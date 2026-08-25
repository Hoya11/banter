"""텍스트+음성 실시간 채팅 API (Phase 1~2) — WebSocket으로 3자 대화를 서빙한다.

- 라디오 모드: 유저가 radio_sec 동안 침묵하면 AI끼리 진행.
- barge-in(§2.1): 스트리밍 도중 유저가 끼어들면 취소하고 끊긴 지점까지만 기록.
- 세션 캡(§5): AI 발화가 max_turns에 도달하면 마무리 멘트 후 종료.
- 음성 동기(voice mode): 소리가 주인공 — 발화 텍스트는 FE가 오디오 재생 시점에
  표시하고(자막), FE의 played ack을 받아야 다음 라디오 턴을 진행한다.
- prefetch(§3.2): 현재 발화가 재생되는 동안 다음 AI 턴(화자선정→대사→합성)을
  백그라운드로 완성해 둔다 — AI끼리의 티키타카는 체감 지연이 0에 수렴.
  유저가 개입하면 폐기하고 실시간 경로로 반응한다.

재생 페이스는 seq 장부로 맞춘다: audio 이벤트마다 단조 증가 seq를 붙이고
FE는 {played, seq}로 응답한다. 서버는 acked=max(acked, seq)만 기억하므로
ack 중복은 무해하고, 유실도 뒤 ack이 덮는다(개수 카운터의 적자 문제 제거).

서버→FE: {hello,voice} {start,speaker} {token,text}* {end,speaker,text}
        {audio,seq,speaker,text,audio|null} {interrupted,...} {error,...} {done}
FE→서버: {"type":"say","text":...} | {"type":"played","seq":N} (일반 텍스트는 say로 폴백)
"""

import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine.graph.graph import (
    MAX_SENTENCES,
    _strip_speaker_prefix,
    finalize_tagged,
    pop_sentences,
    prepare_utterance,
    strip_audio_tags,
)
from engine.graph.state import initial_state
from engine.graph.supervisor import select_next
from engine.personas.loader import get_persona

WEB_DIR = Path(__file__).resolve().parents[2] / 'web'
RADIO_SEC = 5.0  # 유저 침묵이 이 시간을 넘으면 AI 턴을 자동 진행(라디오 모드)
MAX_TURNS = 12  # AI 발화가 이 수에 도달하면 마무리 후 종료(세션 캡)
ACK_SEC = 30.0  # played ack 최대 대기 — 초과 시 장부를 리셋해 세션이 영구 지연되지 않게
DRAIN_SEC = 15.0  # 종료 전 남은 TTS 전송을 기다리는 상한 (마무리 멘트 유실 방지)
FINISH_INTENT = '이제 대화를 자연스럽게 마무리하는 인사를 건네라'


def _parse_incoming(raw: str) -> tuple[str, object]:
    """FE 메시지를 (kind, value)로 푼다.

    - {"type":"played","seq":N} → ('played', N)
    - {"type":"say","text":...} → ('say', text) — 공백뿐이면 noop
    - {"type":"voice","audio":b64,"mime":...} → ('voice', {...}) — STT 대상
    - 그 외 JSON(미지 type·null 등) → ('noop', None): 유저 발화로 오인해
      이력을 오염시키지 않는다. JSON이 아닌 일반 텍스트만 say로 폴백.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        text = raw.strip()
        return ('say', text) if text else ('noop', None)
    if isinstance(data, dict):
        if data.get('type') == 'played':
            seq = data.get('seq')
            return 'played', seq if isinstance(seq, int) else None
        if data.get('type') == 'say':
            text = str(data.get('text') or '').strip()
            return ('say', text) if text else ('noop', None)
        if data.get('type') == 'voice' and data.get('audio'):
            return 'voice', {'audio': data['audio'], 'mime': data.get('mime') or 'audio/webm'}
    return 'noop', None


def _add_user(state: dict, text: str) -> dict:
    """유저 발화를 상태에 주입한다 (AI 연속 카운터 리셋)."""
    msg = {'speaker': 'user', 'text': text, 'ts': 0.0, 'interrupted': False}
    return {
        **state,
        'messages': state['messages'] + [msg],
        'current_speaker': 'user',
        'consecutive_ai_turns': 0,
        'last_user_turn_ts': 0.0,
    }


def _apply_utterance(state: dict, speaker: str, text: str, interrupted: bool = False) -> dict:
    """발화를 상태에 반영한다 (메시지 추가 + 발화 카운트). interrupted면 끊긴 발화로 기록."""
    personas = {k: dict(v) for k, v in state['personas'].items()}
    if speaker in personas:
        personas[speaker]['speak_count'] += 1
    msg = {'speaker': speaker, 'text': text, 'ts': 0.0, 'interrupted': interrupted}
    return {**state, 'messages': state['messages'] + [msg], 'personas': personas}


def create_app(
    utterance_client=None,
    supervisor_client=None,
    tts_client=None,
    stt_client=None,
    max_turns: int = MAX_TURNS,
    radio_sec: float = RADIO_SEC,
    ack_sec: float = ACK_SEC,
) -> FastAPI:
    """WS 채팅 앱을 만든다. LLM/TTS/STT client를 주입받아(테스트는 fake) 엔진을 구동한다."""
    app = FastAPI()
    voice_mode = tts_client is not None

    @app.get('/')
    def index():
        """채팅 UI 페이지를 서빙한다.

        no-store: WS 프로토콜이 바뀔 때 브라우저가 옛 FE를 캐시로 물고 있으면
        이벤트 형식 불일치로 조용히 깨진다(예: audio 이벤트 미처리 → 무음).
        """
        return FileResponse(WEB_DIR / 'index.html', headers={'Cache-Control': 'no-store'})

    async def _synthesize(speaker: str, tagged_text: str):
        """화자 voice(+speed)로 태그 포함 발화를 합성해 base64 mp3를 반환한다.

        실패하면 None — TTS 문제로 대화 전체가 죽지 않게 격리한다.
        voice_id 미설정은 설정 오류이므로 폴백하지 않고 드러낸다(무음 은폐 방지).
        """
        persona = get_persona(speaker)
        if not persona.voice_id:
            print(f'[tts] voice_id 미설정({speaker}) — duo.yaml voice_presets 확인 필요')
            return None
        try:
            data = await tts_client.synthesize(tagged_text, persona.voice_id, persona.voice_speed)
            return base64.b64encode(data).decode('ascii')
        except Exception as exc:
            print(f'[tts] 합성 실패({type(exc).__name__}) — 소리 없이 진행')
            return None

    async def _send_audio(
        ws: WebSocket, ctx: dict, speaker: str, clean: str, audio, cont: bool = False
    ) -> None:
        """audio 이벤트를 seq를 붙여 보낸다 (합성 실패 시 audio=null — FE는 자막만 표시).

        cont=True는 같은 발화의 이어지는 문장 — FE가 새 말풍선 대신 이어붙인다.
        """
        ctx['sent_seq'] += 1
        await ws.send_json(
            {
                'type': 'audio',
                'seq': ctx['sent_seq'],
                'speaker': speaker,
                'text': clean,
                'audio': audio,
                'cont': cont,
            }
        )

    async def _tts_worker(ws: WebSocket, queue: asyncio.Queue, ctx: dict) -> None:
        """TTS 합성 워커 — 큐를 순서대로 소비해 audio 이벤트를 발화 순서대로 보낸다.

        (턴마다 태스크를 띄우면 합성 완료 순서가 뒤섞여 자막·소리 순서가 깨진다.)
        """
        while True:
            speaker, clean, tagged, cont = await queue.get()
            try:
                audio = await _synthesize(speaker, tagged)
                await _send_audio(ws, ctx, speaker, clean, audio, cont)
            except Exception:
                return  # 연결이 닫혔으면 워커 종료
            finally:
                queue.task_done()

    async def _incoming(ws: WebSocket, raw: str) -> tuple[str, object]:
        """수신 메시지를 해석한다. voice(마이크 녹음)는 STT로 전사해 say로 합류시킨다.

        전사 결과는 {'you', text}로 echo — 유저가 자기 발화 인식 결과를 확인.
        STT 미설정·실패·빈 전사는 noop (대화 오염 방지).
        """
        kind, value = _parse_incoming(raw)
        if kind != 'voice':
            return kind, value
        if stt_client is None:
            return 'noop', None
        try:
            audio = base64.b64decode(value['audio'])
            text = await stt_client.transcribe(audio, value['mime'])
        except Exception as exc:
            print(f'[stt] 전사 실패({type(exc).__name__}) — 무시')
            return 'noop', None
        if not text:
            return 'noop', None
        await ws.send_json({'type': 'you', 'text': text})
        return 'say', text

    async def _flush_stale(ws: WebSocket, ctx: dict) -> None:
        """유저 발화 반영 시점 이전의 오디오를 FE가 버리게 한다.

        합성 중(큐 대기 포함)이던 '유저 발화 이전 대사'가 뒤늦게 도착해
        유저 말풍선 뒤에 재생되는 순서 역전을 막는다. 큐에 남은 항목은
        sent_seq+1..sent_seq+qsize로 나갈 예정이므로 그 상한까지 폐기 지시.
        """
        upto = ctx['sent_seq'] + ctx['tts_queue'].qsize()
        if upto > 0:
            await ws.send_json({'type': 'flush', 'seq': upto})

    def _unplayed(ctx: dict) -> bool:
        """아직 재생 확인이 안 된 발화가 있는가 (합성 대기 포함)."""
        return ctx['sent_seq'] > ctx['acked_seq'] or not ctx['tts_queue'].empty()

    def _on_ack(ctx: dict, seq) -> None:
        """played ack 반영 — seq 장부라 중복·유실에 안전하다."""
        if isinstance(seq, int):
            ctx['acked_seq'] = max(ctx['acked_seq'], seq)
        else:  # 구형/비정상 ack — 전량 재생된 것으로 간주
            ctx['acked_seq'] = ctx['sent_seq']

    async def _prefetch_turn(state: dict, finish: bool = False) -> dict:
        """다음 AI 턴을 미리 통째로 계산한다 — 이벤트 전송 없음 (§3.2 prefetch).

        현재 발화가 재생되는 동안 백그라운드에서 화자선정→대사→합성까지 끝내
        재생 종료 시 즉시 방출할 완성품을 만든다. 유저가 개입하면 폐기된다
        (유저 침묵 전제로 만든 맥락이라 재사용 불가 — prefetch의 비용).
        """
        loop = asyncio.get_event_loop()
        sel = await loop.run_in_executor(None, select_next, state, supervisor_client)
        pre = {**state, **sel}
        if finish:
            pre = {**pre, 'current_intent': FINISH_INTENT}
        system, user, speaker = prepare_utterance(pre)
        name = get_persona(speaker).name
        parts: list[str] = []
        if utterance_client is None:
            parts.append(f'({speaker} 발화 스텁)')
        else:
            async for token in utterance_client.complete_stream(system, user):
                parts.append(token)
        tagged = finalize_tagged(''.join(parts), name)
        clean = strip_audio_tags(tagged)
        audio = await _synthesize(speaker, tagged) if (voice_mode and clean) else None
        return {'sel': sel, 'speaker': speaker, 'clean': clean, 'audio': audio}

    async def _commit_prefetched(state: dict, ws: WebSocket, ctx: dict, pre: dict) -> dict:
        """미리 만든 턴을 확정 방출한다 — start/end/audio를 한 번에.

        트레이드오프: 이 턴은 스트리밍이 없어 서버측 barge-in이 불가하다.
        유저 개입은 FE의 재생 중단 + 다음 턴 반영으로 처리되며, 이력에는
        완주로 남는다(§2.1 충실도는 실시간 경로만) — docs/experiments/latency.md 참고.
        """
        state = {**state, **pre['sel']}
        speaker, clean = pre['speaker'], pre['clean']
        if not clean:  # 빈 발화는 방출·기록하지 않는다
            await ws.send_json({'type': 'error', 'speaker': speaker})
            return state
        await ws.send_json({'type': 'start', 'speaker': speaker})
        await ws.send_json({'type': 'end', 'speaker': speaker, 'text': clean})
        if voice_mode:
            await _send_audio(ws, ctx, speaker, clean, pre['audio'])
        return _apply_utterance(state, speaker, clean)

    async def _discard(task) -> None:
        """prefetch를 폐기한다 (유저 개입 등으로 전제가 무너졌을 때)."""
        if task is None:
            return
        if task.done():
            task.exception()  # 'exception never retrieved' 경고 방지
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _stream_turn(state: dict, ws: WebSocket, ctx: dict, finish: bool = False):
        """한 AI 발화를 스트리밍하되, 도중 유저 개입이 오면 취소(barge-in)한다.

        반환: (새 state, barge_text) — barge_text가 있으면 유저가 끼어든 것.
        """
        loop = asyncio.get_event_loop()
        sel = await loop.run_in_executor(None, select_next, state, supervisor_client)
        state = {**state, **sel}
        if finish:
            state = {**state, 'current_intent': FINISH_INTENT}
        system, user, speaker = prepare_utterance(state)
        name = get_persona(speaker).name

        await ws.send_json({'type': 'start', 'speaker': speaker})
        parts: list[str] = []  # barge 시 부분 발화 복원용 원문 토큰
        emitted: list[str] = []  # 확정된 문장(clean) — 자막·이력용
        flags = {'first': True, 'extended': False}

        def _may_emit(sentence: str) -> bool:
            # 문장 상한(§1.4)의 스트리밍판 — 상한 직후 질문 1개는 허용(유저 소환 보호)
            if len(emitted) < MAX_SENTENCES:
                return True
            if (
                len(emitted) == MAX_SENTENCES
                and not flags['extended']
                and sentence.rstrip().endswith('?')
            ):
                flags['extended'] = True
                return True
            return False

        async def _emit(sentence_raw: str) -> bool:
            """완성 문장 하나를 확정한다 — 자막·이력엔 clean, TTS엔 태그 보존본."""
            raw = sentence_raw
            if flags['first']:
                raw = _strip_speaker_prefix(raw, name)
                flags['first'] = False
            clean = strip_audio_tags(raw).strip()
            if not clean:
                return True  # 태그뿐인 조각 — 버리고 계속
            if not _may_emit(clean):
                return False
            emitted.append(clean)
            if voice_mode:
                await ctx['tts_queue'].put((speaker, clean, raw, len(emitted) > 1))
            return True

        async def _run_stream():
            # 문장 단위 flush(§3.1): 첫 문장이 완성되는 즉시 합성 큐로 보낸다 —
            # 발화 전체 생성·합성을 기다리지 않아 유저 응답의 첫 소리가 단축된다.
            if utterance_client is None:
                text = f'({speaker} 발화 스텁)'
                parts.append(text)
                await ws.send_json({'type': 'token', 'text': text})
                await _emit(text)
                return
            buffer = ''
            async for token in utterance_client.complete_stream(system, user):
                parts.append(token)
                await ws.send_json({'type': 'token', 'text': token})
                buffer += token
                done_sentences, buffer = pop_sentences(buffer)
                for s in done_sentences:
                    if not await _emit(s):
                        return  # 상한 도달 — 남은 토큰 생성도 중단(비용 절약)
            if buffer.strip():
                await _emit(buffer.strip())  # 종결부호 없는 잔여(반말체) flush

        async def _cancel(task) -> None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        stream_task = asyncio.create_task(_run_stream())
        barge = None
        try:
            while True:
                recv_task = asyncio.create_task(ws.receive_text())
                done, _ = await asyncio.wait(
                    {stream_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if recv_task in done:
                    kind, value = await _incoming(ws, recv_task.result())
                    if kind == 'say':
                        barge = value
                        break
                    if kind == 'played':  # ack·noop은 barge가 아니다 — 계속 듣는다
                        _on_ack(ctx, value)
                    if stream_task.done():
                        break
                    continue
                await _cancel(recv_task)
                break
        except WebSocketDisconnect:
            await _cancel(stream_task)  # 방치하면 'exception never retrieved'가 남는다
            raise

        # 스트림이 아직이면(유저 발화가 먼저 도착) → barge-in.
        if barge is not None and not stream_task.done():
            await _cancel(stream_task)
            partial = strip_audio_tags(_strip_speaker_prefix(''.join(parts), name))
            await ws.send_json({'type': 'interrupted', 'speaker': speaker, 'text': partial})
            if partial:  # 빈 부분 발화는 이력에 남기지 않는다(프롬프트 오염 방지)
                state = _apply_utterance(state, speaker, partial, interrupted=True)
            return state, barge

        # 스트림이 에러로 끝났으면 부분 발화를 확정하지 않는다(빈/잘린 발화의 이력 오염 방지)
        if stream_task.exception() is not None:
            print(f'[stream] 발화 생성 실패({type(stream_task.exception()).__name__}) — 턴 건너뜀')
            await ws.send_json({'type': 'error', 'speaker': speaker})
            return state, barge

        # 문장들은 스트리밍 중 이미 확정·합성 큐 전송됨 — 여기선 발화 전체를 마감만
        clean = ' '.join(emitted)
        if not clean:  # 빈 발화는 확정하지 않는다 ('도현: ' 이력 오염 방지)
            await ws.send_json({'type': 'error', 'speaker': speaker})
            return state, barge
        await ws.send_json({'type': 'end', 'speaker': speaker, 'text': clean})
        return _apply_utterance(state, speaker, clean), barge

    async def _wait_user(
        ws: WebSocket, ctx: dict, timeout: float, until_acked: bool = False
    ) -> str | None:
        """유저 say를 timeout까지 기다린다. played ack·noop은 소비하며 계속 기다린다.

        until_acked=True면 미재생 발화가 없어지는 순간 바로 반환하고,
        타임아웃 시 장부를 리셋한다 — ack 유실 하나로 세션이 영구 지연되는 것 방지.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            if until_acked and not _unplayed(ctx):
                return None
            remain = deadline - loop.time()
            if remain <= 0:
                if until_acked and _unplayed(ctx):
                    print('[pace] played ack 타임아웃 — 장부 리셋')
                    ctx['acked_seq'] = ctx['sent_seq']
                return None
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=remain)
            except asyncio.TimeoutError:
                continue  # deadline 검사로 되돌아감
            kind, value = await _incoming(ws, raw)
            if kind == 'say':
                return value
            if kind == 'played':
                _on_ack(ctx, value)

    @app.websocket('/ws')
    async def chat(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json(
            {'type': 'hello', 'voice': voice_mode, 'stt': stt_client is not None}
        )
        state = initial_state()
        turn = 0
        pending_user = None  # barge-in으로 받은 유저 발화(다음 턴에 반영)
        ctx = {'sent_seq': 0, 'acked_seq': 0, 'tts_queue': asyncio.Queue()}
        worker = asyncio.create_task(_tts_worker(ws, ctx['tts_queue'], ctx))
        prefetch = None  # 재생 중 미리 만들어두는 다음 AI 턴 (§3.2)
        try:
            while True:
                user_spoke = False
                if pending_user is not None:
                    state = _add_user(state, pending_user)
                    pending_user = None
                    user_spoke = True
                    await _flush_stale(ws, ctx)  # 끊긴 발화의 이미 큐잉된 문장 오디오 폐기
                    await _flush_stale(ws, ctx)
                else:
                    # 음성 동기: 앞선 발화의 재생이 끝나기(played ack)를 기다린 뒤에야
                    # 라디오 침묵 타이머를 돌린다 — 서버가 재생보다 앞서 달리지 않게.
                    # (그동안 prefetch가 백그라운드에서 다음 턴을 완성해 둔다)
                    said = None
                    if voice_mode and _unplayed(ctx):
                        said = await _wait_user(ws, ctx, ack_sec, until_acked=True)
                    if said is None:
                        said = await _wait_user(ws, ctx, radio_sec)
                    if said is not None:
                        state = _add_user(state, said)
                        user_spoke = True
                        await _flush_stale(ws, ctx)

                finish = turn + 1 >= max_turns
                if user_spoke:
                    # 유저가 말함 → "침묵 전제" prefetch는 무효, 실시간 경로로 반응
                    await _discard(prefetch)
                    prefetch = None
                if prefetch is not None:
                    # prefetch 히트 — 완성품을 즉시 방출 (지연 은폐)
                    try:
                        pre = await prefetch
                        state = await _commit_prefetched(state, ws, ctx, pre)
                        barge = None
                    except Exception:  # 준비 실패 시 실시간 경로로 폴백
                        state, barge = await _stream_turn(state, ws, ctx, finish=finish)
                    prefetch = None
                else:
                    state, barge = await _stream_turn(state, ws, ctx, finish=finish)
                turn += 1

                # 세션 캡이 barge보다 우선 — 마무리 턴에 끼어들어도 세션은 종료된다
                if finish:
                    # 마무리 멘트의 TTS가 아직 큐에 있으면 전송을 기다린다 (유실 방지)
                    try:
                        await asyncio.wait_for(ctx['tts_queue'].join(), timeout=DRAIN_SEC)
                    except asyncio.TimeoutError:
                        pass
                    await ws.send_json({'type': 'done'})
                    break
                if barge is not None:
                    pending_user = barge  # 개입 발화를 다음 턴에 반영(세션 이어감)
                elif voice_mode:
                    # 방금 발화가 재생되는 동안 다음 턴을 미리 만든다 (마무리 여부 반영)
                    prefetch = asyncio.create_task(
                        _prefetch_turn(state, finish=turn + 1 >= max_turns)
                    )
        except WebSocketDisconnect:
            return
        finally:
            worker.cancel()
            await _discard(prefetch)

    return app


# uvicorn 기동용 기본 앱 (실제 LLM/TTS client는 main에서 주입 — 아래는 스텁 폴백)
app = create_app()
