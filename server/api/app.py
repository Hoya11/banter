"""텍스트와 음성으로 3자 대화를 제공하는 WebSocket API.

- 라디오 모드: 유저가 radio_sec 동안 침묵하면 AI끼리 진행.
- barge-in: 스트리밍 도중 유저가 끼어들면 취소하고 끊긴 지점까지만 기록.
- 세션 캡: AI 발화가 max_turns에 도달하면 마무리 멘트 후 종료.
- 음성 동기(voice mode): 소리가 주인공. 발화 텍스트는 FE가 오디오 재생 시점에
  표시하고(자막), FE의 played ack을 받아야 다음 라디오 턴을 진행한다.
- prefetch: 현재 발화가 재생되는 동안 다음 AI 턴의 화자 선정, 대사, 합성을
  백그라운드로 완성해 둔다. AI끼리의 티키타카는 체감 지연이 0에 수렴한다.
  유저가 개입하면 폐기하고 실시간 경로로 반응한다.

재생 페이스는 seq 장부로 맞춘다: audio 이벤트마다 단조 증가 seq를 붙이고
FE는 {played, seq}로 응답한다. 서버는 acked=max(acked, seq)만 기억하므로
ack 중복은 무해하고, 유실도 뒤 ack이 덮는다(개수 카운터의 적자 문제 제거).

서버에서 FE: {hello,voice} {session_started,session_id}
        {start,speaker,after_client_event_id?} {token,text}*
        {end,speaker,text} {audio,seq,speaker,text,audio|null} {interrupted,...}
        {error,...} {done}
FE에서 서버: {"type":"session_start"}
        | {"type":"say","text":...,"client_event_id":...}
        | {"type":"hold","client_event_id":...,"audio_stop":...}
        | {"type":"hold_off","client_event_id":...}
        | {"type":"voice","client_event_id":...,"audio":...,"mime":...}
        | {"type":"played","seq":N} (일반 텍스트는 say로 폴백)
"""

import asyncio
import base64
import json
import re
from math import isfinite
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api.event_log import EventName
from engine.graph.graph import (
    MAX_SENTENCES,
    _strip_speaker_prefix,
    finalize_tagged,
    pop_sentences,
    prepare_utterance,
    strip_audio_tags,
)
from engine.graph.state import initial_state
from engine.graph.supervisor import (
    _mechanical_next,
    build_unified_prompt,
    parse_unified_header,
    select_next,
    split_unified,
)
from engine.personas.loader import get_persona

WEB_DIR = Path(__file__).resolve().parents[2] / 'web'
RADIO_SEC = 5.0  # 유저 침묵이 이 시간을 넘으면 AI 턴을 자동 진행(라디오 모드)
MAX_TURNS = 12  # AI 발화가 이 수에 도달하면 마무리 후 종료(세션 캡)
ACK_SEC = 30.0  # played ack 최대 대기 — 초과 시 장부를 리셋해 세션이 영구 지연되지 않게
HOLD_SEC = 15.0  # hold 최대 유지 시간. 전사가 안 오면 라디오로 복귀한다.
MIN_VOICE_BYTES = 6000  # 이보다 짧은 녹음은 무시 (무음·스침 — 환각 전사 방지)
DRAIN_SEC = 15.0  # 종료 전 남은 TTS 전송을 기다리는 상한 (마무리 멘트 유실 방지)
FINISH_INTENT = '이제 대화를 자연스럽게 마무리하는 인사를 건네라'
CLIENT_AUDIO_STOP_MAX_MS = 60_000.0
CLIENT_SEQUENCE_MAX = 2_147_483_647
AUDIO_STOP_OUTCOMES = frozenset({'paused', 'pause_failed', 'queued_only', 'idle'})
CLIENT_EVENT_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
SEGMENT_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')


def _safe_client_event_id(data: dict) -> tuple[str | None, bool]:
    """클라이언트 이벤트 ID를 검증한다.

    필드가 없는 기존 클라이언트는 허용하지만, 필드를 보내고 형식이
    틀린 경우는 없는 ID로 오인하지 않는다.
    """
    if 'client_event_id' not in data:
        return None, True
    value = data.get('client_event_id')
    if isinstance(value, str) and CLIENT_EVENT_ID_PATTERN.fullmatch(value):
        return value, True
    return None, False


def _safe_audio_stop(value: object) -> dict | None:
    """브라우저 재생 중단 결과에서 측정에 필요한 값만 복사한다."""
    if not isinstance(value, dict):
        return None

    outcome = value.get('outcome')
    if outcome not in AUDIO_STOP_OUTCOMES:
        return None

    elapsed = value.get('elapsed_ms')
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not isfinite(elapsed)
        or not 0 <= elapsed <= CLIENT_AUDIO_STOP_MAX_MS
    ):
        return None
    if outcome == 'paused' and elapsed is None:
        return None

    segment_id = value.get('segment_id')
    if segment_id is not None and (
        not isinstance(segment_id, str)
        or SEGMENT_ID_PATTERN.fullmatch(segment_id) is None
    ):
        return None

    sequence = value.get('sequence')
    if sequence is not None and (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 0 <= sequence <= CLIENT_SEQUENCE_MAX
    ):
        return None

    if segment_id is None and sequence is not None:
        segment_id = f'audio-{sequence}'
    return {
        'outcome': outcome,
        'elapsed_ms': round(float(elapsed), 3) if elapsed is not None else None,
        'segment_id': segment_id,
    }


def _parse_incoming(raw: str) -> tuple[str, object]:
    """FE 메시지를 (kind, value)로 푼다.

    - {"type":"played","seq":N} → ('played', N)
    - {"type":"session_start"}는 ('session_start', None)
    - {"type":"say","text":...}: ('say', {text, client_event_id}). 공백뿐이면 noop
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
        if data.get('type') == 'session_start':
            return 'session_start', None
        if data.get('type') == 'played':
            seq = data.get('seq')
            return 'played', seq if isinstance(seq, int) else None
        if data.get('type') == 'say':
            client_event_id, valid_id = _safe_client_event_id(data)
            if not valid_id:
                return 'noop', None
            text = str(data.get('text') or '').strip()
            return (
                ('say', {'text': text, 'client_event_id': client_event_id})
                if text
                else ('noop', None)
            )
        if data.get('type') in {'voice', 'hold', 'hold_off'}:
            client_event_id, valid_id = _safe_client_event_id(data)
            if not valid_id:
                return 'noop', None
            if data.get('type') == 'voice' and data.get('audio'):
                return 'voice', {
                    'audio': data['audio'],
                    'mime': data.get('mime') or 'audio/webm',
                    'client_event_id': client_event_id,
                }
            if data.get('type') == 'hold':
                return 'hold', {
                    'client_event_id': client_event_id,
                    'audio_stop': _safe_audio_stop(data.get('audio_stop')),
                }
            if data.get('type') == 'hold_off':
                return 'hold_off', {'client_event_id': client_event_id}
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
    unified: bool = False,  # v2 통합 생성(D-003) — 화자 선정+발화를 한 호출로
    *,
    event_recorder=None,
    run_id: str | None = None,
    run_metadata: dict | None = None,
) -> FastAPI:
    """WS 채팅 앱을 만든다. LLM/TTS/STT client를 주입받아(테스트는 fake) 엔진을 구동한다."""
    app = FastAPI()
    voice_mode = tts_client is not None
    runtime_run_id = run_id or uuid4().hex
    runtime_metadata = {
        **dict(run_metadata or {}),
        'max_turns': max_turns,
        'radio_sec': radio_sec,
        'ack_sec': ack_sec,
        'unified': unified,
        'protocol_version': 1,
        'voice_mode': voice_mode,
    }

    def _record(
        ctx: dict,
        event: EventName,
        *,
        generation_id: str | None = None,
        segment_id: str | None = None,
        client_event_id: str | None = None,
        client_elapsed_ms: float | None = None,
        outcome: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """관측 실패가 대화 흐름을 끊지 않게 격리한 best-effort 기록."""
        if event_recorder is None:
            return
        try:
            event_recorder.record(
                event,
                run_id=runtime_run_id,
                session_id=ctx['session_id'],
                turn_id=ctx.get('turn_id'),
                generation_id=generation_id,
                segment_id=segment_id,
                client_event_id=client_event_id,
                client_elapsed_ms=client_elapsed_ms,
                outcome=outcome,
                metadata=metadata,
            )
        except Exception as exc:
            print(f'[events] 기록 실패({type(exc).__name__}), 대화 계속')

    def _flush_events() -> None:
        flush = getattr(event_recorder, 'flush', None)
        if flush is None:
            return
        try:
            flush()
        except Exception as exc:
            print(f'[events] 저장 실패({type(exc).__name__}), 대화 종료는 유지')

    def _record_pending_hold(
        ctx: dict,
        generation_id: str | None,
        payload: dict,
    ) -> bool:
        """hold와 재생 중단 결과를 한 수신 단위로 기록한다."""
        client_event_id = payload.get('client_event_id')
        if (
            client_event_id is not None
            and client_event_id in ctx['seen_client_event_ids']
        ):
            return False
        if client_event_id is not None:
            ctx['seen_client_event_ids'].add(client_event_id)

        if ctx.get('next_turn_after_hold_pending'):
            _cancel_pending_hold(ctx, release_presentation=False)
        ctx['tts_epoch'] += 1

        _record(
            ctx,
            EventName.HOLD_RECEIVED,
            generation_id=generation_id,
            client_event_id=client_event_id,
        )
        audio_stop = payload.get('audio_stop')
        if audio_stop is not None and client_event_id is not None:
            _record(
                ctx,
                EventName.AUDIO_STOPPED,
                generation_id=generation_id,
                segment_id=audio_stop.get('segment_id'),
                client_event_id=client_event_id,
                client_elapsed_ms=audio_stop.get('elapsed_ms'),
                outcome=audio_stop['outcome'],
            )
        ctx['next_turn_after_hold_pending'] = True
        ctx['hold_sequence'] += 1
        ctx['active_hold_sequence'] = ctx['hold_sequence']
        ctx['active_hold_client_event_id'] = client_event_id
        return True

    def _arm_presentation_release(ctx: dict, client_event_id: str | None) -> None:
        """다음 start가 해제할 브라우저 입력을 최신 값으로 교체한다."""
        ctx['presentation_release_sequence'] += 1
        ctx['presentation_release_client_event_id'] = client_event_id

    def _cancel_pending_hold(
        ctx: dict,
        client_event_id: str | None = None,
        *,
        release_presentation: bool = True,
    ) -> bool:
        """ID가 있으면 현재 hold와 일치할 때만 취소한다."""
        if not ctx.get('next_turn_after_hold_pending'):
            return False
        active_id = ctx.get('active_hold_client_event_id')
        if client_event_id is not None and client_event_id != active_id:
            return False
        _record(
            ctx,
            EventName.HOLD_CANCELLED,
            client_event_id=active_id,
        )
        if release_presentation:
            _arm_presentation_release(ctx, active_id)
        ctx['next_turn_after_hold_pending'] = False
        ctx['active_hold_sequence'] = None
        ctx['active_hold_client_event_id'] = None
        ctx['hold_until'] = 0.0
        return True

    def _begin_generation(ctx: dict) -> str:
        ctx['generation_seq'] += 1
        generation_id = f'generation-{ctx["generation_seq"]}'
        ctx['active_generation_id'] = generation_id
        _record(ctx, EventName.GENERATION_STARTED, generation_id=generation_id)
        return generation_id

    def _finish_generation(ctx: dict, generation_id: str) -> None:
        if ctx.get('active_generation_id') == generation_id:
            ctx['active_generation_id'] = None

    async def _send_start(ws: WebSocket, ctx: dict, speaker: str) -> None:
        """start 전송과 끼어들기 뒤 첫 응답 관측을 한 지점에서 처리한다."""
        payload = {'type': 'start', 'speaker': speaker}
        release_id = ctx.get('presentation_release_client_event_id')
        release_sequence = ctx['presentation_release_sequence']
        hold_sequence = ctx.get('active_hold_sequence')
        hold_client_event_id = ctx.get('active_hold_client_event_id')
        generation_id = ctx.get('active_generation_id')
        if release_id is not None:
            payload['after_client_event_id'] = release_id
        await ws.send_json(payload)
        if ctx['presentation_release_sequence'] == release_sequence:
            ctx['presentation_release_client_event_id'] = None
        if (
            hold_sequence is not None
            and ctx.get('next_turn_after_hold_pending')
            and ctx.get('active_hold_sequence') == hold_sequence
        ):
            if hold_client_event_id is None or release_id == hold_client_event_id:
                _record(
                    ctx,
                    EventName.NEXT_TURN_STARTED,
                    generation_id=generation_id,
                    client_event_id=hold_client_event_id,
                )
                ctx['next_turn_after_hold_pending'] = False
                ctx['active_hold_sequence'] = None
                ctx['active_hold_client_event_id'] = None
                ctx['hold_until'] = 0.0

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
            speaker, clean, tagged, cont, item_epoch, generation_id = await queue.get()
            try:
                if item_epoch != ctx['tts_epoch']:
                    _record(
                        ctx,
                        EventName.LATE_AUDIO_DROPPED,
                        generation_id=generation_id,
                    )
                    continue
                audio = await _synthesize(speaker, tagged)
                if item_epoch != ctx['tts_epoch']:
                    _record(
                        ctx,
                        EventName.LATE_AUDIO_DROPPED,
                        generation_id=generation_id,
                    )
                    continue
                await _send_audio(ws, ctx, speaker, clean, audio, cont)
            except Exception:
                return  # 연결이 닫혔으면 워커 종료
            finally:
                queue.task_done()

    async def _incoming(ws: WebSocket, ctx: dict, raw: str) -> tuple[str, object]:
        """수신 메시지를 해석한다. voice(마이크 녹음)는 STT로 전사해 say로 합류시킨다.

        hold와 hold_off는 발화권 신호이며 ctx['hold_until']을 갱신한다.
        voice가 해소되면(성공이든 무효든) hold도 함께 푼다.
        전사 결과는 {'you', text}로 echo. STT 미설정·실패·빈·초단 녹음은 noop.
        """
        loop = asyncio.get_event_loop()
        kind, value = _parse_incoming(raw)
        if kind == 'hold':
            accepted = _record_pending_hold(
                ctx,
                ctx.get('active_generation_id'),
                value,
            )
            if not accepted:
                return 'hold_duplicate', value
            ctx['hold_until'] = loop.time() + HOLD_SEC
            return 'hold', value
        if kind == 'hold_off':
            if _cancel_pending_hold(ctx, value.get('client_event_id')):
                ctx['hold_until'] = 0.0
            return 'noop', None
        if kind == 'say':
            if isinstance(value, dict):
                text = value['text']
                client_event_id = value.get('client_event_id')
            else:
                text = value
                client_event_id = None
            ctx['tts_epoch'] += 1
            ctx['hold_until'] = 0.0
            if client_event_id is not None:
                if ctx.get('next_turn_after_hold_pending'):
                    _cancel_pending_hold(ctx, release_presentation=False)
                _arm_presentation_release(ctx, client_event_id)
            elif ctx.get('next_turn_after_hold_pending'):
                _arm_presentation_release(
                    ctx,
                    ctx.get('active_hold_client_event_id'),
                )
            return kind, text
        if kind != 'voice':
            return kind, value
        client_event_id = value.get('client_event_id')
        active_id = ctx.get('active_hold_client_event_id')
        pending_hold = ctx.get('next_turn_after_hold_pending')
        if client_event_id is not None:
            if not pending_hold or client_event_id != active_id:
                return 'noop', None
        elif pending_hold and active_id is not None:
            return 'noop', None
        if not pending_hold:
            ctx['tts_epoch'] += 1
        ctx['hold_until'] = 0.0
        if stt_client is None:
            _cancel_pending_hold(ctx, client_event_id)
            return 'noop', None
        try:
            audio = base64.b64decode(value['audio'])
            if len(audio) < MIN_VOICE_BYTES:  # 무음·스침 — 환각 전사 방지
                print(f'[stt] 녹음 너무 짧음({len(audio)}B) — 무시')
                _cancel_pending_hold(ctx, client_event_id)
                return 'noop', None
            text = await stt_client.transcribe(audio, value['mime'])
        except Exception as exc:
            print(f'[stt] 전사 실패({type(exc).__name__}) — 무시')
            _cancel_pending_hold(ctx, client_event_id)
            return 'noop', None
        if not text:
            print('[stt] 빈 전사 — 무시')
            _cancel_pending_hold(ctx, client_event_id)
            return 'noop', None
        print('[stt] 전사 완료')
        if pending_hold:
            _arm_presentation_release(ctx, active_id)
        await ws.send_json({'type': 'you', 'text': text})
        return 'say', text

    async def _flush_stale(ws: WebSocket, ctx: dict) -> None:
        """유저 발화 반영 시점 이전의 오디오를 FE가 버리게 한다.

        이미 전송된 '유저 발화 이전 대사'가 유저 말풍선 뒤에 재생되는 순서
        역전을 막는다. 아직 합성 중이거나 큐에 남은 항목은 tts_epoch가 달라져
        서버에서 폐기하므로 전송이 끝난 sent_seq까지만 브라우저에 알린다.
        """
        upto = ctx['sent_seq']
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
        """다음 AI 턴을 미리 계산하되 이벤트는 전송하지 않는다.

        현재 발화가 재생되는 동안 백그라운드에서 화자선정→대사→합성까지 끝내
        재생 종료 시 즉시 방출할 완성품을 만든다. 유저가 개입하면 폐기된다
        (유저 침묵 전제로 만든 맥락이라 재사용 불가 — prefetch의 비용).
        """
        if unified:
            # v2(D-003): 한 호출로 화자+대사 — supervisor 호출 없음
            system, user = build_unified_prompt(state, finish=finish)
            parts: list[str] = []
            if utterance_client is None:
                parts.append('{"next_speaker": "ai_a"}\n(발화 스텁)')
            else:
                async for token in utterance_client.complete_stream(system, user):
                    parts.append(token)
            sel, text_raw = split_unified(''.join(parts), state)
            speaker = sel['current_speaker']
            tagged = finalize_tagged(text_raw.strip(), get_persona(speaker).name)
        else:
            loop = asyncio.get_event_loop()
            sel = await loop.run_in_executor(None, select_next, state, supervisor_client)
            pre = {**state, **sel}
            if finish:
                pre = {**pre, 'current_intent': FINISH_INTENT}
            system, user, speaker = prepare_utterance(pre)
            name = get_persona(speaker).name
            parts = []
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
        완주로 남는다. 끊긴 지점 기록은 실시간 경로에서만 정확하다.
        """
        state = {**state, **pre['sel']}
        speaker, clean = pre['speaker'], pre['clean']
        if not clean:  # 빈 발화는 방출·기록하지 않는다
            await ws.send_json({'type': 'error', 'speaker': speaker})
            return state
        await _send_start(ws, ctx, speaker)
        await ws.send_json({'type': 'end', 'speaker': speaker, 'text': clean})
        if voice_mode:
            await _send_audio(ws, ctx, speaker, clean, pre['audio'])
        return _apply_utterance(state, speaker, clean)

    async def _discard(task) -> None:
        """prefetch를 폐기한다 (유저 개입 등으로 전제가 무너졌을 때)."""
        if task is None:
            return
        if task.done():
            try:
                task.exception()  # 'exception never retrieved' 경고 방지
            except asyncio.CancelledError:
                pass
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
        generation_id = _begin_generation(ctx)
        # unified(v2)는 화자를 스트림의 첫 줄(헤더)이 정한다 — 그때까지 hd가 비어 있다
        hd = {'speaker': None, 'name': None, 'sel': None}
        if unified:
            system, user = build_unified_prompt(state, finish=finish)
        else:
            loop = asyncio.get_event_loop()
            sel = await loop.run_in_executor(None, select_next, state, supervisor_client)
            state = {**state, **sel}
            if finish:
                state = {**state, 'current_intent': FINISH_INTENT}
            system, user, speaker_v1 = prepare_utterance(state)
            hd['speaker'] = speaker_v1
            hd['name'] = get_persona(speaker_v1).name
            await _send_start(ws, ctx, speaker_v1)

        async def _resolve_header(first_line: str) -> bool:
            """v2 헤더를 파싱해 화자를 확정하고 start를 보낸다. 반환: 헤더가 유효했는가."""
            selp = parse_unified_header(first_line.strip(), state)
            ok = selp is not None
            if not ok:  # 규칙 위반·파싱 실패 — 기계적 교대 폴백 (본문은 살릴 수 있음)
                print('[unified] 헤더 불량 — 기계적 교대 폴백')
                prev = state['current_speaker']
                nxt = _mechanical_next(prev, state['messages'])
                selp = {
                    'current_speaker': nxt,
                    'consecutive_ai_turns': 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1,
                    'current_intent': None,
                    'topic_stack': list(state.get('topic_stack') or []),
                    'topic_turns': state.get('topic_turns', 0),
                }
            hd['sel'] = selp
            hd['speaker'] = selp['current_speaker']
            hd['name'] = get_persona(hd['speaker']).name
            await _send_start(ws, ctx, hd['speaker'])
            return ok

        parts: list[str] = []  # barge 시 부분 발화 복원용 원문 토큰
        emitted: list[str] = []  # 확정된 문장(clean) — 자막·이력용
        flags = {'first': True, 'extended': False}

        def _may_emit(sentence: str) -> bool:
            # 스트리밍에서도 문장 상한 직후 질문 1개는 허용한다.
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
                raw = _strip_speaker_prefix(raw, hd['name'])
                flags['first'] = False
            clean = strip_audio_tags(raw).strip()
            if not clean:
                return True  # 태그뿐인 조각 — 버리고 계속
            if not _may_emit(clean):
                return False
            emitted.append(clean)
            if voice_mode:
                await ctx['tts_queue'].put(
                    (
                        hd['speaker'],
                        clean,
                        raw,
                        len(emitted) > 1,
                        ctx['tts_epoch'],
                        generation_id,
                    )
                )
            return True

        async def _run_stream():
            # 첫 문장이 완성되는 즉시 합성 큐로 보내 문장 단위로 처리한다.
            # 발화 전체 생성·합성을 기다리지 않아 유저 응답의 첫 소리가 단축된다.
            # unified(v2)는 첫 줄(헤더)을 버퍼링해 화자를 먼저 확정한다.
            if utterance_client is None:
                if unified:
                    await _resolve_header('{"next_speaker": "ai_a"}')
                text = f"({hd['speaker']} 발화 스텁)"
                parts.append(text)
                await ws.send_json({'type': 'token', 'text': text})
                await _emit(text)
                return
            header_pending = unified
            buffer = ''
            async for token in utterance_client.complete_stream(system, user):
                parts.append(token)
                if header_pending:
                    buffer += token
                    if '\n' not in buffer:
                        continue
                    first, _, buffer = buffer.partition('\n')
                    await _resolve_header(first)
                    header_pending = False
                    if buffer:  # 헤더 뒤에 딸려온 본문 조각도 화면에 흘린다
                        await ws.send_json({'type': 'token', 'text': buffer})
                else:
                    await ws.send_json({'type': 'token', 'text': token})
                    buffer += token
                done_sentences, buffer = pop_sentences(buffer)
                for s in done_sentences:
                    if not await _emit(s):
                        return  # 상한 도달 — 남은 토큰 생성도 중단(비용 절약)
            if header_pending:
                # 개행 없이 끝남 — 한 줄이 헤더인지 대사인지 판별
                ok = await _resolve_header(buffer)
                buffer = '' if (ok or buffer.lstrip().startswith('{')) else buffer
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
        grabbed = False  # hold는 내용 없이 발화권만 먼저 잡고 전사는 뒤따라온다.
        try:
            while True:
                recv_task = asyncio.create_task(ws.receive_text())
                done, _ = await asyncio.wait(
                    {stream_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                # 생성 완료와 hold가 동시에 준비되면 완료된 generation을 먼저 비활성화한다.
                # 이후 hold는 취소 대상이 없는 floor-grab으로 기록돼 누락 오진을 막는다.
                if stream_task.done():
                    _finish_generation(ctx, generation_id)
                if recv_task in done:
                    kind, value = await _incoming(ws, ctx, recv_task.result())
                    if kind == 'say':
                        barge = value
                        break
                    if kind == 'hold':
                        if not stream_task.done():
                            grabbed = True  # 누르는 순간 발화를 멈춰 사용자 음성이 묻히지 않게 한다.
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
            _finish_generation(ctx, generation_id)
            raise

        # unified: 헤더가 정한 화자·화제 갱신을 상태에 반영 (v1은 이미 반영됨)
        if hd['sel'] is not None:
            state = {**state, **hd['sel']}

        # 스트림이 아직이면(유저 발화·hold가 먼저 도착) → barge-in.
        if (barge is not None or grabbed) and not stream_task.done():
            await _cancel(stream_task)
            _record(
                ctx,
                EventName.GENERATION_INVALIDATED,
                generation_id=generation_id,
            )
            _finish_generation(ctx, generation_id)
            if hd['speaker'] is None:  # 헤더 전 개입 — 아무것도 표시된 게 없다
                return state, barge
            partial = strip_audio_tags(_strip_speaker_prefix(''.join(parts), hd['name']))
            await ws.send_json({'type': 'interrupted', 'speaker': hd['speaker'], 'text': partial})
            if partial:  # 빈 부분 발화는 이력에 남기지 않는다(프롬프트 오염 방지)
                state = _apply_utterance(state, hd['speaker'], partial, interrupted=True)
            return state, barge

        _finish_generation(ctx, generation_id)

        # 스트림이 에러로 끝났으면 부분 발화를 확정하지 않는다(빈/잘린 발화의 이력 오염 방지)
        if stream_task.exception() is not None:
            print(f'[stream] 발화 생성 실패({type(stream_task.exception()).__name__}) — 턴 건너뜀')
            await ws.send_json({'type': 'error', 'speaker': hd['speaker']})
            return state, barge

        # 문장들은 스트리밍 중 이미 확정·합성 큐 전송됨 — 여기선 발화 전체를 마감만
        clean = ' '.join(emitted)
        if not clean:  # 빈 발화는 확정하지 않는다 ('도현: ' 이력 오염 방지)
            await ws.send_json({'type': 'error', 'speaker': hd['speaker']})
            return state, barge
        await ws.send_json({'type': 'end', 'speaker': hd['speaker'], 'text': clean})
        return _apply_utterance(state, hd['speaker'], clean), barge

    async def _wait_user(
        ws: WebSocket, ctx: dict, timeout: float, until_acked: bool = False
    ) -> str | None:
        """유저 say를 timeout까지 기다린다. played ack·noop은 소비하며 계속 기다린다.

        until_acked=True면 미재생 발화가 없어지는 순간 바로 반환하고,
        타임아웃 시 장부를 리셋한다 — ack 유실 하나로 세션이 영구 지연되는 것 방지.
        """
        loop = asyncio.get_event_loop()
        base_deadline = loop.time() + timeout
        while True:
            if until_acked and not _unplayed(ctx):
                return None
            hold_deadline_active = bool(
                ctx.get('next_turn_after_hold_pending') and ctx.get('hold_until')
            )
            deadline = (
                min(base_deadline, ctx['hold_until'])
                if hold_deadline_active
                else base_deadline
            )
            remain = deadline - loop.time()
            if remain <= 0:
                if hold_deadline_active and ctx['hold_until'] <= base_deadline:
                    return None
                if until_acked and _unplayed(ctx):
                    print('[pace] played ack 타임아웃 — 장부 리셋')
                    ctx['acked_seq'] = ctx['sent_seq']
                return None
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=remain)
            except asyncio.TimeoutError:
                continue  # deadline 검사로 되돌아감
            kind, value = await _incoming(ws, ctx, raw)
            if kind == 'say':
                return value
            if kind == 'played':
                _on_ack(ctx, value)

    @app.websocket('/ws')
    async def chat(ws: WebSocket) -> None:
        await ws.accept()
        try:
            await ws.send_json(
                {'type': 'hello', 'voice': voice_mode, 'stt': stt_client is not None}
            )
            while True:
                kind, _ = _parse_incoming(await ws.receive_text())
                if kind == 'session_start':
                    break
        except WebSocketDisconnect:
            return

        ctx = {
            'session_id': uuid4().hex,
            'turn_id': None,
            'generation_seq': 0,
            'active_generation_id': None,
            'next_turn_after_hold_pending': False,
            'hold_sequence': 0,
            'active_hold_sequence': None,
            'active_hold_client_event_id': None,
            'presentation_release_sequence': 0,
            'presentation_release_client_event_id': None,
            'seen_client_event_ids': set(),
            'tts_epoch': 0,
            'sent_seq': 0,
            'acked_seq': 0,
            'hold_until': 0.0,
            'tts_queue': asyncio.Queue(),
        }
        state = initial_state()
        turn = 0
        pending_user = None  # barge-in으로 받은 유저 발화(다음 턴에 반영)
        loop = asyncio.get_event_loop()
        worker = None
        prefetch = None  # 재생 중 미리 만들어두는 다음 AI 턴
        try:
            _record(ctx, EventName.SESSION_STARTED, metadata=runtime_metadata)
            await ws.send_json(
                {'type': 'session_started', 'session_id': ctx['session_id']}
            )
            worker = asyncio.create_task(_tts_worker(ws, ctx['tts_queue'], ctx))
            while True:
                user_spoke = False
                if pending_user is not None:
                    state = _add_user(state, pending_user)
                    pending_user = None
                    user_spoke = True
                    await _flush_stale(ws, ctx)  # 끊긴 발화의 이미 큐잉된 문장 오디오 폐기
                else:
                    # 음성 동기: 앞선 발화의 재생이 끝나기(played ack)를 기다린 뒤에야
                    # 라디오 침묵 타이머를 돌린다 — 서버가 재생보다 앞서 달리지 않게.
                    # (그동안 prefetch가 백그라운드에서 다음 턴을 완성해 둔다)
                    said = None
                    if voice_mode and _unplayed(ctx):
                        said = await _wait_user(ws, ctx, ack_sec, until_acked=True)
                    if said is None and not ctx.get('next_turn_after_hold_pending'):
                        said = await _wait_user(ws, ctx, radio_sec)
                    # hold 중에는 전사 결과가 올 때까지 새 AI 턴을 열지 않는다.
                    while said is None:
                        remain = ctx['hold_until'] - loop.time()
                        if remain <= 0:
                            _cancel_pending_hold(
                                ctx,
                                ctx.get('active_hold_client_event_id'),
                            )
                            break
                        said = await _wait_user(ws, ctx, min(remain, 1.0))
                    if said is not None:
                        state = _add_user(state, said)
                        user_spoke = True
                        await _flush_stale(ws, ctx)

                ctx['turn_id'] = f'turn-{turn + 1}'
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
                    except WebSocketDisconnect:
                        raise  # 연결 끊김은 폴백 대상이 아니다 — 닫힌 소켓에 재전송 금지
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
            _cancel_pending_hold(ctx)
            await _discard(worker)
            await _discard(prefetch)
            await asyncio.to_thread(_flush_events)

    return app


# uvicorn 기동용 기본 앱 (실제 LLM/TTS client는 main에서 주입 — 아래는 스텁 폴백)
app = create_app()
