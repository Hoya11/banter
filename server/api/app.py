"""텍스트+음성 실시간 채팅 API (Phase 1~2) — WebSocket으로 3자 대화를 서빙한다.

- 라디오 모드: 유저가 radio_sec 동안 침묵하면 AI끼리 진행.
- barge-in(§2.1): 스트리밍 도중 유저가 끼어들면 취소하고 끊긴 지점까지만 기록.
- 세션 캡(§5): AI 발화가 max_turns에 도달하면 마무리 멘트 후 종료.
- 음성 동기(voice mode): 소리가 주인공 — 발화 텍스트는 FE가 오디오 재생 시점에
  표시하고(자막), FE의 played ack을 받아야 다음 라디오 턴을 진행한다.
  서버가 재생보다 앞서 달려 "채팅이 먼저 쌓이고 소리가 뒤늦게 읽는" 어긋남을 막는다.

서버→FE: {hello,voice} {start,speaker} {token,text}* {end,speaker,text}
        {audio,speaker,text,audio|null} {interrupted,...} {error,...} {done}
FE→서버: {"type":"say","text":...} | {"type":"played"} (일반 텍스트는 say로 폴백)
"""

import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine.graph.graph import (
    _strip_speaker_prefix,
    finalize_tagged,
    prepare_utterance,
    strip_audio_tags,
)
from engine.graph.state import initial_state
from engine.graph.supervisor import select_next
from engine.personas.loader import get_persona

WEB_DIR = Path(__file__).resolve().parents[2] / 'web'
RADIO_SEC = 5.0  # 유저 침묵이 이 시간을 넘으면 AI 턴을 자동 진행(라디오 모드)
MAX_TURNS = 12  # AI 발화가 이 수에 도달하면 마무리 후 종료(세션 캡)
ACK_SEC = 30.0  # played ack 최대 대기 — FE가 죽어도 이 이상 세션을 멈추지 않는다
FINISH_INTENT = '이제 대화를 자연스럽게 마무리하는 인사를 건네라'


def _parse_incoming(raw: str) -> tuple[str, str | None]:
    """FE 메시지를 (kind, text)로 푼다. JSON이 아니면 구버전 호환으로 say 취급."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get('type') == 'played':
            return 'played', None
        if isinstance(data, dict) and data.get('type') == 'say':
            return 'say', str(data.get('text') or '')
    except json.JSONDecodeError:
        pass
    return 'say', raw


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
    max_turns: int = MAX_TURNS,
    radio_sec: float = RADIO_SEC,
) -> FastAPI:
    """WS 채팅 앱을 만든다. LLM/TTS client를 주입받아(테스트는 fake) 엔진을 구동한다."""
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
        """
        try:
            persona = get_persona(speaker)
            data = await tts_client.synthesize(
                tagged_text, persona.voice_id or 'alloy', persona.voice_speed
            )
            return base64.b64encode(data).decode('ascii')
        except Exception as exc:
            print(f'[tts] 합성 실패({type(exc).__name__}) — 소리 없이 진행')
            return None

    async def _tts_worker(ws: WebSocket, queue: asyncio.Queue, ctx: dict) -> None:
        """TTS 합성 워커 — 큐를 순서대로 소비해 audio 이벤트를 발화 순서대로 보낸다.

        (턴마다 태스크를 띄우면 합성 완료 순서가 뒤섞여 자막·소리 순서가 깨진다.)
        합성 실패 시에도 audio=null 이벤트를 보내 FE가 텍스트만이라도 표시하게 한다.
        """
        while True:
            speaker, clean, tagged = await queue.get()
            audio = await _synthesize(speaker, tagged)
            try:
                await ws.send_json(
                    {'type': 'audio', 'speaker': speaker, 'text': clean, 'audio': audio}
                )
            except Exception:
                return  # 연결이 닫혔으면 워커 종료

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
        parts: list[str] = []

        async def _run_stream():
            if utterance_client is None:
                parts.append(f'({speaker} 발화 스텁)')
                await ws.send_json({'type': 'token', 'text': parts[-1]})
            else:
                async for token in utterance_client.complete_stream(system, user):
                    parts.append(token)
                    await ws.send_json({'type': 'token', 'text': token})

        async def _cancel(task) -> None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        stream_task = asyncio.create_task(_run_stream())
        barge = None
        while True:
            recv_task = asyncio.create_task(ws.receive_text())
            done, _ = await asyncio.wait(
                {stream_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                kind, text = _parse_incoming(recv_task.result())
                if kind == 'played':  # 재생 완료 ack은 barge가 아니다 — 계속 듣는다
                    ctx['inflight'] = max(0, ctx['inflight'] - 1)
                    if stream_task.done():
                        break
                    continue
                barge = text
                break
            await _cancel(recv_task)
            break

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

        # 최종본 확정 — tagged(오디오 태그 보존)는 TTS용, clean은 화면·이력용
        tagged = finalize_tagged(''.join(parts), name)
        clean = strip_audio_tags(tagged)
        await ws.send_json({'type': 'end', 'speaker': speaker, 'text': clean})
        if voice_mode and clean:
            ctx['inflight'] += 1  # 이 발화의 played ack이 돌아올 때까지 페이스를 잡는다
            await ctx['tts_queue'].put((speaker, clean, tagged))
        return _apply_utterance(state, speaker, clean), barge

    async def _wait_user(
        ws: WebSocket, ctx: dict, timeout: float, until_acked: bool = False
    ) -> str | None:
        """유저 say를 timeout까지 기다린다. played ack은 소비하며 계속 기다린다.

        until_acked=True면 미재생 오디오(inflight)가 0이 되는 순간 바로 반환한다
        — 재생 완료 시점부터 라디오 침묵 타이머를 새로 시작하기 위함.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            if until_acked and ctx['inflight'] <= 0:
                return None
            remain = deadline - loop.time()
            if remain <= 0:
                return None
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=remain)
            except asyncio.TimeoutError:
                return None
            kind, text = _parse_incoming(raw)
            if kind == 'say':
                return text
            ctx['inflight'] = max(0, ctx['inflight'] - 1)

    @app.websocket('/ws')
    async def chat(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json({'type': 'hello', 'voice': voice_mode})
        state = initial_state()
        turn = 0
        pending_user = None  # barge-in으로 받은 유저 발화(다음 턴에 반영)
        ctx = {'inflight': 0, 'tts_queue': asyncio.Queue()}
        worker = asyncio.create_task(_tts_worker(ws, ctx['tts_queue'], ctx))
        try:
            while True:
                if pending_user is not None:
                    state = _add_user(state, pending_user)
                    pending_user = None
                else:
                    # 음성 동기: 앞선 발화의 재생이 끝나기(played ack)를 기다린 뒤에야
                    # 라디오 침묵 타이머를 돌린다 — 서버가 재생보다 앞서 달리지 않게.
                    said = None
                    if voice_mode and ctx['inflight'] > 0:
                        said = await _wait_user(ws, ctx, ACK_SEC, until_acked=True)
                    if said is None:
                        said = await _wait_user(ws, ctx, radio_sec)
                    if said is not None:
                        state = _add_user(state, said)

                finish = turn + 1 >= max_turns
                state, barge = await _stream_turn(state, ws, ctx, finish=finish)
                turn += 1

                # 세션 캡이 barge보다 우선 — 마무리 턴에 끼어들어도 세션은 종료된다
                if finish:
                    await ws.send_json({'type': 'done'})
                    break
                if barge is not None:
                    pending_user = barge  # 개입 발화를 다음 턴에 반영(세션 이어감)
        except WebSocketDisconnect:
            return
        finally:
            worker.cancel()

    return app


# uvicorn 기동용 기본 앱 (실제 LLM/TTS client는 main에서 주입 — 아래는 스텁 폴백)
app = create_app()
