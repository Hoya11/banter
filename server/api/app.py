"""텍스트+음성 실시간 채팅 API (Phase 1~2) — WebSocket으로 3자 대화를 스트리밍 서빙한다.

유저는 WS로 연결·발화하고, 서버는 화자를 정한 뒤 발화를 토큰 단위로 push한다.
- 라디오 모드: 유저가 radio_sec 동안 침묵하면 AI끼리 진행.
- barge-in(§2.1): 스트리밍 도중 유저가 끼어들면 취소하고 끊긴 지점까지만 기록.
- 세션 캡(§5): AI 발화가 max_turns에 도달하면 마무리 멘트 후 종료.
- 음성(Phase 2): tts_client가 있으면 발화 완성 시 화자 voice로 합성한 mp3(base64)를 함께 전송.
프로토콜: {start,speaker} → {token,text}* → {end,speaker,text,audio?|interrupted,speaker,text} → {done}
"""

import asyncio
import base64
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine.graph.graph import _strip_speaker_prefix, prepare_utterance
from engine.graph.state import initial_state
from engine.graph.supervisor import select_next
from engine.personas.loader import get_persona

WEB_DIR = Path(__file__).resolve().parents[2] / 'web'
RADIO_SEC = 5.0  # 유저 침묵이 이 시간을 넘으면 AI 턴을 자동 진행(라디오 모드)
MAX_TURNS = 12  # AI 발화가 이 수에 도달하면 마무리 후 종료(세션 캡)
FINISH_INTENT = '이제 대화를 자연스럽게 마무리하는 인사를 건네라'


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

    @app.get('/')
    def index():
        """채팅 UI 페이지를 서빙한다."""
        return FileResponse(WEB_DIR / 'index.html')

    async def _synthesize(speaker: str, text: str):
        """화자 voice로 발화를 합성해 base64 mp3를 반환한다.

        tts_client가 없거나 합성이 실패하면 None — TTS 문제로 대화 전체가 죽지 않게 격리한다.
        """
        if tts_client is None or not text:
            return None
        try:
            voice = get_persona(speaker).voice_id or 'alloy'
            data = await tts_client.synthesize(text, voice)
            return base64.b64encode(data).decode('ascii')
        except Exception as exc:  # TTS 실패해도 텍스트 대화는 계속
            print(f'[tts] 합성 실패({type(exc).__name__}) — 소리 없이 진행')
            return None

    async def _stream_turn(state: dict, ws: WebSocket, finish: bool = False):
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

        stream_task = asyncio.create_task(_run_stream())
        recv_task = asyncio.create_task(ws.receive_text())
        done, _ = await asyncio.wait(
            {stream_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # 유저가 스트림 도중 끼어듦 → barge-in (부분 발화는 음성 합성하지 않음)
        if recv_task in done and not recv_task.cancelled():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            partial = _strip_speaker_prefix(''.join(parts), name)
            await ws.send_json({'type': 'interrupted', 'speaker': speaker, 'text': partial})
            state = _apply_utterance(state, speaker, partial, interrupted=True)
            return state, recv_task.result()

        # 정상 완료 → 대기 중이던 수신 취소 + 음성 합성
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        text = _strip_speaker_prefix(''.join(parts), name)
        audio = await _synthesize(speaker, text)
        await ws.send_json({'type': 'end', 'speaker': speaker, 'text': text, 'audio': audio})
        return _apply_utterance(state, speaker, text), None

    @app.websocket('/ws')
    async def chat(ws: WebSocket) -> None:
        await ws.accept()
        state = initial_state()
        turn = 0
        pending_user = None  # barge-in으로 받은 유저 발화(다음 턴에 반영)
        try:
            while True:
                if pending_user is not None:
                    state = _add_user(state, pending_user)
                    pending_user = None
                else:
                    try:
                        incoming = await asyncio.wait_for(ws.receive_text(), timeout=radio_sec)
                        state = _add_user(state, incoming)
                    except asyncio.TimeoutError:
                        pass

                finish = turn + 1 >= max_turns
                state, barge = await _stream_turn(state, ws, finish=finish)
                turn += 1

                if barge is not None:
                    pending_user = barge  # 개입 발화를 다음 턴에 반영(세션 이어감)
                    continue
                if finish:
                    await ws.send_json({'type': 'done'})
                    break
        except WebSocketDisconnect:
            return

    return app


# uvicorn 기동용 기본 앱 (실제 LLM/TTS client는 main에서 주입 — 아래는 스텁 폴백)
app = create_app()
