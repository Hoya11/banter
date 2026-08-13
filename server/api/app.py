"""텍스트 실시간 채팅 API (Phase 1) — WebSocket으로 3자 대화를 스트리밍 서빙한다.

유저는 WS로 연결·발화하고, 서버는 화자를 정한 뒤 발화를 토큰 단위로 push한다.
유저가 radio_sec 동안 침묵하면 AI끼리 진행(라디오 모드).
AI 발화가 max_turns에 도달하면 마무리 멘트 후 종료(세션 캡, PRD §5).
프로토콜: {type:'start', speaker} → {type:'token', text}* → {type:'end', speaker, text} → {type:'done'}
"""

import asyncio
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


def _apply_utterance(state: dict, speaker: str, text: str) -> dict:
    """완성된 발화를 상태에 반영한다 (메시지 추가 + 발화 카운트)."""
    personas = {k: dict(v) for k, v in state['personas'].items()}
    if speaker in personas:
        personas[speaker]['speak_count'] += 1
    msg = {'speaker': speaker, 'text': text, 'ts': 0.0, 'interrupted': False}
    return {**state, 'messages': state['messages'] + [msg], 'personas': personas}


def create_app(
    utterance_client=None,
    supervisor_client=None,
    max_turns: int = MAX_TURNS,
    radio_sec: float = RADIO_SEC,
) -> FastAPI:
    """WS 채팅 앱을 만든다. LLM client를 주입받아(테스트는 fake) 엔진을 구동한다."""
    app = FastAPI()

    @app.get('/')
    def index():
        """채팅 UI 페이지를 서빙한다."""
        return FileResponse(WEB_DIR / 'index.html')

    async def _stream_turn(state: dict, ws: WebSocket, finish: bool = False) -> dict:
        loop = asyncio.get_event_loop()
        # 화자 결정 (sync supervisor를 executor로)
        sel = await loop.run_in_executor(None, select_next, state, supervisor_client)
        state = {**state, **sel}
        if finish:
            # 세션 캡 도달 — 의도를 '마무리'로 덮어써 자연스러운 종료 멘트를 유도
            state = {**state, 'current_intent': FINISH_INTENT}
        system, user, speaker = prepare_utterance(state)

        await ws.send_json({'type': 'start', 'speaker': speaker})
        parts: list[str] = []
        if utterance_client is None:
            parts.append(f'({speaker} 발화 스텁)')
            await ws.send_json({'type': 'token', 'text': parts[0]})
        else:
            async for token in utterance_client.complete_stream(system, user):
                parts.append(token)
                await ws.send_json({'type': 'token', 'text': token})

        text = _strip_speaker_prefix(''.join(parts), get_persona(speaker).name)
        await ws.send_json({'type': 'end', 'speaker': speaker, 'text': text})
        return _apply_utterance(state, speaker, text)

    @app.websocket('/ws')
    async def chat(ws: WebSocket) -> None:
        await ws.accept()
        state = initial_state()
        turn = 0
        try:
            while True:
                # 유저 발화 대기 — radio_sec 안에 없으면 라디오 모드
                try:
                    incoming = await asyncio.wait_for(ws.receive_text(), timeout=radio_sec)
                    state = _add_user(state, incoming)
                except asyncio.TimeoutError:
                    pass
                finish = turn + 1 >= max_turns  # 이번 턴이 마지막인가(세션 캡)
                state = await _stream_turn(state, ws, finish=finish)
                turn += 1
                if finish:
                    await ws.send_json({'type': 'done'})
                    break
        except WebSocketDisconnect:
            return

    return app


# uvicorn 기동용 기본 앱 (실제 LLM client는 main에서 주입 — 아래는 스텁 폴백)
app = create_app()
