"""텍스트 실시간 채팅 API (Phase 1) — WebSocket으로 3자 대화를 서빙한다.

유저는 WS로 연결·발화하고, 서버는 엔진 턴을 돌려 AI 발화를 push한다.
유저가 RADIO_SEC 동안 침묵하면 AI끼리 진행(라디오 모드).
엔진(build_graph)은 그대로 재사용하고, 이 파일은 '실시간 연결 위에서 엔진을 돌리는 껍데기'다.
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine.graph.graph import build_graph
from engine.graph.state import initial_state

WEB_DIR = Path(__file__).resolve().parents[2] / 'web'
RADIO_SEC = 5.0  # 유저 침묵이 이 시간을 넘으면 AI 턴을 자동 진행(라디오 모드)


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


def create_app(utterance_client=None, supervisor_client=None) -> FastAPI:
    """WS 채팅 앱을 만든다. LLM client를 주입받아(테스트는 fake) 엔진을 구동한다."""
    app = FastAPI()
    graph = build_graph(utterance_client, supervisor_client)

    @app.get('/')
    def index():
        """채팅 UI 페이지를 서빙한다."""
        return FileResponse(WEB_DIR / 'index.html')

    async def _ai_turn(state: dict) -> dict:
        # sync 엔진 호출을 executor로 — 이벤트 루프를 막지 않는다
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, graph.invoke, state)

    @app.websocket('/ws')
    async def chat(ws: WebSocket) -> None:
        await ws.accept()
        state = initial_state()
        try:
            while True:
                # 유저 발화 대기 — RADIO_SEC 안에 없으면 라디오 모드
                try:
                    text = await asyncio.wait_for(ws.receive_text(), timeout=RADIO_SEC)
                    state = _add_user(state, text)
                except asyncio.TimeoutError:
                    pass
                # AI 한 턴 진행 후 push
                state = await _ai_turn(state)
                last = state['messages'][-1]
                await ws.send_json({'speaker': last['speaker'], 'text': last['text']})
        except WebSocketDisconnect:
            return

    return app


# uvicorn 기동용 기본 앱 (실제 LLM client는 main에서 주입 — 아래는 스텁 폴백)
app = create_app()
