"""WebSocket 채팅 API 배관 테스트.

fake utterance client로 '유저 발화 → 서버가 AI 발화 push' 흐름을 검증한다
(LLM 답변 품질이 아니라 WS 연결·엔진 구동·push 배관).
"""

from fastapi.testclient import TestClient

from api.app import create_app


class FakeUtter:
    def complete(self, system: str, user: str) -> str:
        return '테스트 발화'


def test_ws_pushes_ai_utterance_after_user_message():
    app = create_app(FakeUtter())  # supervisor None → 기계적 교대
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.send_text('안녕')
        data = ws.receive_json()
        assert data['speaker'] in ('ai_a', 'ai_b')
        assert data['text'] == '테스트 발화'
