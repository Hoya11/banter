"""WebSocket 스트리밍 채팅 API 배관 테스트.

fake 스트리밍 client로 '유저 발화 → start/token*/end 스트림' 흐름을 검증한다
(LLM 답변 품질이 아니라 WS 프로토콜·엔진 구동·토큰 push 배관).
"""

from fastapi.testclient import TestClient

from api.app import create_app


class FakeStream:
    async def complete_stream(self, system: str, user: str):
        for token in ['안', '녕']:
            yield token


def test_ws_streams_tokens_then_end():
    app = create_app(FakeStream())  # supervisor None → 기계적 교대(첫 턴 ai_a)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.send_text('하이')
        assert ws.receive_json() == {'type': 'start', 'speaker': 'ai_a'}
        assert ws.receive_json() == {'type': 'token', 'text': '안'}
        assert ws.receive_json() == {'type': 'token', 'text': '녕'}
        assert ws.receive_json() == {'type': 'end', 'speaker': 'ai_a', 'text': '안녕'}


def test_ws_stub_when_no_client():
    app = create_app()  # utterance client 없음 → 발화 스텁
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.send_text('하이')
        assert ws.receive_json()['type'] == 'start'
        assert ws.receive_json()['type'] == 'token'
        end = ws.receive_json()
        assert end['type'] == 'end'
        assert '스텁' in end['text']
