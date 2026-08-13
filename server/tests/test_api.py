"""WebSocket 스트리밍 채팅 API 배관 테스트.

fake 스트리밍 client로 '유저 발화 → start/token*/end 스트림' 흐름을 검증한다
(LLM 답변 품질이 아니라 WS 프로토콜·엔진 구동·토큰 push 배관).
"""

import asyncio
import base64

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
        end = ws.receive_json()
        assert end['type'] == 'end'
        assert end['speaker'] == 'ai_a'
        assert end['text'] == '안녕'
        assert end['audio'] is None  # tts_client 없으면 오디오 없음


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


def test_ws_ends_after_max_turns():
    # 세션 캡: max_turns 도달 시 done 신호 후 종료 (radio_sec 짧게 → 라디오로 자동 진행)
    app = create_app(FakeStream(), max_turns=2, radio_sec=0.01)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        types = []
        for _ in range(40):
            types.append(ws.receive_json()['type'])
            if types[-1] == 'done':
                break
        assert 'done' in types
        assert types.count('end') == 2  # 딱 2턴 후 종료


class SlowStream:
    """토큰 사이에 지연을 둬 스트리밍 도중 개입할 여지를 만드는 fake."""

    async def complete_stream(self, system: str, user: str):
        for token in ['아', '주', '천', '천', '히']:
            await asyncio.sleep(0.1)
            yield token


def test_ws_barge_in_interrupts_stream():
    # 스트리밍 도중 유저가 끼어들면 interrupted 이벤트가 나와야 한다
    app = create_app(SlowStream(), max_turns=99, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.send_text('시작')  # 첫 유저 발화 → AI 턴 시작
        assert ws.receive_json()['type'] == 'start'
        assert ws.receive_json()['type'] == 'token'  # 첫 토큰 나옴(스트림 진행 중)
        ws.send_text('아니 그게 아니고')  # 발화 도중 끼어들기
        interrupted = False
        for _ in range(20):
            msg = ws.receive_json()
            if msg['type'] == 'interrupted':
                interrupted = True
                break
        assert interrupted


class FakeTTS:
    async def synthesize(self, text: str, voice: str) -> bytes:
        return b'FAKEAUDIO'


def test_ws_end_includes_audio_when_tts():
    # tts_client가 있으면 발화 완성 시 base64 오디오가 실려 온다
    app = create_app(FakeStream(), tts_client=FakeTTS())
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.send_text('하이')
        assert ws.receive_json()['type'] == 'start'
        ws.receive_json()  # token 안
        ws.receive_json()  # token 녕
        end = ws.receive_json()
        assert end['type'] == 'end'
        assert end['audio'] == base64.b64encode(b'FAKEAUDIO').decode('ascii')
