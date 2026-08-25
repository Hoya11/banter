"""WebSocket 스트리밍 채팅 API 배관 테스트.

fake 스트리밍 client로 '유저 발화 → start/token*/end 스트림' 흐름을 검증한다
(LLM 답변 품질이 아니라 WS 프로토콜·엔진 구동·토큰 push 배관).
"""

import asyncio
import base64
import json

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
        assert ws.receive_json() == {'type': 'hello', 'voice': False, 'stt': False}  # 접속 시 모드 안내
        ws.send_text('하이')
        assert ws.receive_json() == {'type': 'start', 'speaker': 'ai_a'}
        assert ws.receive_json() == {'type': 'token', 'text': '안'}
        assert ws.receive_json() == {'type': 'token', 'text': '녕'}
        end = ws.receive_json()
        assert end == {'type': 'end', 'speaker': 'ai_a', 'text': '안녕'}  # audio는 별도 이벤트


def test_ws_stub_when_no_client():
    app = create_app()  # utterance client 없음 → 발화 스텁
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
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
        ws.receive_json()  # hello
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


class BrokenStream:
    """첫 토큰 후 터지는 fake — 스트림 중간 실패 재현."""

    async def complete_stream(self, system: str, user: str):
        yield '안'
        raise RuntimeError('LLM 500 mid-stream')


def test_ws_stream_error_skips_turn():
    # 스트림 중간 실패 → end가 아니라 error, 부분 발화가 확정되지 않는다
    app = create_app(BrokenStream(), max_turns=1, radio_sec=0.01)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')
        types = [ws.receive_json()['type'] for _ in range(3)]
        assert types == ['start', 'token', 'error']  # end 없음


def test_ws_done_even_if_barge_on_final_turn():
    # 마무리 턴에 끼어들어도 세션 캡이 우선 — done이 반드시 온다
    app = create_app(SlowStream(), max_turns=1, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('시작')
        assert ws.receive_json()['type'] == 'start'
        assert ws.receive_json()['type'] == 'token'
        ws.send_text('잠깐만')  # 마지막(마무리) 턴에 barge-in
        types = []
        for _ in range(20):
            types.append(ws.receive_json()['type'])
            if types[-1] == 'done':
                break
        assert 'done' in types


class FakeTTS:
    def __init__(self):
        self.calls = []  # (text, voice, speed) — 라우팅 검증용

    async def synthesize(self, text: str, voice: str, speed=None) -> bytes:
        self.calls.append((text, voice, speed))
        return b'FAKEAUDIO'


def test_ws_audio_follows_end_when_tts():
    # tts_client가 있으면 별도 audio 이벤트(자막용 text 포함)가 온다 (비차단 TTS —
    # 문장 단위 합성이 스트림과 동시에 돌아 end와의 순서는 고정되지 않는다)
    app = create_app(FakeStream(), tts_client=FakeTTS())
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        assert ws.receive_json() == {'type': 'hello', 'voice': True, 'stt': False}  # 음성 모드 안내
        ws.send_text('하이')
        audio = None
        for _ in range(10):
            msg = ws.receive_json()
            if msg['type'] == 'audio':
                audio = msg
                break
        assert audio is not None
        assert audio['speaker'] == 'ai_a'
        assert audio['text'] == '안녕'  # FE가 재생 시점에 표시할 자막
        assert audio['audio'] == base64.b64encode(b'FAKEAUDIO').decode('ascii')


def test_ws_sentence_streaming_emits_audio_per_sentence():
    # 문장 단위 flush(§3.1): 두 문장 발화 → audio 이벤트 2개(둘째는 cont=True)
    class TwoSentenceStream:
        async def complete_stream(self, system: str, user: str):
            for token in ['첫 문', '장이다. ', '둘째 문', '장이다.', ' 셋째는 상한.']:
                yield token

    app = create_app(TwoSentenceStream(), tts_client=FakeTTS(), max_turns=1, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')
        audios = []
        for _ in range(30):
            msg = ws.receive_json()
            if msg['type'] == 'audio':
                audios.append(msg)
                ws.send_text(json.dumps({'type': 'played', 'seq': msg['seq']}))
            if msg['type'] == 'done':
                break
        assert [a['text'] for a in audios] == ['첫 문장이다.', '둘째 문장이다.']  # 상한 2 적용
        assert [a['cont'] for a in audios] == [False, True]  # 같은 발화의 이어짐 표시


def test_ws_finish_turn_audio_sent_before_done():
    # 세션 후반 유저 개입 → 마무리 턴이 실시간 경로 → 그래도 audio가 done보다 먼저 온다
    # (TTS drain — 마무리 멘트 유실 방지, 검수 🔴2 회귀)
    import json as _json

    app = create_app(FakeStream(), tts_client=FakeTTS(), max_turns=2, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')  # 턴0 (실시간)
        types = []
        while True:
            msg = ws.receive_json()
            types.append(msg['type'])
            if msg['type'] == 'audio':
                ws.send_text(_json.dumps({'type': 'played', 'seq': msg['seq']}))
                break
        ws.send_text('한마디 더')  # prefetch 폐기 → 마무리 턴 실시간 경로
        while types[-1] != 'done':
            msg = ws.receive_json()
            types.append(msg['type'])
            if msg['type'] == 'audio':
                ws.send_text(_json.dumps({'type': 'played', 'seq': msg['seq']}))
        assert types.count('audio') == 2  # 마무리 멘트의 오디오도 전송됨
        assert types.index('done') > len(types) - 1 - types[::-1].index('audio')  # audio가 done보다 앞


def test_ws_completes_without_any_acks():
    # ack이 전부 유실돼도 ack_sec 후 장부 리셋 — 세션이 영구 지연 없이 완주 (검수 🟡3 회귀)
    app = create_app(FakeStream(), tts_client=FakeTTS(), max_turns=3, radio_sec=0.05, ack_sec=0.2)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')
        types = []
        for _ in range(60):
            types.append(ws.receive_json()['type'])
            if types[-1] == 'done':
                break
        assert 'done' in types  # ack 0개여도 완주


def test_ws_user_say_discards_prefetch_and_reaches_prompt():
    # 유저 발화가 오면 prefetch를 버리고, 다음 턴 프롬프트에 그 발화가 실제로 들어간다
    import json as _json

    class RecStream:
        def __init__(self):
            self.prompts = []

        async def complete_stream(self, system: str, user: str):
            self.prompts.append(user)
            yield '응'

    rec = RecStream()
    app = create_app(rec, tts_client=FakeTTS(), max_turns=5, radio_sec=0.5)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')
        msg = ws.receive_json()
        while msg['type'] != 'audio':
            msg = ws.receive_json()
        ws.send_text(_json.dumps({'type': 'played', 'seq': msg['seq']}))
        ws.send_text(_json.dumps({'type': 'say', 'text': '프리페치버려라42'}))
        started = False  # say 이후 새 턴(start)을 보고 그 턴의 end까지 기다린다
        for _ in range(30):
            msg = ws.receive_json()
            if msg['type'] == 'start':
                started = True
            if msg['type'] == 'end' and started:
                break
        assert any('프리페치버려라42' in p for p in rec.prompts)  # 유저 발화가 프롬프트에 반영


def test_ws_voice_message_transcribed_and_joins_as_say():
    # 🎤 voice 메시지 → STT 전사 → 'you' echo → 유저 발화로 대화 합류 (기존 say 파이프 재사용)
    import json as _json

    class FakeSTT:
        async def transcribe(self, audio: bytes, mime: str = 'audio/webm') -> str:
            assert audio == b'AUDIODATA'
            return '음성으로 말했어요'

    class RecStream:
        def __init__(self):
            self.prompts = []

        async def complete_stream(self, system: str, user: str):
            self.prompts.append(user)
            yield '응'

    rec = RecStream()
    app = create_app(rec, stt_client=FakeSTT(), max_turns=5, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        hello = ws.receive_json()
        assert hello['stt'] is True  # FE가 🎤 버튼을 보여줄 근거
        b64 = base64.b64encode(b'AUDIODATA').decode('ascii')
        ws.send_text(_json.dumps({'type': 'voice', 'audio': b64, 'mime': 'audio/webm'}))
        assert ws.receive_json() == {'type': 'you', 'text': '음성으로 말했어요'}  # 전사 echo
        assert ws.receive_json()['type'] == 'start'  # 전사가 유저 발화로 처리돼 턴 시작
        for _ in range(5):
            if ws.receive_json()['type'] == 'end':
                break
        assert any('음성으로 말했어요' in p for p in rec.prompts)  # 프롬프트 반영


def test_ws_prefetch_serves_next_turn_without_tokens():
    # 음성 모드에서 두 번째 턴부터는 prefetch 완성품이 방출된다
    # — 실시간 스트리밍이 아니므로 token 이벤트는 첫 턴에서만 나온다
    import json as _json

    app = create_app(FakeStream(), tts_client=FakeTTS(), max_turns=3, radio_sec=0.05)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')
        types = []
        for _ in range(40):
            msg = ws.receive_json()
            types.append(msg['type'])
            if msg['type'] == 'audio':
                ws.send_text(_json.dumps({'type': 'played'}))
            if msg['type'] == 'done':
                break
        assert types.count('end') == 3  # 3턴 모두 완주
        assert types.count('audio') == 3  # 각 턴에 자막+오디오
        assert types.count('token') == 2  # FakeStream 토큰 2개 — 첫(실시간) 턴만


def test_ws_waits_played_ack_before_next_radio_turn():
    # 음성 동기: played ack이 와야 다음 라디오 턴이 진행된다 (재생 페이스 조율)
    import json as _json

    app = create_app(FakeStream(), tts_client=FakeTTS(), max_turns=2, radio_sec=0.05)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        ws.receive_json()  # hello
        ws.send_text('하이')
        msg = ws.receive_json()
        while msg['type'] != 'audio':  # 합성이 스트림과 동시라 end와 순서 비고정
            msg = ws.receive_json()
        ws.send_text(_json.dumps({'type': 'played'}))  # 재생 완료 ack → 다음 턴 허용
        types = []
        for _ in range(20):
            msg = ws.receive_json()
            types.append(msg['type'])
            if msg['type'] == 'audio':
                ws.send_text(_json.dumps({'type': 'played'}))
            if msg['type'] == 'done':
                break
        assert 'done' in types  # ack 기반 페이스로 세션이 정상 완주
