"""WebSocket 스트리밍 채팅 API 배관 테스트.

fake 스트리밍 client로 '유저 발화 → start/token*/end 스트림' 흐름을 검증한다
(LLM 답변 품질이 아니라 WS 프로토콜·엔진 구동·토큰 push 배관).
"""

import asyncio
import base64
import json
import threading

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from api.app import create_app
from api.event_log import EventName, JsonlEventRecorder, summarize_hold_latencies


class FakeStream:
    async def complete_stream(self, system: str, user: str):
        for token in ['안', '녕']:
            yield token


def _begin_session(ws):
    ws.send_text(json.dumps({'type': 'session_start'}))
    started = ws.receive_json()
    assert started['type'] == 'session_started'
    assert started['session_id']
    return started


def _start_session(ws):
    hello = ws.receive_json()
    assert hello['type'] == 'hello'
    return hello, _begin_session(ws)


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _gate_first_start(monkeypatch):
    """첫 start는 전송하되 send_json 반환만 늦춰 수신 경합을 만든다."""
    entered = threading.Event()
    cancelled = threading.Event()
    release = threading.Event()
    original_send_json = WebSocket.send_json
    blocked = False

    async def gated_send_json(self, payload, *args, **kwargs):
        nonlocal blocked
        await original_send_json(self, payload, *args, **kwargs)
        if blocked or payload.get('type') != 'start':
            return
        blocked = True
        entered.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            cancelled.set()
            while not release.is_set():
                await asyncio.sleep(0.001)

    monkeypatch.setattr(WebSocket, 'send_json', gated_send_json)
    return entered, cancelled, release


def test_index_exposes_explicit_start_gate():
    response = TestClient(create_app()).get('/')

    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert 'id="session-start"' in response.text
    assert 'id="msg"' in response.text and 'disabled' in response.text
    assert "type: 'session_start'" in response.text


def test_ws_waits_for_explicit_session_start():
    import time

    class RecordingStream:
        def __init__(self):
            self.prompts = []

        async def complete_stream(self, system: str, user: str):
            self.prompts.append(user)
            yield '준비됨'

    stream = RecordingStream()
    app = create_app(stream, max_turns=1, radio_sec=0.01)

    with TestClient(app).websocket_connect('/ws') as ws:
        assert ws.receive_json()['type'] == 'hello'
        ws.send_text(json.dumps({'type': 'say', 'text': '시작 전에 보낸 메시지'}))
        ws.send_text(json.dumps({'type': 'hold'}))
        time.sleep(0.05)
        assert stream.prompts == []

        _begin_session(ws)
        while ws.receive_json()['type'] != 'done':
            pass

    assert len(stream.prompts) == 1
    assert '시작 전에 보낸 메시지' not in stream.prompts[0]


def test_ws_flushes_session_event_when_start_ack_disconnects(tmp_path):
    class DisconnectingWebSocket:
        def __init__(self):
            self.sent = []

        async def accept(self):
            pass

        async def send_json(self, message):
            self.sent.append(message)
            if message['type'] == 'session_started':
                raise WebSocketDisconnect(code=1001)

        async def receive_text(self):
            return json.dumps({'type': 'session_start'})

    path = tmp_path / 'events.jsonl'
    app = create_app(
        FakeStream(),
        event_recorder=JsonlEventRecorder(path),
        run_id='start-disconnect-run',
    )
    endpoint = next(route.endpoint for route in app.routes if route.path == '/ws')
    socket = DisconnectingWebSocket()

    asyncio.run(endpoint(socket))

    assert [message['type'] for message in socket.sent] == ['hello', 'session_started']
    events = _read_events(path)
    assert [event['event'] for event in events] == ['session_started']
    assert events[0]['run_id'] == 'start-disconnect-run'
    assert events[0]['session_id']


def test_ws_new_connection_starts_fresh_session_after_done():
    app = create_app(FakeStream(), max_turns=1, radio_sec=0.01)
    client = TestClient(app)
    session_ids = []

    for _ in range(2):
        with client.websocket_connect('/ws') as ws:
            _, started = _start_session(ws)
            session_ids.append(started['session_id'])
            while ws.receive_json()['type'] != 'done':
                pass

    assert session_ids[0] != session_ids[1]


def test_ws_streams_tokens_then_end():
    app = create_app(FakeStream())  # supervisor None → 기계적 교대(첫 턴 ai_a)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        hello, _ = _start_session(ws)
        assert hello == {'type': 'hello', 'voice': False, 'stt': False}  # 접속 시 모드 안내
        ws.send_text('하이')
        assert ws.receive_json() == {'type': 'start', 'speaker': 'ai_a'}
        assert ws.receive_json() == {'type': 'token', 'text': '안'}
        assert ws.receive_json() == {'type': 'token', 'text': '녕'}
        end = ws.receive_json()
        assert end == {'type': 'end', 'speaker': 'ai_a', 'text': '안녕'}  # audio는 별도 이벤트


def test_ws_json_say_correlates_the_response_start():
    app = create_app(FakeStream(), max_turns=1, radio_sec=100)
    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text(
            json.dumps(
                {
                    'type': 'say',
                    'text': '텍스트로 끼어든다',
                    'client_event_id': 'text-input-1',
                }
            )
        )

        assert ws.receive_json() == {
            'type': 'start',
            'speaker': 'ai_a',
            'after_client_event_id': 'text-input-1',
        }


def test_ws_start_send_does_not_clear_a_newer_say_release(monkeypatch):
    entered, cancelled, release = _gate_first_start(monkeypatch)
    app = create_app(FakeStream(), unified=True, max_turns=5, radio_sec=100)

    with TestClient(app).websocket_connect('/ws') as ws:
        try:
            _start_session(ws)
            ws.send_text(
                json.dumps(
                    {
                        'type': 'say',
                        'text': '첫 입력',
                        'client_event_id': 'rapid-say-1',
                    }
                )
            )
            first_start = ws.receive_json()
            assert first_start['after_client_event_id'] == 'rapid-say-1'
            assert entered.wait(2)

            ws.send_text(
                json.dumps(
                    {
                        'type': 'say',
                        'text': '더 최신 입력',
                        'client_event_id': 'rapid-say-2',
                    }
                )
            )
            assert cancelled.wait(2)
            release.set()

            for _ in range(30):
                message = ws.receive_json()
                if message['type'] == 'start':
                    break
            assert message['type'] == 'start'
            assert message['after_client_event_id'] == 'rapid-say-2'
        finally:
            release.set()


def test_ws_blocked_start_does_not_resolve_a_new_legacy_hold(
    tmp_path, monkeypatch
):
    entered, cancelled, release = _gate_first_start(monkeypatch)
    path = tmp_path / 'events.jsonl'
    app = create_app(
        FakeStream(),
        unified=True,
        max_turns=5,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='legacy-hold-race',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        try:
            _start_session(ws)
            ws.send_text(
                json.dumps(
                    {
                        'type': 'say',
                        'text': '첫 입력',
                        'client_event_id': 'legacy-race-say',
                    }
                )
            )
            assert ws.receive_json()['type'] == 'start'
            assert entered.wait(2)

            ws.send_text(json.dumps({'type': 'hold'}))
            assert cancelled.wait(2)
            release.set()
            while ws.receive_json()['type'] != 'interrupted':
                pass
            ws.send_text(json.dumps({'type': 'hold_off'}))
            ws.send_text(json.dumps({'type': 'say', 'text': '계속 진행'}))
            while ws.receive_json()['type'] != 'start':
                pass
        finally:
            release.set()

    sample = summarize_hold_latencies(path)[0]
    assert sample.hold_to_cancellation_ms is not None
    assert sample.hold_to_next_turn_ms is None
    assert sample.missing == ()


def test_ws_stub_when_no_client():
    app = create_app()  # utterance client 없음 → 발화 스텁
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
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
        _start_session(ws)
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
        _start_session(ws)
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
        _start_session(ws)
        ws.send_text('하이')
        types = [ws.receive_json()['type'] for _ in range(3)]
        assert types == ['start', 'token', 'error']  # end 없음


def test_ws_done_even_if_barge_on_final_turn():
    # 마무리 턴에 끼어들어도 세션 캡이 우선 — done이 반드시 온다
    app = create_app(SlowStream(), max_turns=1, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
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


def test_ws_unified_streams_header_then_sentences():
    # v2 통합 생성 스트리밍: 첫 줄 헤더로 화자 확정(start) → 문장 단위 audio (호출 -1)
    class UnifiedStream:
        async def complete_stream(self, system: str, user: str):
            for token in ['{"next_speaker": "ai_b", "intent": "리액션", "topic": "저녁"}\n', '배고프', '다. ', '뭐 먹지?']:
                yield token

    app = create_app(UnifiedStream(), tts_client=FakeTTS(), max_turns=1, radio_sec=100, unified=True)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('하이')
        events = []
        for _ in range(30):
            msg = ws.receive_json()
            events.append(msg)
            if msg['type'] == 'audio':
                ws.send_text(json.dumps({'type': 'played', 'seq': msg['seq']}))
            if msg['type'] == 'done':
                break
        start = next(e for e in events if e['type'] == 'start')
        assert start['speaker'] == 'ai_b'  # 헤더가 정한 화자
        audio_texts = [e['text'] for e in events if e['type'] == 'audio']
        assert audio_texts == ['배고프다.', '뭐 먹지?']  # 헤더 제외, 문장 단위
        tokens = [e['text'] for e in events if e['type'] == 'token']
        assert all('next_speaker' not in t for t in tokens)  # 헤더는 화면에 새지 않는다


def test_ws_hold_grabs_floor_until_say():
    # hold를 받으면 전사 결과가 올 때까지 새 AI 턴을 열지 않는다.
    import time as _time

    app = create_app(SlowStream(), tts_client=FakeTTS(), max_turns=9, radio_sec=0.05)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass  # 스트림 진행 중 확인
        ws.send_text(
            json.dumps({'type': 'hold', 'client_event_id': 'floor-hold-1'})
        )
        msg = ws.receive_json()
        while msg['type'] != 'interrupted':  # 내용 없이도 즉시 발화 중단
            msg = ws.receive_json()
        _time.sleep(0.5)  # radio_sec(0.05)의 10배 — hold 없으면 새 턴이 이미 열렸을 시간
        ws.send_text(json.dumps({'type': 'say', 'text': '이제 말한다'}))
        starts = 0
        for _ in range(30):
            msg = ws.receive_json()
            if msg['type'] == 'start':
                starts += 1
            if msg['type'] == 'audio':
                ws.send_text(json.dumps({'type': 'played', 'seq': msg['seq']}))
            if msg['type'] == 'end':
                break
        assert starts == 1  # hold 동안 몰래 열린 턴 없음 — say 후의 응답 턴 하나뿐


def test_ws_hold_records_invalidation_and_next_turn_latency(tmp_path):
    path = tmp_path / 'events.jsonl'
    recorder = JsonlEventRecorder(path)
    app = create_app(
        SlowStream(),
        max_turns=9,
        radio_sec=100,
        event_recorder=recorder,
        run_id='integration-run',
    )
    client = TestClient(app)

    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'measured-hold-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        while ws.receive_json()['type'] != 'interrupted':
            pass
        ws.send_text(json.dumps({'type': 'say', 'text': '이제 말한다'}))
        while True:
            next_start = ws.receive_json()
            if next_start['type'] == 'start':
                break
        assert next_start['after_client_event_id'] == 'measured-hold-1'
        while ws.receive_json()['type'] != 'end':
            pass

    samples = summarize_hold_latencies(path)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.run_id == 'integration-run'
    assert sample.generation_id == 'generation-1'
    assert sample.hold_to_invalidation_ms is not None
    assert sample.hold_to_invalidation_ms >= 0
    assert sample.hold_to_next_turn_ms is not None
    assert sample.hold_to_next_turn_ms >= sample.hold_to_invalidation_ms
    assert sample.missing == ()


def test_ws_hold_during_audio_playback_records_next_turn_without_invalidation(tmp_path):
    path = tmp_path / 'events.jsonl'
    recorder = JsonlEventRecorder(path)
    app = create_app(
        FakeStream(),
        tts_client=FakeTTS(),
        max_turns=9,
        radio_sec=100,
        ack_sec=100,
        event_recorder=recorder,
        run_id='voice-run',
    )
    client = TestClient(app)

    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        seen = set()
        while not {'end', 'audio'}.issubset(seen):
            seen.add(ws.receive_json()['type'])
        ws.send_text(json.dumps({'type': 'hold'}))
        ws.send_text(json.dumps({'type': 'say', 'text': '재생 중에 끼어든다'}))
        while ws.receive_json()['type'] != 'start':
            pass

    samples = summarize_hold_latencies(path)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.run_id == 'voice-run'
    assert sample.generation_id is None
    assert sample.invalidation_expected is False
    assert sample.hold_to_invalidation_ms is None
    assert sample.hold_to_next_turn_ms is not None
    assert sample.missing == ()


def test_ws_hold_records_client_audio_stop_once_and_ignores_stale_cancel(tmp_path):
    path = tmp_path / 'events.jsonl'
    app = create_app(
        FakeStream(),
        max_turns=9,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='client-stop-run',
    )
    first_hold = {
        'type': 'hold',
        'client_event_id': 'gesture-1',
        'audio_stop': {
            'outcome': 'paused',
            'elapsed_ms': 12.3456,
            'sequence': 7,
        },
    }

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('events must not contain this user text')
        while ws.receive_json()['type'] != 'end':
            pass

        ws.send_text(json.dumps(first_hold))
        ws.send_text(json.dumps(first_hold))
        ws.send_text(json.dumps({'type': 'hold_off', 'client_event_id': 'gesture-1'}))
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'gesture-2',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        ws.send_text(json.dumps({'type': 'hold_off', 'client_event_id': 'gesture-1'}))
        stale_audio = base64.b64encode(b'raw-audio-must-not-be-recorded').decode('ascii')
        ws.send_text(
            json.dumps(
                {
                    'type': 'voice',
                    'client_event_id': 'gesture-1',
                    'audio': stale_audio,
                    'mime': 'audio/webm',
                }
            )
        )
        ws.send_text(json.dumps({'type': 'say', 'text': '새 hold만 완료한다'}))
        while ws.receive_json()['type'] != 'start':
            pass

    events = _read_events(path)
    holds = [event for event in events if event['event'] == EventName.HOLD_RECEIVED]
    stops = [event for event in events if event['event'] == EventName.AUDIO_STOPPED]
    cancellations = [
        event for event in events if event['event'] == EventName.HOLD_CANCELLED
    ]
    next_turns = [
        event for event in events if event['event'] == EventName.NEXT_TURN_STARTED
    ]

    assert [event.get('client_event_id') for event in holds] == ['gesture-1', 'gesture-2']
    assert [event.get('client_event_id') for event in stops] == ['gesture-1', 'gesture-2']
    assert stops[0]['outcome'] == 'paused'
    assert stops[0]['client_elapsed_ms'] == 12.346
    assert stops[0]['segment_id'] == 'audio-7'
    assert stops[1]['outcome'] == 'idle'
    assert [event.get('client_event_id') for event in cancellations] == ['gesture-1']
    assert next_turns[-1]['client_event_id'] == 'gesture-2'
    assert events.index(stops[0]) == events.index(holds[0]) + 1
    serialized = path.read_text(encoding='utf-8')
    assert 'events must not contain this user text' not in serialized
    assert '새 hold만 완료한다' not in serialized
    assert stale_audio not in serialized


def test_ws_malformed_audio_stop_is_dropped_without_losing_floor_control(tmp_path):
    path = tmp_path / 'events.jsonl'
    app = create_app(
        SlowStream(),
        max_turns=9,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='invalid-stop-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'invalid-stop-1',
                    'audio_stop': {
                        'outcome': 'paused',
                        'elapsed_ms': 60_001,
                        'segment_id': '../not-safe',
                    },
                }
            )
        )
        while ws.receive_json()['type'] != 'interrupted':
            pass
        ws.send_text(json.dumps({'type': 'say', 'text': '측정값은 버리고 대화는 계속'}))
        while ws.receive_json()['type'] != 'start':
            pass

    events = _read_events(path)
    holds = [event for event in events if event['event'] == EventName.HOLD_RECEIVED]
    stops = [event for event in events if event['event'] == EventName.AUDIO_STOPPED]
    next_turns = [
        event for event in events if event['event'] == EventName.NEXT_TURN_STARTED
    ]
    assert holds[-1]['client_event_id'] == 'invalid-stop-1'
    assert stops == []
    assert next_turns[-1]['client_event_id'] == 'invalid-stop-1'


def test_ws_session_event_records_safe_reproducibility_metadata(tmp_path):
    path = tmp_path / 'events.jsonl'
    app = create_app(
        FakeStream(),
        max_turns=7,
        radio_sec=1.25,
        ack_sec=2.5,
        unified=True,
        event_recorder=JsonlEventRecorder(path),
        run_id='metadata-run',
        run_metadata={
            'baseline_id': 'push-to-talk-v1',
            'utterance_model': 'fake-model',
            'api_key': 'must-not-be-recorded',
            'transcript': 'raw conversation must not be recorded',
        },
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _, started = _start_session(ws)
        assert set(started) == {'type', 'session_id'}

    events = _read_events(path)
    session_event = next(
        event for event in events if event['event'] == EventName.SESSION_STARTED
    )
    assert session_event['metadata'] == {
        'baseline_id': 'push-to-talk-v1',
        'utterance_model': 'fake-model',
        'protocol_version': 1,
        'voice_mode': False,
        'radio_sec': 1.25,
        'ack_sec': 2.5,
        'max_turns': 7,
        'unified': True,
    }
    serialized = path.read_text(encoding='utf-8')
    assert 'must-not-be-recorded' not in serialized
    assert 'raw conversation must not be recorded' not in serialized


def test_ws_simultaneous_hold_and_stream_completion_has_no_stale_generation(
    tmp_path, monkeypatch
):
    class ImmediateStream:
        async def complete_stream(self, system: str, user: str):
            yield '완료.'

    original_wait = asyncio.wait
    forced_once = False

    async def force_both_ready_once(
        tasks, *, timeout=None, return_when=asyncio.ALL_COMPLETED
    ):
        nonlocal forced_once
        if not forced_once and return_when == asyncio.FIRST_COMPLETED:
            forced_once = True
            return await original_wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
        return await original_wait(tasks, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(asyncio, 'wait', force_both_ready_once)
    path = tmp_path / 'events.jsonl'
    app = create_app(
        ImmediateStream(),
        max_turns=9,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='race-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        assert ws.receive_json()['type'] == 'start'
        ws.send_text(json.dumps({'type': 'hold'}))
        while ws.receive_json()['type'] != 'end':
            pass
        ws.send_text(json.dumps({'type': 'say', 'text': '이제 말한다'}))
        while ws.receive_json()['type'] != 'start':
            pass
        while ws.receive_json()['type'] != 'end':
            pass

    assert forced_once
    sample = summarize_hold_latencies(path)[0]
    assert sample.generation_id is None
    assert sample.invalidation_expected is False
    assert sample.missing == ()


def test_ws_hold_off_finishes_cancelled_sample(tmp_path):
    path = tmp_path / 'events.jsonl'
    app = create_app(
        SlowStream(),
        max_turns=9,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='hold-off-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass
        ws.send_text(json.dumps({'type': 'hold'}))
        while ws.receive_json()['type'] != 'interrupted':
            pass
        ws.send_text(json.dumps({'type': 'hold_off'}))
        ws.send_text(json.dumps({'type': 'say', 'text': '다시 시작'}))
        while ws.receive_json()['type'] != 'start':
            pass

    sample = summarize_hold_latencies(path)[0]
    assert sample.hold_to_invalidation_ms is not None
    assert sample.hold_to_cancellation_ms is not None
    assert sample.hold_to_next_turn_ms is None
    assert sample.missing == ()


def test_ws_stt_failure_finishes_cancelled_sample(tmp_path):
    class FailingSTT:
        async def transcribe(self, audio: bytes, mime: str = 'audio/webm') -> str:
            raise RuntimeError('stt failed')

    path = tmp_path / 'events.jsonl'
    app = create_app(
        SlowStream(),
        stt_client=FailingSTT(),
        max_turns=9,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='stt-failure-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'stt-failure-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        while ws.receive_json()['type'] != 'interrupted':
            pass
        audio = base64.b64encode(b'A' * 8000).decode('ascii')
        ws.send_text(
            json.dumps(
                {
                    'type': 'voice',
                    'client_event_id': 'stt-failure-1',
                    'audio': audio,
                    'mime': 'audio/webm',
                }
            )
        )
        ws.send_text(json.dumps({'type': 'say', 'text': '텍스트로 다시 시작'}))
        while ws.receive_json()['type'] != 'start':
            pass

    sample = summarize_hold_latencies(path)[0]
    assert sample.hold_to_cancellation_ms is not None
    assert sample.hold_to_next_turn_ms is None
    assert sample.missing == ()
    events = _read_events(path)
    cancelled = next(
        event for event in events if event['event'] == EventName.HOLD_CANCELLED
    )
    assert cancelled['client_event_id'] == 'stt-failure-1'
    assert audio not in path.read_text(encoding='utf-8')


def test_ws_voice_for_cancelled_hold_is_not_transcribed(tmp_path):
    class RecordingSTT:
        def __init__(self):
            self.calls = 0

        async def transcribe(self, audio: bytes, mime: str = 'audio/webm') -> str:
            self.calls += 1
            return '취소된 음성'

    path = tmp_path / 'events.jsonl'
    stt = RecordingSTT()
    app = create_app(
        FakeStream(),
        stt_client=stt,
        max_turns=9,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='cancelled-voice-run',
    )
    audio = base64.b64encode(b'A' * 8000).decode('ascii')

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'cancelled-hold-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        ws.send_text(
            json.dumps(
                {'type': 'hold_off', 'client_event_id': 'cancelled-hold-1'}
            )
        )
        ws.send_text(
            json.dumps(
                {
                    'type': 'voice',
                    'client_event_id': 'cancelled-hold-1',
                    'audio': audio,
                    'mime': 'audio/webm',
                }
            )
        )
        ws.send_text(json.dumps({'type': 'say', 'text': '텍스트만 반영'}))
        while ws.receive_json()['type'] != 'start':
            pass

    assert stt.calls == 0
    sample = summarize_hold_latencies(path)[0]
    assert sample.client_event_id == 'cancelled-hold-1'
    assert sample.hold_to_cancellation_ms is not None
    assert sample.missing == ()


def test_ws_hold_timeout_is_recorded_as_cancellation(tmp_path, monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, 'HOLD_SEC', 0.01)
    path = tmp_path / 'events.jsonl'
    app = create_app(
        FakeStream(),
        max_turns=1,
        radio_sec=1.0,
        event_recorder=JsonlEventRecorder(path),
        run_id='hold-timeout-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'timeout-hold-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        while ws.receive_json()['type'] != 'done':
            pass

    sample = summarize_hold_latencies(path)[0]
    assert sample.hold_to_cancellation_ms is not None
    assert sample.hold_to_cancellation_ms < 200
    assert sample.hold_to_next_turn_ms is None
    assert sample.missing == ()


def test_ws_final_turn_hold_is_closed_as_cancellation(tmp_path):
    path = tmp_path / 'events.jsonl'
    app = create_app(
        SlowStream(),
        max_turns=1,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='final-hold-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'final-hold-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        while ws.receive_json()['type'] != 'done':
            pass

    sample = summarize_hold_latencies(path)[0]
    assert sample.hold_to_invalidation_ms is not None
    assert sample.hold_to_cancellation_ms is not None
    assert sample.hold_to_next_turn_ms is None
    assert sample.missing == ()


def test_ws_hold_drops_tts_that_finishes_after_invalidation(tmp_path):
    import threading

    class SentenceThenWaitStream:
        def __init__(self):
            self.calls = 0

        async def complete_stream(self, system: str, user: str):
            self.calls += 1
            if self.calls == 1:
                yield '이전 문장. '
                await asyncio.sleep(1)
                yield '늦은 문장.'
            else:
                yield '새 응답.'

    class BlockingTTS:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        async def synthesize(self, text: str, voice: str, speed=None) -> bytes:
            self.started.set()
            while not self.release.is_set():
                await asyncio.sleep(0.005)
            return b'audio'

    path = tmp_path / 'events.jsonl'
    stream = SentenceThenWaitStream()
    tts = BlockingTTS()
    app = create_app(
        stream,
        tts_client=tts,
        max_turns=3,
        radio_sec=100,
        event_recorder=JsonlEventRecorder(path),
        run_id='late-tts-run',
    )

    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('시작')
        while ws.receive_json()['type'] != 'token':
            pass
        assert tts.started.wait(timeout=1)
        ws.send_text(
            json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'late-tts-hold-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        while ws.receive_json()['type'] != 'interrupted':
            pass
        tts.release.set()
        ws.send_text(json.dumps({'type': 'say', 'text': '새 입력'}))

        audio_texts = []
        for _ in range(30):
            message = ws.receive_json()
            if message['type'] == 'audio':
                audio_texts.append(message['text'])
                ws.send_text(json.dumps({'type': 'played', 'seq': message['seq']}))
                if message['text'] == '새 응답.':
                    break

    assert audio_texts == ['새 응답.']
    events = _read_events(path)
    assert any(event['event'] == EventName.LATE_AUDIO_DROPPED for event in events)


def test_ws_event_recorder_failure_does_not_break_chat():
    class BrokenRecorder:
        def record(self, *args, **kwargs):
            raise OSError('disk full')

    app = create_app(FakeStream(), event_recorder=BrokenRecorder())
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
        ws.send_text('하이')
        assert ws.receive_json()['type'] == 'start'


class FakeTTS:
    def __init__(self):
        self.calls = []  # (text, voice, speed) — 라우팅 검증용

    async def synthesize(self, text: str, voice: str, speed=None) -> bytes:
        self.calls.append((text, voice, speed))
        return b'FAKEAUDIO'


def test_ws_audio_sequence_resets_on_new_session_connection():
    app = create_app(FakeStream(), tts_client=FakeTTS(), max_turns=1, radio_sec=0.01)
    client = TestClient(app)
    first_audio_sequences = []

    for _ in range(2):
        with client.websocket_connect('/ws') as ws:
            _start_session(ws)
            while True:
                msg = ws.receive_json()
                if msg['type'] == 'audio':
                    first_audio_sequences.append(msg['seq'])
                    ws.send_text(json.dumps({'type': 'played', 'seq': msg['seq']}))
                if msg['type'] == 'done':
                    break

    assert first_audio_sequences == [1, 1]


def test_ws_audio_follows_end_when_tts():
    # tts_client가 있으면 별도 audio 이벤트(자막용 text 포함)가 온다 (비차단 TTS —
    # 문장 단위 합성이 스트림과 동시에 돌아 end와의 순서는 고정되지 않는다)
    app = create_app(FakeStream(), tts_client=FakeTTS())
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        hello, _ = _start_session(ws)
        assert hello == {'type': 'hello', 'voice': True, 'stt': False}  # 음성 모드 안내
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
    # 두 문장 발화는 audio 이벤트 2개로 나가고 둘째는 cont=True다.
    class TwoSentenceStream:
        async def complete_stream(self, system: str, user: str):
            for token in ['첫 문', '장이다. ', '둘째 문', '장이다.', ' 셋째는 상한.']:
                yield token

    app = create_app(TwoSentenceStream(), tts_client=FakeTTS(), max_turns=1, radio_sec=100)
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        _start_session(ws)
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
        _start_session(ws)
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
        _start_session(ws)
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
        _start_session(ws)
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
    # voice 메시지는 STT와 you 응답을 거쳐 기존 say 경로에 합류한다.
    import json as _json

    AUDIO = b'A' * 8000  # MIN_VOICE_BYTES 이상 (무음 필터 통과)

    class FakeSTT:
        async def transcribe(self, audio: bytes, mime: str = 'audio/webm') -> str:
            assert audio == AUDIO
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
        hello, _ = _start_session(ws)
        assert hello['stt'] is True  # FE가 말하기 버튼을 보여줄 근거
        b64 = base64.b64encode(AUDIO).decode('ascii')
        ws.send_text(
            _json.dumps(
                {
                    'type': 'hold',
                    'client_event_id': 'voice-hold-1',
                    'audio_stop': {'outcome': 'idle'},
                }
            )
        )
        ws.send_text(
            _json.dumps(
                {
                    'type': 'voice',
                    'client_event_id': 'voice-hold-1',
                    'audio': b64,
                    'mime': 'audio/webm',
                }
            )
        )
        assert ws.receive_json() == {'type': 'you', 'text': '음성으로 말했어요'}  # 전사 echo
        start = ws.receive_json()
        assert start['type'] == 'start'  # 전사가 유저 발화로 처리돼 턴 시작
        assert start['after_client_event_id'] == 'voice-hold-1'
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
        _start_session(ws)
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
        _start_session(ws)
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
