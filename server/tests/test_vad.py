"""선택형 VAD capability와 기존 발화권 경로의 관측 격리를 검증한다."""

import json

import pytest
from fastapi.testclient import TestClient
from websockets.exceptions import ConnectionClosedError

from api.app import MAX_TURNS, VAD_CONFIG, _parse_incoming, create_app
from api.event_log import JsonlEventRecorder
from test_api import (
    STREAMING_STT_CHUNKS,
    FakeStream,
    FakeStreamingSTTClient,
    FakeStreamingSTTTurn,
    FakeTTS,
    _send_hold,
    _send_stream_chunk,
    _send_stream_commit,
    _send_stream_start,
    _start_session,
)


def _observe(ws, client_event_id, milestone):
    ws.send_json({
        'type': 'vad_observed',
        'client_event_id': client_event_id,
        'milestone': milestone,
    })


def test_vad_requires_explicit_streaming_provider():
    with pytest.raises(ValueError, match='streaming_stt_client'):
        create_app(interaction_mode='vad')
    with pytest.raises(ValueError, match='interaction_mode'):
        create_app(interaction_mode='unknown')


def test_vad_capability_does_not_connect_provider():
    provider = FakeStreamingSTTClient()
    app = create_app(streaming_stt_client=provider, interaction_mode='vad')
    with TestClient(app).websocket_connect('/ws') as ws:
        hello = ws.receive_json()
    assert hello['interaction_mode'] == 'vad'
    assert hello['vad'] == VAD_CONFIG
    assert hello['vad']['max_utterance_ms'] == 12000
    assert hello['stt_stream']['chunk_samples'] == 2400
    assert provider.start_calls == 0


def test_ptt_boots_without_optional_vad_assets(tmp_path, monkeypatch):
    import api.app as app_module

    (tmp_path / 'index.html').write_text('push-to-talk')
    monkeypatch.setattr(app_module, 'WEB_DIR', tmp_path)
    with TestClient(create_app()) as client:
        assert client.get('/').text == 'push-to-talk'
        assert client.get('/vad-assets/missing.onnx').status_code == 404


def test_optional_assets_stay_inside_vad_folder(tmp_path, monkeypatch):
    import api.app as app_module

    assets = tmp_path / 'vendor' / 'vad'
    assets.mkdir(parents=True)
    (assets / 'silero_vad_v5.onnx').write_bytes(b'fake-model')
    (tmp_path / 'private.txt').write_text('private')
    (tmp_path / 'vad-capture.js').write_text('export const fake = true;')
    monkeypatch.setattr(app_module, 'WEB_DIR', tmp_path)
    with TestClient(create_app()) as client:
        assert client.get('/vad-assets/silero_vad_v5.onnx').content == b'fake-model'
        assert client.get('/vad-assets/%2e%2e/%2e%2e/private.txt').status_code == 404
        helper = client.get('/vad-capture.js')
        assert helper.status_code == 200
        assert helper.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('milestone', [None, [], {}, True, 1, 'unknown'])
def test_malformed_vad_observation_is_harmless(milestone):
    assert _parse_incoming(json.dumps({
        'type': 'vad_observed',
        'client_event_id': 'capture-1',
        'milestone': milestone,
    })) == ('noop', None)


@pytest.mark.parametrize('interaction_mode', ['push_to_talk', 'vad'])
def test_vad_observations_dedupe_and_ignore_cancelled_capture(tmp_path, interaction_mode):
    path = tmp_path / 'events.jsonl'
    provider = FakeStreamingSTTClient()
    app = create_app(
        streaming_stt_client=provider,
        interaction_mode=interaction_mode,
        event_recorder=JsonlEventRecorder(path),
        radio_sec=100,
        max_turns=1,
    )
    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        _observe(ws, 'old', 'start')
        _send_hold(ws, 'old')
        _observe(ws, 'old', 'end')
        _observe(ws, 'old', 'start')
        _observe(ws, 'old', 'start')
        ws.send_json({'type': 'hold_off', 'client_event_id': 'old'})
        _send_hold(ws, 'current')
        _observe(ws, 'old', 'end')
        _observe(ws, 'current', 'end')
        _observe(ws, 'current', 'start')
        _observe(ws, 'current', 'end')
        _observe(ws, 'current', 'end')
        ws.send_json({'type': 'say', 'text': '다음 입력', 'client_event_id': 'typed'})
        while ws.receive_json()['type'] != 'done':
            pass

    events = [json.loads(line) for line in path.read_text().splitlines()]
    observed = [(event['event'], event['client_event_id']) for event in events
                if event['event'].startswith('vad_')]
    assert observed == ([
        ('vad_speech_started', 'old'),
        ('vad_speech_started', 'current'),
        ('vad_speech_ended', 'current'),
    ] if interaction_mode == 'vad' else [])
    session = next(event for event in events if event['event'] == 'session_started')
    assert session['metadata']['interaction_mode'] == interaction_mode
    if interaction_mode == 'vad':
        assert session['metadata']['vad_engine'] == 'silero_v5'
        assert session['metadata']['vad_max_utterance_ms'] == 12000


class DisconnectOnceSTT:
    """실제 WebSocket 종료 예외를 내지만 외부 연결은 만들지 않는 공급자."""

    def __init__(self, phase):
        self.phase = phase
        self.start_calls = 0
        self.turns = []

    async def start(self, *, on_delta):
        self.start_calls += 1
        fail_this_turn = self.start_calls == 1
        if fail_this_turn and self.phase == 'start':
            raise ConnectionClosedError(None, None)
        phase = self.phase

        class Turn(FakeStreamingSTTTurn):
            async def append(self, audio):
                if fail_this_turn and phase == 'append':
                    raise ConnectionClosedError(None, None)
                await super().append(audio)

            async def finish(self):
                if fail_this_turn and phase == 'finish':
                    self.finish_calls += 1
                    raise ConnectionClosedError(None, None)
                return await super().finish()

        turn = Turn('연결 오류 뒤 새 발화', on_delta)
        self.turns.append(turn)
        return turn


def _send_utterance(ws, client_event_id, *, interaction_mode='vad'):
    _send_hold(ws, client_event_id)
    if interaction_mode == 'vad':
        _observe(ws, client_event_id, 'start')
    _send_stream_start(ws, client_event_id)
    for sequence, audio in enumerate(STREAMING_STT_CHUNKS, start=1):
        _send_stream_chunk(ws, client_event_id, sequence, audio)
    if interaction_mode == 'vad':
        _observe(ws, client_event_id, 'end')
    _send_stream_commit(ws, client_event_id, 3, 6000)


@pytest.mark.parametrize('phase', ['start', 'append', 'finish'])
@pytest.mark.parametrize('interaction_mode', ['push_to_talk', 'vad'])
def test_stream_disconnect_recovers_after_radio_resumes(tmp_path, phase, interaction_mode):
    """종료 오류 뒤 AI가 다시 말하더라도 같은 세션의 새 음성을 받을 수 있다."""
    class RecordingStream:
        def __init__(self):
            self.prompts = []

        async def complete_stream(self, system, user):
            self.prompts.append(user)
            yield '{"next_speaker": "ai_a"}\n계속 이야기해요.'

    path = tmp_path / 'recovery.jsonl'
    provider = DisconnectOnceSTT(phase)
    stream = RecordingStream()
    app = create_app(
        stream,
        tts_client=FakeTTS(),
        streaming_stt_client=provider,
        interaction_mode=interaction_mode,
        event_recorder=JsonlEventRecorder(path),
        radio_sec=0.2,
        unified=True,
    )
    messages = []
    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        _send_utterance(ws, 'failed-voice', interaction_mode=interaction_mode)
        while True:
            message = ws.receive_json()
            messages.append(message)
            assert message['type'] != 'done', '전사 오류 전에 세션이 종료됨'
            if message['type'] == 'stt_error':
                assert message == {
                    'type': 'stt_error',
                    'client_event_id': 'failed-voice',
                    'code': 'provider_unavailable',
                }
                break
        # A failed voice request releases the floor; it does not end the session.
        while True:
            message = ws.receive_json()
            messages.append(message)
            assert message['type'] != 'done'
            if message['type'] == 'start':
                break

        _send_hold(ws, 'retry-voice')
        _send_stream_start(ws, 'retry-voice')
        ws.send_json({'type': 'hold_off', 'client_event_id': 'failed-voice'})
        _send_stream_chunk(ws, 'failed-voice', 4, STREAMING_STT_CHUNKS[0])
        _send_stream_commit(ws, 'failed-voice', 3, 6000)
        for sequence, audio in enumerate(STREAMING_STT_CHUNKS, start=1):
            _send_stream_chunk(ws, 'retry-voice', sequence, audio)
        _send_stream_commit(ws, 'retry-voice', 3, 6000)

        retry_started = False
        while True:
            message = ws.receive_json()
            messages.append(message)
            assert message['type'] != 'done', '새 발화가 반영되기 전에 세션이 종료됨'
            if message['type'] == 'start' and message.get('after_client_event_id') == 'retry-voice':
                retry_started = True
            if retry_started and message['type'] == 'end':
                break

    assert provider.start_calls == 2
    assert provider.turns[-1].appended == list(STREAMING_STT_CHUNKS)
    assert provider.turns[-1].finish_calls == 1
    if phase != 'start':
        assert provider.turns[0].cancel_calls == 1
    assert [message for message in messages if message['type'] == 'you'] == [{
        'type': 'you', 'client_event_id': 'retry-voice', 'text': '연결 오류 뒤 새 발화',
    }]
    assert any('연결 오류 뒤 새 발화' in prompt for prompt in stream.prompts)
    assert all('임시 전사' not in prompt for prompt in stream.prompts)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(event['client_event_id'], event['outcome']) for event in events
            if event['event'] == 'stt_completed'] == [
        ('failed-voice', 'error'), ('retry-voice', 'success'),
    ]


def test_vad_disconnect_still_reaches_the_default_automatic_session_cap(tmp_path):
    """전사 실패 뒤 재시도가 없으면 기존 12회 AI 턴 제한으로 정상 종료된다."""
    path = tmp_path / 'session-cap.jsonl'
    provider = DisconnectOnceSTT('finish')
    app = create_app(
        FakeStream(),
        tts_client=FakeTTS(),
        streaming_stt_client=provider,
        interaction_mode='vad',
        event_recorder=JsonlEventRecorder(path),
        radio_sec=0.03,
    )
    messages = []
    with TestClient(app).websocket_connect('/ws') as ws:
        _start_session(ws)
        _send_utterance(ws, 'failed-before-radio')
        while True:
            message = ws.receive_json()
            messages.append(message)
            if message['type'] == 'audio':
                ws.send_json({'type': 'played', 'seq': message['seq']})
            if message['type'] == 'done':
                break
    assert [message['code'] for message in messages if message['type'] == 'stt_error'] == [
        'provider_unavailable',
    ]
    assert sum(message['type'] == 'start' for message in messages) == MAX_TURNS == 12
    assert all(message['type'] != 'you' for message in messages)
    assert provider.start_calls == 1
    assert provider.turns[0].cancel_calls == 1
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(event['event'] == 'hold_cancelled' for event in events) == 1
