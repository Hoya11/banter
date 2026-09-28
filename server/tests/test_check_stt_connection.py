"""연결 점검 명령을 가짜 공급자로 검증하며 실제 API와 키 파일은 사용하지 않는다."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from engine.eval.providers import RealtimeTranscriptionError
from scripts import check_stt_connection as probe


class FakeTurn:
    def __init__(self):
        self.cancel_calls = 0

    async def cancel(self):
        self.cancel_calls += 1


class FakeSDK:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


@pytest.fixture
def fakes(monkeypatch):
    sdk = FakeSDK()
    turn = FakeTurn()
    calls = []

    class Provider:
        async def start(self):
            calls.append('start')
            return turn

    provider = Provider()

    def make_sdk(**kwargs):
        assert kwargs == {
            'api_key': 'test-key', 'timeout': probe.CONNECT_SECONDS, 'max_retries': 0,
        }
        return sdk

    def make_provider(**kwargs):
        assert kwargs == {
            'async_client': sdk, 'model': 'gpt-live-transcribe',
            'timeout': probe.CONNECT_SECONDS, 'close_timeout': 1.0,
        }
        return provider

    monkeypatch.setattr(probe, 'AsyncOpenAI', make_sdk)
    monkeypatch.setattr(probe, 'OpenAIRealtimeSTTClient', make_provider)
    monkeypatch.setattr(probe, 'CONNECT_SECONDS', 0.02)
    monkeypatch.setattr(probe, 'CLEANUP_SECONDS', 0.1)
    monkeypatch.setattr(probe, 'SDK_CLOSE_SECONDS', 0.02)
    return SimpleNamespace(sdk=sdk, turn=turn, provider=provider, calls=calls)


def test_default_only_explains_scope_without_loading_keys_or_constructing_clients(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('안내 모드는 키 파일과 공급자를 사용하면 안 된다')

    monkeypatch.setattr(probe, 'load_dotenv', forbidden)
    monkeypatch.setattr(probe, 'AsyncOpenAI', forbidden)
    monkeypatch.setattr(probe, 'check_connection', forbidden)
    assert probe.main([]) == 0
    output = capsys.readouterr().out
    assert '--live' in output
    assert '음성 전송·commit·LLM·TTS 호출 없음' in output
    assert '안내만 출력' in output


def test_live_missing_key_fails_without_constructing_a_client(monkeypatch, capsys):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(probe, 'load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr(probe, 'AsyncOpenAI', lambda **kwargs: pytest.fail('키 없는 연결'))
    assert probe.main(['--live']) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        'status': 'failed', 'code': 'missing_api_key',
    }


def test_success_starts_once_then_cancels_and_closes_sdk(fakes):
    result = asyncio.run(probe.check_connection('test-key'))
    assert result == {'status': 'configured', 'diagnostics': []}
    assert fakes.calls == ['start']
    assert fakes.turn.cancel_calls == 1
    assert fakes.sdk.close_calls == 1


def test_live_preserves_safe_provider_diagnostics_without_raw_error_or_key(
    fakes, monkeypatch, capsys,
):
    async def fail():
        raise RealtimeTranscriptionError(
            'secret-provider-body test-key',
            diagnostics={
                'stage': 'configure_ack', 'provider_error_code': 'invalid_value',
                'message': 'secret-provider-body', 'api_key': 'test-key',
            },
        )

    monkeypatch.setattr(fakes.provider, 'start', fail)
    monkeypatch.setattr(probe, 'load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    assert probe.main(['--live']) == 1
    captured = capsys.readouterr()
    assert 'test-key' not in captured.out + captured.err
    assert 'secret-provider-body' not in captured.out + captured.err
    result = json.loads(captured.out.splitlines()[-1])
    assert result['diagnostics'][0]['stage'] == 'configure_ack'
    assert result['diagnostics'][0]['provider_error_code'] == 'invalid_value'
    assert fakes.sdk.close_calls == 1


@pytest.mark.parametrize('return_late_turn', [False, True])
def test_start_timeout_cancels_request_and_closes_even_a_late_turn(
    fakes, monkeypatch, return_late_turn,
):
    cancelled = []

    async def wait_forever():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.append(True)
            if return_late_turn:
                return fakes.turn
            raise

    monkeypatch.setattr(fakes.provider, 'start', wait_forever)
    result = asyncio.run(probe.check_connection('test-key'))
    assert result['status'] == 'failed'
    assert result['diagnostics'][0] == {'phase': 'connect', 'error_type': 'TimeoutError'}
    assert cancelled == [True]
    assert fakes.turn.cancel_calls == int(return_late_turn)
    assert fakes.sdk.close_calls == 1


@pytest.mark.parametrize('failed_resource', ['turn', 'sdk'])
def test_cleanup_failure_is_nonzero_and_sdk_close_is_always_attempted(
    fakes, monkeypatch, failed_resource, capsys,
):
    async def fail():
        raise RuntimeError('secret-cleanup-response')

    resource = fakes.turn if failed_resource == 'turn' else fakes.sdk
    monkeypatch.setattr(resource, 'cancel' if failed_resource == 'turn' else 'close', fail)
    monkeypatch.setattr(probe, 'load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    assert probe.main(['--live']) == 1
    captured = capsys.readouterr()
    assert 'secret-cleanup-response' not in captured.out + captured.err
    if failed_resource == 'turn':
        assert fakes.sdk.close_calls == 1


def test_sdk_close_timeout_fails_and_cancels_close(fakes, monkeypatch):
    cancelled = []

    async def wait_forever():
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(fakes.sdk, 'close', wait_forever)
    result = asyncio.run(probe.check_connection('test-key'))
    assert result['status'] == 'failed'
    assert result['diagnostics'] == [{'phase': 'sdk_cleanup', 'error_type': 'TimeoutError'}]
    assert cancelled == [True]


def test_external_cancellation_still_closes_start_and_sdk(fakes, monkeypatch):
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_forever():
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        monkeypatch.setattr(fakes.provider, 'start', wait_forever)
        task = asyncio.create_task(probe.check_connection('test-key'))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert fakes.sdk.close_calls == 1

    asyncio.run(scenario())


def test_provider_construction_failure_still_closes_sdk(fakes, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError('secret-configuration')

    monkeypatch.setattr(probe, 'OpenAIRealtimeSTTClient', fail)
    result = asyncio.run(probe.check_connection('test-key'))
    assert result['status'] == 'failed'
    assert result['diagnostics'] == [{'phase': 'connect', 'error_type': 'RuntimeError'}]
    assert fakes.sdk.close_calls == 1


def test_live_loads_only_server_dotenv_and_preserves_runtime_key(monkeypatch, tmp_path, capsys):
    (tmp_path / '.env').write_text('OPENAI_API_KEY=file-test-key\n', encoding='utf-8')
    monkeypatch.setattr(probe, 'SERVER_DIR', tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'runtime-test-key')
    monkeypatch.delenv('PYTHON_DOTENV_DISABLED', raising=False)
    received = []

    async def fake_check(api_key):
        received.append(api_key)
        return {'status': 'configured', 'diagnostics': []}

    monkeypatch.setattr(probe, 'check_connection', fake_check)
    assert probe.main(['--live']) == 0
    assert received == ['runtime-test-key']
    output = capsys.readouterr().out
    assert 'runtime-test-key' not in output
    assert 'file-test-key' not in output


def test_real_adapter_sends_only_one_session_update_then_closes(monkeypatch):
    sent = []
    connected = []
    closed = []

    class Connection:
        def __init__(self):
            self.events = iter([
                {'type': 'session.created', 'session': {'type': 'transcription'}},
                {'type': 'session.updated'},
            ])

        async def recv(self):
            event = next(self.events, None)
            if event is None:
                await asyncio.Future()
            return event

        async def send(self, event):
            sent.append(event)

    class Manager:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, *args):
            closed.append('session')

    def connect(**kwargs):
        connected.append(kwargs)
        return Manager()

    sdk = FakeSDK()
    sdk.realtime = SimpleNamespace(connect=connect)
    monkeypatch.setattr(probe, 'AsyncOpenAI', lambda **kwargs: sdk)
    result = asyncio.run(probe.check_connection('test-key'))
    assert result['status'] == 'configured'
    assert len(connected) == 1
    assert connected[0]['extra_query'] == {'intent': 'transcription'}
    assert [event['type'] for event in sent] == ['session.update']
    configuration = sent[0]['session']
    assert configuration['type'] == 'transcription'
    assert configuration['audio']['input']['format'] == {'type': 'audio/pcm', 'rate': 24000}
    assert configuration['audio']['input']['transcription']['model'] == 'gpt-live-transcribe'
    assert configuration['audio']['input']['turn_detection'] is None
    assert closed == ['session']
    assert sdk.close_calls == 1
