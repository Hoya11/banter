import asyncio
import base64
import gc

import pytest

from engine.eval.providers import (
    OpenAIRealtimeSTTClient,
    RealtimeTranscriptionError,
)


class FakeRealtimeConnection:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.recv_cancelled = 0
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.recv_entered = asyncio.Event()
        self._send_event = asyncio.Event()

    def feed(self, event: dict) -> None:
        self._incoming.put_nowait(event)

    async def send(self, event: dict) -> None:
        self.sent.append(event)
        self._send_event.set()

    async def recv(self) -> dict:
        try:
            self.recv_entered.set()
            event = await self._incoming.get()
            if isinstance(event, BaseException):
                raise event
            return event
        except asyncio.CancelledError:
            self.recv_cancelled += 1
            raise

    async def wait_for_send_count(self, count: int) -> None:
        while len(self.sent) < count:
            self._send_event.clear()
            if len(self.sent) >= count:
                return
            await asyncio.wait_for(self._send_event.wait(), timeout=0.5)


class FakeRealtimeConnectionManager:
    def __init__(self, connection: FakeRealtimeConnection) -> None:
        self.connection = connection
        self.entered = 0
        self.exited = 0
        self.exit_args = None

    async def __aenter__(self) -> FakeRealtimeConnection:
        self.entered += 1
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited += 1
        self.exit_args = (exc_type, exc, traceback)


class FakeRealtimeAPI:
    def __init__(self, manager: FakeRealtimeConnectionManager) -> None:
        self.manager = manager
        self.connect_calls: list[dict] = []

    def connect(self, **kwargs) -> FakeRealtimeConnectionManager:
        self.connect_calls.append(kwargs)
        return self.manager


class FakeAsyncOpenAI:
    def __init__(self, manager: FakeRealtimeConnectionManager) -> None:
        self.realtime = FakeRealtimeAPI(manager)


def make_client():
    connection = FakeRealtimeConnection()
    manager = FakeRealtimeConnectionManager(connection)
    async_client = FakeAsyncOpenAI(manager)
    client = OpenAIRealtimeSTTClient(
        async_client=async_client,
        timeout=0.5,
    )
    return client, async_client, manager, connection


def test_realtime_stt_configures_appends_commits_and_returns_transcript():
    async def scenario() -> None:
        client, async_client, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'session.updated'})
        deltas: list[str] = []

        turn = await client.start(on_delta=deltas.append)

        assert async_client.realtime.connect_calls == [
            {
                'extra_query': {'intent': 'transcription'},
                'websocket_connection_options': {'close_timeout': 1.0},
            }
        ]
        assert connection.sent == [
            {
                'type': 'session.update',
                'session': {
                    'type': 'transcription',
                    'audio': {
                        'input': {
                            'format': {
                                'type': 'audio/pcm',
                                'rate': 24_000,
                            },
                            'transcription': {
                                'model': 'gpt-live-transcribe',
                                'languages': ['ko'],
                                'delay': 'low',
                            },
                            'turn_detection': None,
                        }
                    },
                },
            }
        ]

        pcm = b'\x00\x01\xfe\xff'
        await turn.append(pcm)
        assert connection.sent[1] == {
            'type': 'input_audio_buffer.append',
            'audio': base64.b64encode(pcm).decode('ascii'),
        }

        finish_task = asyncio.create_task(turn.finish())
        await connection.wait_for_send_count(3)
        assert connection.sent[2] == {'type': 'input_audio_buffer.commit'}

        connection.feed(
            {
                'type': 'conversation.item.input_audio_transcription.delta',
                'delta': 'hello ',
            }
        )
        connection.feed(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': ' hello there ',
            }
        )

        assert await finish_task == 'hello there'
        await turn.close()
        assert deltas == ['hello ']
        assert manager.entered == 1
        assert manager.exited == 1
        assert manager.exit_args == (None, None, None)

    asyncio.run(scenario())


def test_realtime_stt_provider_failure_raises_and_closes_connection():
    async def scenario() -> None:
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'session.updated'})
        turn = await client.start()

        finish_task = asyncio.create_task(turn.finish())
        await connection.wait_for_send_count(2)
        connection.feed(
            {'type': 'conversation.item.input_audio_transcription.failed'}
        )

        with pytest.raises(RealtimeTranscriptionError, match='전사에 실패'):
            await finish_task
        await turn.close()
        assert manager.entered == 1
        assert manager.exited == 1

    asyncio.run(scenario())


def test_realtime_stt_cancel_closes_once_and_rejects_more_audio():
    async def scenario() -> None:
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'session.updated'})
        turn = await client.start()
        await asyncio.sleep(0)

        await turn.cancel()
        await turn.close()

        assert connection.sent[0]['type'] == 'session.update'
        assert len(connection.sent) == 1
        assert connection.recv_cancelled == 1
        assert manager.entered == 1
        assert manager.exited == 1
        with pytest.raises(RealtimeTranscriptionError, match='종료된'):
            await turn.append(b'more audio')

    asyncio.run(scenario())


def test_realtime_stt_close_has_a_bounded_provider_exit():
    async def scenario() -> None:
        connection = FakeRealtimeConnection()

        class HangingExitManager(FakeRealtimeConnectionManager):
            async def __aexit__(self, exc_type, exc, traceback) -> None:
                self.exited += 1
                self.exit_args = (exc_type, exc, traceback)
                await asyncio.Event().wait()

        manager = HangingExitManager(connection)
        client = OpenAIRealtimeSTTClient(
            async_client=FakeAsyncOpenAI(manager),
            timeout=0.5,
            close_timeout=0.01,
        )
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'session.updated'})
        turn = await client.start()

        await asyncio.wait_for(turn.cancel(), timeout=0.2)
        await asyncio.wait_for(turn.close(), timeout=0.2)

        assert manager.entered == 1
        assert manager.exited == 1
        assert manager.exit_args == (None, None, None)

    asyncio.run(scenario())


def test_realtime_stt_finish_returns_before_provider_exit_completes():
    async def scenario() -> None:
        connection = FakeRealtimeConnection()
        exit_started = asyncio.Event()
        release_exit = asyncio.Event()

        class GatedExitManager(FakeRealtimeConnectionManager):
            async def __aexit__(self, exc_type, exc, traceback) -> None:
                self.exited += 1
                self.exit_args = (exc_type, exc, traceback)
                exit_started.set()
                await release_exit.wait()

        manager = GatedExitManager(connection)
        client = OpenAIRealtimeSTTClient(
            async_client=FakeAsyncOpenAI(manager),
            timeout=0.5,
            close_timeout=0.5,
        )
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'session.updated'})
        turn = await client.start()

        finish_task = asyncio.create_task(turn.finish())
        await connection.wait_for_send_count(2)
        connection.feed(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': '지연 없는 최종 전사',
            }
        )

        assert await asyncio.wait_for(finish_task, timeout=0.1) == '지연 없는 최종 전사'
        await asyncio.wait_for(exit_started.wait(), timeout=0.1)
        release_exit.set()
        await turn.close()
        assert manager.exited == 1

    asyncio.run(scenario())


def test_realtime_stt_cancel_retrieves_failure_before_commit():
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        reports = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reports.append(context))
        try:
            client, _, manager, connection = make_client()
            connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
            connection.feed({'type': 'session.updated'})
            turn = await client.start()
            connection.feed(
                {'type': 'conversation.item.input_audio_transcription.failed'}
            )
            await asyncio.wait_for(turn._reader_task, timeout=0.1)

            await turn.cancel()
            del turn
            gc.collect()
            await asyncio.sleep(0)

            assert manager.exited == 1
            assert not any(
                context.get('message') == 'Future exception was never retrieved'
                for context in reports
            )
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_realtime_stt_diagnostics_excludes_free_text_and_invalid_metadata():
    from types import SimpleNamespace

    import httpx
    from openai import APIStatusError
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    from engine.eval.providers import realtime_stt_diagnostics

    secret = 'private transcript sk-secret audio-base64'
    closed = ConnectionClosedError(Close(1011, secret), Close(1000, secret), True)
    assert realtime_stt_diagnostics(closed) == {
        'error_type': 'ConnectionClosedError',
        'close_state': 'received_and_sent',
        'close_received_code': 1011,
        'close_sent_code': 1000,
    }
    assert realtime_stt_diagnostics(ConnectionClosedError(None, None)) == {
        'error_type': 'ConnectionClosedError', 'close_state': 'no_close_frame',
    }
    response = httpx.Response(429, request=httpx.Request('GET', 'https://example.invalid'))
    status = APIStatusError(secret, response=response, body={
        'type': 'invalid_request_error', 'code': 'insufficient_quota', 'param': 'audio.input',
        'message': secret,
    })
    assert realtime_stt_diagnostics(status) == {
        'error_type': 'APIStatusError', 'http_status': 429,
        'provider_error_type': 'invalid_request_error',
        'provider_error_code': 'insufficient_quota', 'provider_error_param': 'audio.input',
    }
    unsafe = RealtimeTranscriptionError('safe', diagnostics={
        'message': secret, 'audio': secret, 'transcript': secret,
        'provider_error_code': 'sk-secret', 'provider_error_param': 'a' * 97,
        'provider_error_type': 'bad\nvalue', 'close_received_code': True,
        'http_status': 1000000, 'stage': SimpleNamespace(value='commit'),
    })
    assert realtime_stt_diagnostics(unsafe) == {'error_type': 'RealtimeTranscriptionError'}


@pytest.mark.parametrize('event_type', [
    'error', 'conversation.item.input_audio_transcription.failed',
])
def test_provider_error_details_survive_a_later_commit_socket_failure(event_type):
    from types import SimpleNamespace

    from websockets.exceptions import ConnectionClosedError

    from engine.eval.providers import realtime_stt_diagnostics

    async def scenario():
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'session.updated'})
        turn = await client.start()
        connection.feed(SimpleNamespace(type=event_type, error=SimpleNamespace(
            type='invalid_request_error', code='invalid_value',
            param='session.audio.input.transcription.languages',
            message='sensitive transcript and audio',
        )))
        await asyncio.wait_for(turn._reader_task, timeout=0.1)

        async def fail_send(event):
            raise ConnectionClosedError(None, None)

        connection.send = fail_send
        with pytest.raises(RealtimeTranscriptionError) as caught:
            await turn.finish()
        details = realtime_stt_diagnostics(caught.value)
        assert details['provider_event_type'] == event_type
        assert details['provider_error_code'] == 'invalid_value'
        assert details['provider_error_param'] == 'session.audio.input.transcription.languages'
        assert details['stage'] == 'receive'
        assert 'sensitive' not in repr(details) + str(caught.value)
        assert len(connection.sent) == 1  # No commit/retry after the received failure.
        await turn.close()
        assert manager.exited == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('at', [
    'connect', 'session_create', 'configure_send', 'configure_ack', 'append', 'commit', 'receive',
])
def test_real_connection_closed_error_is_diagnosable_and_a_fresh_turn_recovers(at):
    import traceback

    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    from engine.eval.providers import realtime_stt_diagnostics

    async def scenario():
        client, async_client, manager, connection = make_client()
        secret = 'private transcript and sk-secret-key'

        def closed():
            return ConnectionClosedError(Close(1011, secret), Close(1011, secret), True)

        async def fail_send(event):
            raise closed()

        async def fail_recv():
            raise closed()

        if at == 'connect':
            class FailedEnter(FakeRealtimeConnectionManager):
                async def __aenter__(self):
                    self.entered += 1
                    raise closed()

            manager = FailedEnter(connection)
            async_client.realtime.manager = manager
        elif at == 'session_create':
            connection.recv = fail_recv
        elif at in {'configure_send', 'configure_ack'}:
            connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
            connection.feed(closed())
            if at == 'configure_send':
                connection.send = fail_send
        else:
            connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
            connection.feed({'type': 'session.updated'})

        with pytest.raises(RealtimeTranscriptionError) as caught:
            turn = await client.start()
            if at in {'append', 'commit'}:
                connection.send = fail_send
            elif at == 'receive':
                turn._reader_task.cancel()
                try:
                    await turn._reader_task
                except asyncio.CancelledError:
                    pass
                connection.recv = fail_recv
                turn._reader_task = asyncio.create_task(turn._read_events())
                await asyncio.wait_for(turn._reader_task, timeout=0.1)
            if at == 'append':
                await turn.append(b'\x00\x01')
            else:
                await turn.finish()
        details = realtime_stt_diagnostics(caught.value)
        assert details == {
            'error_type': 'ConnectionClosedError', 'stage': at,
            'close_state': 'received_and_sent',
            'close_received_code': 1011, 'close_sent_code': 1011,
        }
        assert secret not in ''.join(traceback.format_exception(caught.value))
        if at not in {'connect', 'session_create', 'configure_send', 'configure_ack'}:
            await turn.close()
        assert manager.exited == (0 if at == 'connect' else 1)
        assert len(async_client.realtime.connect_calls) == 1

        next_connection = FakeRealtimeConnection()
        next_manager = FakeRealtimeConnectionManager(next_connection)
        async_client.realtime.manager = next_manager
        next_connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        next_connection.feed({'type': 'session.updated'})
        next_turn = await client.start()
        finished = asyncio.create_task(next_turn.finish())
        await next_connection.wait_for_send_count(2)
        next_connection.feed({
            'type': 'conversation.item.input_audio_transcription.completed',
            'transcript': '다음 입력 성공',
        })
        assert await finished == '다음 입력 성공'
        await next_turn.close()
        assert next_manager.entered == next_manager.exited == 1
        assert len(async_client.realtime.connect_calls) == 2

    asyncio.run(scenario())


def test_configuration_error_preserves_safe_provider_fields_and_closes():
    from engine.eval.providers import realtime_stt_diagnostics

    async def scenario():
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        connection.feed({'type': 'error', 'error': {
            'type': 'invalid_request_error', 'code': 'unknown_parameter',
            'param': 'session.audio.input.transcription.delay',
            'message': 'unsafe provider message',
        }})
        with pytest.raises(RealtimeTranscriptionError) as caught:
            await client.start()
        assert realtime_stt_diagnostics(caught.value) == {
            'error_type': 'RealtimeTranscriptionError', 'stage': 'configure_ack',
            'provider_event_type': 'error', 'provider_error_type': 'invalid_request_error',
            'provider_error_code': 'unknown_parameter',
            'provider_error_param': 'session.audio.input.transcription.delay',
        }
        assert manager.exited == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('at', [
    'connect', 'session_create', 'configure_send', 'configure_ack', 'append', 'commit', 'receive',
])
def test_realtime_stt_timeout_contract_is_preserved(at):
    async def scenario():
        client, async_client, manager, connection = make_client()

        async def timeout_send(event):
            raise TimeoutError('fake timeout')

        async def timeout_recv():
            raise TimeoutError('fake timeout')

        if at == 'connect':
            class TimeoutEnter(FakeRealtimeConnectionManager):
                async def __aenter__(self):
                    self.entered += 1
                    raise TimeoutError('fake timeout')

            async_client.realtime.manager = TimeoutEnter(connection)
        elif at == 'session_create':
            connection.recv = timeout_recv
        elif at in {'configure_send', 'configure_ack'}:
            connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
            connection.feed(TimeoutError('fake timeout'))
            if at == 'configure_send':
                connection.send = timeout_send
        else:
            connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
            connection.feed({'type': 'session.updated'})

        with pytest.raises(TimeoutError):
            turn = await client.start()
            if at in {'append', 'commit'}:
                connection.send = timeout_send
            elif at == 'receive':
                turn._reader_task.cancel()
                try:
                    await turn._reader_task
                except asyncio.CancelledError:
                    pass
                connection.recv = timeout_recv
                turn._reader_task = asyncio.create_task(turn._read_events())
                await asyncio.wait_for(turn._reader_task, timeout=0.1)
            if at == 'append':
                await turn.append(b'\x00\x01')
            else:
                await turn.finish()
        if at not in {'connect', 'session_create', 'configure_send', 'configure_ack'}:
            await turn.close()
        assert len(async_client.realtime.connect_calls) == 1

    asyncio.run(scenario())


def test_session_creation_error_is_read_before_a_queued_4000_close_or_any_send():
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    from engine.eval.providers import realtime_stt_diagnostics

    async def scenario():
        client, _, manager, connection = make_client()
        connection.feed({'type': 'error', 'error': {
            'type': 'invalid_request_error', 'code': 'model_not_found', 'param': 'model',
            'message': 'private provider message',
        }})
        connection.feed(ConnectionClosedError(Close(4000, ''), Close(4000, ''), True))

        async def already_closed_send(event):
            raise ConnectionClosedError(Close(4000, ''), Close(4000, ''), True)

        connection.send = already_closed_send
        with pytest.raises(RealtimeTranscriptionError) as caught:
            await client.start()
        assert realtime_stt_diagnostics(caught.value) == {
            'error_type': 'RealtimeTranscriptionError', 'stage': 'session_create',
            'provider_event_type': 'error', 'provider_error_type': 'invalid_request_error',
            'provider_error_code': 'model_not_found', 'provider_error_param': 'model',
        }
        assert connection.sent == []
        assert manager.exited == 1

    asyncio.run(scenario())


def test_session_update_waits_for_transcription_session_created():
    async def scenario():
        client, _, manager, connection = make_client()
        starting = asyncio.create_task(client.start())
        await asyncio.wait_for(connection.recv_entered.wait(), timeout=0.1)
        assert connection.sent == []
        assert not starting.done()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        await connection.wait_for_send_count(1)
        assert connection.sent[0]['type'] == 'session.update'
        assert not starting.done()
        connection.feed({'type': 'session.updated'})
        turn = await asyncio.wait_for(starting, timeout=0.1)
        await turn.close()
        assert manager.exited == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('session', [None, {}, {'type': 'realtime'}, {'type': 'translation'}])
def test_nontranscription_created_session_is_rejected_before_update(session):
    from engine.eval.providers import realtime_stt_diagnostics

    async def scenario():
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': session})
        connection.feed({'type': 'session.updated'})
        with pytest.raises(RealtimeTranscriptionError) as caught:
            await client.start()
        assert realtime_stt_diagnostics(caught.value)['stage'] == 'session_create'
        assert connection.sent == []
        assert manager.exited == 1

    asyncio.run(scenario())


def test_configuration_timeout_bounds_initial_session_and_update_together():
    async def scenario():
        client, _, manager, connection = make_client()
        client.timeout = 0.05
        starting = asyncio.create_task(client.start())
        await asyncio.wait_for(connection.recv_entered.wait(), timeout=0.1)
        await asyncio.sleep(0.03)
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})

        async def delayed_updated():
            await asyncio.sleep(0.03)
            connection.feed({'type': 'session.updated'})

        updating = asyncio.create_task(delayed_updated())
        try:
            with pytest.raises(TimeoutError):
                await starting
        finally:
            await updating
        assert len(connection.sent) <= 1
        assert manager.exited == 1

    asyncio.run(scenario())


def test_real_openai_sdk_serializes_transcription_intent_and_session_model(monkeypatch):
    import json
    from urllib.parse import parse_qs, urlsplit

    import httpx
    from openai import AsyncOpenAI
    import websockets.asyncio.client

    class MemoryWebSocket:
        def __init__(self):
            self.sent = []
            self.events = []
            self.closed = 0
            self.incoming = asyncio.Queue()
            self.feed({'type': 'session.created', 'session': {'type': 'transcription'}})

        def feed(self, event):
            self.incoming.put_nowait(json.dumps(event).encode())

        async def recv(self, *, decode=False):
            assert decode is False
            data = await self.incoming.get()
            self.events.append('recv:' + json.loads(data)['type'])
            return data

        async def send(self, data):
            assert isinstance(data, str)
            payload = json.loads(data)
            self.sent.append(payload)
            self.events.append('send:' + payload['type'])
            if payload['type'] == 'session.update':
                self.feed({'type': 'session.updated', 'session': {'type': 'transcription'}})
            elif payload['type'] == 'input_audio_buffer.commit':
                self.feed({'type': 'conversation.item.input_audio_transcription.delta',
                           'delta': '실제 SDK '})
                self.feed({'type': 'conversation.item.input_audio_transcription.completed',
                           'transcript': '실제 SDK 전송 확인'})

        async def close(self, *, code=1000, reason=''):
            self.closed += 1

    async def scenario():
        socket = MemoryWebSocket()
        urls = []

        async def memory_connect(url, **kwargs):
            urls.append(url)
            return socket

        def forbid_http(request):
            raise AssertionError('No real or fake HTTP call is expected')

        monkeypatch.setattr(websockets.asyncio.client, 'connect', memory_connect)
        sdk = AsyncOpenAI(
            api_key='not-a-real-key', base_url='https://api.openai.com/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(forbid_http)),
        )
        deltas = []
        turn = None
        try:
            client = OpenAIRealtimeSTTClient(async_client=sdk, timeout=0.5)
            turn = await client.start(on_delta=deltas.append)
            assert len(urls) == 1
            url = urlsplit(urls[0])
            assert (url.scheme, url.netloc, url.path) == ('wss', 'api.openai.com', '/v1/realtime')
            assert parse_qs(url.query) == {'intent': ['transcription']}
            assert socket.events[:3] == [
                'recv:session.created', 'send:session.update', 'recv:session.updated',
            ]
            update = socket.sent[0]
            assert update == {
                'type': 'session.update',
                'session': {
                    'type': 'transcription',
                    'audio': {'input': {
                        'format': {'type': 'audio/pcm', 'rate': 24000},
                        'transcription': {
                            'model': 'gpt-live-transcribe', 'languages': ['ko'], 'delay': 'low',
                        },
                        'turn_detection': None,
                    }},
                },
            }
            audio = b'\x00\x00\x01\x00'
            await turn.append(audio)
            assert socket.sent[1]['type'] == 'input_audio_buffer.append'
            assert base64.b64decode(socket.sent[1]['audio']) == audio
            assert await turn.finish() == '실제 SDK 전송 확인'
            assert deltas == ['실제 SDK ']
            assert socket.sent[2] == {'type': 'input_audio_buffer.commit'}
            await turn.close()
            assert socket.closed == 1
        finally:
            if turn is not None:
                await turn.close()
            await sdk.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('waiting_for', ['session_created', 'update_send', 'error_drain'])
def test_startup_deadline_cancels_a_genuinely_stalled_provider(waiting_for):
    async def scenario():
        client, _, manager, connection = make_client()
        client.timeout = 0.02
        send_cancelled = 0
        if waiting_for in {'update_send', 'error_drain'}:
            connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})

            async def stalled_send(event):
                nonlocal send_cancelled
                assert event['type'] == 'session.update'
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    send_cancelled += 1
                    raise

            connection.send = stalled_send
            if waiting_for == 'error_drain':
                from websockets.exceptions import ConnectionClosedError
                from websockets.frames import Close

                async def closed_send(event):
                    raise ConnectionClosedError(Close(4000, ''), Close(4000, ''), True)

                connection.send = closed_send
        starting = asyncio.create_task(client.start())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(starting), timeout=1.0)
        # The inner deadline must finish its cleanup before the outer watchdog.
        assert starting.done()
        assert manager.entered == manager.exited == 1
        assert send_cancelled == (1 if waiting_for == 'update_send' else 0)
        assert connection.recv_cancelled == (0 if waiting_for == 'update_send' else 1)

    asyncio.run(scenario())


def test_cancelling_start_while_waiting_for_created_closes_once():
    async def scenario():
        client, _, manager, connection = make_client()
        starting = asyncio.create_task(client.start())
        await asyncio.wait_for(connection.recv_entered.wait(), timeout=1.0)
        assert connection.sent == []
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, timeout=1.0)
        assert manager.entered == manager.exited == 1
        assert connection.recv_cancelled == 1
        assert connection.sent == []

    asyncio.run(scenario())


@pytest.mark.parametrize('intervening_event', [False, True])
def test_real_sdk_update_send_close_preserves_queued_provider_error(monkeypatch, intervening_event):
    import json
    import traceback

    import httpx
    from openai import AsyncOpenAI
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close
    import websockets.asyncio.client

    from engine.eval.providers import realtime_stt_diagnostics

    secret = 'private transcript and sk-secret-key'

    class MemoryWebSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.sent = []
            self.closed = 0
            self.incoming.put_nowait(json.dumps({
                'type': 'session.created', 'session': {'type': 'transcription'},
            }).encode())

        async def recv(self, *, decode=False):
            assert decode is False
            return await self.incoming.get()

        async def send(self, data):
            self.sent.append(json.loads(data)['type'])
            if intervening_event:
                self.incoming.put_nowait(json.dumps({
                    'type': 'rate_limits.updated', 'rate_limits': [],
                }).encode())
            self.incoming.put_nowait(json.dumps({
                'type': 'error', 'error': {
                    'type': 'invalid_request_error', 'code': 'unknown_parameter',
                    'param': 'session.audio.input.transcription.delay', 'message': secret,
                },
            }).encode())
            raise ConnectionClosedError(Close(4000, secret), Close(4000, secret), True)

        async def close(self, *, code=1000, reason=''):
            self.closed += 1

    async def scenario():
        socket = MemoryWebSocket()

        async def memory_connect(url, **kwargs):
            assert url == 'wss://api.openai.com/v1/realtime?intent=transcription'
            return socket

        def forbid_http(request):
            raise AssertionError('No HTTP call is expected')

        monkeypatch.setattr(websockets.asyncio.client, 'connect', memory_connect)
        sdk = AsyncOpenAI(
            api_key='not-a-real-key', base_url='https://api.openai.com/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(forbid_http)),
        )
        try:
            with pytest.raises(RealtimeTranscriptionError) as caught:
                await OpenAIRealtimeSTTClient(async_client=sdk, timeout=0.5).start()
            details = realtime_stt_diagnostics(caught.value)
            assert details['stage'] == 'configure_send'
            assert details['provider_event_type'] == 'error'
            assert details['provider_error_type'] == 'invalid_request_error'
            assert details['provider_error_code'] == 'unknown_parameter'
            assert details['provider_error_param'] == 'session.audio.input.transcription.delay'
            assert details['close_state'] == 'received_and_sent'
            assert details['close_received_code'] == details['close_sent_code'] == 4000
            assert secret not in str(details)
            assert secret not in ''.join(traceback.format_exception(caught.value))
            assert socket.incoming.empty()
            assert socket.sent == ['session.update']
            assert socket.closed == 1
        finally:
            await sdk.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('incoming_state', ['closed', 'silent'])
def test_update_send_close_without_provider_error_keeps_diagnostics_and_stops_waiting(incoming_state):
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    from engine.eval.providers import realtime_stt_diagnostics

    async def scenario():
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        closed = ConnectionClosedError(Close(4000, ''), Close(4000, ''), True)
        if incoming_state == 'closed':
            connection.feed(closed)

        async def closed_send(event):
            raise closed

        connection.send = closed_send
        starting = asyncio.create_task(client.start())
        with pytest.raises(RealtimeTranscriptionError) as caught:
            # A missing error response must not consume the 0.5 second startup deadline.
            await asyncio.wait_for(asyncio.shield(starting), timeout=0.3)
        assert realtime_stt_diagnostics(caught.value) == {
            'error_type': 'ConnectionClosedError', 'stage': 'configure_send',
            'close_state': 'received_and_sent',
            'close_received_code': 4000, 'close_sent_code': 4000,
        }
        assert manager.entered == manager.exited == 1
        assert connection.recv_cancelled == (1 if incoming_state == 'silent' else 0)
        assert starting.done()

    asyncio.run(scenario())


def test_cancelling_start_during_configure_error_drain_closes_once():
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    async def scenario():
        client, _, manager, connection = make_client()
        connection.feed({'type': 'session.created', 'session': {'type': 'transcription'}})

        drain_entered = asyncio.Event()
        original_recv = connection.recv

        async def drain_recv():
            drain_entered.set()
            return await original_recv()

        async def closed_send(event):
            connection.recv = drain_recv
            raise ConnectionClosedError(Close(4000, ''), Close(4000, ''), True)

        connection.send = closed_send
        starting = asyncio.create_task(client.start())
        await asyncio.wait_for(drain_entered.wait(), timeout=0.5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, timeout=0.5)
        assert manager.entered == manager.exited == 1
        assert connection.recv_cancelled == 1
        assert starting.done()

    asyncio.run(scenario())


class MemoryRealtimeTransport(asyncio.Transport):
    """Exercise native WebSocket close without opening a socket or calling an API."""

    def __init__(self, websocket, *, block_close_write=False, configuration_error=False):
        from websockets.protocol import OPEN
        from websockets.server import ServerProtocol

        self.websocket = websocket
        self.peer = ServerProtocol(state=OPEN)
        self.block_close_write = block_close_write
        self.configuration_error = configuration_error
        self.close_received = asyncio.Event()
        self.closed = asyncio.Event()
        self.aborts = 0
        self.sent_types = []

    def set_write_buffer_limits(self, high=None, low=None):
        pass

    def pause_reading(self):
        pass

    def resume_reading(self):
        pass

    def is_closing(self):
        return self.closed.is_set()

    def feed(self, payload):
        import json

        self.peer.send_text(json.dumps(payload).encode())
        for data in self.peer.data_to_send():
            self.websocket.data_received(data)

    def write(self, data):
        import json

        from websockets.frames import OP_CLOSE, OP_TEXT

        self.peer.receive_data(data)
        for frame in self.peer.events_received():
            if frame.opcode == OP_CLOSE:
                self.close_received.set()
                # Drop the peer's close reply. Native close_timeout must release us.
                self.peer.data_to_send()
                if self.block_close_write:
                    self.websocket.pause_writing()
            elif frame.opcode == OP_TEXT:
                payload = json.loads(frame.data)
                self.sent_types.append(payload['type'])
                if payload['type'] == 'session.update':
                    if self.configuration_error:
                        self.feed({'type': 'error', 'error': {'code': 'invalid_value'}})
                    else:
                        self.feed({'type': 'session.updated'})
                elif payload['type'] == 'input_audio_buffer.commit':
                    self.feed({
                        'type': 'conversation.item.input_audio_transcription.completed',
                        'transcript': 'offline fixture',
                    })

    def abort(self):
        self.aborts += 1
        self.close()

    def close(self):
        if not self.closed.is_set():
            self.closed.set()
            asyncio.get_running_loop().call_soon(self.websocket.connection_lost, None)


def memory_realtime_client(monkeypatch, *, block_close_write=False, configuration_error=False):
    import httpx
    from openai import AsyncOpenAI
    from websockets.asyncio.client import ClientConnection
    import websockets.asyncio.client
    from websockets.client import ClientProtocol
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    from websockets.protocol import OPEN
    from websockets.uri import parse_uri

    transports = []

    async def memory_connect(url, **kwargs):
        websocket = ClientConnection(
            ClientProtocol(parse_uri(url), state=OPEN),
            close_timeout=kwargs.get('close_timeout', 10),
        )
        websocket.response = Response(101, 'Switching Protocols', Headers())
        transport = MemoryRealtimeTransport(
            websocket, block_close_write=block_close_write,
            configuration_error=configuration_error,
        )
        websocket.connection_made(transport)
        transport.feed({'type': 'session.created', 'session': {'type': 'transcription'}})
        transports.append(transport)
        return websocket

    def forbid_http(request):
        raise AssertionError('No HTTP request is allowed in native close tests')

    monkeypatch.setattr(websockets.asyncio.client, 'connect', memory_connect)
    sdk = AsyncOpenAI(
        api_key='not-a-real-key',
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(forbid_http)),
    )
    client = OpenAIRealtimeSTTClient(async_client=sdk, timeout=0.5, close_timeout=0.06)
    return client, sdk, transports


@pytest.mark.parametrize('block_close_write', [False, True])
@pytest.mark.parametrize('ending', ['cancel', 'finish', 'configure_error'])
def test_native_transport_is_released_when_peer_never_acknowledges_close(
    monkeypatch, block_close_write, ending,
):
    from websockets.protocol import CLOSED

    async def scenario():
        client, sdk, transports = memory_realtime_client(
            monkeypatch, block_close_write=block_close_write,
            configuration_error=ending == 'configure_error',
        )
        turn = None
        try:
            if ending == 'configure_error':
                with pytest.raises(RealtimeTranscriptionError, match='설정에 실패'):
                    await client.start()
            else:
                turn = await client.start()
                await turn.append(b'\x00\x00')
                if ending == 'finish':
                    assert await turn.finish() == 'offline fixture'
                await asyncio.wait_for(turn.cancel(), timeout=0.5)
            transport = transports[0]
            assert transport.close_received.is_set()
            assert transport.closed.is_set()
            assert transport.aborts == 1
            await asyncio.wait_for(transport.websocket.wait_closed(), timeout=0.5)
            assert transport.websocket.state == CLOSED
            assert transport.websocket.connection_lost_waiter.done()
            if turn is not None:
                await turn.close()
                assert transport.aborts == 1
                # A new utterance must use an independent, usable connection.
                next_turn = await client.start()
                await next_turn.append(b'\x01\x00')
                assert await next_turn.finish() == 'offline fixture'
                await next_turn.close()
                assert len(transports) == 2
                assert transports[1].closed.is_set()
                await asyncio.wait_for(transports[1].websocket.wait_closed(), timeout=0.5)
                assert transports[1].websocket.state == CLOSED
        finally:
            for transport in transports:
                transport.close()
            await sdk.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('block_close_write', [False, True])
def test_cancelling_close_waiter_does_not_cancel_native_transport_cleanup(
    monkeypatch, block_close_write,
):
    from websockets.protocol import CLOSED

    async def scenario():
        client, sdk, transports = memory_realtime_client(
            monkeypatch, block_close_write=block_close_write,
        )
        try:
            turn = await client.start()
            closing = asyncio.create_task(turn.cancel())
            transport = transports[0]
            await asyncio.wait_for(transport.close_received.wait(), timeout=0.5)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            await asyncio.wait_for(turn.close(), timeout=0.5)
            assert transport.closed.is_set()
            assert transport.aborts == 1
            await asyncio.wait_for(transport.websocket.wait_closed(), timeout=0.5)
            assert transport.websocket.state == CLOSED
        finally:
            for transport in transports:
                transport.close()
            await sdk.close()

    asyncio.run(scenario())
