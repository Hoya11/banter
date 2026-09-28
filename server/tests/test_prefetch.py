"""사전 생성이 늦어지는 경계의 입력 처리와 결과 폐기 회귀 테스트.

외부 공급자와 네트워크 없이 실제 WebSocket 핸들러를 실행한다.
시간 제한은 테스트 정지 방지용이며 성능 측정값이 아니다.
"""

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import WebSocketDisconnect

from api.app import create_app
from api.event_log import EventName

FIRST_TEXT = '첫 번째 발화입니다.'
PREFETCH_TEXT = '사용자 개입 전 미리 준비한 발화입니다.'
REPLY_TEXT = '새로운 입력에 대한 답변입니다.'
USER_TEXT = '기다리는 동안 끼어든 사용자 입력'


class MemorySocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.sent = []
        self.played_received = asyncio.Event()
        self.closed = False
        self.pending_receivers = 0
        self.cancelled_receive_reply = None
        self.returned_on_cancel = False
        self.pause_cancelled_receive = False
        self.receive_cancellation_started = asyncio.Event()
        self.release_cancelled_receive = asyncio.Event()

    async def accept(self):
        pass

    async def receive_text(self):
        if self.closed:
            raise WebSocketDisconnect(code=1001)
        self.pending_receivers += 1
        try:
            value = await self.incoming.get()
        except asyncio.CancelledError:
            if self.cancelled_receive_reply is None:
                raise
            if self.pause_cancelled_receive:
                self.receive_cancellation_started.set()
                try:
                    await self.release_cancelled_receive.wait()
                except asyncio.CancelledError:
                    pass  # 외부 취소와 이미 읽은 메시지 반환이 겹치는 소켓을 재현한다.
            value = self.cancelled_receive_reply
            self.cancelled_receive_reply = None
            self.returned_on_cancel = True
        finally:
            self.pending_receivers -= 1
        if value is None:
            raise WebSocketDisconnect(code=1001)
        if value.get('type') == 'played':
            self.played_received.set()
        return json.dumps(value)

    async def send_json(self, message):
        self.sent.append(message)
        self.outgoing.put_nowait(message)

    def feed(self, message):
        self.incoming.put_nowait(message)

    def disconnect(self):
        self.closed = True
        self.feed(None)

    async def until(self, kind):
        while True:
            message = await self.outgoing.get()
            if message['type'] == kind:
                return message


class GatedStream:
    def __init__(self, *, fail_prefetch=False):
        self.prompts = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.fail_prefetch = fail_prefetch

    async def complete_stream(self, system, user):
        self.prompts.append(user)
        number = len(self.prompts)
        if number == 2:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            if self.fail_prefetch:
                raise RuntimeError('가짜 사전 생성 실패')
        speaker = 'ai_b' if number == 2 else 'ai_a'
        yield json.dumps({'next_speaker': speaker}) + '\n'
        yield {1: FIRST_TEXT, 2: PREFETCH_TEXT}.get(number, REPLY_TEXT)
        if number == 2:
            self.completed.set()


class FakeTTS:
    async def synthesize(self, text, voice, speed=None):
        return b'fake-audio'


class FakeSTT:
    async def start(self, *, on_delta):
        return self

    async def append(self, audio):
        pass

    async def finish(self):
        return USER_TEXT

    async def cancel(self):
        pass


class Recorder:
    def __init__(self):
        self.events = []

    def record(self, event, **fields):
        self.events.append((event, fields))

    def held(self, event_id):
        return any(
            event == EventName.HOLD_RECEIVED
            and fields['client_event_id'] == event_id
            for event, fields in self.events
        )


async def settle():
    # 준비된 작업들이 진행하게 양보한다. 실제 경과 시간이나 코드 위치를 재지 않는다.
    for _ in range(40):
        await asyncio.sleep(0)


@asynccontextmanager
async def waiting_prefetch(*, fail_prefetch=False):
    stream = GatedStream(fail_prefetch=fail_prefetch)
    recorder = Recorder()
    app = create_app(
        stream,
        tts_client=FakeTTS(),
        streaming_stt_client=FakeSTT(),
        unified=True,
        radio_sec=0,
        ack_sec=10,
        max_turns=2,
        event_recorder=recorder,
    )
    endpoint = next(route.endpoint for route in app.routes if route.path == '/ws')
    socket = MemorySocket()
    task = asyncio.create_task(endpoint(socket))
    try:
        await socket.until('hello')
        socket.feed({'type': 'session_start'})
        await socket.until('session_started')
        first_audio = await socket.until('audio')
        assert first_audio['text'] == FIRST_TEXT
        await stream.entered.wait()
        socket.feed({'type': 'played', 'seq': first_audio['seq']})
        await socket.played_received.wait()
        await settle()
        yield socket, stream, recorder, task
    finally:
        socket.disconnect()
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def run(scenario):
    async def bounded():
        async with asyncio.timeout(2):
            await scenario()

    asyncio.run(bounded())


def assert_only_fresh_reply(socket, stream):
    assert len(stream.prompts) == 3
    assert stream.prompts[2].count(USER_TEXT) == 1
    assert PREFETCH_TEXT not in stream.prompts[2]
    assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [
        FIRST_TEXT, REPLY_TEXT,
    ]
    assert len([m for m in socket.sent if m['type'] == 'start']) == 2
    assert socket.sent[-1]['type'] == 'done'


def test_pending_prefetch_say_cancels_old_work_without_spending_a_turn():
    async def scenario():
        async with waiting_prefetch() as (socket, stream, _, task):
            socket.feed({'type': 'say', 'text': USER_TEXT, 'client_event_id': 'new-text'})
            await settle()
            assert stream.cancelled.is_set(), '사용자 입력은 공급자 완료를 기다리면 안 된다'
            assert not stream.release.is_set()
            await socket.until('done')
            await task
            assert_only_fresh_reply(socket, stream)
            assert any(
                m['type'] == 'start' and m.get('after_client_event_id') == 'new-text'
                for m in socket.sent
            )

    run(scenario)


@pytest.mark.parametrize('source', ['say', 'streaming_stt'])
def test_hold_blocks_completed_prefetch_until_new_input_invalidates_it(source):
    async def scenario():
        async with waiting_prefetch() as (socket, stream, recorder, task):
            socket.feed({'type': 'hold', 'client_event_id': 'voice-1'})
            await settle()
            assert recorder.held('voice-1'), '사전 생성 대기 중에도 hold를 처리해야 한다'
            offset = len(socket.sent)
            stream.release.set()
            await stream.completed.wait()
            await settle()
            assert not any(m['type'] in {'start', 'end', 'audio'} for m in socket.sent[offset:])
            if source == 'say':
                socket.feed({'type': 'say', 'text': USER_TEXT, 'client_event_id': 'new-text'})
            else:
                socket.feed({
                    'type': 'voice_stream_start', 'client_event_id': 'voice-1',
                    'encoding': 'pcm_s16le', 'sample_rate_hz': 24000, 'channels': 1,
                })
                for sequence in range(1, 4):
                    socket.feed({
                        'type': 'voice_stream_chunk', 'client_event_id': 'voice-1',
                        'sequence_number': sequence,
                        'audio': base64.b64encode(b'\x00\x00' * 2400).decode(),
                    })
                socket.feed({
                    'type': 'voice_stream_commit', 'client_event_id': 'voice-1',
                    'final_sequence_number': 3, 'total_samples': 7200,
                })
                transcript = await socket.until('you')
                assert transcript['text'] == USER_TEXT
            await socket.until('done')
            await task
            assert_only_fresh_reply(socket, stream)

    run(scenario)


@pytest.mark.parametrize('release_kind', ['hold_off', 'expiry'])
def test_hold_release_resumes_prefetch_without_an_extra_generation(monkeypatch, release_kind):
    if release_kind == 'expiry':
        monkeypatch.setattr('api.app.HOLD_SEC', 0.1)

    async def scenario():
        async with waiting_prefetch() as (socket, stream, recorder, task):
            socket.feed({'type': 'hold', 'client_event_id': 'voice-1'})
            await settle()
            assert recorder.held('voice-1')
            stream.release.set()
            await stream.completed.wait()
            await settle()
            assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [FIRST_TEXT]
            if release_kind == 'hold_off':
                socket.feed({'type': 'hold_off', 'client_event_id': 'voice-1'})
            else:
                error = await socket.until('stt_error')
                assert error['code'] == 'hold_expired'
                assert error['client_event_id'] == 'voice-1'
            await socket.until('done')
            await task
            assert len(stream.prompts) == 2
            assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [
                FIRST_TEXT, PREFETCH_TEXT,
            ]

    run(scenario)


def test_prefetch_completion_and_say_ready_together_prioritize_input():
    async def scenario():
        async with waiting_prefetch() as (socket, stream, _, task):
            # 같은 event loop 차례에 완료와 입력을 준비한다. 완료 알림을 먼저 예약한다.
            stream.release.set()
            socket.feed({'type': 'say', 'text': USER_TEXT, 'client_event_id': 'same-time'})
            await socket.until('done')
            await task
            assert_only_fresh_reply(socket, stream)

    run(scenario)


def test_disconnect_cancels_pending_prefetch_without_provider_completion():
    async def scenario():
        async with waiting_prefetch() as (socket, stream, _, task):
            socket.disconnect()
            await settle()
            assert stream.cancelled.is_set(), '연결 종료는 사전 생성 공급자를 기다리면 안 된다'
            await task
            assert not stream.release.is_set()
            assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [FIRST_TEXT]

    run(scenario)


def test_failed_prefetch_falls_back_to_live_generation():
    async def scenario():
        async with waiting_prefetch(fail_prefetch=True) as (socket, stream, _, task):
            stream.release.set()
            await socket.until('done')
            await task
            assert len(stream.prompts) == 3
            assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [
                FIRST_TEXT, REPLY_TEXT,
            ]
            assert len([m for m in socket.sent if m['type'] == 'start']) == 2

    run(scenario)


def test_session_cancellation_closes_pending_prefetch_and_input_reader():
    async def scenario():
        async with waiting_prefetch() as (socket, stream, _, task):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await settle()
            assert stream.cancelled.is_set()
            assert socket.pending_receivers == 0
            assert not stream.release.is_set()

    run(scenario)


def test_prefetch_completion_preserves_input_returned_during_reader_cancellation():
    async def scenario():
        async with waiting_prefetch() as (socket, stream, _, task):
            # 수신 작업이 취소되는 순간 이미 읽은 메시지를 반환하는 경합을 재현한다.
            socket.cancelled_receive_reply = {
                'type': 'say', 'text': USER_TEXT, 'client_event_id': 'received-on-cancel',
            }
            stream.release.set()
            await socket.until('done')
            await task
            assert socket.returned_on_cancel
            assert_only_fresh_reply(socket, stream)

    run(scenario)


def test_session_cancellation_wins_over_input_returned_by_cancelled_reader():
    async def scenario():
        async with waiting_prefetch() as (socket, stream, _, task):
            socket.cancelled_receive_reply = {
                'type': 'say', 'text': USER_TEXT, 'client_event_id': 'cancelled-session',
            }
            socket.pause_cancelled_receive = True
            stream.release.set()
            await socket.receive_cancellation_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert socket.returned_on_cancel
            assert socket.pending_receivers == 0
            assert len(stream.prompts) == 2
            assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [FIRST_TEXT]
            assert not any(m['type'] == 'done' for m in socket.sent)

    run(scenario)
