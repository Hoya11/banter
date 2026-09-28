"""생성 완료 후 재생 중단 보고를 다음 대화 이력에 반영하는 회귀 테스트.

실제 음성을 재생하거나 공급자를 호출하지 않고, 브라우저 보고와 다음 LLM
프롬프트 사이의 상태 전달을 검증한다. 들은 단어의 위치는 추정하지 않는다.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import WebSocketDisconnect

from api.app import create_app

FIRST_TEXT = '첫 번째 발화의 전체 내용입니다.'
PREFETCH_TEXT = '미리 생성한 두 번째 발화의 전체 내용입니다.'
USER_TEXT = '재생 중 끼어든 사용자 입력'
SECOND_USER_TEXT = '그다음 사용자 입력'
INTERRUPTED = ' [말하다 끊김]'


class MemorySocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def accept(self):
        pass

    async def receive_text(self):
        if self.closed:
            raise WebSocketDisconnect(code=1001)
        message = await self.incoming.get()
        if message is None:
            raise WebSocketDisconnect(code=1001)
        return json.dumps(message)

    async def send_json(self, message):
        self.sent.append(message)
        self.outgoing.put_nowait(message)

    def feed(self, message):
        self.incoming.put_nowait(message)

    async def until(self, kind):
        while True:
            message = await self.outgoing.get()
            if message['type'] == kind:
                return message


class RecordingStream:
    def __init__(self, *, first_text=FIRST_TEXT):
        self.prompts = []
        self.first_text = first_text

    async def complete_stream(self, system, user):
        self.prompts.append(user)
        number = len(self.prompts)
        speaker = 'ai_a' if number % 2 else 'ai_b'
        yield json.dumps({'next_speaker': speaker}) + '\n'
        yield self.first_text if number == 1 else self.text(number)

    @staticmethod
    def text(number):
        return {1: FIRST_TEXT, 2: PREFETCH_TEXT}.get(number, f'후속 발화 {number}의 전체 내용입니다.')


class FakeTTS:
    async def synthesize(self, text, voice, speed=None):
        return b'fake-audio'


async def settle():
    for _ in range(40):
        await asyncio.sleep(0)


def report(audio, *, outcome='paused', segment_id=None):
    return {
        'outcome': outcome,
        'elapsed_ms': 3.2,
        'segment_id': segment_id or f'audio-{audio["seq"]}',
    }


def history_line(prompt, text):
    return next(line for line in prompt.splitlines() if text in line)


@asynccontextmanager
async def playing(*, prefetched=False, max_turns=5, first_text=FIRST_TEXT, first_segments=1, tts_client=None, ack_sec=10):
    stream = RecordingStream(first_text=first_text)
    app = create_app(
        stream, tts_client=tts_client or FakeTTS(), unified=True,
        radio_sec=0, ack_sec=ack_sec, max_turns=max_turns,
    )
    endpoint = next(route.endpoint for route in app.routes if route.path == '/ws')
    socket = MemorySocket()
    task = asyncio.create_task(endpoint(socket))
    try:
        await socket.until('hello')
        socket.feed({'type': 'session_start'})
        await socket.until('session_started')
        audio = await socket.until('audio')
        await settle()  # 생성 종료와 다음 턴의 사전 생성을 완료시킨다.
        for _ in range(first_segments - 1):
            audio = await socket.until('audio')
        if prefetched:
            socket.feed({'type': 'played', 'seq': audio['seq']})
            audio = await socket.until('audio')
            await settle()
        yield socket, stream, audio, task
    finally:
        socket.closed = True
        socket.feed(None)
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


async def send_input(socket, *, text=USER_TEXT, event_id='text-1', audio_stop=None, ack_audio=None):
    message = {'type': 'say', 'text': text, 'client_event_id': event_id}
    if audio_stop is not None:
        message['audio_stop'] = audio_stop
    socket.feed(message)
    if ack_audio is not None:
        socket.feed({'type': 'played', 'seq': ack_audio['seq']})
    audio = await socket.until('audio')
    await settle()
    return audio


@pytest.mark.parametrize('prefetched', [False, True], ids=['streamed', 'prefetched'])
@pytest.mark.parametrize('input_kind', ['hold', 'say'])
def test_paused_completed_utterance_is_marked_in_next_prompt(prefetched, input_kind):
    async def scenario():
        async with playing(prefetched=prefetched) as (socket, stream, audio, _):
            stopped_text = audio['text']
            if input_kind == 'hold':
                socket.feed({
                    'type': 'hold', 'client_event_id': 'hold-1',
                    'audio_stop': report(audio),
                })
                socket.feed({'type': 'played', 'seq': audio['seq']})
                await settle()
                await send_input(socket)
            else:
                await send_input(socket, audio_stop=report(audio), ack_audio=audio)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert history_line(prompt, stopped_text).endswith(stopped_text + INTERRUPTED)
            assert prompt.count(INTERRUPTED) == 1
            assert prompt.count(USER_TEXT) == 1
            if prefetched:
                assert INTERRUPTED not in history_line(prompt, FIRST_TEXT)

    run(scenario)


@pytest.mark.parametrize('outcome', ['queued_only', 'idle', 'pause_failed', 'unknown_segment'])
def test_non_paused_or_unknown_audio_does_not_mark_completed_history(outcome):
    async def scenario():
        async with playing() as (socket, stream, audio, _):
            audio_stop = report(
                audio,
                outcome='paused' if outcome == 'unknown_segment' else outcome,
                segment_id='audio-999' if outcome == 'unknown_segment' else None,
            )
            socket.feed({
                'type': 'hold', 'client_event_id': 'hold-1', 'audio_stop': audio_stop,
            })
            await send_input(socket)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert FIRST_TEXT in prompt
            assert INTERRUPTED not in prompt

    run(scenario)


def test_normal_playback_completion_does_not_mark_history():
    async def scenario():
        async with playing(prefetched=True) as (socket, stream, _, _):
            await send_input(socket)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert FIRST_TEXT in prompt and PREFETCH_TEXT in prompt
            assert INTERRUPTED not in prompt

    run(scenario)


def test_late_pause_for_acknowledged_audio_does_not_change_any_utterance():
    async def scenario():
        async with playing(prefetched=True) as (socket, stream, audio, _):
            assert audio['seq'] == 2
            socket.feed({
                'type': 'hold', 'client_event_id': 'late-hold',
                'audio_stop': report(audio, segment_id='audio-1'),
            })
            await send_input(socket)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert FIRST_TEXT in prompt and PREFETCH_TEXT in prompt
            assert INTERRUPTED not in prompt

    run(scenario)


def test_duplicate_hold_cannot_mark_a_newer_utterance():
    async def scenario():
        async with playing() as (socket, stream, audio, _):
            socket.feed({
                'type': 'hold', 'client_event_id': 'hold-1', 'audio_stop': report(audio),
            })
            reply = await send_input(socket)
            first_prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert history_line(first_prompt, FIRST_TEXT).endswith(INTERRUPTED)
            socket.feed({
                'type': 'hold', 'client_event_id': 'hold-1', 'audio_stop': report(reply),
            })
            await send_input(socket, text=SECOND_USER_TEXT, event_id='text-2')
            prompt = next(p for p in stream.prompts if SECOND_USER_TEXT in p)
            assert prompt.count(INTERRUPTED) == 1
            assert history_line(prompt, FIRST_TEXT).endswith(INTERRUPTED)
            assert INTERRUPTED not in history_line(prompt, reply['text'])

    run(scenario)


def test_hold_off_rebuilds_prefetch_with_the_recorded_playback_interruption():
    async def scenario():
        async with playing(max_turns=2) as (socket, stream, audio, task):
            assert len(stream.prompts) == 2
            assert INTERRUPTED not in stream.prompts[1]
            socket.feed({
                'type': 'hold', 'client_event_id': 'hold-1', 'audio_stop': report(audio),
            })
            socket.feed({'type': 'played', 'seq': audio['seq']})
            await settle()
            socket.feed({'type': 'hold_off', 'client_event_id': 'hold-1'})
            await socket.until('done')
            await task
            assert len(stream.prompts) == 3
            assert history_line(stream.prompts[2], FIRST_TEXT).endswith(FIRST_TEXT + INTERRUPTED)
            assert [m['text'] for m in socket.sent if m['type'] == 'audio'] == [
                FIRST_TEXT, stream.text(3),
            ]

    run(scenario)


def test_second_sentence_audio_marks_the_complete_utterance_once():
    async def scenario():
        full_text = '앞 문장의 전체 내용입니다. 뒤 문장의 전체 내용입니다.'
        async with playing(first_text=full_text, first_segments=2) as (socket, stream, audio, _):
            assert audio['seq'] == 2
            assert audio['cont'] is True
            await send_input(socket, audio_stop=report(audio), ack_audio=audio)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert history_line(prompt, full_text).endswith(full_text + INTERRUPTED)
            assert prompt.count(INTERRUPTED) == 1

    run(scenario)


def test_failed_audio_synthesis_cannot_mark_text_as_playback_interrupted():
    class FailedTTS:
        async def synthesize(self, text, voice, speed=None):
            raise RuntimeError('가짜 합성 실패')

    async def scenario():
        async with playing(tts_client=FailedTTS()) as (socket, stream, audio, _):
            assert audio['audio'] is None
            await send_input(socket, audio_stop=report(audio))
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert FIRST_TEXT in prompt
            assert INTERRUPTED not in prompt

    run(scenario)


def test_pacing_timeout_preserves_still_playing_audio_for_interruption_history():
    async def scenario():
        async with playing(ack_sec=0.02) as (socket, stream, first_audio, _):
            # 실제 played 보고 없이 다음 audio가 오면 재생 확인의 대기 상한을 지난 것이다.
            second_audio = await socket.until('audio')
            assert second_audio['seq'] == 2
            socket.feed({
                'type': 'hold', 'client_event_id': 'hold-after-timeout',
                'audio_stop': report(first_audio),
            })
            await send_input(socket)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert history_line(prompt, FIRST_TEXT).endswith(FIRST_TEXT + INTERRUPTED)
            assert INTERRUPTED not in history_line(prompt, PREFETCH_TEXT)
            assert prompt.count(INTERRUPTED) == 1

    run(scenario)


def test_actual_ack_after_pacing_timeout_preserves_newer_unacknowledged_audio():
    async def scenario():
        async with playing(ack_sec=0.02) as (socket, stream, first_audio, _):
            second_audio = await socket.until('audio')
            third_audio = await socket.until('audio')
            assert third_audio['seq'] == 3
            # 페이스 장부는 2까지 진행했지만 브라우저가 완료 보고한 것은 1뿐이다.
            socket.feed({'type': 'played', 'seq': first_audio['seq']})
            socket.feed({
                'type': 'hold', 'client_event_id': 'hold-after-lower-ack',
                'audio_stop': report(second_audio),
            })
            await send_input(socket)
            prompt = next(p for p in stream.prompts if USER_TEXT in p)
            assert history_line(prompt, PREFETCH_TEXT).endswith(PREFETCH_TEXT + INTERRUPTED)
            assert INTERRUPTED not in history_line(prompt, FIRST_TEXT)
            assert INTERRUPTED not in history_line(prompt, third_audio['text'])
            assert prompt.count(INTERRUPTED) == 1

    run(scenario)
