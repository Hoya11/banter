"""LLMClient 구현체 — judge(judge.py)에 주입한다.

judge용 LLM은 OpenAI로 시작한다(채점 심판은 하나로 고정, Phase 1 [결정]).
judge.py의 LLMClient Protocol을 만족하므로, 앞으로 다른 provider·앙상블로 바꿔도
judge()·rubric·schema 계약은 불변이다.
"""

import asyncio
import base64
import os
import re
from collections.abc import Callable

from openai import APIError, OpenAI
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State


_DIAGNOSTIC_TEXT_FIELDS = frozenset({
    'error_type', 'stage', 'provider_event_type', 'provider_error_type',
    'provider_error_code', 'provider_error_param', 'close_state', 'session_type',
})
_DIAGNOSTIC_NUMBER_FIELDS = {
    'close_received_code': (1000, 4999),
    'close_sent_code': (1000, 4999),
    'http_status': (100, 599),
}


def _safe_diagnostics(values: dict) -> dict[str, str | int]:
    """원문 메시지 대신 짧은 구조화된 식별자와 상태 코드만 보존한다."""
    safe = {}
    for key, value in values.items():
        if key in _DIAGNOSTIC_TEXT_FIELDS:
            if (isinstance(value, str) and len(value) <= 96
                    and not value.lower().startswith(('sk-', 'bearer'))
                    and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.\[\]-]*', value)):
                safe[key] = value
        elif key in _DIAGNOSTIC_NUMBER_FIELDS:
            lower, upper = _DIAGNOSTIC_NUMBER_FIELDS[key]
            if isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper:
                safe[key] = int(value)
    return safe


class RealtimeTranscriptionError(RuntimeError):
    """원문 오디오나 공급자 메시지 없이 진단 필드를 보존하는 STT 오류."""

    def __init__(self, message: str, *, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self._diagnostics = _safe_diagnostics(diagnostics or {})


def realtime_stt_diagnostics(error: BaseException) -> dict[str, str | int]:
    """서버 로그용 안전한 필드. exception str, close reason과 응답 본문은 제외한다."""
    if isinstance(error, RealtimeTranscriptionError):
        return {'error_type': type(error).__name__, **error._diagnostics}
    fields = {'error_type': type(error).__name__}
    if isinstance(error, ConnectionClosed):
        received, sent = error.rcvd, error.sent
        fields['close_state'] = (
            'received_and_sent' if received is not None and sent is not None
            else 'received_only' if received is not None
            else 'sent_only' if sent is not None else 'no_close_frame'
        )
        if received is not None:
            fields['close_received_code'] = received.code
        if sent is not None:
            fields['close_sent_code'] = sent.code
    if isinstance(error, APIError):
        fields.update({
            'provider_error_type': error.type,
            'provider_error_code': error.code,
            'provider_error_param': error.param,
        })
    status = getattr(error, 'status_code', None)
    if status is None:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
    fields['http_status'] = status
    return _safe_diagnostics(fields)


def _transport_failure(error: Exception, stage: str) -> Exception:
    if isinstance(error, (RealtimeTranscriptionError, TimeoutError)):
        return error
    return RealtimeTranscriptionError(
        'Realtime STT 연결 처리에 실패했습니다',
        diagnostics={**realtime_stt_diagnostics(error), 'stage': stage},
    )


def _event_field(event, name: str):
    """SDK 이벤트와 테스트용 dict에서 같은 방식으로 값을 읽는다."""
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _provider_failure(message: str, event, stage: str) -> RealtimeTranscriptionError:
    error = _event_field(event, 'error')
    return RealtimeTranscriptionError(message, diagnostics={
        'stage': stage,
        'provider_event_type': _event_field(event, 'type'),
        'provider_error_type': _event_field(error, 'type'),
        'provider_error_code': _event_field(error, 'code'),
        'provider_error_param': _event_field(error, 'param'),
    })


class _OpenAIRealtimeSTTTurn:
    """OpenAI Realtime 연결 하나로 push-to-talk 한 턴을 처리한다."""

    def __init__(
        self,
        manager,
        connection,
        *,
        model: str,
        sample_rate: int,
        language: str,
        delay: str,
        timeout: float,
        close_timeout: float,
        on_delta: Callable[[str], None] | None,
    ) -> None:
        self._manager = manager
        self._connection = connection
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._delay = delay
        self._timeout = timeout
        self._close_timeout = close_timeout
        self._on_delta = on_delta
        self._completed = asyncio.get_running_loop().create_future()
        self._reader_task = None
        self._committed = False
        self._closed = False
        self._close_task = None

    async def configure(self) -> None:
        """전사 세션 생성을 확인한 뒤 PCM 입력과 수동 턴 확정을 설정한다."""
        try:
            await self._wait_until_created()
        except Exception as exc:
            raise _transport_failure(exc, 'session_create') from None
        await self._send_configuration(
            {
                'type': 'session.update',
                'session': {
                    'type': 'transcription',
                    'audio': {
                        'input': {
                            'format': {
                                'type': 'audio/pcm',
                                'rate': self._sample_rate,
                            },
                            'transcription': {
                                'model': self._model,
                                'languages': [self._language],
                                'delay': self._delay,
                            },
                            'turn_detection': None,
                        }
                    },
                },
            }
        )
        try:
            await self._wait_until_configured()
        except Exception as exc:
            raise _transport_failure(exc, 'configure_ack') from None
        self._reader_task = asyncio.create_task(self._read_events())

    async def _send_configuration(self, event: dict) -> None:
        try:
            await self._connection.send(event)
        except ConnectionClosed as exc:
            # recv can still expose a buffered error after send observes a closed socket.
            failure = await self._read_pending_configuration_error(exc)
            raise failure or _transport_failure(exc, 'configure_send') from None
        except Exception as exc:
            raise _transport_failure(exc, 'configure_send') from None

    async def _read_pending_configuration_error(
        self, closed: ConnectionClosed,
    ) -> RealtimeTranscriptionError | None:
        async def read_error():
            while True:
                event = await self._connection.recv()
                if _event_field(event, 'type') == 'error':
                    failure = _provider_failure(
                        'Realtime STT 세션 설정에 실패했습니다', event, 'configure_send'
                    )
                    close_fields = realtime_stt_diagnostics(closed)
                    close_fields.pop('error_type', None)
                    return RealtimeTranscriptionError(
                        'Realtime STT 세션 설정에 실패했습니다',
                        diagnostics={**close_fields, **failure._diagnostics},
                    )

        try:
            return await asyncio.wait_for(read_error(), timeout=min(0.1, self._timeout))
        except Exception:
            # Diagnostic collection must not replace the original connection failure.
            return None

    async def _wait_until_created(self) -> None:
        while True:
            event = await self._connection.recv()
            event_type = _event_field(event, 'type')
            if event_type == 'error':
                raise _provider_failure(
                    'Realtime STT 세션 생성에 실패했습니다', event, 'session_create'
                )
            if event_type == 'session.created':
                session_type = _event_field(_event_field(event, 'session'), 'type')
                if session_type != 'transcription':
                    raise RealtimeTranscriptionError(
                        'Realtime STT 전사 세션이 생성되지 않았습니다',
                        diagnostics={'stage': 'session_create', 'session_type': session_type},
                    )
                return

    async def _wait_until_configured(self) -> None:
        while True:
            event = await self._connection.recv()
            event_type = _event_field(event, 'type')
            if event_type == 'session.updated':
                return
            if event_type == 'error':
                raise _provider_failure(
                    'Realtime STT 세션 설정에 실패했습니다', event, 'configure_ack'
                )

    async def _read_events(self) -> None:
        try:
            while True:
                event = await self._connection.recv()
                event_type = _event_field(event, 'type')
                if event_type == 'conversation.item.input_audio_transcription.delta':
                    delta = _event_field(event, 'delta')
                    if delta and self._on_delta is not None:
                        try:
                            self._on_delta(delta)
                        except Exception:
                            pass
                elif event_type == 'conversation.item.input_audio_transcription.completed':
                    if not self._completed.done():
                        transcript = str(_event_field(event, 'transcript') or '').strip()
                        self._completed.set_result(transcript)
                    return
                elif event_type in {
                    'conversation.item.input_audio_transcription.failed',
                    'error',
                }:
                    if not self._completed.done():
                        self._completed.set_exception(
                            _provider_failure('Realtime STT 전사에 실패했습니다', event, 'receive')
                        )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._completed.done():
                self._completed.set_exception(_transport_failure(exc, 'receive'))

    def _reader_failure(self) -> BaseException | None:
        if self._completed.done() and not self._completed.cancelled():
            return self._completed.exception()
        return None

    async def append(self, audio: bytes) -> None:
        if self._closed or self._committed:
            raise RealtimeTranscriptionError('종료된 Realtime STT 턴입니다')
        if not audio:
            return
        try:
            if error := self._reader_failure():
                raise error
            await self._connection.send(
                {
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(audio).decode('ascii'),
                }
            )
        except Exception as exc:
            self._schedule_close()
            raise self._reader_failure() or _transport_failure(exc, 'append') from None

    async def finish(self) -> str:
        if self._closed or self._committed:
            raise RealtimeTranscriptionError('이미 확정된 Realtime STT 턴입니다')
        self._committed = True
        try:
            if error := self._reader_failure():
                raise error
            await self._connection.send({'type': 'input_audio_buffer.commit'})
            return await asyncio.wait_for(
                asyncio.shield(self._completed), timeout=self._timeout
            )
        except TimeoutError:
            raise
        except Exception as exc:
            # A received provider error is more specific than a subsequent closed socket.
            raise self._reader_failure() or _transport_failure(exc, 'commit') from None
        finally:
            self._schedule_close()

    async def cancel(self) -> None:
        await self.close()

    async def _close_resources(self) -> None:
        task = self._reader_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if not self._completed.done():
            self._completed.cancel()
        elif not self._completed.cancelled():
            self._completed.exception()
        try:
            await asyncio.wait_for(
                self._manager.__aexit__(None, None, None),
                timeout=self._close_timeout,
            )
        except TimeoutError:
            print('[stt] Realtime 연결 종료 시간이 초과됐습니다')
        except Exception as exc:
            print(f'[stt] Realtime 연결 종료 실패 {realtime_stt_diagnostics(exc)}')
        finally:
            # SDK 3.x wraps the native websocket without exposing an abort method.
            # Cancelling close() can interrupt both its handshake and forced cleanup.
            # Abort any remaining transport, including a stalled write drain.
            websocket = getattr(self._connection, '_connection', None)
            transport = getattr(websocket, 'transport', None)
            if transport is not None and websocket.state is not State.CLOSED:
                transport.abort()

    def _schedule_close(self):
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        return self._close_task

    async def close(self) -> None:
        await asyncio.shield(self._schedule_close())


class OpenAIRealtimeSTTClient:
    """24 kHz PCM push-to-talk 입력을 OpenAI Realtime STT에 연결한다."""

    def __init__(
        self,
        model: str = 'gpt-live-transcribe',
        api_key: str | None = None,
        *,
        sample_rate: int = 24_000,
        language: str = 'ko',
        delay: str = 'low',
        timeout: float = 15.0,
        close_timeout: float = 2.0,
        async_client=None,
    ) -> None:
        key = api_key or os.environ.get('OPENAI_API_KEY')
        if async_client is None and not key:
            raise RuntimeError('OPENAI_API_KEY가 없다 - server/.env에 설정하라')
        if async_client is None:
            from openai import AsyncOpenAI

            async_client = AsyncOpenAI(api_key=key)
        self._client = async_client
        self.model = model
        self.sample_rate = sample_rate
        self.language = language
        self.delay = delay
        self.timeout = timeout
        self.close_timeout = close_timeout

    async def start(self, on_delta: Callable[[str], None] | None = None):
        """새 전사 턴을 열고 session.updated까지 확인한 뒤 반환한다."""
        # Select a transcription session; the model belongs in session audio input.
        manager = self._client.realtime.connect(
            extra_query={'intent': 'transcription'},
            # Let the native close handshake expire before the outer cleanup guard.
            websocket_connection_options={'close_timeout': self.close_timeout / 2},
        )
        try:
            connection = await manager.__aenter__()
        except TimeoutError:
            raise
        except Exception as exc:
            raise _transport_failure(exc, 'connect') from None
        turn = _OpenAIRealtimeSTTTurn(
            manager,
            connection,
            model=self.model,
            sample_rate=self.sample_rate,
            language=self.language,
            delay=self.delay,
            timeout=self.timeout,
            close_timeout=self.close_timeout,
            on_delta=on_delta,
        )
        try:
            await asyncio.wait_for(turn.configure(), timeout=self.timeout)
        except BaseException as exc:
            await turn.close()
            if isinstance(exc, Exception) and not isinstance(exc, TimeoutError):
                raise _transport_failure(exc, 'configure') from None
            raise
        return turn


class OpenAIClient:
    """OpenAI Chat Completions 기반 심판.

    JSON object 모드로 순수 JSON 출력을 강제하고(코드펜스·설명 텍스트 방지),
    temperature=0으로 채점 재현성을 확보한다.
    """

    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        api_key: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        tts_model: str = 'tts-1',
        stt_model: str = 'whisper-1',
    ):
        key = api_key or os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY가 없다 — server/.env에 설정하라')
        self._client = OpenAI(api_key=key)
        self._api_key = key
        self._async_client = None  # 스트리밍용, lazy 생성
        self._model = model
        self._json = json_mode  # judge=True(순수 JSON), 발화=False(자유 텍스트)
        self._temperature = temperature  # judge=0(재현성), 발화=높게(다양성)
        self._tts_model = tts_model  # TTS 모델 (예: tts-1, tts-1-hd, gpt-4o-mini-tts)
        self._stt_model = stt_model  # STT 모델 (예: whisper-1, gpt-4o-mini-transcribe)

    def complete(self, system: str, user: str) -> str:
        kwargs = {}
        if self._json:
            kwargs['response_format'] = {'type': 'json_object'}  # 순수 JSON 강제
        if self._temperature is not None:  # 일부 신형 모델은 temperature 고정(미지원)
            kwargs['temperature'] = self._temperature
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ''

    async def complete_stream(self, system: str, user: str):
        """발화를 토큰 단위로 스트리밍한다 (async generator). 발화 생성 전용."""
        from openai import AsyncOpenAI

        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        kwargs = {}
        if self._temperature is not None:  # 일부 신형 모델은 temperature 고정(미지원)
            kwargs['temperature'] = self._temperature
        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def transcribe(self, audio: bytes, mime: str = 'audio/webm') -> str:
        """음성(bytes)을 한국어 텍스트로 전사한다 (STT)."""
        from openai import AsyncOpenAI

        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        ext = mime.split('/')[-1].split(';')[0] or 'webm'
        # prompt 힌트는 넣지 않는다 — 짧거나 조용한 오디오에서 모델이 힌트 문장을
        # 그대로 환각 전사하는 부작용이 실측됨 (2026-08 로그)
        resp = await self._async_client.audio.transcriptions.create(
            model=self._stt_model,
            file=(f'speech.{ext}', audio, mime),
            language='ko',
        )
        return (resp.text or '').strip()

    async def synthesize(self, text: str, voice: str, speed: float | None = None) -> bytes:
        """텍스트를 음성(mp3 bytes)으로 합성한다 (TTS)."""
        from openai import AsyncOpenAI

        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        kwargs = {'speed': speed} if speed is not None else {}
        resp = await self._async_client.audio.speech.create(
            model=self._tts_model, voice=voice, input=text, response_format='mp3', **kwargs
        )
        return resp.content
