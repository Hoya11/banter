import asyncio

import pytest

from api.stt_stream import StreamingSTTRequest
from engine.eval.providers import RealtimeTranscriptionError, realtime_stt_diagnostics


def test_new_audio_preserves_the_provider_failure_already_waiting_for_delivery():
    async def scenario():
        failure = RealtimeTranscriptionError('초기 설정 실패', diagnostics={
            'stage': 'configure_ack',
            'provider_error_code': 'invalid_value',
            'provider_error_param': 'session',
        })

        class Provider:
            async def start(self, *, on_delta):
                raise failure

        tasks = []
        request = StreamingSTTRequest(
            Provider(), on_delta=lambda text: None, on_ready=lambda: None,
            track_task=tasks.append, max_pending_chunks=2,
            append_timeout=1, queue_timeout=1,
        )
        with pytest.raises(RealtimeTranscriptionError):
            await request.task
        # A raw WebSocket input wins when it arrives together with worker completion.
        with pytest.raises(RealtimeTranscriptionError) as caught:
            request.append(b'\x00\x00')
        assert caught.value is failure
        assert realtime_stt_diagnostics(caught.value) == {
            'error_type': 'RealtimeTranscriptionError',
            'stage': 'configure_ack',
            'provider_error_code': 'invalid_value',
            'provider_error_param': 'session',
        }
        request.cancel()
        assert all(task.done() for task in tasks)

    asyncio.run(scenario())
