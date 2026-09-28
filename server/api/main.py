"""실제 LLM client를 주입하는 uvicorn 진입점.

실행: cd server && PYTHONPATH=. uv run uvicorn api.main:app --reload
(OPENAI_API_KEY 필요. 키 없이 구조만 보려면 api.app:app을 쓰면 발화가 스텁으로 나온다.)
"""

import subprocess
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

from api.app import (
    STREAM_STT_CHUNK_SAMPLES,
    STREAM_STT_ENCODING,
    STREAM_STT_SAMPLE_RATE,
    create_app,
)
from api.event_log import JsonlEventRecorder
from engine.eval.elevenlabs import ElevenLabsClient
from engine.eval.providers import OpenAIClient, OpenAIRealtimeSTTClient

load_dotenv()

UTTERANCE_MODEL = 'gpt-5.6-luna'
SUPERVISOR_MODEL = 'gpt-4o-mini'
FILE_STT_MODEL = 'gpt-4o-mini-transcribe'
STREAM_STT_MODEL = 'gpt-live-transcribe'
STT_DELAY = 'low'
TTS_MODEL = 'eleven_v3'


def _resolve_build_revision() -> str:
    """실행 파일과 함께 남길 짧은 코드 버전을 구한다."""
    override = getenv('BANTER_BUILD_VERSION', '').strip()
    if override:
        return override[:80]

    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ['git', 'describe', '--always', '--dirty'],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    revision = result.stdout.strip()
    return revision[:80] if result.returncode == 0 and revision else 'unknown'


event_log_path = getenv('BANTER_EVENT_LOG')
stt_mode = getenv('BANTER_STT_MODE', 'record_then_transcribe').strip()
if stt_mode not in {'record_then_transcribe', 'streaming_push_to_talk'}:
    raise RuntimeError(
        'BANTER_STT_MODE은 record_then_transcribe 또는 '
        'streaming_push_to_talk이어야 합니다'
    )
interaction_mode = getenv('BANTER_INTERACTION_MODE', 'push_to_talk').strip()
if interaction_mode not in {'push_to_talk', 'vad'}:
    raise RuntimeError('BANTER_INTERACTION_MODE은 push_to_talk 또는 vad여야 합니다')
if interaction_mode == 'vad' and stt_mode != 'streaming_push_to_talk':
    raise RuntimeError('VAD에는 BANTER_STT_MODE=streaming_push_to_talk가 필요합니다')
elevenlabs_api_key = getenv('ELEVENLABS_API_KEY', '').strip()
tts_client = (
    ElevenLabsClient(api_key=elevenlabs_api_key, model_id=TTS_MODEL)
    if elevenlabs_api_key
    else None
)
file_stt_client = (
    OpenAIClient(stt_model=FILE_STT_MODEL)
    if stt_mode == 'record_then_transcribe'
    else None
)
streaming_stt_client = (
    OpenAIRealtimeSTTClient(
        model=STREAM_STT_MODEL,
        sample_rate=STREAM_STT_SAMPLE_RATE,
        delay=STT_DELAY,
    )
    if stt_mode == 'streaming_push_to_talk'
    else None
)
run_metadata = {
    'baseline_id': (
        'vad-v1'
        if interaction_mode == 'vad'
        else 'streaming-stt-v1'
        if stt_mode == 'streaming_push_to_talk'
        else 'push-to-talk-v1'
    ),
    'interaction_mode': interaction_mode,
    'stt_mode': stt_mode,
    'control_transport': 'websocket_json',
    'audio_uplink': (
        'websocket_json_base64_pcm16'
        if stt_mode == 'streaming_push_to_talk'
        else 'websocket_json_base64_webm'
    ),
    'audio_downlink': 'websocket_base64_mp3' if tts_client else 'disabled',
    'orchestration_mode': 'unified_v2',
    'audio_stop_semantics': (
        'vad_start_detected_to_html_media_pause_applied'
        if interaction_mode == 'vad'
        else 'pointerdown_to_html_media_pause_applied'
    ),
    'build_revision': _resolve_build_revision(),
    'utterance_model': UTTERANCE_MODEL,
    'supervisor_model': SUPERVISOR_MODEL,
    'stt_model': (
        STREAM_STT_MODEL
        if stt_mode == 'streaming_push_to_talk'
        else FILE_STT_MODEL
    ),
    'stt_file_model': (
        FILE_STT_MODEL if stt_mode == 'record_then_transcribe' else 'disabled'
    ),
    'stt_stream_encoding': (
        STREAM_STT_ENCODING if stt_mode == 'streaming_push_to_talk' else 'disabled'
    ),
    'stt_stream_sample_rate_hz': (
        STREAM_STT_SAMPLE_RATE if stt_mode == 'streaming_push_to_talk' else 0
    ),
    'stt_stream_chunk_samples': (
        STREAM_STT_CHUNK_SAMPLES if stt_mode == 'streaming_push_to_talk' else 0
    ),
    'stt_delay': STT_DELAY if stt_mode == 'streaming_push_to_talk' else 'disabled',
    'tts_model': TTS_MODEL if tts_client else 'disabled',
    'protocol_version': 1,
}

app = create_app(
    utterance_client=OpenAIClient(model=UTTERANCE_MODEL, temperature=None),
    supervisor_client=OpenAIClient(model=SUPERVISOR_MODEL, json_mode=True),
    # TTS provider 선택. 음성 없이 개발하려면 tts_client=None으로 둔다.
    # 보이스 조합은 duo.yaml의 voice_preset (chris-jessica | liam-laura).
    tts_client=tts_client,
    stt_client=file_stt_client,
    streaming_stt_client=streaming_stt_client,
    interaction_mode=interaction_mode,
    # prefetch가 생성과 합성을 재생 뒤에 숨기므로 발화 사이 틈만 남긴다.
    radio_sec=1.5,
    # supervisor v2는 화자 선정과 발화를 한 번에 처리한다.
    unified=True,
    event_recorder=JsonlEventRecorder(event_log_path) if event_log_path else None,
    run_metadata=run_metadata,
)
