"""실제 서버 진입점의 선택형 provider 조립을 검증한다."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_main_starts_without_elevenlabs_key():
    """ElevenLabs 키가 비어 있으면 외부 호출 없이 텍스트 모드로 조립된다."""
    server_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHON_DOTENV_DISABLED'] = '1'
    env['BANTER_INTERACTION_MODE'] = 'push_to_talk'
    env['BANTER_STT_MODE'] = 'record_then_transcribe'
    env['OPENAI_API_KEY'] = 'test-value'
    env['ELEVENLABS_API_KEY'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(server_dir)
    env['BANTER_BUILD_VERSION'] = 'test-revision'
    env.pop('BANTER_EVENT_LOG', None)

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import api.main; '
                'assert api.main.tts_client is None; '
                'assert api.main.run_metadata["build_revision"] == "test-revision"; '
                'assert api.main.run_metadata["interaction_mode"] == "push_to_talk"; '
                'assert api.main.run_metadata["orchestration_mode"] == "unified_v2"; '
                'assert api.main.run_metadata["utterance_model"] == "gpt-5.6-luna"; '
                'assert api.main.run_metadata["stt_model"] == "gpt-4o-mini-transcribe"; '
                'assert api.main.stt_mode == "record_then_transcribe"; '
                'assert api.main.streaming_stt_client is None; '
                'assert api.main.run_metadata["audio_uplink"] '
                '== "websocket_json_base64_webm"; '
                'assert api.main.run_metadata["audio_stop_semantics"] '
                '== "pointerdown_to_html_media_pause_applied"; '
                'assert api.main.run_metadata["audio_downlink"] == "disabled"; '
                'assert api.main.run_metadata["tts_model"] == "disabled"'
            ),
        ],
        cwd=server_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_main_can_select_streaming_stt_without_connecting():
    """스트리밍 모드는 import 시 외부 연결 없이 provider만 조립한다."""
    server_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHON_DOTENV_DISABLED'] = '1'
    env['BANTER_INTERACTION_MODE'] = 'push_to_talk'
    env['OPENAI_API_KEY'] = 'test-value'
    env['ELEVENLABS_API_KEY'] = ''
    env['BANTER_STT_MODE'] = 'streaming_push_to_talk'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(server_dir)
    env.pop('BANTER_EVENT_LOG', None)

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import api.main; '
                'assert api.main.streaming_stt_client is not None; '
                'assert api.main.run_metadata["baseline_id"] == "streaming-stt-v1"; '
                'assert api.main.run_metadata["stt_model"] == "gpt-live-transcribe"; '
                'assert api.main.file_stt_client is None; '
                'assert api.main.run_metadata["stt_file_model"] '
                '== "disabled"; '
                'assert api.main.run_metadata["audio_uplink"] '
                '== "websocket_json_base64_pcm16"; '
                'assert api.main.run_metadata["stt_stream_sample_rate_hz"] == 24000'
            ),
        ],
        cwd=server_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_main_metadata_does_not_copy_secret_values():
    """실행 정보에는 provider key가 들어가지 않는다."""
    server_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHON_DOTENV_DISABLED'] = '1'
    env['BANTER_INTERACTION_MODE'] = 'push_to_talk'
    env['BANTER_STT_MODE'] = 'record_then_transcribe'
    env['OPENAI_API_KEY'] = 'openai-secret-value'
    env['ELEVENLABS_API_KEY'] = 'elevenlabs-secret-value'
    env['BANTER_BUILD_VERSION'] = 'release-candidate'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(server_dir)
    env.pop('BANTER_EVENT_LOG', None)

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import json, api.main; '
                'data = json.dumps(api.main.run_metadata); '
                'assert "secret-value" not in data; '
                'assert api.main.run_metadata["audio_downlink"] '
                '== "websocket_base64_mp3"; '
                'assert api.main.run_metadata["tts_model"] == "eleven_v3"'
            ),
        ],
        cwd=server_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('interaction,stt_mode,valid', [
    ('vad', 'streaming_push_to_talk', True),
    ('vad', 'record_then_transcribe', False),
    ('unknown', 'streaming_push_to_talk', False),
])
def test_main_vad_selection_is_explicit_and_validated(interaction, stt_mode, valid):
    server_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({
        'OPENAI_API_KEY': 'test-value',
        'ELEVENLABS_API_KEY': '',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHON_DOTENV_DISABLED': '1',
        'PYTHONPATH': str(server_dir),
        'BANTER_INTERACTION_MODE': interaction,
        'BANTER_STT_MODE': stt_mode,
        'BANTER_BUILD_VERSION': 'test-vad',
    })
    env.pop('BANTER_EVENT_LOG', None)
    result = subprocess.run(
        [sys.executable, '-c', (
            'import api.main; '
            'assert api.main.interaction_mode == "vad"; '
            'assert api.main.run_metadata["baseline_id"] == "vad-v1"; '
            'assert api.main.streaming_stt_client is not None; '
            'assert api.main.file_stt_client is None; '
            'assert api.main.run_metadata["audio_stop_semantics"] '
            '== "vad_start_detected_to_html_media_pause_applied"'
        )],
        cwd=server_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if valid:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert ('BANTER_STT_MODE' if interaction == 'vad' else 'BANTER_INTERACTION_MODE') in result.stderr
