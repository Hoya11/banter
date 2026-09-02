"""실제 서버 진입점의 선택형 provider 조립을 검증한다."""

import os
import subprocess
import sys
from pathlib import Path


def test_main_starts_without_elevenlabs_key():
    """ElevenLabs 키가 비어 있으면 외부 호출 없이 텍스트 모드로 조립된다."""
    server_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
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


def test_main_metadata_does_not_copy_secret_values():
    """실행 정보에는 provider key가 들어가지 않는다."""
    server_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
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
