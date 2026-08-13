"""ElevenLabs 클라이언트 계약 테스트 (실 API 호출은 하지 않음).

키 부재 방어와 synthesize 인터페이스(tts_client 계약) 충족만 확인한다.
음질·실호출은 로컬/스테이징에서 청음으로 검증한다.
"""

import pytest

from engine.eval.elevenlabs import ElevenLabsClient


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv('ELEVENLABS_API_KEY', raising=False)
    with pytest.raises(RuntimeError):
        ElevenLabsClient()


def test_satisfies_tts_interface():
    # tts_client 계약: synthesize 메서드 존재
    assert hasattr(ElevenLabsClient, 'synthesize')
