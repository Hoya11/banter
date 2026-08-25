"""페르소나 설정 계약 (D-004 데이터 주도).

코드가 아닌 duo.yaml이 페르소나의 단일 소스. 이 스키마는 그 파일의 형태를 고정한다.
"""

from pydantic import BaseModel


class Persona(BaseModel):
    """대비형 듀오의 한 명."""

    key: str  # 'ai_a' | 'ai_b' — 상태의 current_speaker와 일치
    name: str
    voice_id: str | None  # TTS 보이스 (voice_presets에서 주입)
    voice_speed: float | None = None  # 말 빠르기 (provider 지원 시, 1.0=기본)
    stance: str  # 기본 스탠스 — 대비의 핵심(케미의 절반)
    speech_style: str
    topic_bias: list[str]
