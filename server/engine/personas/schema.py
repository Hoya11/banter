"""페르소나 설정 계약 (D-004 데이터 주도).

코드가 아닌 duo.yaml이 페르소나의 단일 소스. 이 스키마는 그 파일의 형태를 고정한다.
"""

from pydantic import BaseModel


class Persona(BaseModel):
    """대비형 듀오의 한 명."""

    key: str  # 'ai_a' | 'ai_b' — 상태의 current_speaker와 일치
    name: str
    voice_id: str | None  # Phase 2 TTS 보이스 (지금은 None)
    stance: str  # 기본 스탠스 — 대비의 핵심(케미의 절반)
    speech_style: str
    topic_bias: list[str]
