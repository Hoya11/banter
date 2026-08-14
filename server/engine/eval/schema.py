"""judge 입출력 계약 (D-005).

이 스키마와 rubric.yaml이 평가의 '인터페이스'다.
심판 구현(단일 LLM → 앙상블)을 바꿔도 이 계약은 불변으로 유지한다.
"""

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """대화 한 턴. judge 입력(transcript)의 원소.

    interrupted: 끼어들기로 잘린 발화(§2.1). judge가 '끊긴 뒤 수습'을
    평가하려면 이 사실을 알아야 한다 — 없으면 평범한 대화로 채점된다.
    """

    speaker: str  # user | ai_a | ai_b
    text: str
    interrupted: bool = False


class ItemScore(BaseModel):
    """루브릭 항목별 채점 결과 (점수 + 근거)."""

    score: int = Field(ge=0, le=10)
    reason: str


class JudgeResult(BaseModel):
    """judge 출력. scores의 key는 rubric.yaml의 item key와 일치한다."""

    scores: dict[str, ItemScore]
    overall_comment: str
