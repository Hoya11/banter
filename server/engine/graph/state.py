"""3자 대화 오케스트레이터의 LangGraph 상태 (엔진설계 §1.1).

노드들이 이 상태를 읽고 갱신한다. current_speaker·consecutive_ai_turns 같은 필드가
supervisor의 발화권 중재(§1.3)와 실패모드 가드(§1.4)의 판단 근거가 된다.
"""

import operator
from typing import Annotated, Literal, TypedDict

# 발화 주체 — 유저 1명 + AI 2명(대비형 듀오)
Speaker = Literal['user', 'ai_a', 'ai_b']


class Message(TypedDict):
    """대화 한 발화. interrupted는 끼어들기로 잘린 발화 표시(§2.1)."""

    speaker: Speaker
    text: str
    ts: float
    interrupted: bool


class PersonaState(TypedDict):
    """페르소나별 누적 상태 — 발화 균형·입장 일관성 추적."""

    speak_count: int
    last_stance: str


class ConvState(TypedDict):
    """오케스트레이터 전체 상태."""

    # conversation
    messages: Annotated[list[Message], operator.add]  # 노드 반환분을 누적
    topic_stack: list[str]  # 현재 화제(맨 위) + 보류 화제 — supervisor가 판단, 코드가 갱신
    topic_turns: int  # 현재 화제가 이어진 턴 수 — 신선도 가드(§1.4 화제 집착 방지)
    # turn
    current_speaker: Speaker | None
    consecutive_ai_turns: int  # AI 연속 발화 카운터 (유저 소외 방지)
    last_user_turn_ts: float | None
    current_intent: str | None  # supervisor가 정한 이번 발화 의도 (§1.3)
    # personas — key: 'ai_a' | 'ai_b'
    personas: dict[str, PersonaState]
    # control
    pending_user_input: str | None  # 생성 중 도착한 유저 개입 버퍼
    session_elapsed: float
    budget_used: float


def initial_state() -> ConvState:
    """빈 대화의 초기 상태 (도현/소은 듀오)."""
    return {
        'messages': [],
        'topic_stack': [],
        'topic_turns': 0,
        'current_speaker': None,
        'consecutive_ai_turns': 0,
        'last_user_turn_ts': None,
        'current_intent': None,
        'personas': {
            'ai_a': {'speak_count': 0, 'last_stance': ''},
            'ai_b': {'speak_count': 0, 'last_stance': ''},
        },
        'pending_user_input': None,
        'session_elapsed': 0.0,
        'budget_used': 0.0,
    }
