"""화자 선정 v1 — supervisor (§1.3, D-003).

LLM이 대화 이력을 보고 다음 화자와 발화 의도를 판단한다. LLM이 없으면 기계적 교대로 폴백.
규칙 가드(직전 화자 연속 금지)는 전략과 무관하게 항상 적용 — supervisor가 어겨도 강제 교체.
"""

import json

from ..personas.loader import load_personas
from .state import ConvState


def _mechanical_next(prev: str | None) -> str:
    """LLM 없이 쓰는 기계적 교대 (직전 화자의 반대 AI)."""
    if prev == 'ai_a':
        return 'ai_b'
    if prev == 'ai_b':
        return 'ai_a'
    return 'ai_a'  # 첫 턴 또는 유저 직후


def build_supervisor_prompt(state: ConvState) -> tuple[str, str]:
    """이력·로스터로 supervisor 판단용 (system, user) 프롬프트를 만든다."""
    personas = load_personas()
    roster = '\n'.join(f'- {k}: {p.name}, {p.stance}' for k, p in personas.items())
    history = (
        '\n'.join(f'{m["speaker"]}: {m["text"]}' for m in state['messages'])
        or '(아직 발화 없음)'
    )
    system = (
        '너는 3자 수다(유저 1명 + AI 2명)의 진행 감독이다. 다음에 누가 어떤 의도로 말할지 정해라.\n'
        f'AI 로스터:\n{roster}\n'
        '직전 화자가 연속으로 말하지 않게 하고, 대화가 동의만 반복되면 반박·화제 전환 의도를 넣어라.\n'
        '출력은 JSON만: {"next_speaker": "ai_a"|"ai_b", "intent": "이번 발화의 짧은 의도(한국어)"}'
    )
    user = f'대화 이력:\n{history}\n\n다음 화자와 의도를 정해라.'
    return system, user


def select_next(state: ConvState, client=None) -> dict:
    """다음 화자 + 의도를 결정한다.

    client가 없으면 기계적 교대. 있으면 supervisor LLM 판단.
    어느 경우든 규칙 가드(직전 화자 연속 금지)를 항상 적용한다.
    """
    prev = state['current_speaker']
    intent = None

    if client is None:
        nxt = _mechanical_next(prev)
    else:
        try:
            decision = json.loads(client.complete(*build_supervisor_prompt(state)))
            nxt = decision['next_speaker']
            intent = decision.get('intent')
        except (json.JSONDecodeError, KeyError, TypeError):
            nxt = _mechanical_next(prev)
        # 규칙 가드: 유효하지 않거나 직전 화자와 같으면 강제 교체
        if nxt not in ('ai_a', 'ai_b') or (nxt == prev and prev in ('ai_a', 'ai_b')):
            nxt = _mechanical_next(prev)

    # 유저·시작 직후면 AI 연속 카운터를 1로, AI가 이어가면 증가
    consecutive = 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1
    return {'current_speaker': nxt, 'consecutive_ai_turns': consecutive, 'current_intent': intent}
