"""화자 선정 v1 — supervisor (§1.3, D-003).

LLM이 대화 이력을 보고 다음 화자와 발화 의도를 판단한다. LLM이 없으면 기계적 교대로 폴백.
규칙 가드(직전 화자 연속 금지)는 전략과 무관하게 항상 적용 — supervisor가 어겨도 강제 교체.
"""

import json

from ..personas.loader import load_personas
from .state import ConvState


def _mechanical_next(prev: str | None, messages: list | None = None) -> str:
    """LLM 없이 쓰는 기계적 교대 (직전 화자의 반대 AI).

    유저 발화 직후엔 이력에서 마지막 AI 화자를 찾아 그 반대를 고른다
    — 끊긴 화자가 곧바로 또 말하는 것을 막는다.
    """
    if prev == 'ai_a':
        return 'ai_b'
    if prev == 'ai_b':
        return 'ai_a'
    for m in reversed(messages or []):  # 유저 턴을 건너뛰고 마지막 AI를 찾는다
        if m['speaker'] == 'ai_a':
            return 'ai_b'
        if m['speaker'] == 'ai_b':
            return 'ai_a'
    return 'ai_a'  # 첫 턴


def _sanitize_intent(intent) -> str | None:
    """supervisor가 만든 intent를 발화 프롬프트에 넣기 전에 정제한다.

    intent는 (유저 발화가 섞인 이력을 본) LLM의 자유 출력이라 그대로 삽입하면
    타입 오류·프롬프트 인젝션 통로가 된다. 문자열만 허용, 개행 제거, 길이 제한.
    """
    if not isinstance(intent, str) or not intent.strip():
        return None
    return intent.replace('\n', ' ').strip()[:100]


# 같은 화제가 이 턴 수 이상 이어지면 전환·복귀를 지시한다 (소재 고갈 → 단조 방지, §1.4)
TOPIC_STALE_TURNS = 4
TOPIC_STACK_MAX = 5  # 보류 화제 보관 상한 (오래된 것부터 버림)


def _update_topics(stack: list[str], turns: int, topic: str | None) -> tuple[list[str], int]:
    """supervisor가 판단한 현재 화제로 topic_stack과 연속 턴 카운터를 갱신한다.

    - 새 화제 → push (이전 화제는 보류로 남음, 전환 수용 — §1.4)
    - 보류 화제로 복귀 → 스택 맨 위로 끌어올림
    - 같은 화제 지속 → 카운터만 증가
    """
    if not topic:
        return stack, turns
    if stack and stack[-1] == topic:
        return stack, turns + 1
    rest = [t for t in stack if t != topic]
    return (rest + [topic])[-TOPIC_STACK_MAX:], 1


def build_supervisor_prompt(state: ConvState) -> tuple[str, str]:
    """이력·로스터·화제 상태로 supervisor 판단용 (system, user) 프롬프트를 만든다."""
    personas = load_personas()
    roster = '\n'.join(f'- {k}: {p.name}, {p.stance}' for k, p in personas.items())
    history = (
        '\n'.join(
            f'{m["speaker"]}: {m["text"]}' + (' [말하다 끊김]' if m.get('interrupted') else '')
            for m in state['messages']
        )
        or '(아직 발화 없음)'
    )
    system = (
        '너는 3자 수다(유저 1명 + AI 2명)의 진행 감독이다. 다음에 누가 어떤 의도로 말할지 정해라.\n'
        f'AI 로스터:\n{roster}\n'
        '직전 화자가 연속으로 말하지 않게 해라.\n'
        '의도는 다양하게 굴려라 — 예: 딴지/반박, 구체적 경험담 꺼내기, 상대 놀리기, 화제 살짝 틀기, '
        '짧은 리액션만, 되묻기, 유저 소환. 직전 두 턴과 같은 유형의 의도를 반복하지 마라 '
        '(특히 "동의+질문" 패턴의 연속 금지 — 수다가 단조로워진다).\n'
        '직전 발화가 [말하다 끊김]이고 유저가 끼어들었으면, 끊긴 화자가 자연스럽게 양보하고 '
        '유저 발화에 반응하도록 화자·의도를 정해라(매번 사과하지 말 것).\n'
        '출력은 JSON만: {"next_speaker": "ai_a"|"ai_b", "intent": "이번 발화의 짧은 의도(한국어)", '
        '"topic": "지금 나누는 화제(2~6단어)"}'
    )
    # 화제 컨텍스트 — 신선도 가드는 코드(카운터)가 판정하고 지시만 주입한다
    stack = state.get('topic_stack') or []
    lines = []
    if stack:
        lines.append(f'현재 화제: {stack[-1]}')
        if stack[:-1]:
            lines.append(f'보류 중인 화제: {", ".join(stack[:-1])}')
    if state.get('topic_turns', 0) >= TOPIC_STALE_TURNS:
        lines.append(
            '이 화제가 오래 이어졌다. 이번 의도는 화제 전환이어야 한다 — '
            '새로운 화제를 꺼내거나 보류 중인 화제 하나로 자연스럽게 넘어가라.'
        )
    topic_ctx = ('\n' + '\n'.join(lines)) if lines else ''
    user = f'대화 이력:\n{history}{topic_ctx}\n\n다음 화자와 의도를 정해라.'
    return system, user


def select_next(state: ConvState, client=None) -> dict:
    """다음 화자 + 의도를 결정한다.

    client가 없으면 기계적 교대. 있으면 supervisor LLM 판단.
    어느 경우든 규칙 가드(직전 화자 연속 금지)를 항상 적용한다.
    """
    prev = state['current_speaker']
    messages = state['messages']
    intent = None
    stack = list(state.get('topic_stack') or [])
    topic_turns = state.get('topic_turns', 0)

    if client is None:
        nxt = _mechanical_next(prev, messages)
    else:
        try:
            decision = json.loads(client.complete(*build_supervisor_prompt(state)))
            nxt = decision['next_speaker']
            intent = _sanitize_intent(decision.get('intent'))
            # 화제 갱신 — 판단은 LLM이, 신선도 판정은 코드(카운터)가 한다
            topic = _sanitize_intent(decision.get('topic'))
            stack, topic_turns = _update_topics(stack, topic_turns, topic and topic[:40])
        except Exception:  # LLM 호출 실패(429/500/네트워크)·파싱 실패 모두 기계적 교대로 폴백
            print('[supervisor] 판단 실패 — 기계적 교대로 폴백')
            nxt = _mechanical_next(prev, messages)
        # 규칙 가드: 유효하지 않거나 직전 화자와 같으면 강제 교체
        if nxt not in ('ai_a', 'ai_b') or (nxt == prev and prev in ('ai_a', 'ai_b')):
            nxt = _mechanical_next(prev, messages)

    # 유저·시작 직후면 AI 연속 카운터를 1로, AI가 이어가면 증가
    consecutive = 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1
    return {
        'current_speaker': nxt,
        'consecutive_ai_turns': consecutive,
        'current_intent': intent,
        'topic_stack': stack,
        'topic_turns': topic_turns,
    }


# ---------- v2: 통합 생성 (D-003) ----------
# 화자 선정과 발화 생성을 한 LLM 호출로 — 턴당 호출 -1 (지연·비용 절감).
# 출력 계약: 첫 줄 = JSON 헤더 {next_speaker, intent, topic}, 둘째 줄부터 = 그 화자의 대사.
# 규칙 가드·topic 갱신은 v1과 같은 코드를 재사용한다 (전략이 바뀌어도 가드는 불변, §1.3).


def build_unified_prompt(state: ConvState, finish: bool = False) -> tuple[str, str]:
    """v2 통합 생성용 (system, user) 프롬프트 — 두 페르소나·규칙·화제 컨텍스트 포함.

    finish=True면 세션 캡 도달 — 마무리 인사 지시를 얹는다.
    """
    personas = load_personas()
    roster = '\n'.join(
        f'- {k} "{p.name}": {p.stance} / 말투: {p.speech_style} / 관심사: {", ".join(p.topic_bias)}'
        for k, p in personas.items()
    )
    history = (
        '\n'.join(
            f'{m["speaker"]}: {m["text"]}' + (' [말하다 끊김]' if m.get('interrupted') else '')
            for m in state['messages']
        )
        or '(아직 발화 없음)'
    )
    # 다음 턴의 AI 연속 카운트를 예측해 소환 지시 여부를 결정 (§1.3)
    prev = state['current_speaker']
    next_consec = 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1
    system = (
        '너는 3자 수다(유저 1명 + AI 2명)의 다음 발화를 만든다. 누가 말할지 정하고, 그 사람으로서 말해라.\n'
        f'AI 페르소나:\n{roster}\n'
        '규칙:\n'
        '- 직전 화자가 연속으로 말하지 않는다.\n'
        '- 의도는 다양하게: 딴지/반박, 구체적 경험담, 상대 놀리기, 화제 살짝 틀기, 짧은 리액션, 되묻기, 유저 소환. '
        '직전 두 턴과 같은 유형 반복 금지(특히 동의+질문 패턴 연속 금지).\n'
        '- 이력에 나온 인사·리액션 반복 금지, 직전 발화에 새 내용을 얹어 진전.\n'
        '- 직전 발화가 [말하다 끊김]이면 끊긴 화자가 양보하고 유저 발화에 반응(매번 사과 금지).\n'
        '- 발화는 한국어 반말 1~2문장, 이모지 금지. 감정은 [sighs] [tired] [sarcastic] [laughs] [excited] [cheerfully] '
        '태그 0~2개로, 쉼은 <break time="0.4s" />.\n'
        + ('- 지금은 유저(사람)에게 가볍게 말을 걸어 대화에 끌어들여라.\n' if next_consec > 0 and next_consec % 3 == 0 else '')
        + ('- 지금은 세션 마무리 — 대화를 자연스럽게 끝내는 인사를 해라.\n' if finish else '')
        + '출력 형식(정확히 지켜라):\n'
        '첫 줄: {"next_speaker": "ai_a"|"ai_b", "intent": "짧은 의도", "topic": "화제(2~6단어)"}\n'
        '둘째 줄부터: 선택한 화자의 발화만 (화자 표시·따옴표 없이)'
    )
    stack = state.get('topic_stack') or []
    lines = []
    if stack:
        lines.append(f'현재 화제: {stack[-1]}')
        if stack[:-1]:
            lines.append(f'보류 중인 화제: {", ".join(stack[:-1])}')
    if state.get('topic_turns', 0) >= TOPIC_STALE_TURNS:
        lines.append('이 화제가 오래 이어졌다. 새 화제를 꺼내거나 보류 화제로 자연스럽게 넘어가라.')
    topic_ctx = ('\n' + '\n'.join(lines)) if lines else ''
    user = f'대화 이력:\n{history}{topic_ctx}\n\n다음 발화를 만들어라.'
    return system, user


def parse_unified_header(header_line: str, state: ConvState) -> dict | None:
    """v2 첫 줄(JSON 헤더)을 v1과 동일한 가드·화제 갱신을 거친 sel dict로 만든다.

    화자가 무효·연속이면 None — 호출부가 기계적 교대로 폴백한다.
    (v1과 달리 대사가 이미 그 화자 말투로 생성되므로 화자만 바꿔치기할 수 없다.)
    """
    prev = state['current_speaker']
    try:
        head = json.loads(header_line)
        nxt = head['next_speaker']
    except Exception:
        return None
    if nxt not in ('ai_a', 'ai_b') or (nxt == prev and prev in ('ai_a', 'ai_b')):
        return None  # 규칙 가드 위반 — 통합 출력은 정정 불가라 폴백
    intent = _sanitize_intent(head.get('intent'))
    topic = _sanitize_intent(head.get('topic'))
    stack, topic_turns = _update_topics(
        list(state.get('topic_stack') or []), state.get('topic_turns', 0), topic and topic[:40]
    )
    consecutive = 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1
    return {
        'current_speaker': nxt,
        'consecutive_ai_turns': consecutive,
        'current_intent': intent,
        'topic_stack': stack,
        'topic_turns': topic_turns,
    }


def split_unified(raw: str, state: ConvState) -> tuple[dict, str]:
    """v2 원문을 (sel, 대사 원문)으로 분리한다. 헤더 불량이면 기계적 교대로 폴백.

    폴백 시 첫 줄이 JSON 흉내({로 시작)면 버리고, 아니면 전체를 대사로 살린다.
    """
    first, _, rest = raw.partition('\n')
    sel = parse_unified_header(first.strip(), state)
    if sel is not None:
        return sel, rest
    print('[unified] 헤더 파싱 실패 — 기계적 교대 폴백')
    prev = state['current_speaker']
    nxt = _mechanical_next(prev, state['messages'])
    consecutive = 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1
    fallback = {
        'current_speaker': nxt,
        'consecutive_ai_turns': consecutive,
        'current_intent': None,
        'topic_stack': list(state.get('topic_stack') or []),
        'topic_turns': state.get('topic_turns', 0),
    }
    return fallback, (rest if first.lstrip().startswith('{') else raw)
