"""화자 선정(supervisor §1.3)의 결정적 계약 테스트.

supervisor LLM이 없으면 기계적 교대로 폴백하고, 있어도 규칙 가드(직전 화자 연속 금지)가
LLM 결정을 이긴다. AI 둘이 번갈아 말하는 규칙이 §1.4 루프 방지의 뼈대다.
"""

from engine.graph.graph import build_graph
from engine.graph.supervisor import (
    TOPIC_STALE_TURNS,
    _update_topics,
    build_supervisor_prompt,
    select_next,
)


def _state(current, consec=0):
    return {
        'messages': [],
        'topic_stack': [],
        'current_speaker': current,
        'consecutive_ai_turns': consec,
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


def test_fallback_first_turn_starts_ai_a():
    out = select_next(_state(None))  # client 없음 → 기계적
    assert out['current_speaker'] == 'ai_a'
    assert out['consecutive_ai_turns'] == 1


def test_fallback_alternates():
    assert select_next(_state('ai_a', 1))['current_speaker'] == 'ai_b'
    assert select_next(_state('ai_b', 2))['current_speaker'] == 'ai_a'


def test_consecutive_resets_after_user():
    assert select_next(_state('user', 5))['consecutive_ai_turns'] == 1


def test_supervisor_guard_prevents_consecutive():
    # supervisor가 직전 화자(ai_a)를 또 지목해도 규칙 가드가 ai_b로 교체
    class BadSupervisor:
        def complete(self, system: str, user: str) -> str:
            return '{"next_speaker": "ai_a", "intent": "test"}'

    out = select_next(_state('ai_a', 1), BadSupervisor())
    assert out['current_speaker'] == 'ai_b'


def test_supervisor_intent_is_passed_through():
    class Sup:
        def complete(self, system: str, user: str) -> str:
            return '{"next_speaker": "ai_b", "intent": "반박"}'

    out = select_next(_state('ai_a', 1), Sup())
    assert out['current_speaker'] == 'ai_b'
    assert out['current_intent'] == '반박'


def test_supervisor_sanitizes_bad_intent():
    # intent가 문자열이 아니거나(주입 통로) 너무 길면 정제된다
    class DictIntent:
        def complete(self, system: str, user: str) -> str:
            return '{"next_speaker": "ai_b", "intent": {"x": 1}}'

    assert select_next(_state('ai_a', 1), DictIntent())['current_intent'] is None

    class LongIntent:
        def complete(self, system: str, user: str) -> str:
            return '{"next_speaker": "ai_b", "intent": "' + '가' * 300 + '"}'

    out = select_next(_state('ai_a', 1), LongIntent())
    assert len(out['current_intent']) <= 100


def test_supervisor_bad_json_falls_back():
    class Broken:
        def complete(self, system: str, user: str) -> str:
            return '이건 JSON이 아님'

    out = select_next(_state('ai_a', 1), Broken())
    assert out['current_speaker'] == 'ai_b'  # 파싱 실패 → 기계적 교대


def test_supervisor_api_error_falls_back():
    # LLM 호출 자체가 던지는 예외(429/500/네트워크)도 세션을 죽이지 않고 폴백
    class Boom:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError('api 500')

    out = select_next(_state('ai_a', 1), Boom())
    assert out['current_speaker'] == 'ai_b'


def test_fallback_after_user_avoids_last_ai_speaker():
    # 유저 발화 직후엔 이력의 마지막 AI(끊긴 화자)의 반대를 고른다 — ai_a 고정 아님
    state = _state('user', 0)
    state['messages'] = [
        {'speaker': 'ai_a', 'text': 'x', 'ts': 0.0, 'interrupted': True},
        {'speaker': 'user', 'text': 'y', 'ts': 0.0, 'interrupted': False},
    ]
    assert select_next(state)['current_speaker'] == 'ai_b'


def test_update_topics_rules():
    # 새 화제 push (이전 화제는 보류) / 같은 화제는 카운터만 / 보류 복귀는 맨 위로 / 상한 유지
    stack, turns = _update_topics([], 0, '퇴근 후 저녁')
    assert (stack, turns) == (['퇴근 후 저녁'], 1)
    stack, turns = _update_topics(stack, turns, '퇴근 후 저녁')
    assert (stack, turns) == (['퇴근 후 저녁'], 2)
    stack, turns = _update_topics(stack, turns, '주말 계획')
    assert (stack, turns) == (['퇴근 후 저녁', '주말 계획'], 1)
    stack, turns = _update_topics(stack, turns, '퇴근 후 저녁')  # 보류 화제 복귀
    assert (stack, turns) == (['주말 계획', '퇴근 후 저녁'], 1)
    assert _update_topics(stack, 3, None) == (stack, 3)  # 판단 없으면 유지
    many = [f't{i}' for i in range(9)]
    trimmed, _ = _update_topics(many, 1, '새화제')
    assert len(trimmed) <= 5 and trimmed[-1] == '새화제'  # 상한


def test_supervisor_updates_topic_state():
    class Sup:
        def complete(self, system: str, user: str) -> str:
            return '{"next_speaker": "ai_b", "intent": "반박", "topic": "야근 문화"}'

    out = select_next(_state('ai_a', 1), Sup())
    assert out['topic_stack'] == ['야근 문화']
    assert out['topic_turns'] == 1


def test_stale_topic_injects_switch_directive():
    # 신선도 가드는 코드(카운터)가 판정한다 — 임계 이상일 때만 전환 지시가 조립된다
    state = _state('ai_a', 1)
    state['topic_stack'] = ['퇴근 후 저녁']
    state['topic_turns'] = TOPIC_STALE_TURNS
    _, user_stale = build_supervisor_prompt(state)
    assert '화제 전환' in user_stale
    state['topic_turns'] = 1
    _, user_fresh = build_supervisor_prompt(state)
    assert '화제 전환' not in user_fresh


def test_graph_one_pass():
    g = build_graph()
    result = g.invoke(_state(None))
    assert len(result['messages']) == 1
    assert result['messages'][0]['speaker'] == 'ai_a'
    assert result['personas']['ai_a']['speak_count'] == 1


def test_generate_uses_injected_client():
    class FakeUtteranceClient:
        def complete(self, system: str, user: str) -> str:
            return '오늘 진짜 피곤하다'

    g = build_graph(FakeUtteranceClient())
    result = g.invoke(_state(None))
    assert result['messages'][0]['text'] == '오늘 진짜 피곤하다'


def test_generate_strips_speaker_prefix():
    class PrefixClient:
        def complete(self, system: str, user: str) -> str:
            return '도현: 오늘 피곤하다'

    g = build_graph(PrefixClient())
    result = g.invoke(_state(None))
    assert result['messages'][0]['text'] == '오늘 피곤하다'
