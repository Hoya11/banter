"""화자 선정(supervisor §1.3)의 결정적 계약 테스트.

supervisor LLM이 없으면 기계적 교대로 폴백하고, 있어도 규칙 가드(직전 화자 연속 금지)가
LLM 결정을 이긴다. AI 둘이 번갈아 말하는 규칙이 §1.4 루프 방지의 뼈대다.
"""

from engine.graph.graph import build_graph
from engine.graph.supervisor import select_next


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


def test_supervisor_bad_json_falls_back():
    class Broken:
        def complete(self, system: str, user: str) -> str:
            return '이건 JSON이 아님'

    out = select_next(_state('ai_a', 1), Broken())
    assert out['current_speaker'] == 'ai_b'  # 파싱 실패 → 기계적 교대


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
