"""화자 선정 규칙 가드(§1.3)의 결정적 계약 테스트.

AI 둘이 번갈아 말하는 규칙(직전 화자 연속 금지)이 §1.4 루프 방지의 뼈대다.
"""

from engine.graph.graph import build_graph, select_speaker


def _state(current, consec=0):
    return {
        'messages': [],
        'topic_stack': [],
        'current_speaker': current,
        'consecutive_ai_turns': consec,
        'last_user_turn_ts': None,
        'personas': {
            'ai_a': {'speak_count': 0, 'last_stance': ''},
            'ai_b': {'speak_count': 0, 'last_stance': ''},
        },
        'pending_user_input': None,
        'session_elapsed': 0.0,
        'budget_used': 0.0,
    }


def test_first_turn_starts_ai_a():
    out = select_speaker(_state(None))
    assert out['current_speaker'] == 'ai_a'
    assert out['consecutive_ai_turns'] == 1


def test_alternates_ai_a_to_ai_b():
    assert select_speaker(_state('ai_a', 1))['current_speaker'] == 'ai_b'


def test_alternates_ai_b_to_ai_a():
    assert select_speaker(_state('ai_b', 2))['current_speaker'] == 'ai_a'


def test_consecutive_increments_between_ai():
    assert select_speaker(_state('ai_a', 3))['consecutive_ai_turns'] == 4


def test_consecutive_resets_after_user():
    # 유저 발화 직후 AI 턴이면 연속 카운터가 1로 리셋
    assert select_speaker(_state('user', 5))['consecutive_ai_turns'] == 1


def test_graph_one_pass():
    # 한 바퀴 돌면 메시지 1개(ai_a 스텁) + 발화 카운트 갱신
    g = build_graph()
    result = g.invoke(_state(None))
    assert len(result['messages']) == 1
    assert result['messages'][0]['speaker'] == 'ai_a'
    assert result['personas']['ai_a']['speak_count'] == 1
