"""3자 대화 노드 그래프 (엔진설계 §1.2, 화자 선정 v1 기준).

Phase 1 골격: 발화 생성은 스텁(LLM은 C단계), 화자 선정은 규칙 가드(§1.3) 중심.
노드는 상태의 '바뀐 부분'만 dict로 반환하고, LangGraph가 병합한다
(messages는 operator.add reducer로 누적).
"""

from langgraph.graph import END, START, StateGraph

from .state import ConvState


def select_speaker(state: ConvState) -> dict:
    """다음 화자를 정한다 — 규칙 가드(§1.3)를 항상 적용.

    Phase 1 골격은 AI 턴 진행(라디오 모드) 기준으로 ai_a/ai_b 중 다음 화자를 고른다.
    """
    prev = state['current_speaker']
    if prev == 'ai_a':
        nxt = 'ai_b'
    elif prev == 'ai_b':
        nxt = 'ai_a'
    else:  # 첫 턴(None) 또는 유저 발화 직후
        nxt = 'ai_a'
    # 유저·시작 직후면 AI 연속 카운터를 1로, AI가 이어가면 증가시킨다
    consecutive = 1 if prev in (None, 'user') else state['consecutive_ai_turns'] + 1
    return {'current_speaker': nxt, 'consecutive_ai_turns': consecutive}


def generate_utterance(state: ConvState) -> dict:
    """현재 화자의 발화를 생성한다 (Phase 1 스텁 — 실제 LLM 발화는 C단계)."""
    speaker = state['current_speaker']
    msg = {'speaker': speaker, 'text': f'({speaker} 발화 스텁)', 'ts': 0.0, 'interrupted': False}
    return {'messages': [msg]}


def update_state(state: ConvState) -> dict:
    """발화 후 페르소나 발화 카운트를 갱신한다."""
    speaker = state['current_speaker']
    personas = {k: dict(v) for k, v in state['personas'].items()}
    if speaker in personas:
        personas[speaker]['speak_count'] += 1
    return {'personas': personas}


def build_graph():
    """노드 그래프를 조립해 compile한다."""
    g = StateGraph(ConvState)
    g.add_node('select_speaker', select_speaker)
    g.add_node('generate', generate_utterance)
    g.add_node('update', update_state)
    g.add_edge(START, 'select_speaker')
    g.add_edge('select_speaker', 'generate')
    g.add_edge('generate', 'update')
    g.add_edge('update', END)
    return g.compile()
