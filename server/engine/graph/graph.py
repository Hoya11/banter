"""3자 대화 노드 그래프 (엔진설계 §1.2, 화자 선정 v1 기준).

Phase 1 골격: 발화 생성은 스텁(LLM은 C단계), 화자 선정은 규칙 가드(§1.3) 중심.
노드는 상태의 '바뀐 부분'만 dict로 반환하고, LangGraph가 병합한다
(messages는 operator.add reducer로 누적).
"""

from langgraph.graph import END, START, StateGraph

from ..personas.loader import build_persona_prompt, get_persona
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


def generate_utterance(state: ConvState, client=None) -> dict:
    """현재 화자의 발화를 생성한다.

    client가 없으면 스텁(배관 테스트용), 있으면 페르소나 프롬프트 + 대화 이력으로 LLM 발화.
    """
    speaker = state['current_speaker']
    if client is None:
        text = f'({speaker} 발화 스텁)'
    else:
        persona = get_persona(speaker)
        others = [k for k in ('ai_a', 'ai_b') if k != speaker]
        other_name = get_persona(others[0]).name if others else ''
        system = build_persona_prompt(persona, other_name)
        text = client.complete(system, _render_history(state['messages']))
        text = _strip_speaker_prefix(text, persona.name)
    msg = {'speaker': speaker, 'text': text, 'ts': 0.0, 'interrupted': False}
    return {'messages': [msg]}


def _strip_speaker_prefix(text: str, name: str) -> str:
    """LLM이 발화 앞에 붙인 '이름:' 화자 표시를 방어적으로 제거한다."""
    stripped = text.lstrip()
    for prefix in (f'{name}:', f'{name} :'):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return text


def _speaker_label(speaker: str) -> str:
    return '유저' if speaker == 'user' else get_persona(speaker).name


def _render_history(messages: list) -> str:
    """대화 이력을 '이름: 발화' 형태의 user 프롬프트로 렌더한다."""
    if not messages:
        return '대화를 시작해줘. 가볍게 인사하거나 오늘 있었던 얘기로 자연스럽게 시작해.'
    return '\n'.join(f'{_speaker_label(m["speaker"])}: {m["text"]}' for m in messages)


def update_state(state: ConvState) -> dict:
    """발화 후 페르소나 발화 카운트를 갱신한다."""
    speaker = state['current_speaker']
    personas = {k: dict(v) for k, v in state['personas'].items()}
    if speaker in personas:
        personas[speaker]['speak_count'] += 1
    return {'personas': personas}


def build_graph(utterance_client=None):
    """노드 그래프를 조립해 compile한다.

    utterance_client가 없으면 발화는 스텁, 있으면 LLM으로 생성한다.
    """

    def generate(state: ConvState) -> dict:
        return generate_utterance(state, utterance_client)

    g = StateGraph(ConvState)
    g.add_node('select_speaker', select_speaker)
    g.add_node('generate', generate)
    g.add_node('update', update_state)
    g.add_edge(START, 'select_speaker')
    g.add_edge('select_speaker', 'generate')
    g.add_edge('generate', 'update')
    g.add_edge('update', END)
    return g.compile()
