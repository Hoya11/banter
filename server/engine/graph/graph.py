"""3자 대화 노드 그래프 (엔진설계 §1.2).

화자 선정은 supervisor(§1.3)에 위임한다 — LLM이 있으면 맥락 기반 판단,
없으면 기계적 교대로 폴백. 발화 생성은 페르소나 프롬프트 + supervisor 의도로.
"""

from langgraph.graph import END, START, StateGraph

from ..personas.loader import build_persona_prompt, get_persona
from .state import ConvState
from .supervisor import select_next

# AI가 이 턴 수만큼 이어갈 때마다 유저를 대화로 소환한다 (§1.3, 초안 튜닝 변수)
SUMMON_THRESHOLD = 3


def generate_utterance(state: ConvState, client=None) -> dict:
    """현재 화자의 발화를 생성한다.

    client가 없으면 스텁, 있으면 페르소나 프롬프트 + supervisor 의도 + 이력으로 LLM 발화.
    """
    speaker = state['current_speaker']
    if client is None:
        text = f'({speaker} 발화 스텁)'
    else:
        persona = get_persona(speaker)
        others = [k for k in ('ai_a', 'ai_b') if k != speaker]
        other_name = get_persona(others[0]).name if others else ''
        consec = state['consecutive_ai_turns']
        summon = consec > 0 and consec % SUMMON_THRESHOLD == 0
        system = build_persona_prompt(
            persona, other_name, summon_user=summon, intent=state.get('current_intent')
        )
        text = client.complete(system, _render_history(state['messages']))
        text = _strip_speaker_prefix(text, persona.name)
    msg = {'speaker': speaker, 'text': text, 'ts': 0.0, 'interrupted': False}
    return {'messages': [msg]}


def update_state(state: ConvState) -> dict:
    """발화 후 페르소나 발화 카운트를 갱신한다."""
    speaker = state['current_speaker']
    personas = {k: dict(v) for k, v in state['personas'].items()}
    if speaker in personas:
        personas[speaker]['speak_count'] += 1
    return {'personas': personas}


def _speaker_label(speaker: str) -> str:
    return '유저' if speaker == 'user' else get_persona(speaker).name


def _render_history(messages: list) -> str:
    """대화 이력을 '이름: 발화' 형태의 user 프롬프트로 렌더한다."""
    if not messages:
        return '대화를 시작해줘. 가볍게 인사하거나 오늘 있었던 얘기로 자연스럽게 시작해.'
    return '\n'.join(f'{_speaker_label(m["speaker"])}: {m["text"]}' for m in messages)


def _strip_speaker_prefix(text: str, name: str) -> str:
    """LLM이 발화 앞에 붙인 '이름:' 화자 표시를 방어적으로 제거한다."""
    stripped = text.lstrip()
    for prefix in (f'{name}:', f'{name} :'):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return text


def build_graph(utterance_client=None, supervisor_client=None):
    """노드 그래프를 조립해 compile한다.

    supervisor_client가 있으면 맥락 기반 화자 선정, 없으면 기계적 교대.
    utterance_client가 있으면 LLM 발화, 없으면 스텁.
    """

    def select(state: ConvState) -> dict:
        return select_next(state, supervisor_client)

    def generate(state: ConvState) -> dict:
        return generate_utterance(state, utterance_client)

    g = StateGraph(ConvState)
    g.add_node('select_speaker', select)
    g.add_node('generate', generate)
    g.add_node('update', update_state)
    g.add_edge(START, 'select_speaker')
    g.add_edge('select_speaker', 'generate')
    g.add_edge('generate', 'update')
    g.add_edge('update', END)
    return g.compile()
