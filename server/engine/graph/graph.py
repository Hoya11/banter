"""3자 대화 노드 그래프 (엔진설계 §1.2).

화자 선정은 supervisor(§1.3)에 위임한다 — LLM이 있으면 맥락 기반 판단,
없으면 기계적 교대로 폴백. 발화 생성은 페르소나 프롬프트 + supervisor 의도로.
"""

import re

from langgraph.graph import END, START, StateGraph

from ..personas.loader import build_persona_prompt, get_persona
from .state import ConvState
from .supervisor import select_next

# AI가 이 턴 수만큼 이어갈 때마다 유저를 대화로 소환한다 (§1.3, 초안 튜닝 변수)
SUMMON_THRESHOLD = 3


def prepare_utterance(state: ConvState) -> tuple[str, str, str]:
    """현재 화자의 (system, user) 프롬프트와 speaker를 준비한다 (LLM 호출 직전까지).

    발화를 non-stream(generate_utterance)으로 하든 stream(API)으로 하든 이 준비는 공통이다.
    """
    speaker = state['current_speaker']
    persona = get_persona(speaker)
    others = [k for k in ('ai_a', 'ai_b') if k != speaker]
    other_name = get_persona(others[0]).name if others else ''
    consec = state['consecutive_ai_turns']
    summon = consec > 0 and consec % SUMMON_THRESHOLD == 0
    system = build_persona_prompt(
        persona, other_name, summon_user=summon, intent=state.get('current_intent')
    )
    return system, _render_history(state['messages']), speaker


def generate_utterance(state: ConvState, client=None) -> dict:
    """현재 화자의 발화를 생성한다 (non-stream).

    client가 없으면 스텁, 있으면 prepare_utterance 프롬프트로 LLM 발화.
    """
    speaker = state['current_speaker']
    if client is None:
        text = f'({speaker} 발화 스텁)'
    else:
        system, user, _ = prepare_utterance(state)
        text = finalize_utterance(client.complete(system, user), get_persona(speaker).name)
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
    """대화 이력을 '이름: 발화' 형태의 user 프롬프트로 렌더한다 (끊긴 발화는 표시)."""
    if not messages:
        return '대화를 시작해줘. 가볍게 인사하거나 오늘 있었던 얘기로 자연스럽게 시작해.'
    lines = []
    for m in messages:
        text = m['text'] + (' [말하다 끊김]' if m.get('interrupted') else '')
        lines.append(f'{_speaker_label(m["speaker"])}: {text}')
    return '\n'.join(lines)


def _strip_speaker_prefix(text: str, name: str) -> str:
    """LLM이 발화 앞에 붙인 '이름:' 화자 표시를 방어적으로 제거한다."""
    stripped = text.lstrip()
    for prefix in (f'{name}:', f'{name} :'):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return text


# 발화 문장 수 상한 (§1.4 연설화 방지 — 프롬프트+후처리 이중 제어의 후처리 쪽)
MAX_SENTENCES = 2
_SENTENCE_END = ('.', '!', '?', '…')


def limit_sentences(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    """발화를 문장 수 상한으로 자른다. 프롬프트가 어겨도 코드가 보장한다.

    종결부호(.!?…) 기준으로 자르되, 부호 없는 짧은 반말체(예: '그치')는 그대로 둔다.
    """
    ends = []  # 문장 끝 인덱스 목록
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END:
            # 소수점(숫자.숫자)은 문장 끝이 아니다 (예: 3.5시간)
            if ch == '.' and text[i - 1 : i].isdigit() and text[i + 1 : i + 2].isdigit():
                continue
            # 연속 부호('?!', '...')는 한 문장의 끝으로 묶는다
            if i + 1 < len(text) and text[i + 1] in _SENTENCE_END:
                continue
            ends.append(i)
    if len(ends) <= max_sentences:
        return text.strip()
    # 상한 초과 — 단, 바로 다음 문장이 질문(?)이면 한 문장 더 허용한다.
    # 유저 소환 발화가 "리액션. 리액션. 너는 어땠어?" 꼴일 때 질문이 잘리면
    # 소외 방지 장치(§1.3)가 상한(§1.4)에 무력화되기 때문.
    cut = ends[max_sentences - 1]
    nxt = ends[max_sentences]
    if text[nxt] == '?':
        cut = nxt
    return text[: cut + 1].strip()


_AUDIO_TAG_RE = re.compile(r'\[[a-z]+(?: [a-z]+)?\]\s*|<break\s+[^>]*/>\s*')


def strip_audio_tags(text: str) -> str:
    """오디오 태그([sighs] 등)와 <break/>를 제거한다.

    태그는 TTS(v3) 연기 지시용 — 화면·대화 이력에는 깨끗한 텍스트만 남긴다.
    """
    return _AUDIO_TAG_RE.sub('', text).strip()


def finalize_utterance(raw: str, name: str) -> str:
    """LLM 발화 원문에 공통 후처리(프리픽스 제거 → 문장 상한 → 태그 제거)를 적용한다.

    TTS용 태그 포함본이 필요하면 finalize_tagged를 쓴다.
    """
    return strip_audio_tags(finalize_tagged(raw, name))


def finalize_tagged(raw: str, name: str) -> str:
    """프리픽스 제거 + 문장 상한까지만 — 오디오 태그를 보존한 TTS용 최종본."""
    return limit_sentences(_strip_speaker_prefix(raw, name))


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
