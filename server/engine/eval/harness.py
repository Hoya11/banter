"""Phase 1 end-to-end 파일럿: 대화 생성 → judge 채점 → go/no-go.

Phase 1의 핵심 루프. 이 파이프가 페르소나·프롬프트 튜닝의 나침반이다
(감이 아니라 judge 점수로 판단). 결과는 sink로 기록해 버전별 비교에 쓴다.
"""

from pathlib import Path

import yaml

from ..graph.graph import build_graph
from ..graph.state import initial_state
from .judge import judge
from .schema import JudgeResult, Turn
from .scoring import is_go
from .tracing import get_sink

SCENARIOS_PATH = Path(__file__).parent / 'scenarios.yaml'


def run_pilot(
    utterance_client,
    judge_client,
    turns: int = 6,
    rubric_version: str = 'v1',
) -> tuple[list[Turn], JudgeResult, bool]:
    """N턴 라디오 모드 대화를 만들고 judge로 채점해 (transcript, result, go)를 반환한다."""
    graph = build_graph(utterance_client)
    state = initial_state()
    for _ in range(turns):
        state = graph.invoke(state)

    transcript = [Turn(speaker=m['speaker'], text=m['text']) for m in state['messages']]
    result = judge(transcript, judge_client, rubric_version)
    get_sink().record_judge(
        transcript,
        result,
        {'source': 'pilot', 'turns': turns, 'rubric_version': rubric_version},
    )
    return transcript, result, is_go(result)


def load_scenarios() -> list[dict]:
    """scenarios.yaml의 평가 시나리오 셋을 로드한다."""
    data = yaml.safe_load(SCENARIOS_PATH.read_text(encoding='utf-8'))
    if not data or 'scenarios' not in data:
        raise ValueError('scenarios.yaml에 scenarios가 없다')
    return data['scenarios']


def _inject_user(state: dict, text: str) -> dict:
    """스크립트된 유저 발화를 상태에 주입한다 (AI 연속 카운터 리셋)."""
    msg = {'speaker': 'user', 'text': text, 'ts': 0.0, 'interrupted': False}
    return {
        **state,
        'messages': state['messages'] + [msg],
        'current_speaker': 'user',
        'consecutive_ai_turns': 0,
        'last_user_turn_ts': 0.0,
    }


def run_scenario(
    steps: list[dict],
    utterance_client,
    judge_client,
    rubric_version: str = 'v1',
) -> tuple[list[Turn], JudgeResult, bool]:
    """시나리오 step(ai/user)을 순서대로 실행하고 전체 대화를 채점한다."""
    graph = build_graph(utterance_client)
    state = initial_state()
    for step in steps:
        if step['type'] == 'user':
            state = _inject_user(state, step['text'])
        else:
            state = graph.invoke(state)

    transcript = [Turn(speaker=m['speaker'], text=m['text']) for m in state['messages']]
    result = judge(transcript, judge_client, rubric_version)
    get_sink().record_judge(
        transcript, result, {'source': 'scenario', 'rubric_version': rubric_version}
    )
    return transcript, result, is_go(result)


def run_scenario_set(utterance_client, judge_client) -> list[dict]:
    """시나리오 셋 전체를 채점해 시나리오별 결과를 반환한다."""
    out = []
    for scenario in load_scenarios():
        transcript, result, go = run_scenario(
            scenario['steps'], utterance_client, judge_client
        )
        out.append({'id': scenario['id'], 'result': result, 'go': go})
    return out
