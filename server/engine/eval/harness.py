"""Phase 1 end-to-end 파일럿: 대화 생성 → judge 채점 → go/no-go.

Phase 1의 핵심 루프. 이 파이프가 페르소나·프롬프트 튜닝의 나침반이다
(감이 아니라 judge 점수로 판단). 결과는 sink로 기록해 버전별 비교에 쓴다.
"""

import statistics
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
    supervisor_client=None,
) -> tuple[list[Turn], JudgeResult, bool]:
    """N턴 라디오 모드 대화를 만들고 judge로 채점해 (transcript, result, go)를 반환한다."""
    graph = build_graph(utterance_client, supervisor_client)
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


def _interrupt(state: dict, text: str) -> dict:
    """AI 발화 중 유저 개입 — 직전 AI 발화를 끊김 표시하고 유저 발화를 주입한다(§2.1)."""
    messages = [dict(m) for m in state['messages']]
    if messages and messages[-1]['speaker'] in ('ai_a', 'ai_b'):
        messages[-1]['interrupted'] = True
    messages.append({'speaker': 'user', 'text': text, 'ts': 0.0, 'interrupted': False})
    return {
        **state,
        'messages': messages,
        'current_speaker': 'user',
        'consecutive_ai_turns': 0,
        'last_user_turn_ts': 0.0,
    }


def run_scenario(
    steps: list[dict],
    utterance_client,
    judge_client,
    rubric_version: str = 'v1',
    supervisor_client=None,
) -> tuple[list[Turn], JudgeResult, bool]:
    """시나리오 step(ai/user)을 순서대로 실행하고 전체 대화를 채점한다."""
    graph = build_graph(utterance_client, supervisor_client)
    state = initial_state()
    for step in steps:
        if step['type'] == 'user':
            state = _inject_user(state, step['text'])
        elif step['type'] == 'interrupt':
            state = _interrupt(state, step['text'])
        else:
            state = graph.invoke(state)

    transcript = [Turn(speaker=m['speaker'], text=m['text']) for m in state['messages']]
    result = judge(transcript, judge_client, rubric_version)
    get_sink().record_judge(
        transcript, result, {'source': 'scenario', 'rubric_version': rubric_version}
    )
    return transcript, result, is_go(result)


def run_scenario_set(utterance_client, judge_client, supervisor_client=None) -> list[dict]:
    """시나리오 셋 전체를 채점해 시나리오별 결과를 반환한다."""
    out = []
    for scenario in load_scenarios():
        transcript, result, go = run_scenario(
            scenario['steps'],
            utterance_client,
            judge_client,
            supervisor_client=supervisor_client,
        )
        out.append({'id': scenario['id'], 'result': result, 'go': go})
    return out


def score_stats(results: list[JudgeResult]) -> dict:
    """여러 JudgeResult의 항목별 평균·표준편차(모집단)를 낸다."""
    if not results:
        return {}
    stats = {}
    for key in results[0].scores:
        vals = [r.scores[key].score for r in results]
        stats[key] = {
            'mean': statistics.mean(vals),
            'std': statistics.pstdev(vals),
            'values': vals,
        }
    return stats


def score_transcript_repeated(
    transcript: list[Turn], judge_client, n: int = 5, rubric_version: str = 'v1'
) -> dict:
    """고정 대화를 N회 채점해 항목별 통계를 낸다 (judge 순수 변동).

    개별 채점이 실패하면(파싱·검증 오류) 그 회차만 제외하고 나머지로 집계한다.
    """
    results = []
    for _ in range(n):
        try:
            results.append(judge(transcript, judge_client, rubric_version))
        except Exception:
            continue
    return score_stats(results)


def run_scenario_repeated(
    steps: list[dict], utterance_client, judge_client, n: int = 3, supervisor_client=None
) -> dict:
    """시나리오를 N회 실행·채점해 항목별 통계를 낸다 (대화 변동 + judge 변동).

    개별 회차가 실패하면 제외하고 나머지로 집계한다.
    """
    results = []
    for _ in range(n):
        try:
            _, result, _ = run_scenario(
                steps, utterance_client, judge_client, supervisor_client=supervisor_client
            )
            results.append(result)
        except Exception:
            continue
    return score_stats(results)
