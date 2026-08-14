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

    transcript = _to_transcript(state['messages'])
    result = judge(transcript, judge_client, rubric_version)
    get_sink().record_judge(
        transcript,
        result,
        {'source': 'pilot', 'turns': turns, 'rubric_version': rubric_version},
    )
    return transcript, result, is_go(result)


def _to_transcript(messages: list[dict]) -> list[Turn]:
    """상태 메시지를 judge 입력으로 변환한다 — interrupted 플래그를 보존해야
    judge가 끊김을 인지한다(누락 시 barge-in 평가가 무효가 된다)."""
    return [
        Turn(speaker=m['speaker'], text=m['text'], interrupted=m.get('interrupted', False))
        for m in messages
    ]


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


def _truncate_midway(text: str) -> str:
    """발화를 중간에서 자른다 — 실제 barge-in에서 '재생된 지점까지만 남는' 상황 재현.

    절반 지점 이전의 마지막 공백에서 자른다(단어 중간 파손 방지). 결정적이라 테스트 가능.
    """
    half = len(text) // 2
    cut = text.rfind(' ', 0, half)
    return text[: cut if cut > 0 else half].rstrip()


def _interrupt(state: dict, text: str) -> dict:
    """AI 발화 중 유저 개입 — 직전 AI 발화를 실제로 잘라 끊김 처리하고 유저 발화를 주입한다(§2.1).

    플래그만 달면 judge가 온전한 발화를 보게 되어 '수습 능력'이 평가되지 않는다
    (2026-08-14 검수에서 발견·정정). API의 barge-in과 동일하게 부분 발화만 남긴다.
    """
    messages = [dict(m) for m in state['messages']]
    if messages and messages[-1]['speaker'] in ('ai_a', 'ai_b'):
        messages[-1]['text'] = _truncate_midway(messages[-1]['text'])
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

    transcript = _to_transcript(state['messages'])
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
        except Exception as exc:
            # 실패를 무음으로 삼키면 표본 수가 조용히 줄어 통계가 왜곡된다 — 반드시 남긴다
            print(f'[repeat] 채점 실패({type(exc).__name__}) — 회차 제외')
    if len(results) < n:
        print(f'[repeat] 유효 표본 {len(results)}/{n} — 통계 해석 시 주의')
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
        except Exception as exc:
            print(f'[repeat] 회차 실패({type(exc).__name__}) — 제외')
            continue
    return score_stats(results)
