"""judge 신뢰성 스모크 — OPENAI_API_KEY 필요.

노이즈 원인 분리:
  A) 고정 대화 N회 재채점 → judge 순수 변동
  B) 시나리오 N회 재실행 → 대화 변동 + judge 변동
A가 크면 judge 불안정, A는 작고 B가 크면 대화 변동 탓.
실행: cd server && PYTHONPATH=. uv run python scripts/smoke_reliability.py
"""

from dotenv import load_dotenv

from engine.eval.harness import (
    load_scenarios,
    run_scenario,
    run_scenario_repeated,
    score_transcript_repeated,
)
from engine.eval.providers import OpenAIClient

load_dotenv()

N = 5


def _print_stats(title: str, stats: dict) -> None:
    print(f'=== {title} ===')
    for key, s in stats.items():
        print(f"  {key}: 평균 {s['mean']:.1f} ± {s['std']:.2f}  {s['values']}")
    print()


def main() -> None:
    utterance = OpenAIClient(temperature=0.9)
    judge_client = OpenAIClient(json_mode=True)
    supervisor = OpenAIClient(json_mode=True)

    # 변동이 컸던 radio_silence로 검증
    steps = next(s['steps'] for s in load_scenarios() if s['id'] == 'radio_silence')

    # 대화 한 번 생성 → 고정
    transcript, _, _ = run_scenario(steps, utterance, judge_client, supervisor_client=supervisor)

    _print_stats(
        f'A) judge 순수 변동 (고정 대화 {N}회 채점)',
        score_transcript_repeated(transcript, judge_client, n=N),
    )
    _print_stats(
        f'B) 대화+judge 변동 (시나리오 {N}회 실행)',
        run_scenario_repeated(steps, utterance, judge_client, n=N, supervisor_client=supervisor),
    )


if __name__ == '__main__':
    main()
