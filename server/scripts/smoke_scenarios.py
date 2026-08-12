"""시나리오 셋 채점 스모크 — OPENAI_API_KEY 필요.

실행: cd server && PYTHONPATH=. uv run python scripts/smoke_scenarios.py
유저 발화가 포함된 시나리오들을 채점해 항목별·전체 평균을 본다.
"""

from dotenv import load_dotenv

from engine.eval.harness import run_scenario_set
from engine.eval.providers import OpenAIClient

load_dotenv()


def main() -> None:
    utterance = OpenAIClient(temperature=0.9)
    judge_client = OpenAIClient(json_mode=True)
    supervisor = OpenAIClient(json_mode=True)  # 화자 선정: 맥락 기반(§1.3)

    results = run_scenario_set(utterance, judge_client, supervisor_client=supervisor)
    all_scores = []
    for r in results:
        scores = r['result'].scores
        avg = sum(i.score for i in scores.values()) / len(scores)
        all_scores.extend(i.score for i in scores.values())
        print(f"[{r['id']}] {'GO' if r['go'] else 'NO-GO'} 평균 {avg:.1f}")
        for key, item in scores.items():
            print(f'  {key}: {item.score} — {item.reason}')
        print()

    print(f'시나리오 셋 전체 평균: {sum(all_scores) / len(all_scores):.2f}')


if __name__ == '__main__':
    main()
