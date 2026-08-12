"""end-to-end 파일럿 스모크 — OPENAI_API_KEY 필요.

실행: cd server && PYTHONPATH=. uv run python scripts/smoke_pilot.py
대화 생성 → judge 채점 → go/no-go 를 한 번에 돌린다. 실호출이라 로컬에서만.
"""

from dotenv import load_dotenv

from engine.eval.harness import run_pilot
from engine.eval.providers import OpenAIClient

load_dotenv()


def main() -> None:
    utterance = OpenAIClient(temperature=0.9)  # 발화: 다양성
    judge_client = OpenAIClient(json_mode=True)  # 채점: JSON + temp0(재현성)

    transcript, result, go = run_pilot(utterance, judge_client, turns=6)

    print('=== 대화 ===')
    for turn in transcript:
        print(f'{turn.speaker}: {turn.text}')
    print('\n=== 채점 ===')
    for key, item in result.scores.items():
        print(f'{key}: {item.score} — {item.reason}')
    print('overall:', result.overall_comment)
    print('\n판정:', 'GO' if go else 'NO-GO')


if __name__ == '__main__':
    main()
