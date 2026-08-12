"""judge 실호출 스모크 — OPENAI_API_KEY 필요.

실행: cd server && PYTHONPATH=. uv run python scripts/smoke_judge.py

외부 LLM 실호출은 자동화 테스트가 아니라 로컬에서 수동으로 확인한다.
채점 '품질'(점수 타당성)은 이후 eval로 보고, 여기서는 파이프가 도는지만 본다.
"""

from dotenv import load_dotenv

from engine.eval.judge import judge
from engine.eval.providers import OpenAIClient
from engine.eval.schema import Turn
from engine.eval.scoring import is_go
from engine.eval.tracing import get_sink

load_dotenv()

CONVO = [
    Turn(speaker='ai_a', text='오늘 회사에서 진짜 웃긴 일 있었어'),
    Turn(speaker='ai_b', text='뭔데뭔데 빨리 말해봐'),
    Turn(speaker='user', text='나도 오늘 좀 힘들었는데'),
    Turn(speaker='ai_a', text='어? 무슨 일 있었어? 말해봐'),
]


def main() -> None:
    sink = get_sink()
    result = judge(CONVO, OpenAIClient())
    sink.record_judge(CONVO, result, {'rubric_version': 'v1', 'source': 'smoke'})
    for key, item in result.scores.items():
        print(f'{key}: {item.score} — {item.reason}')
    print('overall:', result.overall_comment)
    print('go?', is_go(result))
    print('trace sink:', type(sink).__name__)


if __name__ == '__main__':
    main()
