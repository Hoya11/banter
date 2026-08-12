"""라디오 모드 발화 스모크 — OPENAI_API_KEY 필요.

실행: cd server && PYTHONPATH=. uv run python scripts/smoke_chat.py
AI 둘(도현·소은)이 유저 없이 몇 턴 주고받는다. 실호출이라 로컬에서만 확인.
"""

from dotenv import load_dotenv

from engine.eval.providers import OpenAIClient
from engine.graph.graph import build_graph
from engine.graph.state import initial_state

load_dotenv()


def main() -> None:
    client = OpenAIClient(temperature=0.9)  # 발화는 다양성 위해 temp 높게
    supervisor = OpenAIClient(json_mode=True)  # 화자 선정: 맥락 기반(§1.3)
    graph = build_graph(client, supervisor)
    state = initial_state()
    for _ in range(4):
        state = graph.invoke(state)
        last = state['messages'][-1]
        print(f'{last["speaker"]}: {last["text"]}')


if __name__ == '__main__':
    main()
