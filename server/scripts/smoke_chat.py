"""라디오 모드 발화 스모크 — OPENAI_API_KEY 필요.

실행: cd server && PYTHONPATH=. uv run python scripts/smoke_chat.py
AI 둘(도현·소은)이 유저 없이 몇 턴 주고받는다. 실호출이라 로컬에서만 확인.
"""

from dotenv import load_dotenv

from engine.eval.providers import OpenAIClient
from engine.graph.graph import build_graph

load_dotenv()


def _init_state() -> dict:
    return {
        'messages': [],
        'topic_stack': [],
        'current_speaker': None,
        'consecutive_ai_turns': 0,
        'last_user_turn_ts': None,
        'personas': {
            'ai_a': {'speak_count': 0, 'last_stance': ''},
            'ai_b': {'speak_count': 0, 'last_stance': ''},
        },
        'pending_user_input': None,
        'session_elapsed': 0.0,
        'budget_used': 0.0,
    }


def main() -> None:
    client = OpenAIClient(temperature=0.9)  # 발화는 다양성 위해 temp 높게
    graph = build_graph(client)
    state = _init_state()
    for _ in range(4):
        state = graph.invoke(state)
        last = state['messages'][-1]
        print(f'{last["speaker"]}: {last["text"]}')


if __name__ == '__main__':
    main()
