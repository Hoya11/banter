"""uvicorn 진입점 — 실제 LLM client를 주입한 앱.

실행: cd server && PYTHONPATH=. uv run uvicorn api.main:app --reload
(OPENAI_API_KEY 필요. 키 없이 구조만 보려면 api.app:app을 쓰면 발화가 스텁으로 나온다.)
"""

from dotenv import load_dotenv

from api.app import create_app
from engine.eval.providers import OpenAIClient

load_dotenv()

app = create_app(
    utterance_client=OpenAIClient(temperature=0.9),
    supervisor_client=OpenAIClient(json_mode=True),
)
