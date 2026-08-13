"""uvicorn 진입점 — 실제 LLM client를 주입한 앱.

실행: cd server && PYTHONPATH=. uv run uvicorn api.main:app --reload
(OPENAI_API_KEY 필요. 키 없이 구조만 보려면 api.app:app을 쓰면 발화가 스텁으로 나온다.)
"""

from dotenv import load_dotenv

from api.app import create_app
from engine.eval.providers import OpenAIClient

load_dotenv()

app = create_app(
    utterance_client=OpenAIClient(model='gpt-5.6-luna', temperature=None),  # 발화 생성(temp 고정 모델)
    supervisor_client=OpenAIClient(json_mode=True),  # 화자 선정(gpt-4o-mini, 비용 절약)
    tts_client=OpenAIClient(),  # 발화 → 화자 voice로 음성 합성
)
