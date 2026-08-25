"""uvicorn 진입점 — 실제 LLM client를 주입한 앱.

실행: cd server && PYTHONPATH=. uv run uvicorn api.main:app --reload
(OPENAI_API_KEY 필요. 키 없이 구조만 보려면 api.app:app을 쓰면 발화가 스텁으로 나온다.)
"""

from dotenv import load_dotenv

from api.app import create_app
from engine.eval.elevenlabs import ElevenLabsClient
from engine.eval.providers import OpenAIClient

load_dotenv()

app = create_app(
    utterance_client=OpenAIClient(model='gpt-5.6-luna', temperature=None),  # 발화 생성(temp 고정 모델)
    supervisor_client=OpenAIClient(json_mode=True),  # 화자 선정(gpt-4o-mini, 비용 절약)
    # TTS provider 선택 — 음성 없이 개발하려면 tts_client=None으로.
    # 보이스 조합은 duo.yaml의 voice_preset (chris-jessica | liam-laura).
    # premade 보이스 × v3는 무료 플랜에서 사용 가능 확인(2026-08 청음 비교).
    tts_client=ElevenLabsClient(model_id='eleven_v3'),
    stt_client=OpenAIClient(stt_model='gpt-4o-mini-transcribe'),  # 🎤 전사 (whisper-1보다 한국어 강함)
    # prefetch로 생성·합성이 재생 뒤에 숨으므로, 발화 사이 틈은 유저 발언 기회만큼만
    radio_sec=1.5,
)
