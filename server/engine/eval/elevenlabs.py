"""ElevenLabs TTS 클라이언트 — 한국어 자연스러움.

OpenAIClient와 같은 synthesize(text, voice) 인터페이스를 만족하므로
tts_client 자리에 그대로 주입할 수 있다(조립형 D-002, provider 교체 자유).
voice는 ElevenLabs voice_id(콘솔에서 확인).
"""

import os

import httpx

ELEVEN_URL = 'https://api.elevenlabs.io/v1/text-to-speech'


class ElevenLabsClient:
    """ElevenLabs TTS. voice 인자는 ElevenLabs voice_id."""

    def __init__(
        self, api_key: str | None = None, model_id: str = 'eleven_multilingual_v2'
    ):
        self._key = api_key or os.environ.get('ELEVENLABS_API_KEY')
        if not self._key:
            raise RuntimeError('ELEVENLABS_API_KEY가 없다 — server/.env에 설정하라')
        self._model_id = model_id  # 한국어는 multilingual v2 (저지연은 flash 계열)

    async def synthesize(self, text: str, voice: str) -> bytes:
        """voice=ElevenLabs voice_id. mp3 bytes를 반환한다."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f'{ELEVEN_URL}/{voice}',
                headers={'xi-api-key': self._key, 'accept': 'audio/mpeg'},
                json={'text': text, 'model_id': self._model_id},
            )
            resp.raise_for_status()
            return resp.content
