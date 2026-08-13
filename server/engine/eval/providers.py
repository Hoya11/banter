"""LLMClient 구현체 — judge(judge.py)에 주입한다.

judge용 LLM은 OpenAI로 시작한다(채점 심판은 하나로 고정, Phase 1 [결정]).
judge.py의 LLMClient Protocol을 만족하므로, 앞으로 다른 provider·앙상블로 바꿔도
judge()·rubric·schema 계약은 불변이다.
"""

import os

from openai import OpenAI


class OpenAIClient:
    """OpenAI Chat Completions 기반 심판.

    JSON object 모드로 순수 JSON 출력을 강제하고(코드펜스·설명 텍스트 방지),
    temperature=0으로 채점 재현성을 확보한다.
    """

    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        api_key: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ):
        key = api_key or os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY가 없다 — server/.env에 설정하라')
        self._client = OpenAI(api_key=key)
        self._api_key = key
        self._async_client = None  # 스트리밍용, lazy 생성
        self._model = model
        self._json = json_mode  # judge=True(순수 JSON), 발화=False(자유 텍스트)
        self._temperature = temperature  # judge=0(재현성), 발화=높게(다양성)

    def complete(self, system: str, user: str) -> str:
        kwargs = {}
        if self._json:
            kwargs['response_format'] = {'type': 'json_object'}  # 순수 JSON 강제
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            temperature=self._temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ''

    async def complete_stream(self, system: str, user: str):
        """발화를 토큰 단위로 스트리밍한다 (async generator). 발화 생성 전용."""
        from openai import AsyncOpenAI

        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def synthesize(self, text: str, voice: str) -> bytes:
        """텍스트를 음성(mp3 bytes)으로 합성한다 (TTS)."""
        from openai import AsyncOpenAI

        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        resp = await self._async_client.audio.speech.create(
            model='tts-1', voice=voice, input=text, response_format='mp3'
        )
        return resp.content
