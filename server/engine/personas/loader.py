"""페르소나 로딩 + 발화 프롬프트 구성.

duo.yaml을 읽어 Persona로 검증하고, 발화 생성용 system 프롬프트를 만든다.
보이스는 개별이 아니라 쌍 단위 프리셋(voice_presets)으로 관리한다 — 케미가
쌍 단위이듯(D-004) 목소리 대비(청각적 화자 구분)도 쌍으로 튜닝하기 때문.
"""

from functools import lru_cache
from pathlib import Path

import yaml

from .schema import Persona

DUO_PATH = Path(__file__).parent / 'duo.yaml'


def _resolve_voices(data: dict) -> dict[str, str | None]:
    """활성 voice_preset의 {화자키: voice_id}를 푼다. 프리셋 미지정이면 빈 매핑."""
    preset = data.get('voice_preset')
    if not preset:
        return {}
    presets = data.get('voice_presets') or {}
    if preset not in presets:
        raise ValueError(f'voice_preset "{preset}"이 voice_presets에 없다')
    return presets[preset]


@lru_cache(maxsize=1)  # 매 발화마다 파일 IO를 반복하지 않는다 (이벤트 루프 블로킹 방지)
def load_personas() -> dict[str, Persona]:
    """duo.yaml의 듀오를 {key: Persona}로 로드한다 (voice는 활성 프리셋에서 주입)."""
    data = yaml.safe_load(DUO_PATH.read_text(encoding='utf-8'))
    if not data or 'personas' not in data:
        raise ValueError('duo.yaml에 personas가 없다')
    voices = _resolve_voices(data)
    return {
        key: Persona(**val, voice_id=voices.get(key))
        for key, val in data['personas'].items()
    }


def get_persona(speaker: str) -> Persona:
    """화자 key로 페르소나를 가져온다. 미정의 화자면 에러."""
    personas = load_personas()
    if speaker not in personas:
        raise ValueError(f'알 수 없는 화자: {speaker}')
    return personas[speaker]


def build_persona_prompt(
    persona: Persona,
    other_name: str,
    summon_user: bool = False,
    intent: str | None = None,
) -> str:
    """페르소나로 발화 생성용 system 프롬프트를 만든다.

    발화 길이 상한(1~2문장)으로 §1.4 '연설화'를 프롬프트에서 1차 제어한다.
    intent가 있으면 supervisor가 정한 이번 발화 의도를 반영한다(§1.3).
    summon_user=True면 유저를 대화로 끌어들이도록 지시한다(§1.3 유저 소환, 소외 방지).
    """
    prompt = (
        f'너는 "{persona.name}". {persona.stance}.\n'
        f'말투: {persona.speech_style}.\n'
        f'관심사: {", ".join(persona.topic_bias)}.\n'
        f'유저 1명, 친구 "{other_name}"와 셋이서 퇴근 후 편하게 수다 떠는 중이다.\n'
        '한국어 반말로 1~2문장, 진짜 말하듯 짧게. 길게 설명하거나 착하게 정리하지 마라.\n'
        '상대 말에 매번 동의·응원하지 마라 — 네 스탠스대로 반응해라(딴지·화제 전환도 자연스럽게).\n'
        '이력에 이미 나온 인사·리액션을 반복하지 마라. 직전 발화에 새로운 내용을 얹어 대화를 진전시켜라.\n'
        '네 발화 내용만 출력해라 — "이름:" 같은 화자 표시를 앞에 붙이지 마라.\n'
        '이모지는 쓰지 마라 (음성으로 읽을 대사다).'
    )
    if intent:
        prompt += f'\n이번 발화 의도: {intent}. 이 결대로 말해라.'
    if summon_user:
        prompt += '\n지금은 유저(사람)에게 오늘 어땠는지 가볍게 말을 걸어 대화에 끌어들여라.'
    return prompt
