"""페르소나 로딩 + 발화 프롬프트 구성.

duo.yaml을 읽어 Persona로 검증하고, 발화 생성용 system 프롬프트를 만든다.
"""

from pathlib import Path

import yaml

from .schema import Persona

DUO_PATH = Path(__file__).parent / 'duo.yaml'


def load_personas() -> dict[str, Persona]:
    """duo.yaml의 듀오를 {key: Persona}로 로드한다."""
    data = yaml.safe_load(DUO_PATH.read_text(encoding='utf-8'))
    if not data or 'personas' not in data:
        raise ValueError('duo.yaml에 personas가 없다')
    return {key: Persona(**val) for key, val in data['personas'].items()}


def get_persona(speaker: str) -> Persona:
    """화자 key로 페르소나를 가져온다. 미정의 화자면 에러."""
    personas = load_personas()
    if speaker not in personas:
        raise ValueError(f'알 수 없는 화자: {speaker}')
    return personas[speaker]


def build_persona_prompt(persona: Persona, other_name: str) -> str:
    """페르소나로 발화 생성용 system 프롬프트를 만든다.

    발화 길이 상한(1~3문장)으로 §1.4 '연설화'를 프롬프트에서 1차 제어한다.
    """
    return (
        f'너는 "{persona.name}". {persona.stance}.\n'
        f'말투: {persona.speech_style}.\n'
        f'관심사: {", ".join(persona.topic_bias)}.\n'
        f'유저 1명, 친구 "{other_name}"와 셋이서 퇴근 후 편하게 수다 떠는 중이다.\n'
        '한국어 반말로 1~2문장, 진짜 말하듯 짧게. 길게 설명하거나 착하게 정리하지 마라.\n'
        '상대 말에 매번 동의·응원하지 마라 — 네 스탠스대로 반응해라(딴지·화제 전환도 자연스럽게).\n'
        '네 발화 내용만 출력해라 — "이름:" 같은 화자 표시를 앞에 붙이지 마라.\n'
        '이모지는 쓰지 마라 (음성으로 읽을 대사다).'
    )
