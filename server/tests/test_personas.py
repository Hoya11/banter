"""페르소나 로딩·프롬프트 구성의 결정적 계약 테스트.

대비형 듀오(D-007)의 두 스탠스가 실제로 달라야 하고(케미 전제),
프롬프트에 페르소나 정체성이 실려야 한다.
"""

import pytest

from engine.personas.loader import build_persona_prompt, get_persona, load_personas


def test_loads_duo():
    personas = load_personas()
    assert set(personas) == {'ai_a', 'ai_b'}


def test_stances_contrast():
    # 대비형 전제 — 두 스탠스가 같으면 루프 방지 설계가 무너진다
    personas = load_personas()
    assert personas['ai_a'].stance != personas['ai_b'].stance


def test_get_persona_unknown_raises():
    with pytest.raises(ValueError):
        get_persona('ai_c')


def test_prompt_carries_identity():
    persona = get_persona('ai_a')
    prompt = build_persona_prompt(persona, other_name='소은')
    assert persona.name in prompt  # 자기 이름
    assert '소은' in prompt  # 상대 이름


def test_summon_flag_adds_user_call():
    # summon_user=True일 때만 유저 소환 지시가 붙는다 (§1.3)
    persona = get_persona('ai_a')
    summoned = build_persona_prompt(persona, '소은', summon_user=True)
    normal = build_persona_prompt(persona, '소은', summon_user=False)
    assert '끌어들여' in summoned
    assert '끌어들여' not in normal
