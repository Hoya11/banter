"""발화 문장 상한(§1.4 연설화 방지)의 결정적 계약 테스트.

프롬프트가 어겨도 후처리가 상한을 보장하는지 박제한다.
"""

from engine.graph.graph import finalize_utterance, limit_sentences


def test_trims_to_two_sentences():
    text = '첫 문장이야. 둘째 문장이지. 셋째는 잘려야 해. 넷째도.'
    assert limit_sentences(text) == '첫 문장이야. 둘째 문장이지.'


def test_keeps_short_utterance():
    assert limit_sentences('짧게 한마디.') == '짧게 한마디.'


def test_no_punctuation_passes_through():
    # 종결부호 없는 반말체는 자르지 않는다
    assert limit_sentences('그치 완전 맞아') == '그치 완전 맞아'


def test_decimal_point_not_sentence_end():
    # 소수점은 문장 끝이 아니다 — "2.5랑 3."으로 파손되던 회귀 방지
    assert limit_sentences('2.5랑 3.5 중에 뭐가 나아? 난 몰라.') == '2.5랑 3.5 중에 뭐가 나아? 난 몰라.'
    assert limit_sentences('어제 3.5시간 잤어. 진짜 죽겠다. 셋째는 잘림.') == '어제 3.5시간 잤어. 진짜 죽겠다.'


def test_consecutive_marks_count_as_one():
    # '?!'나 '...'는 한 문장의 끝
    text = '진짜?! 대박이다... 이건 잘려야지. 넷째.'
    assert limit_sentences(text) == '진짜?! 대박이다...'


def test_finalize_strips_prefix_then_limits():
    raw = '도현: 하나. 둘. 셋.'
    assert finalize_utterance(raw, '도현') == '하나. 둘.'


def test_pop_sentences_incremental():
    # 스트리밍 문장 분리 — 경계 확정은 다음 글자가 와야 (연속부호·소수점 유예)
    from engine.graph.graph import pop_sentences

    done, rest = pop_sentences('하나. 둘째가 진행')
    assert done == ['하나.'] and rest == ' 둘째가 진행'
    done, rest = pop_sentences('진짜?')  # 버퍼 끝 — '!'가 이어질 수 있어 유예
    assert done == [] and rest == '진짜?'
    done, rest = pop_sentences('진짜?! 그러')  # 다음 글자 확인 후 확정
    assert done == ['진짜?!'] and rest == ' 그러'
    done, rest = pop_sentences('3.5시간 잤어. 그리고')  # 소수점은 경계 아님
    assert done == ['3.5시간 잤어.'] and rest == ' 그리고'


def test_strip_audio_tags_variants():
    # LLM이 내는 태그 변형은 다 지우고, 한국어 대괄호는 건드리지 않는다 (검수 🟡8 회귀)
    from engine.graph.graph import strip_audio_tags

    assert strip_audio_tags('[sighs] 하아') == '하아'
    assert strip_audio_tags('[Sighs] 하아') == '하아'  # 대문자
    assert strip_audio_tags('[sarcastic tone here] 그래') == '그래'  # 여러 단어
    assert strip_audio_tags('말 사이 <break/> 쉼') == '말 사이 쉼'
    assert strip_audio_tags('말 사이 <break time="0.4s"> 쉼') == '말 사이 쉼'  # self-closing 아님
    assert strip_audio_tags('[도현] 얘기랑 [ㅋㅋ] 는 유지') == '[도현] 얘기랑 [ㅋㅋ] 는 유지'


def test_question_after_limit_is_preserved():
    # 상한 직후 문장이 질문이면 살린다 — 유저 소환 질문(§1.3)이 상한(§1.4)에 잘리는 충돌 방지
    assert limit_sentences('헐. 진짜? 그래서 넌 어땠어?') == '헐. 진짜? 그래서 넌 어땠어?'
    # 질문이 아니면 원래대로 상한 적용
    assert limit_sentences('하나. 둘. 셋이다. 넷.') == '하나. 둘.'
