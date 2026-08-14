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


def test_question_after_limit_is_preserved():
    # 상한 직후 문장이 질문이면 살린다 — 유저 소환 질문(§1.3)이 상한(§1.4)에 잘리는 충돌 방지
    assert limit_sentences('헐. 진짜? 그래서 넌 어땠어?') == '헐. 진짜? 그래서 넌 어땠어?'
    # 질문이 아니면 원래대로 상한 적용
    assert limit_sentences('하나. 둘. 셋이다. 넷.') == '하나. 둘.'
