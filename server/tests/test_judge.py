"""judge() 계약 무결성 테스트 (D-005).

채점 결과의 항목 key가 루브릭 5항목과 정확히 일치하지 않으면 예외가 나야 한다.
LLMClient는 fake로 주입해 '판정 오염 방지 흐름'만 검증한다(LLM 답변 문구 검증 아님).
"""

import pytest

from engine.eval.judge import judge, load_rubric
from engine.eval.schema import Turn

CONVO = [Turn(speaker='ai_a', text='안녕'), Turn(speaker='user', text='야')]


class FakeClient:
    """지정한 JSON 문자열을 그대로 반환하는 fake 심판."""

    def __init__(self, payload: str):
        self._payload = payload

    def complete(self, system: str, user: str) -> str:
        return self._payload


def _payload(keys: list[str]) -> str:
    """주어진 key들로 채점 결과 JSON을 만든다 (모두 8점)."""
    items = ', '.join(f'"{k}": {{"score": 8, "reason": "ok"}}' for k in keys)
    return '{"scores": {' + items + '}, "overall_comment": "good"}'


def _rubric_keys() -> list[str]:
    return [it['key'] for it in load_rubric()['items']]


def test_judge_happy_path():
    # 루브릭 5항목을 모두 채점한 정상 결과
    result = judge(CONVO, FakeClient(_payload(_rubric_keys())))
    assert set(result.scores) == set(_rubric_keys())


def test_judge_rejects_missing_item():
    # 4항목만 채점 → 계약 위반으로 예외
    with pytest.raises(ValueError):
        judge(CONVO, FakeClient(_payload(_rubric_keys()[:4])))


def test_judge_rejects_wrong_key():
    # 개수는 5개지만 첫 key가 오타 → 계약 위반으로 예외
    keys = _rubric_keys()
    keys[0] = 'turntaking'
    with pytest.raises(ValueError):
        judge(CONVO, FakeClient(_payload(keys)))


def test_judge_recovers_misplaced_overall_comment():
    # LLM이 overall_comment를 scores 안에 넣어도 복구해 파싱한다
    keys = _rubric_keys()
    items = ', '.join(f'"{k}": {{"score": 8, "reason": "ok"}}' for k in keys)
    payload = '{"scores": {' + items + ', "overall_comment": "총평"}}'
    result = judge(CONVO, FakeClient(payload))
    assert result.overall_comment == '총평'
    assert set(result.scores) == set(keys)
