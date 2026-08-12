"""go/no-go 컷 판정(D-006)의 결정적 계약 테스트.

컷 경계값이 조용히 바뀌면 Phase 1 종료 판정이 틀어지므로, 경계를 박제한다.
"""

from engine.eval.schema import ItemScore, JudgeResult
from engine.eval.scoring import is_go


def _result(scores: list[int]) -> JudgeResult:
    """점수 리스트로 JudgeResult를 만든다 (근거는 채점 로직과 무관하므로 빈 문자열)."""
    return JudgeResult(
        scores={f'k{i}': ItemScore(score=v, reason='') for i, v in enumerate(scores)},
        overall_comment='',
    )


def test_go_pass():
    # 평균 7.8, 최저 7 — 두 조건 모두 여유 통과
    assert is_go(_result([8, 8, 7, 9, 7])) is True


def test_go_at_boundary():
    # 평균 7.0, 최저 7 — 경계값(>=)이라 통과
    assert is_go(_result([7, 7, 7, 7, 7])) is True


def test_no_go_when_min_below_cut():
    # 평균은 넉넉하지만 최저 항목 5 < 6 → 탈락
    assert is_go(_result([8, 8, 5, 9, 7])) is False


def test_no_go_when_mean_below_cut():
    # 최저는 6이지만 평균 6.4 < 7.0 → 탈락
    assert is_go(_result([7, 7, 6, 6, 6])) is False


def test_no_go_when_empty():
    # 채점된 항목이 없으면 통과로 보지 않는다 (예외도 나지 않아야 함)
    assert is_go(_result([])) is False
