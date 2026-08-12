"""go/no-go 컷 판정 (D-006).

상대 추세가 아니라 절대 컷오프로 Phase 1 종료를 판정한다.
임계값은 초안이며 조정 시 docs/experiments/에 이력을 남긴다
(컷을 사후에 낮춰 통과시키는 자기기만 방지).
"""

from .schema import JudgeResult

# 컷 임계값 (초안 D-006)
CUT_MEAN_MIN = 7.0  # 항목 점수 평균의 최소
CUT_ITEM_MIN = 6  # 최저 항목 점수의 최소


def is_go(result: JudgeResult) -> bool:
    """루브릭 채점 결과가 go 컷을 통과하는지 판정한다.

    통과 조건(D-006): 항목 점수 평균 >= CUT_MEAN_MIN AND 최저 항목 점수 >= CUT_ITEM_MIN.
    """
    scores = [item.score for item in result.scores.values()]
    if not scores:  # 채점된 항목이 없으면 통과로 볼 수 없다
        return False
    mean = sum(scores) / len(scores)
    return mean >= CUT_MEAN_MIN and min(scores) >= CUT_ITEM_MIN
