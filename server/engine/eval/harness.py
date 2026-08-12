"""Phase 1 end-to-end 파일럿: 대화 생성 → judge 채점 → go/no-go.

Phase 1의 핵심 루프. 이 파이프가 페르소나·프롬프트 튜닝의 나침반이다
(감이 아니라 judge 점수로 판단). 결과는 sink로 기록해 버전별 비교에 쓴다.
"""

from ..graph.graph import build_graph
from ..graph.state import initial_state
from .judge import judge
from .schema import JudgeResult, Turn
from .scoring import is_go
from .tracing import get_sink


def run_pilot(
    utterance_client,
    judge_client,
    turns: int = 6,
    rubric_version: str = 'v1',
) -> tuple[list[Turn], JudgeResult, bool]:
    """N턴 라디오 모드 대화를 만들고 judge로 채점해 (transcript, result, go)를 반환한다."""
    graph = build_graph(utterance_client)
    state = initial_state()
    for _ in range(turns):
        state = graph.invoke(state)

    transcript = [Turn(speaker=m['speaker'], text=m['text']) for m in state['messages']]
    result = judge(transcript, judge_client, rubric_version)
    get_sink().record_judge(
        transcript,
        result,
        {'source': 'pilot', 'turns': turns, 'rubric_version': rubric_version},
    )
    return transcript, result, is_go(result)
