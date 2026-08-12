"""평가 트레이싱 sink 추상화.

judge 결과 기록을 인터페이스 뒤로 숨긴다 — sink가 Langfuse든 no-op이든
judge/eval 코드는 안 바뀐다. Langfuse 키 미설정 시 NoopSink로 조용히 넘어가
개발·테스트가 트레이싱 백엔드 없이도 돌아간다.
"""

import os
from typing import Protocol

from .schema import JudgeResult, Turn


class TraceSink(Protocol):
    """judge 채점 결과를 기록하는 계약."""

    def record_judge(
        self, transcript: list[Turn], result: JudgeResult, meta: dict
    ) -> None: ...


class NoopSink:
    """Langfuse 미설정·테스트용. 아무것도 하지 않는다."""

    def record_judge(
        self, transcript: list[Turn], result: JudgeResult, meta: dict
    ) -> None:
        return None


class LangfuseSink:
    """채점 결과를 Langfuse Cloud에 evaluator observation + 항목별 score로 남긴다."""

    def __init__(self, client=None):
        from langfuse import Langfuse

        # env: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
        self._lf = client or Langfuse()

    def record_judge(
        self, transcript: list[Turn], result: JudgeResult, meta: dict
    ) -> None:
        with self._lf.start_as_current_observation(
            name='judge',
            as_type='evaluator',
            input=[t.model_dump() for t in transcript],
            output=result.model_dump(),
            metadata=meta,
        ):
            for key, item in result.scores.items():
                self._lf.score_current_span(
                    name=key,
                    value=item.score,
                    comment=item.reason,
                    data_type='NUMERIC',
                )
        self._lf.flush()  # 스크립트 종료 전 전송 보장


def get_sink() -> TraceSink:
    """env에 Langfuse 키가 있으면 LangfuseSink, 없으면 NoopSink.

    키 로딩 실패 시에도 NoopSink로 폴백해 트레이싱이 판정 흐름을 막지 않는다.
    """
    if os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY'):
        try:
            return LangfuseSink()
        except Exception:
            return NoopSink()
    return NoopSink()
