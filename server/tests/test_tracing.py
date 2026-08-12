"""트레이싱 sink 폴백 계약 테스트.

Langfuse 키가 없으면 개발·테스트가 막히면 안 된다 — get_sink()가 NoopSink로
폴백하고, NoopSink 기록이 예외 없이 no-op인지 박제한다.
"""

from engine.eval.schema import JudgeResult
from engine.eval.tracing import NoopSink, get_sink


def test_get_sink_falls_back_to_noop_without_keys(monkeypatch):
    monkeypatch.delenv('LANGFUSE_PUBLIC_KEY', raising=False)
    monkeypatch.delenv('LANGFUSE_SECRET_KEY', raising=False)
    assert isinstance(get_sink(), NoopSink)


def test_noop_sink_records_without_error():
    result = JudgeResult(scores={}, overall_comment='')
    # 예외 없이 None을 반환해야 한다
    assert NoopSink().record_judge([], result, {}) is None
