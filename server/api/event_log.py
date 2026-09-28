"""재현 가능한 지연 측정을 위한 로컬 JSONL 이벤트 기록기."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import floor, isfinite
from pathlib import Path
from threading import Lock
from time import monotonic

MetadataScalar = str | int | float | bool

# 실행 조건을 재현하는 데 필요한 값만 허용한다. 임의 환경 변수나 client 객체를
# 그대로 직렬화하면 API key와 대화 원문이 섞일 수 있으므로 필드도 함께 제한한다.
METADATA_FIELDS = frozenset(
    {
        'baseline_id',
        'interaction_mode',
        'stt_mode',
        'control_transport',
        'audio_downlink',
        'orchestration_mode',
        'audio_stop_semantics',
        'build_revision',
        'utterance_model',
        'supervisor_model',
        'stt_model',
        'stt_file_model',
        'stt_stream_encoding',
        'stt_stream_sample_rate_hz',
        'stt_stream_chunk_samples',
        'stt_delay',
        'stt_chunk_count',
        'stt_audio_samples',
        'stt_audio_duration_ms',
        'stt_audio_bytes',
        'audio_uplink',
        'tts_model',
        'protocol_version',
        'voice_mode',
        'radio_sec',
        'ack_sec',
        'max_turns',
        'unified',
        'vad_engine',
        'vad_start_ms',
        'vad_end_silence_ms',
        'vad_pre_roll_ms',
        'vad_max_utterance_ms',
    }
)


def _safe_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, MetadataScalar] | None:
    """허용된 스칼라 실행 정보만 복사한다."""
    if not metadata:
        return None
    safe: dict[str, MetadataScalar] = {}
    for key, value in metadata.items():
        if key not in METADATA_FIELDS or value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (str, int)):
            safe[key] = value
        elif isinstance(value, float) and isfinite(value):
            safe[key] = value
    return safe or None


class EventName(StrEnum):
    SESSION_STARTED = 'session_started'
    TURN_STARTED = 'turn_started'
    GENERATION_STARTED = 'generation_started'
    TTS_STARTED = 'tts_started'
    AUDIO_SENT = 'audio_sent'
    HOLD_RECEIVED = 'hold_received'
    HOLD_CANCELLED = 'hold_cancelled'
    GENERATION_INVALIDATED = 'generation_invalidated'
    AUDIO_STOPPED = 'audio_stopped'
    LATE_AUDIO_DROPPED = 'late_audio_dropped'
    PLAYBACK_ACK = 'playback_ack'
    NEXT_TURN_STARTED = 'next_turn_started'
    STT_STREAM_STARTED = 'stt_stream_started'
    STT_PROVIDER_READY = 'stt_provider_ready'
    STT_FIRST_DELTA = 'stt_first_delta'
    STT_INPUT_COMMITTED = 'stt_input_committed'
    STT_COMPLETED = 'stt_completed'
    STT_CLIENT_OBSERVED = 'stt_client_observed'
    VAD_SPEECH_STARTED = 'vad_speech_started'
    VAD_SPEECH_ENDED = 'vad_speech_ended'


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event: EventName
    run_id: str
    session_id: str
    monotonic_ms: float
    turn_id: str | None = None
    generation_id: str | None = None
    segment_id: str | None = None
    client_event_id: str | None = None
    client_elapsed_ms: float | None = None
    outcome: str | None = None
    metadata: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        data = {
            key: value
            for key, value in asdict(self).items()
            if value is not None and key != 'metadata'
        }
        metadata = _safe_metadata(self.metadata)
        if metadata is not None:
            data['metadata'] = metadata
        return data


@dataclass(frozen=True, slots=True)
class HoldLatencySample:
    run_id: str
    session_id: str
    generation_id: str | None
    invalidation_expected: bool
    hold_monotonic_ms: float
    hold_to_invalidation_ms: float | None
    hold_to_next_turn_ms: float | None
    hold_to_cancellation_ms: float | None
    missing: tuple[str, ...]
    client_event_id: str | None = None
    audio_stop_outcome: str | None = None
    pointer_to_audio_stop_ms: float | None = None
    interaction_mode: str = 'push_to_talk'
    vad_start_to_audio_stop_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SttLatencySample:
    run_id: str
    session_id: str
    client_event_id: str
    stt_mode: str
    outcome: str | None
    stream_to_provider_ready_ms: float | None
    stream_to_first_delta_ms: float | None
    input_to_completed_ms: float | None
    release_to_final_ms: float | None
    missing: tuple[str, ...]
    interaction_mode: str = 'push_to_talk'
    vad_end_to_final_ms: float | None = None
    vad_start_observed: bool = False
    vad_end_observed: bool = False


@dataclass(slots=True)
class _PendingStt:
    index: int
    run_id: str
    session_id: str
    client_event_id: str
    stt_mode: str
    stream_started_ms: float | None = None
    provider_ready_ms: float | None = None
    first_delta_ms: float | None = None
    input_committed_ms: float | None = None
    completed_ms: float | None = None
    outcome: str | None = None
    release_to_final_ms: float | None = None
    interaction_mode: str = 'push_to_talk'
    vad_start_observed: bool = False
    vad_end_observed: bool = False


@dataclass(slots=True)
class _PendingHold:
    index: int
    run_id: str
    session_id: str
    generation_id: str | None
    hold_monotonic_ms: float
    client_event_id: str | None = None
    invalidation_ms: float | None = None
    next_turn_ms: float | None = None
    cancellation_ms: float | None = None
    audio_stop_outcome: str | None = None
    client_elapsed_ms: float | None = None
    interaction_mode: str = 'push_to_talk'


class JsonlEventRecorder:
    """이벤트를 메모리에 모은 뒤 JSONL로 저장한다.

    대화 원문이나 음성 데이터 필드는 제공하지 않는다. 지연과 취소 동작을
    재현하는 데 필요한 식별자와 시각만 남기는 것이 기본 계약이다. `record`는
    파일 I/O를 하지 않아 측정 대상인 WebSocket 흐름에 디스크 지연을 더하지 않는다.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._path = Path(path)
        self._clock = clock
        self._lock = Lock()
        self._flush_lock = Lock()
        self._pending: list[str] = []

    def record(
        self,
        event: EventName,
        *,
        run_id: str,
        session_id: str,
        turn_id: str | None = None,
        generation_id: str | None = None,
        segment_id: str | None = None,
        client_event_id: str | None = None,
        client_elapsed_ms: float | None = None,
        outcome: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RuntimeEvent:
        entry = RuntimeEvent(
            event=event,
            run_id=run_id,
            session_id=session_id,
            monotonic_ms=round(self._clock() * 1000, 3),
            turn_id=turn_id,
            generation_id=generation_id,
            segment_id=segment_id,
            client_event_id=client_event_id,
            client_elapsed_ms=client_elapsed_ms,
            outcome=outcome,
            metadata=metadata,
        )
        line = json.dumps(entry.as_dict(), ensure_ascii=False, separators=(',', ':'))

        with self._lock:
            self._pending.append(f'{line}\n')

        return entry

    def flush(self) -> int:
        """대기 중인 이벤트를 저장하고 기록한 줄 수를 반환한다."""
        with self._flush_lock:
            with self._lock:
                lines, self._pending = self._pending, []
            if not lines:
                return 0
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open('a', encoding='utf-8') as file:
                    file.writelines(lines)
            except Exception:
                with self._lock:
                    self._pending = lines + self._pending
                raise
            return len(lines)


def _finish_sample(pending: _PendingHold) -> HoldLatencySample:
    missing = []
    if pending.generation_id is not None and pending.invalidation_ms is None:
        missing.append(EventName.GENERATION_INVALIDATED.value)
    if pending.next_turn_ms is None and pending.cancellation_ms is None:
        missing.append(EventName.NEXT_TURN_STARTED.value)
    if pending.client_event_id is not None and pending.audio_stop_outcome is None:
        missing.append(EventName.AUDIO_STOPPED.value)
    return HoldLatencySample(
        run_id=pending.run_id,
        session_id=pending.session_id,
        generation_id=pending.generation_id,
        invalidation_expected=pending.generation_id is not None,
        hold_monotonic_ms=pending.hold_monotonic_ms,
        hold_to_invalidation_ms=pending.invalidation_ms,
        hold_to_next_turn_ms=pending.next_turn_ms,
        hold_to_cancellation_ms=pending.cancellation_ms,
        missing=tuple(missing),
        client_event_id=pending.client_event_id,
        audio_stop_outcome=pending.audio_stop_outcome,
        pointer_to_audio_stop_ms=(
            pending.client_elapsed_ms
            if pending.audio_stop_outcome == 'paused'
            and pending.interaction_mode == 'push_to_talk'
            else None
        ),
        interaction_mode=pending.interaction_mode,
        vad_start_to_audio_stop_ms=(
            pending.client_elapsed_ms
            if pending.audio_stop_outcome == 'paused' and pending.interaction_mode == 'vad'
            else None
        ),
    )


def summarize_hold_latencies(path: str | Path) -> list[HoldLatencySample]:
    """완료된 JSONL에서 hold별 서버 측 인계 지연을 복원한다.

    파일 기록 순서를 인과 순서로 사용한다. generation 취소는 같은 세션과
    generation_id가 모두 일치할 때만 연결하고, 다음 응답 시작과 hold
    취소는 같은 세션의 서로 대체하는 종료 이벤트로 취급한다. generation 취소가
    필요한 hold의 종료 이벤트는 generation 무효화보다 이를 수 없다.
    """
    relevant = {
        EventName.SESSION_STARTED.value,
        EventName.HOLD_RECEIVED.value,
        EventName.HOLD_CANCELLED.value,
        EventName.GENERATION_INVALIDATED.value,
        EventName.NEXT_TURN_STARTED.value,
        EventName.AUDIO_STOPPED.value,
    }
    pending_by_session: dict[tuple[str, str], _PendingHold] = {}
    replaced_terminal_generations: set[tuple[str, str, str]] = set()
    finished_by_index: dict[int, HoldLatencySample] = {}
    hold_count = 0
    session_interactions: dict[tuple[str, str], str] = {}

    def finish(pending: _PendingHold) -> None:
        finished_by_index[pending.index] = _finish_sample(pending)

    with Path(path).open(encoding='utf-8') as file:
        for line_number, raw in enumerate(file, start=1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: 잘못된 JSON') from exc
            if not isinstance(data, dict):
                raise ValueError(f'{path}:{line_number}: JSON 객체가 필요합니다')

            event = data.get('event')
            if event not in relevant:
                continue

            run_id = data.get('run_id')
            session_id = data.get('session_id')
            timestamp = data.get('monotonic_ms')
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f'{path}:{line_number}: run_id가 필요합니다')
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f'{path}:{line_number}: session_id가 필요합니다')
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not isfinite(timestamp)
            ):
                raise ValueError(f'{path}:{line_number}: monotonic_ms가 필요합니다')
            timestamp = float(timestamp)
            key = (run_id, session_id)

            if event == EventName.SESSION_STARTED.value:
                metadata = data.get('metadata')
                if isinstance(metadata, dict):
                    session_interactions[key] = metadata.get('interaction_mode', 'push_to_talk')
                continue

            if event == EventName.HOLD_RECEIVED.value:
                previous = pending_by_session.pop(key, None)
                if previous is not None:
                    terminal_seen = (
                        previous.next_turn_ms is not None
                        or previous.cancellation_ms is not None
                    )
                    if (
                        previous.generation_id is not None
                        and previous.invalidation_ms is None
                        and terminal_seen
                    ):
                        replaced_terminal_generations.add(
                            (run_id, session_id, previous.generation_id)
                        )
                    finish(previous)
                generation_id = data.get('generation_id')
                if generation_id is not None and not isinstance(generation_id, str):
                    raise ValueError(f'{path}:{line_number}: generation_id 형식이 잘못됐습니다')
                client_event_id = data.get('client_event_id')
                if client_event_id is not None and (
                    not isinstance(client_event_id, str) or not client_event_id
                ):
                    raise ValueError(f'{path}:{line_number}: client_event_id 형식이 잘못됐습니다')
                pending_by_session[key] = _PendingHold(
                    index=hold_count,
                    run_id=run_id,
                    session_id=session_id,
                    generation_id=generation_id,
                    hold_monotonic_ms=timestamp,
                    client_event_id=client_event_id,
                    interaction_mode=session_interactions.get(key, 'push_to_talk'),
                )
                hold_count += 1
                continue

            generation_id = data.get('generation_id')
            if event == EventName.GENERATION_INVALIDATED.value and not isinstance(
                generation_id, str
            ):
                raise ValueError(f'{path}:{line_number}: generation_id가 필요합니다')
            if (
                event == EventName.GENERATION_INVALIDATED.value
                and (run_id, session_id, generation_id)
                in replaced_terminal_generations
            ):
                raise ValueError(
                    f'{path}:{line_number}: generation 무효화 전에 '
                    'hold 종료 이벤트가 기록됐습니다'
                )

            pending = pending_by_session.get(key)
            if pending is None:
                continue
            if event in {
                EventName.NEXT_TURN_STARTED.value,
                EventName.HOLD_CANCELLED.value,
            }:
                client_event_id = data.get('client_event_id')
                if client_event_id is not None and (
                    not isinstance(client_event_id, str) or not client_event_id
                ):
                    raise ValueError(
                        f'{path}:{line_number}: client_event_id 형식이 잘못됐습니다'
                    )
                if (
                    client_event_id is not None
                    and client_event_id != pending.client_event_id
                ):
                    continue
            delta = round(timestamp - pending.hold_monotonic_ms, 3)
            if delta < 0:
                raise ValueError(f'{path}:{line_number}: hold보다 이른 후속 이벤트입니다')

            if event == EventName.AUDIO_STOPPED.value:
                client_event_id = data.get('client_event_id')
                if not isinstance(client_event_id, str) or not client_event_id:
                    raise ValueError(f'{path}:{line_number}: client_event_id가 필요합니다')
                outcome = data.get('outcome')
                if outcome not in {'paused', 'queued_only', 'idle', 'pause_failed'}:
                    raise ValueError(f'{path}:{line_number}: outcome 형식이 잘못됐습니다')
                elapsed = data.get('client_elapsed_ms')
                if elapsed is not None and (
                    isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or not isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError(f'{path}:{line_number}: client_elapsed_ms 형식이 잘못됐습니다')
                if outcome == 'paused' and elapsed is None:
                    raise ValueError(
                        f'{path}:{line_number}: paused 결과에는 '
                        'client_elapsed_ms가 필요합니다'
                    )
                if (
                    pending.client_event_id == client_event_id
                    and pending.audio_stop_outcome is None
                ):
                    pending.audio_stop_outcome = outcome
                    pending.client_elapsed_ms = (
                        round(float(elapsed), 3) if elapsed is not None else None
                    )
            elif (
                event == EventName.GENERATION_INVALIDATED.value
                and pending.generation_id is not None
                and generation_id == pending.generation_id
                and pending.invalidation_ms is None
            ):
                if pending.next_turn_ms is not None or pending.cancellation_ms is not None:
                    raise ValueError(
                        f'{path}:{line_number}: generation 무효화 전에 '
                        'hold 종료 이벤트가 기록됐습니다'
                    )
                pending.invalidation_ms = delta
            elif (
                event == EventName.NEXT_TURN_STARTED.value
                and pending.next_turn_ms is None
                and pending.cancellation_ms is None
            ):
                pending.next_turn_ms = delta
            elif (
                event == EventName.HOLD_CANCELLED.value
                and pending.cancellation_ms is None
                and pending.next_turn_ms is None
            ):
                pending.cancellation_ms = delta

            terminal_ms = (
                pending.next_turn_ms
                if pending.next_turn_ms is not None
                else pending.cancellation_ms
            )
            if (
                pending.invalidation_ms is not None
                and terminal_ms is not None
                and terminal_ms < pending.invalidation_ms
            ):
                raise ValueError(
                    f'{path}:{line_number}: generation 무효화보다 이른 hold 종료 이벤트입니다'
                )

            invalidation_complete = (
                pending.generation_id is None or pending.invalidation_ms is not None
            )
            if invalidation_complete and terminal_ms is not None:
                finish(pending)
                del pending_by_session[key]

    for pending in pending_by_session.values():
        finish(pending)
    return [finished_by_index[index] for index in sorted(finished_by_index)]


def _latency_distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    """작은 기준 측정에도 쓸 수 있는 선형 보간 지연 분포를 만든다."""
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'p50': None, 'p95': None, 'max': None}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = floor(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 3)

    return {
        'count': len(ordered),
        'p50': percentile(0.5),
        'p95': percentile(0.95),
        'max': round(ordered[-1], 3),
    }


def _stt_values(
    samples: Sequence[SttLatencySample],
    field: str,
) -> list[float]:
    return [
        value
        for sample in samples
        if (value := getattr(sample, field)) is not None
    ]


def build_hold_report(samples: Sequence[HoldLatencySample]) -> dict[str, object]:
    """hold 결과와 지연 분포를 포트폴리오 기준선용 요약으로 만든다.

    실제 재생 중단 지연으로 해석할 수 있는 `paused` 결과만 입력 방식에 맞는
    지연 분포에 포함한다. 다른 outcome은 정상적인 제외 사례로 건수만 남긴다.
    """
    outcomes = Counter(
        sample.audio_stop_outcome
        for sample in samples
        if sample.audio_stop_outcome is not None
    )
    paused = [
        sample.pointer_to_audio_stop_ms
        for sample in samples
        if sample.audio_stop_outcome == 'paused'
        and sample.interaction_mode == 'push_to_talk'
        and sample.pointer_to_audio_stop_ms is not None
    ]
    missing_events = Counter(
        event for sample in samples for event in sample.missing
    )

    def present(field: str) -> list[float]:
        return [
            value
            for sample in samples
            if (value := getattr(sample, field)) is not None
        ]

    return {
        'count': len(samples),
        'audio_stop_outcomes': dict(sorted(outcomes.items())),
        'missing_events': dict(sorted(missing_events.items())),
        'pointer_to_audio_stop_ms': _latency_distribution(paused),
        'vad_start_to_audio_stop_ms': _latency_distribution(
            present('vad_start_to_audio_stop_ms')
        ),
        'interaction_modes': dict(sorted(Counter(
            sample.interaction_mode for sample in samples
        ).items())),
        'hold_to_invalidation_ms': _latency_distribution(
            present('hold_to_invalidation_ms')
        ),
        'hold_to_next_turn_ms': _latency_distribution(present('hold_to_next_turn_ms')),
        'hold_to_cancellation_ms': _latency_distribution(
            present('hold_to_cancellation_ms')
        ),
    }


def summarize_stt_latencies(path: str | Path) -> list[SttLatencySample]:
    """JSONL에서 STT 방식별 지연을 대화 원문 없이 복원한다."""
    session_modes: dict[tuple[str, str], str] = {}
    session_interactions: dict[tuple[str, str], str] = {}
    pending: dict[tuple[str, str, str], _PendingStt] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    relevant = {
        EventName.SESSION_STARTED.value,
        EventName.STT_STREAM_STARTED.value,
        EventName.STT_PROVIDER_READY.value,
        EventName.STT_FIRST_DELTA.value,
        EventName.STT_INPUT_COMMITTED.value,
        EventName.STT_COMPLETED.value,
        EventName.STT_CLIENT_OBSERVED.value,
        EventName.VAD_SPEECH_STARTED.value,
        EventName.VAD_SPEECH_ENDED.value,
    }

    with Path(path).open(encoding='utf-8') as file:
        for line_number, raw in enumerate(file, start=1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: 잘못된 JSON') from exc
            if not isinstance(data, dict) or data.get('event') not in relevant:
                continue
            run_id = data.get('run_id')
            session_id = data.get('session_id')
            timestamp = data.get('monotonic_ms')
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f'{path}:{line_number}: run_id가 필요합니다')
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f'{path}:{line_number}: session_id가 필요합니다')
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not isfinite(timestamp)
            ):
                raise ValueError(f'{path}:{line_number}: monotonic_ms가 필요합니다')
            timestamp = float(timestamp)
            session_key = (run_id, session_id)
            event = data['event']
            if event == EventName.SESSION_STARTED.value:
                metadata = data.get('metadata')
                mode = metadata.get('stt_mode') if isinstance(metadata, dict) else None
                if isinstance(mode, str) and mode:
                    session_modes[session_key] = mode
                if isinstance(metadata, dict):
                    session_interactions[session_key] = metadata.get(
                        'interaction_mode', 'push_to_talk'
                    )
                continue

            client_event_id = data.get('client_event_id')
            if not isinstance(client_event_id, str) or not client_event_id:
                raise ValueError(f'{path}:{line_number}: client_event_id가 필요합니다')
            key = (run_id, session_id, client_event_id)
            item = pending.get(key)
            if item is None:
                item = _PendingStt(
                    index=len(ordered_keys),
                    run_id=run_id,
                    session_id=session_id,
                    client_event_id=client_event_id,
                    stt_mode=session_modes.get(session_key, 'unknown'),
                    interaction_mode=session_interactions.get(session_key, 'push_to_talk'),
                )
                pending[key] = item
                ordered_keys.append(key)

            if event == EventName.STT_STREAM_STARTED.value and item.stream_started_ms is None:
                item.stream_started_ms = timestamp
            elif event == EventName.VAD_SPEECH_STARTED.value:
                item.vad_start_observed = True
            elif event == EventName.VAD_SPEECH_ENDED.value:
                item.vad_end_observed = True
            elif event == EventName.STT_PROVIDER_READY.value and item.provider_ready_ms is None:
                item.provider_ready_ms = timestamp
            elif event == EventName.STT_FIRST_DELTA.value and item.first_delta_ms is None:
                item.first_delta_ms = timestamp
            elif event == EventName.STT_INPUT_COMMITTED.value and item.input_committed_ms is None:
                item.input_committed_ms = timestamp
            elif event == EventName.STT_COMPLETED.value and item.completed_ms is None:
                outcome = data.get('outcome')
                if outcome is not None and not isinstance(outcome, str):
                    raise ValueError(f'{path}:{line_number}: outcome 형식이 잘못됐습니다')
                item.completed_ms = timestamp
                item.outcome = outcome
            elif event == EventName.STT_CLIENT_OBSERVED.value:
                elapsed = data.get('client_elapsed_ms')
                if (
                    isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or not isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError(
                        f'{path}:{line_number}: client_elapsed_ms 형식이 잘못됐습니다'
                    )
                if item.release_to_final_ms is None:
                    item.release_to_final_ms = round(float(elapsed), 3)

    results = []
    for key in ordered_keys:
        item = pending[key]
        missing = []
        if item.input_committed_ms is not None and item.completed_ms is None:
            missing.append(EventName.STT_COMPLETED.value)
        if item.outcome == 'success':
            if item.interaction_mode == 'vad':
                if not item.vad_start_observed:
                    missing.append(EventName.VAD_SPEECH_STARTED.value)
                if not item.vad_end_observed:
                    missing.append(EventName.VAD_SPEECH_ENDED.value)
            if item.input_committed_ms is None:
                missing.append(EventName.STT_INPUT_COMMITTED.value)
            if item.stt_mode == 'streaming_push_to_talk':
                if item.stream_started_ms is None:
                    missing.append(EventName.STT_STREAM_STARTED.value)
                if item.provider_ready_ms is None:
                    missing.append(EventName.STT_PROVIDER_READY.value)
                if item.first_delta_ms is None:
                    missing.append(EventName.STT_FIRST_DELTA.value)
            if item.release_to_final_ms is None:
                missing.append(EventName.STT_CLIENT_OBSERVED.value)

        def delta(
            after: float | None,
            before: float | None,
            client_event_id: str = item.client_event_id,
        ) -> float | None:
            if after is None or before is None:
                return None
            value = round(after - before, 3)
            if value < 0:
                raise ValueError(
                    f'{path}: {client_event_id} STT 이벤트 순서가 잘못됐습니다'
                )
            return value

        results.append(
            SttLatencySample(
                run_id=item.run_id,
                session_id=item.session_id,
                client_event_id=item.client_event_id,
                stt_mode=item.stt_mode,
                outcome=item.outcome,
                stream_to_provider_ready_ms=delta(
                    item.provider_ready_ms,
                    item.stream_started_ms,
                ),
                stream_to_first_delta_ms=delta(
                    item.first_delta_ms,
                    item.stream_started_ms,
                ),
                input_to_completed_ms=delta(
                    item.completed_ms,
                    item.input_committed_ms,
                ),
                release_to_final_ms=(
                    item.release_to_final_ms if item.interaction_mode == 'push_to_talk' else None
                ),
                missing=tuple(missing),
                interaction_mode=item.interaction_mode,
                vad_end_to_final_ms=(
                    item.release_to_final_ms
                    if item.interaction_mode == 'vad' and item.vad_end_observed
                    else None
                ),
                vad_start_observed=item.vad_start_observed,
                vad_end_observed=item.vad_end_observed,
            )
        )
    return results


def build_stt_report(
    samples: Sequence[SttLatencySample],
    *,
    skip_first: int = 0,
) -> dict[str, object]:
    """STT와 입력 방식별 지연 분포를 분리하고 기존 PTT 그룹 키는 유지한다."""
    if skip_first < 0:
        raise ValueError('skip_first는 0 이상이어야 합니다')
    report: dict[str, object] = {}
    groups = sorted({(sample.stt_mode, sample.interaction_mode) for sample in samples})
    for mode, interaction in groups:
        all_selected = [
            sample for sample in samples
            if sample.stt_mode == mode and sample.interaction_mode == interaction
        ]
        skipped = min(skip_first, len(all_selected))
        selected = all_selected[skipped:]
        successful = [sample for sample in selected if sample.outcome == 'success']

        group_key = mode if interaction == 'push_to_talk' else f'{mode}:{interaction}'
        client_metric = (
            'vad_end_to_final_ms' if interaction == 'vad' else 'release_to_final_ms'
        )
        report[group_key] = {
            'interaction_mode': interaction,
            'count': len(selected),
            'total_count': len(all_selected),
            'skipped_warmup_count': skipped,
            'success_count': len(successful),
            'outcomes': dict(
                sorted(
                    Counter(
                        sample.outcome
                        for sample in selected
                        if sample.outcome is not None
                    ).items()
                )
            ),
            'missing_events': dict(
                sorted(Counter(event for sample in selected for event in sample.missing).items())
            ),
            'stream_to_provider_ready_ms': _latency_distribution(
                _stt_values(successful, 'stream_to_provider_ready_ms')
            ),
            'stream_to_first_delta_ms': _latency_distribution(
                _stt_values(successful, 'stream_to_first_delta_ms')
            ),
            'input_to_completed_ms': _latency_distribution(
                _stt_values(successful, 'input_to_completed_ms')
            ),
            client_metric: _latency_distribution(
                _stt_values(successful, client_metric)
            ),
        }
    return report


def build_vad_report(path: str | Path) -> dict[str, object]:
    """VAD 관측 지연을 요약한다. 정답 라벨 없는 품질 비율은 만들지 않는다."""
    holds = [
        sample for sample in summarize_hold_latencies(path)
        if sample.interaction_mode == 'vad'
    ]
    stt = [
        sample for sample in summarize_stt_latencies(path)
        if sample.interaction_mode == 'vad'
    ]
    successful = [sample for sample in stt if sample.outcome == 'success']
    return {
        'interaction_mode': 'vad',
        'hold_count': len(holds),
        'observed_start_count': sum(sample.vad_start_observed for sample in stt),
        'observed_end_count': sum(sample.vad_end_observed for sample in stt),
        'cancelled_hold_count': sum(
            sample.hold_to_cancellation_ms is not None for sample in holds
        ),
        'success_count': len(successful),
        'audio_stop_outcomes': dict(sorted(Counter(
            sample.audio_stop_outcome for sample in holds
            if sample.audio_stop_outcome is not None
        ).items())),
        'stt_outcomes': dict(sorted(Counter(
            sample.outcome for sample in stt if sample.outcome is not None
        ).items())),
        'vad_start_to_audio_stop_ms': _latency_distribution([
            sample.vad_start_to_audio_stop_ms
            for sample in holds if sample.vad_start_to_audio_stop_ms is not None
        ]),
        'vad_end_to_final_ms': _latency_distribution(
            _stt_values(successful, 'vad_end_to_final_ms')
        ),
        'missing_events': dict(sorted(Counter(
            event for sample in [*holds, *stt] for event in sample.missing
        ).items())),
        'false_start_rate': None,
        'mid_utterance_cut_rate': None,
        'quality_note': '오감지와 발화 중간 절단 비율은 실제 녹음의 정답 라벨로 별도 측정해야 합니다.',
    }
