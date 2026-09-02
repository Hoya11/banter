import json
import sys

import pytest

from api.event_log import (
    EventName,
    HoldLatencySample,
    JsonlEventRecorder,
    build_hold_report,
    summarize_hold_latencies,
)
from scripts.summarize_barge_in import main as summarize_main


def test_jsonl_event_recorder_writes_identifiers_without_conversation_text(tmp_path):
    path = tmp_path / 'run.jsonl'
    recorder = JsonlEventRecorder(path, clock=lambda: 12.345678)

    entry = recorder.record(
        EventName.HOLD_RECEIVED,
        run_id='run-1',
        session_id='session-1',
        turn_id='turn-3',
        generation_id='generation-2',
    )

    assert not path.exists()  # 측정 중에는 디스크 I/O를 하지 않는다
    assert recorder.flush() == 1
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved == {
        'event': 'hold_received',
        'run_id': 'run-1',
        'session_id': 'session-1',
        'monotonic_ms': 12345.678,
        'turn_id': 'turn-3',
        'generation_id': 'generation-2',
    }
    assert entry.monotonic_ms == saved['monotonic_ms']
    assert 'text' not in saved


def test_jsonl_event_recorder_whitelists_scalar_runtime_metadata(tmp_path):
    path = tmp_path / 'run.jsonl'
    recorder = JsonlEventRecorder(path, clock=lambda: 1.0)

    recorder.record(
        EventName.AUDIO_STOPPED,
        run_id='run',
        session_id='session',
        client_event_id='hold-1',
        client_elapsed_ms=1.25,
        outcome='paused',
        metadata={
            'baseline_id': 'push-to-talk-v1',
            'protocol_version': 1,
            'voice_mode': True,
            'radio_sec': 1.5,
            'api_key': 'must-not-be-saved',
            'text': '대화 원문',
            'build_revision': ['not', 'a', 'scalar'],
        },
    )
    recorder.flush()

    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['client_event_id'] == 'hold-1'
    assert saved['client_elapsed_ms'] == 1.25
    assert saved['outcome'] == 'paused'
    assert saved['metadata'] == {
        'baseline_id': 'push-to-talk-v1',
        'protocol_version': 1,
        'voice_mode': True,
        'radio_sec': 1.5,
    }
    assert 'must-not-be-saved' not in path.read_text(encoding='utf-8')
    assert '대화 원문' not in path.read_text(encoding='utf-8')


def test_summarize_hold_latencies_correlates_generation_and_session(tmp_path):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 's1',
            'generation_id': 'g1',
            'monotonic_ms': 100,
        },
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 's2',
            'generation_id': 'g2',
            'monotonic_ms': 105,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 's1',
            'generation_id': 'stale',
            'monotonic_ms': 110,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 's2',
            'generation_id': 'g2',
            'monotonic_ms': 115,
        },
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 's3',
            'monotonic_ms': 120,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 's1',
            'generation_id': 'g1',
            'monotonic_ms': 125,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 's3',
            'generation_id': 'g3',
            'monotonic_ms': 130,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 's1',
            'generation_id': 'g3',
            'monotonic_ms': 350,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    assert summarize_hold_latencies(path) == [
        HoldLatencySample(
            run_id='run',
            session_id='s1',
            generation_id='g1',
            invalidation_expected=True,
            hold_monotonic_ms=100.0,
            hold_to_invalidation_ms=25.0,
            hold_to_next_turn_ms=250.0,
            hold_to_cancellation_ms=None,
            missing=(),
        ),
        HoldLatencySample(
            run_id='run',
            session_id='s2',
            generation_id='g2',
            invalidation_expected=True,
            hold_monotonic_ms=105.0,
            hold_to_invalidation_ms=10.0,
            hold_to_next_turn_ms=None,
            hold_to_cancellation_ms=None,
            missing=('next_turn_started',),
        ),
        HoldLatencySample(
            run_id='run',
            session_id='s3',
            generation_id=None,
            invalidation_expected=False,
            hold_monotonic_ms=120.0,
            hold_to_invalidation_ms=None,
            hold_to_next_turn_ms=10.0,
            hold_to_cancellation_ms=None,
            missing=(),
        ),
    ]


def test_summarize_audio_stop_correlates_client_event_id_and_session(tmp_path):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 's1',
            'generation_id': 'g1',
            'client_event_id': 'hold-1',
            'monotonic_ms': 100,
        },
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 's2',
            'client_event_id': 'hold-1',
            'monotonic_ms': 105,
        },
        {
            'event': 'audio_stopped',
            'run_id': 'run',
            'session_id': 's1',
            'client_event_id': 'another-hold',
            'client_elapsed_ms': 99,
            'outcome': 'paused',
            'monotonic_ms': 108,
        },
        {
            'event': 'audio_stopped',
            'run_id': 'run',
            'session_id': 's2',
            'client_event_id': 'hold-1',
            'client_elapsed_ms': 2,
            'outcome': 'queued_only',
            'monotonic_ms': 110,
        },
        {
            'event': 'audio_stopped',
            'run_id': 'run',
            'session_id': 's1',
            'client_event_id': 'hold-1',
            'client_elapsed_ms': 4.1254,
            'outcome': 'paused',
            'monotonic_ms': 112,
        },
        {
            'event': 'audio_stopped',
            'run_id': 'run',
            'session_id': 's1',
            'client_event_id': 'hold-1',
            'client_elapsed_ms': 1,
            'outcome': 'paused',
            'monotonic_ms': 113,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 's1',
            'generation_id': 'g1',
            'monotonic_ms': 120,
        },
        {
            'event': 'hold_cancelled',
            'run_id': 'run',
            'session_id': 's2',
            'monotonic_ms': 135,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 's1',
            'monotonic_ms': 150,
        },
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'legacy',
            'monotonic_ms': 200,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 'legacy',
            'monotonic_ms': 210,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    by_session = {
        sample.session_id: sample for sample in summarize_hold_latencies(path)
    }
    assert by_session['s1'].audio_stop_outcome == 'paused'
    assert by_session['s1'].client_event_id == 'hold-1'
    assert by_session['s1'].pointer_to_audio_stop_ms == 4.125
    assert by_session['s1'].missing == ()
    assert by_session['s2'].audio_stop_outcome == 'queued_only'
    assert by_session['s2'].pointer_to_audio_stop_ms is None
    assert by_session['s2'].missing == ()
    assert by_session['legacy'].audio_stop_outcome is None
    assert by_session['legacy'].missing == ()


def test_summarize_reports_missing_audio_stop_only_for_identified_hold(tmp_path):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'client_event_id': 'hold-1',
            'monotonic_ms': 100,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 110,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    sample = summarize_hold_latencies(path)[0]
    assert sample.audio_stop_outcome is None
    assert sample.pointer_to_audio_stop_ms is None
    assert sample.missing == ('audio_stopped',)


def test_summarize_ignores_terminal_event_for_another_client_hold(tmp_path):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'client_event_id': 'hold-b',
            'monotonic_ms': 100,
        },
        {
            'event': 'audio_stopped',
            'run_id': 'run',
            'session_id': 'session',
            'client_event_id': 'hold-b',
            'outcome': 'idle',
            'monotonic_ms': 101,
        },
        {
            'event': 'hold_cancelled',
            'run_id': 'run',
            'session_id': 'session',
            'client_event_id': 'hold-a',
            'monotonic_ms': 110,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 'session',
            'client_event_id': 'hold-b',
            'monotonic_ms': 120,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    sample = summarize_hold_latencies(path)[0]
    assert sample.client_event_id == 'hold-b'
    assert sample.hold_to_cancellation_ms is None
    assert sample.hold_to_next_turn_ms == 20.0
    assert sample.missing == ()


def test_summarize_hold_cancelled_is_alternative_terminal(tmp_path):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 100,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 115,
        },
        {
            'event': 'hold_cancelled',
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 125,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 150,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    assert summarize_hold_latencies(path) == [
        HoldLatencySample(
            run_id='run',
            session_id='session',
            generation_id='generation',
            invalidation_expected=True,
            hold_monotonic_ms=100.0,
            hold_to_invalidation_ms=15.0,
            hold_to_next_turn_ms=None,
            hold_to_cancellation_ms=25.0,
            missing=(),
        )
    ]


def test_summarize_cancelled_hold_only_reports_missing_invalidation(tmp_path):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 100,
        },
        {
            'event': 'hold_cancelled',
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 125,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    sample = summarize_hold_latencies(path)[0]
    assert sample.hold_to_cancellation_ms == 25.0
    assert sample.hold_to_next_turn_ms is None
    assert sample.missing == ('generation_invalidated',)


@pytest.mark.parametrize('terminal_event', ['next_turn_started', 'hold_cancelled'])
def test_summarize_rejects_terminal_timestamp_before_generation_invalidation(
    tmp_path, terminal_event
):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 100,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 120,
        },
        {
            'event': terminal_event,
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 110,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    with pytest.raises(ValueError, match='generation 무효화보다 이른'):
        summarize_hold_latencies(path)


@pytest.mark.parametrize('terminal_event', ['next_turn_started', 'hold_cancelled'])
def test_summarize_rejects_terminal_file_order_before_invalidation(
    tmp_path, terminal_event
):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 100,
        },
        {
            'event': terminal_event,
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 130,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation',
            'monotonic_ms': 120,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    with pytest.raises(ValueError, match='generation 무효화 전에'):
        summarize_hold_latencies(path)


@pytest.mark.parametrize('terminal_event', ['next_turn_started', 'hold_cancelled'])
def test_summarize_rejects_late_invalidation_after_terminal_hold_is_replaced(
    tmp_path, terminal_event
):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation-1',
            'monotonic_ms': 100,
        },
        {
            'event': terminal_event,
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 130,
        },
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation-2',
            'monotonic_ms': 140,
        },
        {
            'event': 'generation_invalidated',
            'run_id': 'run',
            'session_id': 'session',
            'generation_id': 'generation-1',
            'monotonic_ms': 150,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')

    with pytest.raises(ValueError, match='generation 무효화 전에'):
        summarize_hold_latencies(path)


def _report_sample(
    index,
    outcome,
    pointer_ms=None,
    invalidation_ms=None,
    next_turn_ms=None,
    cancellation_ms=None,
):
    return HoldLatencySample(
        run_id='run',
        session_id=f'session-{index}',
        generation_id=f'generation-{index}' if invalidation_ms is not None else None,
        invalidation_expected=invalidation_ms is not None,
        hold_monotonic_ms=float(index),
        hold_to_invalidation_ms=invalidation_ms,
        hold_to_next_turn_ms=next_turn_ms,
        hold_to_cancellation_ms=cancellation_ms,
        missing=(),
        audio_stop_outcome=outcome,
        pointer_to_audio_stop_ms=pointer_ms,
    )


def test_build_hold_report_counts_all_outcomes_but_times_only_paused():
    samples = [
        _report_sample(1, 'paused', 1.0, 10.0, 100.0),
        _report_sample(2, 'paused', 3.0, 30.0, 200.0),
        _report_sample(3, 'queued_only', next_turn_ms=150.0),
        _report_sample(4, 'idle', cancellation_ms=20.0),
        _report_sample(5, 'pause_failed', cancellation_ms=40.0),
    ]

    report = build_hold_report(samples)

    assert report['count'] == 5
    assert report['audio_stop_outcomes'] == {
        'idle': 1,
        'pause_failed': 1,
        'paused': 2,
        'queued_only': 1,
    }
    assert report['missing_events'] == {}
    assert report['pointer_to_audio_stop_ms'] == {
        'count': 2,
        'p50': 2.0,
        'p95': 2.9,
        'max': 3.0,
    }
    assert report['hold_to_invalidation_ms'] == {
        'count': 2,
        'p50': 20.0,
        'p95': 29.0,
        'max': 30.0,
    }
    assert report['hold_to_next_turn_ms'] == {
        'count': 3,
        'p50': 150.0,
        'p95': 195.0,
        'max': 200.0,
    }
    assert report['hold_to_cancellation_ms'] == {
        'count': 2,
        'p50': 30.0,
        'p95': 39.0,
        'max': 40.0,
    }


def test_summary_cli_outputs_report_and_samples(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'run.jsonl'
    events = [
        {
            'event': 'hold_received',
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 100,
        },
        {
            'event': 'next_turn_started',
            'run_id': 'run',
            'session_id': 'session',
            'monotonic_ms': 110,
        },
    ]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['summarize_barge_in.py', str(path)])

    summarize_main()

    output = json.loads(capsys.readouterr().out)
    assert output['summary']['count'] == 1
    assert output['samples'][0]['session_id'] == 'session'


def test_build_hold_report_counts_missing_events():
    sample = HoldLatencySample(
        run_id='run',
        session_id='session',
        generation_id='generation',
        invalidation_expected=True,
        hold_monotonic_ms=1.0,
        hold_to_invalidation_ms=None,
        hold_to_next_turn_ms=None,
        hold_to_cancellation_ms=None,
        missing=('generation_invalidated', 'next_turn_started', 'audio_stopped'),
        client_event_id='hold-1',
    )

    assert build_hold_report([sample])['missing_events'] == {
        'audio_stopped': 1,
        'generation_invalidated': 1,
        'next_turn_started': 1,
    }
