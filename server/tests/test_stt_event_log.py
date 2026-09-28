import json
from dataclasses import replace

from api.event_log import build_stt_report, summarize_stt_latencies


def _write_events(path, events):
    path.write_text(
        ''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in events),
        encoding='utf-8',
    )


def test_stt_report_compares_file_and_streaming_modes(tmp_path):
    path = tmp_path / 'events.jsonl'
    _write_events(
        path,
        [
            {
                'event': 'session_started',
                'run_id': 'run-1',
                'session_id': 'file-session',
                'monotonic_ms': 0,
                'metadata': {'stt_mode': 'record_then_transcribe'},
            },
            {
                'event': 'stt_input_committed',
                'run_id': 'run-1',
                'session_id': 'file-session',
                'client_event_id': 'file-1',
                'monotonic_ms': 100,
            },
            {
                'event': 'stt_completed',
                'run_id': 'run-1',
                'session_id': 'file-session',
                'client_event_id': 'file-1',
                'monotonic_ms': 500,
                'outcome': 'success',
            },
            {
                'event': 'stt_client_observed',
                'run_id': 'run-1',
                'session_id': 'file-session',
                'client_event_id': 'file-1',
                'monotonic_ms': 520,
                'client_elapsed_ms': 430.5,
            },
            {
                'event': 'session_started',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'monotonic_ms': 0,
                'metadata': {'stt_mode': 'streaming_push_to_talk'},
            },
            {
                'event': 'stt_stream_started',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'client_event_id': 'stream-1',
                'monotonic_ms': 50,
            },
            {
                'event': 'stt_provider_ready',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'client_event_id': 'stream-1',
                'monotonic_ms': 90,
            },
            {
                'event': 'stt_first_delta',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'client_event_id': 'stream-1',
                'monotonic_ms': 180,
            },
            {
                'event': 'stt_input_committed',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'client_event_id': 'stream-1',
                'monotonic_ms': 300,
            },
            {
                'event': 'stt_completed',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'client_event_id': 'stream-1',
                'monotonic_ms': 390,
                'outcome': 'success',
            },
            {
                'event': 'stt_client_observed',
                'run_id': 'run-1',
                'session_id': 'stream-session',
                'client_event_id': 'stream-1',
                'monotonic_ms': 400,
                'client_elapsed_ms': 105.25,
            },
        ],
    )

    samples = summarize_stt_latencies(path)
    assert len(samples) == 2
    stream = samples[1]
    assert stream.stream_to_provider_ready_ms == 40
    assert stream.stream_to_first_delta_ms == 130
    assert stream.input_to_completed_ms == 90
    assert stream.release_to_final_ms == 105.25

    report = build_stt_report(samples)
    assert report['record_then_transcribe']['success_count'] == 1
    assert report['streaming_push_to_talk']['success_count'] == 1
    assert report['record_then_transcribe']['release_to_final_ms']['p50'] == 430.5
    assert report['streaming_push_to_talk']['release_to_final_ms']['p50'] == 105.25

    with_warmups = [
        samples[0],
        replace(samples[0], client_event_id='file-2', release_to_final_ms=400),
        samples[1],
        replace(samples[1], client_event_id='stream-2', release_to_final_ms=100),
    ]
    report = build_stt_report(with_warmups, skip_first=1)
    assert report['record_then_transcribe']['count'] == 1
    assert report['record_then_transcribe']['total_count'] == 2
    assert report['record_then_transcribe']['skipped_warmup_count'] == 1
    assert report['record_then_transcribe']['release_to_final_ms']['p50'] == 400
    assert report['streaming_push_to_talk']['release_to_final_ms']['p50'] == 100


def test_stt_report_marks_missing_client_observation(tmp_path):
    path = tmp_path / 'events.jsonl'
    _write_events(
        path,
        [
            {
                'event': 'session_started',
                'run_id': 'run-1',
                'session_id': 'session-1',
                'monotonic_ms': 0,
                'metadata': {'stt_mode': 'streaming_push_to_talk'},
            },
            {
                'event': 'stt_input_committed',
                'run_id': 'run-1',
                'session_id': 'session-1',
                'client_event_id': 'voice-1',
                'monotonic_ms': 10,
            },
            {
                'event': 'stt_completed',
                'run_id': 'run-1',
                'session_id': 'session-1',
                'client_event_id': 'voice-1',
                'monotonic_ms': 20,
                'outcome': 'success',
            },
        ],
    )

    sample = summarize_stt_latencies(path)[0]
    assert sample.stt_mode == 'streaming_push_to_talk'
    assert sample.missing == (
        'stt_stream_started',
        'stt_provider_ready',
        'stt_first_delta',
        'stt_client_observed',
    )
    assert build_stt_report([sample])['streaming_push_to_talk']['missing_events'] == {
        'stt_client_observed': 1,
        'stt_first_delta': 1,
        'stt_provider_ready': 1,
        'stt_stream_started': 1,
    }


def test_stt_report_never_needs_transcript_or_audio_fields(tmp_path):
    path = tmp_path / 'events.jsonl'
    _write_events(
        path,
        [
            {
                'event': 'stt_input_committed',
                'run_id': 'run-1',
                'session_id': 'session-1',
                'client_event_id': 'voice-1',
                'monotonic_ms': 10,
            },
            {
                'event': 'stt_completed',
                'run_id': 'run-1',
                'session_id': 'session-1',
                'client_event_id': 'voice-1',
                'monotonic_ms': 20,
                'outcome': 'error',
            }
        ],
    )

    sample = summarize_stt_latencies(path)[0]
    assert sample.outcome == 'error'
    assert sample.release_to_final_ms is None
    report = build_stt_report([sample])['unknown']
    assert report['success_count'] == 0
    assert report['input_to_completed_ms']['count'] == 0
