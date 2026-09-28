import json
import sys

from api.event_log import (
    build_hold_report,
    build_stt_report,
    build_vad_report,
    summarize_hold_latencies,
    summarize_stt_latencies,
)
from scripts.summarize_vad import main


def _events(interaction, *, with_end=True):
    def event(name, at, **kwargs):
        return {
            'event': name, 'run_id': 'run', 'session_id': interaction,
            'client_event_id': 'capture-1', 'monotonic_ms': at, **kwargs,
        }

    result = [
        event('session_started', 0, metadata={
            'interaction_mode': interaction,
            'stt_mode': 'streaming_push_to_talk',
        }),
        event('hold_received', 10),
        event('audio_stopped', 11, outcome='paused', client_elapsed_ms=4),
    ]
    if interaction == 'vad':
        result.append(event('vad_speech_started', 12))
    result.extend([
        event('stt_stream_started', 15),
        event('stt_provider_ready', 25),
        event('stt_first_delta', 100),
    ])
    if interaction == 'vad' and with_end:
        result.append(event('vad_speech_ended', 110))
    result.extend([
        event('stt_input_committed', 115),
        event('stt_completed', 200, outcome='success'),
        event('stt_client_observed', 205, client_elapsed_ms=95),
        event('next_turn_started', 210),
    ])
    return result


def _write(path, events):
    path.write_text('\n'.join(json.dumps(event) for event in events))


def test_mixed_logs_keep_vad_out_of_pointer_and_release_metrics(tmp_path):
    path = tmp_path / 'events.jsonl'
    _write(path, [*_events('push_to_talk'), *_events('vad')])
    holds = summarize_hold_latencies(path)
    stt = summarize_stt_latencies(path)
    assert holds[0].pointer_to_audio_stop_ms == 4
    assert holds[0].vad_start_to_audio_stop_ms is None
    assert holds[1].pointer_to_audio_stop_ms is None
    assert holds[1].vad_start_to_audio_stop_ms == 4
    hold_report = build_hold_report(holds)
    assert hold_report['pointer_to_audio_stop_ms']['count'] == 1
    assert hold_report['vad_start_to_audio_stop_ms']['count'] == 1
    assert stt[0].release_to_final_ms == 95
    assert stt[0].vad_end_to_final_ms is None
    assert stt[1].release_to_final_ms is None
    assert stt[1].vad_end_to_final_ms == 95
    report = build_stt_report(stt)
    assert report['streaming_push_to_talk']['release_to_final_ms']['count'] == 1
    assert report['streaming_push_to_talk:vad']['vad_end_to_final_ms']['count'] == 1
    assert 'release_to_final_ms' not in report['streaming_push_to_talk:vad']
    vad = build_vad_report(path)
    assert vad['hold_count'] == 1
    assert vad['observed_start_count'] == 1
    assert vad['observed_end_count'] == 1
    assert vad['audio_stop_outcomes'] == {'paused': 1}
    assert vad['stt_outcomes'] == {'success': 1}
    assert vad['false_start_rate'] is None
    assert vad['mid_utterance_cut_rate'] is None
    assert vad['missing_events'] == {}


def test_missing_end_is_not_a_valid_vad_end_to_final_measurement(tmp_path):
    path = tmp_path / 'events.jsonl'
    _write(path, _events('vad', with_end=False))
    sample = summarize_stt_latencies(path)[0]
    assert sample.vad_end_to_final_ms is None
    assert sample.release_to_final_ms is None
    assert 'vad_speech_ended' in sample.missing
    assert build_vad_report(path)['vad_end_to_final_ms']['count'] == 0


def test_vad_summary_distinguishes_idle_empty_from_pause_and_provider_failures(tmp_path):
    reports = []
    for stop_outcome, stt_outcome in [('idle', 'empty'), ('pause_failed', 'error')]:
        events = []
        for event in _events('vad'):
            if event['event'] == 'audio_stopped':
                event['outcome'] = stop_outcome
                event.pop('client_elapsed_ms')
            elif event['event'] == 'stt_completed':
                event['outcome'] = stt_outcome
            elif event['event'] == 'stt_client_observed':
                continue
            elif event['event'] == 'next_turn_started':
                event['event'] = 'hold_cancelled'
            events.append(event)
        path = tmp_path / f'{stop_outcome}.jsonl'
        _write(path, events)
        report = build_vad_report(path)
        assert report['audio_stop_outcomes'] == {stop_outcome: 1}
        assert report['stt_outcomes'] == {stt_outcome: 1}
        assert report['cancelled_hold_count'] == 1
        assert report['success_count'] == 0
        assert report['vad_start_to_audio_stop_ms']['count'] == 0
        assert report['vad_end_to_final_ms']['count'] == 0
        assert report['missing_events'] == {}
        reports.append(report)
    assert reports[0] != reports[1]


def test_missing_vad_outcomes_are_not_invented(tmp_path):
    path = tmp_path / 'missing-outcomes.jsonl'
    _write(path, [
        event for event in _events('vad')
        if event['event'] not in {'audio_stopped', 'stt_completed', 'stt_client_observed'}
    ])
    report = build_vad_report(path)
    assert report['audio_stop_outcomes'] == {}
    assert report['stt_outcomes'] == {}
    assert report['missing_events'] == {'audio_stopped': 1, 'stt_completed': 1}
    assert report['false_start_rate'] is None
    assert report['mid_utterance_cut_rate'] is None


def test_vad_cli_does_not_invent_quality_rates(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'events.jsonl'
    _write(path, _events('vad'))
    monkeypatch.setattr(sys, 'argv', ['summarize_vad.py', str(path)])
    main()
    output = json.loads(capsys.readouterr().out)
    assert output['vad_start_to_audio_stop_ms']['p50'] == 4
    assert output['vad_end_to_final_ms']['p50'] == 95
    assert output['audio_stop_outcomes'] == {'paused': 1}
    assert output['stt_outcomes'] == {'success': 1}
    assert output['false_start_rate'] is None
    assert output['mid_utterance_cut_rate'] is None
