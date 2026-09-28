import assert from 'node:assert/strict';
import test from 'node:test';
import { evaluateDetections, validateLabels } from '../scripts/vad-eval-metrics.mjs';

const audioSha256 = 'a'.repeat(64);
const durationMs = 5000;
const options = { durationMs, audioSha256 };
const utterance = (id, start_ms, end_ms) => ({ id, start_ms, end_ms });
const labels = (utterances = [], extras = {}) => ({
  schema_version: 1, audio_sha256: audioSha256,
  annotation_status: 'complete', source_kind: 'recorded', utterances, ...extras,
});
const detection = (id, start_ms, end_ms, reason = 'silence', audio_start_ms = 0) => (
  { id, start_ms, end_ms, reason, audio_start_ms }
);

test('validates complete labels against the exact audio and preserves input', () => {
  const document = labels([utterance('u1', 0, 1000), utterance('u2', 1000, 2000)]);
  const validated = validateLabels(document, options);
  assert.deepEqual(validated, document);
  assert.notEqual(validated.utterances, document.utterances);
  assert.throws(() => validateLabels(document, { ...options, audioSha256: 'b'.repeat(64) }), /mismatch/);
});

test('rejects invalid schema, status, provenance, bounds, duplicate ids and overlapping labels', () => {
  const invalid = [
    null, [], labels([], { schema_version: 2 }), labels([], { audio_sha256: 'bad' }),
    labels([], { annotation_status: 'partial' }), labels([], { source_kind: 'unknown' }),
    labels([], { source_kind: 'unverified' }), labels(null),
    labels([utterance('', 0, 100)]), labels([utterance('u1', 0, 0)]),
    labels([utterance('u1', -1, 100)]), labels([utterance('u1', 0, 5001)]),
    labels([utterance('u1', NaN, 100)]), labels([utterance('u1', 0, Infinity)]),
    labels([utterance('u1', '0', 100)]),
    labels([utterance('u1', 0, 100), utterance('u1', 200, 300)]),
    labels([utterance('u1', 100, 200), utterance('u2', 0, 50)]),
    labels([utterance('u1', 0, 200), utterance('u2', 100, 300)]),
  ];
  for (const document of invalid) assert.throws(() => validateLabels(document, options));
  assert.throws(() => validateLabels(labels(), { ...options, durationMs: NaN }));
});

test('absent and unreviewed labels never generate quality rates or associations', () => {
  const detections = [detection('d1', 160, 1000)];
  for (const document of [null, labels([utterance('u1', 0, 800)], {
    annotation_status: 'unreviewed', source_kind: 'unverified',
  })]) {
    const report = evaluateDetections(detections, document, options);
    assert.equal(report.detection_count, 1);
    assert.equal(report.quality_metrics, null);
    assert.deepEqual(report.utterance_matches, []);
    assert.deepEqual(report.detection_matches, []);
  }
});

test('pre-roll origin may be unavailable without inventing a timestamp', () => {
  for (const audioStart of [undefined, null]) {
    const event = detection('d1', 160, 1400);
    if (audioStart === undefined) delete event.audio_start_ms;
    else event.audio_start_ms = audioStart;
    const report = evaluateDetections([event], labels([utterance('u1', 0, 800)]), options);
    assert.equal(report.quality_metrics.matched_utterance_count, 1);
    assert.equal(report.matching_policy.pre_roll_used_for_matching, false);
  }
});

test('noise starts stay false despite pre-roll overlap and undetected utterances stay missed', () => {
  const report = evaluateDetections([
    detection('noise', 400, 550, 'silence', 0),
    detection('speech', 1160, 2000, 'silence', 800),
  ], labels([utterance('u1', 0, 300), utterance('u2', 1000, 1500)]), options);
  const quality = report.quality_metrics;
  assert.equal(quality.false_start_count, 1);
  assert.equal(quality.missed_start_count, 1);
  assert.equal(quality.false_start_rate, 0.5);
  assert.equal(quality.missed_start_rate, 0.5);
  assert.deepEqual(quality.denominators, {
    false_start_rate: 2, missed_start_rate: 2, duplicate_start_rate: 2, mid_utterance_cut_rate: 2,
  });
  assert.deepEqual(report.detection_matches.map((d) => d.classification), ['false_start', 'matched']);
});

test('split detections distinguish duplicate starts and count a cut only once per utterance', () => {
  const report = evaluateDetections([
    detection('d1', 160, 400), detection('d2', 500, 700), detection('d3', 800, 1600),
  ], labels([utterance('u1', 0, 1000)]), options);
  const quality = report.quality_metrics;
  assert.equal(quality.false_start_count, 0);
  assert.equal(quality.duplicate_start_count, 2);
  assert.equal(quality.duplicate_start_rate, 2 / 3);
  assert.equal(quality.cut_utterance_count, 1);
  assert.equal(quality.mid_utterance_cut_rate, 1);
  assert.deepEqual(quality.onset_delay_ms, { count: 1, p50: 160, p95: 160, max: 160 });
  assert.deepEqual(quality.endpoint_delay_ms, { count: 1, p50: 600, p95: 600, max: 600 });
  assert.deepEqual(report.utterance_matches[0].detection_ids, ['d1', 'd2', 'd3']);
});

test('a detection spanning two utterances does not invent a later onset', () => {
  const report = evaluateDetections([detection('d1', 160, 2600)], labels([
    utterance('u1', 0, 800), utterance('u2', 1400, 2000),
  ]), options);
  assert.equal(report.quality_metrics.matched_utterance_count, 1);
  assert.equal(report.quality_metrics.missed_start_count, 1);
  assert.equal(report.utterance_matches[0].endpoint_delay_ms, 1800);
  assert.deepEqual(report.utterance_matches[1].detection_ids, []);
});

test('32 ms trailing tolerance is explicit and current speech wins over the preceding tolerance', () => {
  const report = evaluateDetections([
    detection('at-end', 100, 150), detection('at-limit', 132, 160),
    detection('outside', 133, 170), detection('next', 315, 500),
  ], labels([
    utterance('u1', 0, 100), utterance('u2', 200, 300), utterance('u3', 310, 400),
  ]), options);
  assert.equal(report.matching_policy.start_trailing_tolerance_ms, 32);
  assert.deepEqual(report.detection_matches.map((d) => d.utterance_id), ['u1', 'u1', null, 'u3']);
  const justEnded = evaluateDetections([detection('d1', 40, 100)], labels([
    utterance('u1', 0, 10), utterance('u2', 20, 30),
  ]), options);
  assert.equal(justEnded.detection_matches[0].utterance_id, 'u2');
});

test('maximum duration cancellation and unfinished EOF never become valid endpoint samples', () => {
  const report = evaluateDetections([
    detection('cut', 160, 400, 'max_duration'),
    detection('late-cancel', 1160, 1800, 'max_duration'),
    detection('unfinished', 2160, null, 'eof'),
  ], labels([
    utterance('u1', 0, 800), utterance('u2', 1000, 1500), utterance('u3', 2000, 3000),
  ]), options);
  assert.equal(report.cancelled_detection_count, 2);
  assert.equal(report.unfinished_detection_count, 1);
  assert.equal(report.quality_metrics.cut_utterance_count, 1);
  assert.deepEqual(report.quality_metrics.endpoint_delay_ms, { count: 0, p50: null, p95: null, max: null });
});

test('complete empty labels represent a reviewed clip with no target utterances', () => {
  const empty = evaluateDetections([], labels(), options).quality_metrics;
  assert.equal(empty.false_start_rate, null);
  assert.equal(empty.missed_start_rate, null);
  assert.equal(empty.duplicate_start_rate, null);
  assert.equal(empty.mid_utterance_cut_rate, null);
  const noise = evaluateDetections([detection('noise', 160, 800)], labels(), options).quality_metrics;
  assert.equal(noise.false_start_count, 1);
  assert.equal(noise.false_start_rate, 1);
  assert.equal(noise.missed_start_rate, null);
});

test('hand calculated delays use linear percentile interpolation and only first matched onsets', () => {
  const report = evaluateDetections([
    detection('d1', 100, 600), detection('d2', 1200, 1700), detection('d3', 2400, 2900),
  ], labels([
    utterance('u1', 0, 500), utterance('u2', 1000, 1500), utterance('u3', 2000, 2500),
  ], { source_kind: 'synthetic' }), options);
  const expected = { count: 3, p50: 200, p95: 380, max: 400 };
  assert.deepEqual(report.quality_metrics.onset_delay_ms, expected);
  assert.deepEqual(report.quality_metrics.endpoint_delay_ms, expected);
  assert.equal(report.source_kind, 'synthetic');
});

test('malformed detection timelines fail before evaluation', () => {
  for (const detections of [
    [detection('d1', -1, 100)], [detection('d1', 160, 100)],
    [detection('d1', 160, null)], [detection('d1', 160, 6000)],
    [detection('d1', 160, 800, 'silence', 161)],
    [detection('d1', 160, 800, 'unknown')],
    [detection('d1', 160, 800), detection('d1', 1000, 1800)],
    [detection('d1', 1000, 1800), detection('d2', 160, 800)],
  ]) assert.throws(() => evaluateDetections(detections, labels(), options));
});
