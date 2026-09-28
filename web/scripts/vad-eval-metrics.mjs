const HASH = /^[a-f0-9]{64}$/i;
const TRAILING_TOLERANCE_MS = 32;

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteTime(value, maximum) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= maximum;
}

function validateDuration(durationMs) {
  check(finiteTime(durationMs, Number.MAX_VALUE), 'durationMs must be finite and nonnegative');
}

function validId(value) {
  return typeof value === 'string' && value.length > 0 && value.trim() === value;
}

export function validateLabels(document, { durationMs, audioSha256 }) {
  validateDuration(durationMs);
  check(typeof audioSha256 === 'string' && HASH.test(audioSha256), 'Invalid expected audio SHA-256');
  check(document && typeof document === 'object' && !Array.isArray(document), 'Labels must be an object');
  check(document.schema_version === 1, 'Unsupported labels schema_version');
  check(typeof document.audio_sha256 === 'string' && HASH.test(document.audio_sha256),
    'Labels require a 64-digit audio_sha256');
  check(document.audio_sha256.toLowerCase() === audioSha256.toLowerCase(), 'Labels audio SHA-256 mismatch');
  check(['unreviewed', 'complete'].includes(document.annotation_status), 'Invalid annotation_status');
  check(['unverified', 'recorded', 'synthetic'].includes(document.source_kind), 'Invalid source_kind');
  check(document.annotation_status !== 'complete' || document.source_kind !== 'unverified',
    'Complete labels require recorded or synthetic source_kind');
  check(Array.isArray(document.utterances), 'Labels utterances must be an array');
  const ids = new Set();
  let previousEnd = 0;
  const utterances = document.utterances.map((utterance) => {
    check(utterance && typeof utterance === 'object' && !Array.isArray(utterance), 'Invalid utterance');
    check(validId(utterance.id) && !ids.has(utterance.id), 'Utterance ids must be unique nonempty strings');
    check(finiteTime(utterance.start_ms, durationMs) && finiteTime(utterance.end_ms, durationMs)
      && utterance.start_ms < utterance.end_ms, 'Invalid utterance time bounds');
    check(utterance.start_ms >= previousEnd, 'Utterances must be sorted and nonoverlapping');
    ids.add(utterance.id);
    previousEnd = utterance.end_ms;
    return { id: utterance.id, start_ms: utterance.start_ms, end_ms: utterance.end_ms };
  });
  return {
    schema_version: 1,
    audio_sha256: document.audio_sha256.toLowerCase(),
    annotation_status: document.annotation_status,
    source_kind: document.source_kind,
    utterances,
  };
}

function validateDetections(detections, durationMs) {
  check(Array.isArray(detections), 'Detections must be an array');
  const ids = new Set();
  let previousStart = 0;
  for (const detection of detections) {
    check(detection && typeof detection === 'object', 'Invalid detection');
    check(validId(detection.id) && !ids.has(detection.id), 'Detection ids must be unique nonempty strings');
    check(['silence', 'max_duration', 'eof'].includes(detection.reason), 'Invalid detection reason');
    check(finiteTime(detection.start_ms, durationMs) && detection.start_ms >= previousStart,
      'Detection starts must be bounded and sorted');
    check(detection.audio_start_ms == null || finiteTime(detection.audio_start_ms, detection.start_ms),
      'Invalid detection audio_start_ms');
    check((detection.reason === 'eof' && detection.end_ms === null)
      || (finiteTime(detection.end_ms, durationMs) && detection.end_ms >= detection.start_ms),
    'Invalid detection end_ms');
    ids.add(detection.id);
    previousStart = detection.start_ms;
  }
}

const round = (value) => Math.round(value * 1000) / 1000;
const rate = (numerator, denominator) => denominator ? numerator / denominator : null;

function distribution(values) {
  if (!values.length) return { count: 0, p50: null, p95: null, max: null };
  const sorted = [...values].sort((a, b) => a - b);
  const percentile = (fraction) => {
    const index = (sorted.length - 1) * fraction;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    return round(sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower));
  };
  return { count: sorted.length, p50: percentile(0.5), p95: percentile(0.95), max: round(sorted.at(-1)) };
}

export function evaluateDetections(detections, labelsOrNull, { durationMs }) {
  validateDuration(durationMs);
  validateDetections(detections, durationMs);
  // The runner verifies the hash against the WAV via validateLabels before evaluation.
  const labels = labelsOrNull == null ? null : validateLabels(labelsOrNull, {
    durationMs, audioSha256: labelsOrNull.audio_sha256,
  });
  const report = {
    duration_ms: durationMs,
    detection_count: detections.length,
    unfinished_detection_count: detections.filter((d) => d.reason === 'eof').length,
    cancelled_detection_count: detections.filter((d) => d.reason === 'max_duration').length,
    annotation_status: labels?.annotation_status ?? 'absent',
    source_kind: labels?.source_kind ?? 'unverified',
    matching_policy: {
      timeline: 'audio',
      start_rule: 'decision_inside_utterance_first_then_trailing_tolerance',
      start_trailing_tolerance_ms: TRAILING_TOLERANCE_MS,
      pre_roll_used_for_matching: false,
      onset_detection: 'first',
      endpoint_detection: 'last_associated_detection_only_if_silence_at_or_after_label_end',
      cut_rule: 'any_associated_silence_or_max_duration_end_before_label_end',
      percentile_method: 'linear_interpolation',
    },
    quality_metrics: null,
    utterance_matches: [],
    detection_matches: [],
  };
  if (labels?.annotation_status !== 'complete') return report;

  const associated = new Map(labels.utterances.map((u) => [u.id, []]));
  for (const detection of detections) {
    // Prefer an utterance that is still speaking over a preceding label's tolerance.
    const utterance = labels.utterances.find((u) => u.start_ms <= detection.start_ms && detection.start_ms < u.end_ms)
      ?? labels.utterances.filter((u) => u.end_ms <= detection.start_ms
        && detection.start_ms <= u.end_ms + TRAILING_TOLERANCE_MS).at(-1);
    const matches = utterance ? associated.get(utterance.id) : null;
    report.detection_matches.push({
      detection_id: detection.id,
      utterance_id: utterance?.id ?? null,
      classification: matches ? (matches.length ? 'duplicate' : 'matched') : 'false_start',
    });
    matches?.push(detection);
  }

  for (const utterance of labels.utterances) {
    const matches = associated.get(utterance.id);
    const first = matches[0];
    const last = matches.at(-1);
    report.utterance_matches.push({
      utterance_id: utterance.id,
      detection_ids: matches.map((d) => d.id),
      onset_delay_ms: first ? round(first.start_ms - utterance.start_ms) : null,
      endpoint_delay_ms: last?.reason === 'silence' && last.end_ms >= utterance.end_ms
        ? round(last.end_ms - utterance.end_ms) : null,
      cut: matches.some((d) => d.reason !== 'eof' && d.end_ms < utterance.end_ms),
    });
  }
  const utteranceCount = labels.utterances.length;
  const missed = report.utterance_matches.filter((u) => !u.detection_ids.length).length;
  const falseStarts = report.detection_matches.filter((d) => d.classification === 'false_start').length;
  const duplicates = report.detection_matches.filter((d) => d.classification === 'duplicate').length;
  const cuts = report.utterance_matches.filter((u) => u.cut).length;
  report.quality_metrics = {
    utterance_count: utteranceCount,
    matched_utterance_count: utteranceCount - missed,
    missed_start_count: missed,
    false_start_count: falseStarts,
    duplicate_start_count: duplicates,
    cut_utterance_count: cuts,
    false_start_rate: rate(falseStarts, detections.length),
    missed_start_rate: rate(missed, utteranceCount),
    duplicate_start_rate: rate(duplicates, detections.length),
    mid_utterance_cut_rate: rate(cuts, utteranceCount),
    denominators: {
      false_start_rate: detections.length,
      missed_start_rate: utteranceCount,
      duplicate_start_rate: detections.length,
      mid_utterance_cut_rate: utteranceCount,
    },
    onset_delay_ms: distribution(report.utterance_matches.map((u) => u.onset_delay_ms).filter((v) => v !== null)),
    endpoint_delay_ms: distribution(report.utterance_matches.map((u) => u.endpoint_delay_ms).filter((v) => v !== null)),
  };
  return report;
}
