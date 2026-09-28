import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';
import { argumentsFor, frameAudio, markdownReport, replayProbabilities } from '../scripts/evaluate-vad.mjs';
import { evaluateDetections } from '../scripts/vad-eval-metrics.mjs';

const run = promisify(execFile);
const script = new URL('../scripts/evaluate-vad.mjs', import.meta.url);
const config = globalThis.BanterVadCapture.DEFAULT_CONFIG;
const framesFor = (count) => Array.from({ length: count }, (_, i) => ({
  samples: new Float32Array(512).fill(0.1), endMs: (i + 1) * 32,
}));

function silenceWav() {
  const wav = Buffer.alloc(44 + 32000);
  wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(32000, 40);
  return wav;
}

test('evaluation arguments validate variants without modifying live defaults', () => {
  assert.deepEqual(argumentsFor(['--audio', 'a.wav', '--out', 'run']).endMs, [400, 600, 800]);
  assert.deepEqual(argumentsFor(['--audio', 'a.wav', '--out', 'run', '--end-ms', '500,900']).endMs, [500, 900]);
  assert.equal(config.endMs, 600);
  for (const list of ['400,', 'NaN', '0', '-1', 'Infinity', '400,400', '60001']) {
    assert.throws(() => argumentsFor(['--audio', 'a.wav', '--out', 'run', '--end-ms', list]));
  }
  assert.throws(() => argumentsFor(['--audio', 'a.wav']));
  assert.throws(() => argumentsFor(['--audio', 'a.wav', '--write-label-template', 'a.json', '--out', 'run']));
});

test('resampled timestamps follow consumed WAV samples and expose the unprocessed tail', () => {
  const at48k = frameAudio(new Float32Array(4800).fill(0.25), 48000);
  assert.deepEqual(at48k.frames.map((f) => f.endMs), [32, 64, 96]);
  assert.equal(at48k.droppedTailMs, 4);
  assert.ok(at48k.frames.every((f) => f.samples.every((sample) => sample === 0.25)));
  const at44100 = frameAudio(new Float32Array(3000).fill(0.25), 44100);
  assert.equal(at44100.consumed, 2824);
  assert.ok(Math.abs(at44100.frames[1].endMs - 2824 / 44100 * 1000) < 1e-9);
  assert.ok(Math.abs(at44100.droppedTailMs - 176 / 44100 * 1000) < 1e-9);
});

test('endpoint variants reveal a split without changing model probability input', () => {
  const probabilities = [...Array(5).fill(0.9), ...Array(15).fill(0.1),
    ...Array(5).fill(0.9), ...Array(26).fill(0.1)];
  const frames = framesFor(probabilities.length);
  const short = replayProbabilities(frames, probabilities, { ...config, endMs: 400 });
  const baseline = replayProbabilities(frames, probabilities, config);
  assert.deepEqual(short.detections.map((d) => [d.start_ms, d.end_ms]), [[160, 576], [800, 1216]]);
  assert.deepEqual(baseline.detections.map((d) => [d.start_ms, d.end_ms]), [[160, 1408]]);
  const labels = { schema_version: 1, audio_sha256: 'a'.repeat(64),
    annotation_status: 'complete', source_kind: 'synthetic',
    utterances: [{ id: 'one-intended-utterance', start_ms: 0, end_ms: 800 }] };
  const cut = evaluateDetections(short.detections, labels, { durationMs: 1632 }).quality_metrics;
  const whole = evaluateDetections(baseline.detections, labels, { durationMs: 1632 }).quality_metrics;
  assert.equal(cut.cut_utterance_count, 1);
  assert.equal(cut.duplicate_start_count, 1);
  assert.equal(whole.cut_utterance_count, 0);
  assert.equal(whole.endpoint_delay_ms.p50, 608);
});

test('replay leaves EOF unfinished and records max duration as cancellation', () => {
  const incomplete = replayProbabilities(framesFor(5), Array(5).fill(0.9), config);
  assert.equal(incomplete.final_state, 'speaking');
  assert.equal(incomplete.detections[0].end_ms, null);
  assert.equal(incomplete.detections[0].reason, 'eof');
  const limited = replayProbabilities(framesFor(10), Array(10).fill(0.9), {
    ...config, preRollMs: 0, maxMs: 256,
  });
  assert.equal(limited.final_state, 'suppressed');
  assert.equal(limited.detections[0].reason, 'max_duration');
  assert.equal(limited.detections[0].pcm_samples_24k, 6144);
  assert.throws(() => replayProbabilities(framesFor(2), [0.1], config));
});

test('Markdown keeps missing quality values distinct from zero', () => {
  const evaluation = evaluateDetections([], null, { durationMs: 1000 });
  const report = markdownReport({ audio: { sha256: 'a'.repeat(64) },
    annotation_status: 'absent', source_kind: 'unverified', variants: [{ config,
      effective_frame_thresholds: { end_ms: 608 }, detections: [], evaluation }] });
  assert.match(report, /600 \| 608 \| 0 \| 미측정 \| 미측정/);
  assert.match(report, /실제 대화 지연이 아닙니다/);
});

test('CLI creates hash-bound unreviewed labels and refuses to overwrite them', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'banter-vad-cli-'));
  try {
    const audio = join(directory, 'silence.wav');
    const labels = join(directory, 'labels.json');
    const bytes = silenceWav();
    await writeFile(audio, bytes);
    const args = [script.pathname, '--audio', audio, '--write-label-template', labels];
    await run(process.execPath, args, { cwd: directory });
    const document = JSON.parse(await readFile(labels, 'utf8'));
    assert.equal(document.audio_sha256, createHash('sha256').update(bytes).digest('hex'));
    assert.equal(document.annotation_status, 'unreviewed');
    assert.equal(document.source_kind, 'unverified');
    assert.deepEqual(document.utterances, []);
    await assert.rejects(run(process.execPath, args, { cwd: directory }), /EEXIST/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('CLI rejects labels for another WAV before loading the model or writing results', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'banter-vad-hash-'));
  try {
    const audio = join(directory, 'silence.wav');
    const labels = join(directory, 'labels.json');
    await writeFile(audio, silenceWav());
    await writeFile(labels, JSON.stringify({ schema_version: 1, audio_sha256: 'f'.repeat(64),
      annotation_status: 'complete', source_kind: 'synthetic', utterances: [] }));
    await assert.rejects(run(process.execPath, [script.pathname, '--audio', audio,
      '--labels', labels, '--out', join(directory, 'run')]), (error) => {
      assert.match(error.stderr, /SHA-256 mismatch/);
      assert.doesNotMatch(error.stdout, /Loading VAD/);
      return true;
    });
    await assert.rejects(readFile(join(directory, 'run/report.json')), { code: 'ENOENT' });
  } finally { await rm(directory, { recursive: true, force: true }); }
});
