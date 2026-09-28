import { createHash } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import '../vad-capture.js';
import { decodeWav } from './vad-eval-wav.mjs';
import { evaluateDetections, validateLabels } from './vad-eval-metrics.mjs';

const require = createRequire(import.meta.url);
const { createSegmenter, DEFAULT_CONFIG } = globalThis.BanterVadCapture;
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const MAX_BYTES = 128 * 1024 * 1024;
const HELP = `로컬 녹음 VAD 평가 (외부 API와 마이크 사용 없음)

npm run eval:vad -- --audio recording.wav --write-label-template labels.json
npm run eval:vad -- --audio recording.wav --labels labels.json --end-ms 400,600,800 --out results/run-001

--audio                 PCM 또는 float WAV 파일 (최대 5분)
--write-label-template  음성 해시가 포함된 미검토 라벨 파일 생성 후 종료
--labels                수동 라벨 JSON. 생략하거나 unreviewed이면 품질 수치는 미측정
--end-ms                종료 침묵 설정 목록. 기본 400,600,800; 라이브 설정은 변경하지 않음
--out                   결과를 저장할 새 디렉터리 (기존 디렉터리 덮어쓰기 금지)
--help                  사용법
`;

export function argumentsFor(argv) {
  const { values } = parseArgs({ args: argv, options: {
    audio: { type: 'string' }, labels: { type: 'string' },
    'write-label-template': { type: 'string' }, 'end-ms': { type: 'string' },
    out: { type: 'string' }, help: { type: 'boolean' },
  }, allowPositionals: false });
  if (values.help) return { help: true };
  if (!values.audio?.trim()) throw new Error('--audio에 WAV 파일을 지정해 주세요.');
  if (values['write-label-template']) {
    if (values.out || values.labels || values['end-ms']) {
      throw new Error('라벨 양식 생성과 평가는 별도 명령으로 실행해 주세요.');
    }
    return { audio: values.audio, template: values['write-label-template'] };
  }
  if (!values.out?.trim()) throw new Error('--out에 새 결과 디렉터리를 지정해 주세요.');
  const entries = (values['end-ms'] ?? '400,600,800').split(',');
  if (entries.length > 12 || entries.some((value) => !value.trim())) {
    throw new Error('--end-ms는 비어 있지 않은 숫자 1~12개여야 합니다.');
  }
  const endMs = entries.map(Number);
  if (new Set(endMs).size !== endMs.length) throw new Error('--end-ms에 중복 설정이 있습니다.');
  for (const end of endMs) createSegmenter({ config: { endMs: end } }).destroy();
  return { audio: values.audio, labels: values.labels, out: values.out, endMs };
}

// Match the pinned microphone worklet resampler, including its source consumption.
// At 44.1 kHz a frame consumes 1412 samples; frameIndex * 32 would drift from the WAV.
export function frameAudio(samples, sampleRate) {
  const { Resampler } = require('@ricky0123/vad-web/dist/resampler.js');
  const resampler = new Resampler({
    nativeSampleRate: sampleRate, targetSampleRate: 16000, targetFrameSize: 512,
  });
  const frames = [];
  const sourceSamplesPerFrame = Math.ceil(512 * sampleRate / 16000);
  let consumed = 0;
  for (let offset = 0; offset < samples.length; offset += 4096) {
    for (const frame of resampler.process(samples.subarray(offset, offset + 4096))) {
      consumed += sourceSamplesPerFrame;
      frames.push({ samples: frame, endMs: consumed * 1000 / sampleRate });
    }
  }
  return { frames, consumed, droppedTailMs: (samples.length - consumed) * 1000 / sampleRate };
}

export function replayProbabilities(frames, probabilities, config) {
  if (frames.length !== probabilities.length) throw new Error('프레임과 확률 개수가 다릅니다.');
  const detections = [];
  let active;
  const segmenter = createSegmenter({
    config,
    onStart({ detectedAt }) {
      active = { id: `d${detections.length + 1}`, start_ms: detectedAt, end_ms: null,
        reason: 'eof', pcm_samples_24k: 0, pcm_chunk_count: 0 };
      detections.push(active);
    },
    onChunk(samples) {
      active.pcm_samples_24k += samples.length;
      active.pcm_chunk_count++;
    },
    onEnd({ detectedAt, reason }) {
      active.end_ms = detectedAt;
      active.reason = reason;
      active = undefined;
    },
  });
  try {
    frames.forEach((frame, index) => {
      segmenter.process({ isSpeech: probabilities[index] }, frame.samples, frame.endMs);
    });
    return { detections, final_state: segmenter.getState() };
  } finally {
    // EOF is an observation limit, never an invented silence/commit event.
    segmenter.destroy();
  }
}

async function dependencyInfo() {
  const packages = {};
  for (const [name, expected] of [['@ricky0123/vad-web', '0.0.30'], ['onnxruntime-web', '1.22.0']]) {
    const path = new URL(`../node_modules/${name}/package.json`, import.meta.url);
    const value = JSON.parse(await readFile(path, 'utf8'));
    if (value.version !== expected) throw new Error(`${name} ${expected}가 필요합니다. npm ci --ignore-scripts를 실행해 주세요.`);
    packages[name] = value.version;
  }
  const modelBytes = await readFile(require.resolve('@ricky0123/vad-web/dist/silero_vad_v5.onnx'));
  return { packages, modelBytes };
}

async function infer(frames, modelBytes) {
  const ort = await import('onnxruntime-web');
  const { SileroV5 } = require('@ricky0123/vad-web/dist/models/v5.js');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  const model = await SileroV5.new(ort, async () => new Uint8Array(modelBytes));
  try {
    const probabilities = new Float32Array(frames.length);
    for (let index = 0; index < frames.length; index++) {
      probabilities[index] = (await model.process(frames[index].samples)).isSpeech;
      if (!Number.isFinite(probabilities[index]) || probabilities[index] < 0 || probabilities[index] > 1) {
        throw new Error(`모델이 프레임 ${index + 1}에 잘못된 확률을 반환했습니다.`);
      }
    }
    return probabilities;
  } finally {
    await model.release();
  }
}

function display(value) {
  return value === null || value === undefined ? '미측정' : String(Math.round(value * 100) / 100);
}

export function markdownReport(report) {
  const lines = [
    '# 로컬 녹음 VAD 평가', '',
    `음성 SHA-256: ${report.audio.sha256}`, '',
    `라벨 상태: ${report.annotation_status}. 입력 출처: ${report.source_kind}.`, '',
    '아래 시간은 녹음의 샘플 위치 기준입니다. 추론 실행 시간, 브라우저 재생 중단, STT 또는 실제 대화 지연이 아닙니다.', '',
    '| 종료 침묵 설정 ms | 프레임 기준 ms | 시작 수 | 오감지 수 | 놓친 발화 수 | 중간 절단 발화 수 | 미완료 수 | 취소 수 | 시작 대기 p50 ms | 종료 대기 p50 ms |',
    '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
  ];
  for (const variant of report.variants) {
    const m = variant.evaluation.quality_metrics;
    lines.push(`| ${variant.config.endMs} | ${variant.effective_frame_thresholds.end_ms} | ${variant.detections.length} | ${display(m?.false_start_count)} | ${display(m?.missed_start_count)} | ${display(m?.cut_utterance_count)} | ${variant.evaluation.unfinished_detection_count} | ${variant.evaluation.cancelled_detection_count} | ${display(m?.onset_delay_ms.p50)} | ${display(m?.endpoint_delay_ms.p50)} |`);
  }
  lines.push('', '비율의 분모, p95, 발화별 대응과 개별 감지 구간은 report.json에서 확인합니다.', '',
    '입력 끝에 침묵을 덧붙이지 않습니다. 종료 판정 전에 파일이 끝난 발화는 미완료로 남깁니다.', '',
    '합성 음성이나 무음으로 확인한 결과는 실제 사용자 음성 품질의 근거로 사용하지 않습니다.', '');
  return lines.join('\n');
}

export async function main(argv = process.argv.slice(2)) {
  const args = argumentsFor(argv);
  if (args.help) { console.log(HELP); return; }
  const audioPath = resolve(args.audio);
  const info = await stat(audioPath);
  if (!info.isFile() || info.size > MAX_BYTES) throw new Error('WAV는 128 MiB 이하의 일반 파일이어야 합니다.');
  const bytes = await readFile(audioPath);
  const decoded = decodeWav(bytes);
  const audioHash = sha256(bytes);
  if (args.template) {
    const target = resolve(args.template);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, JSON.stringify({
      schema_version: 1, audio_sha256: audioHash, annotation_status: 'unreviewed',
      source_kind: 'unverified', utterances: [],
    }, null, 2) + '\n', { flag: 'wx' });
    console.log(`미검토 라벨 양식을 저장했습니다: ${target}`);
    console.log('녹음 전체의 의도된 사용자 발화를 표시한 뒤 complete와 입력 출처를 설정해 주세요.');
    return;
  }
  let labels = null;
  let labelsHash = null;
  if (args.labels) {
    const labelBytes = await readFile(resolve(args.labels));
    labelsHash = sha256(labelBytes);
    labels = validateLabels(JSON.parse(labelBytes.toString('utf8')), {
      durationMs: decoded.durationMs, audioSha256: audioHash,
    });
  }
  // Validate everything before inference; do not replace an earlier experiment.
  const output = resolve(args.out);
  try { await stat(output); throw new Error('결과 경로가 이미 있습니다. 새 디렉터리를 지정해 주세요.'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const { packages, modelBytes } = await dependencyInfo();
  const framed = frameAudio(decoded.samples, decoded.sampleRate);
  if (!framed.frames.length) throw new Error('녹음이 VAD 한 프레임(약 32 ms)보다 짧습니다.');
  console.log(`${display(decoded.durationMs / 1000)}초 녹음의 ${framed.frames.length}개 프레임을 로컬 모델로 평가합니다.`);
  const probabilities = await infer(framed.frames, modelBytes);
  const variants = args.endMs.map((endMs) => {
    const config = { ...DEFAULT_CONFIG, endMs };
    const replay = replayProbabilities(framed.frames, probabilities, config);
    return { config, effective_frame_thresholds: {
      start_ms: Math.ceil(config.startMs / 32) * 32,
      end_ms: Math.ceil(config.endMs / 32) * 32,
      pre_roll_ms: Math.floor(config.preRollMs / 32) * 32,
    }, ...replay, evaluation: evaluateDetections(replay.detections, labels, { durationMs: decoded.durationMs }) };
  });
  const report = {
    schema_version: 1, evaluation_kind: 'offline_recording', created_at: new Date().toISOString(),
    annotation_status: labels?.annotation_status ?? 'absent', source_kind: labels?.source_kind ?? 'unverified',
    audio: { name: basename(audioPath), sha256: audioHash, duration_ms: decoded.durationMs,
      sample_rate: decoded.sampleRate, channels: decoded.channels, bits_per_sample: decoded.bitsPerSample,
      channel_policy: decoded.channels === 2 ? 'arithmetic_mean' : 'mono' },
    labels_sha256: labelsHash, labels,
    runtime: { node: process.version, platform: process.platform, arch: process.arch,
      packages, model: 'silero_v5', model_sha256: sha256(modelBytes), execution_provider: 'wasm', threads: 1 },
    implementation_sha256: {
      segmenter: sha256(await readFile(new URL('../vad-capture.js', import.meta.url))),
      evaluator: sha256(await readFile(new URL(import.meta.url))),
      metrics: sha256(await readFile(new URL('./vad-eval-metrics.mjs', import.meta.url))),
      wav_decoder: sha256(await readFile(new URL('./vad-eval-wav.mjs', import.meta.url))),
    },
    timeline: { unit: 'ms', basis: 'consumed_source_samples', nominal_frame_ms: 32,
      frame_count: framed.frames.length, analyzed_until_ms: framed.consumed * 1000 / decoded.sampleRate,
      unprocessed_tail_ms: framed.droppedTailMs, appended_silence_ms: 0 },
    limitations: [
      'These are recording sample positions, not inference time or live latency.',
      'No microphone processing, acoustic echo, browser interruption, STT, LLM, or TTS is evaluated.',
      'Quality metrics require complete human labels; synthetic inputs do not establish user voice quality.',
      'Probabilities are inferred once with continuous model state and reused across endpoint settings.',
    ],
    variants,
  };
  const frameLog = framed.frames.map((frame, index) => JSON.stringify({
    frame_end_ms: frame.endMs, is_speech: probabilities[index],
  })).join('\n') + '\n';
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  await writeFile(resolve(output, 'report.md'), markdownReport(report), { flag: 'wx' });
  await writeFile(resolve(output, 'frames.jsonl'), frameLog, { flag: 'wx' });
  console.log(`결과를 저장했습니다: ${output}`);
  console.log(labels?.annotation_status === 'complete' ? '녹음 라벨 기준 평가이며 실제 대화 성능은 별도 측정해야 합니다.' : '수동 라벨이 완성되지 않아 품질 수치는 미측정으로 남겼습니다.');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(`VAD 평가 실패: ${error.message}`); process.exitCode = 1; });
}
