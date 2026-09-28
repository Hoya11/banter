import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const code = await readFile(new URL('../vad-capture.js', import.meta.url), 'utf8');
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
};

async function setup(settings = {}) {
  const events = [];
  const scripts = [];
  const probabilities = new WeakMap();
  let current = true;
  let frameTime = 0;
  let options;
  let released = 0;
  let stopped = 0;
  let disconnected = 0;
  let activeRuns = 0;
  let maximumRuns = 0;
  const endedListeners = new Set();
  const track = {
    readyState: settings.endedOnGrant ? 'ended' : 'live',
    stop() { stopped++; this.readyState = 'ended'; },
    addEventListener(type, listener) {
      if (type === 'ended') endedListeners.add(listener);
    },
    removeEventListener(type, listener) {
      if (type === 'ended') endedListeners.delete(listener);
    },
    end() {
      this.readyState = 'ended';
      for (const listener of [...endedListeners]) listener({ type: 'ended' });
    },
  };
  const stream = { getTracks: () => [track] };
  const source = {
    frameProcessor: { audioBuffer: [] },
    model: { async release() { released++; } },
    _vadNode: { disconnect() { disconnected++; }, port: { close() {} } },
    _mediaStreamAudioSourceNode: { disconnect() { disconnected++; } },
    async start() {
      if (settings.getStream) await options.getStream();
      settings.startEntered?.resolve();
      if (settings.startGate) await settings.startGate.promise;
      if (settings.startError) throw settings.startError;
    },
    async destroy() {
      if (settings.partial) throw new Error('MicVAD has null stream');
      await options.pauseStream(stream);
      await this.model.release();
    },
    async processFrame(frame) {
      activeRuns++;
      maximumRuns = Math.max(maximumRuns, activeRuns);
      try {
        if (settings.frameGate) await settings.frameGate.promise;
        if (settings.frameError) throw settings.frameError;
        frameTime += 32;
        options.onFrameProcessed({ isSpeech: probabilities.get(frame) }, frame);
        this.frameProcessor.audioBuffer.push(frame);
      } finally { activeRuns--; }
    },
  };
  const context = vm.createContext({
    Float32Array, Error, performance: { now: () => frameTime },
    document: {
      createElement: () => ({ remove() {} }),
      head: { appendChild(element) { scripts.push(element.src); queueMicrotask(element.onload); } },
    },
    navigator: { mediaDevices: { async getUserMedia(constraints) {
      assert.equal(constraints.audio.echoCancellation, true);
      if (settings.permissionGate) return settings.permissionGate.promise;
      return stream;
    } } },
    vad: { MicVAD: { async new(value) {
      options = value;
      if (settings.createGate) await settings.createGate.promise;
      return source;
    } } },
  });
  vm.runInContext(code, context);
  const creating = context.BanterVadCapture.create({
    audioContext: {}, isCurrent: () => current,
    onStart(value) { events.push({ type: 'start', ...value }); settings.onStart?.(); },
    onChunk(samples) { events.push({ type: 'chunk', samples }); settings.onChunk?.(); },
    onEnd(value) { events.push({ type: 'end', ...value }); },
    onError(error) { events.push({ type: 'error', error }); },
  });
  const send = (probability, samples = new Float32Array(512).fill(0.25)) => {
    probabilities.set(samples, probability);
    return source.processFrame(samples);
  };
  const api = {
    creating, events, scripts, source, stream, track, endedListeners, send,
    stale() { current = false; },
    async frames(probability, count) { for (let i = 0; i < count; i++) await send(probability); },
    counts: () => ({ released, stopped, disconnected, maximumRuns }),
    options: () => options,
    chunks: () => events.filter((e) => e.type === 'chunk').map((e) => e.samples),
  };
  if (settings.createGate) return api;
  api.controller = await creating;
  if (!settings.noStart) await api.controller.start();
  return api;
}

test('loads pinned local runtime only and defers microphone startup', async () => {
  const h = await setup({ noStart: true });
  assert.deepEqual(h.scripts, ['/vad-assets/ort.wasm.min.js', '/vad-assets/vad.bundle.min.js']);
  const o = h.options();
  assert.equal(o.model, 'v5');
  assert.equal(o.startOnLoad, false);
  assert.equal(o.processorType, 'AudioWorklet');
  assert.equal(o.baseAssetPath, '/vad-assets/');
  assert.equal(o.onnxWASMBasePath, '/vad-assets/');
  const ort = { env: { wasm: {} } };
  o.ortConfig(ort);
  assert.equal(ort.env.wasm.numThreads, 1);
  assert.equal(ort.env.wasm.proxy, false);
  await h.send(1);
  assert.equal(h.events.length, 0);
  await h.controller.destroy();
});

test('short impulse and separated positive frames do not interrupt', async () => {
  const h = await setup();
  await h.frames(0.9, 4);
  await h.frames(0.1, 19);
  for (let i = 0; i < 20; i++) {
    await h.send(0.9);
    await h.send(0.1);
  }
  assert.equal(h.events.length, 0);
  await h.controller.destroy();
});

test('start precedes bounded pre-roll chunks, tail flush precedes silence end', async () => {
  const h = await setup();
  await h.frames(0.1, 100);
  await h.frames(0.9, 4);
  assert.equal(h.events.length, 0);
  await h.frames(0.9, 1);
  assert.equal(h.events[0].type, 'start');
  assert.equal(h.events[0].detectedAt, 105 * 32);
  await h.frames(0.1, 18);
  assert.equal(h.events.some((e) => e.type === 'end'), false);
  await h.frames(0.1, 1);
  assert.equal(h.events.at(-1).type, 'end');
  assert.equal(h.events.at(-1).reason, 'silence');
  assert.equal(h.events.at(-2).type, 'chunk');
  const chunks = h.chunks();
  assert.equal(chunks.reduce((n, chunk) => n + chunk.length, 0), (8 + 5 + 19) * 768);
  assert.ok(chunks.every((chunk) => chunk.length > 0 && chunk.length <= 2400));
  assert.ok(chunks.at(-1).length < 2400);
  await h.frames(0.1, 30);
  assert.equal(h.events.filter((e) => e.type === 'end').length, 1);
  await h.controller.destroy();
});

test('short pauses and uncertain frames never accumulate into a mid-utterance cut', async () => {
  const h = await setup();
  await h.frames(0.9, 5);
  for (let i = 0; i < 4; i++) {
    await h.frames(0.1, 12);
    await h.frames(0.9, 1);
  }
  await h.frames(0.1, 18);
  await h.frames(0.4, 1);
  await h.frames(0.1, 18);
  assert.equal(h.events.filter((e) => e.type === 'start').length, 1);
  assert.equal(h.events.some((e) => e.type === 'end'), false);
  await h.frames(0.1, 1);
  assert.equal(h.events.at(-1).reason, 'silence');
  await h.controller.destroy();
});

test('reset cancels active speech and requires fresh silence before rearming', async () => {
  const h = await setup();
  await h.frames(0.9, 8);
  h.controller.reset();
  const length = h.events.length;
  await h.frames(0.9, 20);
  await h.frames(0.1, 18);
  await h.frames(0.9, 1);
  await h.frames(0.1, 19);
  assert.equal(h.events.length, length);
  await h.frames(0.9, 5);
  await h.frames(0.1, 19);
  assert.equal(h.events.filter((e) => e.type === 'start').length, 2);
  assert.equal(h.events.filter((e) => e.type === 'end').length, 1);
  assert.equal(h.events.at(-1).reason, 'silence');
  await h.controller.destroy();
});

test('repeated utterances restart interpolation and keep independent boundaries', async () => {
  const h = await setup();
  for (let i = 0; i < 3; i++) {
    await h.frames(0.9, 5);
    await h.frames(0.1, 19);
  }
  assert.equal(h.events.filter((e) => e.type === 'start').length, 3);
  assert.equal(h.events.filter((e) => e.type === 'end').length, 3);
  assert.equal(h.chunks().reduce((n, chunk) => n + chunk.length, 0), 3 * 24 * 768);
  await h.controller.destroy();
});

test('16k to 24k conversion preserves continuous interpolation across frame edges', async () => {
  const h = await setup();
  const input = new Float32Array(24 * 512);
  for (let frameIndex = 0; frameIndex < 24; frameIndex++) {
    const frame = new Float32Array(512);
    for (let i = 0; i < frame.length; i++) frame[i] = Math.sin((frameIndex * 512 + i) * 0.011);
    input.set(frame, frameIndex * 512);
    await h.send(frameIndex < 5 ? 0.9 : 0.1, frame);
  }
  const output = Float32Array.from(h.chunks().flatMap((chunk) => [...chunk]));
  assert.equal(output.length, input.length * 1.5);
  for (let i = 0; i < output.length; i++) {
    const position = i * 2 / 3;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const expected = input[left] + (input[right] - input[left]) * (position - left);
    assert.ok(Math.abs(output[i] - expected) < 1e-6, `sample ${i}`);
  }
  await h.controller.destroy();
});

test('12s includes pre-roll and cancels continuous speech until silence', async () => {
  const h = await setup();
  await h.frames(0.1, 20);
  await h.frames(0.9, 400);
  assert.equal(h.events.filter((e) => e.type === 'start').length, 1);
  assert.equal(h.events.at(-1).reason, 'max_duration');
  assert.equal(h.chunks().reduce((n, chunk) => n + chunk.length, 0), 12000 * 24);
  assert.equal(h.source.frameProcessor.audioBuffer.length, 0);
  await h.frames(0.1, 19);
  await h.frames(0.9, 5);
  assert.equal(h.events.filter((e) => e.type === 'start').length, 2);
  await h.controller.destroy();
});

test('reentrant start cancellation prevents chunks and stale completion', async () => {
  let h;
  h = await setup({ onStart: () => h.controller.reset() });
  await h.frames(0.9, 5);
  await h.frames(0.1, 19);
  assert.deepEqual(h.events.map((e) => e.type), ['start']);
  await h.controller.destroy();
});

test('reentrant chunk cancellation stops the current frame and tail', async () => {
  let h;
  h = await setup({ onChunk: () => h.controller.reset() });
  await h.frames(0.9, 5);
  await h.frames(0.1, 19);
  assert.deepEqual(h.events.map((e) => e.type), ['start', 'chunk']);
  await h.controller.destroy();
});

test('inference is serial and reset invalidates in-flight frames', async () => {
  const gate = deferred();
  const h = await setup({ frameGate: gate });
  const work = Array.from({ length: 10 }, () => h.send(0.9));
  await Promise.resolve();
  h.controller.reset();
  gate.resolve();
  await Promise.all(work);
  assert.equal(h.events.length, 0);
  assert.equal(h.counts().maximumRuns, 1);
  await h.frames(0.1, 19);
  await h.frames(0.9, 5);
  assert.equal(h.events[0].type, 'start');
  await h.controller.destroy();
});

test('queue overload fails once, bounds queued inference, and releases resources', async () => {
  const gate = deferred();
  const h = await setup({ frameGate: gate, getStream: true });
  const work = Array.from({ length: 50 }, () => h.send(0.9));
  assert.equal(h.events.filter((e) => e.type === 'error').length, 1);
  assert.ok(h.counts().stopped > 0);
  gate.resolve();
  await Promise.all(work);
  await h.controller.destroy();
  assert.equal(h.counts().released, 1);
  assert.equal(h.events.some((e) => e.type === 'start'), false);
});

test('stale identity or destroyed controller ignores later frames', async () => {
  for (const action of ['stale', 'destroy']) {
    const h = await setup();
    if (action === 'stale') h.stale();
    else await h.controller.destroy();
    await h.frames(0.9, 10);
    await h.controller.destroy();
    assert.equal(h.events.length, 0);
    assert.equal(h.counts().released, 1);
  }
});

test('destroy during a pending microphone grant closes its eventual stream', async () => {
  const permission = deferred();
  const h = await setup({ getStream: true, permissionGate: permission, noStart: true, partial: true });
  const starting = h.controller.start();
  const destroying = h.controller.destroy();
  permission.resolve(h.stream);
  await assert.rejects(starting, /취소/);
  await destroying;
  assert.ok(h.counts().stopped > 0);
  assert.equal(h.counts().released, 1);
  assert.equal(h.events.length, 0);
});

test('failed start releases loaded model and acquired stream', async () => {
  const h = await setup({ noStart: true, getStream: true, partial: true, startError: new Error('worklet failed') });
  await assert.rejects(h.controller.start(), /worklet failed/);
  await h.controller.destroy();
  assert.equal(h.events.filter((e) => e.type === 'error').length, 1);
  assert.equal(h.counts().released, 1);
  assert.ok(h.counts().stopped > 0);
  assert.equal(h.counts().disconnected, 2);
});

test('microphone ending while idle or speaking fails once without committing speech', async () => {
  for (const speaking of [false, true]) {
    const h = await setup({ getStream: true });
    if (speaking) await h.frames(0.9, 5);
    const chunkCount = h.chunks().length;
    h.track.end();
    assert.equal(h.events.filter((e) => e.type === 'error').length, 1);
    assert.ok(h.counts().stopped > 0, 'release tracks immediately');
    h.track.end();
    await h.frames(0.9, 5);
    await h.controller.destroy();
    assert.equal(h.events.filter((e) => e.type === 'error').length, 1);
    assert.equal(h.events.some((e) => e.type === 'end'), false);
    assert.equal(h.chunks().length, chunkCount);
    assert.equal(h.counts().released, 1);
    assert.equal(h.endedListeners.size, 0);
  }
});

test('intentional microphone stop removes handlers and ignores delayed ended callbacks', async () => {
  const h = await setup({ getStream: true });
  const delayedEnded = [...h.endedListeners][0];
  assert.equal(typeof delayedEnded, 'function');
  await h.controller.destroy();
  const next = await setup({ getStream: true });
  delayedEnded();
  h.track.end();
  assert.equal(h.events.length, 0);
  assert.equal(h.endedListeners.size, 0);
  await next.frames(0.9, 5);
  assert.equal(next.events[0].type, 'start');
  assert.equal(next.events.some((e) => e.type === 'error'), false);
  await next.controller.destroy();
});

test('microphone already ended on grant rejects startup and releases its model', async () => {
  const h = await setup({ getStream: true, noStart: true, endedOnGrant: true, partial: true });
  await assert.rejects(h.controller.start(), /마이크.*종료/);
  await h.controller.destroy();
  assert.equal(h.events.filter((e) => e.type === 'error').length, 1);
  assert.equal(h.counts().released, 1);
  assert.equal(h.endedListeners.size, 0);
});

test('microphone ending during worklet startup reports failure before startup completes', async () => {
  const gate = deferred();
  const entered = deferred();
  const h = await setup({ getStream: true, noStart: true, startGate: gate, startEntered: entered });
  const starting = h.controller.start();
  await entered.promise;
  h.track.end();
  assert.equal(h.events.filter((e) => e.type === 'error').length, 1);
  assert.ok(h.counts().stopped > 0);
  gate.resolve();
  await starting;
  await h.controller.destroy();
  assert.equal(h.counts().released, 1);
  assert.equal(h.endedListeners.size, 0);
});

test('late detector creation releases model when requesting identity is obsolete', async () => {
  const gate = deferred();
  const h = await setup({ createGate: gate, partial: true });
  h.stale();
  gate.resolve();
  const controller = await h.creating;
  await controller.start();
  assert.equal(h.counts().released, 1);
  assert.equal(h.events.length, 0);
});

test('model errors and malformed frames fail safely without finalizing speech', async () => {
  for (const modelError of [true, false]) {
    const h = await setup(modelError ? { frameError: new Error('model failed') } : {});
    await h.send(modelError ? 0.9 : Number.NaN);
    await h.controller.destroy();
    assert.deepEqual(h.events.map((e) => e.type), ['error']);
    assert.equal(h.counts().released, 1);
  }
});

test('pinned MicVAD retention cleanup preserves recurrent-model probability inputs', async () => {
  const require = createRequire(import.meta.url);
  const { FrameProcessor } = require('@ricky0123/vad-web/dist/frame-processor.js');
  const { MicVAD } = require('@ricky0123/vad-web/dist/real-time-vad.js');
  const options = {
    positiveSpeechThreshold: 0.6, negativeSpeechThreshold: 0.35,
    preSpeechPadMs: 0, redemptionMs: 600, minSpeechMs: 160,
    submitUserSpeechOnPause: false,
  };
  async function exercise(clearBuffer) {
    let recurrence = 0;
    let resets = 0;
    let maximumRetention = 0;
    const probabilities = [];
    // Use the pinned real wrapper/processor with a fake recurrent model only.
    const model = {
      async process(frame) {
        recurrence = (recurrence + 1) % 17;
        return { isSpeech: frame[0] + recurrence * 0.0001 };
      },
      reset_state() { recurrence = 0; resets++; },
    };
    const processor = new FrameProcessor(model.process, model.reset_state, options, 32);
    const mic = new MicVAD({
      onFrameProcessed: (probs) => probabilities.push(probs.isSpeech),
      onSpeechStart() {}, onSpeechRealStart() {}, onSpeechEnd() {}, onVADMisfire() {},
    }, processor, model, 512);
    assert.equal(mic.frameProcessor, processor);
    assert.equal(mic.model, model);
    processor.resume();
    for (let i = 0; i < 1000; i++) {
      await mic.processFrame(new Float32Array(512).fill(i % 200 < 160 ? 0.9 : 0.1));
      maximumRetention = Math.max(maximumRetention, processor.audioBuffer.length);
      if (clearBuffer) processor.audioBuffer = [];
    }
    return { probabilities, resets, maximumRetention };
  }
  const original = await exercise(false);
  const bounded = await exercise(true);
  assert.deepEqual(original.probabilities, bounded.probabilities);
  assert.equal(original.resets, bounded.resets);
  assert.equal(bounded.maximumRetention, 1);
  assert.ok(original.maximumRetention > bounded.maximumRetention);
});

function offlineSegmenter(settings = {}) {
  const events = [];
  const context = vm.createContext({ Float32Array, Error });
  vm.runInContext(code, context);
  const api = context.BanterVadCapture;
  const segmenter = api.createSegmenter({
    ...settings,
    onStart(value) { events.push({ type: 'start', ...value }); settings.onStart?.(); },
    onChunk(samples) { events.push({ type: 'chunk', samples }); settings.onChunk?.(); },
    onEnd(value) { events.push({ type: 'end', ...value }); settings.onEnd?.(); },
  });
  let time = 0;
  return {
    api, segmenter, events,
    frame(probability, samples = new Float32Array(512).fill(0.25)) {
      time += 32;
      segmenter.process({ isSpeech: probability }, samples, time);
    },
    frames(probability, count) { for (let i = 0; i < count; i++) this.frame(probability); },
  };
}

test('shared segmenter reproduces live decisions and exact PCM without browser dependencies', async () => {
  const live = await setup();
  const offline = offlineSegmenter();
  const sequence = [
    ...Array(10).fill(0.1), ...Array(4).fill(0.9), ...Array(2).fill(0.1),
    ...Array(7).fill(0.9), ...Array(12).fill(0.1), ...Array(2).fill(0.4),
    ...Array(3).fill(0.9), ...Array(19).fill(0.1), ...Array(400).fill(0.9),
    ...Array(19).fill(0.1), ...Array(5).fill(0.9), ...Array(19).fill(0.1),
  ];
  for (const [index, probability] of sequence.entries()) {
    const samples = Float32Array.from({ length: 512 }, (_, j) => Math.sin((index * 512 + j) * 0.017));
    await live.send(probability, samples);
    offline.frame(probability, samples);
  }
  assert.deepEqual(offline.events, live.events);
  assert.deepEqual(offline.events.filter(e => e.type === 'end').map(e => e.reason), [
    'silence', 'max_duration', 'silence',
  ]);
  assert.equal(offline.segmenter.getState(), 'idle');
  await live.controller.destroy();
  offline.segmenter.destroy();
});

test('offline endMs variants change only the configured silence boundary and leave defaults frozen', () => {
  const fast = offlineSegmenter({ config: { endMs: 320 } });
  const baseline = offlineSegmenter();
  assert.ok(Object.isFrozen(fast.api.DEFAULT_CONFIG));
  assert.deepEqual({ ...fast.api.DEFAULT_CONFIG }, {
    startMs: 160, endMs: 600, preRollMs: 256, maxMs: 12000,
    positiveThreshold: 0.6, negativeThreshold: 0.35,
  });
  for (const h of [fast, baseline]) {
    h.frames(0.9, 5);
    h.frames(0.1, 10);
  }
  assert.equal(fast.events.at(-1).type, 'end');
  assert.equal(fast.events.at(-1).detectedAt, 480);
  assert.equal(baseline.segmenter.getState(), 'speaking');
  baseline.frames(0.1, 9);
  assert.equal(baseline.events.at(-1).detectedAt, 768);
  assert.deepEqual(fast.events[0], baseline.events[0]);
  assert.throws(() => { fast.api.DEFAULT_CONFIG.endMs = 320; }, TypeError);
});

test('shared segmenter rejects unknown, nonfinite and impossible configurations', () => {
  const api = offlineSegmenter().api;
  for (const config of [
    null, [], 12, '600', { endms: 600 }, { [Symbol('unknown')]: 1 },
    { endMs: Number.NaN }, { startMs: Infinity }, { maxMs: '12000' },
    { startMs: 0 }, { endMs: -1 }, { preRollMs: -1 }, { maxMs: 31 },
    { endMs: 60001 }, { positiveThreshold: 0 }, { positiveThreshold: 1.1 },
    { negativeThreshold: 0 }, { negativeThreshold: 0.6 },
    { startMs: 161, preRollMs: 256, maxMs: 447 },
  ]) {
    assert.throws(() => api.createSegmenter({ config }), Error, `config ${String(config)}`);
  }
});

test('shared segmenter validates frame and timestamp before changing its state', () => {
  const { segmenter } = offlineSegmenter();
  for (const [probabilities, frame, detectedAt] of [
    [{ isSpeech: Number.NaN }, new Float32Array(512), 32],
    [{ isSpeech: 1.1 }, new Float32Array(512), 32],
    [{ isSpeech: -0.1 }, new Float32Array(512), 32],
    [{ isSpeech: 0.9 }, new Float32Array(511), 32],
    [{ isSpeech: 0.9 }, new Float64Array(512), 32],
    [{ isSpeech: 0.9 }, new Float32Array(512).fill(Infinity), 32],
    [{ isSpeech: 0.9 }, new Float32Array(512), Number.NaN],
    [{ isSpeech: 0.9 }, new Float32Array(512), -1],
  ]) {
    assert.throws(() => segmenter.process(probabilities, frame, detectedAt), Error);
    assert.equal(segmenter.getState(), 'idle');
  }
});

test('shared segmenter exposes partial states and never commits an unfinished EOF segment', () => {
  const h = offlineSegmenter();
  assert.equal(h.segmenter.getState(), 'idle');
  h.frames(0.9, 4);
  assert.equal(h.segmenter.getState(), 'candidate');
  h.frame(0.9);
  assert.equal(h.segmenter.getState(), 'speaking');
  h.frames(0.1, 18);
  assert.equal(h.events.some(e => e.type === 'end'), false);
  const eventCount = h.events.length;
  h.segmenter.destroy();
  h.frames(0.1, 19);
  assert.equal(h.events.length, eventCount);
  assert.equal(h.segmenter.getState(), 'suppressed');
});

test('shared segmenter keeps reset and destroy safe during callback reentry', () => {
  for (const callback of ['onStart', 'onChunk']) {
    for (const action of ['reset', 'destroy']) {
      let h;
      h = offlineSegmenter({ [callback]: () => h.segmenter[action]() });
      h.frames(0.9, 5);
      h.frames(0.1, 19);
      assert.deepEqual(h.events.map(e => e.type), callback === 'onStart' ? ['start'] : ['start', 'chunk']);
      assert.equal(h.segmenter.getState(), action === 'reset' ? 'idle' : 'suppressed');
    }
  }
  let current = true;
  const stale = offlineSegmenter({ isCurrent: () => current });
  stale.frames(0.9, 4);
  current = false;
  stale.frames(0.9, 20);
  assert.equal(stale.events.length, 0);
});

test('a configured limit at the opening buffer ends exactly at its maximum', () => {
  const h = offlineSegmenter({ config: { startMs: 32, preRollMs: 0, maxMs: 32 } });
  h.frame(0.9);
  assert.deepEqual(h.events.map(e => e.type), ['start', 'chunk', 'end']);
  assert.equal(h.events[1].samples.length, 768);
  assert.equal(h.events[2].reason, 'max_duration');
  assert.equal(h.segmenter.getState(), 'suppressed');
});
