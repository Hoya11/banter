import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(resolve(HERE, '..', 'index.html'), 'utf8');
const workletSource = readFileSync(resolve(HERE, '..', 'pcm-capture-worklet.js'), 'utf8');
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
assert.equal(scripts.length, 1, 'index.html must contain exactly one inline script');
const appSource = scripts[0][1];

function deferred() {
  let resolvePromise;
  let rejectPromise;
  const promise = new Promise((resolve, reject) => {
    resolvePromise = resolve;
    rejectPromise = reject;
  });
  return { promise, resolve: resolvePromise, reject: rejectPromise };
}

async function drainMicrotasks() {
  for (let index = 0; index < 20; index += 1) await Promise.resolve();
}

class FakeElement {
  constructor(ownerDocument, tagName = 'div', id = '') {
    this.ownerDocument = ownerDocument;
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.listeners = new Map();
    this.attributes = new Map();
    this.className = '';
    this.disabled = false;
    this.hidden = false;
    this.open = false;
    this.placeholder = '';
    this.style = {};
    this._id = '';
    this._textContent = '';
    this.classList = {
      add: (...names) => {
        const classes = new Set(this.className.split(/\s+/).filter(Boolean));
        names.forEach((name) => classes.add(name));
        this.className = [...classes].join(' ');
      },
      remove: (...names) => {
        const removed = new Set(names);
        this.className = this.className
          .split(/\s+/)
          .filter((name) => name && !removed.has(name))
          .join(' ');
      },
      contains: (name) => this.className.split(/\s+/).includes(name),
    };
    this.id = id;
  }

  get id() {
    return this._id;
  }

  set id(value) {
    if (this._id && this.ownerDocument.elements.get(this._id) === this) {
      this.ownerDocument.elements.delete(this._id);
    }
    this._id = value;
    if (value) this.ownerDocument.elements.set(value, this);
  }

  get textContent() {
    if (this.children.length) return this.children.map((child) => child.textContent).join('');
    return this._textContent;
  }

  set textContent(value) {
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = [];
    this._textContent = String(value);
  }

  get scrollHeight() {
    return this.children.length;
  }

  append(...children) {
    for (const child of children) {
      child.parentNode = this;
      this.children.push(child);
    }
  }

  replaceChildren(...children) {
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = [];
    this._textContent = '';
    this.append(...children);
  }

  remove() {
    if (this.parentNode) {
      const index = this.parentNode.children.indexOf(this);
      if (index >= 0) this.parentNode.children.splice(index, 1);
      this.parentNode = null;
    }
    if (this.id && this.ownerDocument.elements.get(this.id) === this) {
      this.ownerDocument.elements.delete(this.id);
    }
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  click() {
    return this.listeners.get('click')?.({ preventDefault() {} });
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
}

class FakeDocument {
  constructor() {
    this.elements = new Map();
    this.activeElement = null;
    this.body = new FakeElement(this, 'body');
    const tags = {
      log: 'div',
      'session-status': 'p',
      'session-start': 'button',
      form: 'form',
      msg: 'input',
      send: 'button',
      mic: 'button',
      'session-label': 'span',
      'room-title': 'h1',
      'room-description': 'p',
      'person-ai_a': 'article',
      'person-ai_b': 'article',
      'person-status-ai_a': 'p',
      'person-status-ai_b': 'p',
      'mic-control': 'div',
      'mic-state': 'span',
      'mic-action': 'p',
      'input-mode': 'span',
      'input-hint': 'p',
      'text-entry': 'details',
      'conversation-details': 'details',
    };
    for (const [id, tag] of Object.entries(tags)) new FakeElement(this, tag, id);
  }

  getElementById(id) {
    return this.elements.get(id) ?? null;
  }

  createElement(tagName) {
    return new FakeElement(this, tagName);
  }
}

function createHarness({
  getUserMedia,
  performanceTimes = [0],
  fileReaderError = false,
  audioContextSampleRate = 24000,
  vadCreate,
  vadStart,
  realSegmenter = false,
} = {}) {
  const document = new FakeDocument();
  const clock = [...performanceTimes];
  const timeouts = [];
  const vadControllers = [];
  const windowListeners = new Map();
  let lastClockValue = clock.at(-1) ?? 0;

  class FakeAudio {
    static instances = [];

    constructor() {
      this.src = '';
      this.onended = null;
      this.onerror = null;
      this.playCalls = [];
      this.pauseCount = 0;
      this.loadCount = 0;
      this.paused = true;
      this.pauseError = null;
      this.pauseLeavesPlaying = false;
      FakeAudio.instances.push(this);
    }

    play() {
      const call = deferred();
      this.playCalls.push(call);
      this.paused = false;
      if (this.playCalls.length === 1) call.resolve(); // start 버튼의 무음 unlock
      return call.promise;
    }

    pause() {
      this.pauseCount += 1;
      if (this.pauseError) throw this.pauseError;
      if (!this.pauseLeavesPlaying) this.paused = true;
    }

    removeAttribute(name) {
      if (name === 'src') this.src = '';
    }

    load() {
      this.loadCount += 1;
    }

    endCurrent() {
      this.paused = true;
      this.onended?.();
    }
  }

  class FakeWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
    static instances = [];

    constructor(url) {
      this.url = url;
      this.readyState = FakeWebSocket.CONNECTING;
      this.bufferedAmount = 0;
      this.sent = [];
      this.closeCalls = [];
      this.onopen = null;
      this.onmessage = null;
      this.onerror = null;
      this.onclose = null;
      FakeWebSocket.instances.push(this);
    }

    open() {
      this.readyState = FakeWebSocket.OPEN;
      this.onopen?.();
    }

    receive(payload) {
      this.onmessage?.({ data: JSON.stringify(payload) });
    }

    send(raw) {
      this.sent.push(JSON.parse(raw));
    }

    close(code) {
      this.closeCalls.push(code);
      this.readyState = FakeWebSocket.CLOSED;
      this.onclose?.({ code });
    }
  }

  class FakeMediaRecorder {
    static instances = [];

    constructor(stream) {
      this.stream = stream;
      this.state = 'inactive';
      this.mimeType = 'audio/webm';
      this.ondataavailable = null;
      this.onstop = null;
      FakeMediaRecorder.instances.push(this);
    }

    start() {
      this.state = 'recording';
    }

    stop() {
      this.state = 'inactive';
      this.onstop?.();
    }
  }

  class FakeConnectableNode {
    constructor() {
      this.connections = [];
      this.disconnected = false;
    }

    connect(target) {
      this.connections.push(target);
      return target;
    }

    disconnect() {
      this.disconnected = true;
      this.connections.length = 0;
    }
  }

  class FakeAudioWorkletNode extends FakeConnectableNode {
    static instances = [];

    constructor(context, name, options) {
      super();
      this.context = context;
      this.name = name;
      this.options = options;
      this.flushSamples = null;
      this.port = {
        onmessage: null,
        sent: [],
        postMessage: (message) => {
          this.port.sent.push(message);
          if (message?.type !== 'flush') return;
          if (this.flushSamples?.length) {
            this.emit({ type: 'audio', samples: this.flushSamples });
          }
          this.emit({ type: 'flushed' });
        },
      };
      FakeAudioWorkletNode.instances.push(this);
    }

    emit(data) {
      this.port.onmessage?.({ data });
    }
  }

  class FakeAudioContext {
    static instances = [];

    constructor(options = {}) {
      this.requestedSampleRate = options.sampleRate;
      this.sampleRate = audioContextSampleRate;
      this.destination = {};
      this.moduleUrls = [];
      this.resumeCalls = 0;
      this.closeCalls = 0;
      this.sources = [];
      this.gains = [];
      this.audioWorklet = {
        addModule: async (url) => {
          this.moduleUrls.push(url);
        },
      };
      FakeAudioContext.instances.push(this);
    }

    createMediaStreamSource(stream) {
      const source = new FakeConnectableNode();
      source.stream = stream;
      this.sources.push(source);
      return source;
    }

    createGain() {
      const gain = new FakeConnectableNode();
      gain.gain = { value: 1 };
      this.gains.push(gain);
      return gain;
    }

    async resume() {
      this.resumeCalls += 1;
    }

    close() {
      this.closeCalls += 1;
      return Promise.resolve();
    }
  }

  class FakeFileReader {
    readAsDataURL() {
      if (fileReaderError) {
        queueMicrotask(() => this.onerror?.(new Error('read failed')));
      } else {
        this.result = 'data:audio/webm;base64,';
        queueMicrotask(() => this.onload?.());
      }
    }
  }

  const microphone = getUserMedia ?? (() => Promise.reject(new Error('not configured')));
  const segmenterContext = { Float32Array };
  if (realSegmenter) vm.runInNewContext(
    readFileSync(resolve(HERE, '..', 'vad-capture.js'), 'utf8'), segmenterContext,
  );
  const context = {
    BanterVadCapture: {
      create: async (options) => {
        const segmenter = realSegmenter
          ? segmenterContext.BanterVadCapture.createSegmenter(options) : null;
        const controller = {
          segmenter,
          frames(speech, count) {
            for (let index = 0; index < count; index++) {
              segmenter.process({ isSpeech: speech ? 0.9 : 0.1 }, new Float32Array(512), 100);
            }
          },
          options, resetCalls: 0, destroyCalls: 0, startCalls: 0,
          async start() {
            this.startCalls += 1;
            if (vadStart) await vadStart(this);
          },
          reset() { this.resetCalls += 1; segmenter?.reset(); },
          async destroy() { this.destroyCalls += 1; segmenter?.destroy(); },
          speechStart(detectedAt = 100) { options.onStart({ detectedAt }); },
          chunk(samples = new Float32Array(2400)) { options.onChunk(samples); },
          speechEnd(detectedAt = 1000, reason = 'silence') {
            options.onEnd({ detectedAt, reason });
          },
        };
        vadControllers.push(controller);
        return vadCreate ? vadCreate(controller) : controller;
      },
    },
    Audio: FakeAudio,
    AudioContext: FakeAudioContext,
    AudioWorkletNode: FakeAudioWorkletNode,
    Blob,
    FileReader: FakeFileReader,
    MediaRecorder: FakeMediaRecorder,
    WebSocket: FakeWebSocket,
    console,
    document,
    location: { protocol: 'http:', host: 'localhost:8000' },
    navigator: { mediaDevices: { getUserMedia: microphone } },
    performance: {
      now: () => {
        if (clock.length) lastClockValue = clock.shift();
        return lastClockValue;
      },
    },
    queueMicrotask,
    addEventListener: (type, listener) => windowListeners.set(type, listener),
    btoa: (binary) => Buffer.from(binary, 'binary').toString('base64'),
    setTimeout: (callback) => {
      timeouts.push(callback);
      return timeouts.length;
    },
  };
  vm.runInNewContext(appSource, context, { filename: 'web/index.html' });

  const elements = Object.fromEntries(document.elements);
  const audio = FakeAudio.instances[0];

  async function startConnecting({ voice = true, stt = false, sttStream = null, vad = false } = {}) {
    await elements['session-start'].click();
    const socket = FakeWebSocket.instances.at(-1);
    assert.ok(socket, 'start must create a WebSocket');
    socket.open();
    socket.receive({ type: 'hello', voice, stt, ...(sttStream ? { stt_stream: sttStream } : {}),
      ...(vad ? { interaction_mode: 'vad', vad: { engine: 'silero_v5' } } : {}) });
    await drainMicrotasks();
    return socket;
  }

  async function startRunning(options) {
    const socket = await startConnecting(options);
    assert.equal(socket.sent.filter((event) => event.type === 'session_start').length, 1);
    socket.receive({ type: 'session_started', session_id: 'test-session' });
    return socket;
  }

  return {
    startConnecting,
    audio,
    document,
    elements,
    AudioContext: FakeAudioContext,
    AudioWorkletNode: FakeAudioWorkletNode,
    MediaRecorder: FakeMediaRecorder,
    sockets: FakeWebSocket.instances,
    startRunning,
    timeouts,
    vadControllers,
    blurWindow: () => windowListeners.get('blur')?.(),
  };
}

function subtitles(log) {
  return log.children.map((message) => message.children[1]?.textContent ?? '');
}

const PCM_STREAM_CONFIG = Object.freeze({
  encoding: 'pcm_s16le',
  sample_rate_hz: 24000,
  channels: 1,
  chunk_samples: 2400,
});

function decodePcm16(base64) {
  const bytes = Buffer.from(base64, 'base64');
  const samples = [];
  for (let offset = 0; offset < bytes.length; offset += 2) {
    samples.push(bytes.readInt16LE(offset));
  }
  return samples;
}

test('voice room hides transcripts and keeps text entry optional without focusing it', async () => {
  const harness = createHarness();
  const ui = harness.elements;
  assert.equal(ui['session-start'].hidden, false);
  assert.equal(ui['text-entry'].hidden, true);
  assert.equal(ui['conversation-details'].hidden, true);
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'off');
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'idle');
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'idle');

  await harness.startRunning({ voice: true, sttStream: PCM_STREAM_CONFIG });
  assert.equal(ui['session-start'].hidden, true);
  assert.equal(ui['text-entry'].hidden, false);
  assert.equal(ui['text-entry'].open, false);
  assert.equal(ui['conversation-details'].hidden, true);
  assert.notEqual(harness.document.activeElement, ui.msg);
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'ready');
});

test('text-only fallback exposes conversation and separates generation from speaking', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: false });
  const ui = harness.elements;
  assert.equal(ui['conversation-details'].hidden, false);
  assert.equal(ui['conversation-details'].open, true);
  assert.equal(ui['text-entry'].hidden, false);
  assert.equal(ui['text-entry'].open, true);
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'unavailable');
  socket.receive({ type: 'start', speaker: 'ai_b' });
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'idle');
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'thinking');
  socket.receive({ type: 'end', speaker: 'ai_b', text: '텍스트 응답' });
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'idle');
});

test('participant speaks only after audio starts and remains accurate through the last queued audio', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: true });
  const ui = harness.elements;
  socket.receive({ type: 'start', speaker: 'ai_a' });
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'idle');
  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '첫 문장', audio: 'YQ==' });
  socket.receive({ type: 'audio', seq: 2, speaker: 'ai_b', text: '마지막 문장', audio: 'Yg==' });
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'idle', 'an unresolved play call is not audible playback');
  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'speaking');
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'idle');
  socket.receive({ type: 'done' });
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'speaking');
  harness.audio.endCurrent();
  assert.equal(ui['person-ai_a'].getAttribute('data-state'), 'idle');
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'idle');
  harness.audio.playCalls[2].resolve();
  await drainMicrotasks();
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'speaking');
  harness.audio.endCurrent();
  assert.equal(ui['person-ai_b'].getAttribute('data-state'), 'idle');
  assert.equal(ui['session-start'].hidden, false);
});

test('push-to-talk shows permission preparation before capture and transcript processing after release', async () => {
  const permission = deferred();
  const harness = createHarness({ getUserMedia: () => permission.promise });
  const socket = await harness.startRunning({ voice: true, sttStream: PCM_STREAM_CONFIG });
  const ui = harness.elements;
  const pending = ui.mic.onpointerdown();
  await drainMicrotasks();
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'preparing');
  permission.resolve({ getTracks: () => [{ stop() {} }] });
  await pending;
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'recording');
  const node = harness.AudioWorkletNode.instances.at(-1);
  for (let index = 0; index < 3; index++) node.emit({ type: 'audio', samples: new Float32Array(2400) });
  ui.mic.onpointerup();
  await drainMicrotasks();
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'processing');
  const commit = socket.sent.find((message) => message.type === 'voice_stream_commit');
  assert.ok(commit);
  socket.receive({ type: 'you', client_event_id: commit.client_event_id, text: '확정된 발화' });
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'ready');
});

test('Space and Enter hold one PTT request despite repeats and commit once on key release', async () => {
  for (const key of [' ', 'Enter']) {
    let permissionCalls = 0;
    let prevented = 0;
    const harness = createHarness({ getUserMedia: async () => {
      permissionCalls++;
      return { getTracks: () => [{ stop() {} }] };
    } });
    const socket = await harness.startRunning({ sttStream: PCM_STREAM_CONFIG });
    const mic = harness.elements.mic;
    const keyEvent = (repeat = false) => ({ key, repeat, preventDefault() { prevented++; } });
    await mic.listeners.get('keydown')(keyEvent());
    await mic.listeners.get('keydown')(keyEvent(true));
    assert.equal(permissionCalls, 1);
    assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 1);
    assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'recording');
    const node = harness.AudioWorkletNode.instances.at(-1);
    for (let index = 0; index < 3; index++) node.emit({ type: 'audio', samples: new Float32Array(2400) });
    mic.listeners.get('keyup')(keyEvent());
    mic.listeners.get('keyup')(keyEvent());
    await drainMicrotasks();
    assert.equal(prevented, 4);
    assert.equal(socket.sent.filter((message) => message.type === 'voice_stream_commit').length, 1);
    assert.equal(socket.sent.some((message) => message.type === 'hold_off'), false);
    assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'processing');
  }
});

test('microphone and window blur cancel keyboard permission waits and stop a late granted stream', async () => {
  for (const blurTarget of ['microphone', 'window']) {
    const permission = deferred();
    let stopped = 0;
    const harness = createHarness({ getUserMedia: () => permission.promise });
    const socket = await harness.startRunning({ sttStream: PCM_STREAM_CONFIG });
    const mic = harness.elements.mic;
    const pending = mic.listeners.get('keydown')({ key: ' ', repeat: false, preventDefault() {} });
    await drainMicrotasks();
    assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'preparing');
    if (blurTarget === 'microphone') mic.listeners.get('blur')();
    else harness.blurWindow();
    mic.listeners.get('keyup')({ key: ' ', preventDefault() {} });
    assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'ready');
    const hold = socket.sent.find((message) => message.type === 'hold');
    assert.deepEqual(socket.sent.filter((message) => message.type === 'hold_off'), [
      { type: 'hold_off', client_event_id: hold.client_event_id },
    ]);
    permission.resolve({ getTracks: () => [{ stop() { stopped++; } }] });
    await pending;
    assert.equal(stopped, 1);
    assert.equal(harness.AudioWorkletNode.instances.length, 0);
    assert.equal(socket.sent.some((message) => message.type === 'voice_stream_start'), false);
    assert.equal(socket.sent.some((message) => message.type === 'voice_stream_commit'), false);
    assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'ready');
  }
});

test('a delayed previous recorder stop cannot clear the current microphone recording state', async () => {
  const streams = [0, 0].map(() => ({
    stopped: 0,
    getTracks() { return [{ stop: () => { this.stopped++; } }]; },
  }));
  let streamIndex = 0;
  const harness = createHarness({ getUserMedia: async () => streams[streamIndex++] });
  const socket = await harness.startRunning({ voice: true, stt: true });
  await harness.elements.mic.onpointerdown();
  const first = harness.MediaRecorder.instances[0];
  const previousOnStop = first.onstop;
  first.ondataavailable({ data: new Uint8Array(9000) });
  first.stop = () => { first.state = 'inactive'; };
  harness.elements.mic.onpointerup();
  await harness.elements.mic.onpointerdown();
  const second = harness.MediaRecorder.instances[1];
  assert.equal(second.state, 'recording');
  const sentCount = socket.sent.length;
  await previousOnStop();
  assert.equal(streams[0].stopped, 1);
  assert.equal(streams[1].stopped, 0);
  assert.equal(socket.sent.length, sentCount, 'the stale recording must not submit audio');
  assert.equal(second.state, 'recording');
  assert.equal(harness.elements.mic.classList.contains('rec'), true);
  assert.equal(harness.elements.mic.getAttribute('aria-pressed'), 'true');
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'recording');
});

test('missing or failed voice audio reveals readable text while restart isolates late failures', async () => {
  for (const failure of ['missing_audio', 'error_event', 'play_rejection']) {
    const harness = createHarness();
    const socket = await harness.startRunning({ voice: true });
    const conversation = harness.elements['conversation-details'];
    assert.equal(conversation.hidden, true);
    socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '읽을 수 있는 응답',
      audio: failure === 'missing_audio' ? null : 'YQ==' });
    if (failure === 'error_event') harness.audio.onerror();
    else if (failure === 'play_rejection') harness.audio.playCalls.at(-1).reject(new Error('play failed'));
    await drainMicrotasks();
    assert.equal(conversation.hidden, false);
    assert.equal(conversation.open, true);
    assert.deepEqual(subtitles(harness.elements.log), ['읽을 수 있는 응답']);
    assert.deepEqual(socket.sent.filter((message) => message.type === 'played'), [{ type: 'played', seq: 1 }]);

    socket.receive({ type: 'audio', seq: 2, speaker: 'ai_b', text: '이전 세션 대기 음성', audio: 'Yg==' });
    const staleError = harness.audio.onerror;
    const stalePlay = harness.audio.playCalls.at(-1);
    socket.receive({ type: 'done' });
    await harness.startRunning({ voice: true });
    assert.equal(conversation.hidden, true);
    assert.equal(conversation.open, false);
    staleError();
    stalePlay.reject(new Error('previous session play failed'));
    await drainMicrotasks();
    assert.equal(conversation.hidden, true);
    assert.equal(conversation.open, false);
    assert.equal(harness.elements.log.children.length, 0);
  }
});

test('done preserves the current and queued audio until playback finishes', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: true });
  const pauseAfterUnlock = harness.audio.pauseCount;

  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '첫 문장', audio: 'YQ==', cont: false });
  socket.receive({ type: 'audio', seq: 2, speaker: 'ai_b', text: '둘째 문장', audio: 'Yg==', cont: false });
  assert.equal(harness.audio.playCalls.length, 2); // unlock + 첫 문장, 둘째 문장은 queue
  const loadBeforeDone = harness.audio.loadCount;
  const sourceBeforeDone = harness.audio.src;

  socket.receive({ type: 'done' });
  assert.equal(harness.audio.pauseCount, pauseAfterUnlock, 'done must not pause current audio');
  assert.equal(harness.audio.loadCount, loadBeforeDone, 'done must not reset current audio');
  assert.equal(harness.audio.src, sourceBeforeDone, 'done must preserve the playing source');

  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();
  assert.deepEqual(subtitles(harness.elements.log), ['첫 문장']);

  harness.audio.endCurrent();
  assert.equal(harness.audio.playCalls.length, 3, 'queued audio must start after current audio');
  harness.audio.playCalls[2].resolve();
  await drainMicrotasks();
  assert.deepEqual(subtitles(harness.elements.log), ['첫 문장', '둘째 문장']);
});

test('restart clears presentation state and opens a fresh WebSocket', async () => {
  const harness = createHarness();
  const firstSocket = await harness.startRunning({ voice: true });
  firstSocket.receive({ type: 'you', text: '이전 대화' });
  firstSocket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '재생 중', audio: 'YQ==', cont: false });
  firstSocket.receive({ type: 'audio', seq: 2, speaker: 'ai_b', text: '대기 중', audio: 'Yg==', cont: false });
  firstSocket.receive({ type: 'done' });

  await harness.elements['session-start'].click();
  assert.equal(harness.sockets.length, 2);
  assert.notEqual(harness.sockets[1], firstSocket);
  assert.equal(harness.sockets[1].url, 'ws://localhost:8000/ws');
  assert.equal(harness.elements.log.children.length, 0);
  assert.ok(harness.audio.loadCount > 0, 'restart must reset the reusable Audio element');

  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();
  harness.audio.endCurrent();
  assert.equal(harness.audio.playCalls.length, 2, 'audio queued by the old session must be cleared');
});

test('a stale player.play resolution cannot add subtitles after reset', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: true });
  socket.receive({ type: 'audio', seq: 7, speaker: 'ai_a', text: '늦은 자막', audio: 'YQ==', cont: false });
  socket.receive({ type: 'done' });

  await harness.elements['session-start'].click();
  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();

  assert.equal(harness.elements.log.children.length, 0);
  assert.deepEqual(
    harness.sockets[1].sent.filter((message) => message.type === 'played'),
    [],
    'the stale callback must not ack into the new session',
  );
});

test('a stale player.play rejection cannot leak into the new session', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: true });
  socket.receive({ type: 'audio', seq: 8, speaker: 'ai_b', text: '취소된 자막', audio: 'Yg==', cont: false });
  socket.receive({ type: 'done' });

  await harness.elements['session-start'].click();
  const newSocket = harness.sockets[1];
  newSocket.open();
  newSocket.receive({ type: 'hello', voice: true, stt: false });
  newSocket.receive({ type: 'session_started', session_id: 'new-session' });
  const abortError = Object.assign(new Error('playback reset'), { name: 'AbortError' });
  harness.audio.playCalls[1].reject(abortError);
  await drainMicrotasks();

  assert.equal(harness.elements.log.children.length, 0);
  assert.deepEqual(
    newSocket.sent.filter((message) => message.type === 'played'),
    [],
    'the stale rejection must not ack into the new session',
  );
});

test('pointerdown pauses audible playback and includes the stop metric in hold', async () => {
  const permission = deferred();
  const harness = createHarness({
    getUserMedia: () => permission.promise,
    performanceTimes: [10, 13.75],
  });
  const socket = await harness.startRunning({ voice: true, stt: true });
  const pauseAfterUnlock = harness.audio.pauseCount;
  socket.receive({ type: 'audio', seq: 7, speaker: 'ai_a', text: '재생 중', audio: 'YQ==', cont: false });
  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();

  const recordingStart = harness.elements.mic.onpointerdown();
  const hold = socket.sent.find((message) => message.type === 'hold');

  assert.equal(harness.audio.pauseCount, pauseAfterUnlock + 1);
  assert.deepEqual(hold, {
    type: 'hold',
    client_event_id: 'test-session-1',
    audio_stop: {
      outcome: 'paused',
      elapsed_ms: 3.75,
      segment_id: 'audio-7',
    },
  });
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'played'),
    [{ type: 'played', seq: 7 }],
    'the interrupted segment must still be acknowledged once',
  );
  assert.ok(
    socket.sent.findIndex((message) => message.type === 'hold')
      < socket.sent.findIndex((message) => message.type === 'played'),
    'hold must reach the WebSocket before the discard acknowledgement',
  );
  assert.deepEqual(subtitles(harness.elements.log), ['재생 중']);

  harness.elements.mic.onpointerup();
  permission.resolve({ getTracks: () => [{ stop() {} }] });
  await recordingStart;
});

test('queued audio is cleared without counting it as audible stop latency', async () => {
  const permission = deferred();
  const harness = createHarness({
    getUserMedia: () => permission.promise,
    performanceTimes: [20, 20.4],
  });
  const socket = await harness.startRunning({ voice: true, stt: true });
  socket.receive({ type: 'audio', seq: 2, speaker: 'ai_a', text: '재생 대기', audio: 'YQ==', cont: false });
  socket.receive({ type: 'audio', seq: 3, speaker: 'ai_b', text: '큐 대기', audio: 'Yg==', cont: false });

  const recordingStart = harness.elements.mic.onpointerdown();
  const hold = socket.sent.find((message) => message.type === 'hold');

  assert.deepEqual(hold.audio_stop, {
    outcome: 'queued_only',
    elapsed_ms: 0.4,
    segment_id: 'audio-2',
  });
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'played'),
    [{ type: 'played', seq: 3 }],
    'the highest discarded sequence must acknowledge the whole queue',
  );

  harness.elements.mic.onpointerup();
  permission.resolve({ getTracks: () => [{ stop() {} }] });
  await recordingStart;
});

test('a failed pause is recorded separately from successful stop latency', async () => {
  const permission = deferred();
  const harness = createHarness({
    getUserMedia: () => permission.promise,
    performanceTimes: [30, 31],
  });
  const socket = await harness.startRunning({ voice: true, stt: true });
  socket.receive({ type: 'audio', seq: 4, speaker: 'ai_b', text: '재생 중', audio: 'Yg==', cont: false });
  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();
  harness.audio.pauseLeavesPlaying = true;

  const recordingStart = harness.elements.mic.onpointerdown();
  const hold = socket.sent.find((message) => message.type === 'hold');

  assert.equal(hold.audio_stop.outcome, 'pause_failed');
  assert.equal(hold.audio_stop.elapsed_ms, 1);
  assert.equal(hold.audio_stop.segment_id, 'audio-4');

  harness.elements.mic.onpointerup();
  permission.resolve({ getTracks: () => [{ stop() {} }] });
  await recordingStart;
});

test('recorded voice carries the same client event id as its hold', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
    performanceTimes: [40, 40.2],
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  const mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  const hold = socket.sent.find((message) => message.type === 'hold');
  const voice = socket.sent.find((message) => message.type === 'voice');
  assert.equal(hold.client_event_id, 'test-session-1');
  assert.equal(hold.audio_stop.outcome, 'idle');
  assert.equal(voice.client_event_id, hold.client_event_id);
  assert.equal(socket.sent.filter((message) => message.type === 'hold_off').length, 0);

  harness.timeouts[0]();
  assert.equal(harness.elements['session-status'].textContent, '음성 확인이 늦어지고 있어요. 다시 시도해 주세요.');
  await harness.elements.mic.onpointerdown();
  assert.equal(harness.MediaRecorder.instances.length, 2);
  assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 2);
  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  harness.elements.msg.value = '전사 대신 텍스트';
  harness.elements.form.onsubmit({ preventDefault() {} });
  const say = socket.sent.find((message) => message.type === 'say');
  socket.receive({
    type: 'start',
    speaker: 'ai_a',
    after_client_event_id: say.client_event_id,
  });
  await harness.elements.mic.onpointerdown();
  assert.equal(harness.MediaRecorder.instances.length, 3);
  assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 3);
  harness.elements.mic.onpointerup();
});

test('text input supersedes a pending file transcript and ignores its late error', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  const mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();
  const voice = socket.sent.find((message) => message.type === 'voice');

  harness.elements.msg.value = '텍스트로 먼저 계속할게';
  harness.elements.form.onsubmit({ preventDefault() {} });
  assert.equal(harness.elements['session-status'].textContent, '대화 중, 언제든 끼어들 수 있어요.');
  assert.deepEqual(subtitles(harness.elements.log), ['텍스트로 먼저 계속할게']);

  socket.receive({
    type: 'stt_error',
    client_event_id: voice.client_event_id,
    code: 'provider_unavailable',
  });
  assert.equal(harness.elements['session-status'].textContent, '대화 중, 언제든 끼어들 수 있어요.');
  assert.deepEqual(subtitles(harness.elements.log), ['텍스트로 먼저 계속할게']);
});

test('a timed out file transcript cannot disturb the next voice request', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  let mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();
  const firstVoice = socket.sent.find((message) => message.type === 'voice');
  harness.timeouts[0]();

  await harness.elements.mic.onpointerdown();
  mediaRecorder = harness.MediaRecorder.instances[1];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();
  const secondVoice = socket.sent.filter((message) => message.type === 'voice')[1];

  socket.receive({
    type: 'you',
    client_event_id: firstVoice.client_event_id,
    text: '늦은 첫 전사',
  });
  socket.receive({
    type: 'stt_error',
    client_event_id: firstVoice.client_event_id,
    code: 'provider_unavailable',
  });
  assert.equal(harness.elements['session-status'].textContent, '음성을 확인하고 있어요.');
  assert.deepEqual(subtitles(harness.elements.log), ['듣는 중...']);
  assert.equal(
    socket.sent.some(
      (message) => message.type === 'stt_observed'
        && message.client_event_id === firstVoice.client_event_id,
    ),
    false,
  );

  socket.receive({
    type: 'you',
    client_event_id: secondVoice.client_event_id,
    text: '두 번째 전사',
  });
  assert.equal(harness.elements['session-status'].textContent, '대화 중, 언제든 끼어들 수 있어요.');
  assert.deepEqual(subtitles(harness.elements.log), ['두 번째 전사']);
  assert.equal(
    socket.sent.filter(
      (message) => message.type === 'stt_observed'
        && message.client_event_id === secondVoice.client_event_id,
    ).length,
    1,
  );
});

test('a matching file STT error clears pending state and allows another recording', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  const mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();
  const voice = socket.sent.find((message) => message.type === 'voice');

  socket.receive({
    type: 'stt_error',
    client_event_id: voice.client_event_id,
    code: 'empty_transcript',
  });
  assert.equal(harness.elements.log.children.length, 0);
  assert.equal(harness.elements['session-status'].textContent, '음성을 알아듣지 못했어요. 다시 시도해 주세요.');

  await harness.elements.mic.onpointerdown();
  assert.equal(harness.MediaRecorder.instances.length, 2);
  assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 2);
  harness.elements.mic.onpointerup();
  await drainMicrotasks();
});

test('microphone permission failure releases its hold exactly once', async () => {
  const harness = createHarness({
    getUserMedia: () => Promise.reject(new Error('permission denied')),
    performanceTimes: [50, 50.1],
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  harness.elements.mic.onpointerup();
  harness.elements.mic.onpointerleave();

  const hold = socket.sent.find((message) => message.type === 'hold');
  const releases = socket.sent.filter((message) => message.type === 'hold_off');
  assert.equal(hold.audio_stop.outcome, 'idle');
  assert.deepEqual(releases, [{ type: 'hold_off', client_event_id: hold.client_event_id }]);
});

test('late getUserMedia after pointerup stops tracks without creating a recorder', async () => {
  const permission = deferred();
  const harness = createHarness({
    getUserMedia: () => permission.promise,
    performanceTimes: [60, 60.125],
  });
  const socket = await harness.startRunning({ voice: false, stt: true });
  let stopCount = 0;
  const stream = { getTracks: () => [{ stop: () => { stopCount += 1; } }] };

  const recordingStart = harness.elements.mic.onpointerdown();
  await drainMicrotasks();
  harness.elements.mic.onpointerup();
  harness.elements.mic.onpointerleave();
  permission.resolve(stream);
  await recordingStart;

  assert.equal(stopCount, 1);
  assert.equal(harness.MediaRecorder.instances.length, 0);
  const hold = socket.sent.find((message) => message.type === 'hold');
  assert.deepEqual(hold.audio_stop, { outcome: 'idle', elapsed_ms: 0.125 });
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'hold_off'),
    [{ type: 'hold_off', client_event_id: hold.client_event_id }],
  );
  assert.equal(harness.elements['session-status'].textContent, '대화 중, 언제든 끼어들 수 있어요.');
});

test('a short recorded voice releases its hold and explains how to retry', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  const mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(1000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  const hold = socket.sent.find((message) => message.type === 'hold');
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'hold_off'),
    [{ type: 'hold_off', client_event_id: hold.client_event_id }],
  );
  assert.equal(socket.sent.some((message) => message.type === 'voice'), false);
  assert.equal(harness.elements['session-status'].textContent, '조금 더 길게 말해 주세요.');
});

test('hold expiry during microphone permission unlocks the next recording', async () => {
  const firstPermission = deferred();
  let permissionCalls = 0;
  let firstStopCount = 0;
  const firstStream = {
    getTracks: () => [{ stop: () => { firstStopCount += 1; } }],
  };
  const nextStream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => {
      permissionCalls += 1;
      return permissionCalls === 1 ? firstPermission.promise : Promise.resolve(nextStream);
    },
  });
  const socket = await harness.startRunning({
    voice: false,
    stt: false,
    sttStream: PCM_STREAM_CONFIG,
  });

  const firstStart = harness.elements.mic.onpointerdown();
  const firstHold = socket.sent.find((message) => message.type === 'hold');
  socket.receive({
    type: 'stt_error',
    client_event_id: firstHold.client_event_id,
    code: 'hold_expired',
  });
  assert.equal(harness.elements['session-status'].textContent, '음성을 알아듣지 못했어요. 다시 시도해 주세요.');
  firstPermission.resolve(firstStream);
  await firstStart;

  assert.equal(firstStopCount, 1);
  assert.equal(harness.AudioWorkletNode.instances.length, 0);
  await harness.elements.mic.onpointerdown();
  assert.equal(harness.elements['session-status'].textContent, '듣고 있어요. 말이 끝나면 버튼에서 손을 떼세요.');
  assert.equal(harness.AudioWorkletNode.instances.length, 1);
  assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 2);
  harness.elements.mic.onpointerup();
  await drainMicrotasks();
});

test('text input clears a previous microphone error status', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: false, stt: true });
  socket.receive({ type: 'stt_error', code: 'provider_unavailable' });
  assert.equal(harness.elements['session-status'].textContent, '음성 인식 연결에 문제가 생겼어요. 다시 말해 주세요.');

  harness.elements.msg.value = '텍스트로 계속할게';
  harness.elements.form.onsubmit({ preventDefault() {} });

  assert.equal(harness.elements['session-status'].textContent, '대화 중, 언제든 끼어들 수 있어요.');
  assert.equal(socket.sent.at(-1).type, 'say');
});

test('restart uses the new session id and resets the hold sequence', async () => {
  const harness = createHarness({
    getUserMedia: () => Promise.reject(new Error('permission denied')),
    performanceTimes: [70, 70.1, 80, 80.1],
  });
  const firstSocket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  const firstHold = firstSocket.sent.find((message) => message.type === 'hold');
  assert.equal(firstHold.client_event_id, 'test-session-1');

  firstSocket.receive({ type: 'done' });
  await harness.elements['session-start'].click();
  const secondSocket = harness.sockets.at(-1);
  secondSocket.open();
  secondSocket.receive({ type: 'hello', voice: false, stt: true });
  secondSocket.receive({ type: 'session_started', session_id: 'new-session' });

  await harness.elements.mic.onpointerdown();
  const secondHold = secondSocket.sent.find((message) => message.type === 'hold');
  assert.equal(secondHold.client_event_id, 'new-session-1');
});

test('file reader failure releases its hold and removes the pending transcript', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
    performanceTimes: [90, 90.1],
    fileReaderError: true,
  });
  const socket = await harness.startRunning({ voice: false, stt: true });

  await harness.elements.mic.onpointerdown();
  const mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  const hold = socket.sent.find((message) => message.type === 'hold');
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'hold_off'),
    [{ type: 'hold_off', client_event_id: hold.client_event_id }],
  );
  assert.equal(socket.sent.some((message) => message.type === 'voice'), false);
  assert.equal(harness.elements.log.children.length, 0);
  assert.equal(harness.elements['session-status'].textContent, '음성을 준비하지 못했어요. 다시 시도해 주세요.');
});

test('duplicate playback failure callbacks cannot skip the next queued item', async () => {
  const harness = createHarness();
  const socket = await harness.startRunning({ voice: true });
  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '첫째', audio: 'YQ==', cont: false });
  socket.receive({ type: 'audio', seq: 2, speaker: 'ai_b', text: '둘째', audio: 'Yg==', cont: false });

  harness.audio.onerror?.();
  harness.audio.playCalls[1].reject(new Error('same playback failure'));
  await drainMicrotasks();

  assert.equal(harness.audio.playCalls.length, 3, 'the second queued item must start exactly once');
  assert.deepEqual(socket.sent.filter((message) => message.type === 'played'), [{ type: 'played', seq: 1 }]);
  assert.equal(harness.elements['conversation-details'].hidden, false);
  assert.equal(harness.elements['conversation-details'].open, true);
  harness.audio.playCalls[2].resolve();
  await drainMicrotasks();
  assert.deepEqual(subtitles(harness.elements.log), ['첫째', '둘째']);
});

test('stale start and audio after hold stay suppressed until the matching turn starts', async () => {
  const permission = deferred();
  const harness = createHarness({
    getUserMedia: () => permission.promise,
    performanceTimes: [100, 100.1],
  });
  const socket = await harness.startRunning({ voice: true, stt: true });
  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '재생 중', audio: 'YQ==', cont: false });
  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();

  const recordingStart = harness.elements.mic.onpointerdown();
  const hold = socket.sent.find((message) => message.type === 'hold');
  socket.receive({ type: 'start', speaker: 'ai_a' });
  socket.receive({ type: 'audio', seq: 2, speaker: 'ai_a', text: '늦게 도착', audio: 'Yg==', cont: true });
  assert.equal(harness.audio.playCalls.length, 2, 'late audio must not restart playback');
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'played'),
    [{ type: 'played', seq: 1 }, { type: 'played', seq: 2 }],
  );

  socket.receive({
    type: 'start',
    speaker: 'ai_b',
    after_client_event_id: hold.client_event_id,
  });
  socket.receive({ type: 'audio', seq: 3, speaker: 'ai_b', text: '새 응답', audio: 'Yw==', cont: false });
  assert.equal(harness.audio.playCalls.length, 3, 'the next turn may play normally');

  harness.elements.mic.onpointerup();
  permission.resolve({ getTracks: () => [{ stop() {} }] });
  await recordingStart;
});

test('text interruption also waits for its matching start before resuming audio', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
    performanceTimes: [110, 110.1],
  });
  const socket = await harness.startRunning({ voice: true, stt: true });
  await harness.elements.mic.onpointerdown();
  const mediaRecorder = harness.MediaRecorder.instances[0];
  mediaRecorder.ondataavailable?.({ data: new Uint8Array(9000) });
  harness.elements.mic.onpointerup();
  const hold = socket.sent.find((message) => message.type === 'hold');
  harness.elements.msg.value = '텍스트로 끼어든다';

  harness.elements.form.onsubmit({ preventDefault() {} });
  const say = socket.sent.find((message) => message.type === 'say');
  assert.equal(say.client_event_id, 'test-session-2');
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'hold_off'),
    [{ type: 'hold_off', client_event_id: hold.client_event_id }],
  );
  assert.deepEqual(subtitles(harness.elements.log), ['텍스트로 끼어든다']);
  await drainMicrotasks();
  assert.equal(socket.sent.some((message) => message.type === 'voice'), false);

  socket.receive({
    type: 'start',
    speaker: 'ai_a',
    after_client_event_id: hold.client_event_id,
  });
  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '이전 응답', audio: 'YQ==', cont: false });
  assert.equal(harness.audio.playCalls.length, 1, 'stale audio must remain suppressed');

  socket.receive({
    type: 'start',
    speaker: 'ai_b',
    after_client_event_id: say.client_event_id,
  });
  socket.receive({ type: 'audio', seq: 2, speaker: 'ai_b', text: '새 응답', audio: 'Yg==', cont: false });
  assert.equal(harness.audio.playCalls.length, 2, 'matching response may resume playback');

});

test('PCM capability sends hold before stream start', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({ getUserMedia: () => Promise.resolve(stream) });
  const socket = await harness.startRunning({
    voice: false,
    stt: false,
    sttStream: PCM_STREAM_CONFIG,
  });

  assert.equal(harness.AudioContext.instances.length, 1);
  assert.deepEqual(harness.AudioContext.instances[0].moduleUrls, ['/pcm-capture-worklet.js']);

  await harness.elements.mic.onpointerdown();

  const hold = socket.sent.find((message) => message.type === 'hold');
  const start = socket.sent.find((message) => message.type === 'voice_stream_start');
  assert.ok(hold);
  assert.deepEqual(start, {
    type: 'voice_stream_start',
    client_event_id: hold.client_event_id,
    encoding: 'pcm_s16le',
    sample_rate_hz: 24000,
    channels: 1,
  });
  assert.ok(socket.sent.indexOf(hold) < socket.sent.indexOf(start));
  assert.equal(harness.MediaRecorder.instances.length, 0);
  assert.equal(harness.AudioContext.instances[0].requestedSampleRate, 24000);
  assert.deepEqual(harness.AudioContext.instances[0].moduleUrls, ['/pcm-capture-worklet.js']);

  harness.elements.mic.onpointerup();
  await drainMicrotasks();
});

test('PCM chunks use little endian sequence and commit after flush', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({ getUserMedia: () => Promise.resolve(stream) });
  const socket = await harness.startRunning({
    voice: false,
    stt: false,
    sttStream: PCM_STREAM_CONFIG,
  });
  await harness.elements.mic.onpointerdown();

  const node = harness.AudioWorkletNode.instances[0];
  const first = new Float32Array(2400);
  first.set([-1, -0.5, 0, 0.5, 1]);
  const second = new Float32Array(2400).fill(0.25);
  node.emit({ type: 'audio', samples: first });
  node.emit({ type: 'audio', samples: second });
  node.flushSamples = new Float32Array(1200).fill(-0.25);

  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  const hold = socket.sent.find((message) => message.type === 'hold');
  const start = socket.sent.find((message) => message.type === 'voice_stream_start');
  const chunks = socket.sent.filter((message) => message.type === 'voice_stream_chunk');
  const commit = socket.sent.find((message) => message.type === 'voice_stream_commit');
  assert.equal(chunks.length, 3);
  assert.deepEqual(chunks.map((message) => message.sequence_number), [1, 2, 3]);
  assert.deepEqual(
    chunks.map((message) => message.client_event_id),
    [hold.client_event_id, hold.client_event_id, hold.client_event_id],
  );
  assert.deepEqual(decodePcm16(chunks[0].audio).slice(0, 5), [
    -32768,
    -16384,
    0,
    16384,
    32767,
  ]);
  assert.equal(decodePcm16(chunks[0].audio).length, 2400);
  assert.deepEqual(commit, {
    type: 'voice_stream_commit',
    client_event_id: hold.client_event_id,
    final_sequence_number: 3,
    total_samples: 6000,
  });
  assert.equal(start.client_event_id, hold.client_event_id);
  assert.ok(socket.sent.indexOf(chunks[2]) < socket.sent.indexOf(commit));
  assert.deepEqual(node.port.sent.map((message) => message.type), ['flush']);
});

test('final streaming transcript reports release latency with the matching event id', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({
    getUserMedia: () => Promise.resolve(stream),
    performanceTimes: [100, 100.25, 140, 287.5],
  });
  const socket = await harness.startRunning({
    voice: false,
    stt: false,
    sttStream: PCM_STREAM_CONFIG,
  });
  await harness.elements.mic.onpointerdown();

  const node = harness.AudioWorkletNode.instances[0];
  node.emit({ type: 'audio', samples: new Float32Array(2400).fill(0.25) });
  node.emit({ type: 'audio', samples: new Float32Array(2400).fill(0.25) });
  node.flushSamples = new Float32Array(1200).fill(0.25);
  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  const commit = socket.sent.find((message) => message.type === 'voice_stream_commit');
  assert.ok(commit);
  assert.equal(harness.elements['session-status'].textContent, '음성을 확인하고 있어요.');
  socket.receive({
    type: 'you',
    client_event_id: commit.client_event_id,
    text: '최종 전사',
  });

  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'stt_observed'),
    [{
      type: 'stt_observed',
      client_event_id: commit.client_event_id,
      milestone: 'final',
      elapsed_ms: 147.5,
    }],
  );
  assert.equal(harness.elements['session-status'].textContent, '대화 중, 언제든 끼어들 수 있어요.');
});

test('short PCM input releases hold without commit', async () => {
  const stream = { getTracks: () => [{ stop() {} }] };
  const harness = createHarness({ getUserMedia: () => Promise.resolve(stream) });
  const socket = await harness.startRunning({
    voice: false,
    stt: false,
    sttStream: PCM_STREAM_CONFIG,
  });
  await harness.elements.mic.onpointerdown();

  const node = harness.AudioWorkletNode.instances[0];
  node.flushSamples = new Float32Array(100).fill(0.25);
  harness.elements.mic.onpointerup();
  await drainMicrotasks();

  const hold = socket.sent.find((message) => message.type === 'hold');
  assert.deepEqual(
    socket.sent.filter((message) => message.type === 'hold_off'),
    [{ type: 'hold_off', client_event_id: hold.client_event_id }],
  );
  assert.equal(socket.sent.some((message) => message.type === 'voice_stream_commit'), false);
});

test('stale worklet callback after restart sends no audio', async () => {
  let stopCount = 0;
  const stream = { getTracks: () => [{ stop: () => { stopCount += 1; } }] };
  const harness = createHarness({ getUserMedia: () => Promise.resolve(stream) });
  const firstSocket = await harness.startRunning({
    voice: false,
    stt: false,
    sttStream: PCM_STREAM_CONFIG,
  });
  await harness.elements.mic.onpointerdown();

  const context = harness.AudioContext.instances[0];
  const node = harness.AudioWorkletNode.instances[0];
  const staleCallback = node.port.onmessage;
  firstSocket.receive({ type: 'done' });
  await harness.elements['session-start'].click();
  const secondSocket = harness.sockets.at(-1);
  secondSocket.open();
  secondSocket.receive({
    type: 'hello',
    voice: false,
    stt: false,
    stt_stream: PCM_STREAM_CONFIG,
  });
  secondSocket.receive({ type: 'session_started', session_id: 'new-session' });
  const firstCount = firstSocket.sent.length;
  const secondCount = secondSocket.sent.length;

  staleCallback({ data: { type: 'audio', samples: new Float32Array(2400).fill(0.5) } });
  staleCallback({ data: { type: 'flushed' } });

  assert.equal(firstSocket.sent.length, firstCount);
  assert.equal(secondSocket.sent.length, secondCount);
  assert.equal(stopCount, 1);
  assert.equal(context.closeCalls, 1);
  assert.equal(node.disconnected, true);
});

test('PCM worklet emits full chunks and flushes the remainder', () => {
  const posted = [];
  let processorName = null;
  let ProcessorClass = null;

  class FakeAudioWorkletProcessor {
    constructor() {
      this.port = {
        onmessage: null,
        postMessage: (message, transfers = []) => posted.push({ message, transfers }),
      };
    }
  }

  vm.runInNewContext(workletSource, {
    AudioWorkletProcessor: FakeAudioWorkletProcessor,
    registerProcessor: (name, processor) => {
      processorName = name;
      ProcessorClass = processor;
    },
  }, { filename: 'web/pcm-capture-worklet.js' });

  assert.equal(processorName, 'pcm-capture-processor');
  const processor = new ProcessorClass();
  const first = new Float32Array(2000).fill(0.25);
  const second = new Float32Array(1000).fill(-0.25);
  assert.equal(processor.process([[first]]), true);
  assert.equal(posted.length, 0);
  assert.equal(processor.process([[second]]), true);
  assert.equal(posted.length, 1);
  assert.equal(posted[0].message.type, 'audio');
  assert.equal(posted[0].message.samples.length, 2400);
  assert.equal(posted[0].message.samples[1999], 0.25);
  assert.equal(posted[0].message.samples[2000], -0.25);

  processor.port.onmessage({ data: { type: 'flush' } });
  assert.deepEqual(posted.map(({ message }) => message.type), ['audio', 'audio', 'flushed']);
  assert.equal(posted[1].message.samples.length, 600);
  assert.equal(posted[1].message.samples[0], -0.25);
  assert.equal(posted[0].transfers[0], posted[0].message.samples.buffer);
  assert.equal(posted[1].transfers[0], posted[1].message.samples.buffer);

  processor.process([[new Float32Array(2400).fill(0.75)]]);
  assert.deepEqual(posted.map(({ message }) => message.type), ['audio', 'audio', 'flushed']);

  processor.port.onmessage({ data: { type: 'flush' } });
  assert.deepEqual(
    posted.map(({ message }) => message.type),
    ['audio', 'audio', 'flushed', 'flushed'],
  );
});

async function startVad(harness, { voice = false } = {}) {
  const socket = await harness.startRunning({ voice, sttStream: PCM_STREAM_CONFIG, vad: true });
  return { socket, detector: harness.vadControllers.at(-1) };
}

function completeVadSpeech(detector, endAt = 1000) {
  detector.speechStart();
  detector.chunk(new Float32Array(2400).fill(0.25));
  detector.chunk(new Float32Array(2400).fill(0.5));
  detector.chunk(new Float32Array(1200).fill(-0.5));
  detector.speechEnd(endAt);
}

test('VAD readiness waits for microphone startup and tracks capture versus transcript processing', async () => {
  const start = deferred();
  const harness = createHarness({ vadStart: () => start.promise });
  const socket = await harness.startConnecting({ voice: true, sttStream: PCM_STREAM_CONFIG, vad: true });
  const ui = harness.elements;
  assert.equal(socket.sent.length, 0, 'no AI session before the microphone is ready');
  assert.equal(harness.document.body.getAttribute('data-session'), 'starting');
  const detector = harness.vadControllers.at(-1);
  assert.ok(detector);
  assert.equal(detector.startCalls, 1);
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'preparing');
  start.resolve();
  await drainMicrotasks();
  assert.equal(socket.sent.filter((event) => event.type === 'session_start').length, 1);
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'preparing');
  socket.receive({ type: 'session_started', session_id: 'test-session' });
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'listening');
  detector.speechStart();
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'recording');
  for (let index = 0; index < 3; index++) detector.chunk();
  detector.speechEnd();
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'processing');
  const commit = socket.sent.find((message) => message.type === 'voice_stream_commit');
  socket.receive({ type: 'you', client_event_id: commit.client_event_id, text: '자동 감지 발화' });
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'listening');
  await ui.mic.click();
  assert.equal(ui['mic-control'].getAttribute('data-state'), 'off');
});

test('VAD is ready at session start and only claims hold after speech detection', async () => {
  const harness = createHarness({ performanceTimes: [105] });
  const socket = await harness.startRunning({ voice: true, sttStream: PCM_STREAM_CONFIG, vad: true });
  assert.equal(harness.elements.mic.textContent, '마이크 끄기');
  let prevented = 0;
  for (const key of [' ', 'Enter']) {
    const event = { key, repeat: false, preventDefault() { prevented++; } };
    await harness.elements.mic.listeners.get('keydown')(event);
    harness.elements.mic.listeners.get('keyup')(event);
  }
  assert.equal(prevented, 0, 'VAD keyboard activation remains the native button click');
  assert.equal(harness.vadControllers.length, 1);
  assert.equal(socket.sent.some((message) => message.type === 'hold'), false);
  await harness.elements.mic.onpointerdown();
  assert.equal(harness.vadControllers.length, 1);
  const detector = harness.vadControllers[0];
  assert.equal(detector.startCalls, 1);
  assert.equal(socket.sent.some((m) => m.type === 'hold'), false);
  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '재생', audio: 'YQ==' });
  harness.audio.playCalls[1].resolve();
  await drainMicrotasks();
  detector.speechStart(100);
  const hold = socket.sent.find((m) => m.type === 'hold');
  assert.deepEqual(hold.audio_stop, { outcome: 'paused', elapsed_ms: 5, segment_id: 'audio-1' });
  assert.ok(socket.sent.indexOf(hold) < socket.sent.findIndex((m) => m.type === 'voice_stream_start'));
  assert.equal(socket.sent.filter((m) => m.type === 'vad_observed' && m.milestone === 'start').length, 1);
  assert.equal(harness.audio.paused, true);
  assert.equal(harness.elements.mic.getAttribute('aria-pressed'), 'true');
  harness.elements.mic.onpointerup();
  assert.equal(socket.sent.some((m) => m.type === 'voice_stream_commit'), false);
});

test('VAD flushes tail before the shared commit and keeps listening between utterances', async () => {
  const harness = createHarness({ performanceTimes: [100, 1400] });
  const { socket, detector } = await startVad(harness);
  completeVadSpeech(detector);
  const commit = socket.sent.find((m) => m.type === 'voice_stream_commit');
  const chunks = socket.sent.filter((m) => m.type === 'voice_stream_chunk');
  assert.deepEqual(chunks.map((m) => m.sequence_number), [1, 2, 3]);
  assert.equal(decodePcm16(chunks[2].audio).length, 1200);
  assert.equal(commit.total_samples, 6000);
  assert.equal(commit.final_sequence_number, 3);
  assert.ok(socket.sent.indexOf(chunks[2]) < socket.sent.indexOf(commit));
  assert.equal(socket.sent.at(-2).milestone, 'end');
  assert.equal(detector.destroyCalls, 0);
  assert.equal(harness.elements.mic.textContent, '마이크 끄기');
  socket.receive({ type: 'you', text: '첫 발화', client_event_id: commit.client_event_id });
  assert.equal(socket.sent.at(-1).elapsed_ms, 400);
  assert.deepEqual(subtitles(harness.elements.log), ['첫 발화']);
  completeVadSpeech(detector, 2000);
  const commits = socket.sent.filter((m) => m.type === 'voice_stream_commit');
  assert.equal(commits.length, 2);
  assert.notEqual(commits[0].client_event_id, commits[1].client_event_id);
  assert.equal(commits[1].final_sequence_number, 3);
});

test('new VAD speech supersedes a pending final and isolates its late events', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  completeVadSpeech(detector);
  const previous = socket.sent.find((m) => m.type === 'voice_stream_commit').client_event_id;
  detector.speechStart();
  const current = socket.sent.filter((m) => m.type === 'hold').at(-1).client_event_id;
  socket.receive({ type: 'you', text: '늦은 결과', client_event_id: previous });
  socket.receive({ type: 'stt_error', code: 'empty_transcript', client_event_id: previous });
  socket.receive({ type: 'start', speaker: 'ai_a', after_client_event_id: previous });
  socket.receive({ type: 'audio', seq: 1, speaker: 'ai_a', text: '이전 응답', audio: 'YQ==' });
  assert.equal(harness.audio.playCalls.length, 1);
  assert.equal(harness.elements.log.children.length, 0);
  assert.equal(harness.elements.mic.classList.contains('rec'), true);
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'recording');
  detector.chunk();
  assert.equal(socket.sent.at(-1).client_event_id, current);
  assert.equal(detector.resetCalls, 0);
});

test('text submission cancels VAD speech while retaining the listener', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  detector.speechStart();
  detector.chunk();
  const id = socket.sent.find((m) => m.type === 'hold').client_event_id;
  harness.elements.msg.value = '텍스트가 우선';
  harness.elements.form.onsubmit({ preventDefault() {} });
  assert.deepEqual(socket.sent.filter((m) => m.type === 'hold_off'), [{ type: 'hold_off', client_event_id: id }]);
  assert.equal(detector.resetCalls, 1);
  assert.equal(detector.destroyCalls, 0);
  const count = socket.sent.length;
  detector.chunk();
  detector.speechEnd();
  assert.equal(socket.sent.length, count);
  assert.equal(socket.sent.at(-1).type, 'say');
  assert.deepEqual(subtitles(harness.elements.log), ['텍스트가 우선']);
});

test('matching VAD provider error cancels only the utterance and permits another', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  detector.speechStart();
  const id = socket.sent.find((m) => m.type === 'hold').client_event_id;
  socket.receive({ type: 'stt_error', client_event_id: id, code: 'hold_expired' });
  assert.equal(detector.resetCalls, 1);
  assert.equal(detector.destroyCalls, 0);
  assert.equal(harness.elements.mic.classList.contains('rec'), false);
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'listening');
  detector.speechStart();
  assert.equal(socket.sent.filter((m) => m.type === 'hold').length, 2);
});

test('VAD microphone off cancels a pending transcript and suppresses stale callbacks', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  completeVadSpeech(detector);
  const id = socket.sent.find((m) => m.type === 'voice_stream_commit').client_event_id;
  await harness.elements.mic.click();
  assert.equal(detector.destroyCalls, 1);
  assert.equal(harness.elements.mic.textContent, '마이크 켜기');
  assert.equal(harness.elements.mic.getAttribute('aria-pressed'), 'false');
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'off');
  assert.deepEqual(socket.sent.at(-1), { type: 'hold_off', client_event_id: id });
  const count = socket.sent.length;
  detector.speechStart();
  detector.chunk();
  detector.speechEnd();
  socket.receive({ type: 'you', client_event_id: id, text: '취소 후 결과' });
  assert.equal(socket.sent.length, count);
  assert.equal(harness.elements.log.children.length, 0);
});

test('VAD microphone loss cancels speech and allows retry without stale callbacks', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  detector.speechStart();
  detector.chunk();
  const previous = socket.sent.find((m) => m.type === 'hold').client_event_id;
  detector.options.onError(new Error('microphone ended'));
  assert.equal(detector.destroyCalls, 1);
  assert.equal(harness.elements.mic.textContent, '마이크 켜기');
  assert.equal(harness.elements.mic.getAttribute('aria-pressed'), 'false');
  assert.equal(harness.elements.mic.classList.contains('rec'), false);
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'off');
  assert.deepEqual(socket.sent.filter((m) => m.type === 'hold_off'), [
    { type: 'hold_off', client_event_id: previous },
  ]);
  assert.equal(socket.sent.some((m) => m.type === 'voice_stream_commit'), false);

  await harness.elements.mic.click();
  const next = harness.vadControllers.at(-1);
  const count = socket.sent.length;
  detector.options.onError(new Error('late microphone ended'));
  detector.speechStart();
  detector.chunk();
  detector.speechEnd();
  socket.receive({ type: 'you', client_event_id: previous, text: '늦은 결과' });
  assert.equal(socket.sent.length, count);
  assert.equal(next.destroyCalls, 0);
  assert.equal(harness.elements.mic.textContent, '마이크 끄기');
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'listening');
  assert.equal(harness.elements.log.children.length, 0);
  completeVadSpeech(next);
  const commit = socket.sent.find((m) => m.type === 'voice_stream_commit');
  assert.notEqual(commit.client_event_id, previous);
  socket.receive({ type: 'you', client_event_id: commit.client_event_id, text: '다시 말하기' });
  assert.deepEqual(subtitles(harness.elements.log), ['다시 말하기']);
});

test('VAD maximum duration cancels without committing a cut-off utterance', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  detector.speechStart();
  detector.chunk();
  detector.speechEnd(12000, 'max_duration');
  assert.equal(socket.sent.at(-1).type, 'hold_off');
  assert.equal(socket.sent.some((m) => m.type === 'voice_stream_commit'), false);
  assert.equal(socket.sent.some((m) => m.type === 'vad_observed' && m.milestone === 'end'), false);
  assert.equal(detector.resetCalls, 1);
  assert.equal(detector.destroyCalls, 0);
});

test('VAD initialization finishing after microphone off is destroyed without starting', async () => {
  const loading = deferred();
  const harness = createHarness({ vadCreate: async (controller) => {
    await loading.promise;
    return controller;
  } });
  const socket = await harness.startConnecting({ sttStream: PCM_STREAM_CONFIG, vad: true });
  const detector = harness.vadControllers[0];
  assert.ok(detector);
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'preparing');
  await harness.elements.mic.click();
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'off');
  loading.resolve();
  await drainMicrotasks();
  assert.equal(detector.startCalls, 0);
  assert.equal(detector.destroyCalls, 1);
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'off');
  assert.equal(socket.sent.some((m) => m.type === 'hold'), false);
});

test('VAD initialization failure does not start AI and is recoverable through session retry', async () => {
  let attempts = 0;
  const harness = createHarness({ vadCreate: async (controller) => {
    attempts += 1;
    if (attempts === 1) throw new Error('assets unavailable');
    return controller;
  } });
  const socket = await harness.startConnecting({ sttStream: PCM_STREAM_CONFIG, vad: true });
  assert.equal(harness.document.body.getAttribute('data-session'), 'error');
  assert.equal(socket.sent.length, 0);
  assert.ok(socket.closeCalls.length);
  await harness.startRunning({ sttStream: PCM_STREAM_CONFIG, vad: true });
  assert.equal(harness.vadControllers.at(-1).startCalls, 1);
  assert.equal(harness.elements.mic.textContent, '마이크 끄기');
});

test('VAD listener and pending speech cannot leak across a session restart', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  detector.speechStart();
  detector.chunk();
  socket.receive({ type: 'done' });
  assert.equal(detector.destroyCalls, 1);
  await harness.elements['session-start'].click();
  const next = harness.sockets.at(-1);
  next.open();
  next.receive({ type: 'hello', voice: false, stt_stream: PCM_STREAM_CONFIG,
    interaction_mode: 'vad', vad: { engine: 'silero_v5' } });
  await drainMicrotasks();
  next.receive({ type: 'session_started', session_id: 'new' });
  const count = next.sent.length;
  detector.speechStart();
  detector.chunk();
  detector.speechEnd();
  detector.options.onError(new Error('late'));
  assert.equal(next.sent.length, count);
  assert.equal(harness.elements.mic.textContent, '마이크 끄기');
  harness.vadControllers.at(-1).speechStart();
  assert.equal(next.sent.find((m) => m.type === 'hold').client_event_id, 'new-1');
});

test('VAD pending transcript timeout cancels that request without disabling listening', async () => {
  const harness = createHarness();
  const { socket, detector } = await startVad(harness);
  completeVadSpeech(detector);
  const id = socket.sent.find((m) => m.type === 'voice_stream_commit').client_event_id;
  harness.timeouts.at(-1)();
  assert.deepEqual(socket.sent.at(-1), { type: 'hold_off', client_event_id: id });
  assert.equal(detector.destroyCalls, 0);
  assert.equal(harness.elements.log.children.length, 0);
  detector.speechStart();
  socket.receive({ type: 'you', client_event_id: id, text: '늦은 결과' });
  assert.equal(harness.elements.mic.classList.contains('rec'), true);
  assert.equal(harness.elements.log.children.length, 0);
});


for (const failure of ['provider_error', 'timeout']) {
  test(`finished VAD ${failure} preserves the actual segmenter and next speech candidate`, async () => {
    const harness = createHarness({ realSegmenter: true });
    const { socket, detector } = await startVad(harness);
    detector.frames(true, 5);
    detector.frames(false, 19);
    const first = socket.sent.find((event) => event.type === 'voice_stream_commit');
    assert.ok(first);
    assert.equal(detector.segmenter.getState(), 'idle');
    detector.frames(true, 3); // Next speech began before the previous result arrived.
    if (failure === 'provider_error') {
      socket.receive({ type: 'stt_error', client_event_id: first.client_event_id,
        code: 'provider_unavailable' });
    } else {
      harness.timeouts.at(-1)();
    }
    assert.equal(detector.resetCalls, 0);
    detector.frames(true, 2);
    assert.equal(detector.segmenter.getState(), 'speaking');
    assert.equal(socket.sent.filter((event) => event.type === 'hold').length, 2);
    detector.frames(false, 19);
    const commits = socket.sent.filter((event) => event.type === 'voice_stream_commit');
    assert.equal(commits.length, 2);
    assert.notEqual(commits[0].client_event_id, commits[1].client_event_id);
    socket.receive({ type: 'you', client_event_id: commits[1].client_event_id, text: '다시 말한 내용' });
    assert.deepEqual(subtitles(harness.elements.log), ['다시 말한 내용']);
  });
}

test('active VAD failure still waits for silence to avoid splitting the same utterance', async () => {
  const harness = createHarness({ realSegmenter: true });
  const { socket, detector } = await startVad(harness);
  detector.frames(true, 5);
  const first = socket.sent.find((event) => event.type === 'hold');
  socket.receive({ type: 'stt_error', client_event_id: first.client_event_id,
    code: 'provider_unavailable' });
  assert.equal(detector.segmenter.getState(), 'suppressed');
  detector.frames(true, 20);
  assert.equal(socket.sent.filter((event) => event.type === 'hold').length, 1);
  detector.frames(false, 19);
  detector.frames(true, 5);
  assert.equal(socket.sent.filter((event) => event.type === 'hold').length, 2);
});

test('VAD startup ignores premature speech and does not reset an already ended segment', async () => {
  const harness = createHarness({ realSegmenter: true });
  const socket = await harness.startConnecting({ sttStream: PCM_STREAM_CONFIG, vad: true });
  const detector = harness.vadControllers[0];
  detector.frames(true, 5);
  detector.frames(false, 19);
  assert.deepEqual(socket.sent.map((event) => event.type), ['session_start']);
  socket.receive({ type: 'session_started', session_id: 'ready' });
  assert.equal(detector.resetCalls, 0);
  detector.frames(true, 5);
  assert.equal(socket.sent.find((event) => event.type === 'hold').client_event_id, 'ready-1');
});

test('VAD startup cancels an unfinished pre-session segment before accepting fresh speech', async () => {
  const harness = createHarness({ realSegmenter: true });
  const socket = await harness.startConnecting({ sttStream: PCM_STREAM_CONFIG, vad: true });
  const detector = harness.vadControllers[0];
  detector.frames(true, 5);
  socket.receive({ type: 'session_started', session_id: 'ready' });
  assert.equal(detector.segmenter.getState(), 'suppressed');
  detector.frames(true, 5);
  assert.equal(socket.sent.some((event) => event.type === 'hold'), false);
  detector.frames(false, 19);
  detector.frames(true, 5);
  assert.equal(socket.sent.find((event) => event.type === 'hold').client_event_id, 'ready-1');
});

test('microphone startup failure closes the waiting session without starting AI', async () => {
  const harness = createHarness({ vadStart: async () => { throw new Error('NotAllowedError'); } });
  const socket = await harness.startConnecting({ sttStream: PCM_STREAM_CONFIG, vad: true });
  assert.equal(socket.sent.length, 0);
  assert.equal(harness.document.body.getAttribute('data-session'), 'error');
  assert.equal(harness.vadControllers[0].destroyCalls, 1);
  assert.equal(harness.elements['session-start'].disabled, false);
});

test('late microphone readiness after startup cancel cannot start AI in a replacement session', async () => {
  const pending = deferred();
  let attempts = 0;
  const harness = createHarness({ vadStart: async () => { if (++attempts === 1) await pending.promise; } });
  const previous = await harness.startConnecting({ sttStream: PCM_STREAM_CONFIG, vad: true });
  await harness.elements.mic.click();
  const socket = await harness.startRunning({ sttStream: PCM_STREAM_CONFIG, vad: true });
  pending.resolve();
  await drainMicrotasks();
  assert.equal(previous.sent.length, 0);
  assert.equal(socket.sent.filter((event) => event.type === 'session_start').length, 1);
  assert.equal(harness.vadControllers[0].destroyCalls, 1);
  assert.equal(harness.vadControllers[1].destroyCalls, 0);
  assert.equal(harness.elements['mic-control'].getAttribute('data-state'), 'listening');
});
