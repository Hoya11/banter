import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(resolve(HERE, '..', 'index.html'), 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
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
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
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

function createHarness({ getUserMedia, performanceTimes = [0], fileReaderError = false } = {}) {
  const document = new FakeDocument();
  const clock = [...performanceTimes];
  const timeouts = [];
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
  const context = {
    Audio: FakeAudio,
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
    setTimeout: (callback) => {
      timeouts.push(callback);
      return timeouts.length;
    },
  };
  vm.runInNewContext(appSource, context, { filename: 'web/index.html' });

  const elements = Object.fromEntries(
    ['log', 'session-status', 'session-start', 'form', 'msg', 'send', 'mic'].map((id) => [
      id,
      document.getElementById(id),
    ]),
  );
  const audio = FakeAudio.instances[0];

  async function startRunning({ voice = true, stt = false } = {}) {
    await elements['session-start'].click();
    const socket = FakeWebSocket.instances.at(-1);
    assert.ok(socket, 'start must create a WebSocket');
    socket.open();
    socket.receive({ type: 'hello', voice, stt });
    socket.receive({ type: 'session_started', session_id: 'test-session' });
    return socket;
  }

  return {
    audio,
    document,
    elements,
    MediaRecorder: FakeMediaRecorder,
    sockets: FakeWebSocket.instances,
    startRunning,
    timeouts,
  };
}

function subtitles(log) {
  return log.children.map((message) => message.children[1]?.textContent ?? '');
}

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
  await harness.elements.mic.onpointerdown();
  assert.equal(harness.MediaRecorder.instances.length, 1);
  assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 1);

  harness.elements.msg.value = '전사 대신 텍스트';
  harness.elements.form.onsubmit({ preventDefault() {} });
  const say = socket.sent.find((message) => message.type === 'say');
  socket.receive({
    type: 'start',
    speaker: 'ai_a',
    after_client_event_id: say.client_event_id,
  });
  await harness.elements.mic.onpointerdown();
  assert.equal(harness.MediaRecorder.instances.length, 2);
  assert.equal(socket.sent.filter((message) => message.type === 'hold').length, 2);
  harness.elements.mic.onpointerup();
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
  harness.audio.playCalls[2].resolve();
  await drainMicrotasks();
  assert.deepEqual(subtitles(harness.elements.log), ['둘째']);
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
