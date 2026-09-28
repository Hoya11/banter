/* Local Silero V5 adapter. Decision times are configuration, not measured quality. */
(() => {
  'use strict';

  const ASSETS = '/vad-assets/';
  const INPUT_RATE = 16000;
  const FRAME_SAMPLES = 512;
  const CHUNK_SAMPLES = 2400;
  const DEFAULT_CONFIG = Object.freeze({
    startMs: 160, endMs: 600, preRollMs: 256, maxMs: 12000,
    positiveThreshold: 0.6, negativeThreshold: 0.35,
  });
  const MAX_QUEUED_FRAMES = 32;
  let scriptsPromise;

  function script(src) {
    return new Promise((resolve, reject) => {
      const element = document.createElement('script');
      element.src = src;
      element.onload = resolve;
      element.onerror = () => {
        element.remove();
        reject(new Error('VAD 파일을 불러오지 못했습니다. 로컬 자산 준비 상태를 확인해 주세요.'));
      };
      document.head.appendChild(element);
    });
  }

  async function loadDetector() {
    if (!scriptsPromise) {
      scriptsPromise = (async () => {
        await script(`${ASSETS}ort.wasm.min.js`);
        await script(`${ASSETS}vad.bundle.min.js`);
        if (!globalThis.vad?.MicVAD) throw new Error('VAD 초기화 파일이 올바르지 않습니다.');
      })().catch((error) => {
        scriptsPromise = undefined;
        throw error;
      });
    }
    await scriptsPromise;
    return globalThis.vad.MicVAD;
  }

  // Carry interpolation position across frames; 16 kHz becomes exactly 24 kHz.
  // A final repeated endpoint supplies the final fractional sample on flush.
  class StreamResampler {
    constructor(emit) {
      this.emit = emit;
      this.inputCount = 0;
      this.outputCount = 0;
      this.previous = 0;
      this.pending = new Float32Array(CHUNK_SAMPLES);
      this.pendingLength = 0;
    }

    append(value) {
      this.pending[this.pendingLength++] = value;
      this.outputCount++;
      if (this.pendingLength === CHUNK_SAMPLES) this.flushChunk();
    }

    push(frame, current) {
      for (const value of frame) {
        const index = this.inputCount++;
        // Integer thirds avoid accumulating a floating-point timing error.
        while (this.outputCount * 2 <= index * 3) {
          const position = this.outputCount * 2 / 3;
          const fraction = position - (index - 1);
          this.append(index === 0 ? value : this.previous + (value - this.previous) * fraction);
          if (!current()) return;
        }
        this.previous = value;
      }
    }

    flushChunk() {
      if (!this.pendingLength) return;
      const chunk = this.pending.slice(0, this.pendingLength);
      this.pendingLength = 0;
      this.emit(chunk);
    }

    finish(current) {
      const expected = Math.floor(this.inputCount * 3 / 2);
      while (this.outputCount < expected) {
        this.append(this.previous);
        if (!current()) return;
      }
      this.flushChunk();
    }
  }

  function segmentConfig(config) {
    if (!config || Object.prototype.toString.call(config) !== '[object Object]') {
      throw new Error('VAD 설정은 객체여야 합니다.');
    }
    for (const key of Reflect.ownKeys(config)) {
      if (typeof key !== 'string' || !Object.hasOwn(DEFAULT_CONFIG, key)) {
        throw new Error(`알 수 없는 VAD 설정: ${String(key)}`);
      }
    }
    const settings = { ...DEFAULT_CONFIG, ...config };
    for (const [key, value] of Object.entries(settings)) {
      if (!Number.isFinite(value)) throw new Error(`VAD 설정 ${key}는 유한한 수여야 합니다.`);
    }
    for (const key of ['startMs', 'endMs', 'preRollMs', 'maxMs']) {
      const minimum = key === 'preRollMs' ? 0 : key === 'maxMs' ? 32 : 1;
      if (settings[key] < minimum || settings[key] > 60000) {
        throw new Error(`VAD 설정 ${key}는 ${minimum}~60000 ms 범위여야 합니다.`);
      }
    }
    if (settings.positiveThreshold <= 0 || settings.positiveThreshold > 1 ||
        settings.negativeThreshold <= 0 || settings.negativeThreshold >= settings.positiveThreshold) {
      throw new Error('VAD 확률 설정은 0 < negativeThreshold < positiveThreshold <= 1이어야 합니다.');
    }
    const openingSamples = (Math.ceil(settings.startMs / 32) +
      Math.floor(settings.preRollMs / 32)) * FRAME_SAMPLES;
    if (openingSamples > Math.floor(settings.maxMs * INPUT_RATE / 1000)) {
      throw new Error('VAD 최대 길이는 시작 확인과 직전 버퍼를 포함할 수 있어야 합니다.');
    }
    return settings;
  }

  // The live microphone and offline evaluation use the same frame decisions.
  // Only received frames advance time; end-of-file never commits a partial segment.
  function createSegmenter({
    config = {}, isCurrent = () => true, onStart = () => {}, onChunk = () => {}, onEnd = () => {},
  } = {}) {
    const settings = segmentConfig(config);
    const startSamples = settings.startMs * INPUT_RATE / 1000;
    const endSamples = settings.endMs * INPUT_RATE / 1000;
    const preRollSamples = settings.preRollMs * INPUT_RATE / 1000;
    const maxSamples = Math.floor(settings.maxMs * INPUT_RATE / 1000);
    let disposed = false;
    let generation = 0;
    let state = 'idle';
    let speechSamples = 0;
    let silenceSamples = 0;
    let utteranceSamples = 0;
    let preRoll = [];
    let candidate = [];
    let resampler;
    const current = () => !disposed && isCurrent();
    const activeGeneration = (value) => current() && generation === value;

    function clearSegment(nextState) {
      state = nextState;
      speechSamples = 0;
      silenceSamples = 0;
      utteranceSamples = 0;
      preRoll = [];
      candidate = [];
      resampler = undefined;
    }

    function keepPreRoll(frame) {
      preRoll.push(frame);
      while (preRoll.length * FRAME_SAMPLES > preRollSamples) preRoll.shift();
    }

    function end(reason, detectedAt, frameGeneration) {
      const endingResampler = resampler;
      // Reentrant client callbacks may cancel/reset/destroy this very segment.
      endingResampler.finish(() => activeGeneration(frameGeneration));
      if (!activeGeneration(frameGeneration)) return;
      clearSegment(reason === 'max_duration' ? 'suppressed' : 'idle');
      onEnd({ detectedAt, reason });
    }

    function feed(frame, frameGeneration) {
      const remaining = maxSamples - utteranceSamples;
      const audio = frame.subarray(0, remaining);
      utteranceSamples += audio.length;
      resampler.push(audio, () => activeGeneration(frameGeneration));
    }

    function process(probabilities, frame, detectedAt) {
      if (!current()) return;
      const frameGeneration = generation;
      if (!(frame instanceof Float32Array) || frame.length !== FRAME_SAMPLES ||
          !Number.isFinite(probabilities?.isSpeech) || probabilities.isSpeech < 0 ||
          probabilities.isSpeech > 1 || !frame.every(Number.isFinite)) {
        throw new Error('VAD가 올바르지 않은 오디오 프레임을 반환했습니다.');
      }
      const speech = probabilities.isSpeech >= settings.positiveThreshold;
      const silent = probabilities.isSpeech < settings.negativeThreshold;
      if (!Number.isFinite(detectedAt) || detectedAt < 0) {
        throw new Error('VAD 감지 시각은 0 이상의 유한한 수여야 합니다.');
      }

      if (state === 'suppressed') {
        silenceSamples = silent ? silenceSamples + frame.length : 0;
        if (silenceSamples >= endSamples) clearSegment('idle');
        return;
      }
      if (state === 'speaking') {
        feed(frame, frameGeneration);
        if (!activeGeneration(frameGeneration)) return;
        silenceSamples = silent ? silenceSamples + frame.length : 0;
        if (utteranceSamples >= maxSamples) end('max_duration', detectedAt, frameGeneration);
        else if (silenceSamples >= endSamples) end('silence', detectedAt, frameGeneration);
        return;
      }

      if (!speech) {
        // An impulse does not accumulate toward the next start decision.
        for (const buffered of candidate) keepPreRoll(buffered);
        candidate = [];
        speechSamples = 0;
        state = 'idle';
        keepPreRoll(frame.slice());
        return;
      }
      state = 'candidate';
      candidate.push(frame.slice());
      speechSamples += frame.length;
      if (speechSamples < startSamples) return;

      state = 'speaking';
      silenceSamples = 0;
      resampler = new StreamResampler(onChunk);
      const opening = [...preRoll, ...candidate];
      preRoll = [];
      candidate = [];
      onStart({ detectedAt });
      if (!activeGeneration(frameGeneration)) return;
      for (const buffered of opening) {
        feed(buffered, frameGeneration);
        if (!activeGeneration(frameGeneration)) return;
      }
      if (utteranceSamples >= maxSamples) end('max_duration', detectedAt, frameGeneration);
    }

    return {
      process,
      reset() {
        if (disposed) return;
        generation++;
        clearSegment('suppressed');
      },
      destroy() {
        if (disposed) return;
        disposed = true;
        generation++;
        clearSegment('suppressed');
      },
      getState: () => state,
    };
  }

  async function create({ audioContext, isCurrent, onStart, onChunk, onEnd, onError }) {
    const MicVAD = await loadDetector();
    let source;
    let disposed = false;
    let started = false;
    let startPromise;
    let destroyPromise;
    let generation = 0;
    let processingGeneration = -1;
    let queuedFrames = 0;
    let queue = Promise.resolve();
    let failed = false;
    const streams = new Set();
    const trackEndHandlers = new Map();
    const current = () => !disposed && isCurrent();
    const activeGeneration = (value) => current() && generation === value;

    const segmenter = createSegmenter({ isCurrent: current, onStart, onChunk, onEnd });

    function stopTracks() {
      for (const [track, handler] of trackEndHandlers) {
        track.removeEventListener('ended', handler);
      }
      trackEndHandlers.clear();
      for (const stream of streams) {
        for (const track of stream.getTracks()) track.stop();
      }
    }

    function fail(error) {
      if (disposed || failed) return;
      failed = true;
      const report = isCurrent();
      void controller.destroy();
      if (report) onError(error instanceof Error ? error : new Error(String(error)));
    }

    function processProbabilities(probabilities, frame) {
      if (!current()) {
        void controller.destroy();
        return;
      }
      if (!started || processingGeneration !== generation) return;
      try {
        segmenter.process(probabilities, frame, performance.now());
      } catch (error) {
        fail(error);
      }
    }

    const controller = {
      start() {
        if (disposed || !isCurrent()) return Promise.resolve();
        if (startPromise) return startPromise;
        started = true;
        startPromise = source.start().then(() => {
          if (!current()) void controller.destroy();
        }).catch((error) => {
          fail(error);
          throw error;
        });
        return startPromise;
      },
      reset() {
        if (disposed) return;
        generation++;
        segmenter.reset();
      },
      destroy() {
        if (destroyPromise) return destroyPromise;
        disposed = true;
        started = false;
        generation++;
        segmenter.destroy();
        stopTracks();
        destroyPromise = (async () => {
          await startPromise?.catch(() => {});
          await queue;
          stopTracks();
          if (!source) return;
          try {
            await source.destroy();
          } catch {
            // vad-web 0.0.30 destroy assumes successful microphone/node setup.
            // Release a loaded model even when permission/worklet setup failed.
            await source.model?.release().catch(() => {});
          } finally {
            source._mediaStreamAudioSourceNode?.disconnect();
            source._vadNode?.disconnect();
            source._vadNode?.port?.close();
            streams.clear();
          }
        })();
        return destroyPromise;
      },
    };

    const getStream = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: true, autoGainControl: true, noiseSuppression: true,
      } });
      streams.add(stream);
      if (!current()) {
        stopTracks();
        throw new Error('VAD 시작 요청이 취소되었습니다.');
      }
      for (const track of stream.getTracks()) {
        const onEnded = () => fail(new Error('마이크 입력이 종료되었습니다. 다시 켜 주세요.'));
        trackEndHandlers.set(track, onEnded);
        track.addEventListener('ended', onEnded, { once: true });
        // Permission or device loss can precede handler registration.
        if (track.readyState === 'ended') {
          throw new Error('마이크 입력이 종료되었습니다. 다시 켜 주세요.');
        }
      }
      return stream;
    };

    try {
      source = await MicVAD.new({
        audioContext, model: 'v5', startOnLoad: false, processorType: 'AudioWorklet',
        baseAssetPath: ASSETS, onnxWASMBasePath: ASSETS,
        ortConfig: (ort) => {
          ort.env.logLevel = 'error';
          ort.env.wasm.numThreads = 1;
          ort.env.wasm.proxy = false;
        },
        getStream, resumeStream: getStream,
        pauseStream: async (stream) => { for (const track of stream.getTracks()) track.stop(); },
        onFrameProcessed: processProbabilities,
        onSpeechStart() {}, onSpeechRealStart() {}, onSpeechEnd() {}, onVADMisfire() {},
        positiveSpeechThreshold: DEFAULT_CONFIG.positiveThreshold,
        negativeSpeechThreshold: DEFAULT_CONFIG.negativeThreshold,
        minSpeechMs: DEFAULT_CONFIG.startMs, redemptionMs: DEFAULT_CONFIG.endMs, preSpeechPadMs: 0,
      });
      // Pinned MicVAD's worklet dispatch does not serialize stateful ONNX runs.
      // Bound queued work and its unused completed-segment audio retention.
      const processFrame = source.processFrame.bind(source);
      source.processFrame = (frame) => {
        if (!current()) {
          void controller.destroy();
          return Promise.resolve();
        }
        if (++queuedFrames > MAX_QUEUED_FRAMES) {
          queuedFrames--;
          fail(new Error('VAD 처리가 오디오 입력을 따라가지 못했습니다. 다시 시작해 주세요.'));
          return Promise.resolve();
        }
        const frameGeneration = generation;
        queue = queue.then(async () => {
          if (!activeGeneration(frameGeneration)) return;
          processingGeneration = frameGeneration;
          await processFrame(frame);
        }).catch(fail).finally(() => {
          // The library's speech callbacks are unused; preserve model state only.
          if (source.frameProcessor) source.frameProcessor.audioBuffer = [];
          queuedFrames--;
        });
        return queue;
      };
      if (!current()) await controller.destroy();
      return controller;
    } catch (error) {
      await controller.destroy();
      throw error;
    }
  }

  globalThis.BanterVadCapture = Object.freeze({ create, createSegmenter, DEFAULT_CONFIG });
})();
