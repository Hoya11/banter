const MAX_FILE_BYTES = 128 * 1024 * 1024;
const MAX_DURATION_SECONDS = 5 * 60;

function invalid(reason) {
  throw new Error(`Invalid WAV: ${reason}`);
}

/**
 * Decode a bounded RIFF/WAVE file without resampling or normalizing its volume.
 * Signed integer PCM is scaled to [-1, 1); finite IEEE float samples are preserved.
 * Stereo input is converted to mono by averaging its two channels per frame.
 */
export function decodeWav(buffer) {
  if (!(buffer instanceof Uint8Array)) {
    throw new TypeError('WAV input must be a Uint8Array or Buffer');
  }
  if (buffer.byteLength > MAX_FILE_BYTES) invalid('file exceeds 128 MiB');
  if (buffer.byteLength < 12) invalid('truncated RIFF header');
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const tag = (offset) => String.fromCharCode(
    view.getUint8(offset), view.getUint8(offset + 1),
    view.getUint8(offset + 2), view.getUint8(offset + 3),
  );
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
    invalid('expected little-endian RIFF/WAVE');
  }
  const riffEnd = view.getUint32(4, true) + 8;
  if (riffEnd !== buffer.byteLength || riffEnd < 12) {
    invalid('RIFF size does not match the input length');
  }

  let format = null;
  let data = null;
  let offset = 12;
  while (offset < riffEnd) {
    if (riffEnd - offset < 8) invalid('truncated chunk header');
    const name = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const start = offset + 8;
    const end = start + size;
    const paddedEnd = end + (size % 2);
    if (paddedEnd > riffEnd) invalid('chunk or padding exceeds RIFF bounds');
    if (name === 'fmt ') {
      if (format !== null) invalid('multiple fmt chunks');
      if (size < 16 || size === 17) invalid('invalid fmt chunk size');
      if (size >= 18 && view.getUint16(start + 16, true) !== size - 18) {
        invalid('fmt extension size does not match its chunk');
      }
      const encoding = view.getUint16(start, true);
      if (encoding === 0xfffe) invalid('WAVE_FORMAT_EXTENSIBLE is not supported');
      const channels = view.getUint16(start + 2, true);
      const sampleRate = view.getUint32(start + 4, true);
      const byteRate = view.getUint32(start + 8, true);
      const blockAlign = view.getUint16(start + 12, true);
      const bitsPerSample = view.getUint16(start + 14, true);
      if (encoding !== 1 && encoding !== 3) invalid('unsupported sample encoding');
      if (channels !== 1 && channels !== 2) invalid('only mono and stereo are supported');
      if (sampleRate < 16_000 || sampleRate > 96_000) {
        invalid('sample rate must be between 16000 and 96000 Hz');
      }
      if ((encoding === 1 && ![16, 24, 32].includes(bitsPerSample)) ||
          (encoding === 3 && bitsPerSample !== 32)) {
        invalid('expected PCM16/24/32 or IEEE float32 samples');
      }
      if (blockAlign !== channels * (bitsPerSample / 8) ||
          byteRate !== sampleRate * blockAlign) {
        invalid('inconsistent block alignment or byte rate');
      }
      format = { encoding, channels, sampleRate, blockAlign, bitsPerSample };
    } else if (name === 'data') {
      if (data !== null) invalid('multiple data chunks');
      data = { start, size };
    }
    offset = paddedEnd;
  }
  if (format === null || data === null) invalid('missing fmt or data chunk');
  if (data.size === 0) invalid('empty audio data');
  const { encoding, channels, sampleRate, blockAlign, bitsPerSample } = format;
  if (data.size % blockAlign !== 0) invalid('data contains an incomplete sample frame');
  const frameCount = data.size / blockAlign;
  if (frameCount > sampleRate * MAX_DURATION_SECONDS) {
    invalid('audio duration exceeds 5 minutes');
  }
  const bytesPerSample = bitsPerSample / 8;
  const readSample = (position) => {
    if (encoding === 3) {
      const value = view.getFloat32(position, true);
      if (!Number.isFinite(value)) invalid('non-finite IEEE float sample');
      return value;
    }
    if (bitsPerSample === 16) return view.getInt16(position, true) / 0x8000;
    if (bitsPerSample === 32) return view.getInt32(position, true) / 0x80000000;
    const unsigned = view.getUint8(position) |
      (view.getUint8(position + 1) << 8) | (view.getUint8(position + 2) << 16);
    const signed = unsigned & 0x800000 ? unsigned - 0x1000000 : unsigned;
    return signed / 0x800000;
  };
  const samples = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame++) {
    const position = data.start + frame * blockAlign;
    const left = readSample(position);
    samples[frame] = channels === 1
      ? left
      : (left + readSample(position + bytesPerSample)) / 2;
  }
  return {
    samples, sampleRate, channels, bitsPerSample,
    durationMs: frameCount / sampleRate * 1000,
  };
}
