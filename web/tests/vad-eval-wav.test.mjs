import assert from 'node:assert/strict';
import test from 'node:test';

import { decodeWav } from '../scripts/vad-eval-wav.mjs';

function chunk(name, payload, padding = true) {
  const header = Buffer.alloc(8);
  header.write(name, 0, 'ascii');
  header.writeUInt32LE(payload.length, 4);
  return Buffer.concat([header, payload, Buffer.alloc(padding ? payload.length % 2 : 0)]);
}

function riff(chunks) {
  const contents = Buffer.concat([Buffer.from('WAVE'), ...chunks]);
  const header = Buffer.alloc(8);
  header.write('RIFF', 0, 'ascii');
  header.writeUInt32LE(contents.length, 4);
  return Buffer.concat([header, contents]);
}

function format({ encoding = 1, channels = 1, rate = 16_000, bits = 16 } = {}) {
  const bytes = Buffer.alloc(16);
  bytes.writeUInt16LE(encoding, 0);
  bytes.writeUInt16LE(channels, 2);
  bytes.writeUInt32LE(rate, 4);
  bytes.writeUInt32LE(rate * channels * (bits / 8), 8);
  bytes.writeUInt16LE(channels * (bits / 8), 12);
  bytes.writeUInt16LE(bits, 14);
  return bytes;
}

function audio(values, { encoding = 1, bits = 16 } = {}) {
  const bytes = Buffer.alloc(values.length * bits / 8);
  values.forEach((value, index) => {
    const offset = index * bits / 8;
    if (encoding === 3) bytes.writeFloatLE(value, offset);
    else bytes.writeIntLE(value, offset, bits / 8);
  });
  return bytes;
}

function wav(values, options = {}) {
  return riff([chunk('fmt ', format(options)), chunk('data', audio(values, options))]);
}

test('PCM16 decodes exact normalized samples and reports original metadata', () => {
  const result = decodeWav(wav([-32768, -16384, 0, 16384, 32767]));
  assert.ok(result.samples instanceof Float32Array);
  assert.deepEqual([...result.samples], [-1, -0.5, 0, 0.5, 32767 / 32768]);
  assert.equal(result.sampleRate, 16000);
  assert.equal(result.channels, 1);
  assert.equal(result.bitsPerSample, 16);
  assert.equal(result.durationMs, 5 / 16000 * 1000);
});

test('PCM24 sign extension preserves negative and positive low bits', () => {
  const values = [-8388608, -4194304, -1, 0, 1, 4194304, 8388607];
  const result = decodeWav(wav(values, { bits: 24 }));
  assert.deepEqual([...result.samples], values.map(value => Math.fround(value / 8388608)));
  assert.equal(result.bitsPerSample, 24);
});

test('PCM32 decodes full-scale signed integers to Float32', () => {
  const values = [-2147483648, -1073741824, -1, 0, 1, 1073741824, 2147483647];
  const result = decodeWav(wav(values, { bits: 32, rate: 96000 }));
  assert.deepEqual([...result.samples], values.map(value => Math.fround(value / 2147483648)));
  assert.equal(result.sampleRate, 96000);
});

test('IEEE float32 preserves finite samples without volume normalization or clipping', () => {
  const values = [-2, -1, -0.125, 0, 0.5, 1, 2];
  const result = decodeWav(wav(values, { encoding: 3, bits: 32, rate: 44100 }));
  assert.deepEqual([...result.samples], values);
  assert.equal(result.sampleRate, 44100);
});

test('stereo is averaged per frame while original channel count is reported', () => {
  const result = decodeWav(wav([-32768, 16384, 16384, 0, 8192, -8192], { channels: 2 }));
  assert.deepEqual([...result.samples], [-0.25, 0.25, 0]);
  assert.equal(result.channels, 2);
  assert.equal(result.durationMs, 3 / 16000 * 1000);
});

test('float stereo averages large finite values without intermediate Float32 overflow', () => {
  const large = Math.fround(3e38);
  const result = decodeWav(wav([large, large], { encoding: 3, bits: 32, channels: 2 }));
  assert.equal(result.samples[0], large);
});

test('Uint8Array subviews decode only their selected bytes', () => {
  const input = wav([16384]);
  const backing = new Uint8Array(input.length + 11);
  backing.set(input, 5);
  assert.deepEqual([...decodeWav(backing.subarray(5, 5 + input.length)).samples], [0.5]);
  assert.throws(() => decodeWav(input.buffer), /Uint8Array or Buffer/);
});

test('data may precede fmt, and unknown chunks and odd padding are skipped', () => {
  const result = decodeWav(riff([
    chunk('JUNK', Buffer.from([3, 4, 5])),
    chunk('data', audio([-16384])),
    chunk('LIST', Buffer.from([1])),
    chunk('fmt ', format()),
  ]));
  assert.deepEqual([...result.samples], [-0.5]);
});

test('float format can declare an empty extension and a fact chunk', () => {
  const fmt = Buffer.concat([format({ encoding: 3, bits: 32 }), Buffer.alloc(2)]);
  const result = decodeWav(riff([
    chunk('fmt ', fmt), chunk('fact', Buffer.from([1, 0, 0, 0])),
    chunk('data', audio([0.25], { encoding: 3, bits: 32 })),
  ]));
  assert.equal(result.samples[0], 0.25);
});

test('rejects missing fmt, missing data, and empty audio', () => {
  for (const input of [riff([]), riff([chunk('fmt ', format())]), riff([chunk('data', audio([0]))])]) {
    assert.throws(() => decodeWav(input), /missing fmt or data/);
  }
  assert.throws(() => decodeWav(wav([])), /empty audio/);
});

test('rejects multiple fmt or data chunks even if the duplicate is empty', () => {
  assert.throws(() => decodeWav(riff([
    chunk('fmt ', format()), chunk('fmt ', format()), chunk('data', audio([0])),
  ])), /multiple fmt/);
  assert.throws(() => decodeWav(riff([
    chunk('fmt ', format()), chunk('data', audio([0])), chunk('data', Buffer.alloc(0)),
  ])), /multiple data/);
});

test('rejects truncated headers, wrong signatures, and declared RIFF length mismatches', () => {
  for (const size of [0, 4, 11]) assert.throws(() => decodeWav(Buffer.alloc(size)), /RIFF header/);
  for (const signature of ['RIFX', 'RF64']) {
    const input = wav([0]);
    input.write(signature, 0, 'ascii');
    assert.throws(() => decodeWav(input), /little-endian RIFF\/WAVE/);
  }
  const wrongForm = wav([0]);
  wrongForm.write('AVI ', 8, 'ascii');
  assert.throws(() => decodeWav(wrongForm), /RIFF\/WAVE/);
  for (const declared of [0, 3, 4, 0xffffffff]) {
    const input = wav([0]);
    input.writeUInt32LE(declared, 4);
    assert.throws(() => decodeWav(input), /RIFF size/);
  }
  assert.throws(() => decodeWav(Buffer.concat([wav([0]), Buffer.from([0])])), /RIFF size/);
  assert.throws(() => decodeWav(wav([0]).subarray(0, -1)), /RIFF size/);
});

test('rejects incomplete chunk headers, payloads, and absent odd padding', () => {
  assert.throws(() => decodeWav(riff([Buffer.from('bad')])), /chunk header/);
  const oversized = chunk('data', audio([0]));
  oversized.writeUInt32LE(0xffffffff, 4);
  assert.throws(() => decodeWav(riff([chunk('fmt ', format()), oversized])), /RIFF bounds/);
  assert.throws(() => decodeWav(riff([
    chunk('fmt ', format()), chunk('data', audio([0])), chunk('JUNK', Buffer.from([1]), false),
  ])), /padding exceeds RIFF bounds/);
});

test('rejects malformed fmt sizes and explicit WAVE_FORMAT_EXTENSIBLE', () => {
  for (const size of [0, 15, 17]) {
    assert.throws(() => decodeWav(riff([
      chunk('fmt ', Buffer.alloc(size)), chunk('data', audio([0])),
    ])), /fmt chunk size/);
  }
  const badExtension = Buffer.concat([format(), Buffer.from([5, 0])]);
  assert.throws(() => decodeWav(riff([
    chunk('fmt ', badExtension), chunk('data', audio([0])),
  ])), /extension size/);
  const extensible = Buffer.alloc(40);
  format({ encoding: 0xfffe }).copy(extensible);
  extensible.writeUInt16LE(22, 16);
  assert.throws(() => decodeWav(riff([
    chunk('fmt ', extensible), chunk('data', audio([0])),
  ])), /WAVE_FORMAT_EXTENSIBLE is not supported/);
});

test('rejects unsupported encodings, channel counts, bit depths, and sample rates', () => {
  for (const options of [
    { encoding: 6 }, { channels: 0 }, { channels: 3 },
    { bits: 8 }, { encoding: 3, bits: 16 }, { rate: 0 }, { rate: 15999 }, { rate: 96001 },
  ]) {
    assert.throws(() => decodeWav(riff([
      chunk('fmt ', format(options)), chunk('data', audio([0])),
    ])), /Invalid WAV/);
  }
});

test('rejects inconsistent block alignment, byte rate, and partial stereo frames', () => {
  for (const field of [8, 12]) {
    const fmt = format();
    if (field === 8) fmt.writeUInt32LE(1234, field);
    else fmt.writeUInt16LE(1, field);
    assert.throws(() => decodeWav(riff([
      chunk('fmt ', fmt), chunk('data', audio([0])),
    ])), /block alignment or byte rate/);
  }
  assert.throws(() => decodeWav(riff([
    chunk('fmt ', format({ channels: 2 })), chunk('data', audio([0, 1, 2])),
  ])), /incomplete sample frame/);
});

test('rejects non-finite float values in either stereo channel', () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    for (const values of [[value, 0], [0, value]]) {
      assert.throws(() => decodeWav(wav(values, { encoding: 3, bits: 32, channels: 2 })), /non-finite/);
    }
  }
});

test('rejects more than five minutes before allocating decoded samples', () => {
  // This duration-boundary fixture stays in memory; no audio file is created.
  const input = riff([
    chunk('fmt ', format()), chunk('data', Buffer.alloc((16000 * 300 + 1) * 2)),
  ]);
  assert.throws(() => decodeWav(input), /duration exceeds 5 minutes/);
});

test('rejects files above 128 MiB before reading their header', () => {
  assert.throws(() => decodeWav(new Uint8Array(128 * 1024 * 1024 + 1)), /exceeds 128 MiB/);
});
