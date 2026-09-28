import { copyFile, mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const web = new URL('../', import.meta.url);
const output = new URL('vendor/vad/', web);
const packages = [
  ['@ricky0123/vad-web', '0.0.30', [
    ['dist/bundle.min.js', 'vad.bundle.min.js'],
    ['dist/bundle.min.js.LICENSE.txt', 'vad.bundle.LICENSE.txt'],
    ['dist/vad.worklet.bundle.min.js', 'vad.worklet.bundle.min.js'],
    ['dist/silero_vad_v5.onnx', 'silero_vad_v5.onnx'],
  ]],
  ['onnxruntime-web', '1.22.0', [
    ['dist/ort.wasm.min.js', 'ort.wasm.min.js'],
    ['dist/ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.mjs'],
    ['dist/ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.wasm'],
  ]],
];

await mkdir(output, { recursive: true });
for (const [name, version, files] of packages) {
  const root = new URL(`node_modules/${name}/`, web);
  const pkg = JSON.parse(await readFile(new URL('package.json', root), 'utf8'));
  if (pkg.version !== version) throw new Error(`Expected ${name} ${version}, found ${pkg.version}`);
  for (const [source, destination] of files) {
    await copyFile(new URL(source, root), new URL(destination, output));
  }
}
// The npm tarballs omit their root LICENSE files. Preserve the upstream texts:
// vad-web 0.0.30 (ISC) and bundled Silero model (MIT):
// https://github.com/ricky0123/vad/blob/d08d8be6604cc81aa367ea10218b8120c4505c04/LICENSE
// ONNX Runtime 1.22.0 (MIT):
// https://github.com/microsoft/onnxruntime/blob/v1.22.0/LICENSE
for (const license of ['vad-web-LICENSE.txt', 'onnxruntime-LICENSE.txt']) {
  await copyFile(new URL(`licenses/${license}`, web), new URL(license, output));
}
console.log(`Prepared local VAD assets in ${fileURLToPath(output)}`);
