# CompiledModel (LiteRT Next) spike

Standalone proof-of-concept that the prebuilt LiteRT Next runtime works end-to-end via
Dart FFI, independent of the existing `Interpreter` path. See `../../COMPILEDMODEL_PLAN.md`
for the full plan and the cross-platform symbol/packaging analysis.

## Status (all verified on macOS arm64)

- ✅ `smoke.dart` — lib loads, symbols resolve, CompiledModel created (CPU); Metal GPU
  accelerator auto-registers.
- ✅ `infer.dart` — full tensor I/O; CPU vs GPU(Metal) numerical parity.
- ✅ `bench.dart` — Interpreter vs CompiledModel (CPU / GPU-FP16 / GPU-FP32), warm, sync.
- ✅ `bench_async.dart` — async pipelined throughput (sync vs depth-4/8).

### Findings

**CompiledModel beats the classic Interpreter everywhere** (CPU and GPU). GPU vs Interpreter
(sync, warm, µs/inf): yolov8n 64×, selfie_multiclass seg 62×, pose_heavy 23×, efficientdet
16×, face_detection_back 7.5×.

**Sync vs async is the key subtlety.** Synchronous per-call timing pays full
encode→submit→sync every call and makes the GPU look bad on small models. **Async pipelining
(`LiteRtRunCompiledModelAsync`) gives the GPU 2.8–4.9× throughput; the CPU gains ~nothing.**
This FLIPS the small-model verdict:

| face_detection_short_range | sync | async-8 |
|---|--:|--:|
| GPU/Metal | 773 µs | **158 µs** |
| CPU | 566 µs | 580 µs |

So with an async-pipelined design (= a live camera feed), **GPU wins across the board,
including tiny face models.** Caveat: async numbers are *throughput*, not single-shot
latency, and the benchmark feeds a static input (no per-frame upload cost).

**Async recycle idiom:** `LiteRtGetTensorBufferEvent` is borrowed (do NOT destroy) →
`LiteRtWaitEvent` → `LiteRtClearTensorBufferEvent(buf)` before reusing the output slot.
(`LiteRtSetTensorBufferEvent(buf, null)` returns InvalidArgument — wrong approach.)

**FP16 vs FP32:** GPU defaults to FP16 (fast); FP32 via opaque `gpu_options` TOML
`precision=2`. FP16 faster as expected (yolov8n FP16 1676 vs FP32 2058 µs). Ship FP16
default, FP32 opt-in (GPU-only; CPU is always FP32).

Next: wire CompiledModel into the package (separate `LiteRtNextBindings`, additive), then
measure real end-to-end fps in the `face_detection_tflite` pipeline.

## Running it (macOS arm64)

The prebuilt runtime libs are **not committed** (they're large and platform-specific).
Extract them from the official PyPI wheel:

```sh
cd spike/compiled_model
pip download ai-edge-litert==2.1.5 --no-deps -d /tmp/litert_whl
# (or download the macosx_12_0_arm64 wheel from https://pypi.org/project/ai-edge-litert/#files)
unzip -o /tmp/litert_whl/ai_edge_litert-2.1.5-*macosx*arm64.whl -d /tmp/litert_whl/x
cp /tmp/litert_whl/x/ai_edge_litert/libLiteRt.dylib .
cp /tmp/litert_whl/x/ai_edge_litert/libLiteRtMetalAccelerator.dylib .

dart pub get
dart run bin/smoke.dart [path/to/model.tflite]
```

`smoke.dart` opens `./libLiteRt.dylib` by relative path; the runtime auto-loads the
co-located `libLiteRtMetalAccelerator.dylib`. Default model is the repo's
`example/assets/simple_model.tflite`.

## C API call sequence (from official `litert_cc_sdk` headers)

```
LiteRtCreateEnvironment(0, NULL, &env)
LiteRtCreateOptions(&opts); LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu /*=1*/ | Gpu /*=2*/)
LiteRtCreateModelFromFile("model.tflite", &model)
LiteRtCreateCompiledModel(env, model, opts, &compiled)
// next milestone — tensor I/O:
//   LiteRtGetCompiledModelInputBufferRequirements + LiteRtCreateManagedTensorBufferFromRequirements
//   LiteRtLockTensorBuffer / write / LiteRtUnlockTensorBuffer
//   LiteRtRunCompiledModel(compiled, sigIndex, nIn, inBufs, nOut, outBufs)
LiteRtDestroyCompiledModel / Model / Options / Environment
```
