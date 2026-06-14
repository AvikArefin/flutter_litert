# CompiledModel (LiteRT Next): measured results (macOS arm64)

All numbers from the spikes in `bin/` (`dart run`, warm loops, median µs unless noted),
LiteRT `ai-edge-litert` 2.1.5. Interpreter side = bundled classic `libtensorflowlite_c-mac`
(TF 2.20.0), so Interpreter-vs-CompiledModel multiples mix "GPU" + "newer runtime"; the
**GPU-vs-CompiledModel-CPU** comparison is the clean one.

## Headline (`bin/sweep.dart`, 32 models)

- **Median CompiledModel GPU-async vs Interpreter-XNNPACK: ~19×.**
- **Median CompiledModel-CPU vs Interpreter-XNNPACK: ~2×.**

Selected (median µs):

| model | size | interp_xnn | cm_cpu | gpu_sync | gpu_async | best |
|---|--|--:|--:|--:|--:|--:|
| selfie_multiclass (seg) | 16M | 98,425 | 17,058 | 1,606 | 1,349 | 73× |
| yolov8n | 12M | 107,199 | 20,129 | 1,700 | 1,488 | 72× |
| efficientdet_lite2 | 22M | 83,609 | 25,109 | 2,586 | 2,391 | 35× |
| pose_landmark_heavy | 26M | 49,506 | 16,551 | 2,091 | 1,866 | 27× |
| face_detection_back | 308K | 5,588 | 3,786 | 768 | 285 | 20× |
| face_detection_short_range | 224K | 927 | 575 | 784 | 187 | 5× |

Rule: **GPU wins big above ~1 to 2 ms of compute; tiny sub-ms models favor CPU.** It's *compute*,
not category: `face_detection_back` (heavy) wins on GPU, `face_detection_front` (tiny) on CPU.

## Async (`bin/async_probe.dart`)

Single-shot `runAsync` (event-poll) is ~2× faster than sync on GPU and ~matches a depth-8
pipeline; CPU runs synchronously (asyncOut=0). e.g. face_detection_back GPU: sync ~610 to 850,
async ~293 to 303, pipeline8 ~290. → single-shot async captures the win; no streaming API needed.

## Isolate (`bin/isolate_probe.dart`)

Metal works inside a worker isolate. Cross-isolate overhead: plain `SendPort.send` ~+230 µs on
face_detection_back, `TransferableTypedData` ~+20 µs (near-free). Isolate value = UI-thread
offload, not inference speed.

## Accelerator fallback (`bin/fallback_probe.dart` + `python/`)

Google's `HardwareAccelerator` docstring: *"Using GPU or NPU alone may fail… For robust
execution, combine with CPU as fallback: `GPU | CPU`."* Confirmed: GPU-only (mask 2) returns
`504 (Compilation)` for several models; **GPU|CPU (mask 3) makes them compile/run**:

| model | CPU | GPU(2) | GPU\|CPU(3) |
|---|--:|--:|--:|
| mobilefacenet | 2309 | 504 | 626 (3.7× faster) |
| species_classifier_f16 | 1152 | 504 | 1434 |
| superanimal_ssdlite_f16 | 4972 | 504 | 5895 |
| superanimal_rtmpose_f16 | 5363 | 504 | 11350 |

Fallback makes them **run**, not always **fast** (partition overhead can exceed the GPU gain).

## Diagnosis: the failures are NOT dtype (`bin/dtype_analysis.dart`)

Every failing model has **Float32 I/O, no quantization** (cross-checked via the classic TFLite C
API too). The `504`s are unsupported GPU ops (`DEQUANTIZE`, `RELU_0_TO_1`, `L2_NORMALIZATION`),
fixed by fallback; not a dtype problem. Status: `3 = RuntimeFailure`, `504 = Compilation`
(confirmed via `LiteRtGetStatusString`).

## Two confirmed upstream bugs (reproduced via Google's own Python API, `python/`)

NOT our FFI; reproduced with plain `CompiledModel.from_file(model, hardware_accel=GPU|CPU)`:
- **`face_detection_full_range_sparse`** → SIGABRT: `DENSIFY: Operation is not supported` →
  uncaught `std::bad_optional_access`. Crashes even with CPU fallback. Classic Metal runs it fine.
- **`gesture_embedder`** → `RuntimeError` at `litert_compiled_model.h:1486`: GATHER shape mismatch
  / Metal kernel `Unable to parse bc coord for BATCH axis`. Fails even with fallback. Classic Metal
  runs it fine.

(Repros staged for reporting; not filed.)


## Host-memory I/O: official, working, opt-in (`bin/hostmem_probe2.dart`, 2026-06-10)

Official host-side zero-copy uses `LiteRtCreateTensorBufferFromHostMemory`: the
caller owns a `LITERT_HOST_MEMORY_BUFFER_ALIGNMENT`-aligned buffer for the whole
tensor-buffer lifetime, and LiteRT wraps it at buffer creation. This is distinct
from holding a transient lock pointer open while preprocessing/decoding; that
lock-callback experiment was removed because it crashed under concurrent
multi-model Metal load.

`hostmem_probe2.dart` fixes the earlier flawed probe by using a fresh
`CompiledModel` per buffer mode and binding one buffer kind for the model's
whole lifetime. Outputs were bit-identical to managed buffers in all successful
modes, but performance was model/accelerator-dependent:

| model | accelerator | managed+lock | host-memory | result |
|---|---|--:|--:|---|
| face_detection_back | CPU | 4020 µs | 4060 µs | host slightly slower |
| face_detection_back | GPU strict | 906 µs | 1395 µs | host slower |
| face_detection_back | GPU\|CPU | 685 µs | 723 µs | host slightly slower |
| face_detection_short_range | CPU | 605 µs | 651 µs | host slower |
| face_detection_short_range | GPU strict | 485 µs | 867 µs | host much slower |
| face_detection_short_range | GPU\|CPU | 817 µs | 538 µs | host faster |

⇒ Decision: package API exposes `TensorBufferMode.hostMemory` as an **opt-in**
mode; `TensorBufferMode.managed` remains the default. Host memory is worth
benchmarking per model/accelerator, not enabling globally.

### Public API `runAsync()` + direct host-memory check (`bin/buffer_mode_bench.dart`)

This uses the actual Dart `CompiledModel.fromFile(..., tensorBufferMode: ...)`
API. Each row is 3 repeats, 150 measured iterations + 30 warmup per repeat,
using rotated path order and median-of-medians. Outputs matched exactly
(`max|diff|=0`) wherever both modes compiled.

Paths:
- `managed.run`: default managed buffers + `runAsync()`.
- `host.run`: official host-memory buffers + `runAsync()` (still returns output
  copies).
- `host.direct`: official host-memory buffers + `writeInput` →
  `dispatchAsync` → `readOutput` callbacks, so callers can consume output
  without per-run output copies.

| model | accelerator | managed.run | host.run | host.direct | direct vs managed | result |
|---|---|--:|--:|--:|--:|---|
| species_classifier_float16 | CPU | 1134 µs | 1144 µs | 1115 µs | 1.017× | small direct win |
| species_classifier_float16 | GPU strict | 504 | 504 | 504 | n/a | all fail strict GPU |
| species_classifier_float16 | GPU\|CPU | 1276 µs | 1276 µs | 1286 µs | 0.992× | direct slower |
| superanimal_ssdlite_float16 | CPU | 4972 µs | 4920 µs | 4860 µs | 1.023× | small direct win |
| superanimal_ssdlite_float16 | GPU strict | 504 | 504 | 504 | n/a | all fail strict GPU |
| superanimal_ssdlite_float16 | GPU\|CPU | 6126 µs | 6160 µs | 6058 µs | 1.011× | small direct win |
| superanimal_rtmpose_s_float16 | CPU | 5483 µs | 5478 µs | 5414 µs | 1.013× | small direct win |
| superanimal_rtmpose_s_float16 | GPU strict | 504 | 504 | 504 | n/a | all fail strict GPU |
| superanimal_rtmpose_s_float16 | GPU\|CPU | 10171 µs | 10159 µs | 10254 µs | 0.992× | direct slower |
| face_detection_back | CPU | 4104 µs | 4109 µs | 4033 µs | 1.018× | small direct win |
| face_detection_back | GPU strict | 451 µs | 472 µs | 507 µs | 0.890× | managed wins |
| face_detection_back | GPU\|CPU | 450 µs | 467 µs | 502 µs | 0.896× | managed wins |
| selfie_segmenter | CPU | 1689 µs | 1667 µs | 1692 µs | 0.998× | neutral/slower |
| selfie_segmenter | GPU strict | 710 µs | 699 µs | 727 µs | 0.977× | managed/host.run win |
| selfie_segmenter | GPU\|CPU | 544 µs | 578 µs | 603 µs | 0.902× | managed wins |

⇒ Updated decision: `TensorBufferMode.hostMemory` and the direct
`writeInput`/`dispatchAsync`/`readOutput` API stay **opt-in only**. They are
correct and sometimes shave ~1 to 2% on CPU/fallback-heavy models, but they are
not a safe default and are often slower on Metal GPU. Managed buffers remain
the default for GPU-heavy face/segmentation models.

The animal models are fallback-heavy on Metal (`DEQUANTIZE` /
`RELU_0_TO_1` unsupported), so host memory does not change the main bottleneck.
For true GPU zero-copy, the next frontier is GPU-native buffer interop (camera
textures / platform GPU buffers), not host-memory wrapping from Dart arrays.

## Async wait: blocking `LiteRtWaitEvent(-1)` replaces event polling (2026-06-10)

Verified against the official LiteRT source (clone at `_litert_spike/LiteRT`,
commit ea79caf): `LiteRtTensorBufferT::Lock` waits on an attached event with
timeout -1 (`tensor_buffer.cc:988`), sync Run waits the same way
(`compiled_model.cc:1817 to 1833`), and **no official code polls with timeout 0**.
`runAsync` therefore now does a blocking `LiteRtWaitEvent(event, -1)` per
output instead of the previous `waitEvent(0)` + `Future.delayed(Duration.zero)`
poll loop (which also had an unbounded-loop edge case).

Measured in-app (macOS arm64, debug, `cm_inference_only_test`, median ms,
GPU|CPU `runAsync`): back 0.435 → 0.468, face_landmark 0.493 → 0.416, iris
0.647 → 0.680; parity within noise. Sync `run()`'s extra ~0.9 ms on GPU is
NOT the event wait (it persists either way); async dispatch + event wait stays
~3× faster than sync `run()` on Metal even when the wait blocks.

## In-app per-model engine table (`cm_inference_only_test`, median ms)

| model | interp | cm CPU run | cm GPU\|CPU run | cm GPU\|CPU runAsync |
|---|--:|--:|--:|--:|
| face_detection_back | 6.16 | 4.11 | 1.36 | **0.44** |
| face_landmark | 1.39 | 0.84 | 0.57 | **0.49** |
| iris_landmark | 1.47 | **0.50** | 1.09 | 0.65 |

⇒ Detection + mesh belong on GPU|CPU async; **iris (64×64) is fastest on
CompiledModel-CPU**, consistent with the "GPU wins above ~1 to 2 ms compute"
rule. face_detection_tflite now pins iris to `{cpu}`.

## Whole-pipeline A/B variance lesson (2026-06-10)

The same `compiledmodel_ab_test` on the same code measured CM fast-mode at
0.85× (midday, loaded machine: interp fast mean 14.8 ms, heavy mean/median
skew) and 1.31× (quiet machine: interp fast mean 6.2 ms, mean≈median). The
interpreter engine was unchanged between runs; absolute numbers and
cross-run ratios from this benchmark are unreliable; only the within-run
paired ratio on a quiet machine is meaningful. Post-change quiet-machine runs
(two consecutive, ratios reproduce within ±0.03): fast 1.31×/1.32×, full
1.33×/1.30× (1 face); 1.03×/1.04×, 1.06×/1.04× (4 faces, decode-dominated);
mesh deviation unchanged (1.61 px); 300-iteration stress pass.

## iOS: packaging + accelerator discovery + API-version pin (2026-06-11)

Verified against the LiteRT source (clones under `_litert_spike/`):

- **Accelerator discovery is filename-based with no usable process fallback for
  the Metal plugin.** `RegisterGpuAccelerator`
  (`litert/runtime/accelerators/gpu_registry.cc`) dlopen's
  `<RuntimeLibraryDir>/libLiteRtMetalAccelerator.dylib` and looks up the
  exported `LiteRtAcceleratorImpl` def (probe with
  `try_default_on_failure=false`). The RTLD_DEFAULT fallback only applies to a
  `LiteRtRegisterGpuAccelerator` function symbol, which the official Metal
  accelerator does NOT export (checked on macOS + iOS prebuilts). ⇒ The
  accelerator file must keep its exact dylib name on disk; pre-dlopen'ing it
  ourselves cannot replace the directory scan.
- **iOS packaging: library-type xcframeworks** (`xcodebuild -create-xcframework
  -library libLiteRt.dylib ...`), NOT `.framework` bundles. Frameworks force a
  binary rename (no `lib` prefix / `.dylib` suffix), which breaks the scan;
  bare-dylib xcframework slices are embedded by Xcode/SPM into
  `Runner.app/Frameworks/` under their original names (verified in a simulator
  build). The Dart loader opens `<app>/Frameworks/libLiteRt.dylib` and passes
  that directory as `kLiteRtEnvOptionTagRuntimeLibraryDir`.
- **iOS prebuilts must be pinned to the v2.1.5-era C API.**
  `google-ai-edge/LiteRT` commit `1ac2a58f` (2026-06-01) added a leading
  `LiteRtEnvironment` parameter to `LiteRtCreateModelFromFile/FromBuffer`;
  prebuilts updated after that date crash v2.1.5-shaped bindings
  (`fromFile` → status 500 FileIO, `fromBuffer` → SIGSEGV in the flatbuffer
  verifier; both reproduced on the simulator with `bc426d8` binaries).
  Pin: commit `1adc2475829fbe52d5670873821a45bea8779532` (2026-05-28, last
  prebuilt update before the change). All 38 bound symbols exist there and all
  bound-function signatures are byte-identical to v2.1.5
  (`.github/workflows/build-litert-ios.yml` defaults to this ref).
- **iOS-simulator Metal event wedge (sequence-dependent):** closing a Metal
  model that ran `runAsync` and then waiting on an async event from a freshly
  created environment can hang forever in
  `MTLSimSharedEvent waitUntilSignaledValue` (sync XPC to MTLSimDriver; no
  signal ever arrives). Reproduced twice with GPU|CPU-async → close →
  strict-GPU-async; sync-then-async sequences pass. Simulator-only driver
  path (MTLSimDriver); real devices use a different Metal stack. The
  integration test keeps the fallback test sync-only so CI stays
  deterministic.

## Android: CompiledModel CPU path (2026-06-11)

- Bundled `libLiteRt.so` from the same API-pinned prebuilt commit (`1adc2475`,
  arm64-v8a + x86_64, both 16 KB page-size aligned) via a Gradle
  download-at-build task (release `litert-android-v1.0.0`), additively next to
  the classic Maven `litert:1.4.1` artifacts. Loader dlopens the bare soname.
- All 38 bound symbols are exported (with `@@VERS_1.0` symbol versioning;
  plain-name dlsym still resolves them).
- **The OpenCL/GL GPU accelerator is deliberately NOT bundled yet:** when
  `libLiteRtClGlAccelerator.so` registers in an environment without working
  OpenCL (every emulator), `LiteRtCreateCompiledModel` with GPU|CPU fails with
  status 3 (RuntimeFailure) instead of falling back to CPU; registered-but-
  broken GPU is worse than no GPU. Re-add once validated on real hardware
  (single `include()` line in android/build.gradle.kts).
- Emulator (Android 16 arm64, Pixel 8 AVD): CPU inference, fromBuffer/runAsync
  parity, host-memory parity and GPU|CPU-fallback-to-CPU all pass; strict GPU
  skips.
