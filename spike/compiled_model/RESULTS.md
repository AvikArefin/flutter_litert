# CompiledModel (LiteRT Next) — measured results (macOS arm64)

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

Rule: **GPU wins big above ~1–2 ms of compute; tiny sub-ms models favor CPU.** It's *compute*,
not category — `face_detection_back` (heavy) wins on GPU, `face_detection_front` (tiny) on CPU.

## Async (`bin/async_probe.dart`)

Single-shot `runAsync` (event-poll) is ~2× faster than sync on GPU and ~matches a depth-8
pipeline; CPU runs synchronously (asyncOut=0). e.g. face_detection_back GPU: sync ~610–850,
async ~293–303, pipeline8 ~290. → single-shot async captures the win; no streaming API needed.

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
fixed by fallback — not a dtype problem. Status: `3 = RuntimeFailure`, `504 = Compilation`
(confirmed via `LiteRtGetStatusString`).

## Two confirmed upstream bugs (reproduced via Google's own Python API — `python/`)

NOT our FFI — reproduced with plain `CompiledModel.from_file(model, hardware_accel=GPU|CPU)`:
- **`face_detection_full_range_sparse`** → SIGABRT: `DENSIFY: Operation is not supported` →
  uncaught `std::bad_optional_access`. Crashes even with CPU fallback. Classic Metal runs it fine.
- **`gesture_embedder`** → `RuntimeError` at `litert_compiled_model.h:1486`: GATHER shape mismatch
  / Metal kernel `Unable to parse bc coord for BATCH axis`. Fails even with fallback. Classic Metal
  runs it fine.

(Repros staged for reporting; not filed.)
