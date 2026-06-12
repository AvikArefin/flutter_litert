# CompiledModel (LiteRT Next) Support — Implementation Plan

> ## ⚑ CURRENT STATUS (authoritative — 2026-06-09)
>
> The sections below this banner are **historical exploration** (the "Scenario B / unified
> runtime replacement / resolved on every platform" language is **superseded** — we are NOT
> replacing the classic runtime). This banner is the canonical state.
>
> **Approach: purely ADDITIVE.** A separate `libLiteRt` runtime + accelerator plugin is bundled
> *alongside* the untouched classic `libtensorflowlite_c`; CompiledModel uses its own
> `litert_loader` + hand-written `litert_ffi` bindings and a `CompiledModel` class parallel to
> `Interpreter`. The Interpreter path is unchanged.
>
> **Done & verified (macOS arm64):**
> - `CompiledModel.fromFile(path, {Set<Accelerator> accelerators, Precision precision}).run()/runAsync()/close()`
>   — sync + non-blocking async; `Accelerator{cpu,gpu,npu}`, `Precision{fp16,fp32}`.
> - **Accelerator selection = `Set<Accelerator>`** OR'd into the native bitmask, matching Google's
>   `HardwareAccelerator` IntFlag (unordered set, default `{cpu}`). `{gpu,cpu}` = robust fallback
>   (mask 3, the documented pattern); `{gpu}` = strict (mask 2).
> - macOS `libLiteRt.dylib` + `libLiteRtMetalAccelerator.dylib` bundled (podspec **and** SwiftPM);
>   `flutter analyze` clean; smoke test passes (CPU + fallback; strict-GPU skips headless).
>
> **Measured results:** see `spike/compiled_model/RESULTS.md` (GPU 19–73× on heavy models, async
> ~2×, fallback fixes the `504` op-coverage failures, dtype is NOT the cause, 2 confirmed upstream
> bugs). Reusable probes in `spike/compiled_model/bin/` + `python/`.
>
> **Next:** (1) wire CompiledModel into the `face_detection` app + `flutter run` (real GPU/stage-timer
> proof); (2) Linux/Windows bundling via download-at-build; (3) later polish: dtype-general I/O
> (not currently blocking), `IsolateCompiledModel`, input-shape accessors. **Not doing:**
> unified-runtime replacement; from-source builds.
>
> **Update 2026-06-11 — iOS + cross-platform CI landed:**
> - **iOS WORKS on BOTH channels (simulator-verified, incl. strict-GPU Metal).** Binaries
>   pinned to LiteRT commit `1adc2475` (last prebuilt with the v2.1.5 C API — newer prebuilts
>   add a `LiteRtEnvironment` param to `LiteRtCreateModelFromFile/FromBuffer` and crash
>   v2.1.5-shaped bindings; see RESULTS.md). Release: `litert-ios-v1.0.0`. Rebuild workflow:
>   `.github/workflows/build-litert-ios.yml`.
>   - **SwiftPM:** library-type (bare-dylib) xcframeworks via `binaryTarget` url+checksum;
>     loader dlopens `<app>/Frameworks/libLiteRt.dylib` and the Metal accelerator
>     auto-registers via the RuntimeLibraryDir file-name scan.
>   - **CocoaPods:** rejects bare-dylib xcframeworks, so it vendors framework-wrapped
>     variants (`ios/LiteRt.xcframework` + `ios/LiteRtMetalAccelerator.xcframework`, built by
>     `scripts/wrap_litert_ios_frameworks.sh`, committed to git, pub-ignored + podspec
>     download). The framework rename breaks the accelerator scan, so
>     `ios/Classes/litert_gpu_accelerator_shim.c` exports `LiteRtRegisterGpuAccelerator`
>     (found by the runtime's RTLD_DEFAULT probe) and registers the def manually; it must be
>     anchored via `FlutterLitertRetainLiteRtGpuShim()` in the plugin or the linker drops it.
>   - Known: iOS-simulator-only Metal event wedge on async→close→async sequences
>     (MTLSimDriver; see RESULTS.md) — integration test keeps the fallback test sync-only.
> - **Windows 76-byte struct fix DONE** (`LiteRtLayout`/`LiteRtRankedTensorType` are now opaque
>   with per-ABI sizes; MSVC packs the rank/has_strides bitfields differently).
> - **Android CompiledModel CPU path WORKS (emulator-verified).** `libLiteRt.so` from the
>   same pinned commit (arm64-v8a + x86_64, 16 KB-aligned, release `litert-android-v1.0.0`)
>   bundled via a Gradle download-at-build task in `android/build.gradle.kts`, additively
>   next to the classic Maven artifacts; loader dlopens the bare soname. The OpenCL/GL GPU
>   accelerator is deliberately NOT bundled yet: registered-but-broken OpenCL (every
>   emulator) makes even GPU|CPU compilation fail with RuntimeFailure instead of falling
>   back — re-add after real-device GPU validation (single `include()` line).
> - **CI added** (`flutter-ci.yml`) — all five platforms now run real CompiledModel
>   inference: macOS host tests, Linux + Windows smoke tests against `libLiteRt` extracted
>   from the official PyPI `ai-edge-litert==2.1.5` wheel (`LITERT_LIB_PATH`), an
>   iOS-simulator integration-test job, and an Android-emulator integration-test job
>   (ubuntu + KVM, API 34 x86_64).
> - **Still pending for iOS:** real-device run + archive/TestFlight validation (embedded-dylib
>   signing on the SwiftPM channel; the CocoaPods channel ships conventional frameworks).

---

## Goal

Add support for the **LiteRT Next `CompiledModel` API** as an **additive** inference
path alongside the existing classic `Interpreter` API. Interpreter stays exactly as it
is; `CompiledModel` is a new, parallel runtime users can opt into — primarily to unlock
**GPU acceleration on desktop (Windows/Linux)** and zero-copy / async I/O where it
actually helps.

## Mental model (important — corrects a common assumption)

The intuition "the Interpreter stuff all stays, CompiledModel just adds on top" is
**definitely correct at the package/Dart level**. At the **native level it is an open
question** that must be resolved empirically (see Spike 0) — do NOT treat either layout
below as settled fact.

- ✅ **Package/Dart level (certain):** purely additive. We keep `Interpreter`,
  `InterpreterOptions`, all delegates, all existing bindings. We add a new `CompiledModel`
  class + supporting types next to them. No breaking changes.
> **🔎 FINDING (Android, resolved 2026-06-08):** Scenario B is **confirmed on Android.**
> The Maven artifact `com.google.ai.edge.litert:litert:2.1.5` ships a single unified
> `jni/<abi>/libLiteRt.so` whose string table contains the **full classic interpreter C
> API** (`TfLiteInterpreterCreate`, `TfLiteInterpreterAllocateTensors`, `TfLiteModelCreate`,
> … — 46 `TfLiteInterpreter*` names) **and** the **full CompiledModel C API**
> (`LiteRtCreateCompiledModel`, `LiteRtDestroyCompiledModel`, `LiteRtCompiledModel*`,
> `TensorBuffer`, `Environment` — 25 `LiteRtCompiledModel*` names). The GPU accelerator is
> a **separate plugin** `libLiteRtClGlAccelerator.so` loaded by the runtime. So on Android:
> *replace* the classic lib with the unified `libLiteRt.so` (superset) + add the
> accelerator plugin.
> Caveats: (1) this is the **Android** artifact only — Android ABIs (arm64-v8a, armeabi-v7a,
> x86_64); it says **nothing about desktop** Windows/Linux/macOS distribution, which is
> still open and is the real risk for the Windows-GPU goal. (2) Evidence is via `strings`
> (symbol-name presence), not a true `readelf --dyn-syms` export dump — confirm before
> betting weeks. (3) Repo currently pins `litert:1.4.1`; latest is **2.1.5** (2026-05-13) —
> bump regardless.

- ❓ **Native level (resolved for Android = B; UNKNOWN for desktop — verify per platform):**
  there are two plausible packaging models, and they imply very different work:

  **Scenario A — separate libraries (additive).** The classic `libtensorflowlite_c`
  stays untouched; LiteRT Next ships as an *additional* runtime library
  (`libLiteRtRuntimeCApi-*`) + accelerator plugins bundled alongside it.
  ```
  libtensorflowlite_c-*       (classic Interpreter C API)   ← unchanged
  libLiteRtRuntimeCApi-*      (LiteRT Next CompiledModel)    ← NEW, added alongside
  libLiteRtGpuAccelerator-*   (GPU backend plugin)           ← NEW, platform-dependent
  ```

  **Scenario B — one unified LiteRT runtime (replacement).** LiteRT is the rebrand of
  TensorFlow Lite, and the modern LiteRT runtime may expose **both** the legacy
  `TfLiteInterpreter` C API **and** the new `LiteRtCompiledModel` API in a **single**
  library. In that case we *replace* the current `libtensorflowlite_c-*` with a newer
  unified LiteRT runtime that is a superset — closer to the original "regenerate the one
  lib with CompiledModel added on" intuition.
  ```
  libLiteRt-* (or unified libtensorflowlite_c-*)  exposes BOTH:
     - TfLiteInterpreter C API  (Interpreter path, unchanged behavior)
     - LiteRtCompiledModel C API (new path)
  + accelerator plugin(s) as above
  ```

  **Evidence currently leans toward B on Android:** Android already pulls a *single*
  Maven artifact — `com.google.ai.edge.litert:litert:1.4.1` — not a classic + next pair.
  That suggests a unified runtime there. Desktop may differ.

  **The deciding factor** is whether the newer unified runtime is symbol/ABI-compatible
  enough to drop in as a replacement for the TF 2.20.0-era `libtensorflowlite_c` this repo
  bundles *without regressing the Interpreter path*. If yes → Scenario B (replace). If the
  classic API drifts/breaks → Scenario A (keep old, add new). **Resolve this by inspecting
  exported symbols of the candidate artifacts before committing to a bundling approach.**

- (Windows GPU only) `dxil.dll` / `dxcompiler.dll` — D3D shader compiler DLLs, needed
  under either scenario.

## Current native packaging (as of this plan)

| Platform | Classic lib source | How bundled |
|----------|-------------------|-------------|
| Android  | Maven `com.google.ai.edge.litert:litert:1.4.1` + `litert-gpu:1.4.1` | Gradle dependency |
| iOS      | `TensorFlowLiteC.xcframework` (+ Metal, CoreML) | podspec / SwiftPM |
| macOS    | `libtensorflowlite_c-mac.dylib` (+ gpu, coreml) | podspec vendored |
| Windows  | `libtensorflowlite_c-win.dll` | CMake bundled |
| Linux    | `libtensorflowlite_c-linux.so` | CMake bundled |

**Key observation:** Android already consumes the **new LiteRT Maven packages** (1.4.1),
which plausibly already contain the LiteRT Next / CompiledModel APIs. That makes **Android
the cheapest first target** to prove the Dart + FFI shape end-to-end before touching
desktop native packaging.

---

## The gating question (do this FIRST — it decides everything)

> **For each target platform, can we obtain a consumable, prebuilt LiteRT Next runtime +
> GPU accelerator binary, or must we build it from source?**

> **🔎 FINDING (Desktop, resolved 2026-06-08): PREBUILTS EXIST. No source build needed.**
> The PyPI package **`ai-edge-litert` 2.1.5** ships native desktop wheels for
> `win_amd64`, `manylinux_2_27_x86_64`, `manylinux_2_27_aarch64`, and `macosx_12_0_arm64`.
> Inside the **Windows wheel**:
> - `libLiteRt.dll` (5.6 MB) — **unified runtime with BOTH APIs** (`TfLiteInterpreter*` ×40,
>   `LiteRtCompiledModel*` ×17 incl. `LiteRtCreateCompiledModel`).
> - `libLiteRtWebGpuAccelerator.dll` (22 MB) — **real Windows GPU accelerator**: WebGPU via
>   **Dawn** (702 refs), backed by **D3D12** (183) and Vulkan (198); dxil/dxcompiler refs
>   (26). Exports `LiteRtWebGpuAccelerator`.
> The **Linux wheel** has the same pair (`libLiteRt.so` + `libLiteRtWebGpuAccelerator.so`)
> plus NPU vendor plugins (Qualcomm/MediaTek/Samsung/Intel OpenVINO/Google Tensor).
> ⇒ Desktop GPU via CompiledModel is a **fetch-extract-bundle-FFI** job, NOT a from-source
> build. This overturns the earlier doc-based assumption that desktop GPU prebuilts weren't
> published.
> **Remaining confirmations:** (1) verify Apache-2.0 license covers redistributing the
> binaries extracted from the wheels (almost certainly yes); (2) `dumpbin /exports` on
> `libLiteRt.dll` to confirm the symbols are truly *exported* (strings only proves name
> presence); (3) confirm runtime deps — likely needs `dxil.dll` / `dxcompiler.dll`
> alongside the GPU accelerator; (4) the `.pyd` Python wrappers are separable — we bundle
> only the plain C/C++ shared libs.

> **🔎 FINDING (iOS, resolved 2026-06-08): iOS is READY — it moved to Swift Package
> Manager, not abandoned.** The old CocoaPods nightly pods (`TensorFlowLiteC`, `LiteRTC`)
> froze ~2025-06 because iOS migrated to an in-repo Swift package. In `google-ai-edge/LiteRT`:
> - `litert/swift/Sources/` = full Swift API incl. **`CompiledModel.swift`**,
>   `Environment.swift`, `Model.swift`, `TensorBuffer.swift`, `Options.swift`, `LiteRtC.h`.
> - `litert/prebuilt/ios_arm64/` + `ios_sim_arm64/` = prebuilt **`libLiteRt.dylib`** +
>   **`libLiteRtMetalAccelerator.dylib`** (Metal GPU), stored as Git LFS.
> iOS is distributed via GitHub repo / SPM / Git-LFS binaries — NOT PyPI/Maven, so it has
> no "2.1.5" coordinate but is current.

### Prebuilt availability matrix (all 5 platforms have runtime + GPU accelerator)

| Platform | Runtime | GPU accelerator | Channel |
|----------|---------|-----------------|---------|
| Android | `libLiteRt.so` | `libLiteRtClGlAccelerator.so` (OpenCL/GL) | Maven `litert:2.1.5` |
| Windows | `libLiteRt.dll` | `libLiteRtWebGpuAccelerator.dll` (Dawn/D3D12) | PyPI `ai-edge-litert` 2.1.5 |
| Linux | `libLiteRt.so` | `libLiteRtWebGpuAccelerator.so` (Dawn/D3D12/Vulkan) | PyPI 2.1.5 |
| macOS arm64 | `libLiteRt.dylib` | (in arm64 wheel — verify Metal vs WebGPU) | PyPI 2.1.5 (**arm64 only, no Intel**) |
| iOS | `libLiteRt.dylib` | `libLiteRtMetalAccelerator.dylib` (Metal) | GitHub repo / SPM (Git LFS) |

Also: GitHub releases publish `litert_cc_sdk.zip` per version (v2.1.5, 2026-05-18) — a C++
SDK, another desktop integration option besides the PyPI wheels.

Version state: stable = **2.1.5**; **2.2.0** exists only as PyPI nightlies
(`2.2.0.dev20260607`). Repo currently pins Android `litert:1.4.1` → bump to 2.1.5.

This was previously the single biggest risk and rate-limiter on the project. **It is now
resolved in our favor on every platform.** If a target ever lacks a prebuilt, that platform
falls back to a from-source build decision — but none do today.

**Spike 0 — artifact availability + packaging model (1–2 days, do before any code):**
1. Android: confirm whether `litert:1.4.1` Maven artifact exposes the CompiledModel C API
   symbols (it's the easiest win — already a dependency).
2. Desktop: determine whether prebuilt LiteRT Next runtime + GPU accelerator libs exist
   for Windows/Linux/macOS, or if they must be built from the LiteRT source tree.
3. **Resolve Scenario A vs B (the native packaging question):** dump exported symbols of
   the candidate LiteRT artifacts and the current classic libs. Does a single unified
   LiteRT runtime expose BOTH `TfLite*` (interpreter) and `LiteRt*` (compiled model)
   symbols? Is it a safe drop-in replacement for the TF 2.20.0-era `libtensorflowlite_c`
   without regressing the Interpreter path? → answers whether we *replace* or *add*.
   - macOS/Linux: `nm -gU` / `nm -D` on the `.dylib`/`.so`.
   - Windows: `dumpbin /exports` on the `.dll`.
4. Inventory required runtime deps (e.g. Windows GPU path needs `dxil.dll` /
   `dxcompiler.dll`; GPU backend is WebGPU/D3D12-based).

**Decision gate:** Only proceed to native bundling on platforms where a consumable
artifact exists. Everywhere else, scope down to "build-from-source research spike" and
decide separately whether it's worth it.

---

## Phasing

### Phase 1 — Android proof of concept (lowest risk)
Rationale: the new LiteRT runtime is already a Gradle dependency, so no new native
distribution work is needed to validate the API design.

- [ ] Confirm CompiledModel symbols are reachable from `litert:1.4.1`.
- [ ] Write minimal FFI bindings for the core LiteRT Next C API:
  `LiteRtEnvironment`, `LiteRtModel` (load from file/buffer), `LiteRtCompiledModel`
  (create with accelerator selection), `LiteRtTensorBuffer`, invoke, read outputs,
  destroy/cleanup.
- [ ] Build a tiny internal (non-public) smoke test: load a `.tflite`, create a
  CompiledModel with GPU accelerator, run one inference, compare output vs Interpreter.
- [ ] Validate on a real Android device (GPU path actually executes, not silent CPU
  fallback — see lessons from the GPU delegate fallback work).

### Phase 2 — Public Dart API design
Keep it parallel to Interpreter, not bolted into it. CompiledModel is **not** a
`TfLiteDelegate*`, so it must **not** go through `InterpreterOptions.addDelegate()`.

Sketch:
```dart
final model = LiteRtModel.fromAsset('model.tflite');   // or fromBytes/fromFile
final compiled = LiteRtCompiledModel(
  model,
  accelerator: Accelerator.gpu,   // gpu | cpu | npu (platform-dependent)
);
final outputs = await compiled.run(inputs);            // async-first
compiled.delete();
model.delete();
```
- [ ] Decide async-first vs sync (CompiledModel supports async dispatch — a real
  advantage; lean async).
- [ ] Decide `TensorBuffer` exposure: hide it behind a simple `run()` for v1, expose
  zero-copy buffers later if profiling justifies it.
- [ ] Conditional exports mirroring the existing pattern
  (`compiled_model_native.dart` / `_web.dart` / `_unsupported.dart`).
- [ ] Web: `LiteRtInterpreter` (LiteRT.js) already covers the WebGPU CompiledModel-style
  path — decide whether to unify naming or keep separate.

### Phase 3 — Desktop native bundling (gated by Spike 0)
Only for platforms with a consumable artifact.
- [ ] Linux: add LiteRt runtime `.so` + GPU accelerator `.so` to `linux/CMakeLists.txt`
  `bundled_libraries` (additive — leave classic `.so` in place).
- [ ] Windows: add LiteRt runtime `.dll` + GPU accelerator `.dll` **+ `dxil.dll` /
  `dxcompiler.dll`** to `windows/CMakeLists.txt` bundling.
- [ ] macOS: add LiteRt runtime `.dylib` to podspec `vendored_libraries` (Metal-backed;
  note GPU likely no faster than existing Metal delegate — see Phase 5).
- [ ] Mirror the existing prebuilt-binary distribution approach (GitHub Releases +
  fetch-at-build, like the Flex delegate DLL).

### Phase 4 — `PerformanceConfig` integration
- [ ] Decide whether `PerformanceConfig.gpu()` can transparently route Windows/Linux to
  the CompiledModel GPU path while keeping Interpreter+XNNPACK as the default/fallback.
- [ ] Graceful fallback: if CompiledModel GPU init fails, fall back to Interpreter (and
  surface it — do NOT silently swallow, per the GPU fallback lessons).

### Phase 5 — Validation, benchmarking, CI
- [ ] **Benchmark honestly:** for each real model, measure CompiledModel-GPU vs
  Interpreter (Metal/XNNPACK). Expect:
  - macOS: roughly a wash on compute (same Metal kernels); only I/O-bound, high-frequency
    pipelines (small per-frame models) may benefit from zero-copy/async.
  - Desktop Windows/Linux: this is where GPU is genuinely *new* value vs CPU-only XNNPACK.
- [ ] Add Windows CI baseline FIRST (build + analyze + test + XNNPACK smoke) — independent
  of GPU, useful regardless.
- [ ] Hosted CI can only prove "builds, symbols load, create doesn't crash." **Real GPU
  execution must be validated on actual hardware** (e.g. a self-hosted Windows GPU
  runner / Boot Camp box), not GitHub-hosted runners.

---

## Explicit non-goals / guardrails
- ❌ Do **not** modify or risk the existing Interpreter path. Everything is additive.
- ❌ Do **not** try to expose CompiledModel via `addDelegate()` — it is not a delegate.
- ❌ Do **not** ship a desktop GPU path that silently falls back to CPU without surfacing
  it (repeat of the delegate-fallback bug that invalidated early benchmarks).
- ❌ Do **not** invest in desktop-from-source GPU builds until Spike 0 proves prebuilts
  are unavailable AND the value is confirmed by benchmarking.

## Open questions
- Does `litert:1.4.1` (Android) actually expose CompiledModel C symbols, or only the
  Kotlin/Java API?
- Are there official prebuilt LiteRT Next desktop runtime + GPU accelerator binaries, and
  under what license / distribution?
- What is the exact runtime dependency set for the Windows GPU (WebGPU/D3D12) backend?
- Is there meaningful real-world I/O-bound benefit on the face/hand per-frame pipelines,
  or is preprocessing the actual bottleneck? (Profile before committing.)

## Effort estimate (rough, post-Spike-0)
- If consumable prebuilts exist: Android PoC + Dart API ~1 week; per-desktop-platform
  bundling + CI hardening ~1–2 weeks; benchmarking/polish ~1 week.
- If desktop requires building from source: add multiple weeks and significant risk;
  reconsider scope.
