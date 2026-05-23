## 2.5.5

* Add SPM support for iOS: TensorFlowLiteC, TensorFlowLiteCMetal and TensorFlowLiteCCoreML are now declared as binary targets in Package.swift so the plugin works with Flutter Swift Package Manager integration.
* Fix duplicate XNNPack symbol linker errors when flutter litert flex is used alongside flutter litert by removing XNNPack definitions from TFLiteFlex and hiding overlapping symbols in TensorFlowLiteC via nmedit.
* Fix stale flex dedup marker in podspec that caused nmedit to be skipped on re-downloaded xcframeworks.

## 2.5.4

* Fix WASM compatibility: replace dart:io import in camera_frame.dart with flutter/foundation.dart to allow package to compile under the WASM runtime.

## 2.5.3

* prepareCameraFrameFromImage and prepareCameraFrame now auto-detect isBgra based on platform. macOS uses BGRA, Windows and Linux use RGBA. The isBgra parameter is now nullable and no longer needs to be passed manually.

## 2.5.2

* Update documentation

## 2.5.1

* Add `decodeBitmap(Uint8List bytes)` free function: decodes encoded image bytes (JPEG, PNG, etc.) to a `web.ImageBitmap` via `createImageBitmap`, off the main thread.
* Add `WebGpuFallback` mixin: transparent WebGPU-to-WASM runtime fallback for web detector classes. Provides `withFallback<T>()` which catches GPU errors, swaps all runners to WASM via `swapToWasm()`, and retries once. Apply with `with WebGpuFallback`; implement `activeAccelerator` and `swapToWasm()`.
* Both exported from `package:flutter_litert/flutter_litert.dart` on web.

## 2.5.0

* Add `LiteRtInterpreter`, an alternative web inference path backed by Google's official LiteRT.js runtime (`@litertjs/core`). Selectable at construction time via `LiteRtInterpreter.fromBytes(bytes, accelerator: 'webgpu' | 'wasm')`, with automatic fallback from `webgpu` to `wasm` when ops aren't supported by the GPU delegate.
  * Surface chosen to match the `Interpreter` hot path used by detector packages: `fromBytes`, `getInputTensor` / `getOutputTensors`, `runForMultipleInputs(inputs, outputs)`. `runForMultipleInputs` is async (LiteRT.js `run` returns a `Promise`).
  * Output buffers can be supplied as `Float32List`, `ByteBuffer`, or the legacy nested `List<List<List<double>>>` shape used by tflite-js callers; the float-typed buffer paths take a single bulk copy.
  * Read paths use `JSFloat32Array.toDart` directly, skipping the `dataSync().dartify()` round-trip.
  * Faster output readback in the existing tflite-js `Interpreter._tensorFromJSTensor`: replaces `dataSync().dartify() as List<double>` + `Float32List.fromList(...)` with a single bulk copy via `JSTensorExtensions.dataSyncFloat32`. ~25 ms / call savings on a 705k-element YOLOv8n output.
  * Auto-loader: by default the first `LiteRtInterpreter.fromBytes(...)` call programmatically appends a `<script type="module">` to `<head>` that imports `@litertjs/core` from jsDelivr and calls `loadLiteRt(...)`; consumers don't have to touch their `web/index.html`. Override URLs (for self-hosting / strict CSP) or disable auto-loading via `configureLiteRtLoader(moduleUrl: ..., wasmUrl: ..., autoLoad: ...)`. Existing host-page loaders that assign `window.LiteRt` and dispatch a `litert-ready` event still work.
  * Pure additive: native and unsupported targets are unchanged; the existing tflite-js `Interpreter` remains the default web runtime.

## 2.4.1

* Make `camera_overlay.dart` WASM-compatible on Flutter Web

## 2.4.0

* Add painter primitives `drawLandmarkMarker`, `drawSkeletonConnections`, and `drawBoundingBoxOutline` for reuse by detector example apps and overlay widgets. Pure Dart + `dart:ui`, no new dependencies.

## 2.3.0

* Add camera-overlay helpers used across detector example apps: `rotationForFrame`, `detectionSize`, `coverFitScaleOffset`, `barQuarterTurns`, and `FpsCounter`. All pure Dart + Flutter SDK, no new dependencies. Lets example apps drop ~200 lines of duplicated orientation / sizing / FPS boilerplate.

## 2.2.2

* Add `prepareCameraFrameFromImage`, a duck-typed wrapper around `prepareCameraFrame` that accepts a `CameraImage`-shaped object directly (any object exposing `width`, `height`, `planes` with `bytes`/`bytesPerRow`/`bytesPerPixel`). Lets detector packages expose one-line camera-stream APIs without adding `package:camera` as a dependency here. Pure Dart, no new dependencies.

## 2.2.1

* Add `prepareCameraFrame` helper plus `CameraFrame`, `CameraFrameConversion`, and `CameraFrameRotation` types. Describes a camera frame (YUV420 or packed BGRA/RGBA) in a pure-Dart descriptor that detector packages can hand to their existing detection isolate, moving the `cvtColor` / `rotate` work off the UI thread without adding `opencv_dart` as a dependency here.
* Add `CameraPlane` typedef (structurally identical to `YuvPlane`; use whichever name reads better at the call site).
* Add `TensorFloat32Views` (native only): captures `Float32List` views of an `Interpreter`'s input/output tensors once after `allocateTensors`, letting detector packages reuse the same view wrappers on every inference instead of recreating them per-call. Pure Dart, no new dependencies.

## 2.2.0

* Add `packYuv420` helper for packing NV12 / NV21 / I420 camera frames into a contiguous buffer

## 2.1.0

* Minor performance/accuracy optimizations: 
  * Remove unnecessary rounding in `fillNHWC4D`  
  * Add direct `Float32List` fast paths for common tensor flattening shapes

## 2.0.13

* Fix Android JVM target mismatch: bump Java compile target to 17 to match Kotlin target set by Flutter toolchain

## 2.0.12

* Fix Android Flutter beta builds by aligning Kotlin and Java JVM targets to 11

## 2.0.11

* Fix edge case in output buffer allocation

## 2.0.10

* Update documentation

## 2.0.9

* Enable XNNPACK delegate on Android (ARM NEON SIMD acceleration in auto mode)
* Allow explicit `PerformanceConfig.xnnpack()` on iOS
* Initialize XNNPackDelegateOptions from native defaults (preserves QS8/QU8 quantization flags)

## 2.0.8

* Add Windows XNNPack delegate support (2-5x CPU inference speedup via SIMD)
* Add CI workflow to build Windows TFLite C DLL from source with XNNPack symbols

## 2.0.7

* Fix Android custom ops library alignment for 16 KB page-size devices

## 2.0.6

* Add useIsolateInterpreter parameter to skip nested isolate creation

## 2.0.5

* Fix native crash during repeated inference by removing unsafe output tensor writeback

## 2.0.4

* Fix macOS native crashes by disabling auto IsolateInterpreter for no-delegate interpreters.

## 2.0.3

* Fix WASM compatibility: move `dart:isolate` imports behind conditional exports so web compilation path is WASM-safe

## 2.0.2

* Fix: use-after-free when interpreter reads model weights from freed buffer, transfer buffer ownership from `Model` to `Interpreter`

## 2.0.1

* Add `IsolateWorkerBase` for shared isolate lifecycle management
* Add `RoundRobinPool` generic round-robin pool utility
* Add `TensorType` enum, `LandmarkMixin`, `listUtils` shared helpers
* Add weighted NMS with spatial grid optimization to `nms()`
* Consolidate platform-specific byte conversion into shared implementation
* Consolidate platform-specific tensor logic (native/web/unsupported)
* Consolidate desktop library loading into `DelegateLibraryLoader`
* Remove dead files: `all_unsupported.dart`, `version.dart`, `flutter_litert_method_channel.dart`, `flutter_litert_platform_interface.dart`
* Fix: `Model` buffer leak, delegate options leak, stale tensor cache

## 2.0.0

**Breaking:** `Point.x` and `Point.y` changed from `int` to `double`.

* Upgrade `Point` to double-precision with optional `z` depth, `==`/`hashCode`, `toMap()`/`fromMap()`, `is3D`
* Add shared `BoundingBox` class (4-corner Point-based, supports rotated boxes)
  * `BoundingBox.ltrb()` factory for axis-aligned boxes
  * `left`/`top`/`right`/`bottom` convenience getters
  * `width`, `height`, `center`, `corners` computed properties
  * `toMap()`/`fromMap()` serialization

## 1.4.0
* Fix tensor cache bug, add shared Point class, dedup internals

## 1.3.1
* Add NaN handling to `clamp01()`, returns 0.0 for NaN inputs

## 1.3.0
* Add `IsolateRpcClient` and `setupIsolateHandshake` for reusable isolate request/response communication

## 1.2.0
* Add shared ML utility functions
  * `sigmoid`, `sigmoidClipped`, `clip`, `clamp01`, `argSortDesc`, `median`, `normalizeRadians` (math utilities)
  * `iouXYXY`, `nms` (non-maximum suppression)
  * `computeLetterboxParams`, `computeAspectPadParams`, `LetterboxParams`, `AspectPadParams` (image preprocessing)
  * `bgrBytesToRgbFloat32`, `bgrBytesToSignedFloat32`, `fillNHWC4DFromBgrBytes` (image-to-tensor conversion)
  * `allocTensorShape`, `createOutputBuffers`, `zeroOutputBuffers`, `createNHWCTensor4D`, `fillNHWC4D`, `flattenDynamicTensor` (tensor allocation)
  * `decodeDetectionOutputs`, `transpose2D`, `concat0`, `ensure2D`, `xywhToXyxy` (model output decoding)
  * `postProcessDetections`, `Detection`, `decodeAndSplitOutputs` (end-to-end detection post-processing with NMS)

## 1.1.1
* Fix package layout to follow Pub conventions

## 1.1.0
* Add `PerformanceConfig` and `PerformanceMode`, 
* Add `InterpreterFactory` and `InterpreterPool`
* Add `generateAnchors()` and `SSDAnchorOptions`
* Add `scaleFromLetterbox()` utility for letterbox-to-original coordinate mapping

## 1.0.3
* Add `SignatureRunner` for on-device training workflows (`train`, `infer`, `get_weights`, `set_weights` signatures)
* Add Linux FlexDelegate support via `flutter_litert_flex` (Linux x86_64, built from TF 2.20.0 source). All three desktop platforms (macOS, Windows, Linux) now fully support on-device training with `SELECT_TF_OPS` models and checkpoint save/restore.
* Add `Interpreter.signatureCount`, `signatureKeys`, `getSignatureKey()`, `getSignatureRunner()`
* Add `SignatureRunner.cancel()`, `getInputTensors()`, `getOutputTensors()`, `lastNativeInferenceDurationMicroSeconds`

## 1.0.2
* Add native dylibs to SPM Package.swift 
* Update Dart loading paths for SPM bundle

## 1.0.1
* Improve Custom Ops documentation

## 1.0.0
* Upgrade Linux TFLite native library from 2.9.3 to 2.20.0 (built from source via CMake + Ninja + GCC x86_64)
* First stable release: 
  * All platforms are on updated 2.20.0 library files, official final stable release of TFLite
  * Pre-bundling works on supported native platforms: users no longer need to bundle native libraries manually as was required with `tflite_flutter`
  * Custom ops supported, see [face_detection_tflite v5.0.2](https://pub.dev/packages/face_detection_tflite/versions/5.0.2) `example` directory for a working example (the binary segmentation model selfie_segmenter.tflite uses custom ops)
  * Web support (experimental) functional, see [pose_detection v1.0.1](https://pub.dev/packages/pose_detection/versions/1.0.1) `web_example` directory for a working example

## 0.2.2
* Update dependencies

## 0.2.1
* Update documentation

## 0.2.0
* Web support (experimental)

## 0.1.16
* Register iOS pluginClass

## 0.1.15
* Add missing null check in interpreter teardown path on macOS

## 0.1.14
* Improve IsolateInterpreter shutdown reliability on iOS to prevent rare use-after-free when closing during active inference

## 0.1.13
* Add Swift Package Manager (SPM) support for iOS and macOS

## 0.1.12
* Upgrade Windows TFLite native library from 2.18.0 to 2.20.0 (built from source via CMake + Ninja + MSVC x64)

## 0.1.11
* Fix iOS: download xcframeworks at pod install time so static linking works on first build

## 0.1.10
* Fix macOS: bundle native libraries in pub package so `flutter test` works without manual setup

## 0.1.9
* Fix iOS and macOS podspec compatibility with Ruby 3.4+ (Prism parser)

## 0.1.8
* Upgrade iOS TensorFlow Lite from 2.17.0 (CocoaPods) to 2.20.0 (built from source via Bazel)
* Replace CocoaPods TensorFlowLiteSwift dependency with vendored xcframeworks (TensorFlowLiteC, Metal delegate, CoreML delegate)
* All xcframeworks support device arm64 + simulator arm64/x86_64 (Apple Silicon and Intel Macs)

## 0.1.7
* Improved documentation

## 0.1.6
* Upgrade macOS TFLite native library from 2.17.1 to 2.20.0 (latest stable, universal binary: arm64 + x86_64)
* Update all C API headers to TFLite 2.20.0
* Regenerate FFI bindings (`TfLiteOperatorCreate` now takes 4 params, `TfLiteOperatorCreateWithData` removed, new `kTfLiteOutputShapeNotKnown` status, new builtin ops)
* Rebuild macOS custom ops dylib against 2.20.0

## 0.1.5
* Upgrade macOS TFLite native library from 2.11.0 to 2.17.1 (universal binary: arm64 + x86_64)
* Update all C API headers to TFLite 2.17.1 (including new `TfLiteOperator` API replacing `TfLiteRegistrationExternal`)
* Regenerate FFI bindings with new APIs (SignatureRunner, TfLiteInterpreterCancel, and more)
* Rebuild macOS custom ops dylib as universal binary (arm64 + x86_64)

## 0.1.4
* Bundle `libtensorflowlite_c-win.dll` from flutter_litert Windows plugin instead of downstream packages

## 0.1.3
* Fix Windows: build and bundle custom ops DLL (tflite_custom_ops.dll) for MediaPipe models
* Fix heap corruption crash when switching between segmentation models (custom op name string was freed prematurely)

## 0.1.2
* Fix Linux: build and bundle custom ops library (libtflite_custom_ops.so) so MediaPipe models with custom ops (e.g. selfie segmentation) work on Linux

## 0.1.1
* Update AndroidManifest.xml

## 0.1.0
* Fix IsolateInterpreter thread-safety bug causing intermittent native crashes when hardware delegates are active

## 0.0.1
* Initial release, forked from tflite_flutter_custom v1.2.5
* Rebranded to flutter_litert for LiteRT ecosystem
* All native libraries bundled automatically
* Custom ops support (MediaPipe models)
* Full platform support: Android, iOS, macOS, Windows, Linux
