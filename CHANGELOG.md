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
* Add NaN handling to `clamp01()` — returns 0.0 for NaN inputs

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
  * Pre-bundling works on all platforms: users no longer need to do bundle libs as was required with `tflite_flutter`
  * Custom ops supported - see [face_detection_tflite v5.0.2](https://pub.dev/packages/face_detection_tflite/versions/5.0.2) `example` directory for a working example (the binary segmentation model selfie_segmenter.tflite uses custom ops)
  * Web support (experimental) functional - see [pose_detection v1.0.1](https://pub.dev/packages/pose_detection/versions/1.0.1) `web_example` directory for a working example

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
