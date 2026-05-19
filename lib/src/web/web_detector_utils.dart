import 'dart:js_interop';
import 'dart:typed_data';

import 'package:web/web.dart' as web;

/// Decodes encoded image bytes (JPEG, PNG, etc.) to an [web.ImageBitmap].
///
/// Uses `createImageBitmap`, which decodes off the main thread and avoids the
/// HTMLImageElement load-event roundtrip. Returns null if decoding fails.
Future<web.ImageBitmap?> decodeBitmap(Uint8List bytes) async {
  final web.Blob blob = web.Blob([bytes.toJS].toJS);
  try {
    return await web.window.createImageBitmap(blob).toDart;
  } catch (_) {
    return null;
  }
}

/// Mixin that adds transparent WebGPU-to-WASM runtime fallback to a web
/// detector class.
///
/// Apply with `with WebGpuFallback`. The applying class must provide:
/// - `String? get activeAccelerator`: the current backend
/// - `Future<void> swapToWasm()`: dispose and re-init all runners on WASM
///
/// Then wrap each public inference call with [withFallback]:
/// ```dart
/// Future<List<Result>> detect(Uint8List bytes) async {
///   ...
///   return withFallback(() => _detectInner(bytes));
/// }
/// ```
mixin WebGpuFallback {
  bool _fellBackToWasm = false;

  /// True once the detector has irreversibly fallen back from WebGPU to WASM
  /// after a runtime GPU error.
  bool get fellBackToWasm => _fellBackToWasm;

  /// The accelerator currently in use. Provided by the applying class.
  String? get activeAccelerator;

  /// Disposes and re-initializes all model runners on WASM. Called once on
  /// the first runtime GPU error. Provided by the applying class.
  Future<void> swapToWasm();

  /// Runs [fn]. If a GPU error occurs on the WebGPU path, transparently swaps
  /// all runners to WASM via [swapToWasm] and retries [fn] once.
  Future<T> withFallback<T>(Future<T> Function() fn) async {
    try {
      return await fn();
    } catch (e) {
      if (activeAccelerator == 'webgpu' && !_fellBackToWasm) {
        _fellBackToWasm = true;
        await swapToWasm();
        return fn();
      }
      rethrow;
    }
  }
}
