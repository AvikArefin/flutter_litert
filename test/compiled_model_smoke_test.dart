@TestOn('mac-os')
library;

import 'dart:io';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Exercises the real package CompiledModel path end-to-end:
/// loads libLiteRt via the package loader, compiles a real .tflite, sets up
/// managed tensor buffers, and frees them. Requires the LiteRT Next runtime:
///   LITERT_LIB_PATH=/path/to/libLiteRt.dylib flutter test test/compiled_model_smoke_test.dart
void main() {
  const model = 'example/assets/simple_model.tflite';

  test('CompiledModel compiles a model on the CPU accelerator', () {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);
    expect(cm, isNotNull);
  });

  test('CompiledModel compiles model bytes on the CPU accelerator', () {
    final bytes = File(model).readAsBytesSync();
    final cm = CompiledModel.fromBuffer(bytes, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);
    expect(cm, isNotNull);
  });

  // NOTE: The GPU (Metal) accelerator needs a real GPU/Metal device context,
  // which the headless `flutter test` runner does not provide — there it fails
  // with kLiteRtStatusErrorCompilation (504) because the Metal accelerator can't
  // register. The GPU path is validated separately via the standalone spike
  // (`dart run` has Metal access) and in a real `flutter run` app. These tests
  // assert success when Metal is available and skip gracefully when it isn't.
  test(
    'CompiledModel compiles on the GPU (Metal) accelerator when available',
    () {
      _gpuTestOrSkip(
        () => CompiledModel.fromFile(model, accelerators: {Accelerator.gpu}),
      );
    },
  );

  test('CompiledModel compiles with GPU and CPU fallback', () {
    final cm = CompiledModel.fromFile(
      model,
      accelerators: {Accelerator.gpu, Accelerator.cpu},
    );
    addTearDown(cm.close);
    expect(cm, isNotNull);
  });

  test('CompiledModel GPU FP32 precision option compiles when available', () {
    _gpuTestOrSkip(
      () => CompiledModel.fromFile(
        model,
        accelerators: {Accelerator.gpu},
        precision: Precision.fp32,
      ),
    );
  });
}

void _gpuTestOrSkip(CompiledModel Function() build) {
  CompiledModel cm;
  try {
    cm = build();
  } on StateError catch (e) {
    if (e.message.contains('LiteRtStatus=504')) {
      markTestSkipped(
        'Metal GPU accelerator unavailable in headless test runner.',
      );
      return;
    }
    rethrow;
  }
  addTearDown(cm.close);
  expect(cm, isNotNull);
}
