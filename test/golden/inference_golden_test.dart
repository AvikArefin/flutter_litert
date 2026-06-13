@TestOn('mac-os || linux || windows')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Bit-exact inference golden for the seeded benchmark input.
///
/// Guards Dart-side I/O plumbing changes (tensor staging, output copies,
/// views): any byte-level deviation in model outputs fails this test with
/// zero statistical ambiguity, unlike tolerance-based comparisons.
///
/// Regenerate (only when the model, runtime binary, or input seed changes):
///   GOLDEN_REGEN=1 flutter test test/golden/inference_golden_test.dart
const _model = 'test/assets/face_detection_short_range.tflite';
const _golden = 'test/assets/goldens/face_detection_outputs.bin';

const _inLen = 128 * 128 * 3;

Float32List _makeInput() {
  // Same generator as test/benchmark/engine_overhead_benchmark_test.dart.
  final rng = math.Random(42);
  final input = Float32List(_inLen);
  for (var i = 0; i < input.length; i++) {
    input[i] = rng.nextDouble() * 2 - 1;
  }
  return input;
}

Uint8List _runAndConcatOutputs() {
  final interpreter = Interpreter.fromFile(File(_model));
  interpreter.allocateTensors();
  try {
    interpreter.getInputTensor(0).setTo(_makeInput());
    interpreter.invoke();
    final outputs = interpreter.getOutputTensors();
    final builder = BytesBuilder(copy: true);
    for (final tensor in outputs) {
      builder.add(tensor.data);
    }
    return builder.toBytes();
  } finally {
    interpreter.close();
  }
}

void main() {
  test('inference outputs are bit-identical to golden', () {
    final actual = _runAndConcatOutputs();
    final goldenFile = File(_golden);

    if (Platform.environment['GOLDEN_REGEN'] == '1') {
      goldenFile.parent.createSync(recursive: true);
      goldenFile.writeAsBytesSync(actual, flush: true);
      // ignore: avoid_print
      print('Golden regenerated: $_golden (${actual.length} bytes)');
      return;
    }

    expect(
      goldenFile.existsSync(),
      isTrue,
      reason:
          'Golden file missing. Generate it on a known-good revision with: '
          'GOLDEN_REGEN=1 flutter test test/golden/inference_golden_test.dart',
    );
    final expected = goldenFile.readAsBytesSync();
    expect(
      actual.length,
      expected.length,
      reason: 'Output byte length changed.',
    );
    expect(
      actual,
      equals(expected),
      reason: 'Inference outputs deviate from golden at the byte level.',
    );
  });
}
