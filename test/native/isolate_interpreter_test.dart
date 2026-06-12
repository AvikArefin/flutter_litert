@TestOn('mac-os || linux || windows')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

File get _modelFile => File(
  '${Directory.current.path}/test/assets/face_detection_short_range.tflite',
);

const _inLen = 128 * 128 * 3;
const _boxesLen = 896 * 16;
const _scoresLen = 896 * 1;

Float32List _seededInput(int seed) {
  final rng = math.Random(seed);
  final input = Float32List(_inLen);
  for (var i = 0; i < input.length; i++) {
    input[i] = rng.nextDouble() * 2 - 1;
  }
  return input;
}

/// Single-shot reference outputs for [input] on a fresh interpreter.
(Float32List, Float32List) _reference(Float32List input) {
  final interpreter = Interpreter.fromFile(_modelFile);
  interpreter.allocateTensors();
  try {
    final boxes = Float32List(_boxesLen);
    final scores = Float32List(_scoresLen);
    interpreter.runForMultipleInputs(
      [input],
      {0: boxes, 1: scores},
    );
    return (boxes, scores);
  } finally {
    interpreter.close();
  }
}

void main() {
  group('IsolateInterpreter run semantics', () {
    late Interpreter interpreter;
    late IsolateInterpreter isolateInterpreter;

    setUp(() async {
      interpreter = Interpreter.fromFile(_modelFile);
      interpreter.allocateTensors();
      isolateInterpreter = await IsolateInterpreter.create(
        address: interpreter.address,
      );
    });

    tearDown(() async {
      await isolateInterpreter.close();
      if (!interpreter.isDeleted) interpreter.close();
    });

    test('overlapping calls both complete with correct outputs', () async {
      final inputA = _seededInput(1);
      final inputB = _seededInput(2);
      final (refBoxesA, refScoresA) = _reference(inputA);
      final (refBoxesB, refScoresB) = _reference(inputB);

      final boxesA = Float32List(_boxesLen);
      final scoresA = Float32List(_scoresLen);
      final boxesB = Float32List(_boxesLen);
      final scoresB = Float32List(_scoresLen);

      // Issue the second call while the first is still in flight.
      final first = isolateInterpreter.runForMultipleInputs(
        [inputA],
        {0: boxesA, 1: scoresA},
      );
      final second = isolateInterpreter.runForMultipleInputs(
        [inputB],
        {0: boxesB, 1: scoresB},
      );
      await Future.wait([first, second]);

      expect(boxesA, equals(refBoxesA), reason: 'first call outputs');
      expect(scoresA, equals(refScoresA), reason: 'first call outputs');
      expect(boxesB, equals(refBoxesB), reason: 'second call outputs');
      expect(scoresB, equals(refScoresB), reason: 'second call outputs');
    });

    test('sequential awaited calls produce correct outputs', () async {
      for (final seed in [3, 4]) {
        final input = _seededInput(seed);
        final (refBoxes, refScores) = _reference(input);
        final boxes = Float32List(_boxesLen);
        final scores = Float32List(_scoresLen);
        await isolateInterpreter.runForMultipleInputs(
          [input],
          {0: boxes, 1: scores},
        );
        expect(boxes, equals(refBoxes));
        expect(scores, equals(refScores));
      }
    });

    test('run after close throws StateError', () async {
      await isolateInterpreter.close();
      expect(
        () => isolateInterpreter.runForMultipleInputs(
          [_seededInput(5)],
          {0: Float32List(_boxesLen), 1: Float32List(_scoresLen)},
        ),
        throwsA(isA<StateError>()),
      );
    });
  });
}
