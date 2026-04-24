import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

File get _modelFile =>
    File('${Directory.current.path}/test/assets/training_model.tflite');

void main() {
  group('TensorFloat32Views.capture', () {
    late Interpreter interpreter;

    setUp(() {
      interpreter = Interpreter.fromFile(_modelFile);
      interpreter.allocateTensors();
    });

    tearDown(() {
      if (!interpreter.isDeleted) interpreter.close();
    });

    test('returns one view per input tensor', () {
      final views = TensorFloat32Views.capture(interpreter);
      expect(views.inputs.length, equals(interpreter.getInputTensors().length));
      for (final Float32List v in views.inputs) {
        expect(v, isA<Float32List>());
      }
    });

    test('returns one view per output tensor', () {
      final views = TensorFloat32Views.capture(interpreter);
      expect(
        views.outputs.length,
        equals(interpreter.getOutputTensors().length),
      );
      for (final Float32List v in views.outputs) {
        expect(v, isA<Float32List>());
      }
    });

    test('input view length matches tensor byte size / 4', () {
      final views = TensorFloat32Views.capture(interpreter);
      for (int i = 0; i < views.inputs.length; i++) {
        final Tensor t = interpreter.getInputTensor(i);
        expect(
          views.inputs[i].lengthInBytes,
          equals(t.numBytes()),
          reason: 'input $i view bytes should equal tensor byte size',
        );
      }
    });

    test('output view length matches tensor byte size / 4', () {
      final views = TensorFloat32Views.capture(interpreter);
      for (int i = 0; i < views.outputs.length; i++) {
        final Tensor t = interpreter.getOutputTensor(i);
        expect(
          views.outputs[i].lengthInBytes,
          equals(t.numBytes()),
          reason: 'output $i view bytes should equal tensor byte size',
        );
      }
    });

    test('returned view lists are unmodifiable', () {
      final views = TensorFloat32Views.capture(interpreter);
      expect(
        () => views.inputs.add(Float32List(1)),
        throwsA(isA<UnsupportedError>()),
      );
      expect(
        () => views.outputs.add(Float32List(1)),
        throwsA(isA<UnsupportedError>()),
      );
    });
  });
}
