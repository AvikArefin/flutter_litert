import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late File modelFile;

  setUpAll(() async {
    // Copy the bundled asset to a temp file so the interpreter can read it.
    final data = await rootBundle.load('assets/training_model.tflite');
    final tmpDir = await Directory.systemTemp.createTemp('litert_test_');
    modelFile = File('${tmpDir.path}/training_model.tflite');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  });

  group('FlexDelegate on iOS', () {
    testWidgets('isAvailable returns true when flex is bundled',
        (tester) async {
      expect(FlexDelegate.isAvailable, isTrue);
    });

    testWidgets('constructor creates a valid delegate', (tester) async {
      final delegate = FlexDelegate();
      expect(delegate, isNotNull);
      delegate.delete();
    });

    testWidgets('delete throws on double-delete', (tester) async {
      final delegate = FlexDelegate();
      delegate.delete();
      expect(() => delegate.delete(), throwsA(isA<StateError>()));
    });

    testWidgets('inference with FlexDelegate works', (tester) async {
      final flex = FlexDelegate();
      final opts = InterpreterOptions()..addDelegate(flex);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      final infer = interpreter.getSignatureRunner('infer');
      final output = [
        [0.0],
      ];
      infer.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': output},
      );
      // Untrained model, w=0 b=0 → output should be ~0
      expect(output[0][0], closeTo(0.0, 1e-5));
      infer.close();

      interpreter.close();
      flex.delete();
      opts.delete();
    });

    testWidgets('training with FlexDelegate works and loss decreases',
        (tester) async {
      final flex = FlexDelegate();
      final opts = InterpreterOptions()..addDelegate(flex);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      final train = interpreter.getSignatureRunner('train');
      final loss = Float32List(1);

      // First step
      train.run(
        {
          'x': [
            [1.0],
          ],
          'y': [
            [2.0],
          ],
        },
        {'loss': loss},
      );
      final firstLoss = loss[0];
      expect(firstLoss, closeTo(4.0, 0.01));

      // Train 100 more steps
      for (var i = 0; i < 100; i++) {
        train.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss},
        );
      }
      expect(loss[0], lessThan(firstLoss));
      train.close();

      // Verify prediction improved
      final infer = interpreter.getSignatureRunner('infer');
      final output = [
        [0.0],
      ];
      infer.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': output},
      );
      expect(output[0][0], greaterThan(1.0));
      infer.close();

      interpreter.close();
      flex.delete();
      opts.delete();
    });

    testWidgets('weight persistence works with FlexDelegate', (tester) async {
      final flex = FlexDelegate();
      final opts = InterpreterOptions()..addDelegate(flex);
      var interpreter = Interpreter.fromFile(modelFile, options: opts);

      // Train
      final train = interpreter.getSignatureRunner('train');
      final loss = Float32List(1);
      for (var i = 0; i < 50; i++) {
        train.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss},
        );
      }
      train.close();

      // Get trained weights
      final getW = interpreter.getSignatureRunner('get_weights');
      final w = [
        [0.0],
      ];
      final b = [0.0];
      getW.run({}, {'w': w, 'b': b});
      getW.close();
      expect(w[0][0], isNot(closeTo(0.0, 1e-5)));

      // Record trained prediction
      final inferA = interpreter.getSignatureRunner('infer');
      final predA = [
        [0.0],
      ];
      inferA.run(
        {
          'x': [
            [1.0],
          ],
        },
        {'output': predA},
      );
      inferA.close();

      // Fresh interpreter
      interpreter.close();
      flex.delete();
      opts.delete();

      final flex2 = FlexDelegate();
      final opts2 = InterpreterOptions()..addDelegate(flex2);
      interpreter = Interpreter.fromFile(modelFile, options: opts2);

      // Fresh should predict ~0
      final inferFresh = interpreter.getSignatureRunner('infer');
      final predFresh = [
        [0.0],
      ];
      inferFresh.run(
        {
          'x': [
            [1.0],
          ],
        },
        {'output': predFresh},
      );
      inferFresh.close();
      expect(predFresh[0][0], closeTo(0.0, 1e-5));

      // Restore weights
      final setW = interpreter.getSignatureRunner('set_weights');
      setW.run({'w': w, 'b': b}, {});
      setW.close();

      // Should match trained prediction
      final inferB = interpreter.getSignatureRunner('infer');
      final predB = [
        [0.0],
      ];
      inferB.run(
        {
          'x': [
            [1.0],
          ],
        },
        {'output': predB},
      );
      inferB.close();

      expect(predB[0][0], closeTo(predA[0][0], 1e-5));
      expect(predB[0][0], greaterThan(0.5));

      interpreter.close();
      flex2.delete();
      opts2.delete();
    });
  });
}
