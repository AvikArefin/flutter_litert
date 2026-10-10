@TestOn('browser')
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// A float32 graph with two inputs, computing `a + 2 * b` on [1, 4] tensors,
/// embedded because browser tests cannot read the file system.
///
/// Models with more than one input take a different branch when the inputs
/// are handed to LiteRT.js `run()`: they are passed as a JS array. That array
/// used to be built from a `cast()` view, which dart2js does not turn into a
/// `JSArray`, so every multi-input model failed at run time while the
/// single-input models the other tests use kept working.
const _add2ModelBase64 =
    'HAAAAFRGTDMUACAABAAIAAwAEAAUAAAAGAAcABQAAAADAAAARAAAADgAAABwAQAADAAAANwAAAA8'
    'AAAACAAAAAQEAAAABAAA/AMAAEADAAD0AwAA8AMAACgBAADoAAAAAQAAAFwBAAACAAAAmAIAAPgB'
    'AAABAAAAEAAAAAAACgAQAAwACAAEAAoAAAAMAAAAHAAAACAAAAAPAAAAc2VydmluZ19kZWZhdWx0'
    'AAEAAAAQAAAAAgAAAEQAAAAoAAAA5P///wQAAAAEAAAACAAAAG91dHB1dF8wAAAAAAgADAAIAAQA'
    'CAAAAAEAAAAEAAAABgAAAGFyZ3NfMQAAdv3//wQAAAAGAAAAYXJnc18wAAACAAAASAAAAAQAAADI'
    '////CAAAAAcAAAAXAAAAa2VlcF9zdGFibGVobG9fY29uc3RhbnQAvv3//wQAAAAEAAAAdHJ1ZQgA'
    'DAAEAAgACAAAAAgAAAAGAAAAEwAAAG1pbl9ydW50aW1lX3ZlcnNpb24A+v3//wQAAAAQAAAAMS41'
    'LjAAAAAAAAAAAAAAAA8AAABNTElSIENvbnZlcnRlZC4AAAAOABgABAAIAAwAEAAUAA4AAABAAAAA'
    'MAAAACQAAAAUAAAABAAAAAQAAABtYWluAAAAAAIAAADkAAAAQAAAAAEAAAAEAAAAAgAAAAAAAAAB'
    'AAAABQAAAPQBAACQAQAAQAEAAAABAABYAAAAAAAOABgACAAMABAABwAUAA4AAAAAAAALAQAAABgA'
    'AAAMAAAABAAAAAT+//8BAAAABAAAAAIAAAAAAAAAAwAAAAAACgAIAAAAAAAEAAoAAAABAAAAiv7/'
    '/wAAAAE0AAAABQAAAAgAAAA0AAAAHwAAAHNlcnZpbmdfZGVmYXVsdF9vdXRwdXRfMF9vdXRwdXQA'
    'AgAAAAEAAAAEAAAA3P7//wAADgAUAAAACAAMAAcAEAAOAAAAAAAAFRgAAAAMAAAABAAAAKD+//8B'
    'AAAAAwAAAAIAAAABAAAAAgAAAAwAEAALAAAADAAEAAwAAAASAAAAAAAAEgEAAAAu////AAAAASQA'
    'AAAEAAAACAAAACQAAAAPAAAAX19tYWluX18uQWRkMjsxAAIAAAABAAAABAAAAAz///9q////AAAA'
    'ASQAAAADAAAACAAAABwAAAAOAAAAX19tYWluX18uQWRkMjsAAAAAAACk////AAAGAAgABAAGAAAA'
    'BAAAAAQAAAAAAABAtv///wAAAAEsAAAAAgAAAAgAAAAwAAAAFgAAAHNlcnZpbmdfZGVmYXVsdF9h'
    'cmdzXzEAAAIAAAABAAAABAAAAAQABgAEAAAAAAAWABgACAAAAAwAEAAUAAAAAAAAAAcAFgAAAAAA'
    'AAEsAAAAAQAAAAgAAAAsAAAAFgAAAHNlcnZpbmdfZGVmYXVsdF9hcmdzXzAAAAIAAAABAAAABAAA'
    'APz///8EAAQABAAAAA==';

void main() {
  final Uint8List modelBytes = base64Decode(_add2ModelBase64);
  final a = Float32List.fromList([1, 2, 3, 4]);
  final b = Float32List.fromList([10, 20, 30, 40]);
  const expected = [21.0, 42.0, 63.0, 84.0];

  // These tests exercise the real LiteRT.js runtime, auto-loaded from
  // cdn.jsdelivr.net on first use; they need network access.
  group('web multi-input models with LiteRT.js (network)', () {
    test('LiteRtInterpreter.runForMultipleInputs takes two inputs', () async {
      final lrt = await LiteRtInterpreter.fromBytes(modelBytes);
      expect(lrt.getInputTensor(0).shape, [1, 4]);
      expect(lrt.getInputTensor(1).shape, [1, 4]);

      final output = Float32List(4);
      await lrt.runForMultipleInputs([a, b], {0: output});
      for (int i = 0; i < 4; i++) {
        expect(output[i], closeTo(expected[i], 1e-6));
      }
      lrt.close();
    });

    test('CompiledModel.runAsync takes two inputs', () async {
      final cm = await CompiledModel.fromBufferAsync(modelBytes);
      expect(cm.inputCount, 2);
      expect(cm.outputCount, 1);

      final outputs = await cm.runAsync([a, b]);
      expect(outputs, hasLength(1));
      expect(outputs[0], hasLength(4));
      for (int i = 0; i < 4; i++) {
        expect(outputs[0][i], closeTo(expected[i], 1e-6));
      }
      cm.close();
    });
  });
}
