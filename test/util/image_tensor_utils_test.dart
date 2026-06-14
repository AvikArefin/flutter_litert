import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('rgbaToRgbFloat32', () {
    test('drops alpha and normalizes channels to 0..1', () {
      final rgba = Uint8List.fromList([0, 127, 255, 10, 255, 0, 127, 200]);
      final out = Float32List(6);
      rgbaToRgbFloat32(rgba, out);
      expect(out[0], closeTo(0.0, 1e-9));
      expect(out[1], closeTo(127 / 255.0, 1e-9));
      expect(out[2], closeTo(1.0, 1e-9));
      // Second pixel: alpha (200) discarded.
      expect(out[3], closeTo(1.0, 1e-9));
      expect(out[4], closeTo(0.0, 1e-9));
      expect(out[5], closeTo(127 / 255.0, 1e-9));
    });
  });

  group('rgbaToSignedRgbFloat32', () {
    test('drops alpha and normalizes channels to -1..1', () {
      final rgba = Uint8List.fromList([0, 127, 255, 10]);
      final out = Float32List(3);
      rgbaToSignedRgbFloat32(rgba, out);
      expect(out[0], closeTo(-1.0, 1e-9));
      expect(out[1], closeTo(127 / 127.5 - 1.0, 1e-9));
      expect(out[2], closeTo(255 / 127.5 - 1.0, 1e-9));
    });
  });
}
