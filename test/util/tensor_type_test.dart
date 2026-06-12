import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('TensorType.fromValue', () {
    test('round-trips every declared type', () {
      for (final t in TensorType.values) {
        expect(TensorType.fromValue(t.value), same(t));
      }
    });

    test('enum declaration order matches TfLiteType values', () {
      // fromValue relies on value == declaration index; this pins the
      // invariant against future enum reordering or gaps.
      for (var i = 0; i < TensorType.values.length; i++) {
        expect(TensorType.values[i].value, i);
      }
    });

    test('unknown values map to noType', () {
      expect(TensorType.fromValue(-1), TensorType.noType);
      expect(TensorType.fromValue(TensorType.values.length), TensorType.noType);
      expect(TensorType.fromValue(99), TensorType.noType);
    });
  });
}
