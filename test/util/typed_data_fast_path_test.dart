import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert/src/util/list_utils.dart' as list_utils;
import 'package:flutter_test/flutter_test.dart';

/// Covers the typed-data fast paths in ByteConversionUtils and the
/// flat-typed-data input shape handling in list_utils.
void main() {
  group('convertObjectToBytes typed-data fast paths', () {
    test('Float32List produces identical bytes to the element-wise path', () {
      final values = [1.1, -2.5, 0.0, 3.75];
      final viaList = ByteConversionUtils.convertObjectToBytes(
        values,
        TensorType.float32,
      );
      final viaTyped = ByteConversionUtils.convertObjectToBytes(
        Float32List.fromList(values),
        TensorType.float32,
      );
      expect(viaTyped, viaList);
    });

    test('Float32List view with non-zero offset converts correctly', () {
      final backing = Float32List.fromList([9.0, 1.5, -2.0, 4.0]);
      final view = Float32List.view(backing.buffer, 4, 3); // skip first
      final bytes = ByteConversionUtils.convertObjectToBytes(
        view,
        TensorType.float32,
      );
      final roundTrip =
          ByteConversionUtils.convertBytesToObject(bytes, TensorType.float32, [
                3,
              ])
              as List;
      expect(roundTrip, [1.5, -2.0, 4.0]);
    });

    test('Float64List narrows to float32 bytes', () {
      final bytes = ByteConversionUtils.convertObjectToBytes(
        Float64List.fromList([1.5, -2.0]),
        TensorType.float32,
      );
      expect(bytes.length, 8);
      final roundTrip =
          ByteConversionUtils.convertBytesToObject(bytes, TensorType.float32, [
                2,
              ])
              as List;
      expect(roundTrip, [1.5, -2.0]);
    });

    test('Int32List, Int64List, Int16List, Int8List match element-wise', () {
      final values = [1, -2, 127, 0];
      for (final (typed, type) in [
        (Int32List.fromList(values), TensorType.int32),
        (Int64List.fromList(values), TensorType.int64),
        (Int16List.fromList(values), TensorType.int16),
        (Int8List.fromList(values), TensorType.int8),
      ]) {
        final viaList = ByteConversionUtils.convertObjectToBytes(values, type);
        final viaTyped = ByteConversionUtils.convertObjectToBytes(typed, type);
        expect(viaTyped, viaList, reason: '$type');
      }
    });

    test('typed data mismatched with tensor type still throws', () {
      expect(
        () => ByteConversionUtils.convertObjectToBytes(
          Float32List.fromList([1.0]),
          TensorType.int32,
        ),
        throwsA(isA<ByteConversionError>()),
      );
    });

    test('nested list single-buffer fill matches legacy bytes', () {
      final nested = [
        [1.5, -2.0],
        [3.25, 4.0],
      ];
      final bytes = ByteConversionUtils.convertObjectToBytes(
        nested,
        TensorType.float32,
      );
      expect(
        bytes,
        Float32List.fromList([1.5, -2.0, 3.25, 4.0]).buffer.asUint8List(),
      );
      expect(
        () => ByteConversionUtils.convertObjectToBytes([
          [1.5, 'oops'],
        ], TensorType.float32),
        throwsA(isA<ByteConversionError>()),
      );
    });
  });

  group('convertBytesToObject offset views', () {
    test('reads through a non-zero-offset Uint8List view', () {
      final floats = Float32List.fromList([7.0, 1.5, -2.0]);
      final all = floats.buffer.asUint8List();
      // Misaligned-on-purpose view start is not possible for float32 data,
      // but an offset view (skipping the first float) must decode correctly.
      final view = Uint8List.view(all.buffer, 4, 8);
      final out =
          ByteConversionUtils.convertBytesToObject(view, TensorType.float32, [
                2,
              ])
              as List;
      expect(out, [1.5, -2.0]);
    });
  });

  group('getInputShapeIfDifferent with flat typed data', () {
    test('matching element count is treated as already shaped', () {
      expect(
        list_utils.getInputShapeIfDifferent(Float32List(12), [1, 2, 2, 3]),
        isNull,
      );
    });

    test('mismatched element count still requests a 1-D resize', () {
      expect(
        list_utils.getInputShapeIfDifferent(Float32List(5), [1, 2, 2, 3]),
        [5],
      );
    });

    test('raw bytes are never treated as a shape', () {
      expect(
        list_utils.getInputShapeIfDifferent(Uint8List(48), [1, 2, 2, 3]),
        isNull,
      );
    });

    test('nested lists keep their existing shape semantics', () {
      expect(
        list_utils.getInputShapeIfDifferent(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ],
          [2, 2],
        ),
        isNull,
      );
      expect(
        list_utils.getInputShapeIfDifferent(
          [
            [1.0, 2.0, 3.0],
          ],
          [2, 2],
        ),
        [1, 3],
      );
    });
  });
}
