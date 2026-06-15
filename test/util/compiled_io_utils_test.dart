import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('compiledFloatCount', () {
    test('divides byte size by 4', () {
      expect(compiledFloatCount(4), 1);
      expect(compiledFloatCount(224 * 224 * 3 * 4), 224 * 224 * 3);
      expect(compiledFloatCount(252), 63); // 21 landmarks * 3
    });

    test('throws on non-positive or non-multiple-of-4 sizes', () {
      expect(() => compiledFloatCount(0), throwsUnsupportedError);
      expect(() => compiledFloatCount(-4), throwsUnsupportedError);
      expect(() => compiledFloatCount(5), throwsUnsupportedError);
    });
  });

  group('squareSideFromFloats', () {
    test('derives the side of a square [1,S,S,3] tensor', () {
      expect(squareSideFromFloats(224 * 224 * 3), 224);
      expect(squareSideFromFloats(192 * 192 * 3), 192);
      expect(squareSideFromFloats(256 * 256 * 3), 256);
    });

    test('honours a non-default channel count', () {
      expect(squareSideFromFloats(64 * 64, channels: 1), 64);
    });

    test('throws when not divisible by channels', () {
      expect(
        () => squareSideFromFloats(224 * 224 * 3 + 1),
        throwsUnsupportedError,
      );
      expect(() => squareSideFromFloats(0), throwsUnsupportedError);
    });

    test('throws when the per-channel area is not a perfect square', () {
      // 3*5*3 floats => area 15, not a perfect square.
      expect(() => squareSideFromFloats(15 * 3), throwsUnsupportedError);
    });
  });

  group('indexWhereFloatCount', () {
    test('returns the first matching index', () {
      expect(indexWhereFloatCount([63, 1, 1, 63], (f) => f == 1), 1);
      expect(indexWhereFloatCount([2016 * 18, 2016], (f) => f == 2016), 1);
    });

    test('returns -1 when nothing matches', () {
      expect(indexWhereFloatCount([63, 1], (f) => f == 999), -1);
    });
  });
}
