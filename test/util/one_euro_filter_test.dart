import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('OneEuroFilter', () {
    test('returns the first sample unchanged', () {
      final f = OneEuroFilter();
      expect(f.filter(3.7, 0.0), 3.7);
    });

    test('a constant signal passes through unchanged', () {
      final f = OneEuroFilter();
      f.filter(5.0, 0.0);
      for (var i = 1; i <= 10; i++) {
        expect(f.filter(5.0, i * 0.1), closeTo(5.0, 1e-9));
      }
    });

    test('a step is smoothed: first output lies between old and new value', () {
      final f = OneEuroFilter();
      f.filter(0.0, 0.0);
      f.filter(0.0, 0.1);
      final out = f.filter(10.0, 0.2);
      expect(out, greaterThan(0.0));
      expect(out, lessThan(10.0));
    });

    test('reduces jitter amplitude on an oscillating signal', () {
      final f = OneEuroFilter();
      double maxOut = double.negativeInfinity;
      double minOut = double.infinity;
      for (var i = 0; i < 40; i++) {
        final x = i.isEven ? 0.0 : 1.0; // input amplitude 1.0
        final out = f.filter(x, i * (1 / 30.0));
        if (i >= 10) {
          maxOut = out > maxOut ? out : maxOut;
          minOut = out < minOut ? out : minOut;
        }
      }
      expect(
        maxOut - minOut,
        lessThan(1.0),
        reason: 'smoothed amplitude must be below the input amplitude',
      );
    });

    test('reset() makes the next sample pass through unchanged', () {
      final f = OneEuroFilter();
      f.filter(1.0, 0.0);
      f.filter(2.0, 0.1);
      f.reset();
      expect(f.filter(9.0, 0.2), 9.0);
    });
  });
}
