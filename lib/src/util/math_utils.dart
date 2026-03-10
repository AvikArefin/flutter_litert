import 'dart:math' as math;

/// Sigmoid activation function.
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

/// Sigmoid with input clipping to prevent overflow.
double sigmoidClipped(double x, {double limit = 80.0}) =>
    sigmoid(clip(x, -limit, limit));

/// Clamps [v] to the range [0.0, 1.0]. Returns 0.0 for NaN inputs.
double clamp01(double v) =>
    v.isNaN ? 0.0 : (v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));

/// Clamps [v] to the range [lo, hi].
double clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

/// Returns indices that sort [a] in descending order.
List<int> argSortDesc(List<double> a) {
  final List<int> idx = List<int>.generate(a.length, (i) => i);
  idx.sort((i, j) => a[j].compareTo(a[i]));
  return idx;
}

/// Returns the median of a non-empty list.
double median(List<double> a) {
  if (a.isEmpty) return double.nan;

  final List<double> b = List<double>.from(a)..sort();
  final int n = b.length;
  if (n.isOdd) return b[n ~/ 2];

  return 0.5 * (b[n ~/ 2 - 1] + b[n ~/ 2]);
}

/// Normalizes an angle in radians to the range [-pi, pi].
double normalizeRadians(double angle) {
  return angle - 2 * math.pi * ((angle + math.pi) / (2 * math.pi)).floor();
}
