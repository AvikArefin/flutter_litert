import 'dart:math' as math;

import 'math_utils.dart';

/// Computes the Intersection over Union (IoU) between two XYXY-format bounding boxes.
///
/// Both [a] and [b] must be [x1, y1, x2, y2] format.
double iouXYXY(List<double> a, List<double> b) {
  final double xx1 = math.max(a[0], b[0]);
  final double yy1 = math.max(a[1], b[1]);
  final double xx2 = math.min(a[2], b[2]);
  final double yy2 = math.min(a[3], b[3]);
  final double interW = math.max(0.0, xx2 - xx1);
  final double interH = math.max(0.0, yy2 - yy1);
  final double inter = interW * interH;
  final double areaA = math.max(0.0, a[2] - a[0]) * math.max(0.0, a[3] - a[1]);
  final double areaB = math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-7);
}

/// Non-Maximum Suppression over XYXY-format bounding boxes.
///
/// Returns the indices of kept detections, sorted by descending score and
/// capped at [maxDet] results. Boxes whose IoU with an already-kept box
/// exceeds [iouThres] are suppressed.
///
/// Parameters:
/// - [boxes]: List of bounding boxes in [x1, y1, x2, y2] format.
/// - [scores]: Confidence score for each box.
/// - [iouThres]: IoU threshold above which a box is suppressed (default 0.45).
/// - [maxDet]: Maximum number of detections to return (default 100).
List<int> nms(
  List<List<double>> boxes,
  List<double> scores, {
  double iouThres = 0.45,
  int maxDet = 100,
}) {
  if (boxes.isEmpty) return <int>[];

  final List<int> order = argSortDesc(scores);
  final List<int> keep = <int>[];

  double area(List<double> b) =>
      math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);

  final List<double> areas = boxes.map(area).toList();
  final List<bool> suppressed = List<bool>.filled(order.length, false);

  for (int m = 0; m < order.length; m++) {
    if (suppressed[m]) continue;
    final int i = order[m];
    keep.add(i);
    if (keep.length >= maxDet) break;
    for (int n = m + 1; n < order.length; n++) {
      if (suppressed[n]) continue;
      final int j = order[n];
      final double xx1 = math.max(boxes[i][0], boxes[j][0]);
      final double yy1 = math.max(boxes[i][1], boxes[j][1]);
      final double xx2 = math.min(boxes[i][2], boxes[j][2]);
      final double yy2 = math.min(boxes[i][3], boxes[j][3]);
      final double inter = math.max(0.0, xx2 - xx1) * math.max(0.0, yy2 - yy1);
      final double u = areas[i] + areas[j] - inter + 1e-7;
      if (inter / u > iouThres) suppressed[n] = true;
    }
  }

  return keep;
}
