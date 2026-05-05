import 'math_utils.dart';
import 'nms_utils.dart';
import 'detection_utils.dart';

/// A single object detection result from a detection model.
///
/// Contains the detected class ID, confidence score, and bounding box coordinates.
class Detection {
  /// Detected class ID.
  final int cls;

  /// Confidence score for the detection (0.0 to 1.0).
  final double score;

  /// Bounding box in XYXY format `[x1, y1, x2, y2]` in pixel coordinates.
  final List<double> bboxXYXY;

  Detection({required this.cls, required this.score, required this.bboxXYXY});
}

/// Decodes raw detection model outputs and splits each row into xywh, rest, and C.
///
/// Returns a list of maps with keys:
/// - `xywh`: first 4 values (bounding box center-xy + width/height)
/// - `rest`: remaining values (objectness + class logits)
/// - `C`: total number of channels in the row
List<Map<String, dynamic>> decodeAndSplitOutputs(List<dynamic> outputs) {
  final List<List<double>> out = decodeDetectionOutputs(outputs);
  final int channels = out[0].length;
  return out
      .map(
        (row) => {
          'xywh': row.sublist(0, 4),
          'rest': row.sublist(4),
          'C': channels,
        },
      )
      .toList();
}

/// Post-processes detection model outputs into [Detection] results.
///
/// Decodes model outputs, applies confidence filtering, optional class filtering,
/// top-k pre-NMS selection, NMS, and coordinate transformation from letterbox
/// space to original image coordinates.
///
/// Parameters:
/// - [outputs]: Raw model output tensors.
/// - [inputWidth]: Width of the model input tensor (used for coordinate de-normalization).
/// - [inputHeight]: Height of the model input tensor (used for coordinate de-normalization).
/// - [r]: Letterbox scale ratio.
/// - [dw]: Horizontal letterbox padding in pixels.
/// - [dh]: Vertical letterbox padding in pixels.
/// - [imageWidth]: Original image width for coordinate clamping.
/// - [imageHeight]: Original image height for coordinate clamping.
/// - [confThres]: Minimum confidence score to keep a detection.
/// - [iouThres]: IoU threshold for NMS.
/// - [topkPreNms]: Number of top candidates to keep before NMS (0 = auto-scale).
/// - [maxDet]: Maximum number of detections to return after NMS.
/// - [filterClassId]: If non-null, only detections with this class ID are kept.
List<Detection> postProcessDetections({
  required List<dynamic> outputs,
  required int inputWidth,
  required int inputHeight,
  required double r,
  required int dw,
  required int dh,
  required int imageWidth,
  required int imageHeight,
  required double confThres,
  required double iouThres,
  required int topkPreNms,
  required int maxDet,
  int? filterClassId,
}) {
  final List<Map<String, dynamic>> decoded = decodeAndSplitOutputs(outputs);
  final List<int> clsIds = <int>[];
  final List<double> scores = <double>[];
  final List<List<double>> xywhs = <List<double>>[];

  for (final Map<String, dynamic> row in decoded) {
    final int C = row['C'] as int;
    final List<double> xywh = (row['xywh'] as List)
        .map((v) => (v as num).toDouble())
        .toList();
    final List<double> rest = (row['rest'] as List)
        .map((v) => (v as num).toDouble())
        .toList();

    if (C == 84) {
      int argMax = 0;
      double best = -1e9;
      for (int i = 0; i < rest.length; i++) {
        final double s = sigmoid(rest[i]);
        if (s > best) {
          best = s;
          argMax = i;
        }
      }
      scores.add(best);
      clsIds.add(argMax);
      xywhs.add(xywh);
    } else {
      final double obj = sigmoid(rest[0]);
      final List<double> clsLogits = rest.sublist(1, 81);
      int argMax = 0;
      double best = -1e9;
      for (int i = 0; i < clsLogits.length; i++) {
        final double s = sigmoid(clsLogits[i]);
        if (s > best) {
          best = s;
          argMax = i;
        }
      }
      scores.add(obj * best);
      clsIds.add(argMax);
      xywhs.add(xywh);
    }
  }

  final List<int> keep0 = <int>[];
  for (int i = 0; i < scores.length; i++) {
    if (scores[i] >= confThres) keep0.add(i);
  }
  if (keep0.isEmpty) return <Detection>[];

  final List<List<double>> keptXywh = [for (final int i in keep0) xywhs[i]];
  final List<int> keptCls = [for (final int i in keep0) clsIds[i]];
  final List<double> keptScore = [for (final int i in keep0) scores[i]];

  if (keptXywh.isNotEmpty && median([for (final v in keptXywh) v[2]]) <= 2.0) {
    for (final List<double> v in keptXywh) {
      v[0] *= inputWidth.toDouble();
      v[1] *= inputHeight.toDouble();
      v[2] *= inputWidth.toDouble();
      v[3] *= inputHeight.toDouble();
    }
  }

  final List<List<double>> boxesLtr = [
    for (final List<double> v in keptXywh) xywhToXyxy(v),
  ];
  final List<List<double>> boxes = <List<double>>[];
  for (final List<double> b in boxesLtr) {
    boxes.add(scaleFromLetterbox(b, r, dw, dh));
  }
  final double iw = imageWidth.toDouble();
  final double ih = imageHeight.toDouble();
  for (final List<double> b in boxes) {
    b[0] = b[0].clamp(0.0, iw);
    b[2] = b[2].clamp(0.0, iw);
    b[1] = b[1].clamp(0.0, ih);
    b[3] = b[3].clamp(0.0, ih);
  }

  final int effectiveTopk;
  if (topkPreNms > 0) {
    effectiveTopk = topkPreNms;
  } else {
    const int basePixels = 640 * 640;
    const int baseCandidates = 100;
    final int imagePixels = imageWidth * imageHeight;
    final double scale = imagePixels / basePixels;
    effectiveTopk = (baseCandidates * scale).round().clamp(20, 200);
  }

  if (effectiveTopk > 0 && keptScore.length > effectiveTopk) {
    final List<int> ord = argSortDesc(keptScore).take(effectiveTopk).toList();
    final List<List<double>> sortedBoxes = <List<double>>[];
    final List<double> sortedScores = <double>[];
    final List<int> sortedCls = <int>[];
    for (final int i in ord) {
      sortedBoxes.add(boxes[i]);
      sortedScores.add(keptScore[i]);
      sortedCls.add(keptCls[i]);
    }
    boxes
      ..clear()
      ..addAll(sortedBoxes);
    keptScore
      ..clear()
      ..addAll(sortedScores);
    keptCls
      ..clear()
      ..addAll(sortedCls);
  }

  if (filterClassId != null) {
    final List<List<double>> fBoxes = <List<double>>[];
    final List<double> fScores = <double>[];
    final List<int> fCls = <int>[];
    for (int i = 0; i < keptCls.length; i++) {
      if (keptCls[i] == filterClassId) {
        fBoxes.add(boxes[i]);
        fScores.add(keptScore[i]);
        fCls.add(keptCls[i]);
      }
    }
    boxes
      ..clear()
      ..addAll(fBoxes);
    keptScore
      ..clear()
      ..addAll(fScores);
    keptCls
      ..clear()
      ..addAll(fCls);
  }

  final List<int> keep = nms(
    boxes,
    keptScore,
    iouThres: iouThres,
    maxDet: maxDet,
  );
  final List<Detection> out = <Detection>[];
  for (final int i in keep) {
    out.add(
      Detection(cls: keptCls[i], score: keptScore[i], bboxXYXY: boxes[i]),
    );
  }
  return out;
}

/// Transposes a 2D list (swaps rows and columns).
List<List<double>> transpose2D(List<List<double>> a) {
  if (a.isEmpty) return <List<double>>[];
  final int rows = a.length, cols = a[0].length;
  final List<List<double>> out = List.generate(
    cols,
    (_) => List<double>.filled(rows, 0.0),
  );
  for (int r = 0; r < rows; r++) {
    final List<double> row = a[r];
    for (int c = 0; c < cols; c++) {
      out[c][r] = row[c];
    }
  }
  return out;
}

/// Concatenates a list of 2D matrices along axis 0.
List<List<double>> concat0(List<List<List<double>>> parts) {
  final List<List<double>> out = <List<double>>[];
  for (final List<List<double>> p in parts) {
    out.addAll(p);
  }
  return out;
}

/// Ensures a dynamic list is a proper 2D `List<List<double>>`.
List<List<double>> ensure2D(List<dynamic> raw) {
  return raw
      .map<List<double>>(
        (e) => (e as List).map((v) => (v as num).toDouble()).toList(),
      )
      .toList();
}

/// Converts XYWH bounding box format to XYXY format.
List<double> xywhToXyxy(List<double> xywh) {
  final double cx = xywh[0], cy = xywh[1], w = xywh[2], h = xywh[3];
  return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
}

/// Decodes raw detection model outputs into a standardized 2D matrix.
/// Handles both `[1, numBoxes, 5 + classes]` and
/// `[1, 5 + classes, numBoxes]` formats.
List<List<double>> decodeDetectionOutputs(List<dynamic> outputs) {
  final List<List<List<double>>> parts = <List<List<double>>>[];
  for (final raw in outputs) {
    final List<dynamic> t3d = raw as List;
    if (t3d.length != 1) throw StateError('Unexpected output rank');

    final List<List<double>> out2d = ensure2D(t3d[0]);
    if (out2d.isEmpty) continue;

    final int rows = out2d.length;
    final int cols = out2d[0].length;
    if (rows < cols && (rows == 84 || rows == 85)) {
      parts.add(transpose2D(out2d));
    } else {
      parts.add(out2d);
    }
  }

  final List<List<double>> out = concat0(parts);
  if (out.isEmpty || out[0].length < 84) {
    throw StateError('Expected channels >=84');
  }
  return out;
}
