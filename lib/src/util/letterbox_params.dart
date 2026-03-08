import 'dart:math' as math;

/// Parameters for letterbox preprocessing (aspect-preserving resize + padding).
class LetterboxParams {
  final double scale;
  final int newWidth;
  final int newHeight;
  final int padLeft;
  final int padTop;
  final int padRight;
  final int padBottom;

  const LetterboxParams({
    required this.scale,
    required this.newWidth,
    required this.newHeight,
    required this.padLeft,
    required this.padTop,
    required this.padRight,
    required this.padBottom,
  });
}

/// Computes letterbox parameters for resizing [srcWidth]x[srcHeight] to fit
/// within [targetWidth]x[targetHeight] while preserving aspect ratio.
///
/// The scale factor is `min(targetWidth/srcWidth, targetHeight/srcHeight)`.
/// The image is resized to [newWidth]x[newHeight] and then padded symmetrically
/// to exactly [targetWidth]x[targetHeight]. Any remainder pixel goes to the
/// right/bottom pad (matching integer truncation of the left/top half).
LetterboxParams computeLetterboxParams({
  required int srcWidth,
  required int srcHeight,
  required int targetWidth,
  required int targetHeight,
}) {
  final double scale = math.min(
    targetWidth / srcWidth,
    targetHeight / srcHeight,
  );
  final int newWidth = (srcWidth * scale).round();
  final int newHeight = (srcHeight * scale).round();
  final (padLeft, padRight) = _centeredPad(targetWidth, newWidth);
  final (padTop, padBottom) = _centeredPad(targetHeight, newHeight);

  return LetterboxParams(
    scale: scale,
    newWidth: newWidth,
    newHeight: newHeight,
    padLeft: padLeft,
    padTop: padTop,
    padRight: padRight,
    padBottom: padBottom,
  );
}

/// Parameters for aspect-preserving resize with centered padding.
class AspectPadParams {
  final int newWidth;
  final int newHeight;
  final int padTop;
  final int padBottom;
  final int padLeft;
  final int padRight;

  const AspectPadParams({
    required this.newWidth,
    required this.newHeight,
    required this.padTop,
    required this.padBottom,
    required this.padLeft,
    required this.padRight,
  });
}

/// Computes aspect-preserving resize dimensions and centered padding.
///
/// Selects the smaller of the two axis scale factors so the resized image
/// never exceeds [targetWidth]x[targetHeight], then pads symmetrically to
/// reach the target size. Any remainder pixel goes to the bottom/right pad.
AspectPadParams computeAspectPadParams({
  required int srcWidth,
  required int srcHeight,
  required int targetWidth,
  required int targetHeight,
}) {
  final double asw = targetWidth / srcWidth;
  final double ash = targetHeight / srcHeight;

  final int newWidth;
  final int newHeight;
  if (asw < ash) {
    newWidth = (srcWidth * asw).toInt();
    newHeight = (srcHeight * asw).toInt();
  } else {
    newWidth = (srcWidth * ash).toInt();
    newHeight = (srcHeight * ash).toInt();
  }

  final (padLeft, padRight) = _centeredPad(targetWidth, newWidth);
  final (padTop, padBottom) = _centeredPad(targetHeight, newHeight);

  return AspectPadParams(
    newWidth: newWidth,
    newHeight: newHeight,
    padTop: padTop,
    padBottom: padBottom,
    padLeft: padLeft,
    padRight: padRight,
  );
}

(int, int) _centeredPad(int total, int used) {
  final int before = (total - used) ~/ 2;
  return (before, total - used - before);
}
