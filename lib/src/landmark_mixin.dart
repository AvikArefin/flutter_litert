import 'point.dart';

/// Mixin providing normalized coordinate utilities for landmarks with x/y pixel coordinates.
mixin LandmarkMixin {
  double get x;
  double get y;

  double xNorm(int imageWidth) => (x / imageWidth).clamp(0.0, 1.0);
  double yNorm(int imageHeight) => (y / imageHeight).clamp(0.0, 1.0);
  Point toPixel(int imageWidth, int imageHeight) => Point(x, y);
}
