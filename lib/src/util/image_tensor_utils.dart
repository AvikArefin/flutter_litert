import 'dart:typed_data';

/// Converts BGR bytes to a flat Float32List with `0.0..1.0` normalization.
///
/// Performs a BGR-to-RGB channel swap and divides each value by 255.
///
/// Parameters:
/// - [bytes]: Raw BGR image bytes (length must be totalPixels * 3)
/// - [totalPixels]: Total number of pixels (width * height)
/// - [buffer]: Optional pre-allocated Float32List of length totalPixels * 3 to reuse
///
/// Returns a flat Float32List with normalized RGB pixel values in `0.0..1.0`.
Float32List bgrBytesToRgbFloat32({
  required Uint8List bytes,
  required int totalPixels,
  Float32List? buffer,
}) => _bgrToRgbFloat32(
  bytes: bytes,
  totalPixels: totalPixels,
  scale: 1.0 / 255.0,
  offset: 0.0,
  buffer: buffer,
);

/// Converts BGR bytes to a flat Float32List with `-1.0..1.0` normalization.
///
/// Performs a BGR-to-RGB channel swap and normalizes via (value / 127.5) - 1.0,
/// as used by models expecting signed normalized input.
///
/// Parameters:
/// - [bytes]: Raw BGR image bytes (length must be totalPixels * 3)
/// - [totalPixels]: Total number of pixels (width * height)
/// - [buffer]: Optional pre-allocated Float32List of length totalPixels * 3 to reuse
///
/// Returns a flat Float32List with normalized RGB pixel values in `-1.0..1.0`.
Float32List bgrBytesToSignedFloat32({
  required Uint8List bytes,
  required int totalPixels,
  Float32List? buffer,
}) => _bgrToRgbFloat32(
  bytes: bytes,
  totalPixels: totalPixels,
  scale: 1.0 / 127.5,
  offset: -1.0,
  buffer: buffer,
);

Float32List _bgrToRgbFloat32({
  required Uint8List bytes,
  required int totalPixels,
  required double scale,
  required double offset,
  Float32List? buffer,
}) {
  final int size = totalPixels * 3;
  final Float32List tensor = buffer ?? Float32List(size);

  for (int i = 0, j = 0; i < size && j < size; i += 3, j += 3) {
    tensor[j] = bytes[i + 2] * scale + offset;
    tensor[j + 1] = bytes[i + 1] * scale + offset;
    tensor[j + 2] = bytes[i] * scale + offset;
  }
  return tensor;
}

/// Fills a 4D NHWC tensor in-place from raw BGR bytes with a BGR-to-RGB channel swap.
///
/// The [tensor] must already be allocated as `[1][height][width][3]`.
/// Use [scale] and [offset] to control normalization:
/// - For `0.0..1.0`: scale=1/255, offset=0.0 (default)
/// - For `-1.0..1.0`: scale=1/127.5, offset=-1.0
///
/// Parameters:
/// - [bytes]: Raw BGR image bytes (length must be width * height * 3)
/// - [tensor]: Pre-allocated 4D tensor `[1][height][width][3]` to fill
/// - [width]: Image width in pixels
/// - [height]: Image height in pixels
/// - [scale]: Multiplier applied to each byte value before adding offset
/// - [offset]: Value added after scaling
void fillNHWC4DFromBgrBytes({
  required Uint8List bytes,
  required List<List<List<List<double>>>> tensor,
  required int width,
  required int height,
  double scale = 1.0 / 255.0,
  double offset = 0.0,
}) {
  int byteIndex = 0;

  for (int y = 0; y < height; y++) {
    final List<List<double>> row = tensor[0][y];
    for (int x = 0; x < width; x++) {
      final List<double> pixel = row[x];
      pixel[0] = bytes[byteIndex + 2] * scale + offset;
      pixel[1] = bytes[byteIndex + 1] * scale + offset;
      pixel[2] = bytes[byteIndex] * scale + offset;
      byteIndex += 3;
    }
  }
}
