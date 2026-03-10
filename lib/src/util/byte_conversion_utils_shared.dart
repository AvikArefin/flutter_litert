import 'dart:convert';
import 'dart:typed_data';

/// Size of int32 in bytes (avoids dart:ffi dependency for web compatibility).
const int kInt32ByteSize = 4;

class ByteConversionError extends ArgumentError {
  ByteConversionError({required this.input, required this.tensorType})
    : super(
        'The input element is ${input.runtimeType} while tensor data type is $tensorType',
      );

  final Object input;
  final Object tensorType;
}

/// Converts a float32 value to float16 bytes (little-endian).
Uint8List floatToFloat16Bytes(double value) {
  int float16 = float32ToFloat16(value);
  final ByteData byteDataBuffer = ByteData(2)
    ..setUint16(0, float16, Endian.little);
  return Uint8List.fromList(byteDataBuffer.buffer.asUint8List());
}

/// Converts float16 bytes to a float32 value.
double bytesToFloat32(Uint8List bytes) {
  final ByteData byteDataBuffer = ByteData(2);
  int float16 = byteDataBuffer.buffer
      .asUint8List()
      .buffer
      .asByteData()
      .getUint16(0, Endian.little);
  return float16ToFloat32(float16);
}

/// Converts a float32 to float16 representation (as int).
int float32ToFloat16(double value) {
  final Float32List float32Buffer = Float32List(1);
  final Uint32List int32Buffer = float32Buffer.buffer.asUint32List();

  float32Buffer[0] = value;
  int f = int32Buffer[0];
  int sign = (f >> 16) & 0x8000;
  int exponent = (f >> 23) & 0xFF;
  int mantissa = f & 0x007FFFFF;

  if (exponent == 0) return sign;
  if (exponent == 255) return sign | 0x7C00;

  exponent = exponent - 127 + 15;
  if (exponent >= 31) return sign | 0x7C00;
  if (exponent <= 0) return sign;

  // Implement rounding
  int roundMantissa = (mantissa >> 13) + ((mantissa >> 12) & 1);

  return sign | (exponent << 10) | roundMantissa;
}

/// Converts a float16 (as int) to float32.
double float16ToFloat32(int value) {
  final Float32List float32Buffer = Float32List(1);
  final Uint32List int32Buffer = float32Buffer.buffer.asUint32List();

  int sign = (value & 0x8000) << 16;
  int exponent = (value & 0x7C00) >> 10;
  int mantissa = (value & 0x03FF) << 13;

  if (exponent == 0) {
    if (mantissa == 0) return sign == 0 ? 0.0 : -0.0;
    while ((mantissa & 0x00800000) == 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    exponent += 1;
  } else if (exponent == 31) {
    if (mantissa == 0) {
      return sign == 0 ? double.infinity : double.negativeInfinity;
    }
    return double.nan;
  }

  exponent = exponent - 15 + 127;
  int32Buffer[0] = sign | (exponent << 23) | mantissa;

  return float32Buffer[0];
}

/// Decodes a TensorFlow string tensor's binary format to a `List<String>`.
List<String> decodeTFStrings(Uint8List bytes) {
  List<String> decodedStrings = [];

  int numStrings = ByteData.view(
    bytes.sublist(0, kInt32ByteSize).buffer,
  ).getInt32(0, Endian.little);

  for (int s = 0; s < numStrings; s++) {
    int startIdx = ByteData.view(
      bytes.sublist((1 + s) * kInt32ByteSize, (2 + s) * kInt32ByteSize).buffer,
    ).getInt32(0, Endian.little);
    int endIdx = ByteData.view(
      bytes.sublist((2 + s) * kInt32ByteSize, (3 + s) * kInt32ByteSize).buffer,
    ).getInt32(0, Endian.little);

    decodedStrings.add(utf8.decode(bytes.sublist(startIdx, endIdx)));
  }

  return decodedStrings;
}
