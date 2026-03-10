// Unsupported platform stub for Tensor.
//
// This file is used when compiling for platforms where neither dart:io
// nor dart:js_interop is available.

import 'dart:typed_data';

import '../quantization_params.dart';

/// TensorFlowLite tensor.
class Tensor {
  Tensor(dynamic tensor) {
    throw UnsupportedError('Tensor is not supported on this platform');
  }

  /// Name of the tensor element.
  String get name =>
      throw UnsupportedError('Tensor.name is not supported on this platform');

  /// Data type of the tensor element.
  TensorType get type =>
      throw UnsupportedError('Tensor.type is not supported on this platform');

  /// Dimensions of the tensor.
  List<int> get shape =>
      throw UnsupportedError('Tensor.shape is not supported on this platform');

  /// Underlying data buffer as bytes.
  Uint8List get data =>
      throw UnsupportedError('Tensor.data is not supported on this platform');

  /// Quantization Params associated with the model, [only Android]
  QuantizationParams get params =>
      throw UnsupportedError('Tensor.params is not supported on this platform');

  /// Updates the underlying data buffer with new bytes.
  set data(Uint8List bytes) =>
      throw UnsupportedError('Tensor.data= is not supported on this platform');

  /// Returns number of dimensions
  int numDimensions() => throw UnsupportedError(
    'Tensor.numDimensions is not supported on this platform',
  );

  /// Returns the size, in bytes, of the tensor data.
  int numBytes() => throw UnsupportedError(
    'Tensor.numBytes is not supported on this platform',
  );

  /// Returns the number of elements in a flattened (1-D) view of the tensor.
  int numElements() => throw UnsupportedError(
    'Tensor.numElements is not supported on this platform',
  );

  /// Returns the number of elements in a flattened (1-D) view of the tensor's shape.
  static int computeNumElements(List<int> shape) => throw UnsupportedError(
    'Tensor.computeNumElements is not supported on this platform',
  );

  /// Returns shape of an object as an int list
  static List<int> computeShapeOf(Object o) => throw UnsupportedError(
    'Tensor.computeShapeOf is not supported on this platform',
  );

  /// Returns the number of dimensions of a multi-dimensional array, otherwise 0.
  static int computeNumDimensions(Object? o) => throw UnsupportedError(
    'Tensor.computeNumDimensions is not supported on this platform',
  );

  /// Recursively populates the shape dimensions for a given (multi-dimensional) array)
  static void fillShape(Object o, int dim, List<int>? shape) =>
      throw UnsupportedError(
        'Tensor.fillShape is not supported on this platform',
      );

  /// Returns data type of given object
  static int dataTypeOf(Object o) => throw UnsupportedError(
    'Tensor.dataTypeOf is not supported on this platform',
  );

  void setTo(Object src) =>
      throw UnsupportedError('Tensor.setTo is not supported on this platform');

  Object copyTo(Object dst) =>
      throw UnsupportedError('Tensor.copyTo is not supported on this platform');

  List<int>? getInputShapeIfDifferent(Object? input) => throw UnsupportedError(
    'Tensor.getInputShapeIfDifferent is not supported on this platform',
  );

  @override
  String toString() => 'Tensor(unsupported platform stub)';
}

enum TensorType {
  noType(0),
  float32(1),
  int32(2),
  uint8(3),
  int64(4),
  string(5),
  boolean(6),
  int16(7),
  complex64(8),
  int8(9),
  float16(10),
  float64(11),
  complex128(12),
  uint64(13),
  resource(14),
  variant(15),
  uint32(16),
  uint16(17),
  int4(18);

  const TensorType(this.value);

  static TensorType fromValue(int tfLiteValue) {
    switch (tfLiteValue) {
      case 1:
        return TensorType.float32;
      case 2:
        return TensorType.int32;
      case 3:
        return TensorType.uint8;
      case 4:
        return TensorType.int64;
      case 5:
        return TensorType.string;
      case 6:
        return TensorType.boolean;
      case 7:
        return TensorType.int16;
      case 8:
        return TensorType.complex64;
      case 9:
        return TensorType.int8;
      case 10:
        return TensorType.float16;
      case 11:
        return TensorType.float64;
      case 12:
        return TensorType.complex128;
      case 13:
        return TensorType.uint64;
      case 14:
        return TensorType.resource;
      case 15:
        return TensorType.variant;
      case 16:
        return TensorType.uint32;
      case 17:
        return TensorType.uint16;
      case 18:
        return TensorType.int4;
      default:
        return TensorType.noType;
    }
  }

  final int value;

  @override
  String toString() => name;
}
