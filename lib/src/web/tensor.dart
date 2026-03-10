import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../quantization_params.dart';
import '../util/byte_conversion_utils_web.dart';
import '../util/list_shape_extension.dart';
import '../util/tensor_shape_utils.dart' as shape_utils;

/// Web implementation of Tensor.
///
/// On web, tensors are lightweight data containers rather than FFI pointer
/// wrappers. They hold tensor metadata (name, shape, type) and optionally
/// a data buffer.
class Tensor {
  final String _name;
  final TensorType _type;
  final List<int> _shape;
  Uint8List _data;

  Tensor.fromMetadata({
    required String name,
    required TensorType type,
    required List<int> shape,
  }) : _name = name,
       _type = type,
       _shape = shape,
       _data = Uint8List(_computeByteSize(type, shape));

  /// Creates a Tensor compatible with the native constructor signature.
  /// On web this is only used internally.
  Tensor(dynamic tensor)
    : _name = '',
      _type = TensorType.float32,
      _shape = [],
      _data = Uint8List(0);

  /// Name of the tensor element.
  String get name => _name;

  /// Data type of the tensor element.
  TensorType get type => _type;

  /// Dimensions of the tensor.
  List<int> get shape => _shape;

  /// Underlying data buffer as bytes.
  Uint8List get data => _data.asUnmodifiableView();

  /// Quantization params (not available on web).
  QuantizationParams get params => QuantizationParams(1.0, 0);

  /// Updates the underlying data buffer.
  set data(Uint8List bytes) {
    _data = bytes;
  }

  /// Returns number of dimensions.
  int numDimensions() => _shape.length;

  /// Returns the size, in bytes, of the tensor data.
  int numBytes() => _data.length;

  /// Returns the number of elements in a flattened (1-D) view of the tensor.
  int numElements() => computeNumElements(_shape);

  /// Returns the number of elements in a flattened (1-D) view of the tensor's shape.
  static int computeNumElements(List<int> shape) =>
      shape_utils.computeNumElements(shape);

  /// Returns shape of an object as an int list.
  static List<int> computeShapeOf(Object o) => shape_utils.computeShapeOf(o);

  /// Returns the number of dimensions of a multi-dimensional array, otherwise 0.
  static int computeNumDimensions(Object? o) =>
      shape_utils.computeNumDimensions(o);

  /// Recursively populates the shape dimensions.
  static void fillShape(Object o, int dim, List<int>? shape) =>
      shape_utils.fillShape(o, dim, shape);

  /// Returns data type of given object.
  static int dataTypeOf(Object o) {
    while (o is List) {
      o = o.elementAt(0);
    }
    if (o is double) return TensorType.float32.value;
    if (o is int) return TensorType.int32.value;
    if (o is String) return TensorType.string.value;
    if (o is bool) return TensorType.boolean.value;
    throw ArgumentError(
      'DataType error: cannot resolve DataType of ${o.runtimeType}',
    );
  }

  void setTo(Object src) {
    _data = ByteConversionUtils.convertObjectToBytes(src, _type);
  }

  Object copyTo(Object dst) {
    Object obj;
    if (dst is Uint8List) {
      obj = _data;
    } else if (dst is ByteBuffer) {
      ByteData bdata = dst.asByteData();
      for (int i = 0; i < bdata.lengthInBytes; i++) {
        bdata.setUint8(i, _data[i]);
      }
      obj = bdata.buffer;
    } else {
      obj = ByteConversionUtils.convertBytesToObject(_data, _type, _shape);
    }
    if (obj is List && dst is List) {
      _duplicateList(obj, dst);
    } else {
      dst = obj;
    }
    return obj;
  }

  void _duplicateList(List obj, List dst) {
    var objShape = obj.shape;
    var dstShape = dst.shape;
    var equal = true;
    if (objShape.length == dst.shape.length) {
      for (var i = 0; i < objShape.length; i++) {
        if (objShape[i] != dstShape[i]) {
          equal = false;
          break;
        }
      }
    } else {
      equal = false;
    }
    if (!equal) {
      throw ArgumentError(
        'Output object shape mismatch, interpreter returned output of shape: ${obj.shape} while shape of output provided as argument in run is: ${dst.shape}',
      );
    }
    for (var i = 0; i < obj.length; i++) {
      dst[i] = obj[i];
    }
  }

  List<int>? getInputShapeIfDifferent(Object? input) {
    if (input == null) return null;
    if (input is ByteBuffer || input is Uint8List) return null;
    final inputShape = computeShapeOf(input);
    if (listEquals(inputShape, _shape)) return null;
    return inputShape;
  }

  @override
  String toString() =>
      'Tensor{name: $_name, type: $_type, shape: $_shape, data: ${_data.length}}';
}

/// Computes the byte size for a given tensor type and shape.
int _computeByteSize(TensorType type, List<int> shape) {
  int numElements = 1;
  for (final dim in shape) {
    numElements *= dim;
  }
  switch (type) {
    case TensorType.float32:
    case TensorType.int32:
      return numElements * 4;
    case TensorType.float64:
    case TensorType.int64:
      return numElements * 8;
    case TensorType.float16:
    case TensorType.int16:
    case TensorType.uint16:
      return numElements * 2;
    case TensorType.int8:
    case TensorType.uint8:
      return numElements;
    default:
      return numElements * 4;
  }
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
    for (final t in TensorType.values) {
      if (t.value == tfLiteValue) return t;
    }
    return TensorType.noType;
  }

  final int value;

  @override
  String toString() => name;
}
