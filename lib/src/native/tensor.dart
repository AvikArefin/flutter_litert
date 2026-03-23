/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import '../bindings/bindings.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';
import '../util/byte_conversion_utils_native.dart';

import '../ffi/helper.dart';
import '../quantization_params.dart';
import '../tensor_type.dart';
import '../util/list_utils.dart' as list_utils;
import '../util/tensor_shape_utils.dart' as shape_utils;

export '../bindings/tensorflow_lite_bindings_generated.dart' show TfLiteType;
export '../tensor_type.dart';

/// LiteRT tensor.
class Tensor {
  final Pointer<TfLiteTensor> _tensor;

  /// Creates a tensor wrapper around a native tensor pointer.
  Tensor(this._tensor) {
    ArgumentError.checkNotNull(_tensor);
  }

  /// Name of the tensor element.
  String get name =>
      tfliteBinding.TfLiteTensorName(_tensor).cast<Utf8>().toDartString();

  /// Data type of the tensor element.
  TensorType get type =>
      TensorType.fromValue(tfliteBinding.TfLiteTensorType(_tensor));

  /// Dimensions of the tensor.
  List<int> get shape => List.generate(
    tfliteBinding.TfLiteTensorNumDims(_tensor),
    (i) => tfliteBinding.TfLiteTensorDim(_tensor, i),
  );

  /// Underlying data buffer as bytes.
  Uint8List get data {
    final data = cast<Uint8>(tfliteBinding.TfLiteTensorData(_tensor));
    return data
        .asTypedList(tfliteBinding.TfLiteTensorByteSize(_tensor))
        .asUnmodifiableView();
  }

  /// Quantization params associated with the tensor.
  QuantizationParams get params {
    final ref = tfliteBinding.TfLiteTensorQuantizationParams(_tensor);
    return QuantizationParams(ref.scale, ref.zero_point);
  }

  /// Updates the underlying data buffer with new bytes.
  ///
  /// The size must match the size of the tensor.
  set data(Uint8List bytes) {
    final tensorByteSize = tfliteBinding.TfLiteTensorByteSize(_tensor);
    checkArgument(tensorByteSize == bytes.length);
    final data = cast<Uint8>(tfliteBinding.TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    final externalTypedData = data.asTypedList(tensorByteSize);
    externalTypedData.setRange(0, tensorByteSize, bytes);
  }

  /// Returns number of dimensions
  int numDimensions() {
    return tfliteBinding.TfLiteTensorNumDims(_tensor);
  }

  /// Returns the size, in bytes, of the tensor data.
  int numBytes() {
    return tfliteBinding.TfLiteTensorByteSize(_tensor);
  }

  /// Returns the number of elements in a flattened (1-D) view of the tensor.
  int numElements() {
    return shape_utils.computeNumElements(shape);
  }

  /// Copies the given [src] data into this tensor.
  void setTo(Object src) {
    Uint8List bytes = _convertObjectToBytes(src);
    int size = bytes.length;

    // String tensors require buffer reallocation because the pre-allocated
    // buffer size does not match the encoded string data size.
    if (type == TensorType.string) {
      final reallocStatus = tfliteBinding.TfLiteTensorRealloc(size, _tensor);
      checkState(
        reallocStatus == TfLiteStatus.kTfLiteOk,
        message:
            'TfLiteTensorRealloc failed for string tensor '
            '(requested $size bytes, status=$reallocStatus).',
      );
    }

    final ptr = calloc<Uint8>(size);
    checkState(isNotNull(ptr), message: 'unallocated');
    final externalTypedData = ptr.asTypedList(size);
    externalTypedData.setRange(0, bytes.length, bytes);
    try {
      checkState(
        tfliteBinding.TfLiteTensorCopyFromBuffer(
              _tensor,
              ptr.cast(),
              bytes.length,
            ) ==
            TfLiteStatus.kTfLiteOk,
        message:
            'TfLiteTensorCopyFromBuffer failed '
            '(buffer=$size bytes, tensor=${numBytes()} bytes).',
      );
    } finally {
      calloc.free(ptr);
    }
  }

  /// Copies this tensor's data into [dst].
  Object copyTo(Object dst) {
    int size = tfliteBinding.TfLiteTensorByteSize(_tensor);
    final ptr = calloc<Uint8>(size);
    checkState(isNotNull(ptr), message: 'unallocated');
    final externalTypedData = ptr.asTypedList(size);
    checkState(
      tfliteBinding.TfLiteTensorCopyToBuffer(_tensor, ptr.cast(), size) ==
          TfLiteStatus.kTfLiteOk,
    );
    // Clone the data, because once `free(ptr)`, `externalTypedData` will be
    // volatile
    final bytes = externalTypedData.sublist(0);
    late Object obj;
    if (dst is Uint8List) {
      obj = bytes;
    } else if (dst is ByteBuffer) {
      ByteData bdata = dst.asByteData();
      for (int i = 0; i < bdata.lengthInBytes; i++) {
        bdata.setUint8(i, bytes[i]);
      }
      obj = bdata.buffer;
    } else {
      obj = _convertBytesToObject(bytes);
    }
    calloc.free(ptr);
    if (obj is List && dst is List) {
      list_utils.duplicateList(obj, dst);
    }
    return obj;
  }

  Uint8List _convertObjectToBytes(Object o) {
    return ByteConversionUtils.convertObjectToBytes(o, type);
  }

  Object _convertBytesToObject(Uint8List bytes) {
    return ByteConversionUtils.convertBytesToObject(bytes, type, shape);
  }

  List<int>? getInputShapeIfDifferent(Object? input) =>
      list_utils.getInputShapeIfDifferent(input, shape);

  @override
  String toString() {
    return 'Tensor{_tensor: $_tensor, name: $name, type: $type, shape: $shape, data: ${data.length}}';
  }
}
