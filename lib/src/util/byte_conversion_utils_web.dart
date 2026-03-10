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

import 'dart:typed_data';

import '../web/tensor.dart';
import 'byte_conversion_utils_shared.dart' as shared;
import 'list_shape_extension.dart';

typedef ByteConversionError = shared.ByteConversionError;

class ByteConversionUtils {
  static Uint8List convertObjectToBytes(Object o, TensorType tensorType) {
    if (o is Uint8List) {
      return o;
    }
    if (o is ByteBuffer) {
      return o.asUint8List();
    }
    List<int> bytes = <int>[];
    if (o is List) {
      for (var e in o) {
        bytes.addAll(convertObjectToBytes(e, tensorType));
      }
    } else {
      return _convertElementToBytes(o, tensorType);
    }
    return Uint8List.fromList(bytes);
  }

  static Uint8List _convertElementToBytes(Object o, TensorType tensorType) {
    // Float32
    if (tensorType == TensorType.float32) {
      if (o is num) {
        var buffer = Uint8List(4).buffer;
        var bdata = ByteData.view(buffer);
        bdata.setFloat32(0, o.toDouble(), Endian.little);
        return buffer.asUint8List();
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    // Uint8
    if (tensorType == TensorType.uint8) {
      if (o is int) {
        var buffer = Uint8List(1).buffer;
        var bdata = ByteData.view(buffer);
        bdata.setUint8(0, o);
        return buffer.asUint8List();
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    // Int32
    if (tensorType == TensorType.int32) {
      if (o is int) {
        var buffer = Uint8List(4).buffer;
        var bdata = ByteData.view(buffer);
        bdata.setInt32(0, o, Endian.little);
        return buffer.asUint8List();
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    // Int64
    if (tensorType == TensorType.int64) {
      if (o is int) {
        var buffer = Uint8List(8).buffer;
        var bdata = ByteData.view(buffer);
        bdata.setInt64(0, o, Endian.big);
        return buffer.asUint8List();
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    // Int16
    if (tensorType == TensorType.int16) {
      if (o is int) {
        var buffer = Uint8List(2).buffer;
        var bdata = ByteData.view(buffer);
        bdata.setInt16(0, o, Endian.little);
        return buffer.asUint8List();
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    // Float16
    if (tensorType == TensorType.float16) {
      if (o is num) {
        return ByteConversionUtils.floatToFloat16Bytes(o.toDouble());
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    // Int8
    if (tensorType == TensorType.int8) {
      if (o is int) {
        var buffer = Uint8List(1).buffer;
        var bdata = ByteData.view(buffer);
        bdata.setInt8(0, o);
        return buffer.asUint8List();
      }
      throw ByteConversionError(input: o, tensorType: tensorType);
    }

    throw ArgumentError(
      'The input data tfliteType ${o.runtimeType} is unsupported',
    );
  }

  /// Decodes a TensorFlow string to a `List<String>`
  static List<String> decodeTFStrings(Uint8List bytes) =>
      shared.decodeTFStrings(bytes);

  static Object convertBytesToObject(
    Uint8List bytes,
    TensorType tensorType,
    List<int> shape,
  ) {
    // stores flattened data
    List<dynamic> list = [];
    if (tensorType == TensorType.int32) {
      for (var i = 0; i < bytes.length; i += 4) {
        list.add(ByteData.view(bytes.buffer).getInt32(i, Endian.little));
      }
      return list.reshape<int>(shape);
    } else if (tensorType == TensorType.float32) {
      for (var i = 0; i < bytes.length; i += 4) {
        list.add(ByteData.view(bytes.buffer).getFloat32(i, Endian.little));
      }
      return list.reshape<double>(shape);
    } else if (tensorType == TensorType.int16) {
      for (var i = 0; i < bytes.length; i += 2) {
        list.add(ByteData.view(bytes.buffer).getInt16(i, Endian.little));
      }
      return list.reshape<int>(shape);
    } else if (tensorType == TensorType.float16) {
      for (var i = 0; i < bytes.length; i += 2) {
        int float16 = ByteData.view(bytes.buffer).getUint16(i, Endian.little);
        double float32 = shared.float16ToFloat32(float16);
        list.add(float32);
      }
      return list.reshape<double>(shape);
    } else if (tensorType == TensorType.int8) {
      for (var i = 0; i < bytes.length; i += 1) {
        list.add(ByteData.view(bytes.buffer).getInt8(i));
      }
      return list.reshape<int>(shape);
    } else if (tensorType == TensorType.uint8) {
      for (var i = 0; i < bytes.length; i += 1) {
        list.add(ByteData.view(bytes.buffer).getUint8(i));
      }
      return list.reshape<int>(shape);
    } else if (tensorType == TensorType.int64) {
      for (var i = 0; i < bytes.length; i += 8) {
        list.add(ByteData.view(bytes.buffer).getInt64(i));
      }
      return list.reshape<int>(shape);
    } else if (tensorType == TensorType.string) {
      list.add(decodeTFStrings(bytes));
      return list;
    }
    throw UnsupportedError("$tensorType is not Supported.");
  }

  static Uint8List floatToFloat16Bytes(double value) =>
      shared.floatToFloat16Bytes(value);

  static double bytesToFloat32(Uint8List bytes) => shared.bytesToFloat32(bytes);
}
