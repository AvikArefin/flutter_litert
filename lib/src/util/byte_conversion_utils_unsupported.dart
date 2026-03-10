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

import '../tensor.dart';
import 'byte_conversion_utils_shared.dart' as shared;

typedef ByteConversionError = shared.ByteConversionError;

class ByteConversionUtils {
  static Uint8List convertObjectToBytes(Object o, TensorType tensorType) {
    throw UnsupportedError(
      'ByteConversionUtils is not supported on this platform',
    );
  }

  static List<String> decodeTFStrings(Uint8List bytes) {
    throw UnsupportedError(
      'ByteConversionUtils is not supported on this platform',
    );
  }

  static Object convertBytesToObject(
    Uint8List bytes,
    TensorType tensorType,
    List<int> shape,
  ) {
    throw UnsupportedError(
      'ByteConversionUtils is not supported on this platform',
    );
  }

  static Uint8List floatToFloat16Bytes(double value) {
    throw UnsupportedError(
      'ByteConversionUtils is not supported on this platform',
    );
  }

  static double bytesToFloat32(Uint8List bytes) {
    throw UnsupportedError(
      'ByteConversionUtils is not supported on this platform',
    );
  }
}
