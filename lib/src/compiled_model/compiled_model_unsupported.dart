/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
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

/// Hardware accelerator requested for LiteRT Next compilation.
enum Accelerator { cpu, gpu, npu }

/// GPU precision mode for LiteRT Next compilation.
enum Precision { fp16, fp32 }

/// LiteRT Next CompiledModel inference API.
class CompiledModel {
  CompiledModel._();

  /// Creates a compiled model from a model file.
  static CompiledModel fromFile(
    String path, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Creates a compiled model from model bytes.
  static CompiledModel fromBuffer(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Runs inference with Float32 input tensors and returns Float32 outputs.
  List<Float32List> run(List<Float32List> inputs) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Runs inference asynchronously when the selected accelerator supports it.
  Future<List<Float32List>> runAsync(List<Float32List> inputs) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Releases native CompiledModel resources.
  void close() {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }
}
