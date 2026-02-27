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

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';

import '../bindings/bindings.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';
import '../ffi/helper.dart';
import 'tensor.dart';

/// A runner for a specific model signature, enabling named tensor access.
///
/// Signatures allow models to expose multiple named entry points. This is the
/// foundation for on-device training, where a model typically exposes:
///
/// - `train`:   runs one training step (forward pass + backprop + optimizer)
/// - `infer`:   runs inference / prediction
/// - `save`:    saves model weights to a checkpoint file
/// - `restore`: restores model weights from a checkpoint file
///
/// ## On-device training example
///
/// ```dart
/// // Load a model that was exported with training signatures.
/// final interpreter = await Interpreter.fromAsset('trainable_model.tflite');
///
/// // Optionally inspect available signatures:
/// print(interpreter.signatureKeys); // ['train', 'infer', 'save', 'restore']
///
/// // --- Training loop ---
/// final trainRunner = interpreter.getSignatureRunner('train');
///
/// for (int epoch = 0; epoch < 10; epoch++) {
///   final lossBuffer = Float32List(1);
///   trainRunner.run(
///     {'x': imageData, 'y': labels},
///     {'loss': lossBuffer},
///   );
///   print('Epoch $epoch loss: ${lossBuffer[0]}');
/// }
/// trainRunner.close();
///
/// // --- Inference ---
/// final inferRunner = interpreter.getSignatureRunner('infer');
/// final predictions = Float32List(10);
/// inferRunner.run({'x': testImage}, {'output': predictions});
/// inferRunner.close();
///
/// // --- Save weights ---
/// final saveRunner = interpreter.getSignatureRunner('save');
/// saveRunner.run({'checkpoint_path': '/path/to/checkpoint'}, {});
/// saveRunner.close();
///
/// interpreter.close();
/// ```
///
/// ## Building a training-capable model (Python)
///
/// Export a TensorFlow model with `train`, `infer`, `save`, and `restore`
/// signatures, then convert with `SELECT_TF_OPS` enabled:
///
/// ```python
/// converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
/// converter.target_spec.supported_ops = [
///   tf.lite.OpsSet.TFLITE_BUILTINS,
///   tf.lite.OpsSet.SELECT_TF_OPS,
/// ]
/// converter.experimental_enable_resource_variables = True
/// tflite_model = converter.convert()
/// ```
class SignatureRunner {
  final Pointer<TfLiteSignatureRunner> _runner;
  bool _closed = false;
  bool _allocated = false;

  /// Creates a [SignatureRunner] from a raw native pointer.
  ///
  /// Typically obtained via [Interpreter.getSignatureRunner].
  SignatureRunner(this._runner) {
    checkArgument(
      isNotNull(_runner),
      message: 'SignatureRunner pointer is null.',
    );
  }

  // ---------------------------------------------------------------------------
  // Input tensors
  // ---------------------------------------------------------------------------

  /// The number of input tensors for this signature.
  int get inputCount =>
      tfliteBinding.TfLiteSignatureRunnerGetInputCount(_runner);

  /// Returns the name of the input tensor at [index].
  String getInputName(int index) {
    return tfliteBinding.TfLiteSignatureRunnerGetInputName(
      _runner,
      index,
    ).cast<Utf8>().toDartString();
  }

  /// The names of all input tensors for this signature.
  List<String> get inputNames =>
      List.generate(inputCount, getInputName, growable: false);

  /// Returns the input [Tensor] identified by [name].
  ///
  /// Throws [ArgumentError] if [name] is not a valid input name.
  Tensor getInputTensor(String name) {
    final namePtr = name.toNativeUtf8();
    try {
      final tensor = tfliteBinding.TfLiteSignatureRunnerGetInputTensor(
        _runner,
        namePtr.cast(),
      );
      checkArgument(
        isNotNull(tensor),
        message:
            'Input tensor "$name" not found. '
            'Valid input names: ${inputNames.join(', ')}',
      );
      return Tensor(tensor);
    } finally {
      calloc.free(namePtr);
    }
  }

  /// Resizes the input tensor identified by [name] to [shape].
  ///
  /// [allocateTensors] must be called again after any resize before invoking.
  void resizeInputTensor(String name, List<int> shape) {
    final namePtr = name.toNativeUtf8();
    final dimensionSize = shape.length;
    final dimensions = calloc<Int>(dimensionSize);
    try {
      final externalTypedData = dimensions.cast<Int32>().asTypedList(
        dimensionSize,
      );
      externalTypedData.setRange(0, dimensionSize, shape);
      final status = tfliteBinding.TfLiteSignatureRunnerResizeInputTensor(
        _runner,
        namePtr.cast(),
        dimensions,
        dimensionSize,
      );
      checkState(status == TfLiteStatus.kTfLiteOk);
      _allocated = false;
    } finally {
      calloc.free(namePtr);
      calloc.free(dimensions);
    }
  }

  // ---------------------------------------------------------------------------
  // Allocation & invocation
  // ---------------------------------------------------------------------------

  /// Allocates memory for all tensors in this signature.
  ///
  /// Must be called after [getSignatureRunner] and after any
  /// [resizeInputTensor] calls, before the first [invoke].
  void allocateTensors() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    checkState(
      tfliteBinding.TfLiteSignatureRunnerAllocateTensors(_runner) ==
          TfLiteStatus.kTfLiteOk,
    );
    _allocated = true;
  }

  /// Runs this signature.
  ///
  /// Ensure input tensors are populated and [allocateTensors] has been called
  /// (or call the higher-level [run] method which handles both automatically).
  void invoke() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    if (!_allocated) {
      allocateTensors();
    }
    checkState(
      tfliteBinding.TfLiteSignatureRunnerInvoke(_runner) ==
          TfLiteStatus.kTfLiteOk,
    );
  }

  // ---------------------------------------------------------------------------
  // Output tensors
  // ---------------------------------------------------------------------------

  /// The number of output tensors for this signature.
  int get outputCount =>
      tfliteBinding.TfLiteSignatureRunnerGetOutputCount(_runner);

  /// Returns the name of the output tensor at [index].
  String getOutputName(int index) {
    return tfliteBinding.TfLiteSignatureRunnerGetOutputName(
      _runner,
      index,
    ).cast<Utf8>().toDartString();
  }

  /// The names of all output tensors for this signature.
  List<String> get outputNames =>
      List.generate(outputCount, getOutputName, growable: false);

  /// Returns the output [Tensor] identified by [name].
  ///
  /// Throws [ArgumentError] if [name] is not a valid output name.
  Tensor getOutputTensor(String name) {
    final namePtr = name.toNativeUtf8();
    try {
      final tensor = tfliteBinding.TfLiteSignatureRunnerGetOutputTensor(
        _runner,
        namePtr.cast(),
      );
      checkArgument(
        isNotNull(tensor),
        message:
            'Output tensor "$name" not found. '
            'Valid output names: ${outputNames.join(', ')}',
      );
      return Tensor(tensor);
    } finally {
      calloc.free(namePtr);
    }
  }

  // ---------------------------------------------------------------------------
  // High-level run API
  // ---------------------------------------------------------------------------

  /// Runs this signature with named [inputs], writing results to [outputs].
  ///
  /// Automatically resizes input tensors if shapes differ, allocates tensors
  /// if needed, and copies output data into the provided objects.
  ///
  /// [inputs]  — map from input tensor name to data (List, Uint8List, etc.)
  /// [outputs] — map from output tensor name to a pre-allocated object that
  ///             will receive the output data (List, Float32List, etc.).
  ///             Pass an empty map if the signature produces no outputs you
  ///             need to read (e.g. the `save` signature).
  ///
  /// Example — training step:
  /// ```dart
  /// final lossBuffer = Float32List(1);
  /// trainRunner.run(
  ///   {'x': imageData, 'y': labels},
  ///   {'loss': lossBuffer},
  /// );
  /// ```
  ///
  /// Example — checkpoint save:
  /// ```dart
  /// saveRunner.run({'checkpoint_path': '/data/model.ckpt'}, {});
  /// ```
  void run(Map<String, Object> inputs, Map<String, Object> outputs) {
    if (inputs.isEmpty) {
      // Some signatures (like `save`) may need their inputs set separately.
      // We allow empty inputs here but let the user set tensors manually.
    }

    // Resize input tensors if shapes differ.
    for (final entry in inputs.entries) {
      final tensor = getInputTensor(entry.key);
      final newShape = tensor.getInputShapeIfDifferent(entry.value);
      if (newShape != null) {
        resizeInputTensor(entry.key, newShape);
      }
    }

    if (!_allocated) {
      allocateTensors();
    }

    // Copy input data into native tensors.
    for (final entry in inputs.entries) {
      getInputTensor(entry.key).setTo(entry.value);
    }

    invoke();

    // Copy output data back to Dart objects.
    for (final entry in outputs.entries) {
      getOutputTensor(entry.key).copyTo(entry.value);
    }
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /// Destroys this signature runner and releases native resources.
  ///
  /// Must be called before the [Interpreter] that created this runner is
  /// closed. After calling [close], this object must not be used.
  void close() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    tfliteBinding.TfLiteSignatureRunnerDelete(_runner);
    _closed = true;
  }

  /// Whether [close] has been called on this runner.
  bool get isClosed => _closed;

  @override
  String toString() =>
      'SignatureRunner{inputs: $inputNames, outputs: $outputNames, '
      'closed: $_closed}';
}
