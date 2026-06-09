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

import 'dart:async';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../bindings/litert_ffi.dart';
import '../bindings/litert_loader.dart';

const int _kLiteRtStatusOk = 0;
const int _kLiteRtStatusErrorTimeoutExpired = 7;
const int _kLiteRtHwAcceleratorCpu = 1;
const int _kLiteRtHwAcceleratorGpu = 2;
const int _kLiteRtHwAcceleratorNpu = 4;
const int _kLiteRtElementTypeFloat32 = 1;
const int _kLiteRtTensorBufferLockModeRead = 0;
const int _kLiteRtTensorBufferLockModeWrite = 1;
const int _kLiteRtDelegatePrecisionFp32 = 2;
const int _kRankedTensorTypeLayoutOffset = 4;
const int _kLiteRtLayoutSize = 68;
const int _kLiteRtAnySize = 16;
const int _kLiteRtAnyValueOffset = 8;
const int _kLiteRtEnvOptionSize = 24;
const int _kLiteRtEnvOptionValueOffset = 8;

/// Hardware accelerator requested for LiteRT Next compilation.
enum Accelerator { cpu, gpu, npu }

/// GPU precision mode for LiteRT Next compilation.
enum Precision { fp16, fp32 }

/// LiteRT Next CompiledModel inference API.
class CompiledModel {
  CompiledModel._(
    this._rt,
    this._environment,
    this._options,
    this._model,
    this._modelBuffer,
    this._compiledModel,
    this._inputBuffers,
    this._outputBuffers,
    this._inputByteSizes,
    this._outputByteSizes,
    this._gpuOptionsIdentifier,
  ) : _inputCount = _inputByteSizes.length,
      _outputCount = _outputByteSizes.length;

  final LiteRtBindings _rt;
  final Pointer<Void> _environment;
  final Pointer<Void> _options;
  final Pointer<Void> _model;
  final Pointer<Uint8>? _modelBuffer;
  final Pointer<Void> _compiledModel;
  final Pointer<Pointer<Void>> _inputBuffers;
  final Pointer<Pointer<Void>> _outputBuffers;
  final List<int> _inputByteSizes;
  final List<int> _outputByteSizes;
  final Pointer<Utf8>? _gpuOptionsIdentifier;
  final int _inputCount;
  final int _outputCount;

  bool _closed = false;

  /// Number of input tensors for the default signature.
  int get inputCount => _inputCount;

  /// Number of output tensors for the default signature.
  int get outputCount => _outputCount;

  /// Byte size of each input tensor's managed buffer, index-aligned with [run].
  List<int> get inputByteSizes => List.unmodifiable(_inputByteSizes);

  /// Byte size of each output tensor's managed buffer, index-aligned with [run]'s result.
  List<int> get outputByteSizes => List.unmodifiable(_outputByteSizes);

  /// Creates a compiled model from a model file.
  static CompiledModel fromFile(
    String path, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) {
    return _fromSource(
      accelerators: accelerators,
      precision: precision,
      createModel: (rt) => _createModelFromFile(rt, path),
    );
  }

  /// Creates a compiled model from model bytes.
  ///
  /// The bytes are copied into a native buffer owned by this [CompiledModel] so
  /// the model source remains alive until [close] releases the LiteRT model.
  static CompiledModel fromBuffer(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) {
    return _fromSource(
      accelerators: accelerators,
      precision: precision,
      createModel: (rt) => _createModelFromBuffer(rt, bytes),
    );
  }

  static CompiledModel _fromSource({
    required Set<Accelerator> accelerators,
    required Precision precision,
    required _ModelSource Function(LiteRtBindings rt) createModel,
  }) {
    _checkStructLayouts();
    final acceleratorMask = _acceleratorMask(accelerators);

    final rt = litert;
    Pointer<Void> environment = nullptr;
    Pointer<Void> options = nullptr;
    Pointer<Void> model = nullptr;
    Pointer<Uint8>? modelBuffer;
    Pointer<Void> compiledModel = nullptr;
    Pointer<Pointer<Void>>? inputBuffers;
    Pointer<Pointer<Void>>? outputBuffers;
    Pointer<Utf8>? gpuOptionsIdentifier;
    var createdInputBuffers = 0;
    var createdOutputBuffers = 0;

    try {
      environment = _createEnvironment(rt);
      options = _createOptions(rt, acceleratorMask);
      if (accelerators.contains(Accelerator.gpu) &&
          precision == Precision.fp32) {
        gpuOptionsIdentifier = _addGpuFp32Options(rt, options);
      }
      final source = createModel(rt);
      model = source.model;
      modelBuffer = source.modelBuffer;
      compiledModel = _createCompiledModel(rt, environment, model, options);

      final signature = _getModelSignature(rt, model);
      final inputCount = _getSignatureInputCount(rt, signature);
      final outputCount = _getSignatureOutputCount(rt, signature);

      inputBuffers = calloc<Pointer<Void>>(inputCount);
      final inputByteSizes = <int>[];
      for (var i = 0; i < inputCount; i++) {
        final buffer = _createInputBuffer(
          rt,
          environment,
          compiledModel,
          signature,
          i,
          inputByteSizes,
        );
        inputBuffers[i] = buffer;
        createdInputBuffers++;
      }

      outputBuffers = calloc<Pointer<Void>>(outputCount);
      final outputByteSizes = <int>[];
      final outputLayouts = calloc<LiteRtLayout>(outputCount);
      try {
        _check(
          'LiteRtGetCompiledModelOutputTensorLayouts',
          rt.getCompiledModelOutputTensorLayouts(
            compiledModel,
            0,
            outputCount,
            outputLayouts,
            1,
          ),
        );
        for (var i = 0; i < outputCount; i++) {
          final buffer = _createOutputBuffer(
            rt,
            environment,
            compiledModel,
            signature,
            i,
            outputLayouts + i,
            outputByteSizes,
          );
          outputBuffers[i] = buffer;
          createdOutputBuffers++;
        }
      } finally {
        calloc.free(outputLayouts);
      }

      return CompiledModel._(
        rt,
        environment,
        options,
        model,
        modelBuffer,
        compiledModel,
        inputBuffers,
        outputBuffers,
        inputByteSizes,
        outputByteSizes,
        gpuOptionsIdentifier,
      );
    } catch (_) {
      _releaseNative(
        rt,
        inputBuffers: inputBuffers,
        inputCount: createdInputBuffers,
        outputBuffers: outputBuffers,
        outputCount: createdOutputBuffers,
        compiledModel: compiledModel,
        model: model,
        modelBuffer: modelBuffer,
        options: options,
        environment: environment,
        gpuOptionsIdentifier: gpuOptionsIdentifier,
      );
      rethrow;
    }
  }

  /// Runs inference with Float32 input tensors and returns Float32 outputs.
  List<Float32List> run(List<Float32List> inputs) {
    _ensureOpen();
    if (inputs.length != _inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected $_inputCount input tensors.',
      );
    }

    for (var i = 0; i < _inputCount; i++) {
      _writeInput(i, inputs[i]);
    }

    _check(
      'LiteRtRunCompiledModel',
      _rt.runCompiledModel(
        _compiledModel,
        0,
        _inputCount,
        _inputBuffers,
        _outputCount,
        _outputBuffers,
      ),
    );

    return List<Float32List>.generate(_outputCount, _readOutput);
  }

  /// Runs inference asynchronously when the selected accelerator supports it.
  Future<List<Float32List>> runAsync(List<Float32List> inputs) async {
    _ensureOpen();
    if (inputs.length != _inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected $_inputCount input tensors.',
      );
    }

    for (var i = 0; i < _inputCount; i++) {
      _writeInput(i, inputs[i]);
    }

    final asyncOut = calloc<Uint8>();
    try {
      _check(
        'LiteRtRunCompiledModelAsync',
        _rt.runCompiledModelAsync(
          _compiledModel,
          0,
          _inputCount,
          _inputBuffers,
          _outputCount,
          _outputBuffers,
          asyncOut,
        ),
      );

      if (asyncOut.value != 0) {
        await _waitForAsyncOutputs();
      }

      return List<Float32List>.generate(_outputCount, _readOutput);
    } finally {
      calloc.free(asyncOut);
    }
  }

  /// Releases native CompiledModel resources.
  void close() {
    if (_closed) return;
    _releaseNative(
      _rt,
      inputBuffers: _inputBuffers,
      inputCount: _inputCount,
      outputBuffers: _outputBuffers,
      outputCount: _outputCount,
      compiledModel: _compiledModel,
      model: _model,
      modelBuffer: _modelBuffer,
      options: _options,
      environment: _environment,
      gpuOptionsIdentifier: _gpuOptionsIdentifier,
    );
    _closed = true;
  }

  void _writeInput(int index, Float32List input) {
    final expectedBytes = _inputByteSizes[index];
    final expectedFloats = expectedBytes ~/ sizeOf<Float>();
    if (input.length != expectedFloats) {
      throw ArgumentError.value(
        input.length,
        'inputs[$index].length',
        'Expected $expectedFloats float32 values.',
      );
    }

    final hostAddress = calloc<Pointer<Void>>();
    var locked = false;
    try {
      _check(
        'LiteRtLockTensorBuffer input[$index]',
        _rt.lockTensorBuffer(
          _inputBuffers[index],
          hostAddress,
          _kLiteRtTensorBufferLockModeWrite,
        ),
      );
      locked = true;
      final bytes = hostAddress.value.cast<Uint8>().asTypedList(expectedBytes);
      final floats = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        expectedFloats,
      );
      floats.setAll(0, input);
    } finally {
      if (locked) {
        _check(
          'LiteRtUnlockTensorBuffer input[$index]',
          _rt.unlockTensorBuffer(_inputBuffers[index]),
        );
      }
      calloc.free(hostAddress);
    }
  }

  Future<void> _waitForAsyncOutputs() async {
    final hasEvent = calloc<Uint8>();
    final eventOut = calloc<Pointer<Void>>();
    final pending = List<bool>.filled(_outputCount, true);
    var pendingCount = _outputCount;
    var sawAnyEvent = false;

    try {
      while (pendingCount > 0) {
        _ensureOpen();
        for (var i = 0; i < _outputCount; i++) {
          if (!pending[i]) continue;

          hasEvent.value = 0;
          _check(
            'LiteRtHasTensorBufferEvent output[$i]',
            _rt.hasTensorBufferEvent(_outputBuffers[i], hasEvent),
          );
          if (hasEvent.value == 0) {
            if (sawAnyEvent) {
              pending[i] = false;
              pendingCount--;
            }
            continue;
          }
          sawAnyEvent = true;

          eventOut.value = nullptr;
          _check(
            'LiteRtGetTensorBufferEvent output[$i]',
            _rt.getTensorBufferEvent(_outputBuffers[i], eventOut),
          );
          if (eventOut.value == nullptr) {
            pending[i] = false;
            pendingCount--;
            continue;
          }

          // Timeout 0 polls completion without blocking the Dart isolate.
          final waitStatus = _rt.waitEvent(eventOut.value, 0);
          if (waitStatus == _kLiteRtStatusErrorTimeoutExpired) continue;
          _check('LiteRtWaitEvent output[$i]', waitStatus);

          _check(
            'LiteRtClearTensorBufferEvent output[$i]',
            _rt.clearTensorBufferEvent(_outputBuffers[i]),
          );
          pending[i] = false;
          pendingCount--;
        }

        if (pendingCount > 0) {
          await Future<void>.delayed(Duration.zero);
        }
      }
    } finally {
      calloc.free(eventOut);
      calloc.free(hasEvent);
    }
  }

  Float32List _readOutput(int index) {
    final byteSize = _outputByteSizes[index];
    final floatCount = byteSize ~/ sizeOf<Float>();
    final hostAddress = calloc<Pointer<Void>>();
    var locked = false;
    try {
      _check(
        'LiteRtLockTensorBuffer output[$index]',
        _rt.lockTensorBuffer(
          _outputBuffers[index],
          hostAddress,
          _kLiteRtTensorBufferLockModeRead,
        ),
      );
      locked = true;
      final bytes = hostAddress.value.cast<Uint8>().asTypedList(byteSize);
      final floats = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        floatCount,
      );
      return Float32List.fromList(floats);
    } finally {
      if (locked) {
        _check(
          'LiteRtUnlockTensorBuffer output[$index]',
          _rt.unlockTensorBuffer(_outputBuffers[index]),
        );
      }
      calloc.free(hostAddress);
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('CompiledModel is already closed.');
    }
  }
}

Pointer<Void> _createEnvironment(LiteRtBindings rt) {
  final runtimeDir = litertRuntimeDir;
  final out = calloc<Pointer<Void>>();
  Pointer<LiteRtEnvOption>? envOptions;
  Pointer<Utf8>? runtimeDirPtr;
  try {
    if (runtimeDir == null) {
      _check(
        'LiteRtCreateEnvironment',
        rt.createEnvironment(0, nullptr.cast<LiteRtEnvOption>(), out),
      );
    } else {
      envOptions = calloc<LiteRtEnvOption>();
      runtimeDirPtr = runtimeDir.toNativeUtf8();
      envOptions.ref
        ..tag = kLiteRtEnvOptionTagRuntimeLibraryDir
        ..value.type = kLiteRtAnyTypeString
        ..value.value.strValue = runtimeDirPtr;
      _check(
        'LiteRtCreateEnvironment',
        rt.createEnvironment(1, envOptions, out),
      );
    }
    return out.value;
  } finally {
    if (envOptions != null) {
      calloc.free(envOptions);
    }
    if (runtimeDirPtr != null) {
      malloc.free(runtimeDirPtr);
    }
    calloc.free(out);
  }
}

Pointer<Void> _createOptions(LiteRtBindings rt, int acceleratorMask) {
  final out = calloc<Pointer<Void>>();
  try {
    _check('LiteRtCreateOptions', rt.createOptions(out));
    final options = out.value;
    _check(
      'LiteRtSetOptionsHardwareAccelerators',
      rt.setOptionsHardwareAccelerators(options, acceleratorMask),
    );
    return options;
  } catch (_) {
    if (out.value != nullptr) {
      rt.destroyOptions(out.value);
    }
    rethrow;
  } finally {
    calloc.free(out);
  }
}

Pointer<Utf8> _addGpuFp32Options(LiteRtBindings rt, Pointer<Void> options) {
  final identifier = 'gpu_options'.toNativeUtf8();
  final payload = 'precision = $_kLiteRtDelegatePrecisionFp32\n'.toNativeUtf8();
  final opaqueOut = calloc<Pointer<Void>>();
  var payloadTransferred = false;
  var optionsTransferred = false;

  try {
    _check(
      'LiteRtCreateOpaqueOptions',
      rt.createOpaqueOptions(
        identifier,
        payload.cast<Void>(),
        malloc.nativeFree
            .cast<NativeFunction<LiteRtOpaquePayloadDeleterNative>>(),
        opaqueOut,
      ),
    );
    payloadTransferred = true;
    _check(
      'LiteRtAddOpaqueOptions',
      rt.addOpaqueOptions(options, opaqueOut.value),
    );
    optionsTransferred = true;
    return identifier;
  } finally {
    if (!optionsTransferred && opaqueOut.value != nullptr) {
      rt.destroyOpaqueOptions(opaqueOut.value);
    }
    if (!payloadTransferred) {
      malloc.free(payload);
    }
    calloc.free(opaqueOut);
  }
}

final class _ModelSource {
  const _ModelSource(this.model, [this.modelBuffer]);

  final Pointer<Void> model;
  final Pointer<Uint8>? modelBuffer;
}

_ModelSource _createModelFromFile(LiteRtBindings rt, String path) {
  final pathPtr = path.toNativeUtf8();
  final out = calloc<Pointer<Void>>();
  try {
    _check('LiteRtCreateModelFromFile', rt.createModelFromFile(pathPtr, out));
    return _ModelSource(out.value);
  } finally {
    malloc.free(pathPtr);
    calloc.free(out);
  }
}

_ModelSource _createModelFromBuffer(LiteRtBindings rt, Uint8List bytes) {
  if (bytes.isEmpty) {
    throw ArgumentError.value(
      bytes.length,
      'bytes.length',
      'Must be non-zero.',
    );
  }

  final buffer = malloc<Uint8>(bytes.length);
  buffer.asTypedList(bytes.length).setAll(0, bytes);
  final out = calloc<Pointer<Void>>();

  try {
    _check(
      'LiteRtCreateModelFromBuffer',
      rt.createModelFromBuffer(buffer.cast<Void>(), bytes.length, out),
    );
    return _ModelSource(out.value, buffer);
  } catch (_) {
    malloc.free(buffer);
    rethrow;
  } finally {
    calloc.free(out);
  }
}

Pointer<Void> _createCompiledModel(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> model,
  Pointer<Void> options,
) {
  final out = calloc<Pointer<Void>>();
  try {
    _check(
      'LiteRtCreateCompiledModel',
      rt.createCompiledModel(environment, model, options, out),
    );
    return out.value;
  } finally {
    calloc.free(out);
  }
}

Pointer<Void> _getModelSignature(LiteRtBindings rt, Pointer<Void> model) {
  final out = calloc<Pointer<Void>>();
  try {
    _check('LiteRtGetModelSignature', rt.getModelSignature(model, 0, out));
    return out.value;
  } finally {
    calloc.free(out);
  }
}

int _getSignatureInputCount(LiteRtBindings rt, Pointer<Void> signature) {
  final out = calloc<IntPtr>();
  try {
    _check(
      'LiteRtGetNumSignatureInputs',
      rt.getNumSignatureInputs(signature, out),
    );
    return out.value;
  } finally {
    calloc.free(out);
  }
}

int _getSignatureOutputCount(LiteRtBindings rt, Pointer<Void> signature) {
  final out = calloc<IntPtr>();
  try {
    _check(
      'LiteRtGetNumSignatureOutputs',
      rt.getNumSignatureOutputs(signature, out),
    );
    return out.value;
  } finally {
    calloc.free(out);
  }
}

Pointer<Void> _createInputBuffer(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> compiledModel,
  Pointer<Void> signature,
  int index,
  List<int> inputByteSizes,
) {
  return _withRankedTensorType(
    rt,
    'input[$index]',
    signature,
    index,
    isInput: true,
    (tensorType) {
      final layoutPtr =
          (tensorType.cast<Uint8>() + _kRankedTensorTypeLayoutOffset)
              .cast<LiteRtLayout>();
      _check(
        'LiteRtGetCompiledModelInputTensorLayout input[$index]',
        rt.getCompiledModelInputTensorLayout(
          compiledModel,
          0,
          index,
          layoutPtr,
        ),
      );
      return _createBufferFromRequirements(
        rt,
        environment,
        compiledModel,
        index,
        tensorType,
        inputByteSizes,
        isInput: true,
      );
    },
  );
}

Pointer<Void> _createOutputBuffer(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> compiledModel,
  Pointer<Void> signature,
  int index,
  Pointer<LiteRtLayout> outputLayout,
  List<int> outputByteSizes,
) {
  return _withRankedTensorType(
    rt,
    'output[$index]',
    signature,
    index,
    isInput: false,
    (tensorType) {
      (tensorType.cast<Uint8>() + _kRankedTensorTypeLayoutOffset)
          .asTypedList(_kLiteRtLayoutSize)
          .setAll(
            0,
            outputLayout.cast<Uint8>().asTypedList(_kLiteRtLayoutSize),
          );
      return _createBufferFromRequirements(
        rt,
        environment,
        compiledModel,
        index,
        tensorType,
        outputByteSizes,
        isInput: false,
      );
    },
  );
}

Pointer<Void> _withRankedTensorType(
  LiteRtBindings rt,
  String label,
  Pointer<Void> signature,
  int index,
  Pointer<Void> Function(Pointer<LiteRtRankedTensorType>) action, {
  required bool isInput,
}) {
  final tensorOut = calloc<Pointer<Void>>();
  final tensorType = calloc<LiteRtRankedTensorType>();
  try {
    final tensorStatus = isInput
        ? rt.getSignatureInputTensorByIndex(signature, index, tensorOut)
        : rt.getSignatureOutputTensorByIndex(signature, index, tensorOut);
    _check('LiteRtGetSignatureTensorByIndex $label', tensorStatus);
    _check(
      'LiteRtGetRankedTensorType $label',
      rt.getRankedTensorType(tensorOut.value, tensorType),
    );
    return action(tensorType);
  } finally {
    calloc.free(tensorType);
    calloc.free(tensorOut);
  }
}

Pointer<Void> _createBufferFromRequirements(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> compiledModel,
  int index,
  Pointer<LiteRtRankedTensorType> tensorType,
  List<int> byteSizes, {
  required bool isInput,
}) {
  final requirementsOut = calloc<Pointer<Void>>();
  final sizeOut = calloc<IntPtr>();
  final bufferOut = calloc<Pointer<Void>>();
  final label = isInput ? 'input[$index]' : 'output[$index]';

  try {
    final requirementsStatus = isInput
        ? rt.getCompiledModelInputBufferRequirements(
            compiledModel,
            0,
            index,
            requirementsOut,
          )
        : rt.getCompiledModelOutputBufferRequirements(
            compiledModel,
            0,
            index,
            requirementsOut,
          );
    _check(
      'LiteRtGetCompiledModelBufferRequirements $label',
      requirementsStatus,
    );
    _check(
      'LiteRtGetTensorBufferRequirementsBufferSize $label',
      rt.getTensorBufferRequirementsBufferSize(requirementsOut.value, sizeOut),
    );
    _checkFloat32Tensor(label, tensorType, sizeOut.value);
    _check(
      'LiteRtCreateManagedTensorBufferFromRequirements $label',
      rt.createManagedTensorBufferFromRequirements(
        environment,
        tensorType,
        requirementsOut.value,
        bufferOut,
      ),
    );
    byteSizes.add(sizeOut.value);
    return bufferOut.value;
  } finally {
    // The requirements handle is borrowed from LiteRtCompiledModel; only the
    // holder pointer is owned here.
    calloc.free(bufferOut);
    calloc.free(sizeOut);
    calloc.free(requirementsOut);
  }
}

void _releaseNative(
  LiteRtBindings rt, {
  Pointer<Pointer<Void>>? inputBuffers,
  int inputCount = 0,
  Pointer<Pointer<Void>>? outputBuffers,
  int outputCount = 0,
  Pointer<Void>? compiledModel,
  Pointer<Void>? model,
  Pointer<Uint8>? modelBuffer,
  Pointer<Void>? options,
  Pointer<Void>? environment,
  Pointer<Utf8>? gpuOptionsIdentifier,
}) {
  if (outputBuffers != null) {
    for (var i = 0; i < outputCount; i++) {
      if (outputBuffers[i] != nullptr) {
        rt.destroyTensorBuffer(outputBuffers[i]);
      }
    }
    calloc.free(outputBuffers);
  }
  if (inputBuffers != null) {
    for (var i = 0; i < inputCount; i++) {
      if (inputBuffers[i] != nullptr) {
        rt.destroyTensorBuffer(inputBuffers[i]);
      }
    }
    calloc.free(inputBuffers);
  }
  if (compiledModel != null && compiledModel != nullptr) {
    rt.destroyCompiledModel(compiledModel);
  }
  if (model != null && model != nullptr) {
    rt.destroyModel(model);
  }
  if (modelBuffer != null && modelBuffer != nullptr) {
    malloc.free(modelBuffer);
  }
  if (options != null && options != nullptr) {
    rt.destroyOptions(options);
  }
  if (environment != null && environment != nullptr) {
    rt.destroyEnvironment(environment);
  }
  if (gpuOptionsIdentifier != null && gpuOptionsIdentifier != nullptr) {
    malloc.free(gpuOptionsIdentifier);
  }
}

int _acceleratorMask(Set<Accelerator> accelerators) {
  if (accelerators.isEmpty) {
    throw ArgumentError.value(
      accelerators,
      'accelerators',
      'Must contain at least one accelerator.',
    );
  }

  return accelerators.fold(0, (mask, accelerator) {
    final value = switch (accelerator) {
      Accelerator.cpu => _kLiteRtHwAcceleratorCpu,
      Accelerator.gpu => _kLiteRtHwAcceleratorGpu,
      Accelerator.npu => _kLiteRtHwAcceleratorNpu,
    };
    return mask | value;
  });
}

void _checkFloat32Tensor(
  String label,
  Pointer<LiteRtRankedTensorType> tensorType,
  int byteSize,
) {
  if (tensorType.ref.elementType != _kLiteRtElementTypeFloat32) {
    throw UnsupportedError(
      'CompiledModel.run supports Float32 tensors only; $label has LiteRT '
      'element type ${tensorType.ref.elementType}.',
    );
  }
  if (byteSize % sizeOf<Float>() != 0) {
    throw StateError('$label byte size $byteSize is not float32-aligned.');
  }
}

void _checkStructLayouts() {
  if (sizeOf<LiteRtLayout>() != _kLiteRtLayoutSize) {
    throw StateError(
      'LiteRtLayout size ${sizeOf<LiteRtLayout>()} != $_kLiteRtLayoutSize.',
    );
  }
  if (sizeOf<LiteRtAny>() != _kLiteRtAnySize) {
    throw StateError(
      'LiteRtAny size ${sizeOf<LiteRtAny>()} != $_kLiteRtAnySize.',
    );
  }
  if (sizeOf<LiteRtEnvOption>() != _kLiteRtEnvOptionSize) {
    throw StateError(
      'LiteRtEnvOption size ${sizeOf<LiteRtEnvOption>()} != '
      '$_kLiteRtEnvOptionSize.',
    );
  }
  _checkLiteRtAnyValueOffset();
  _checkLiteRtEnvOptionValueOffset();
  if (sizeOf<LiteRtRankedTensorType>() != 72) {
    throw StateError(
      'LiteRtRankedTensorType size ${sizeOf<LiteRtRankedTensorType>()} != 72.',
    );
  }
}

void _checkLiteRtAnyValueOffset() {
  final any = calloc<LiteRtAny>();
  try {
    any.ref.value.intValue = 0x0102030405060708;
    final bytes = any.cast<Uint8>().asTypedList(_kLiteRtAnySize);
    final value = ByteData.sublistView(
      bytes,
    ).getInt64(_kLiteRtAnyValueOffset, Endian.host);
    if (value != 0x0102030405060708) {
      throw StateError(
        'LiteRtAny union offset does not match $_kLiteRtAnyValueOffset.',
      );
    }
  } finally {
    calloc.free(any);
  }
}

void _checkLiteRtEnvOptionValueOffset() {
  final option = calloc<LiteRtEnvOption>();
  try {
    option.ref.value.type = 0x01020304;
    final bytes = option.cast<Uint8>().asTypedList(_kLiteRtEnvOptionSize);
    final value = ByteData.sublistView(
      bytes,
    ).getInt32(_kLiteRtEnvOptionValueOffset, Endian.host);
    if (value != 0x01020304) {
      throw StateError(
        'LiteRtEnvOption.value offset does not match '
        '$_kLiteRtEnvOptionValueOffset.',
      );
    }
  } finally {
    calloc.free(option);
  }
}

void _check(String operation, int status) {
  if (status != _kLiteRtStatusOk) {
    throw StateError('$operation failed with LiteRtStatus=$status.');
  }
}
