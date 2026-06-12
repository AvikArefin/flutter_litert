import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

const String kLiteRtLib = '/tmp/cm_spike/libLiteRt.dylib';
const int kOk = 0;
const int kCpu = 1;
const int kGpu = 2;
const int kLiteRtLayoutSize = 68;
const int kLiteRtRankedTensorTypeSize = 72;

typedef _P = Pointer<Void>;
typedef _PP = Pointer<Pointer<Void>>;

final class LiteRtLayout extends Struct {
  @Uint32()
  external int bitfields;

  @Array(8)
  external Array<Int32> dimensions;

  @Array(8)
  external Array<Uint32> strides;
}

final class LiteRtRankedTensorType extends Struct {
  @Int32()
  external int elementType;

  external LiteRtLayout layout;
}

final class LiteRtQuantizationPerTensor extends Struct {
  @Float()
  external double scale;

  @Int64()
  external int zeroPoint;
}

final class LiteRtQuantizationPerChannel extends Struct {
  @Int32()
  external int quantizedDimension;

  @Uint64()
  external int numChannels;

  external Pointer<Float> scales;
  external Pointer<Int64> zeroPoints;
}

final class ModelCase {
  const ModelCase(this.label, this.path);

  final String label;
  final String path;
}

const List<ModelCase> kModels = [
  ModelCase(
    'gesture_embedder',
    '/Users/hugocornellier/IdeaProjects/hand_detection/assets/models/gesture_embedder.tflite',
  ),
  ModelCase(
    'superanimal_ssdlite_f16',
    '/Users/hugocornellier/IdeaProjects/animal_detection/assets/models/superanimal_ssdlite_float16.tflite',
  ),
  ModelCase(
    'species_classifier_f16',
    '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/species_classifier_float16.tflite',
  ),
  ModelCase(
    'mobilefacenet',
    '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/mobilefacenet.tflite',
  ),
  ModelCase(
    'full_range_sparse',
    '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_full_range_sparse.tflite',
  ),
  ModelCase(
    'face_detection_back(CTRL)',
    '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_back.tflite',
  ),
];

final Map<int, String> kElementTypeNames = {
  0: 'None',
  1: 'Float32',
  2: 'Int32',
  3: 'UInt8',
  4: 'Int64',
  5: 'TfString',
  6: 'Bool',
  7: 'Int16',
  8: 'Complex64',
  9: 'Int8',
  10: 'Float16',
  11: 'Float64',
  12: 'Complex128',
  13: 'UInt64',
  14: 'TfResource',
  15: 'TfVariant',
  16: 'UInt32',
  17: 'UInt16',
  18: 'Int4',
  19: 'BFloat16',
  20: 'Int2',
};

final Map<int, String> kQuantizationTypeNames = {
  0: 'none',
  1: 'per-tensor',
  2: 'per-channel',
  3: 'block-wise',
};

final Map<int, String> kStatusNames = {
  0: 'kLiteRtStatusOk',
  1: 'kLiteRtStatusErrorInvalidArgument',
  2: 'kLiteRtStatusErrorMemoryAllocationFailure',
  3: 'kLiteRtStatusErrorRuntimeFailure',
  4: 'kLiteRtStatusErrorMissingInputTensor',
  5: 'kLiteRtStatusErrorUnsupported',
  6: 'kLiteRtStatusErrorNotFound',
  7: 'kLiteRtStatusErrorTimeoutExpired',
  8: 'kLiteRtStatusErrorWrongVersion',
  9: 'kLiteRtStatusErrorUnknown',
  10: 'kLiteRtStatusErrorAlreadyExists',
  100: 'kLiteRtStatusCancelled',
  500: 'kLiteRtStatusErrorFileIO',
  501: 'kLiteRtStatusErrorInvalidFlatbuffer',
  502: 'kLiteRtStatusErrorDynamicLoading',
  503: 'kLiteRtStatusErrorSerialization',
  504: 'kLiteRtStatusErrorCompilation',
  1000: 'kLiteRtStatusErrorIndexOOB',
  1001: 'kLiteRtStatusErrorInvalidIrType',
  1002: 'kLiteRtStatusErrorInvalidGraphInvariant',
  1003: 'kLiteRtStatusErrorGraphModification',
  1500: 'kLiteRtStatusErrorInvalidToolConfig',
  2000: 'kLiteRtStatusLegalizeNoMatch',
  2001: 'kLiteRtStatusErrorInvalidLegalization',
  3000: 'kLiteRtStatusPatternNoMatch',
  3001: 'kLiteRtStatusInvalidTransformation',
  4000: 'kLiteRtStatusErrorUnsupportedRuntimeVersion',
  4001: 'kLiteRtStatusErrorUnsupportedCompilerVersion',
  4002: 'kLiteRtStatusErrorIncompatibleByteCodeVersion',
  5000: 'kLiteRtStatusErrorUnsupportedOpShapeInferer',
  5001: 'kLiteRtStatusErrorShapeInferenceFailed',
};

final class _Api {
  _Api(DynamicLibrary rt)
    : createEnv = rt.lookupFunction<
        Int32 Function(Int32, _P, _PP),
        int Function(int, _P, _PP)
      >('LiteRtCreateEnvironment'),
      destroyEnv = rt.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyEnvironment',
      ),
      createOpts = rt.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
        'LiteRtCreateOptions',
      ),
      destroyOpts = rt.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyOptions',
      ),
      setAccel = rt
          .lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
            'LiteRtSetOptionsHardwareAccelerators',
          ),
      createModelFromFile = rt.lookupFunction<
        Int32 Function(Pointer<Utf8>, _PP),
        int Function(Pointer<Utf8>, _PP)
      >('LiteRtCreateModelFromFile'),
      destroyModel = rt.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyModel',
      ),
      createCompiledModel = rt.lookupFunction<
        Int32 Function(_P, _P, _P, _PP),
        int Function(_P, _P, _P, _PP)
      >('LiteRtCreateCompiledModel'),
      destroyCompiledModel = rt
          .lookupFunction<Void Function(_P), void Function(_P)>(
            'LiteRtDestroyCompiledModel',
          ),
      getStatusString = rt.lookupFunction<
        Pointer<Utf8> Function(Int32),
        Pointer<Utf8> Function(int)
      >('LiteRtGetStatusString'),
      getNumModelSignatures = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumModelSignatures'),
      getModelSignature = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetModelSignature'),
      getSignatureKey = rt
          .lookupFunction<Int32 Function(_P, _PP), int Function(_P, _PP)>(
            'LiteRtGetSignatureKey',
          ),
      getNumSignatureInputs = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureInputs'),
      getNumSignatureOutputs = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureOutputs'),
      getSignatureInputName = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureInputName'),
      getSignatureOutputName = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureOutputName'),
      getSignatureInputTensorByIndex = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureInputTensorByIndex'),
      getSignatureOutputTensorByIndex = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureOutputTensorByIndex'),
      getMainModelSubgraphIndex = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetMainModelSubgraphIndex'),
      getModelSubgraph = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetModelSubgraph'),
      getNumSubgraphInputs = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSubgraphInputs'),
      getNumSubgraphOutputs = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSubgraphOutputs'),
      getSubgraphInput = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSubgraphInput'),
      getSubgraphOutput = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSubgraphOutput'),
      getTensorName = rt
          .lookupFunction<Int32 Function(_P, _PP), int Function(_P, _PP)>(
            'LiteRtGetTensorName',
          ),
      getTensorIndex = rt.lookupFunction<
        Int32 Function(_P, Pointer<Uint32>),
        int Function(_P, Pointer<Uint32>)
      >('LiteRtGetTensorIndex'),
      getRankedTensorType = rt.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
        int Function(_P, Pointer<LiteRtRankedTensorType>)
      >('LiteRtGetRankedTensorType'),
      getQuantizationTypeId = rt.lookupFunction<
        Int32 Function(_P, Pointer<Int32>),
        int Function(_P, Pointer<Int32>)
      >('LiteRtGetQuantizationTypeId'),
      getPerTensorQuantization = rt.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtQuantizationPerTensor>),
        int Function(_P, Pointer<LiteRtQuantizationPerTensor>)
      >('LiteRtGetPerTensorQuantization'),
      getPerChannelQuantization = rt.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtQuantizationPerChannel>),
        int Function(_P, Pointer<LiteRtQuantizationPerChannel>)
      >('LiteRtGetPerChannelQuantization');

  final int Function(int, _P, _PP) createEnv;
  final void Function(_P) destroyEnv;
  final int Function(_PP) createOpts;
  final void Function(_P) destroyOpts;
  final int Function(_P, int) setAccel;
  final int Function(Pointer<Utf8>, _PP) createModelFromFile;
  final void Function(_P) destroyModel;
  final int Function(_P, _P, _P, _PP) createCompiledModel;
  final void Function(_P) destroyCompiledModel;
  final Pointer<Utf8> Function(int) getStatusString;
  final int Function(_P, Pointer<IntPtr>) getNumModelSignatures;
  final int Function(_P, int, _PP) getModelSignature;
  final int Function(_P, _PP) getSignatureKey;
  final int Function(_P, Pointer<IntPtr>) getNumSignatureInputs;
  final int Function(_P, Pointer<IntPtr>) getNumSignatureOutputs;
  final int Function(_P, int, _PP) getSignatureInputName;
  final int Function(_P, int, _PP) getSignatureOutputName;
  final int Function(_P, int, _PP) getSignatureInputTensorByIndex;
  final int Function(_P, int, _PP) getSignatureOutputTensorByIndex;
  final int Function(_P, Pointer<IntPtr>) getMainModelSubgraphIndex;
  final int Function(_P, int, _PP) getModelSubgraph;
  final int Function(_P, Pointer<IntPtr>) getNumSubgraphInputs;
  final int Function(_P, Pointer<IntPtr>) getNumSubgraphOutputs;
  final int Function(_P, int, _PP) getSubgraphInput;
  final int Function(_P, int, _PP) getSubgraphOutput;
  final int Function(_P, _PP) getTensorName;
  final int Function(_P, Pointer<Uint32>) getTensorIndex;
  final int Function(_P, Pointer<LiteRtRankedTensorType>) getRankedTensorType;
  final int Function(_P, Pointer<Int32>) getQuantizationTypeId;
  final int Function(_P, Pointer<LiteRtQuantizationPerTensor>)
  getPerTensorQuantization;
  final int Function(_P, Pointer<LiteRtQuantizationPerChannel>)
  getPerChannelQuantization;
}

void main(List<String> args) async {
  if (sizeOf<LiteRtLayout>() != kLiteRtLayoutSize ||
      sizeOf<LiteRtRankedTensorType>() != kLiteRtRankedTensorTypeSize ||
      sizeOf<LiteRtQuantizationPerTensor>() != 16 ||
      sizeOf<LiteRtQuantizationPerChannel>() != 32) {
    throw StateError(
      'Unexpected LiteRT struct sizes: '
      'layout=${sizeOf<LiteRtLayout>()}, '
      'ranked=${sizeOf<LiteRtRankedTensorType>()}, '
      'perTensorQ=${sizeOf<LiteRtQuantizationPerTensor>()}, '
      'perChannelQ=${sizeOf<LiteRtQuantizationPerChannel>()}',
    );
  }

  final api = _Api(DynamicLibrary.open(kLiteRtLib));
  if (args.isNotEmpty && args.first == '--compile-probe') {
    _runCompileProbe(api, args.skip(1).toList());
    return;
  }

  for (final model in kModels) {
    print('=== ${model.label} ===');
    print('path: ${model.path}');
    if (!File(model.path).existsSync()) {
      print('missing: true');
      print('');
      continue;
    }

    _printTensorMetadata(api, model);
    print('compile:');
    for (final accel in const [kCpu, kGpu]) {
      final output = await _runCompileChild(model, accel);
      stdout.write(output);
    }
    print('');
  }
}

void _printTensorMetadata(_Api api, ModelCase modelCase) {
  final pathPtr = modelCase.path.toNativeUtf8();
  final modelOut = calloc<Pointer<Void>>();
  try {
    final modelStatus = api.createModelFromFile(pathPtr, modelOut);
    print('metadata_model_status: ${_status(api, modelStatus)}');
    if (modelStatus != kOk) return;

    final model = modelOut.value;
    final source = _openIoSource(api, model);
    try {
      print('io_source: ${source.description}');
      for (var i = 0; i < source.numInputs; i++) {
        final tensorOut = calloc<Pointer<Void>>();
        final status = source.getInputTensor(i, tensorOut);
        try {
          if (status != kOk) {
            print('  input[$i]: tensor_status=${_status(api, status)}');
            continue;
          }
          _printTensor(api, 'input', i, tensorOut.value, source.inputName(i));
        } finally {
          calloc.free(tensorOut);
        }
      }
      for (var i = 0; i < source.numOutputs; i++) {
        final tensorOut = calloc<Pointer<Void>>();
        final status = source.getOutputTensor(i, tensorOut);
        try {
          if (status != kOk) {
            print('  output[$i]: tensor_status=${_status(api, status)}');
            continue;
          }
          _printTensor(api, 'output', i, tensorOut.value, source.outputName(i));
        } finally {
          calloc.free(tensorOut);
        }
      }
    } finally {
      source.close();
      api.destroyModel(model);
    }
  } finally {
    malloc.free(pathPtr);
    calloc.free(modelOut);
  }
}

_IoSource _openIoSource(_Api api, _P model) {
  final numSignaturesOut = calloc<IntPtr>();
  final sigOut = calloc<Pointer<Void>>();
  try {
    final numSigStatus = api.getNumModelSignatures(model, numSignaturesOut);
    if (numSigStatus == kOk && numSignaturesOut.value > 0) {
      final sigStatus = api.getModelSignature(model, 0, sigOut);
      if (sigStatus == kOk && sigOut.value != nullptr) {
        return _SignatureIoSource(api, sigOut.value, 0);
      }
    }
  } finally {
    calloc.free(sigOut);
    calloc.free(numSignaturesOut);
  }

  final subgraphIndexOut = calloc<IntPtr>();
  final subgraphOut = calloc<Pointer<Void>>();
  try {
    _ck(
      api,
      'LiteRtGetMainModelSubgraphIndex',
      api.getMainModelSubgraphIndex(model, subgraphIndexOut),
    );
    _ck(
      api,
      'LiteRtGetModelSubgraph',
      api.getModelSubgraph(model, subgraphIndexOut.value, subgraphOut),
    );
    return _SubgraphIoSource(api, subgraphOut.value, subgraphIndexOut.value);
  } finally {
    calloc.free(subgraphOut);
    calloc.free(subgraphIndexOut);
  }
}

abstract interface class _IoSource {
  String get description;
  int get numInputs;
  int get numOutputs;
  int getInputTensor(int index, _PP tensorOut);
  int getOutputTensor(int index, _PP tensorOut);
  String? inputName(int index);
  String? outputName(int index);
  void close();
}

final class _SignatureIoSource implements _IoSource {
  _SignatureIoSource(this.api, this.signature, this.index) {
    final keyOut = calloc<Pointer<Void>>();
    final nInOut = calloc<IntPtr>();
    final nOutOut = calloc<IntPtr>();
    try {
      final keyStatus = api.getSignatureKey(signature, keyOut);
      key =
          keyStatus == kOk && keyOut.value != nullptr
              ? keyOut.value.cast<Utf8>().toDartString()
              : '';
      _ck(
        api,
        'LiteRtGetNumSignatureInputs',
        api.getNumSignatureInputs(signature, nInOut),
      );
      _ck(
        api,
        'LiteRtGetNumSignatureOutputs',
        api.getNumSignatureOutputs(signature, nOutOut),
      );
      numInputs = nInOut.value;
      numOutputs = nOutOut.value;
    } finally {
      calloc.free(nOutOut);
      calloc.free(nInOut);
      calloc.free(keyOut);
    }
  }

  final _Api api;
  final _P signature;
  final int index;
  late final String key;
  @override
  late final int numInputs;
  @override
  late final int numOutputs;

  @override
  String get description =>
      'signature[$index] key="$key" '
      'inputs=$numInputs outputs=$numOutputs';

  @override
  int getInputTensor(int index, _PP tensorOut) =>
      api.getSignatureInputTensorByIndex(signature, index, tensorOut);

  @override
  int getOutputTensor(int index, _PP tensorOut) =>
      api.getSignatureOutputTensorByIndex(signature, index, tensorOut);

  @override
  String? inputName(int index) =>
      _signatureName(api, signature, index, api.getSignatureInputName);

  @override
  String? outputName(int index) =>
      _signatureName(api, signature, index, api.getSignatureOutputName);

  @override
  void close() {}
}

final class _SubgraphIoSource implements _IoSource {
  _SubgraphIoSource(this.api, this.subgraph, this.index) {
    final nInOut = calloc<IntPtr>();
    final nOutOut = calloc<IntPtr>();
    try {
      _ck(
        api,
        'LiteRtGetNumSubgraphInputs',
        api.getNumSubgraphInputs(subgraph, nInOut),
      );
      _ck(
        api,
        'LiteRtGetNumSubgraphOutputs',
        api.getNumSubgraphOutputs(subgraph, nOutOut),
      );
      numInputs = nInOut.value;
      numOutputs = nOutOut.value;
    } finally {
      calloc.free(nOutOut);
      calloc.free(nInOut);
    }
  }

  final _Api api;
  final _P subgraph;
  final int index;
  @override
  late final int numInputs;
  @override
  late final int numOutputs;

  @override
  String get description =>
      'main_subgraph[$index] inputs=$numInputs outputs=$numOutputs';

  @override
  int getInputTensor(int index, _PP tensorOut) =>
      api.getSubgraphInput(subgraph, index, tensorOut);

  @override
  int getOutputTensor(int index, _PP tensorOut) =>
      api.getSubgraphOutput(subgraph, index, tensorOut);

  @override
  String? inputName(int index) => null;

  @override
  String? outputName(int index) => null;

  @override
  void close() {}
}

String? _signatureName(
  _Api api,
  _P signature,
  int index,
  int Function(_P, int, _PP) getName,
) {
  final nameOut = calloc<Pointer<Void>>();
  try {
    final status = getName(signature, index, nameOut);
    if (status != kOk || nameOut.value == nullptr) return null;
    return nameOut.value.cast<Utf8>().toDartString();
  } finally {
    calloc.free(nameOut);
  }
}

void _printTensor(
  _Api api,
  String ioKind,
  int ioIndex,
  _P tensor,
  String? signatureName,
) {
  final rankedType = calloc<LiteRtRankedTensorType>();
  final tensorIndexOut = calloc<Uint32>();
  try {
    final rankedStatus = api.getRankedTensorType(tensor, rankedType);
    if (rankedStatus != kOk) {
      print('  $ioKind[$ioIndex]: ranked_status=${_status(api, rankedStatus)}');
      return;
    }

    final tensorIndexStatus = api.getTensorIndex(tensor, tensorIndexOut);
    final tensorIndex = tensorIndexStatus == kOk ? tensorIndexOut.value : null;
    final tensorName = _tensorName(api, tensor);
    final elementType = rankedType.ref.elementType;
    final dims = _dims(rankedType.ref.layout);
    final q = _quantization(api, tensor);
    final parts = [
      '  $ioKind[$ioIndex]',
      if (signatureName != null) 'signature_name="$signatureName"',
      if (tensorName != null && tensorName.isNotEmpty)
        'tensor_name="$tensorName"',
      if (tensorIndex != null) 'tensor_index=$tensorIndex',
      'element_type=$elementType(${_elementName(elementType)})',
      'quantization=$q',
      'shape=$dims',
    ];
    print(parts.join(' '));
  } finally {
    calloc.free(tensorIndexOut);
    calloc.free(rankedType);
  }
}

String? _tensorName(_Api api, _P tensor) {
  final nameOut = calloc<Pointer<Void>>();
  try {
    final status = api.getTensorName(tensor, nameOut);
    if (status != kOk || nameOut.value == nullptr) return null;
    return nameOut.value.cast<Utf8>().toDartString();
  } finally {
    calloc.free(nameOut);
  }
}

String _quantization(_Api api, _P tensor) {
  final qTypeOut = calloc<Int32>();
  try {
    final qStatus = api.getQuantizationTypeId(tensor, qTypeOut);
    if (qStatus != kOk) {
      return 'status=${_status(api, qStatus)}';
    }
    final qType = qTypeOut.value;
    final qName = kQuantizationTypeNames[qType] ?? 'unknown';
    if (qType == 0) return 'none(qid=0)';
    if (qType == 1) {
      final q = calloc<LiteRtQuantizationPerTensor>();
      try {
        final status = api.getPerTensorQuantization(tensor, q);
        if (status != kOk) {
          return 'per-tensor(qid=1,status=${_status(api, status)})';
        }
        return 'per-tensor(qid=1,scale=${q.ref.scale},zero_point=${q.ref.zeroPoint})';
      } finally {
        calloc.free(q);
      }
    }
    if (qType == 2) {
      final q = calloc<LiteRtQuantizationPerChannel>();
      try {
        final status = api.getPerChannelQuantization(tensor, q);
        if (status != kOk) {
          return 'per-channel(qid=2,status=${_status(api, status)})';
        }
        return 'per-channel(qid=2,quantized_dimension=${q.ref.quantizedDimension},num_channels=${q.ref.numChannels})';
      } finally {
        calloc.free(q);
      }
    }
    return '$qName(qid=$qType)';
  } finally {
    calloc.free(qTypeOut);
  }
}

List<int> _dims(LiteRtLayout layout) {
  final rank = layout.bitfields & 0x7f;
  return [for (var i = 0; i < rank; i++) layout.dimensions[i]];
}

Future<String> _runCompileChild(ModelCase model, int accel) async {
  final scriptPath = Platform.script.toFilePath();
  final result = await Process.run(Platform.resolvedExecutable, [
    scriptPath,
    '--compile-probe',
    model.label,
    accel.toString(),
    model.path,
  ], workingDirectory: Directory.current.path);

  final buffer = StringBuffer();
  final stdoutText = (result.stdout as String).trimRight();
  final stderrText = (result.stderr as String).trimRight();
  if (stdoutText.isNotEmpty) {
    for (final line in stdoutText.split('\n')) {
      buffer.writeln('  $line');
    }
  }
  if (result.exitCode != 0) {
    buffer.writeln(
      '  ${_accelName(accel)}($accel): child_exit_code=${result.exitCode}',
    );
  }
  if (stderrText.isNotEmpty) {
    buffer.writeln('  ${_accelName(accel)}($accel): stderr:');
    for (final line in stderrText.split('\n')) {
      buffer.writeln('    $line');
    }
  }
  return buffer.toString();
}

void _runCompileProbe(_Api api, List<String> args) {
  if (args.length != 3) {
    stderr.writeln('expected: --compile-probe <label> <accel> <path>');
    exitCode = 64;
    return;
  }

  final label = args[0];
  final accel = int.parse(args[1]);
  final path = args[2];
  final envOut = calloc<Pointer<Void>>();
  final optsOut = calloc<Pointer<Void>>();
  final modelOut = calloc<Pointer<Void>>();
  final cmOut = calloc<Pointer<Void>>();
  final pathPtr = path.toNativeUtf8();
  _P? env;
  _P? opts;
  _P? model;
  _P? cm;

  try {
    final envStatus = api.createEnv(0, nullptr, envOut);
    if (envStatus == kOk) env = envOut.value;
    final optsStatus = envStatus == kOk ? api.createOpts(optsOut) : -1;
    if (optsStatus == kOk) opts = optsOut.value;
    final accelStatus = optsStatus == kOk ? api.setAccel(opts!, accel) : -1;
    final modelStatus = api.createModelFromFile(pathPtr, modelOut);
    if (modelStatus == kOk) model = modelOut.value;
    var compileStatus = -1;
    if (envStatus == kOk &&
        optsStatus == kOk &&
        accelStatus == kOk &&
        modelStatus == kOk) {
      compileStatus = api.createCompiledModel(env!, model!, opts!, cmOut);
      if (compileStatus == kOk) cm = cmOut.value;
    }

    print(
      '${_accelName(accel)}($accel): '
      'LiteRtCreateModelFromFile=${_status(api, modelStatus)}; '
      'LiteRtCreateCompiledModel=${compileStatus == -1 ? 'not-attempted' : _status(api, compileStatus)}'
      '${envStatus == kOk && optsStatus == kOk && accelStatus == kOk ? '' : '; setup_statuses env=${_status(api, envStatus)} opts=${_status(api, optsStatus)} accel=${_status(api, accelStatus)}'}'
      ' label=$label',
    );
  } finally {
    if (cm != null && cm != nullptr) api.destroyCompiledModel(cm);
    if (model != null && model != nullptr) api.destroyModel(model);
    if (opts != null && opts != nullptr) api.destroyOpts(opts);
    if (env != null && env != nullptr) api.destroyEnv(env);
    malloc.free(pathPtr);
    calloc.free(cmOut);
    calloc.free(modelOut);
    calloc.free(optsOut);
    calloc.free(envOut);
  }
}

String _elementName(int elementType) =>
    kElementTypeNames[elementType] ?? 'Unknown';

String _accelName(int accel) => switch (accel) {
  kCpu => 'CPU',
  kGpu => 'GPU',
  _ => 'ACCEL',
};

String _status(_Api api, int status) {
  if (status == -1) return 'not-attempted';
  final enumName = kStatusNames[status] ?? 'LiteRtStatusUnknownEnum';
  final cString = api.getStatusString(status);
  final suffix =
      cString == nullptr ? '' : ',${cString.cast<Utf8>().toDartString()}';
  return '$status($enumName$suffix)';
}

void _ck(_Api api, String op, int status) {
  if (status != kOk) {
    throw StateError('$op failed with ${_status(api, status)}');
  }
}
