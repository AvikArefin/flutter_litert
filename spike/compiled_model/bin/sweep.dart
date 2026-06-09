import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

const int kOk = 0;
const int kTimeout = 7;
const int kCpu = 1;
const int kGpu = 2;
const int kLockWrite = 1;
const int kLiteRtElementTypeFloat32 = 1;
const int kTfLiteFloat32 = 1;
const int kRankedTensorTypeLayoutOffset = 4;
const int kLiteRtLayoutSize = 68;
const int kWarmup = 15;
const int kTimed = 80;
const int kStdoutFd = 1;
const int kStderrFd = 2;
const int kOpenWriteOnly = 1;
const String kCellArg = '--cell';
const String kInterpXnn = 'interp_xnn';
const String kCmCpu = 'cm_cpu';
const String kCmGpuSync = 'cm_gpu_sync';
const String kCmGpuAsync = 'cm_gpu_async';

const List<String> kDefaultDirs = [
  '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets',
  '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models',
  '/Users/hugocornellier/IdeaProjects/hand_detection/assets/models',
  '/Users/hugocornellier/IdeaProjects/pose_detection/assets/models',
  '/Users/hugocornellier/IdeaProjects/object_detection/assets/models',
  '/Users/hugocornellier/IdeaProjects/animal_detection/assets/models',
];

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

final class _Cell {
  const _Cell.ok(this.us) : err = null;
  const _Cell.err(this.err) : us = null;

  final double? us;
  final String? err;

  bool get isOk => us != null;

  String get text => us == null ? 'ERR($err)' : us!.toStringAsFixed(1);
}

final class _BenchRow {
  const _BenchRow({
    required this.label,
    required this.bytes,
    required this.interpXnn,
    required this.cmCpu,
    required this.cmGpuSync,
    required this.cmGpuAsync,
  });

  final String label;
  final int bytes;
  final _Cell interpXnn;
  final _Cell cmCpu;
  final _Cell cmGpuSync;
  final _Cell cmGpuAsync;

  String get bestSpeedup {
    final interp = interpXnn.us;
    if (interp == null) return '-';
    final cmValues = [
      cmCpu.us,
      cmGpuSync.us,
      cmGpuAsync.us,
    ].whereType<double>().toList();
    if (cmValues.isEmpty) return '-';
    cmValues.sort();
    return '${(interp / cmValues.first).toStringAsFixed(2)}x';
  }

  bool get allOk =>
      interpXnn.isOk && cmCpu.isOk && cmGpuSync.isOk && cmGpuAsync.isOk;
}

final class _DtypeException implements Exception {
  const _DtypeException();

  @override
  String toString() => 'dtype';
}

final class _PosixApi {
  _PosixApi(DynamicLibrary lib)
      : open = lib.lookupFunction<Int32 Function(Pointer<Utf8>, Int32),
            int Function(Pointer<Utf8>, int)>('open'),
        dup =
            lib.lookupFunction<Int32 Function(Int32), int Function(int)>('dup'),
        dup2 = lib.lookupFunction<Int32 Function(Int32, Int32),
            int Function(int, int)>('dup2'),
        close = lib.lookupFunction<Int32 Function(Int32), int Function(int)>(
          'close',
        ),
        fflush = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'fflush',
        );

  final int Function(Pointer<Utf8>, int) open;
  final int Function(int) dup;
  final int Function(int, int) dup2;
  final int Function(int) close;
  final int Function(_P) fflush;
}

final class _TfLiteApi {
  _TfLiteApi(DynamicLibrary lib)
      : modelCreate = lib.lookupFunction<_P Function(Pointer<Utf8>),
            _P Function(Pointer<Utf8>)>('TfLiteModelCreateFromFile'),
        modelDelete = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'TfLiteModelDelete',
        ),
        optsCreate = lib.lookupFunction<_P Function(), _P Function()>(
          'TfLiteInterpreterOptionsCreate',
        ),
        optsDelete = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'TfLiteInterpreterOptionsDelete',
        ),
        xnnCreate = lib.lookupFunction<_P Function(_P), _P Function(_P)>(
          'TfLiteXNNPackDelegateCreate',
        ),
        xnnDelete = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'TfLiteXNNPackDelegateDelete',
        ),
        addDelegate =
            lib.lookupFunction<Void Function(_P, _P), void Function(_P, _P)>(
          'TfLiteInterpreterOptionsAddDelegate',
        ),
        interpCreate =
            lib.lookupFunction<_P Function(_P, _P), _P Function(_P, _P)>(
          'TfLiteInterpreterCreate',
        ),
        interpDelete = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'TfLiteInterpreterDelete',
        ),
        allocate = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'TfLiteInterpreterAllocateTensors',
        ),
        inputCount = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'TfLiteInterpreterGetInputTensorCount',
        ),
        inputTensor =
            lib.lookupFunction<_P Function(_P, Int32), _P Function(_P, int)>(
          'TfLiteInterpreterGetInputTensor',
        ),
        tensorData = lib.lookupFunction<_P Function(_P), _P Function(_P)>(
          'TfLiteTensorData',
        ),
        tensorBytes = lib.lookupFunction<IntPtr Function(_P), int Function(_P)>(
          'TfLiteTensorByteSize',
        ),
        tensorType = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'TfLiteTensorType',
        ),
        invoke = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'TfLiteInterpreterInvoke',
        );

  final _P Function(Pointer<Utf8>) modelCreate;
  final void Function(_P) modelDelete;
  final _P Function() optsCreate;
  final void Function(_P) optsDelete;
  final _P Function(_P) xnnCreate;
  final void Function(_P) xnnDelete;
  final void Function(_P, _P) addDelegate;
  final _P Function(_P, _P) interpCreate;
  final void Function(_P) interpDelete;
  final int Function(_P) allocate;
  final int Function(_P) inputCount;
  final _P Function(_P, int) inputTensor;
  final _P Function(_P) tensorData;
  final int Function(_P) tensorBytes;
  final int Function(_P) tensorType;
  final int Function(_P) invoke;
}

final class _LiteRtApi {
  _LiteRtApi(DynamicLibrary lib)
      : createEnv = lib.lookupFunction<Int32 Function(Int32, _P, _PP),
            int Function(int, _P, _PP)>('LiteRtCreateEnvironment'),
        destroyEnv = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyEnvironment',
        ),
        createOpts = lib.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
          'LiteRtCreateOptions',
        ),
        destroyOpts = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyOptions',
        ),
        setAccel = lib
            .lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
          'LiteRtSetOptionsHardwareAccelerators',
        ),
        modelFromFile = lib.lookupFunction<Int32 Function(Pointer<Utf8>, _PP),
            int Function(Pointer<Utf8>, _PP)>('LiteRtCreateModelFromFile'),
        destroyModel = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyModel',
        ),
        createCM = lib.lookupFunction<Int32 Function(_P, _P, _P, _PP),
            int Function(_P, _P, _P, _PP)>('LiteRtCreateCompiledModel'),
        destroyCM = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyCompiledModel',
        ),
        getSig = lib.lookupFunction<Int32 Function(_P, IntPtr, _PP),
            int Function(_P, int, _PP)>('LiteRtGetModelSignature'),
        numIn = lib.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
            int Function(_P, Pointer<IntPtr>)>('LiteRtGetNumSignatureInputs'),
        numOut = lib.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
            int Function(_P, Pointer<IntPtr>)>('LiteRtGetNumSignatureOutputs'),
        inTensor = lib.lookupFunction<Int32 Function(_P, IntPtr, _PP),
            int Function(_P, int, _PP)>('LiteRtGetSignatureInputTensorByIndex'),
        outTensor = lib.lookupFunction<Int32 Function(_P, IntPtr, _PP),
            int Function(_P, int, _PP)>(
          'LiteRtGetSignatureOutputTensorByIndex',
        ),
        rankedType = lib.lookupFunction<
            Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
            int Function(_P, Pointer<LiteRtRankedTensorType>)>(
          'LiteRtGetRankedTensorType',
        ),
        inReq = lib.lookupFunction<Int32 Function(_P, IntPtr, IntPtr, _PP),
            int Function(_P, int, int, _PP)>(
          'LiteRtGetCompiledModelInputBufferRequirements',
        ),
        outReq = lib.lookupFunction<Int32 Function(_P, IntPtr, IntPtr, _PP),
            int Function(_P, int, int, _PP)>(
          'LiteRtGetCompiledModelOutputBufferRequirements',
        ),
        reqSize = lib.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
            int Function(_P, Pointer<IntPtr>)>(
          'LiteRtGetTensorBufferRequirementsBufferSize',
        ),
        createBuf = lib.lookupFunction<
            Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
            int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP)>(
          'LiteRtCreateManagedTensorBufferFromRequirements',
        ),
        destroyBuf = lib.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyTensorBuffer',
        ),
        lock = lib.lookupFunction<Int32 Function(_P, _PP, Int32),
            int Function(_P, _PP, int)>('LiteRtLockTensorBuffer'),
        unlock = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'LiteRtUnlockTensorBuffer',
        ),
        getInLayout = lib.lookupFunction<
            Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
            int Function(_P, int, int, Pointer<LiteRtLayout>)>(
          'LiteRtGetCompiledModelInputTensorLayout',
        ),
        getOutLayouts = lib.lookupFunction<
            Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
            int Function(_P, int, int, Pointer<LiteRtLayout>, int)>(
          'LiteRtGetCompiledModelOutputTensorLayouts',
        ),
        runSync = lib.lookupFunction<
            Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
            int Function(_P, int, int, _PP, int, _PP)>(
          'LiteRtRunCompiledModel',
        ),
        runAsync = lib.lookupFunction<
            Int32 Function(
                _P, IntPtr, IntPtr, _PP, IntPtr, _PP, Pointer<Uint8>),
            int Function(_P, int, int, _PP, int, _PP, Pointer<Uint8>)>(
          'LiteRtRunCompiledModelAsync',
        ),
        hasEvent = lib.lookupFunction<Int32 Function(_P, Pointer<Uint8>),
            int Function(_P, Pointer<Uint8>)>('LiteRtHasTensorBufferEvent'),
        getEvent =
            lib.lookupFunction<Int32 Function(_P, _PP), int Function(_P, _PP)>(
          'LiteRtGetTensorBufferEvent',
        ),
        clearEvent = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'LiteRtClearTensorBufferEvent',
        ),
        waitEvent = lib
            .lookupFunction<Int32 Function(_P, Int64), int Function(_P, int)>(
          'LiteRtWaitEvent',
        );

  final int Function(int, _P, _PP) createEnv;
  final void Function(_P) destroyEnv;
  final int Function(_PP) createOpts;
  final void Function(_P) destroyOpts;
  final int Function(_P, int) setAccel;
  final int Function(Pointer<Utf8>, _PP) modelFromFile;
  final void Function(_P) destroyModel;
  final int Function(_P, _P, _P, _PP) createCM;
  final void Function(_P) destroyCM;
  final int Function(_P, int, _PP) getSig;
  final int Function(_P, Pointer<IntPtr>) numIn;
  final int Function(_P, Pointer<IntPtr>) numOut;
  final int Function(_P, int, _PP) inTensor;
  final int Function(_P, int, _PP) outTensor;
  final int Function(_P, Pointer<LiteRtRankedTensorType>) rankedType;
  final int Function(_P, int, int, _PP) inReq;
  final int Function(_P, int, int, _PP) outReq;
  final int Function(_P, Pointer<IntPtr>) reqSize;
  final int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP) createBuf;
  final void Function(_P) destroyBuf;
  final int Function(_P, _PP, int) lock;
  final int Function(_P) unlock;
  final int Function(_P, int, int, Pointer<LiteRtLayout>) getInLayout;
  final int Function(_P, int, int, Pointer<LiteRtLayout>, int) getOutLayouts;
  final int Function(_P, int, int, _PP, int, _PP) runSync;
  final int Function(_P, int, int, _PP, int, _PP, Pointer<Uint8>) runAsync;
  final int Function(_P, Pointer<Uint8>) hasEvent;
  final int Function(_P, _PP) getEvent;
  final int Function(_P) clearEvent;
  final int Function(_P, int) waitEvent;
}

final class _InterpreterSession {
  _InterpreterSession(this.api, String path) {
    final pathPtr = path.toNativeUtf8();
    try {
      model = api.modelCreate(pathPtr);
      if (model == nullptr) throw StateError('model');
      opts = api.optsCreate();
      if (opts == nullptr) throw StateError('opts');
      delegate = api.xnnCreate(nullptr);
      if (delegate == nullptr) throw StateError('xnnpack');
      api.addDelegate(opts, delegate);
      interp = api.interpCreate(model, opts);
      if (interp == nullptr) throw StateError('interpreter');
      ckTf('AllocateTensors', api.allocate(interp));
      _fillInputs();
    } catch (_) {
      close();
      rethrow;
    } finally {
      malloc.free(pathPtr);
    }
  }

  final _TfLiteApi api;
  _P model = nullptr;
  _P opts = nullptr;
  _P delegate = nullptr;
  _P interp = nullptr;

  double measureMedian({required int warmup, required int iters}) {
    return _timeMedian(() {
      ckTf('Invoke', api.invoke(interp));
    }, warmup, iters);
  }

  void close() {
    if (interp != nullptr) {
      api.interpDelete(interp);
      interp = nullptr;
    }
    if (delegate != nullptr) {
      api.xnnDelete(delegate);
      delegate = nullptr;
    }
    if (opts != nullptr) {
      api.optsDelete(opts);
      opts = nullptr;
    }
    if (model != nullptr) {
      api.modelDelete(model);
      model = nullptr;
    }
  }

  void _fillInputs() {
    final count = api.inputCount(interp);
    for (var i = 0; i < count; i++) {
      final tensor = api.inputTensor(interp, i);
      if (tensor == nullptr) throw StateError('input$i');
      final size = api.tensorBytes(tensor);
      final data = api.tensorData(tensor);
      if (data == nullptr) throw StateError('input_data$i');
      final bytes = data.cast<Uint8>().asTypedList(size);
      if (api.tensorType(tensor) == kTfLiteFloat32 &&
          size % sizeOf<Float>() == 0) {
        final floats = Float32List.view(
          bytes.buffer,
          bytes.offsetInBytes,
          size ~/ sizeOf<Float>(),
        );
        for (var k = 0; k < floats.length; k++) {
          floats[k] = (k % 17) * 0.03125;
        }
      } else {
        for (var k = 0; k < bytes.length; k++) {
          bytes[k] = k & 0xff;
        }
      }
    }
  }
}

final class _CompiledSession {
  _CompiledSession(this.api, String path, this.accel) {
    final envOut = calloc<Pointer<Void>>();
    final optsOut = calloc<Pointer<Void>>();
    final modelOut = calloc<Pointer<Void>>();
    final cmOut = calloc<Pointer<Void>>();
    final sigOut = calloc<Pointer<Void>>();
    final nInP = calloc<IntPtr>();
    final nOutP = calloc<IntPtr>();
    final pathPtr = path.toNativeUtf8();
    try {
      ckRt('LiteRtCreateEnvironment', api.createEnv(0, nullptr, envOut));
      env = envOut.value;
      ckRt('LiteRtCreateOptions', api.createOpts(optsOut));
      opts = optsOut.value;
      ckRt(
        'LiteRtSetOptionsHardwareAccelerators',
        api.setAccel(opts, accel),
      );
      ckRt('LiteRtCreateModelFromFile', api.modelFromFile(pathPtr, modelOut));
      model = modelOut.value;
      ckRt('LiteRtGetModelSignature', api.getSig(model, 0, sigOut));
      sig = sigOut.value;
      ckRt('LiteRtGetNumSignatureInputs', api.numIn(sig, nInP));
      ckRt('LiteRtGetNumSignatureOutputs', api.numOut(sig, nOutP));
      nIn = nInP.value;
      nOut = nOutP.value;
      _checkFloatInputs();

      ckRt('LiteRtCreateCompiledModel', api.createCM(env, model, opts, cmOut));
      cm = cmOut.value;

      inBufs = calloc<Pointer<Void>>(nIn);
      inputByteSizes = List<int>.filled(nIn, 0);
      for (var i = 0; i < nIn; i++) {
        inBufs[i] = _createInputBuffer(i);
        _writeInput(i);
      }

      outLayouts = calloc<LiteRtLayout>(nOut);
      ckRt(
        'LiteRtGetCompiledModelOutputTensorLayouts',
        api.getOutLayouts(cm, 0, nOut, outLayouts, 1),
      );
      singleOutSet = makeOutSet();
    } catch (_) {
      close();
      rethrow;
    } finally {
      malloc.free(pathPtr);
      calloc.free(nOutP);
      calloc.free(nInP);
      calloc.free(sigOut);
      calloc.free(cmOut);
      calloc.free(modelOut);
      calloc.free(optsOut);
      calloc.free(envOut);
    }
  }

  final _LiteRtApi api;
  final int accel;
  _P env = nullptr;
  _P opts = nullptr;
  _P model = nullptr;
  _P cm = nullptr;
  _P sig = nullptr;
  int nIn = 0;
  int nOut = 0;
  _PP inBufs = nullptr;
  List<int> inputByteSizes = const [];
  Pointer<LiteRtLayout> outLayouts = nullptr;
  _PP singleOutSet = nullptr;

  double measureSyncMedian({required int warmup, required int iters}) {
    return _timeMedian(() => _runSync(singleOutSet), warmup, iters);
  }

  double measureAsyncMedian({required int warmup, required int iters}) {
    final asyncOut = calloc<Uint8>();
    try {
      return _timeMedian(() {
        asyncOut.value = 0;
        _runAsyncSingle(singleOutSet, asyncOut);
      }, warmup, iters);
    } finally {
      calloc.free(asyncOut);
    }
  }

  _PP makeOutSet() {
    final set = calloc<Pointer<Void>>(nOut);
    for (var i = 0; i < nOut; i++) {
      set[i] = _createOutputBuffer(i);
    }
    return set;
  }

  void destroyOutSet(_PP set) {
    for (var i = 0; i < nOut; i++) {
      if (set[i] != nullptr) api.destroyBuf(set[i]);
    }
    calloc.free(set);
  }

  void close() {
    if (singleOutSet != nullptr) {
      destroyOutSet(singleOutSet);
      singleOutSet = nullptr;
    }
    if (outLayouts != nullptr) {
      calloc.free(outLayouts);
      outLayouts = nullptr;
    }
    if (inBufs != nullptr) {
      for (var i = 0; i < nIn; i++) {
        if (inBufs[i] != nullptr) api.destroyBuf(inBufs[i]);
      }
      calloc.free(inBufs);
      inBufs = nullptr;
    }
    if (cm != nullptr) {
      api.destroyCM(cm);
      cm = nullptr;
    }
    if (model != nullptr) {
      api.destroyModel(model);
      model = nullptr;
    }
    if (opts != nullptr) {
      api.destroyOpts(opts);
      opts = nullptr;
    }
    if (env != nullptr) {
      api.destroyEnv(env);
      env = nullptr;
    }
  }

  void _checkFloatInputs() {
    for (var i = 0; i < nIn; i++) {
      final tensorOut = calloc<Pointer<Void>>();
      final tensorType = calloc<LiteRtRankedTensorType>();
      try {
        ckRt('LiteRtGetSignatureInputTensorByIndex',
            api.inTensor(sig, i, tensorOut));
        ckRt(
          'LiteRtGetRankedTensorType',
          api.rankedType(tensorOut.value, tensorType),
        );
        if (tensorType.ref.elementType != kLiteRtElementTypeFloat32) {
          throw const _DtypeException();
        }
      } finally {
        calloc.free(tensorType);
        calloc.free(tensorOut);
      }
    }
  }

  _P _createInputBuffer(int index) {
    final tensorOut = calloc<Pointer<Void>>();
    final tensorType = calloc<LiteRtRankedTensorType>();
    final reqOut = calloc<Pointer<Void>>();
    final sizeOut = calloc<IntPtr>();
    final bufOut = calloc<Pointer<Void>>();
    try {
      ckRt('LiteRtGetSignatureInputTensorByIndex',
          api.inTensor(sig, index, tensorOut));
      ckRt(
        'LiteRtGetRankedTensorType',
        api.rankedType(tensorOut.value, tensorType),
      );
      if (tensorType.ref.elementType != kLiteRtElementTypeFloat32) {
        throw const _DtypeException();
      }
      ckRt(
        'LiteRtGetCompiledModelInputTensorLayout',
        api.getInLayout(
          cm,
          0,
          index,
          (tensorType.cast<Uint8>() + kRankedTensorTypeLayoutOffset)
              .cast<LiteRtLayout>(),
        ),
      );
      ckRt(
        'LiteRtGetCompiledModelInputBufferRequirements',
        api.inReq(cm, 0, index, reqOut),
      );
      ckRt(
        'LiteRtGetTensorBufferRequirementsBufferSize',
        api.reqSize(reqOut.value, sizeOut),
      );
      inputByteSizes[index] = sizeOut.value;
      ckRt(
        'LiteRtCreateManagedTensorBufferFromRequirements',
        api.createBuf(env, tensorType, reqOut.value, bufOut),
      );
      return bufOut.value;
    } finally {
      calloc.free(bufOut);
      calloc.free(sizeOut);
      calloc.free(reqOut);
      calloc.free(tensorType);
      calloc.free(tensorOut);
    }
  }

  _P _createOutputBuffer(int index) {
    final tensorOut = calloc<Pointer<Void>>();
    final tensorType = calloc<LiteRtRankedTensorType>();
    final reqOut = calloc<Pointer<Void>>();
    final bufOut = calloc<Pointer<Void>>();
    try {
      ckRt('LiteRtGetSignatureOutputTensorByIndex',
          api.outTensor(sig, index, tensorOut));
      ckRt(
        'LiteRtGetRankedTensorType',
        api.rankedType(tensorOut.value, tensorType),
      );
      (tensorType.cast<Uint8>() + kRankedTensorTypeLayoutOffset)
          .asTypedList(kLiteRtLayoutSize)
          .setAll(
            0,
            (outLayouts + index).cast<Uint8>().asTypedList(kLiteRtLayoutSize),
          );
      ckRt(
        'LiteRtGetCompiledModelOutputBufferRequirements',
        api.outReq(cm, 0, index, reqOut),
      );
      ckRt(
        'LiteRtCreateManagedTensorBufferFromRequirements',
        api.createBuf(env, tensorType, reqOut.value, bufOut),
      );
      return bufOut.value;
    } finally {
      calloc.free(bufOut);
      calloc.free(reqOut);
      calloc.free(tensorType);
      calloc.free(tensorOut);
    }
  }

  void _writeInput(int index) {
    final addrOut = calloc<Pointer<Void>>();
    var locked = false;
    try {
      ckRt(
        'LiteRtLockTensorBuffer input',
        api.lock(inBufs[index], addrOut, kLockWrite),
      );
      locked = true;
      final byteSize = inputByteSizes[index];
      final bytes = addrOut.value.cast<Uint8>().asTypedList(byteSize);
      final floats = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        byteSize ~/ sizeOf<Float>(),
      );
      for (var i = 0; i < floats.length; i++) {
        floats[i] = (i % 17) * 0.03125;
      }
    } finally {
      if (locked) {
        ckRt('LiteRtUnlockTensorBuffer input', api.unlock(inBufs[index]));
      }
      calloc.free(addrOut);
    }
  }

  void _runSync(_PP outSet) {
    ckRt('LiteRtRunCompiledModel',
        api.runSync(cm, 0, nIn, inBufs, nOut, outSet));
  }

  void _runAsyncSingle(_PP outSet, Pointer<Uint8> asyncOut) {
    ckRt(
      'LiteRtRunCompiledModelAsync',
      api.runAsync(cm, 0, nIn, inBufs, nOut, outSet, asyncOut),
    );
    if (asyncOut.value != 0) {
      _pollOutputSetSignaled(outSet);
    }
  }

  void _pollOutputSetSignaled(_PP set) {
    final hasEvent = calloc<Uint8>();
    final eventOut = calloc<Pointer<Void>>();
    final pending = List<bool>.filled(nOut, false);
    var pendingCount = 0;

    try {
      for (var i = 0; i < nOut; i++) {
        hasEvent.value = 0;
        ckRt('LiteRtHasTensorBufferEvent', api.hasEvent(set[i], hasEvent));
        if (hasEvent.value == 0) continue;
        pending[i] = true;
        pendingCount++;
      }

      while (pendingCount > 0) {
        for (var i = 0; i < nOut; i++) {
          if (!pending[i]) continue;

          hasEvent.value = 0;
          ckRt('LiteRtHasTensorBufferEvent', api.hasEvent(set[i], hasEvent));
          if (hasEvent.value == 0) {
            pending[i] = false;
            pendingCount--;
            continue;
          }

          eventOut.value = nullptr;
          ckRt('LiteRtGetTensorBufferEvent', api.getEvent(set[i], eventOut));
          if (eventOut.value == nullptr) {
            pending[i] = false;
            pendingCount--;
            continue;
          }

          final waitStatus = api.waitEvent(eventOut.value, 0);
          if (waitStatus == kTimeout) continue;
          ckRt('LiteRtWaitEvent', waitStatus);
          ckRt('LiteRtClearTensorBufferEvent', api.clearEvent(set[i]));
          pending[i] = false;
          pendingCount--;
        }
      }
    } finally {
      calloc.free(eventOut);
      calloc.free(hasEvent);
    }
  }
}

void main(List<String> args) {
  if (args.isNotEmpty && args.first == kCellArg) {
    _runChildCell(args);
    return;
  }

  if (sizeOf<LiteRtLayout>() != kLiteRtLayoutSize ||
      sizeOf<LiteRtRankedTensorType>() != 72) {
    throw StateError(
      'Unexpected LiteRT struct sizes: layout=${sizeOf<LiteRtLayout>()}, '
      'ranked=${sizeOf<LiteRtRankedTensorType>()}',
    );
  }

  final dirs = args.isEmpty ? kDefaultDirs : args;
  final modelFiles = _discoverModels(dirs);
  modelFiles.sort((a, b) {
    final bySize = a.lengthSync().compareTo(b.lengthSync());
    if (bySize != 0) return bySize;
    return a.path.compareTo(b.path);
  });

  final rows = <_BenchRow>[];
  for (final file in modelFiles) {
    final path = file.path;
    rows.add(_BenchRow(
      label: _basename(path),
      bytes: file.lengthSync(),
      interpXnn: _measureChildCell(kInterpXnn, path),
      cmCpu: _measureChildCell(kCmCpu, path),
      cmGpuSync: _measureChildCell(kCmGpuSync, path),
      cmGpuAsync: _measureChildCell(kCmGpuAsync, path),
    ));
  }

  _printTable(rows);
  _printSummary(rows);
}

void _runChildCell(List<String> args) {
  if (args.length != 3) {
    print('ERR\targs');
    return;
  }

  final config = args[1];
  final path = args[2];
  final posix = _PosixApi(DynamicLibrary.process());
  final cell = _withNativeOutputMuted(posix, () {
    return _measureCell(() {
      switch (config) {
        case kInterpXnn:
          return _benchInterpreter(
            _TfLiteApi(DynamicLibrary.open('./libtensorflowlite_c-mac.dylib')),
            path,
          );
        case kCmCpu:
          return _benchCompiled(
            _LiteRtApi(DynamicLibrary.open('./libLiteRt.dylib')),
            path,
            kCpu,
            async: false,
          );
        case kCmGpuSync:
          return _benchCompiled(
            _LiteRtApi(DynamicLibrary.open('./libLiteRt.dylib')),
            path,
            kGpu,
            async: false,
          );
        case kCmGpuAsync:
          return _benchCompiled(
            _LiteRtApi(DynamicLibrary.open('./libLiteRt.dylib')),
            path,
            kGpu,
            async: true,
          );
        default:
          throw StateError('config');
      }
    });
  });

  if (cell.isOk) {
    print('OK\t${cell.us!.toStringAsPrecision(12)}');
  } else {
    print('ERR\t${cell.err}');
  }
}

_Cell _measureChildCell(String config, String path) {
  final scriptPath = Platform.script.toFilePath();
  final result = Process.runSync(
    Platform.resolvedExecutable,
    [scriptPath, kCellArg, config, path],
    workingDirectory: Directory.current.path,
  );
  if (result.exitCode != 0) {
    return const _Cell.err('crash');
  }

  final output = (result.stdout as String).trim();
  if (output.isEmpty) return const _Cell.err('child');
  final line = output.split('\n').last.trim();
  final parts = line.split('\t');
  if (parts.length != 2) return const _Cell.err('child');
  switch (parts[0]) {
    case 'OK':
      final us = double.tryParse(parts[1]);
      return us == null ? const _Cell.err('child') : _Cell.ok(us);
    case 'ERR':
      return _Cell.err(parts[1].isEmpty ? 'error' : parts[1]);
    default:
      return const _Cell.err('child');
  }
}

List<File> _discoverModels(List<String> roots) {
  final files = <File>[];
  for (final root in roots) {
    final type = FileSystemEntity.typeSync(root);
    if (type == FileSystemEntityType.file) {
      if (root.endsWith('.tflite') && !root.contains('/build/')) {
        files.add(File(root));
      }
      continue;
    }
    if (type != FileSystemEntityType.directory) continue;
    for (final entity in Directory(root).listSync(recursive: true)) {
      if (entity is! File) continue;
      if (!entity.path.endsWith('.tflite')) continue;
      if (entity.path.contains('/build/')) continue;
      files.add(entity);
    }
  }
  return files;
}

double _benchInterpreter(_TfLiteApi api, String path) {
  final session = _InterpreterSession(api, path);
  try {
    return session.measureMedian(warmup: kWarmup, iters: kTimed);
  } finally {
    session.close();
  }
}

double _benchCompiled(_LiteRtApi api, String path, int accel,
    {required bool async}) {
  final session = _CompiledSession(api, path, accel);
  try {
    return async
        ? session.measureAsyncMedian(warmup: kWarmup, iters: kTimed)
        : session.measureSyncMedian(warmup: kWarmup, iters: kTimed);
  } finally {
    session.close();
  }
}

_Cell _measureCell(double Function() f) {
  try {
    return _Cell.ok(f());
  } catch (e) {
    return _Cell.err(_shortReason(e));
  }
}

double _timeMedian(void Function() f, int warmup, int iters) {
  for (var i = 0; i < warmup; i++) {
    f();
  }
  final samples = Float64List(iters);
  for (var i = 0; i < iters; i++) {
    final sw = Stopwatch()..start();
    f();
    sw.stop();
    samples[i] = sw.elapsedTicks * 1000000 / sw.frequency;
  }
  samples.sort();
  return samples[iters ~/ 2];
}

T _withNativeOutputMuted<T>(_PosixApi posix, T Function() f) {
  final devNullPath = '/dev/null'.toNativeUtf8();
  final devNull = posix.open(devNullPath, kOpenWriteOnly);
  malloc.free(devNullPath);
  if (devNull < 0) return f();

  final savedStdout = posix.dup(kStdoutFd);
  final savedStderr = posix.dup(kStderrFd);
  if (savedStdout < 0 || savedStderr < 0) {
    if (savedStdout >= 0) posix.close(savedStdout);
    if (savedStderr >= 0) posix.close(savedStderr);
    posix.close(devNull);
    return f();
  }

  posix.fflush(nullptr);
  posix.dup2(devNull, kStdoutFd);
  posix.dup2(devNull, kStderrFd);
  try {
    return f();
  } finally {
    posix.fflush(nullptr);
    posix.dup2(savedStdout, kStdoutFd);
    posix.dup2(savedStderr, kStderrFd);
    posix.close(savedStdout);
    posix.close(savedStderr);
    posix.close(devNull);
  }
}

void _printTable(List<_BenchRow> rows) {
  const headers = [
    'model',
    'size',
    'interp_xnn',
    'cm_cpu',
    'cm_gpu_sync',
    'cm_gpu_async',
    'best_speedup_vs_interp',
  ];
  final widths = List<int>.from(headers.map((h) => h.length));
  for (final row in rows) {
    final values = _rowValues(row);
    for (var i = 0; i < widths.length; i++) {
      if (values[i].length > widths[i]) widths[i] = values[i].length;
    }
  }

  print(_formatRow(headers, widths));
  print(widths.map((w) => '-' * w).join(' | '));
  for (final row in rows) {
    print(_formatRow(_rowValues(row), widths));
  }
}

void _printSummary(List<_BenchRow> rows) {
  final complete = rows.where((r) => r.allOk).toList();
  final gpuAsyncSpeedups =
      complete.map((r) => r.interpXnn.us! / r.cmGpuAsync.us!).toList();
  final cpuSpeedups =
      complete.map((r) => r.interpXnn.us! / r.cmCpu.us!).toList();
  print(
    'summary: complete_models=${complete.length}/${rows.length} '
    'median_cm_gpu_async_vs_interp=${_formatMedianSpeedup(gpuAsyncSpeedups)}',
  );
  print(
    'summary: complete_models=${complete.length}/${rows.length} '
    'median_cm_cpu_vs_interp=${_formatMedianSpeedup(cpuSpeedups)}',
  );
}

List<String> _rowValues(_BenchRow row) => [
      row.label,
      _formatBytes(row.bytes),
      row.interpXnn.text,
      row.cmCpu.text,
      row.cmGpuSync.text,
      row.cmGpuAsync.text,
      row.bestSpeedup,
    ];

String _formatRow(List<String> values, List<int> widths) {
  return List<String>.generate(values.length, (i) {
    final value = values[i];
    return i == 0 ? value.padRight(widths[i]) : value.padLeft(widths[i]);
  }).join(' | ');
}

String _formatMedianSpeedup(List<double> values) {
  if (values.isEmpty) return 'n/a';
  values.sort();
  final median = values[values.length ~/ 2];
  return '${median.toStringAsFixed(2)}x';
}

String _formatBytes(int bytes) {
  const units = ['B', 'K', 'M', 'G'];
  var value = bytes.toDouble();
  var unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  if (unit == 0) return '${bytes}B';
  return '${value.toStringAsFixed(value >= 10 ? 0 : 1)}${units[unit]}';
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  return normalized.substring(normalized.lastIndexOf('/') + 1);
}

String _shortReason(Object error) {
  if (error is _DtypeException) return 'dtype';
  final text = error.toString();
  if (text == 'dtype' || text.contains('dtype')) return 'dtype';
  final status = RegExp(r'LiteRtStatus=(-?\d+)').firstMatch(text);
  if (status != null) return 'status${status.group(1)}';
  final arrow = RegExp(r'[->]\s*(-?\d+)').firstMatch(text);
  if (arrow != null) return 'status${arrow.group(1)}';
  if (text.contains('AllocateTensors')) return 'alloc';
  if (text.contains('Invoke')) return 'invoke';
  if (text.contains('xnnpack')) return 'xnnpack';
  final cleaned = text
      .replaceFirst('Bad state: ', '')
      .replaceFirst('Invalid argument(s): ', '')
      .replaceAll(RegExp(r'[^A-Za-z0-9_]+'), '_')
      .replaceAll(RegExp(r'_+'), '_')
      .replaceAll(RegExp(r'^_|_$'), '');
  if (cleaned.isEmpty) return 'error';
  return cleaned.length <= 14 ? cleaned : cleaned.substring(0, 14);
}

void ckRt(String op, int status) {
  if (status != kOk) {
    throw StateError('$op failed with LiteRtStatus=$status');
  }
}

void ckTf(String op, int status) {
  if (status != kOk) {
    throw StateError('$op failed with TfLiteStatus=$status');
  }
}
