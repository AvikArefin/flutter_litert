import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

const String kDefaultModel =
    '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_back.tflite';
const int kOk = 0;
const int kTimeout = 7;
const int kCpu = 1;
const int kGpu = 2;
const int kLockWrite = 1;
const int kRankedTensorTypeLayoutOffset = 4;
const int kLiteRtLayoutSize = 68;

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
      modelFromFile = rt.lookupFunction<
        Int32 Function(Pointer<Utf8>, _PP),
        int Function(Pointer<Utf8>, _PP)
      >('LiteRtCreateModelFromFile'),
      destroyModel = rt.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyModel',
      ),
      createCM = rt.lookupFunction<
        Int32 Function(_P, _P, _P, _PP),
        int Function(_P, _P, _P, _PP)
      >('LiteRtCreateCompiledModel'),
      destroyCM = rt.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyCompiledModel',
      ),
      getSig = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetModelSignature'),
      numIn = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureInputs'),
      numOut = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureOutputs'),
      inTensor = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureInputTensorByIndex'),
      outTensor = rt.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureOutputTensorByIndex'),
      rankedType = rt.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
        int Function(_P, Pointer<LiteRtRankedTensorType>)
      >('LiteRtGetRankedTensorType'),
      inReq = rt.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelInputBufferRequirements'),
      outReq = rt.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelOutputBufferRequirements'),
      reqSize = rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetTensorBufferRequirementsBufferSize'),
      createBuf = rt.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
        int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP)
      >('LiteRtCreateManagedTensorBufferFromRequirements'),
      destroyBuf = rt.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyTensorBuffer',
      ),
      lock = rt.lookupFunction<
        Int32 Function(_P, _PP, Int32),
        int Function(_P, _PP, int)
      >('LiteRtLockTensorBuffer'),
      unlock = rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
        'LiteRtUnlockTensorBuffer',
      ),
      getInLayout = rt.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
        int Function(_P, int, int, Pointer<LiteRtLayout>)
      >('LiteRtGetCompiledModelInputTensorLayout'),
      getOutLayouts = rt.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
        int Function(_P, int, int, Pointer<LiteRtLayout>, int)
      >('LiteRtGetCompiledModelOutputTensorLayouts'),
      runSync = rt.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
        int Function(_P, int, int, _PP, int, _PP)
      >('LiteRtRunCompiledModel'),
      runAsync = rt.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP, Pointer<Uint8>),
        int Function(_P, int, int, _PP, int, _PP, Pointer<Uint8>)
      >('LiteRtRunCompiledModelAsync'),
      hasEvent = rt.lookupFunction<
        Int32 Function(_P, Pointer<Uint8>),
        int Function(_P, Pointer<Uint8>)
      >('LiteRtHasTensorBufferEvent'),
      getEvent = rt
          .lookupFunction<Int32 Function(_P, _PP), int Function(_P, _PP)>(
            'LiteRtGetTensorBufferEvent',
          ),
      clearEvent = rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
        'LiteRtClearTensorBufferEvent',
      ),
      waitEvent = rt
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

final class _AsyncSingleResult {
  const _AsyncSingleResult(this.medianUs, this.asyncOut);

  final int medianUs;
  final int asyncOut;
}

final class _Session {
  _Session(this.api, String path, this.accel) {
    final envOut = calloc<Pointer<Void>>();
    final optsOut = calloc<Pointer<Void>>();
    final modelOut = calloc<Pointer<Void>>();
    final cmOut = calloc<Pointer<Void>>();
    final sigOut = calloc<Pointer<Void>>();
    final nInP = calloc<IntPtr>();
    final nOutP = calloc<IntPtr>();
    final pathPtr = path.toNativeUtf8();
    try {
      ck('LiteRtCreateEnvironment', api.createEnv(0, nullptr, envOut));
      env = envOut.value;
      ck('LiteRtCreateOptions', api.createOpts(optsOut));
      opts = optsOut.value;
      ck('LiteRtSetOptionsHardwareAccelerators', api.setAccel(opts, accel));
      ck('LiteRtCreateModelFromFile', api.modelFromFile(pathPtr, modelOut));
      model = modelOut.value;
      ck('LiteRtCreateCompiledModel', api.createCM(env, model, opts, cmOut));
      cm = cmOut.value;
      ck('LiteRtGetModelSignature', api.getSig(model, 0, sigOut));
      sig = sigOut.value;
      ck('LiteRtGetNumSignatureInputs', api.numIn(sig, nInP));
      ck('LiteRtGetNumSignatureOutputs', api.numOut(sig, nOutP));
      nIn = nInP.value;
      nOut = nOutP.value;

      inBufs = calloc<Pointer<Void>>(nIn);
      inputByteSizes = List<int>.filled(nIn, 0);
      for (var i = 0; i < nIn; i++) {
        inBufs[i] = _createInputBuffer(i);
        _writeInput(i);
      }

      outLayouts = calloc<LiteRtLayout>(nOut);
      ck(
        'LiteRtGetCompiledModelOutputTensorLayouts',
        api.getOutLayouts(cm, 0, nOut, outLayouts, 1),
      );
      singleOutSet = makeOutSet();
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

  final _Api api;
  final int accel;
  late final _P env;
  late final _P opts;
  late final _P model;
  late final _P cm;
  late final _P sig;
  late final int nIn;
  late final int nOut;
  late final _PP inBufs;
  late final List<int> inputByteSizes;
  late final Pointer<LiteRtLayout> outLayouts;
  late final _PP singleOutSet;

  int measureSyncMedian({required int warmup, required int iters}) {
    for (var i = 0; i < warmup; i++) {
      _runSync(singleOutSet);
    }

    final samples = <int>[];
    for (var i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      _runSync(singleOutSet);
      sw.stop();
      samples.add(sw.elapsedMicroseconds);
    }
    return _median(samples);
  }

  _AsyncSingleResult measureAsyncSingleMedian({
    required int warmup,
    required int iters,
  }) {
    var observedAsyncOut = 0;
    final asyncOut = calloc<Uint8>();
    try {
      for (var i = 0; i < warmup; i++) {
        final flag = _runAsyncSingle(singleOutSet, asyncOut);
        observedAsyncOut |= flag;
      }

      final samples = <int>[];
      for (var i = 0; i < iters; i++) {
        asyncOut.value = 0;
        final sw = Stopwatch()..start();
        final flag = _runAsyncSingle(singleOutSet, asyncOut);
        sw.stop();
        observedAsyncOut |= flag;
        samples.add(sw.elapsedMicroseconds);
      }
      return _AsyncSingleResult(_median(samples), observedAsyncOut);
    } finally {
      calloc.free(asyncOut);
    }
  }

  double measurePipeline({
    required int depth,
    required int warmup,
    required int iters,
  }) {
    final sets = List<_PP>.generate(depth, (_) => makeOutSet());
    final asyncOut = calloc<Uint8>();
    try {
      void submit(int slot) {
        asyncOut.value = 0;
        ck(
          'LiteRtRunCompiledModelAsync',
          api.runAsync(cm, 0, nIn, inBufs, nOut, sets[slot], asyncOut),
        );
      }

      for (var i = 0; i < warmup; i++) {
        final slot = i % depth;
        submit(slot);
        _waitOutputSetBlocking(sets[slot]);
      }

      final sw = Stopwatch()..start();
      for (var i = 0; i < iters; i++) {
        final slot = i % depth;
        if (i >= depth) {
          _waitOutputSetBlocking(sets[slot]);
        }
        submit(slot);
      }
      for (var slot = 0; slot < depth; slot++) {
        _waitOutputSetBlocking(sets[slot]);
      }
      sw.stop();
      return sw.elapsedMicroseconds / iters;
    } finally {
      calloc.free(asyncOut);
      for (final set in sets) {
        destroyOutSet(set);
      }
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
      if (set[i] != nullptr) {
        api.destroyBuf(set[i]);
      }
    }
    calloc.free(set);
  }

  void close() {
    destroyOutSet(singleOutSet);
    calloc.free(outLayouts);
    for (var i = 0; i < nIn; i++) {
      if (inBufs[i] != nullptr) {
        api.destroyBuf(inBufs[i]);
      }
    }
    calloc.free(inBufs);
    api.destroyCM(cm);
    api.destroyModel(model);
    api.destroyOpts(opts);
    api.destroyEnv(env);
  }

  _P _createInputBuffer(int index) {
    final tensorOut = calloc<Pointer<Void>>();
    final tensorType = calloc<LiteRtRankedTensorType>();
    final reqOut = calloc<Pointer<Void>>();
    final sizeOut = calloc<IntPtr>();
    final bufOut = calloc<Pointer<Void>>();
    try {
      ck(
        'LiteRtGetSignatureInputTensorByIndex',
        api.inTensor(sig, index, tensorOut),
      );
      ck(
        'LiteRtGetRankedTensorType',
        api.rankedType(tensorOut.value, tensorType),
      );
      ck(
        'LiteRtGetCompiledModelInputTensorLayout',
        api.getInLayout(
          cm,
          0,
          index,
          (tensorType.cast<Uint8>() + kRankedTensorTypeLayoutOffset)
              .cast<LiteRtLayout>(),
        ),
      );
      ck(
        'LiteRtGetCompiledModelInputBufferRequirements',
        api.inReq(cm, 0, index, reqOut),
      );
      ck(
        'LiteRtGetTensorBufferRequirementsBufferSize',
        api.reqSize(reqOut.value, sizeOut),
      );
      inputByteSizes[index] = sizeOut.value;
      ck(
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
      ck(
        'LiteRtGetSignatureOutputTensorByIndex',
        api.outTensor(sig, index, tensorOut),
      );
      ck(
        'LiteRtGetRankedTensorType',
        api.rankedType(tensorOut.value, tensorType),
      );
      (tensorType.cast<Uint8>() + kRankedTensorTypeLayoutOffset)
          .asTypedList(kLiteRtLayoutSize)
          .setAll(
            0,
            (outLayouts + index).cast<Uint8>().asTypedList(kLiteRtLayoutSize),
          );
      ck(
        'LiteRtGetCompiledModelOutputBufferRequirements',
        api.outReq(cm, 0, index, reqOut),
      );
      ck(
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
    final byteSize = inputByteSizes[index];
    try {
      ck(
        'LiteRtLockTensorBuffer input',
        api.lock(inBufs[index], addrOut, kLockWrite),
      );
      final floats = Float32List.view(
        addrOut.value.cast<Uint8>().asTypedList(byteSize).buffer,
        addrOut.value.cast<Uint8>().asTypedList(byteSize).offsetInBytes,
        byteSize ~/ sizeOf<Float>(),
      );
      for (var i = 0; i < floats.length; i++) {
        floats[i] = (i % 17) * 0.03125;
      }
    } finally {
      ck('LiteRtUnlockTensorBuffer input', api.unlock(inBufs[index]));
      calloc.free(addrOut);
    }
  }

  void _runSync(_PP outSet) {
    ck('LiteRtRunCompiledModel', api.runSync(cm, 0, nIn, inBufs, nOut, outSet));
  }

  int _runAsyncSingle(_PP outSet, Pointer<Uint8> asyncOut) {
    asyncOut.value = 0;
    ck(
      'LiteRtRunCompiledModelAsync',
      api.runAsync(cm, 0, nIn, inBufs, nOut, outSet, asyncOut),
    );
    if (asyncOut.value != 0) {
      _pollOutputSetSignaled(outSet);
    }
    return asyncOut.value;
  }

  void _pollOutputSetSignaled(_PP set) {
    final hasEvent = calloc<Uint8>();
    final eventOut = calloc<Pointer<Void>>();
    final pending = List<bool>.filled(nOut, false);
    var pendingCount = 0;

    try {
      for (var i = 0; i < nOut; i++) {
        hasEvent.value = 0;
        ck('LiteRtHasTensorBufferEvent', api.hasEvent(set[i], hasEvent));
        if (hasEvent.value == 0) continue;
        pending[i] = true;
        pendingCount++;
      }

      while (pendingCount > 0) {
        for (var i = 0; i < nOut; i++) {
          if (!pending[i]) continue;

          hasEvent.value = 0;
          ck('LiteRtHasTensorBufferEvent', api.hasEvent(set[i], hasEvent));
          if (hasEvent.value == 0) {
            pending[i] = false;
            pendingCount--;
            continue;
          }

          eventOut.value = nullptr;
          ck('LiteRtGetTensorBufferEvent', api.getEvent(set[i], eventOut));
          if (eventOut.value == nullptr) {
            pending[i] = false;
            pendingCount--;
            continue;
          }

          final waitStatus = api.waitEvent(eventOut.value, 0);
          if (waitStatus == kTimeout) continue;
          ck('LiteRtWaitEvent', waitStatus);

          ck('LiteRtClearTensorBufferEvent', api.clearEvent(set[i]));
          pending[i] = false;
          pendingCount--;
        }
      }
    } finally {
      calloc.free(eventOut);
      calloc.free(hasEvent);
    }
  }

  void _waitOutputSetBlocking(_PP set) {
    final hasEvent = calloc<Uint8>();
    final eventOut = calloc<Pointer<Void>>();
    try {
      for (var i = 0; i < nOut; i++) {
        hasEvent.value = 0;
        ck('LiteRtHasTensorBufferEvent', api.hasEvent(set[i], hasEvent));
        if (hasEvent.value == 0) continue;

        eventOut.value = nullptr;
        ck('LiteRtGetTensorBufferEvent', api.getEvent(set[i], eventOut));
        if (eventOut.value == nullptr) continue;

        ck('LiteRtWaitEvent', api.waitEvent(eventOut.value, -1));
        ck('LiteRtClearTensorBufferEvent', api.clearEvent(set[i]));
      }
    } finally {
      calloc.free(eventOut);
      calloc.free(hasEvent);
    }
  }
}

void main(List<String> args) {
  final modelPath = args.isEmpty ? kDefaultModel : args.first;
  if (!File(modelPath).existsSync()) {
    throw ArgumentError('Model file not found: $modelPath');
  }
  if (sizeOf<LiteRtLayout>() != kLiteRtLayoutSize ||
      sizeOf<LiteRtRankedTensorType>() != 72) {
    throw StateError(
      'Unexpected LiteRT struct sizes: layout=${sizeOf<LiteRtLayout>()}, '
      'ranked=${sizeOf<LiteRtRankedTensorType>()}',
    );
  }

  final api = _Api(DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib'));
  print('model: ${modelPath.split('/').last}');
  print(
    'accelerator,sync_median_us,async_single_median_us,asyncOut,pipeline8_us_per_inf',
  );

  for (final accel in const [kGpu, kCpu]) {
    final session = _Session(api, modelPath, accel);
    try {
      final sync = session.measureSyncMedian(warmup: 30, iters: 101);
      final asyncSingle = session.measureAsyncSingleMedian(
        warmup: 30,
        iters: 101,
      );
      final pipeline = session.measurePipeline(
        depth: 8,
        warmup: 30,
        iters: 300,
      );
      print(
        '${_accelName(accel)},$sync,${asyncSingle.medianUs},'
        '${asyncSingle.asyncOut},${pipeline.toStringAsFixed(1)}',
      );
    } finally {
      session.close();
    }
  }
}

String _accelName(int accel) => accel == kGpu ? 'GPU' : 'CPU';

int _median(List<int> samples) {
  final sorted = [...samples]..sort();
  return sorted[sorted.length ~/ 2];
}

void ck(String op, int status) {
  if (status != kOk) {
    throw StateError('$op failed with LiteRtStatus=$status');
  }
}
