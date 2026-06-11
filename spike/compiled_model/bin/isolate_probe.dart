import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

const List<String> kDefaultModels = [
  '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_back.tflite',
  '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/yolov8n_float32.tflite',
];

const String kLiteRtPath = '/tmp/cm_spike/libLiteRt.dylib';
const int kOk = 0;
const int kTimeout = 7;
const int kCpu = 1;
const int kGpu = 2;
const int kLockRead = 0;
const int kLockWrite = 1;
const int kRankedTensorTypeLayoutOffset = 4;
const int kLiteRtLayoutSize = 68;
const int kLiteRtRankedTensorTypeSize = 72;
const int kWarmupIters = 30;
const int kTimedIters = 200;

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
      : createEnv = rt.lookupFunction<Int32 Function(Int32, _P, _PP),
            int Function(int, _P, _PP)>('LiteRtCreateEnvironment'),
        destroyEnv = rt.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyEnvironment',
        ),
        createOpts = rt.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
          'LiteRtCreateOptions',
        ),
        destroyOpts = rt.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyOptions',
        ),
        setAccel =
            rt.lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
          'LiteRtSetOptionsHardwareAccelerators',
        ),
        modelFromFile = rt.lookupFunction<Int32 Function(Pointer<Utf8>, _PP),
            int Function(Pointer<Utf8>, _PP)>('LiteRtCreateModelFromFile'),
        destroyModel = rt.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyModel',
        ),
        createCM = rt.lookupFunction<Int32 Function(_P, _P, _P, _PP),
            int Function(_P, _P, _P, _PP)>('LiteRtCreateCompiledModel'),
        destroyCM = rt.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyCompiledModel',
        ),
        getSig = rt.lookupFunction<Int32 Function(_P, IntPtr, _PP),
            int Function(_P, int, _PP)>('LiteRtGetModelSignature'),
        numIn = rt.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
            int Function(_P, Pointer<IntPtr>)>('LiteRtGetNumSignatureInputs'),
        numOut = rt.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
            int Function(_P, Pointer<IntPtr>)>('LiteRtGetNumSignatureOutputs'),
        inTensor = rt.lookupFunction<Int32 Function(_P, IntPtr, _PP),
            int Function(_P, int, _PP)>('LiteRtGetSignatureInputTensorByIndex'),
        outTensor = rt.lookupFunction<
            Int32 Function(_P, IntPtr, _PP),
            int Function(
                _P, int, _PP)>('LiteRtGetSignatureOutputTensorByIndex'),
        rankedType = rt.lookupFunction<
            Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
            int Function(_P,
                Pointer<LiteRtRankedTensorType>)>('LiteRtGetRankedTensorType'),
        inReq = rt.lookupFunction<
            Int32 Function(_P, IntPtr, IntPtr, _PP),
            int Function(_P, int, int,
                _PP)>('LiteRtGetCompiledModelInputBufferRequirements'),
        outReq = rt.lookupFunction<
            Int32 Function(_P, IntPtr, IntPtr, _PP),
            int Function(_P, int, int,
                _PP)>('LiteRtGetCompiledModelOutputBufferRequirements'),
        reqSize = rt.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
                int Function(_P, Pointer<IntPtr>)>(
            'LiteRtGetTensorBufferRequirementsBufferSize'),
        createBuf = rt.lookupFunction<
            Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
            int Function(_P, Pointer<LiteRtRankedTensorType>, _P,
                _PP)>('LiteRtCreateManagedTensorBufferFromRequirements'),
        destroyBuf = rt.lookupFunction<Void Function(_P), void Function(_P)>(
          'LiteRtDestroyTensorBuffer',
        ),
        lock = rt.lookupFunction<Int32 Function(_P, _PP, Int32),
            int Function(_P, _PP, int)>('LiteRtLockTensorBuffer'),
        unlock = rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'LiteRtUnlockTensorBuffer',
        ),
        getInLayout = rt.lookupFunction<
                Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
                int Function(_P, int, int, Pointer<LiteRtLayout>)>(
            'LiteRtGetCompiledModelInputTensorLayout'),
        getOutLayouts = rt.lookupFunction<
            Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
            int Function(_P, int, int, Pointer<LiteRtLayout>,
                int)>('LiteRtGetCompiledModelOutputTensorLayouts'),
        runAsync = rt.lookupFunction<
                Int32 Function(
                  _P,
                  IntPtr,
                  IntPtr,
                  _PP,
                  IntPtr,
                  _PP,
                  Pointer<Uint8>,
                ),
                int Function(_P, int, int, _PP, int, _PP, Pointer<Uint8>)>(
            'LiteRtRunCompiledModelAsync'),
        hasEvent = rt.lookupFunction<Int32 Function(_P, Pointer<Uint8>),
            int Function(_P, Pointer<Uint8>)>('LiteRtHasTensorBufferEvent'),
        getEvent =
            rt.lookupFunction<Int32 Function(_P, _PP), int Function(_P, _PP)>(
          'LiteRtGetTensorBufferEvent',
        ),
        clearEvent = rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
          'LiteRtClearTensorBufferEvent',
        ),
        waitEvent =
            rt.lookupFunction<Int32 Function(_P, Int64), int Function(_P, int)>(
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
  final int Function(_P, int, int, _PP, int, _PP, Pointer<Uint8>) runAsync;
  final int Function(_P, Pointer<Uint8>) hasEvent;
  final int Function(_P, _PP) getEvent;
  final int Function(_P) clearEvent;
  final int Function(_P, int) waitEvent;
}

final class _RunResult {
  const _RunResult(this.output, this.asyncOut);

  final Float32List output;
  final int asyncOut;
}

final class _DirectMeasure {
  const _DirectMeasure(this.medianUs, this.asyncOutObserved, this.checksum);

  final int medianUs;
  final int asyncOutObserved;
  final double checksum;
}

final class _IsolateMeasure {
  const _IsolateMeasure(this.medianUs, this.checksum);

  final int medianUs;
  final double checksum;
}

final class _Worker {
  _Worker(this.isolate, this.receivePort, this.iterator, this.sendPort);

  final Isolate isolate;
  final ReceivePort receivePort;
  final StreamIterator<dynamic> iterator;
  final SendPort sendPort;
}

final class _WorkerStats {
  const _WorkerStats(this.asyncOutObserved, this.gpuRan);

  final int asyncOutObserved;
  final bool gpuRan;
}

final class _Row {
  const _Row({
    required this.accel,
    required this.direct,
    required this.copy,
    required this.transferable,
    required this.workerAsyncOut,
    required this.workerGpuRan,
  });

  final int accel;
  final int direct;
  final int copy;
  final int transferable;
  final int workerAsyncOut;
  final bool workerGpuRan;

  int get copyOverhead => copy - direct;
  int get transferableOverhead => transferable - direct;
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
      if (nIn != 1) {
        throw StateError('Expected one model input, got $nIn');
      }

      inBufs = calloc<Pointer<Void>>(nIn);
      inputByteSizes = List<int>.filled(nIn, 0);
      for (var i = 0; i < nIn; i++) {
        inBufs[i] = _createInputBuffer(i);
      }

      outLayouts = calloc<LiteRtLayout>(nOut);
      outputByteSizes = List<int>.filled(nOut, 0);
      ck(
        'LiteRtGetCompiledModelOutputTensorLayouts',
        api.getOutLayouts(cm, 0, nOut, outLayouts, 1),
      );
      singleOutSet = _makeOutSet();
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
  late final List<int> outputByteSizes;
  late final _PP singleOutSet;
  var asyncOutObserved = 0;

  int get inputFloats => inputByteSizes[0] ~/ sizeOf<Float>();

  int get outputFloats {
    var total = 0;
    for (final bytes in outputByteSizes) {
      total += bytes ~/ sizeOf<Float>();
    }
    return total;
  }

  _RunResult runAsync(Float32List input) {
    _writeInput(0, input);
    final asyncOut = calloc<Uint8>();
    try {
      asyncOut.value = 0;
      ck(
        'LiteRtRunCompiledModelAsync',
        api.runAsync(cm, 0, nIn, inBufs, nOut, singleOutSet, asyncOut),
      );
      if (asyncOut.value != 0) {
        _pollOutputSetSignaled(singleOutSet);
      }
      asyncOutObserved |= asyncOut.value;
      return _RunResult(_readOutputs(singleOutSet), asyncOut.value);
    } finally {
      calloc.free(asyncOut);
    }
  }

  void close() {
    _destroyOutSet(singleOutSet);
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

  _PP _makeOutSet() {
    final set = calloc<Pointer<Void>>(nOut);
    for (var i = 0; i < nOut; i++) {
      set[i] = _createOutputBuffer(i);
    }
    return set;
  }

  _P _createOutputBuffer(int index) {
    final tensorOut = calloc<Pointer<Void>>();
    final tensorType = calloc<LiteRtRankedTensorType>();
    final reqOut = calloc<Pointer<Void>>();
    final sizeOut = calloc<IntPtr>();
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
        'LiteRtGetTensorBufferRequirementsBufferSize',
        api.reqSize(reqOut.value, sizeOut),
      );
      outputByteSizes[index] = sizeOut.value;
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

  void _destroyOutSet(_PP set) {
    for (var i = 0; i < nOut; i++) {
      if (set[i] != nullptr) {
        api.destroyBuf(set[i]);
      }
    }
    calloc.free(set);
  }

  void _writeInput(int index, Float32List input) {
    final expected = inputByteSizes[index] ~/ sizeOf<Float>();
    if (input.length != expected) {
      throw ArgumentError(
        'Input length ${input.length} does not match expected $expected',
      );
    }

    final addrOut = calloc<Pointer<Void>>();
    try {
      ck(
        'LiteRtLockTensorBuffer input',
        api.lock(inBufs[index], addrOut, kLockWrite),
      );
      final bytes = addrOut.value.cast<Uint8>().asTypedList(
            inputByteSizes[index],
          );
      final floats = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        expected,
      );
      floats.setAll(0, input);
    } finally {
      ck('LiteRtUnlockTensorBuffer input', api.unlock(inBufs[index]));
      calloc.free(addrOut);
    }
  }

  Float32List _readOutputs(_PP set) {
    final result = Float32List(outputFloats);
    final addrOut = calloc<Pointer<Void>>();
    var offset = 0;
    try {
      for (var i = 0; i < nOut; i++) {
        ck(
          'LiteRtLockTensorBuffer output',
          api.lock(set[i], addrOut, kLockRead),
        );
        final byteSize = outputByteSizes[i];
        final bytes = addrOut.value.cast<Uint8>().asTypedList(byteSize);
        final floats = Float32List.view(
          bytes.buffer,
          bytes.offsetInBytes,
          byteSize ~/ sizeOf<Float>(),
        );
        result.setRange(offset, offset + floats.length, floats);
        offset += floats.length;
        ck('LiteRtUnlockTensorBuffer output', api.unlock(set[i]));
      }
      return result;
    } finally {
      calloc.free(addrOut);
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
}

Future<void> main(List<String> args) async {
  _checkStructSizes();
  final models = args.isEmpty ? kDefaultModels : args;
  for (var i = 0; i < models.length; i++) {
    final modelPath = models[i];
    if (!File(modelPath).existsSync()) {
      throw ArgumentError('Model file not found: $modelPath');
    }
    if (i > 0) {
      print('');
    }
    await _runModel(modelPath);
  }
}

Future<void> _runModel(String modelPath) async {
  final rows = <_Row>[];
  print('model: ${modelPath.split('/').last}');
  print('warmup=$kWarmupIters timed=$kTimedIters');

  for (final accel in const [kGpu, kCpu]) {
    final api = _Api(DynamicLibrary.open(kLiteRtPath));
    final directSession = _Session(api, modelPath, accel);
    late final Float32List input;
    try {
      input = _makeInput(directSession.inputFloats);
      final direct = _measureDirect(directSession, input);
      final worker = await _spawnWorker(modelPath, accel);
      try {
        await _assertWorkerShape(
          worker,
          directSession.inputFloats,
          directSession.outputFloats,
        );
        final copy = await _measureIsolateCopy(worker, input);
        final transferable = await _measureIsolateTransferable(worker, input);
        final stats = await _closeWorker(worker);
        rows.add(
          _Row(
            accel: accel,
            direct: direct.medianUs,
            copy: copy.medianUs,
            transferable: transferable.medianUs,
            workerAsyncOut: stats.asyncOutObserved,
            workerGpuRan: stats.gpuRan,
          ),
        );
      } catch (_) {
        worker.isolate.kill(priority: Isolate.immediate);
        worker.receivePort.close();
        await worker.iterator.cancel();
        rethrow;
      }
    } finally {
      directSession.close();
    }
  }

  final gpuRow = rows.firstWhere((row) => row.accel == kGpu);
  print(
    'worker_gpu_ran=${gpuRow.workerGpuRan} '
    'worker_gpu_asyncOut=${gpuRow.workerAsyncOut}',
  );
  print(
    'accelerator,direct_async_single_us,isolate_roundtrip_send_copy_us,'
    'isolate_roundtrip_transferable_us,copy_overhead_us,'
    'transferable_overhead_us,worker_asyncOut',
  );
  for (final row in rows) {
    print(
      '${_accelName(row.accel)},${row.direct},${row.copy},'
      '${row.transferable},${row.copyOverhead},'
      '${row.transferableOverhead},${row.workerAsyncOut}',
    );
  }
}

_DirectMeasure _measureDirect(_Session session, Float32List input) {
  var asyncOutObserved = 0;
  var checksum = 0.0;
  for (var i = 0; i < kWarmupIters; i++) {
    final result = session.runAsync(input);
    asyncOutObserved |= result.asyncOut;
    checksum += _touch(result.output);
  }

  final samples = <int>[];
  for (var i = 0; i < kTimedIters; i++) {
    final sw = Stopwatch()..start();
    final result = session.runAsync(input);
    checksum += _touch(result.output);
    sw.stop();
    asyncOutObserved |= result.asyncOut;
    samples.add(sw.elapsedMicroseconds);
  }
  return _DirectMeasure(_median(samples), asyncOutObserved, checksum);
}

Future<_IsolateMeasure> _measureIsolateCopy(
  _Worker worker,
  Float32List input,
) async {
  var checksum = 0.0;
  for (var i = 0; i < kWarmupIters; i++) {
    worker.sendPort.send(input);
    final output = await _nextCopyOutput(worker);
    checksum += _touch(output);
  }

  final samples = <int>[];
  for (var i = 0; i < kTimedIters; i++) {
    final sw = Stopwatch()..start();
    worker.sendPort.send(input);
    final output = await _nextCopyOutput(worker);
    checksum += _touch(output);
    sw.stop();
    samples.add(sw.elapsedMicroseconds);
  }
  return _IsolateMeasure(_median(samples), checksum);
}

Future<_IsolateMeasure> _measureIsolateTransferable(
  _Worker worker,
  Float32List input,
) async {
  var checksum = 0.0;
  for (var i = 0; i < kWarmupIters; i++) {
    worker.sendPort.send(_transferableFromFloats(input));
    final output = await _nextTransferableOutput(worker);
    checksum += _touch(output);
  }

  final samples = <int>[];
  for (var i = 0; i < kTimedIters; i++) {
    final sw = Stopwatch()..start();
    worker.sendPort.send(_transferableFromFloats(input));
    final output = await _nextTransferableOutput(worker);
    checksum += _touch(output);
    sw.stop();
    samples.add(sw.elapsedMicroseconds);
  }
  return _IsolateMeasure(_median(samples), checksum);
}

Future<_Worker> _spawnWorker(String modelPath, int accel) async {
  final receivePort = ReceivePort();
  final iterator = StreamIterator<dynamic>(receivePort);
  final isolate = await Isolate.spawn(_workerMain, <Object?>[
    modelPath,
    accel,
    receivePort.sendPort,
  ]);
  if (!await iterator.moveNext()) {
    receivePort.close();
    throw StateError('Worker exited before ready');
  }
  final ready = iterator.current;
  if (ready is Map && ready['type'] == 'error') {
    receivePort.close();
    throw StateError(
      'Worker init failed: ${ready['error']}\n${ready['stack']}',
    );
  }
  if (ready is! Map || ready['type'] != 'ready') {
    receivePort.close();
    throw StateError('Unexpected worker ready message: $ready');
  }
  return _Worker(isolate, receivePort, iterator, ready['sendPort'] as SendPort);
}

Future<void> _assertWorkerShape(
  _Worker worker,
  int inputFloats,
  int outputFloats,
) async {
  final current = worker.iterator.current as Map;
  final workerInputFloats = current['inputFloats'] as int;
  final workerOutputFloats = current['outputFloats'] as int;
  if (workerInputFloats != inputFloats || workerOutputFloats != outputFloats) {
    throw StateError(
      'Worker tensor shape mismatch: '
      'input $workerInputFloats/$inputFloats, '
      'output $workerOutputFloats/$outputFloats',
    );
  }
}

Future<_WorkerStats> _closeWorker(_Worker worker) async {
  worker.sendPort.send('close');
  if (!await worker.iterator.moveNext()) {
    worker.receivePort.close();
    return const _WorkerStats(0, false);
  }
  final msg = worker.iterator.current;
  await worker.iterator.cancel();
  worker.receivePort.close();
  if (msg is! Map || msg['type'] != 'closed') {
    throw StateError('Unexpected worker close message: $msg');
  }
  return _WorkerStats(msg['asyncOutObserved'] as int, msg['gpuRan'] as bool);
}

Future<Float32List> _nextCopyOutput(_Worker worker) async {
  if (!await worker.iterator.moveNext()) {
    throw StateError('Worker exited while waiting for copy output');
  }
  final msg = worker.iterator.current;
  if (msg is Map && msg['type'] == 'error') {
    throw StateError('Worker run failed: ${msg['error']}\n${msg['stack']}');
  }
  if (msg is! Float32List) {
    throw StateError('Expected Float32List output, got ${msg.runtimeType}');
  }
  return msg;
}

Future<Float32List> _nextTransferableOutput(_Worker worker) async {
  if (!await worker.iterator.moveNext()) {
    throw StateError('Worker exited while waiting for transferable output');
  }
  final msg = worker.iterator.current;
  if (msg is Map && msg['type'] == 'error') {
    throw StateError('Worker run failed: ${msg['error']}\n${msg['stack']}');
  }
  if (msg is! TransferableTypedData) {
    throw StateError(
      'Expected TransferableTypedData output, got ${msg.runtimeType}',
    );
  }
  return _floatsFromTransferable(msg);
}

void _workerMain(List<Object?> args) {
  final modelPath = args[0]! as String;
  final accel = args[1]! as int;
  final reply = args[2]! as SendPort;
  final commandPort = ReceivePort();
  _Session? session;
  try {
    _checkStructSizes();
    final api = _Api(DynamicLibrary.open(kLiteRtPath));
    session = _Session(api, modelPath, accel);
    reply.send(<String, Object?>{
      'type': 'ready',
      'sendPort': commandPort.sendPort,
      'inputFloats': session.inputFloats,
      'outputFloats': session.outputFloats,
      'gpuRan': accel == kGpu,
    });

    commandPort.listen((message) {
      try {
        if (message == 'close') {
          final asyncOutObserved = session!.asyncOutObserved;
          session!.close();
          session = null;
          commandPort.close();
          reply.send(<String, Object?>{
            'type': 'closed',
            'asyncOutObserved': asyncOutObserved,
            'gpuRan': accel == kGpu,
          });
          return;
        }

        if (message is Float32List) {
          final result = session!.runAsync(message);
          reply.send(result.output);
          return;
        }

        if (message is TransferableTypedData) {
          final input = _floatsFromTransferable(message);
          final result = session!.runAsync(input);
          reply.send(_transferableFromFloats(result.output));
          return;
        }

        throw ArgumentError(
          'Unexpected worker message: ${message.runtimeType}',
        );
      } catch (error, stack) {
        reply.send(<String, Object?>{
          'type': 'error',
          'error': '$error',
          'stack': '$stack',
        });
      }
    });
  } catch (error, stack) {
    session?.close();
    commandPort.close();
    reply.send(<String, Object?>{
      'type': 'error',
      'error': '$error',
      'stack': '$stack',
    });
  }
}

Float32List _makeInput(int length) {
  final input = Float32List(length);
  for (var i = 0; i < input.length; i++) {
    input[i] = (i % 17) * 0.03125;
  }
  return input;
}

TransferableTypedData _transferableFromFloats(Float32List floats) {
  return TransferableTypedData.fromList([
    floats.buffer.asUint8List(floats.offsetInBytes, floats.lengthInBytes),
  ]);
}

Float32List _floatsFromTransferable(TransferableTypedData data) {
  final bytes = data.materialize().asUint8List();
  return Float32List.view(
    bytes.buffer,
    bytes.offsetInBytes,
    bytes.lengthInBytes ~/ sizeOf<Float>(),
  );
}

double _touch(Float32List values) {
  if (values.isEmpty) return 0;
  return values.first + values.last + values.length;
}

String _accelName(int accel) => accel == kGpu ? 'GPU' : 'CPU';

int _median(List<int> samples) {
  final sorted = [...samples]..sort();
  return sorted[sorted.length ~/ 2];
}

void _checkStructSizes() {
  if (sizeOf<LiteRtLayout>() != kLiteRtLayoutSize ||
      sizeOf<LiteRtRankedTensorType>() != kLiteRtRankedTensorTypeSize) {
    throw StateError(
      'Unexpected LiteRT struct sizes: layout=${sizeOf<LiteRtLayout>()}, '
      'ranked=${sizeOf<LiteRtRankedTensorType>()}',
    );
  }
}

void ck(String op, int status) {
  if (status != kOk) {
    throw StateError('$op failed with LiteRtStatus=$status');
  }
}
