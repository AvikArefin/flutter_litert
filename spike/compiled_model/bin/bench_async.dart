// Async / pipelined CompiledModel throughput vs sync, GPU (Metal) on macOS.
// Tests whether overlapping encode+sync of run N+1 with compute of run N raises
// effective throughput above the synchronous per-call latency.
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

const int kOk = 0, kGpu = 2, kCpu = 1, kLockWrite = 1;

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

typedef _P = Pointer<Void>;
typedef _PP = Pointer<Pointer<Void>>;
late final DynamicLibrary _rt;

void main(List<String> args) {
  final model = args.isNotEmpty
      ? args[0]
      : '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_short_range.tflite';
  _rt = DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib');
  print('model: ${model.split('/').last}');
  for (final accel in [kGpu, kCpu]) {
    final name = accel == kGpu ? 'GPU/Metal' : 'CPU';
    final sync = _bench(model, accel, depth: 1, warmup: 30, iters: 300);
    final async4 = _bench(model, accel, depth: 4, warmup: 30, iters: 300);
    final async8 = _bench(model, accel, depth: 8, warmup: 30, iters: 300);
    print('\n[$name]');
    print('  sync (depth 1)      : ${sync.toStringAsFixed(1)} µs/inf');
    print('  async pipeline (4)  : ${async4.toStringAsFixed(1)} µs/inf');
    print('  async pipeline (8)  : ${async8.toStringAsFixed(1)} µs/inf');
  }
}

double _bench(
  String path,
  int accel, {
  required int depth,
  required int warmup,
  required int iters,
}) {
  final createEnv = _rt
      .lookupFunction<
        Int32 Function(Int32, _P, _PP),
        int Function(int, _P, _PP)
      >('LiteRtCreateEnvironment');
  final createOpts = _rt.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
    'LiteRtCreateOptions',
  );
  final setAccel = _rt
      .lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
        'LiteRtSetOptionsHardwareAccelerators',
      );
  final modelFromFile = _rt
      .lookupFunction<
        Int32 Function(Pointer<Utf8>, _PP),
        int Function(Pointer<Utf8>, _PP)
      >('LiteRtCreateModelFromFile');
  final createCM = _rt
      .lookupFunction<
        Int32 Function(_P, _P, _P, _PP),
        int Function(_P, _P, _P, _PP)
      >('LiteRtCreateCompiledModel');
  final getSig = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetModelSignature');
  final numIn = _rt
      .lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureInputs');
  final numOut = _rt
      .lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureOutputs');
  final inTensorF = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureInputTensorByIndex');
  final outTensorF = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureOutputTensorByIndex');
  final rankedType = _rt
      .lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
        int Function(_P, Pointer<LiteRtRankedTensorType>)
      >('LiteRtGetRankedTensorType');
  final inReq = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelInputBufferRequirements');
  final outReq = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelOutputBufferRequirements');
  final createBuf = _rt
      .lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
        int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP)
      >('LiteRtCreateManagedTensorBufferFromRequirements');
  final lock = _rt
      .lookupFunction<
        Int32 Function(_P, _PP, Int32),
        int Function(_P, _PP, int)
      >('LiteRtLockTensorBuffer');
  final unlock = _rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'LiteRtUnlockTensorBuffer',
  );
  final reqSize = _rt
      .lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetTensorBufferRequirementsBufferSize');
  final getInLayout = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
        int Function(_P, int, int, Pointer<LiteRtLayout>)
      >('LiteRtGetCompiledModelInputTensorLayout');
  final getOutLayouts = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
        int Function(_P, int, int, Pointer<LiteRtLayout>, int)
      >('LiteRtGetCompiledModelOutputTensorLayouts');
  final runSync = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
        int Function(_P, int, int, _PP, int, _PP)
      >('LiteRtRunCompiledModel');
  final runAsync = _rt
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP, Pointer<Uint8>),
        int Function(_P, int, int, _PP, int, _PP, Pointer<Uint8>)
      >('LiteRtRunCompiledModelAsync');
  final hasEvent = _rt
      .lookupFunction<
        Int32 Function(_P, Pointer<Uint8>),
        int Function(_P, Pointer<Uint8>)
      >('LiteRtHasTensorBufferEvent');
  final getEvent = _rt
      .lookupFunction<Int32 Function(_P, _PP), int Function(_P, _PP)>(
        'LiteRtGetTensorBufferEvent',
      );
  final clearEvent = _rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'LiteRtClearTensorBufferEvent',
  );
  final waitEvent = _rt
      .lookupFunction<Int32 Function(_P, Int64), int Function(_P, int)>(
        'LiteRtWaitEvent',
      );
  void ck(String w, int s) {
    if (s != kOk) throw StateError('$w → $s');
  }

  final envOut = calloc<Pointer<Void>>();
  ck('env', createEnv(0, nullptr, envOut));
  final env = envOut.value;
  final optsOut = calloc<Pointer<Void>>();
  ck('opts', createOpts(optsOut));
  final opts = optsOut.value;
  ck('accel', setAccel(opts, accel));
  final modelOut = calloc<Pointer<Void>>();
  ck('model', modelFromFile(path.toNativeUtf8(), modelOut));
  final model = modelOut.value;
  final cmOut = calloc<Pointer<Void>>();
  ck('cm', createCM(env, model, opts, cmOut));
  final cm = cmOut.value;
  final sigOut = calloc<Pointer<Void>>();
  ck('sig', getSig(model, 0, sigOut));
  final sig = sigOut.value;
  final nInP = calloc<IntPtr>(), nOutP = calloc<IntPtr>();
  ck('nIn', numIn(sig, nInP));
  ck('nOut', numOut(sig, nOutP));
  final nIn = nInP.value, nOut = nOutP.value;

  // shared input set
  final inBufs = calloc<Pointer<Void>>(nIn);
  for (var i = 0; i < nIn; i++) {
    final tOut = calloc<Pointer<Void>>();
    ck('inT', inTensorF(sig, i, tOut));
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('rtt', rankedType(tOut.value, rtt));
    ck(
      'inLayout',
      getInLayout(cm, 0, i, (rtt.cast<Uint8>() + 4).cast<LiteRtLayout>()),
    );
    final reqOut = calloc<Pointer<Void>>();
    ck('inReq', inReq(cm, 0, i, reqOut));
    final szP = calloc<IntPtr>();
    ck('sz', reqSize(reqOut.value, szP));
    final bufOut = calloc<Pointer<Void>>();
    ck('inBuf', createBuf(env, rtt, reqOut.value, bufOut));
    inBufs[i] = bufOut.value;
    final addrOut = calloc<Pointer<Void>>();
    ck('lock', lock(bufOut.value, addrOut, kLockWrite));
    final fl = Float32List.view(
      addrOut.value.cast<Uint8>().asTypedList(szP.value).buffer,
      0,
      szP.value ~/ 4,
    );
    for (var k = 0; k < fl.length; k++) fl[k] = (k % 10) * 0.1;
    ck('unlock', unlock(bufOut.value));
  }
  // K output sets
  final outLayouts = calloc<LiteRtLayout>(nOut);
  ck('outL', getOutLayouts(cm, 0, nOut, outLayouts, 1));
  _PP makeOutSet() {
    final set = calloc<Pointer<Void>>(nOut);
    for (var j = 0; j < nOut; j++) {
      final tOut = calloc<Pointer<Void>>();
      ck('outT', outTensorF(sig, j, tOut));
      final rtt = calloc<LiteRtRankedTensorType>();
      ck('rtt', rankedType(tOut.value, rtt));
      (rtt.cast<Uint8>() + 4)
          .asTypedList(68)
          .setAll(0, (outLayouts + j).cast<Uint8>().asTypedList(68));
      final reqOut = calloc<Pointer<Void>>();
      ck('outReq', outReq(cm, 0, j, reqOut));
      final bufOut = calloc<Pointer<Void>>();
      ck('outBuf', createBuf(env, rtt, reqOut.value, bufOut));
      set[j] = bufOut.value;
    }
    return set;
  }

  final sets = List.generate(depth < 1 ? 1 : depth, (_) => makeOutSet());

  final asyncFlag = calloc<Uint8>();
  final evOut = calloc<Pointer<Void>>();
  final hasEventOut = calloc<Uint8>();

  void submit(int slot) {
    if (depth <= 1) {
      ck('run', runSync(cm, 0, nIn, inBufs, nOut, sets[0]));
    } else {
      asyncFlag.value = 0;
      ck('runAsync', runAsync(cm, 0, nIn, inBufs, nOut, sets[slot], asyncFlag));
    }
  }

  void wait(int slot) {
    if (depth <= 1) return; // sync already complete
    for (var j = 0; j < nOut; j++) {
      ck('hasEvent', hasEvent(sets[slot][j], hasEventOut));
      if (hasEventOut.value == 0) continue;
      ck('getEvent', getEvent(sets[slot][j], evOut));
      if (evOut.value != nullptr) {
        // GetTensorBufferEvent is a borrowed handle; ClearTensorBufferEvent
        // destroys and detaches the owned event from the buffer.
        ck('waitEvent', waitEvent(evOut.value, -1));
        ck('clearEvent', clearEvent(sets[slot][j]));
      }
    }
  }

  // warmup
  for (var i = 0; i < warmup; i++) {
    submit(i % sets.length);
    wait(i % sets.length);
  }

  final sw = Stopwatch()..start();
  if (depth <= 1) {
    for (var i = 0; i < iters; i++) {
      submit(0);
    }
  } else {
    final k = sets.length;
    for (var i = 0; i < iters; i++) {
      final slot = i % k;
      if (i >= k)
        wait(slot); // ensure this slot's prior run finished before reuse
      submit(slot);
    }
    for (var s = 0; s < k; s++) wait(s); // drain
  }
  sw.stop();
  return sw.elapsedMicroseconds / iters;
}
