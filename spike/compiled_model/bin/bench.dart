// In-process microbenchmark: classic Interpreter vs LiteRT Next CompiledModel,
// same model + same input, warm loop, on macOS. Pure Dart FFI to both runtimes.
//   Interpreter  -> libtensorflowlite_c-mac.dylib (TfLiteInterpreter* C API)
//   CompiledModel-> libLiteRt.dylib (LiteRt* C API), CPU + Metal GPU accelerators.
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

const int kOk = 0;
const int kCpu = 1, kGpu = 2, kHostMemory = 1, kLockWrite = 1, kLockRead = 0;
const int kDelegatePrecisionFp32 = 2;

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
typedef _OpaquePayloadDeleterNative = Void Function(Pointer<Void>);

late final DynamicLibrary _rt; // libLiteRt
late final DynamicLibrary _tf; // classic tflite c

void main(List<String> args) {
  final model = args.isNotEmpty
      ? args[0]
      : '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_short_range.tflite';
  final warmup = 30, iters = 300;
  _rt = DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib');
  _tf = DynamicLibrary.open('/tmp/cm_spike/libtensorflowlite_c-mac.dylib');

  print('model: ${model.split('/').last}');
  print(
    'warmup=$warmup  timed iters=$iters  (per-inference Invoke/Run only)\n',
  );

  final rows = <(String, ({double meanUs, double medianUs, double minUs}))>[];
  rows.add((
    'Interpreter (plain CPU)',
    _benchInterpreter(model, xnnpack: false, warmup: warmup, iters: iters),
  ));
  rows.add((
    'Interpreter (XNNPACK)',
    _benchInterpreter(model, xnnpack: true, warmup: warmup, iters: iters),
  ));
  rows.add((
    'CompiledModel (CPU)',
    _benchCompiled(model, kCpu, warmup: warmup, iters: iters),
  ));
  rows.add((
    'CompiledModel (GPU/Metal)',
    _benchCompiled(model, kGpu, warmup: warmup, iters: iters),
  ));
  rows.add((
    'CompiledModel (GPU FP32)',
    _benchCompiled(model, kGpu, gpuFp32: true, warmup: warmup, iters: iters),
  ));

  print(
    '\n${'runtime'.padRight(28)}${'median µs'.padLeft(12)}${'mean µs'.padLeft(12)}${'min µs'.padLeft(12)}',
  );
  print('-' * 64);
  for (final (name, r) in rows) {
    print(
      name.padRight(28) +
          r.medianUs.toStringAsFixed(1).padLeft(12) +
          r.meanUs.toStringAsFixed(1).padLeft(12) +
          r.minUs.toStringAsFixed(1).padLeft(12),
    );
  }
}

({double meanUs, double medianUs, double minUs}) _time(
  void Function() f,
  int warmup,
  int iters,
) {
  for (var i = 0; i < warmup; i++) f();
  final s = Int64List(iters);
  for (var i = 0; i < iters; i++) {
    final sw = Stopwatch()..start();
    f();
    sw.stop();
    s[i] = sw.elapsedMicroseconds;
  }
  s.sort();
  var sum = 0;
  for (final v in s) sum += v;
  return (
    meanUs: sum / iters,
    medianUs: s[iters ~/ 2].toDouble(),
    minUs: s.first.toDouble(),
  );
}

// ---------- classic Interpreter ----------
({double meanUs, double medianUs, double minUs}) _benchInterpreter(
  String path, {
  required bool xnnpack,
  required int warmup,
  required int iters,
}) {
  final modelCreate = _tf
      .lookupFunction<_P Function(Pointer<Utf8>), _P Function(Pointer<Utf8>)>(
    'TfLiteModelCreateFromFile',
  );
  final optsCreate = _tf.lookupFunction<_P Function(), _P Function()>(
    'TfLiteInterpreterOptionsCreate',
  );
  final xnnCreate = _tf.lookupFunction<_P Function(_P), _P Function(_P)>(
    'TfLiteXNNPackDelegateCreate',
  );
  final addDelegate =
      _tf.lookupFunction<Void Function(_P, _P), void Function(_P, _P)>(
    'TfLiteInterpreterOptionsAddDelegate',
  );
  final interpCreate =
      _tf.lookupFunction<_P Function(_P, _P), _P Function(_P, _P)>(
    'TfLiteInterpreterCreate',
  );
  final allocate = _tf.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'TfLiteInterpreterAllocateTensors',
  );
  final inputTensor =
      _tf.lookupFunction<_P Function(_P, Int32), _P Function(_P, int)>(
    'TfLiteInterpreterGetInputTensor',
  );
  final tensorData = _tf.lookupFunction<_P Function(_P), _P Function(_P)>(
    'TfLiteTensorData',
  );
  final tensorBytes = _tf.lookupFunction<IntPtr Function(_P), int Function(_P)>(
    'TfLiteTensorByteSize',
  );
  final invoke = _tf.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'TfLiteInterpreterInvoke',
  );

  final model = modelCreate(path.toNativeUtf8());
  final opts = optsCreate();
  if (xnnpack) {
    final d = xnnCreate(nullptr);
    if (d != nullptr) addDelegate(opts, d);
  }
  final interp = interpCreate(model, opts);
  if (allocate(interp) != kOk) throw StateError('AllocateTensors failed');
  final inT = inputTensor(interp, 0);
  final size = tensorBytes(inT);
  final data = tensorData(inT).cast<Uint8>().asTypedList(size);
  final floats = Float32List.view(data.buffer, data.offsetInBytes, size ~/ 4);
  for (var k = 0; k < floats.length; k++) floats[k] = (k % 10) * 0.1;

  return _time(
    () {
      if (invoke(interp) != kOk) throw StateError('Invoke failed');
    },
    warmup,
    iters,
  );
}

// ---------- CompiledModel ----------
({double meanUs, double medianUs, double minUs}) _benchCompiled(
  String path,
  int accel, {
  bool gpuFp32 = false,
  required int warmup,
  required int iters,
}) {
  final createEnv = _rt.lookupFunction<Int32 Function(Int32, _P, _PP),
      int Function(int, _P, _PP)>('LiteRtCreateEnvironment');
  final createOpts = _rt.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
    'LiteRtCreateOptions',
  );
  final setAccel =
      _rt.lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
    'LiteRtSetOptionsHardwareAccelerators',
  );
  final createOpaque = _rt.lookupFunction<
      Int32 Function(
        Pointer<Utf8>,
        _P,
        Pointer<NativeFunction<_OpaquePayloadDeleterNative>>,
        _PP,
      ),
      int Function(
        Pointer<Utf8>,
        _P,
        Pointer<NativeFunction<_OpaquePayloadDeleterNative>>,
        _PP,
      )>('LiteRtCreateOpaqueOptions');
  final addOpaque =
      _rt.lookupFunction<Int32 Function(_P, _P), int Function(_P, _P)>(
    'LiteRtAddOpaqueOptions',
  );
  final modelFromFile = _rt.lookupFunction<Int32 Function(Pointer<Utf8>, _PP),
      int Function(Pointer<Utf8>, _PP)>('LiteRtCreateModelFromFile');
  final createCM = _rt.lookupFunction<Int32 Function(_P, _P, _P, _PP),
      int Function(_P, _P, _P, _PP)>('LiteRtCreateCompiledModel');
  final getSig = _rt.lookupFunction<Int32 Function(_P, IntPtr, _PP),
      int Function(_P, int, _PP)>('LiteRtGetModelSignature');
  final numIn = _rt.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
      int Function(_P, Pointer<IntPtr>)>('LiteRtGetNumSignatureInputs');
  final numOut = _rt.lookupFunction<Int32 Function(_P, Pointer<IntPtr>),
      int Function(_P, Pointer<IntPtr>)>('LiteRtGetNumSignatureOutputs');
  final inTensor = _rt.lookupFunction<Int32 Function(_P, IntPtr, _PP),
      int Function(_P, int, _PP)>('LiteRtGetSignatureInputTensorByIndex');
  final outTensor = _rt.lookupFunction<Int32 Function(_P, IntPtr, _PP),
      int Function(_P, int, _PP)>('LiteRtGetSignatureOutputTensorByIndex');
  final rankedType = _rt.lookupFunction<
      Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
      int Function(
          _P, Pointer<LiteRtRankedTensorType>)>('LiteRtGetRankedTensorType');
  final inReq = _rt.lookupFunction<
      Int32 Function(_P, IntPtr, IntPtr, _PP),
      int Function(
          _P, int, int, _PP)>('LiteRtGetCompiledModelInputBufferRequirements');
  final outReq = _rt.lookupFunction<
      Int32 Function(_P, IntPtr, IntPtr, _PP),
      int Function(
          _P, int, int, _PP)>('LiteRtGetCompiledModelOutputBufferRequirements');
  final createBuf = _rt.lookupFunction<
      Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
      int Function(_P, Pointer<LiteRtRankedTensorType>, _P,
          _PP)>('LiteRtCreateManagedTensorBufferFromRequirements');
  final lock = _rt.lookupFunction<Int32 Function(_P, _PP, Int32),
      int Function(_P, _PP, int)>('LiteRtLockTensorBuffer');
  final unlock = _rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'LiteRtUnlockTensorBuffer',
  );
  final run = _rt.lookupFunction<
      Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
      int Function(_P, int, int, _PP, int, _PP)>('LiteRtRunCompiledModel');
  final getInLayout = _rt.lookupFunction<
      Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
      int Function(_P, int, int,
          Pointer<LiteRtLayout>)>('LiteRtGetCompiledModelInputTensorLayout');
  final getOutLayouts = _rt.lookupFunction<
      Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
      int Function(_P, int, int, Pointer<LiteRtLayout>,
          int)>('LiteRtGetCompiledModelOutputTensorLayouts');
  void ck(String w, int s) {
    if (s != kOk) throw StateError('$w → $s (accel=$accel)');
  }

  final envOut = calloc<Pointer<Void>>();
  ck('env', createEnv(0, nullptr, envOut));
  final env = envOut.value;
  final optsOut = calloc<Pointer<Void>>();
  ck('opts', createOpts(optsOut));
  final opts = optsOut.value;
  ck('accel', setAccel(opts, accel));
  if (gpuFp32) {
    final id = 'gpu_options'.toNativeUtf8();
    final payload = 'precision = $kDelegatePrecisionFp32\n'.toNativeUtf8();
    final opaqueOut = calloc<Pointer<Void>>();
    ck(
      'gpuOpaque',
      createOpaque(
        id,
        payload.cast<Void>(),
        malloc.nativeFree.cast<NativeFunction<_OpaquePayloadDeleterNative>>(),
        opaqueOut,
      ),
    );
    ck('addGpuOpaque', addOpaque(opts, opaqueOut.value));
  }
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

  final inBufs = calloc<Pointer<Void>>(nIn);
  for (var i = 0; i < nIn; i++) {
    final tOut = calloc<Pointer<Void>>();
    ck('inT', inTensor(sig, i, tOut));
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('rtt', rankedType(tOut.value, rtt));
    ck(
      'inLayout',
      getInLayout(cm, 0, i, (rtt.cast<Uint8>() + 4).cast<LiteRtLayout>()),
    );
    final reqOut = calloc<Pointer<Void>>();
    ck('inReq', inReq(cm, 0, i, reqOut));
    final bufOut = calloc<Pointer<Void>>();
    ck('inBuf', createBuf(env, rtt, reqOut.value, bufOut));
    inBufs[i] = bufOut.value;
    final addrOut = calloc<Pointer<Void>>();
    ck('lock', lock(bufOut.value, addrOut, kLockWrite));
    // fill with a ramp; size from lock not needed for timing correctness
    final szP = calloc<IntPtr>();
    final reqSize = _rt.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P,
            Pointer<IntPtr>)>('LiteRtGetTensorBufferRequirementsBufferSize');
    ck('sz', reqSize(reqOut.value, szP));
    final fl = Float32List.view(
      addrOut.value.cast<Uint8>().asTypedList(szP.value).buffer,
      0,
      szP.value ~/ 4,
    );
    for (var k = 0; k < fl.length; k++) fl[k] = (k % 10) * 0.1;
    ck('unlock', unlock(bufOut.value));
  }
  final outBufs = calloc<Pointer<Void>>(nOut);
  final outLayouts = calloc<LiteRtLayout>(nOut);
  ck('outLayouts', getOutLayouts(cm, 0, nOut, outLayouts, 1));
  for (var j = 0; j < nOut; j++) {
    final tOut = calloc<Pointer<Void>>();
    ck('outT', outTensor(sig, j, tOut));
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('rtt', rankedType(tOut.value, rtt));
    (rtt.cast<Uint8>() + 4)
        .asTypedList(68)
        .setAll(0, (outLayouts + j).cast<Uint8>().asTypedList(68));
    final reqOut = calloc<Pointer<Void>>();
    ck('outReq', outReq(cm, 0, j, reqOut));
    final bufOut = calloc<Pointer<Void>>();
    ck('outBuf', createBuf(env, rtt, reqOut.value, bufOut));
    outBufs[j] = bufOut.value;
  }

  return _time(
    () {
      if (run(cm, 0, nIn, inBufs, nOut, outBufs) != kOk)
        throw StateError('run failed');
    },
    warmup,
    iters,
  );
}
