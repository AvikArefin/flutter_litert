// Probe: official zero-copy via LiteRtCreateTensorBufferFromHostMemory.
// Per Google's LiteRT Next docs, zero-copy I/O = wrap caller-owned 64-byte-
// aligned host memory as TensorBuffers at creation time (no per-run locks).
// Questions answered here, per accelerator (CPU / GPU Metal / GPU|CPU):
//   1. Which buffer types do the compiled model's requirements support?
//      (is kLiteRtTensorBufferTypeHostMemory=1 among them on Metal?)
//   2. Does Run() accept host-memory buffers, and is the output identical
//      to the managed-buffer path?
//   3. Latency: managed+lock I/O vs host-memory views (median µs).
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

const int kOk = 0;
const int kCpu = 1, kGpu = 2, kLockWrite = 1, kLockRead = 0;

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

void ck(String w, int s) {
  if (s != kOk) throw StateError('$w → $s');
}

void main(List<String> args) {
  final model = args.isNotEmpty
      ? args[0]
      : '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_back.tflite';
  _rt = DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib');
  print('model: ${model.split('/').last}\n');

  for (final (name, accel) in [
    ('CPU', kCpu),
    ('GPU strict', kGpu),
    ('GPU|CPU', kGpu | kCpu),
  ]) {
    print('=== accelerator: $name (mask $accel) ===');
    try {
      _probe(model, accel);
    } catch (e) {
      print('  FAILED: $e');
    }
    print('');
  }
}

void _probe(String path, int accel) {
  final createEnv = _rt.lookupFunction<Int32 Function(IntPtr, _P, _PP),
      int Function(int, _P, _PP)>('LiteRtCreateEnvironment');
  final createOpts = _rt.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
      'LiteRtCreateOptions');
  final setAccel =
      _rt.lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
          'LiteRtSetOptionsHardwareAccelerators');
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
  final reqSize = _rt.lookupFunction<
      Int32 Function(_P, Pointer<IntPtr>),
      int Function(
          _P, Pointer<IntPtr>)>('LiteRtGetTensorBufferRequirementsBufferSize');
  final reqNumTypes = _rt.lookupFunction<Int32 Function(_P, Pointer<Int32>),
          int Function(_P, Pointer<Int32>)>(
      'LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes');
  final reqTypeAt = _rt.lookupFunction<
          Int32 Function(_P, Int32, Pointer<Int32>),
          int Function(_P, int, Pointer<Int32>)>(
      'LiteRtGetTensorBufferRequirementsSupportedTensorBufferType');
  final createManaged = _rt.lookupFunction<
      Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
      int Function(_P, Pointer<LiteRtRankedTensorType>, _P,
          _PP)>('LiteRtCreateManagedTensorBufferFromRequirements');
  final createFromHost = _rt.lookupFunction<
      Int32 Function(Pointer<LiteRtRankedTensorType>, _P, IntPtr, _P, _PP),
      int Function(Pointer<LiteRtRankedTensorType>, _P, int, _P,
          _PP)>('LiteRtCreateTensorBufferFromHostMemory');
  final lock = _rt.lookupFunction<Int32 Function(_P, _PP, Int32),
      int Function(_P, _PP, int)>('LiteRtLockTensorBuffer');
  final unlock = _rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
      'LiteRtUnlockTensorBuffer');
  final run = _rt.lookupFunction<
      Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
      int Function(_P, int, int, _PP, int, _PP)>('LiteRtRunCompiledModel');

  // --- setup ---
  final envOut = calloc<Pointer<Void>>();
  ck('env', createEnv(0, nullptr, envOut));
  final env = envOut.value;
  final optsOut = calloc<Pointer<Void>>();
  ck('opts', createOpts(optsOut));
  ck('accel', setAccel(optsOut.value, accel));
  final modelOut = calloc<Pointer<Void>>();
  ck('model', modelFromFile(path.toNativeUtf8(), modelOut));
  final model = modelOut.value;
  final cmOut = calloc<Pointer<Void>>();
  ck('cm', createCM(env, model, optsOut.value, cmOut));
  final cm = cmOut.value;
  final sigOut = calloc<Pointer<Void>>();
  ck('sig', getSig(model, 0, sigOut));
  final sig = sigOut.value;
  final nInP = calloc<IntPtr>(), nOutP = calloc<IntPtr>();
  ck('nIn', numIn(sig, nInP));
  ck('nOut', numOut(sig, nOutP));
  final nIn = nInP.value, nOut = nOutP.value;

  // --- 1. dump supported buffer types from requirements ---
  String typesOf(_P req) {
    final nP = calloc<Int32>();
    ck('numTypes', reqNumTypes(req, nP));
    final names = <String>[];
    for (var i = 0; i < nP.value; i++) {
      final tP = calloc<Int32>();
      ck('typeAt', reqTypeAt(req, i, tP));
      names.add(tP.value.toString());
      calloc.free(tP);
    }
    calloc.free(nP);
    return names.join(',');
  }

  final reqs = <_P>[];
  final sizes = <int>[];
  final rtts = <Pointer<LiteRtRankedTensorType>>[];
  for (var i = 0; i < nIn + nOut; i++) {
    final isIn = i < nIn;
    final idx = isIn ? i : i - nIn;
    final tOut = calloc<Pointer<Void>>();
    ck('tensor', isIn ? inTensor(sig, idx, tOut) : outTensor(sig, idx, tOut));
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('rtt', rankedType(tOut.value, rtt));
    rtts.add(rtt);
    final rOut = calloc<Pointer<Void>>();
    ck('req', isIn ? inReq(cm, 0, idx, rOut) : outReq(cm, 0, idx, rOut));
    reqs.add(rOut.value);
    final sP = calloc<IntPtr>();
    ck('size', reqSize(rOut.value, sP));
    sizes.add(sP.value);
    final label = isIn ? 'input[$idx]' : 'output[$idx]';
    print('  $label bytes=${sP.value} supportedTypes=[${typesOf(rOut.value)}]'
        ' (1=HostMemory, 30=MetalBuffer)');
  }

  // --- 2. managed-buffer reference run ---
  final managed = <_P>[];
  for (var i = 0; i < nIn + nOut; i++) {
    final bOut = calloc<Pointer<Void>>();
    ck('managedBuf', createManaged(env, rtts[i], reqs[i], bOut));
    managed.add(bOut.value);
  }
  final inBufs = calloc<Pointer<Void>>(nIn);
  final outBufs = calloc<Pointer<Void>>(nOut);

  void writeInput(_P buf, int bytes) {
    final hp = calloc<Pointer<Void>>();
    ck('lockW', lock(buf, hp, kLockWrite));
    final f = hp.value.cast<Float>().asTypedList(bytes ~/ 4);
    for (var i = 0; i < f.length; i++) {
      f[i] = (i % 255) / 255.0 - 0.5;
    }
    ck('unlockW', unlock(buf));
    calloc.free(hp);
  }

  Float32List readOutput(_P buf, int bytes) {
    final hp = calloc<Pointer<Void>>();
    ck('lockR', lock(buf, hp, kLockRead));
    final out =
        Float32List.fromList(hp.value.cast<Float>().asTypedList(bytes ~/ 4));
    ck('unlockR', unlock(buf));
    calloc.free(hp);
    return out;
  }

  for (var i = 0; i < nIn; i++) {
    inBufs[i] = managed[i];
    writeInput(managed[i], sizes[i]);
  }
  for (var i = 0; i < nOut; i++) {
    outBufs[i] = managed[nIn + i];
  }
  ck('run managed', run(cm, 0, nIn, inBufs, nOut, outBufs));
  final refOut = [
    for (var i = 0; i < nOut; i++) readOutput(managed[nIn + i], sizes[nIn + i]),
  ];

  // --- 3. host-memory buffers (64-byte aligned, caller-owned) ---
  final hostPtrs = <Pointer<Float>>[];
  final hostBufs = <_P>[];
  for (var i = 0; i < nIn + nOut; i++) {
    final raw = calloc<Uint8>(sizes[i] + 64);
    final aligned = Pointer<Float>.fromAddress((raw.address + 63) & ~63);
    hostPtrs.add(aligned);
    final bOut = calloc<Pointer<Void>>();
    final st = createFromHost(rtts[i], aligned.cast(), sizes[i], nullptr, bOut);
    if (st != kOk) {
      print(
          '  CreateTensorBufferFromHostMemory(${i < nIn ? "input" : "output"}) → $st');
      return;
    }
    hostBufs.add(bOut.value);
  }
  for (var i = 0; i < nIn; i++) {
    inBufs[i] = hostBufs[i];
    final f = hostPtrs[i].asTypedList(sizes[i] ~/ 4);
    for (var j = 0; j < f.length; j++) {
      f[j] = (j % 255) / 255.0 - 0.5;
    }
  }
  for (var i = 0; i < nOut; i++) {
    outBufs[i] = hostBufs[nIn + i];
  }
  final hostRunSt = run(cm, 0, nIn, inBufs, nOut, outBufs);
  if (hostRunSt != kOk) {
    print('  Run(host-memory buffers) → status $hostRunSt — NOT SUPPORTED');
    return;
  }

  // --- 4. compare outputs ---
  var maxDiff = 0.0;
  for (var i = 0; i < nOut; i++) {
    final got = hostPtrs[nIn + i].asTypedList(sizes[nIn + i] ~/ 4);
    for (var j = 0; j < got.length; j++) {
      final d = (got[j] - refOut[i][j]).abs();
      if (d > maxDiff) maxDiff = d;
    }
  }
  print(
      '  Run(host-memory) OK; max |diff| vs managed = ${maxDiff.toStringAsFixed(6)}');

  // --- 5. timing: managed+lock I/O vs host-memory views ---
  double med(List<int> xs) {
    final s = [...xs]..sort();
    return s[s.length ~/ 2].toDouble();
  }

  final tManaged = <int>[];
  for (var i = 0; i < nIn; i++) {
    inBufs[i] = managed[i];
  }
  for (var i = 0; i < nOut; i++) {
    outBufs[i] = managed[nIn + i];
  }
  for (var it = 0; it < 330; it++) {
    final sw = Stopwatch()..start();
    for (var i = 0; i < nIn; i++) {
      writeInput(managed[i], sizes[i]);
    }
    ck('run', run(cm, 0, nIn, inBufs, nOut, outBufs));
    for (var i = 0; i < nOut; i++) {
      readOutput(managed[nIn + i], sizes[nIn + i]);
    }
    sw.stop();
    if (it >= 30) tManaged.add(sw.elapsedMicroseconds);
  }

  final tHost = <int>[];
  for (var i = 0; i < nIn; i++) {
    inBufs[i] = hostBufs[i];
  }
  for (var i = 0; i < nOut; i++) {
    outBufs[i] = hostBufs[nIn + i];
  }
  for (var it = 0; it < 330; it++) {
    final sw = Stopwatch()..start();
    for (var i = 0; i < nIn; i++) {
      final f = hostPtrs[i].asTypedList(sizes[i] ~/ 4);
      for (var j = 0; j < f.length; j++) {
        f[j] = (j % 255) / 255.0 - 0.5;
      }
    }
    ck('run', run(cm, 0, nIn, inBufs, nOut, outBufs));
    var acc = 0.0;
    for (var i = 0; i < nOut; i++) {
      acc += hostPtrs[nIn + i].asTypedList(1)[0];
    }
    sw.stop();
    if (acc.isNaN) print('nan');
    if (it >= 30) tHost.add(sw.elapsedMicroseconds);
  }

  print('  managed+lock : median ${med(tManaged).toStringAsFixed(1)} µs');
  print('  host-memory  : median ${med(tHost).toStringAsFixed(1)} µs');
}
