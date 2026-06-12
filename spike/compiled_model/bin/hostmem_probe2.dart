// Probe v2: host-memory TensorBuffers, done right.
// v1 (hostmem_probe.dart) bound managed buffers first, then host buffers,
// then managed again ON THE SAME CompiledModel — the "Failed to allocate
// tensors" came from re-binding buffer sets, not from host memory itself.
// Here each mode gets a FRESH CompiledModel and one buffer kind bound for
// its whole life, 330 runs (30 warmup), symmetric fill/read methodology:
//   fill = setAll from a precomputed Float32List (managed: under write lock;
//          host: directly into the wrapped memory)
//   read = sum over all output floats (managed: under read lock; host: direct)
// ignore_for_file: avoid_print
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

class _Fns {
  final createEnv = _rt.lookupFunction<
    Int32 Function(IntPtr, _P, _PP),
    int Function(int, _P, _PP)
  >('LiteRtCreateEnvironment');
  final createOpts = _rt.lookupFunction<Int32 Function(_PP), int Function(_PP)>(
    'LiteRtCreateOptions',
  );
  final setAccel = _rt
      .lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
        'LiteRtSetOptionsHardwareAccelerators',
      );
  final modelFromFile = _rt.lookupFunction<
    Int32 Function(Pointer<Utf8>, _PP),
    int Function(Pointer<Utf8>, _PP)
  >('LiteRtCreateModelFromFile');
  final createCM = _rt.lookupFunction<
    Int32 Function(_P, _P, _P, _PP),
    int Function(_P, _P, _P, _PP)
  >('LiteRtCreateCompiledModel');
  final getSig = _rt.lookupFunction<
    Int32 Function(_P, IntPtr, _PP),
    int Function(_P, int, _PP)
  >('LiteRtGetModelSignature');
  final numIn = _rt.lookupFunction<
    Int32 Function(_P, Pointer<IntPtr>),
    int Function(_P, Pointer<IntPtr>)
  >('LiteRtGetNumSignatureInputs');
  final numOut = _rt.lookupFunction<
    Int32 Function(_P, Pointer<IntPtr>),
    int Function(_P, Pointer<IntPtr>)
  >('LiteRtGetNumSignatureOutputs');
  final inTensor = _rt.lookupFunction<
    Int32 Function(_P, IntPtr, _PP),
    int Function(_P, int, _PP)
  >('LiteRtGetSignatureInputTensorByIndex');
  final outTensor = _rt.lookupFunction<
    Int32 Function(_P, IntPtr, _PP),
    int Function(_P, int, _PP)
  >('LiteRtGetSignatureOutputTensorByIndex');
  final rankedType = _rt.lookupFunction<
    Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
    int Function(_P, Pointer<LiteRtRankedTensorType>)
  >('LiteRtGetRankedTensorType');
  final inReq = _rt.lookupFunction<
    Int32 Function(_P, IntPtr, IntPtr, _PP),
    int Function(_P, int, int, _PP)
  >('LiteRtGetCompiledModelInputBufferRequirements');
  final outReq = _rt.lookupFunction<
    Int32 Function(_P, IntPtr, IntPtr, _PP),
    int Function(_P, int, int, _PP)
  >('LiteRtGetCompiledModelOutputBufferRequirements');
  final reqSize = _rt.lookupFunction<
    Int32 Function(_P, Pointer<IntPtr>),
    int Function(_P, Pointer<IntPtr>)
  >('LiteRtGetTensorBufferRequirementsBufferSize');
  final createManaged = _rt.lookupFunction<
    Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
    int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP)
  >('LiteRtCreateManagedTensorBufferFromRequirements');
  final createFromHost = _rt.lookupFunction<
    Int32 Function(Pointer<LiteRtRankedTensorType>, _P, IntPtr, _P, _PP),
    int Function(Pointer<LiteRtRankedTensorType>, _P, int, _P, _PP)
  >('LiteRtCreateTensorBufferFromHostMemory');
  final lock = _rt.lookupFunction<
    Int32 Function(_P, _PP, Int32),
    int Function(_P, _PP, int)
  >('LiteRtLockTensorBuffer');
  final unlock = _rt.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'LiteRtUnlockTensorBuffer',
  );
  final run = _rt.lookupFunction<
    Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
    int Function(_P, int, int, _PP, int, _PP)
  >('LiteRtRunCompiledModel');
}

void main(List<String> args) {
  final model =
      args.isNotEmpty
          ? args[0]
          : '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_back.tflite';
  _rt = DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib');
  print(
    'model: ${model.split('/').last}  (330 runs/mode, 30 warmup, median µs)\n',
  );
  final f = _Fns();

  for (final (name, accel) in [
    ('CPU', kCpu),
    ('GPU strict', kGpu),
    ('GPU|CPU', kGpu | kCpu),
  ]) {
    print('=== accelerator: $name (mask $accel) ===');
    List<Float32List>? ref;
    for (final hostMode in [false, true]) {
      final label = hostMode ? 'host-memory' : 'managed+lock';
      try {
        final (outs, medUs) = _runMode(f, model, accel, hostMode);
        if (ref == null) {
          ref = outs;
        } else {
          var maxDiff = 0.0;
          for (var i = 0; i < outs.length; i++) {
            for (var j = 0; j < outs[i].length; j++) {
              final d = (outs[i][j] - ref[i][j]).abs();
              if (d > maxDiff) maxDiff = d;
            }
          }
          print(
            '  $label: median ${medUs.toStringAsFixed(1)} µs   '
            'max|diff| vs managed = ${maxDiff.toStringAsFixed(6)}',
          );
          continue;
        }
        print('  $label: median ${medUs.toStringAsFixed(1)} µs');
      } catch (e) {
        print('  $label: FAILED $e');
      }
    }
    print('');
  }
}

(List<Float32List>, double) _runMode(
  _Fns f,
  String path,
  int accel,
  bool hostMode,
) {
  final envOut = calloc<Pointer<Void>>();
  ck('env', f.createEnv(0, nullptr, envOut));
  final env = envOut.value;
  final optsOut = calloc<Pointer<Void>>();
  ck('opts', f.createOpts(optsOut));
  ck('accel', f.setAccel(optsOut.value, accel));
  final modelOut = calloc<Pointer<Void>>();
  ck('model', f.modelFromFile(path.toNativeUtf8(), modelOut));
  final model = modelOut.value;
  final cmOut = calloc<Pointer<Void>>();
  ck('cm', f.createCM(env, model, optsOut.value, cmOut));
  final cm = cmOut.value;
  final sigOut = calloc<Pointer<Void>>();
  ck('sig', f.getSig(model, 0, sigOut));
  final sig = sigOut.value;
  final nInP = calloc<IntPtr>(), nOutP = calloc<IntPtr>();
  ck('nIn', f.numIn(sig, nInP));
  ck('nOut', f.numOut(sig, nOutP));
  final nIn = nInP.value, nOut = nOutP.value;
  final n = nIn + nOut;

  final sizes = <int>[];
  final bufs = <_P>[];
  final hostPtrs = <Pointer<Float>?>[];
  for (var i = 0; i < n; i++) {
    final isIn = i < nIn;
    final idx = isIn ? i : i - nIn;
    final tOut = calloc<Pointer<Void>>();
    ck(
      'tensor',
      isIn ? f.inTensor(sig, idx, tOut) : f.outTensor(sig, idx, tOut),
    );
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('rtt', f.rankedType(tOut.value, rtt));
    final rOut = calloc<Pointer<Void>>();
    ck('req', isIn ? f.inReq(cm, 0, idx, rOut) : f.outReq(cm, 0, idx, rOut));
    final sP = calloc<IntPtr>();
    ck('size', f.reqSize(rOut.value, sP));
    sizes.add(sP.value);
    final bOut = calloc<Pointer<Void>>();
    if (hostMode) {
      // 16 KB page alignment: lets Metal wrap without copy if it wants to.
      final raw = calloc<Uint8>(sP.value + 16384);
      final aligned = Pointer<Float>.fromAddress(
        (raw.address + 16383) & ~16383,
      );
      hostPtrs.add(aligned);
      ck(
        'hostBuf',
        f.createFromHost(rtt, aligned.cast(), sP.value, nullptr, bOut),
      );
    } else {
      hostPtrs.add(null);
      ck('managedBuf', f.createManaged(env, rtt, rOut.value, bOut));
    }
    bufs.add(bOut.value);
  }

  final inBufs = calloc<Pointer<Void>>(nIn);
  final outBufs = calloc<Pointer<Void>>(nOut);
  for (var i = 0; i < nIn; i++) {
    inBufs[i] = bufs[i];
  }
  for (var i = 0; i < nOut; i++) {
    outBufs[i] = bufs[nIn + i];
  }

  // Same deterministic input for every mode.
  final src = Float32List(sizes[0] ~/ 4);
  for (var j = 0; j < src.length; j++) {
    src[j] = (j % 255) / 255.0 - 0.5;
  }

  double fillAndRunAndSum() {
    // fill
    for (var i = 0; i < nIn; i++) {
      if (hostMode) {
        hostPtrs[i]!.asTypedList(sizes[i] ~/ 4).setAll(0, src);
      } else {
        final hp = calloc<Pointer<Void>>();
        ck('lockW', f.lock(bufs[i], hp, kLockWrite));
        hp.value.cast<Float>().asTypedList(sizes[i] ~/ 4).setAll(0, src);
        ck('unlockW', f.unlock(bufs[i]));
        calloc.free(hp);
      }
    }
    ck('run', f.run(cm, 0, nIn, inBufs, nOut, outBufs));
    // read: sum all output floats
    var acc = 0.0;
    for (var i = 0; i < nOut; i++) {
      final k = nIn + i;
      if (hostMode) {
        final v = hostPtrs[k]!.asTypedList(sizes[k] ~/ 4);
        for (var j = 0; j < v.length; j++) {
          acc += v[j];
        }
      } else {
        final hp = calloc<Pointer<Void>>();
        ck('lockR', f.lock(bufs[k], hp, kLockRead));
        final v = hp.value.cast<Float>().asTypedList(sizes[k] ~/ 4);
        for (var j = 0; j < v.length; j++) {
          acc += v[j];
        }
        ck('unlockR', f.unlock(bufs[k]));
        calloc.free(hp);
      }
    }
    return acc;
  }

  final times = <int>[];
  for (var it = 0; it < 330; it++) {
    final sw = Stopwatch()..start();
    final acc = fillAndRunAndSum();
    sw.stop();
    if (acc.isNaN) throw StateError('NaN output at iteration $it');
    if (it >= 30) times.add(sw.elapsedMicroseconds);
  }
  times.sort();
  final med = times[times.length ~/ 2].toDouble();

  // capture final outputs for cross-mode comparison
  final outs = <Float32List>[];
  for (var i = 0; i < nOut; i++) {
    final k = nIn + i;
    if (hostMode) {
      outs.add(Float32List.fromList(hostPtrs[k]!.asTypedList(sizes[k] ~/ 4)));
    } else {
      final hp = calloc<Pointer<Void>>();
      ck('lockR', f.lock(bufs[k], hp, kLockRead));
      outs.add(
        Float32List.fromList(hp.value.cast<Float>().asTypedList(sizes[k] ~/ 4)),
      );
      ck('unlockR', f.unlock(bufs[k]));
      calloc.free(hp);
    }
  }
  return (outs, med);
}
