// LiteRT Next CompiledModel — full inference + CPU-vs-GPU numerical parity (macOS).
// Runs the same deterministic input through the CPU accelerator and the Metal GPU
// accelerator and compares outputs. Self-contained proof that CompiledModel runs
// correctly and that GPU works — no Interpreter / Flutter dependency needed.
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

const int kOk = 0;
const int kCpu = 1; // kLiteRtHwAcceleratorCpu
const int kGpu = 2; // kLiteRtHwAcceleratorGpu
const int kHostMemory = 1; // kLiteRtTensorBufferTypeHostMemory
const int kLockWrite = 1, kLockRead = 0;

typedef _P = Pointer<Void>;
typedef _PP = Pointer<Pointer<Void>>;

// LiteRtLayout: { uint rank:7; bool has_strides:1; int32 dims[8]; uint32 strides[8]; } = 68 bytes
final class LiteRtLayout extends Struct {
  @Uint32()
  external int bitfields;
  @Array(8)
  external Array<Int32> dimensions;
  @Array(8)
  external Array<Uint32> strides;
}

// LiteRtRankedTensorType: { int32 element_type; LiteRtLayout layout; } = 72 bytes
final class LiteRtRankedTensorType extends Struct {
  @Int32()
  external int elementType;
  external LiteRtLayout layout;
}

late final DynamicLibrary _lib;

void main(List<String> args) {
  final modelPath = args.isNotEmpty
      ? args[0]
      : '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/simple_model.tflite';
  _lib = DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib');

  if (sizeOf<LiteRtRankedTensorType>() != 72) {
    throw StateError(
      'RankedTensorType size ${sizeOf<LiteRtRankedTensorType>()} != 72',
    );
  }
  print(
    'struct sizes ok (RankedTensorType=${sizeOf<LiteRtRankedTensorType>()})',
  );

  final cpu = _runOnce(modelPath, kCpu);
  print('✓ CPU inference: ${cpu.length} floats, head=${_head(cpu)}');
  final gpu = _runOnce(modelPath, kGpu);
  print('✓ GPU inference: ${gpu.length} floats, head=${_head(gpu)}');

  if (cpu.length != gpu.length) {
    throw StateError('output length mismatch ${cpu.length} vs ${gpu.length}');
  }
  double maxAbs = 0;
  for (var i = 0; i < cpu.length; i++) {
    final d = (cpu[i] - gpu[i]).abs();
    if (d > maxAbs) maxAbs = d;
  }
  print('\nmax |CPU - GPU| diff over ${cpu.length} elements = $maxAbs');
  print(
    maxAbs < 1e-2
        ? '🎉 PARITY PASSED — CompiledModel runs correctly on CPU and GPU (Metal).'
        : '⚠️  diff above tolerance — investigate (dtype/quantization?)',
  );
}

String _head(Float32List v) => List.generate(
  v.length < 5 ? v.length : 5,
  (i) => v[i].toStringAsFixed(4),
).toString();

Float32List _runOnce(String modelPath, int accel) {
  final createEnv = _lib
      .lookupFunction<
        Int32 Function(Int32, _P, _PP),
        int Function(int, _P, _PP)
      >('LiteRtCreateEnvironment');
  final createOpts = _lib
      .lookupFunction<Int32 Function(_PP), int Function(_PP)>(
        'LiteRtCreateOptions',
      );
  final setAccel = _lib
      .lookupFunction<Int32 Function(_P, Int32), int Function(_P, int)>(
        'LiteRtSetOptionsHardwareAccelerators',
      );
  final modelFromFile = _lib
      .lookupFunction<
        Int32 Function(Pointer<Utf8>, _PP),
        int Function(Pointer<Utf8>, _PP)
      >('LiteRtCreateModelFromFile');
  final createCM = _lib
      .lookupFunction<
        Int32 Function(_P, _P, _P, _PP),
        int Function(_P, _P, _P, _PP)
      >('LiteRtCreateCompiledModel');
  final getSig = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetModelSignature');
  final numIn = _lib
      .lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureInputs');
  final numOut = _lib
      .lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureOutputs');
  final inTensor = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureInputTensorByIndex');
  final outTensor = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureOutputTensorByIndex');
  final rankedType = _lib
      .lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
        int Function(_P, Pointer<LiteRtRankedTensorType>)
      >('LiteRtGetRankedTensorType');
  final inReq = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelInputBufferRequirements');
  final outReq = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelOutputBufferRequirements');
  final reqSize = _lib
      .lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetTensorBufferRequirementsBufferSize');
  final createBuf = _lib
      .lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
        int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP)
      >('LiteRtCreateManagedTensorBufferFromRequirements');
  final lock = _lib
      .lookupFunction<
        Int32 Function(_P, _PP, Int32),
        int Function(_P, _PP, int)
      >('LiteRtLockTensorBuffer');
  final unlock = _lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
    'LiteRtUnlockTensorBuffer',
  );
  final getBufType = _lib
      .lookupFunction<
        Int32 Function(_P, Pointer<Int32>),
        int Function(_P, Pointer<Int32>)
      >('LiteRtGetTensorBufferType');
  final run = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
        int Function(_P, int, int, _PP, int, _PP)
      >('LiteRtRunCompiledModel');
  final getInLayout = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
        int Function(_P, int, int, Pointer<LiteRtLayout>)
      >('LiteRtGetCompiledModelInputTensorLayout');
  final getOutLayouts = _lib
      .lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
        int Function(_P, int, int, Pointer<LiteRtLayout>, int)
      >('LiteRtGetCompiledModelOutputTensorLayouts');

  int ck(String w, int s) {
    if (s != kOk) throw StateError('$w → LiteRtStatus=$s (accel=$accel)');
    return s;
  }

  final envOut = calloc<Pointer<Void>>();
  ck('CreateEnvironment', createEnv(0, nullptr, envOut));
  final env = envOut.value;
  final optsOut = calloc<Pointer<Void>>();
  ck('CreateOptions', createOpts(optsOut));
  final opts = optsOut.value;
  ck('SetHardwareAccelerators', setAccel(opts, accel));
  final namePtr = modelPath.toNativeUtf8();
  final modelOut = calloc<Pointer<Void>>();
  ck('CreateModelFromFile', modelFromFile(namePtr, modelOut));
  final model = modelOut.value;
  final cmOut = calloc<Pointer<Void>>();
  ck('CreateCompiledModel', createCM(env, model, opts, cmOut));
  final cm = cmOut.value;

  final sigOut = calloc<Pointer<Void>>();
  ck('GetModelSignature', getSig(model, 0, sigOut));
  final sig = sigOut.value;
  final nInP = calloc<IntPtr>(), nOutP = calloc<IntPtr>();
  ck('NumInputs', numIn(sig, nInP));
  ck('NumOutputs', numOut(sig, nOutP));
  final nIn = nInP.value, nOut = nOutP.value;

  final inBufs = calloc<Pointer<Void>>(nIn);
  for (var i = 0; i < nIn; i++) {
    final tOut = calloc<Pointer<Void>>();
    ck('InputTensor[$i]', inTensor(sig, i, tOut));
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('RankedType in[$i]', rankedType(tOut.value, rtt));
    // Override declared layout with the concrete runtime layout (declared may be dynamic).
    final inLayoutPtr = (rtt.cast<Uint8>() + 4).cast<LiteRtLayout>();
    ck('InLayout[$i]', getInLayout(cm, 0, i, inLayoutPtr));
    final reqOut = calloc<Pointer<Void>>();
    ck('InputReq[$i]', inReq(cm, 0, i, reqOut));
    final szP = calloc<IntPtr>();
    ck('ReqSize in[$i]', reqSize(reqOut.value, szP));
    final size = szP.value;
    final bufOut = calloc<Pointer<Void>>();
    ck('CreateBuf in[$i]', createBuf(env, rtt, reqOut.value, bufOut));
    inBufs[i] = bufOut.value;
    if (i == 0) {
      final tp = calloc<Int32>();
      ck('GetBufType', getBufType(bufOut.value, tp));
      print(
        '  [accel=$accel] input buffer type allocated = ${tp.value} '
        '(1=HostMemory, 30=MetalBuffer, 31=MetalBufferFp16)',
      );
      calloc.free(tp);
    }
    final addrOut = calloc<Pointer<Void>>();
    ck('Lock in[$i]', lock(bufOut.value, addrOut, kLockWrite));
    final bytes = addrOut.value.cast<Uint8>().asTypedList(size);
    final floats = Float32List.view(
      bytes.buffer,
      bytes.offsetInBytes,
      size ~/ 4,
    );
    for (var k = 0; k < floats.length; k++) {
      floats[k] = (k % 10) * 0.1;
    }
    ck('Unlock in[$i]', unlock(bufOut.value));
    calloc.free(tOut);
    calloc.free(rtt);
    calloc.free(reqOut);
    calloc.free(szP);
    calloc.free(bufOut);
    calloc.free(addrOut);
  }

  final outBufs = calloc<Pointer<Void>>(nOut);
  final outSizes = <int>[];
  // Fetch concrete runtime layouts for all outputs once (update_allocation=true).
  final outLayouts = calloc<LiteRtLayout>(nOut);
  ck('OutLayouts', getOutLayouts(cm, 0, nOut, outLayouts, 1));
  for (var j = 0; j < nOut; j++) {
    final tOut = calloc<Pointer<Void>>();
    ck('OutputTensor[$j]', outTensor(sig, j, tOut));
    final rtt = calloc<LiteRtRankedTensorType>();
    ck('RankedType out[$j]', rankedType(tOut.value, rtt));
    // Override declared layout with the runtime layout for output j.
    (rtt.cast<Uint8>() + 4)
        .asTypedList(68)
        .setAll(0, (outLayouts + j).cast<Uint8>().asTypedList(68));
    final reqOut = calloc<Pointer<Void>>();
    ck('OutputReq[$j]', outReq(cm, 0, j, reqOut));
    final szP = calloc<IntPtr>();
    ck('ReqSize out[$j]', reqSize(reqOut.value, szP));
    outSizes.add(szP.value);
    final bufOut = calloc<Pointer<Void>>();
    ck('CreateBuf out[$j]', createBuf(env, rtt, reqOut.value, bufOut));
    outBufs[j] = bufOut.value;
    calloc.free(tOut);
    calloc.free(rtt);
    calloc.free(reqOut);
    calloc.free(szP);
    calloc.free(bufOut);
  }

  ck('RunCompiledModel', run(cm, 0, nIn, inBufs, nOut, outBufs));

  final addrOut = calloc<Pointer<Void>>();
  ck('Lock out[0]', lock(outBufs[0], addrOut, kLockRead));
  final ob = addrOut.value.cast<Uint8>().asTypedList(outSizes[0]);
  final result = Float32List.fromList(
    Float32List.view(ob.buffer, ob.offsetInBytes, outSizes[0] ~/ 4),
  );
  ck('Unlock out[0]', unlock(outBufs[0]));
  calloc.free(addrOut);
  return result;
}
