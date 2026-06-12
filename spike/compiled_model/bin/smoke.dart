// Standalone LiteRT Next CompiledModel smoke test (macOS arm64).
// Proves: the prebuilt libLiteRt.dylib loads via Dart FFI, its symbols resolve,
// and a CompiledModel can be created from a real .tflite on the CPU accelerator.
// No tensor I/O yet — that's the next milestone.
import 'dart:ffi';
import 'package:ffi/ffi.dart';

const int kOk = 0;
const int kCpu = 1; // kLiteRtHwAcceleratorCpu = 1 << 0

typedef _CreateEnvC =
    Int32 Function(Int32, Pointer<Void>, Pointer<Pointer<Void>>);
typedef _CreateEnv = int Function(int, Pointer<Void>, Pointer<Pointer<Void>>);

typedef _CreateOptsC = Int32 Function(Pointer<Pointer<Void>>);
typedef _CreateOpts = int Function(Pointer<Pointer<Void>>);

typedef _SetAccelC = Int32 Function(Pointer<Void>, Int32);
typedef _SetAccel = int Function(Pointer<Void>, int);

typedef _ModelFromFileC = Int32 Function(Pointer<Utf8>, Pointer<Pointer<Void>>);
typedef _ModelFromFile = int Function(Pointer<Utf8>, Pointer<Pointer<Void>>);

typedef _CreateCMC =
    Int32 Function(
      Pointer<Void>,
      Pointer<Void>,
      Pointer<Void>,
      Pointer<Pointer<Void>>,
    );
typedef _CreateCM =
    int Function(
      Pointer<Void>,
      Pointer<Void>,
      Pointer<Void>,
      Pointer<Pointer<Void>>,
    );

typedef _DestroyC = Void Function(Pointer<Void>);
typedef _Destroy = void Function(Pointer<Void>);

void main(List<String> args) {
  final modelPath =
      args.isNotEmpty
          ? args[0]
          : '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/simple_model.tflite';

  final lib = DynamicLibrary.open('/tmp/cm_spike/libLiteRt.dylib');
  print('✓ libLiteRt.dylib loaded');

  final createEnv = lib.lookupFunction<_CreateEnvC, _CreateEnv>(
    'LiteRtCreateEnvironment',
  );
  final createOpts = lib.lookupFunction<_CreateOptsC, _CreateOpts>(
    'LiteRtCreateOptions',
  );
  final setAccel = lib.lookupFunction<_SetAccelC, _SetAccel>(
    'LiteRtSetOptionsHardwareAccelerators',
  );
  final modelFromFile = lib.lookupFunction<_ModelFromFileC, _ModelFromFile>(
    'LiteRtCreateModelFromFile',
  );
  final createCM = lib.lookupFunction<_CreateCMC, _CreateCM>(
    'LiteRtCreateCompiledModel',
  );
  final destroyCM = lib.lookupFunction<_DestroyC, _Destroy>(
    'LiteRtDestroyCompiledModel',
  );
  final destroyModel = lib.lookupFunction<_DestroyC, _Destroy>(
    'LiteRtDestroyModel',
  );
  final destroyOpts = lib.lookupFunction<_DestroyC, _Destroy>(
    'LiteRtDestroyOptions',
  );
  final destroyEnv = lib.lookupFunction<_DestroyC, _Destroy>(
    'LiteRtDestroyEnvironment',
  );
  print('✓ all FFI symbols resolved');

  final envOut = calloc<Pointer<Void>>();
  var st = createEnv(0, nullptr, envOut);
  _check('LiteRtCreateEnvironment', st);
  final env = envOut.value;

  final optsOut = calloc<Pointer<Void>>();
  st = createOpts(optsOut);
  _check('LiteRtCreateOptions', st);
  final opts = optsOut.value;
  st = setAccel(opts, kCpu);
  _check('LiteRtSetOptionsHardwareAccelerators(CPU)', st);

  final namePtr = modelPath.toNativeUtf8();
  final modelOut = calloc<Pointer<Void>>();
  st = modelFromFile(namePtr, modelOut);
  _check('LiteRtCreateModelFromFile', st);
  final model = modelOut.value;
  print('✓ model loaded: $modelPath');

  final cmOut = calloc<Pointer<Void>>();
  st = createCM(env, model, opts, cmOut);
  _check('LiteRtCreateCompiledModel(CPU)', st);
  print('✓ CompiledModel created on CPU accelerator');

  destroyCM(cmOut.value);
  destroyModel(model);
  destroyOpts(opts);
  destroyEnv(env);
  calloc.free(envOut);
  calloc.free(optsOut);
  calloc.free(modelOut);
  calloc.free(cmOut);
  calloc.free(namePtr);
  print(
    '\n🎉 SMOKE TEST PASSED — CompiledModel works end-to-end via FFI on macOS.',
  );
}

void _check(String what, int status) {
  if (status != kOk) {
    print('✗ $what failed with LiteRtStatus=$status');
    throw StateError('$what returned $status');
  }
  print('  · $what → ok');
}
