import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

const int kOk = 0;
const int kCpu = 1;
const int kGpu = 2;
const int kGpuCpu = 3;
const int kLockWrite = 1;
const int kLiteRtElementTypeFloat32 = 1;
const int kRankedTensorTypeLayoutOffset = 4;
const int kLiteRtLayoutSize = 68;
const int kLiteRtRankedTensorTypeSize = 72;
const int kWarmup = 40;
const int kTimed = 40;
const int kStdoutFd = 1;
const int kStderrFd = 2;
const int kOpenWriteOnly = 1;
const String kCellArg = '--cell';

const List<String> kDefaultModels = [
  '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/mobilefacenet.tflite',
  '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/superanimal_rtmpose_s_float16.tflite',
  '/Users/hugocornellier/IdeaProjects/flutter_litert/example/assets/species_classifier_float16.tflite',
  '/Users/hugocornellier/IdeaProjects/animal_detection/assets/models/superanimal_ssdlite_float16.tflite',
  '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/face_detection_full_range_sparse.tflite',
];

const List<(String, int)> kMasks = [
  ('CPU(1)', kCpu),
  ('GPU(2)', kGpu),
  ('GPU|CPU(3)', kGpuCpu),
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

final class _Row {
  const _Row({
    required this.model,
    required this.mask,
    required this.compiles,
    required this.ran,
    required this.result,
  });

  final String model;
  final String mask;
  final bool compiles;
  final bool ran;
  final String result;
}

final class _Cell {
  const _Cell({
    required this.compiles,
    required this.ran,
    required this.result,
  });

  final bool compiles;
  final bool ran;
  final String result;

  String toLine() => [compiles ? 'Y' : 'N', ran ? 'Y' : 'N', result].join('\t');
}

final class _DtypeException implements Exception {
  const _DtypeException();
}

final class _LiteRtStatusException implements Exception {
  const _LiteRtStatusException(this.op, this.status, this.afterCompile);

  final String op;
  final int status;
  final bool afterCompile;
}

final class _PosixApi {
  _PosixApi(DynamicLibrary lib)
    : open = lib.lookupFunction<
        Int32 Function(Pointer<Utf8>, Int32),
        int Function(Pointer<Utf8>, int)
      >('open'),
      dup = lib.lookupFunction<Int32 Function(Int32), int Function(int)>('dup'),
      dup2 = lib
          .lookupFunction<Int32 Function(Int32, Int32), int Function(int, int)>(
            'dup2',
          ),
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

final class _LiteRtApi {
  _LiteRtApi(DynamicLibrary lib)
    : createEnv = lib.lookupFunction<
        Int32 Function(Int32, _P, _PP),
        int Function(int, _P, _PP)
      >('LiteRtCreateEnvironment'),
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
      modelFromFile = lib.lookupFunction<
        Int32 Function(Pointer<Utf8>, _PP),
        int Function(Pointer<Utf8>, _PP)
      >('LiteRtCreateModelFromFile'),
      destroyModel = lib.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyModel',
      ),
      createCM = lib.lookupFunction<
        Int32 Function(_P, _P, _P, _PP),
        int Function(_P, _P, _P, _PP)
      >('LiteRtCreateCompiledModel'),
      destroyCM = lib.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyCompiledModel',
      ),
      getSig = lib.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetModelSignature'),
      numIn = lib.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureInputs'),
      numOut = lib.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetNumSignatureOutputs'),
      inTensor = lib.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureInputTensorByIndex'),
      outTensor = lib.lookupFunction<
        Int32 Function(_P, IntPtr, _PP),
        int Function(_P, int, _PP)
      >('LiteRtGetSignatureOutputTensorByIndex'),
      rankedType = lib.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>),
        int Function(_P, Pointer<LiteRtRankedTensorType>)
      >('LiteRtGetRankedTensorType'),
      inReq = lib.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelInputBufferRequirements'),
      outReq = lib.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP),
        int Function(_P, int, int, _PP)
      >('LiteRtGetCompiledModelOutputBufferRequirements'),
      reqSize = lib.lookupFunction<
        Int32 Function(_P, Pointer<IntPtr>),
        int Function(_P, Pointer<IntPtr>)
      >('LiteRtGetTensorBufferRequirementsBufferSize'),
      createBuf = lib.lookupFunction<
        Int32 Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP),
        int Function(_P, Pointer<LiteRtRankedTensorType>, _P, _PP)
      >('LiteRtCreateManagedTensorBufferFromRequirements'),
      destroyBuf = lib.lookupFunction<Void Function(_P), void Function(_P)>(
        'LiteRtDestroyTensorBuffer',
      ),
      lock = lib.lookupFunction<
        Int32 Function(_P, _PP, Int32),
        int Function(_P, _PP, int)
      >('LiteRtLockTensorBuffer'),
      unlock = lib.lookupFunction<Int32 Function(_P), int Function(_P)>(
        'LiteRtUnlockTensorBuffer',
      ),
      getInLayout = lib.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>),
        int Function(_P, int, int, Pointer<LiteRtLayout>)
      >('LiteRtGetCompiledModelInputTensorLayout'),
      getOutLayouts = lib.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, Pointer<LiteRtLayout>, Uint8),
        int Function(_P, int, int, Pointer<LiteRtLayout>, int)
      >('LiteRtGetCompiledModelOutputTensorLayouts'),
      runSync = lib.lookupFunction<
        Int32 Function(_P, IntPtr, IntPtr, _PP, IntPtr, _PP),
        int Function(_P, int, int, _PP, int, _PP)
      >('LiteRtRunCompiledModel');

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
}

final class _CompiledSession {
  _CompiledSession(this.api, String path, this.mask) {
    final envOut = calloc<Pointer<Void>>();
    final optsOut = calloc<Pointer<Void>>();
    final modelOut = calloc<Pointer<Void>>();
    final cmOut = calloc<Pointer<Void>>();
    final sigOut = calloc<Pointer<Void>>();
    final nInP = calloc<IntPtr>();
    final nOutP = calloc<IntPtr>();
    final pathPtr = path.toNativeUtf8();
    try {
      _ckRt('LiteRtCreateEnvironment', api.createEnv(0, nullptr, envOut));
      env = envOut.value;
      _ckRt('LiteRtCreateOptions', api.createOpts(optsOut));
      opts = optsOut.value;
      _ckRt('LiteRtSetOptionsHardwareAccelerators', api.setAccel(opts, mask));
      _ckRt('LiteRtCreateModelFromFile', api.modelFromFile(pathPtr, modelOut));
      model = modelOut.value;
      _ckRt('LiteRtGetModelSignature', api.getSig(model, 0, sigOut));
      sig = sigOut.value;
      _ckRt('LiteRtGetNumSignatureInputs', api.numIn(sig, nInP));
      _ckRt('LiteRtGetNumSignatureOutputs', api.numOut(sig, nOutP));
      nIn = nInP.value;
      nOut = nOutP.value;
      _ckRt('LiteRtCreateCompiledModel', api.createCM(env, model, opts, cmOut));
      cm = cmOut.value;

      inBufs = calloc<Pointer<Void>>(nIn);
      inputByteSizes = List<int>.filled(nIn, 0);
      for (var i = 0; i < nIn; i++) {
        inBufs[i] = _createInputBuffer(i);
        _writeInput(i);
      }

      outLayouts = calloc<LiteRtLayout>(nOut);
      _ckRt(
        'LiteRtGetCompiledModelOutputTensorLayouts',
        api.getOutLayouts(cm, 0, nOut, outLayouts, 1),
        afterCompile: true,
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
  final int mask;
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
    for (var i = 0; i < warmup; i++) {
      _runSync(singleOutSet);
    }
    final samples = Float64List(iters);
    for (var i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      _runSync(singleOutSet);
      sw.stop();
      samples[i] = sw.elapsedTicks * 1000000 / sw.frequency;
    }
    samples.sort();
    return samples[iters ~/ 2];
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

  _P _createInputBuffer(int index) {
    final tensorOut = calloc<Pointer<Void>>();
    final tensorType = calloc<LiteRtRankedTensorType>();
    final reqOut = calloc<Pointer<Void>>();
    final sizeOut = calloc<IntPtr>();
    final bufOut = calloc<Pointer<Void>>();
    try {
      _ckRt(
        'LiteRtGetSignatureInputTensorByIndex',
        api.inTensor(sig, index, tensorOut),
        afterCompile: true,
      );
      _ckRt(
        'LiteRtGetRankedTensorType',
        api.rankedType(tensorOut.value, tensorType),
        afterCompile: true,
      );
      if (tensorType.ref.elementType != kLiteRtElementTypeFloat32) {
        throw const _DtypeException();
      }
      _ckRt(
        'LiteRtGetCompiledModelInputTensorLayout',
        api.getInLayout(
          cm,
          0,
          index,
          (tensorType.cast<Uint8>() + kRankedTensorTypeLayoutOffset)
              .cast<LiteRtLayout>(),
        ),
        afterCompile: true,
      );
      _ckRt(
        'LiteRtGetCompiledModelInputBufferRequirements',
        api.inReq(cm, 0, index, reqOut),
        afterCompile: true,
      );
      _ckRt(
        'LiteRtGetTensorBufferRequirementsBufferSize',
        api.reqSize(reqOut.value, sizeOut),
        afterCompile: true,
      );
      inputByteSizes[index] = sizeOut.value;
      _ckRt(
        'LiteRtCreateManagedTensorBufferFromRequirements',
        api.createBuf(env, tensorType, reqOut.value, bufOut),
        afterCompile: true,
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
      _ckRt(
        'LiteRtGetSignatureOutputTensorByIndex',
        api.outTensor(sig, index, tensorOut),
        afterCompile: true,
      );
      _ckRt(
        'LiteRtGetRankedTensorType',
        api.rankedType(tensorOut.value, tensorType),
        afterCompile: true,
      );
      (tensorType.cast<Uint8>() + kRankedTensorTypeLayoutOffset)
          .asTypedList(kLiteRtLayoutSize)
          .setAll(
            0,
            (outLayouts + index).cast<Uint8>().asTypedList(kLiteRtLayoutSize),
          );
      _ckRt(
        'LiteRtGetCompiledModelOutputBufferRequirements',
        api.outReq(cm, 0, index, reqOut),
        afterCompile: true,
      );
      _ckRt(
        'LiteRtCreateManagedTensorBufferFromRequirements',
        api.createBuf(env, tensorType, reqOut.value, bufOut),
        afterCompile: true,
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
      _ckRt(
        'LiteRtLockTensorBuffer',
        api.lock(inBufs[index], addrOut, kLockWrite),
        afterCompile: true,
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
        floats[i] = (i % 257) / 256.0;
      }
    } finally {
      if (locked) {
        _ckRt(
          'LiteRtUnlockTensorBuffer',
          api.unlock(inBufs[index]),
          afterCompile: true,
        );
      }
      calloc.free(addrOut);
    }
  }

  void _runSync(_PP outSet) {
    _ckRt(
      'LiteRtRunCompiledModel',
      api.runSync(cm, 0, nIn, inBufs, nOut, outSet),
      afterCompile: true,
    );
  }
}

void main(List<String> args) {
  if (args.isNotEmpty && args.first == kCellArg) {
    _runChildCell(args);
    return;
  }

  _checkStructSizes();
  final models = args.isEmpty ? kDefaultModels : args;
  final rows = <_Row>[];

  for (final model in models) {
    for (final (maskName, mask) in kMasks) {
      final cell =
          File(model).existsSync()
              ? _measureChildCell(model, mask)
              : const _Cell(
                compiles: false,
                ran: false,
                result: 'ERR(missing-model)',
              );
      rows.add(
        _Row(
          model: _basename(model),
          mask: maskName,
          compiles: cell.compiles,
          ran: cell.ran,
          result: cell.result,
        ),
      );
    }
  }

  _printTable(rows);
}

void _runChildCell(List<String> args) {
  if (args.length != 3) {
    print(
      const _Cell(compiles: false, ran: false, result: 'ERR(args)').toLine(),
    );
    return;
  }

  _checkStructSizes();
  final path = args[1];
  final mask = int.tryParse(args[2]);
  if (mask == null) {
    print(
      const _Cell(compiles: false, ran: false, result: 'ERR(mask)').toLine(),
    );
    return;
  }

  final posix = _PosixApi(DynamicLibrary.process());
  final cell = _withNativeOutputMuted(posix, () => _measureCell(path, mask));
  print(cell.toLine());
}

_Cell _measureChildCell(String path, int mask) {
  final scriptPath = Platform.script.toFilePath();
  final result = Process.runSync(Platform.resolvedExecutable, [
    scriptPath,
    kCellArg,
    path,
    '$mask',
  ], workingDirectory: Directory.current.path);
  if (result.exitCode != 0) {
    return _Cell(
      compiles: false,
      ran: false,
      result: 'ERR(crash:${result.exitCode})',
    );
  }

  final output = (result.stdout as String).trim();
  if (output.isEmpty) {
    return const _Cell(compiles: false, ran: false, result: 'ERR(child)');
  }
  final parts = output.split('\n').last.trim().split('\t');
  if (parts.length != 3) {
    return const _Cell(compiles: false, ran: false, result: 'ERR(child)');
  }
  return _Cell(
    compiles: parts[0] == 'Y',
    ran: parts[1] == 'Y',
    result: parts[2],
  );
}

_Cell _measureCell(String path, int mask) {
  _CompiledSession? session;
  try {
    final api = _LiteRtApi(DynamicLibrary.open('./libLiteRt.dylib'));
    session = _CompiledSession(api, path, mask);
    final median = session.measureSyncMedian(warmup: kWarmup, iters: kTimed);
    return _Cell(compiles: true, ran: true, result: median.toStringAsFixed(1));
  } on _DtypeException {
    return const _Cell(
      compiles: true,
      ran: false,
      result: 'ERR(non-float32-input)',
    );
  } on _LiteRtStatusException catch (e) {
    return _Cell(
      compiles: e.afterCompile,
      ran: false,
      result:
          e.afterCompile
              ? 'ERR(${e.op}:LiteRtStatus=${e.status})'
              : 'LiteRtStatus=${e.status}',
    );
  } catch (e) {
    return _Cell(
      compiles: session != null,
      ran: false,
      result: 'ERR(${_shortReason(e)})',
    );
  } finally {
    session?.close();
  }
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

void _printTable(List<_Row> rows) {
  const headers = ['model', 'mask', 'compiles', 'ran', 'median_us_or_status'];
  final widths = List<int>.from(headers.map((h) => h.length));
  for (final row in rows) {
    final values = _rowValues(row);
    for (var i = 0; i < values.length; i++) {
      if (values[i].length > widths[i]) widths[i] = values[i].length;
    }
  }

  print(_formatRow(headers, widths));
  print(widths.map((w) => '-' * w).join(' | '));
  for (final row in rows) {
    print(_formatRow(_rowValues(row), widths));
  }
}

List<String> _rowValues(_Row row) => [
  row.model,
  row.mask,
  row.compiles ? 'yes' : 'no',
  row.ran ? 'yes' : 'no',
  row.result,
];

String _formatRow(List<String> values, List<int> widths) {
  return List<String>.generate(values.length, (i) {
    final value = values[i];
    return i == 0 ? value.padRight(widths[i]) : value.padLeft(widths[i]);
  }).join(' | ');
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  return normalized.substring(normalized.lastIndexOf('/') + 1);
}

String _shortReason(Object error) {
  final text = error.toString();
  final cleaned = text
      .replaceFirst('Bad state: ', '')
      .replaceFirst('Invalid argument(s): ', '')
      .replaceAll(RegExp(r'[^A-Za-z0-9_]+'), '_')
      .replaceAll(RegExp(r'_+'), '_')
      .replaceAll(RegExp(r'^_|_$'), '');
  if (cleaned.isEmpty) return 'error';
  return cleaned.length <= 40 ? cleaned : cleaned.substring(0, 40);
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

void _ckRt(String op, int status, {bool afterCompile = false}) {
  if (status != kOk) {
    throw _LiteRtStatusException(op, status, afterCompile);
  }
}
