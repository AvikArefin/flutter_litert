import 'dart:io';
import 'dart:math' as math;

import 'native/interpreter.dart';
import 'native/interpreter_options.dart';
import 'native/delegate.dart';
import 'native/isolate_interpreter.dart';
import 'delegates/xnnpack_delegate_native.dart';
import 'delegates/metal_delegate_native.dart';
import 'delegates/gpu_delegate_native.dart';
import 'delegates/coreml_delegate_native.dart';
import 'performance_config.dart';

class InterpreterFactory {
  static (InterpreterOptions, Delegate?) create(
    PerformanceConfig? config, {
    bool addMediaPipeCustomOps = false,
  }) {
    final options = InterpreterOptions();
    if (addMediaPipeCustomOps) options.addMediaPipeCustomOps();
    final effectiveConfig = config ?? const PerformanceConfig();
    final threadCount =
        effectiveConfig.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);
    options.threads = threadCount;

    if (effectiveConfig.mode == PerformanceMode.disabled) {
      return (options, null);
    }
    if (effectiveConfig.mode == PerformanceMode.auto) {
      return _createAutoMode(options, threadCount);
    }
    if (effectiveConfig.mode == PerformanceMode.xnnpack) {
      return _createXnnpack(options, threadCount);
    }
    if (effectiveConfig.mode == PerformanceMode.gpu) {
      return _createGpu(options, threadCount);
    }
    if (effectiveConfig.mode == PerformanceMode.coreml) {
      return _createCoreml(options, threadCount);
    }
    return (options, null);
  }

  static Future<IsolateInterpreter?> createIsolateIfNeeded(
    Interpreter interpreter,
    Delegate? delegate,
  ) async {
    if (delegate != null) return null;
    return IsolateInterpreter.create(address: interpreter.address);
  }

  static (InterpreterOptions, Delegate?) _createAutoMode(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (Platform.isMacOS || Platform.isLinux) {
      return _createXnnpack(options, threadCount);
    }
    if (Platform.isIOS) {
      return _createGpu(options, threadCount);
    }
    return (options, null);
  }

  static (InterpreterOptions, Delegate?) _createXnnpack(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (!Platform.isMacOS && !Platform.isLinux) {
      return (options, null);
    }
    try {
      final xnnpackDelegate = XNNPackDelegate(
        options: XNNPackDelegateOptions(numThreads: threadCount),
      );
      options.addDelegate(xnnpackDelegate);
      return (options, xnnpackDelegate);
    } catch (_) {
      return (options, null);
    }
  }

  static (InterpreterOptions, Delegate?) _createGpu(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (!Platform.isIOS && !Platform.isMacOS && !Platform.isAndroid) {
      return (options, null);
    }
    try {
      final gpuDelegate = (Platform.isIOS || Platform.isMacOS)
          ? GpuDelegate()
          : GpuDelegateV2() as Delegate;
      options.addDelegate(gpuDelegate);
      return (options, gpuDelegate);
    } catch (_) {
      return (options, null);
    }
  }

  static (InterpreterOptions, Delegate?) _createCoreml(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (!Platform.isIOS && !Platform.isMacOS) {
      return (options, null);
    }
    try {
      final coremlDelegate = CoreMlDelegate(
        options: CoreMlDelegateOptions(enabledDevices: 1),
      );
      options.addDelegate(coremlDelegate);
      return (options, coremlDelegate);
    } catch (_) {
      return (options, null);
    }
  }
}
