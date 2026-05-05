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

/// Factory for creating interpreter options with the package's platform
/// delegate mapping.
class InterpreterFactory {
  /// Creates [InterpreterOptions] and an optional [Delegate] based on [config].
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
      return _createGpu(options);
    }
    if (effectiveConfig.mode == PerformanceMode.coreml) {
      return _createCoreml(options);
    }
    return (options, null);
  }

  /// Creates an [IsolateInterpreter] if conditions allow isolate-based inference.
  ///
  /// Returns null if [useIsolateInterpreter] is false, a delegate is active,
  /// or the platform is macOS (where isolate sharing is unstable).
  static Future<IsolateInterpreter?> createIsolateIfNeeded(
    Interpreter interpreter,
    Delegate? delegate, {
    bool useIsolateInterpreter = true,
  }) async {
    if (!useIsolateInterpreter) return null;
    if (delegate != null) return null;
    // Sharing a native interpreter across isolates is unstable on macOS.
    if (Platform.isMacOS) return null;
    return IsolateInterpreter.create(address: interpreter.address);
  }

  static (InterpreterOptions, Delegate?) _createAutoMode(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (Platform.isIOS) {
      return _createGpu(options);
    }
    // Android, macOS, Linux, Windows, all use XNNPACK.
    return _createXnnpack(options, threadCount);
  }

  static (InterpreterOptions, Delegate?) _createXnnpack(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (!Platform.isAndroid &&
        !Platform.isIOS &&
        !Platform.isMacOS &&
        !Platform.isLinux &&
        !Platform.isWindows) {
      return (options, null);
    }
    try {
      final xnnOpts = XNNPackDelegateOptions(numThreads: threadCount);
      final xnnpackDelegate = XNNPackDelegate(options: xnnOpts);
      xnnOpts.delete();
      options.addDelegate(xnnpackDelegate);
      return (options, xnnpackDelegate);
    } catch (_) {
      return (options, null);
    }
  }

  static (InterpreterOptions, Delegate?) _createGpu(
    InterpreterOptions options,
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
  ) {
    if (!Platform.isIOS && !Platform.isMacOS) {
      return (options, null);
    }
    try {
      final coreOpts = CoreMlDelegateOptions(enabledDevices: 1);
      final coremlDelegate = CoreMlDelegate(options: coreOpts);
      coreOpts.delete();
      options.addDelegate(coremlDelegate);
      return (options, coremlDelegate);
    } catch (_) {
      return (options, null);
    }
  }
}
