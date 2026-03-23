/// Hardware acceleration mode for LiteRT inference.
enum PerformanceMode {
  /// No hardware acceleration.
  disabled,

  /// XNNPACK delegate (CPU-optimized, macOS/Linux).
  xnnpack,

  /// GPU delegate (iOS Metal / Android OpenGL).
  gpu,

  /// CoreML delegate (iOS/macOS only).
  coreml,

  /// Automatically select the best delegate for the current platform.
  auto,
}

/// Configuration for interpreter hardware acceleration and threading.
class PerformanceConfig {
  /// The hardware acceleration mode.
  final PerformanceMode mode;

  /// Number of threads for inference. Defaults to min(4, CPU count) if null.
  final int? numThreads;

  /// Creates a performance config with the given [mode] and optional [numThreads].
  const PerformanceConfig({this.mode = PerformanceMode.auto, this.numThreads});

  /// Creates an XNNPACK-accelerated config.
  const PerformanceConfig.xnnpack({this.numThreads})
    : mode = PerformanceMode.xnnpack;

  /// Creates a GPU-accelerated config.
  const PerformanceConfig.gpu({this.numThreads}) : mode = PerformanceMode.gpu;

  /// Creates a CoreML-accelerated config.
  const PerformanceConfig.coreml({this.numThreads})
    : mode = PerformanceMode.coreml;

  /// Creates a config that auto-selects the best delegate.
  const PerformanceConfig.auto({this.numThreads}) : mode = PerformanceMode.auto;

  /// A config with no hardware acceleration.
  static const PerformanceConfig disabled = PerformanceConfig(
    mode: PerformanceMode.disabled,
  );
}
