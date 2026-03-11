enum PerformanceMode { disabled, xnnpack, gpu, coreml, auto }

class PerformanceConfig {
  final PerformanceMode mode;
  final int? numThreads;

  const PerformanceConfig({this.mode = PerformanceMode.auto, this.numThreads});

  const PerformanceConfig.xnnpack({this.numThreads})
    : mode = PerformanceMode.xnnpack;

  const PerformanceConfig.gpu({this.numThreads}) : mode = PerformanceMode.gpu;

  const PerformanceConfig.coreml({this.numThreads})
    : mode = PerformanceMode.coreml;

  const PerformanceConfig.auto({this.numThreads}) : mode = PerformanceMode.auto;

  static const PerformanceConfig disabled = PerformanceConfig(
    mode: PerformanceMode.disabled,
  );
}
