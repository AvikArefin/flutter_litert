// Public API benchmark: CompiledModel managed buffers vs official host-memory
// buffers using runAsync().
//
// Run from repo root, for example:
//   dart run spike/compiled_model/bin/buffer_mode_bench.dart \
//     /path/to/model.tflite
// ignore_for_file: avoid_print, depend_on_referenced_packages
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/src/compiled_model/compiled_model.dart';

void main(List<String> args) async {
  final model =
      args.isNotEmpty
          ? args[0]
          : 'example/assets/superanimal_rtmpose_s_float16.tflite';
  const warmup = 30;
  const iterations = 150;
  const repeats = 3;

  print('model: ${model.split('/').last}');
  print('path: $model');
  print('$repeats repeats, $iterations measured iterations, $warmup warmup');

  for (final config in _configs) {
    print('\n=== ${config.name} ===');
    final byPath = <_Path, List<_BenchResult>>{
      for (final path in _paths) path: <_BenchResult>[],
    };
    final failures = <_Path, Object>{};

    for (var repeat = 0; repeat < repeats; repeat++) {
      print('-- repeat ${repeat + 1} --');
      _BenchResult? reference;
      for (final path in _rotatedPaths(repeat)) {
        try {
          final result = await _bench(
            model,
            accelerators: config.accelerators,
            path: path,
            warmup: warmup,
            iterations: iterations,
          );
          final diff =
              reference == null
                  ? null
                  : _maxDiff(reference.lastOutputs, result.lastOutputs);
          reference ??= result;
          byPath[path]!.add(result);
          print(
            '${path.name.padRight(14)} median '
            '${result.medianUs.toStringAsFixed(1).padLeft(8)} us, '
            'mean ${result.meanUs.toStringAsFixed(1).padLeft(8)} us'
            '${diff == null ? '' : ', max|diff|=${diff.toStringAsExponential(2)}'}',
          );
        } catch (e) {
          failures[path] = e;
          print('${path.name.padRight(14)} FAILED $e');
        }
      }
    }

    _printSummary(byPath, failures);
  }
}

const _configs = <_Config>[
  _Config('CPU', {Accelerator.cpu}),
  _Config('GPU strict', {Accelerator.gpu}),
  _Config('GPU|CPU', {Accelerator.gpu, Accelerator.cpu}),
];

const _paths = <_Path>[
  _Path('managed.run', TensorBufferMode.managed, false),
  _Path('host.run', TensorBufferMode.hostMemory, false),
  _Path('host.direct', TensorBufferMode.hostMemory, true),
];

Iterable<_Path> _rotatedPaths(int repeat) sync* {
  for (var i = 0; i < _paths.length; i++) {
    yield _paths[(i + repeat) % _paths.length];
  }
}

final class _Config {
  const _Config(this.name, this.accelerators);

  final String name;
  final Set<Accelerator> accelerators;
}

final class _Path {
  const _Path(this.name, this.tensorBufferMode, this.direct);

  final String name;
  final TensorBufferMode tensorBufferMode;
  final bool direct;
}

final class _BenchResult {
  const _BenchResult(this.medianUs, this.meanUs, this.lastOutputs);

  final double medianUs;
  final double meanUs;
  final List<Float32List> lastOutputs;
}

Future<_BenchResult> _bench(
  String model, {
  required Set<Accelerator> accelerators,
  required _Path path,
  required int warmup,
  required int iterations,
}) async {
  final cm = CompiledModel.fromFile(
    model,
    accelerators: accelerators,
    tensorBufferMode: path.tensorBufferMode,
  );
  try {
    final inputs = List<Float32List>.generate(cm.inputCount, (inputIndex) {
      final input = Float32List(cm.inputByteSizes[inputIndex] ~/ 4);
      for (var i = 0; i < input.length; i++) {
        input[i] = ((i + inputIndex) % 255) / 255.0 - 0.5;
      }
      return input;
    });

    for (var i = 0; i < warmup; i++) {
      await _runAndSum(cm, inputs, direct: path.direct);
    }

    final timings = <int>[];
    var accumulator = 0.0;
    for (var i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      final sum = await _runAndSum(cm, inputs, direct: path.direct);
      sw.stop();
      accumulator += sum;
      timings.add(sw.elapsedMicroseconds);
    }
    if (accumulator.isNaN) {
      throw StateError('NaN output accumulator');
    }
    final outputs = await _captureOutputs(cm, inputs, direct: path.direct);
    timings.sort();
    final mean = timings.reduce((a, b) => a + b) / timings.length;
    return _BenchResult(timings[timings.length ~/ 2].toDouble(), mean, outputs);
  } finally {
    cm.close();
  }
}

Future<double> _runAndSum(
  CompiledModel cm,
  List<Float32List> inputs, {
  required bool direct,
}) async {
  if (!direct) {
    final outputs = await cm.runAsync(inputs);
    return _sumOutputs(outputs);
  }

  for (var i = 0; i < inputs.length; i++) {
    cm.writeInput(i, (view) => view.setAll(0, inputs[i]));
  }
  await cm.dispatchAsync();

  var sum = 0.0;
  for (var i = 0; i < cm.outputCount; i++) {
    cm.readOutput(i, (view) {
      for (var j = 0; j < view.length; j++) {
        sum += view[j];
      }
    });
  }
  return sum;
}

Future<List<Float32List>> _captureOutputs(
  CompiledModel cm,
  List<Float32List> inputs, {
  required bool direct,
}) async {
  if (!direct) {
    return cm.runAsync(inputs);
  }

  for (var i = 0; i < inputs.length; i++) {
    cm.writeInput(i, (view) => view.setAll(0, inputs[i]));
  }
  await cm.dispatchAsync();

  return List<Float32List>.generate(
    cm.outputCount,
    (i) => cm.readOutput(i, Float32List.fromList),
  );
}

double _sumOutputs(List<Float32List> outputs) {
  var sum = 0.0;
  for (final output in outputs) {
    for (var i = 0; i < output.length; i++) {
      sum += output[i];
    }
  }
  return sum;
}

double _maxDiff(List<Float32List> a, List<Float32List> b) {
  if (a.length != b.length) return double.infinity;
  var maxDiff = 0.0;
  for (var i = 0; i < a.length; i++) {
    if (a[i].length != b[i].length) return double.infinity;
    for (var j = 0; j < a[i].length; j++) {
      maxDiff = math.max(maxDiff, (a[i][j] - b[i][j]).abs());
    }
  }
  return maxDiff;
}

void _printSummary(
  Map<_Path, List<_BenchResult>> byPath,
  Map<_Path, Object> failures,
) {
  print('summary:');
  final summaries = <_Path, _Summary>{};
  for (final path in _paths) {
    final results = byPath[path]!;
    if (results.isEmpty) {
      print('  ${path.name.padRight(14)} FAILED ${failures[path]}');
      continue;
    }
    final summary = _summarize(results);
    summaries[path] = summary;
    print(
      '  ${path.name.padRight(14)} median-of-medians '
      '${summary.medianOfMediansUs.toStringAsFixed(1).padLeft(8)} us, '
      'mean-of-means ${summary.meanOfMeansUs.toStringAsFixed(1).padLeft(8)} us',
    );
  }

  final managed = summaries[_paths[0]];
  final hostRun = summaries[_paths[1]];
  final hostDirect = summaries[_paths[2]];
  if (hostDirect != null) {
    if (managed != null) {
      print(
        '  host.direct vs managed.run: '
        '${_speedup(managed, hostDirect).toStringAsFixed(3)}x '
        '(median-of-medians)',
      );
    }
    if (hostRun != null) {
      print(
        '  host.direct vs host.run:    '
        '${_speedup(hostRun, hostDirect).toStringAsFixed(3)}x '
        '(median-of-medians)',
      );
    }
  }
}

_Summary _summarize(List<_BenchResult> results) {
  final medians = results.map((r) => r.medianUs).toList()..sort();
  final meanOfMeans =
      results.map((r) => r.meanUs).reduce((a, b) => a + b) / results.length;
  return _Summary(medians[medians.length ~/ 2], meanOfMeans);
}

double _speedup(_Summary before, _Summary after) =>
    before.medianOfMediansUs / after.medianOfMediansUs;

final class _Summary {
  const _Summary(this.medianOfMediansUs, this.meanOfMeansUs);

  final double medianOfMediansUs;
  final double meanOfMeansUs;
}
