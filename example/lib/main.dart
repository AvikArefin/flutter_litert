// This example requires bundled assets (the .tflite model, label map, and
// sample images) that live in the example/ directory of the repository.
// If you are copying this file from pub.dev, clone the full repo and run
// the app from the example/ subdirectory so that the assets are available:
//
//   git clone https://github.com/hugocornellier/flutter_litert
//   cd flutter_litert/example
//   flutter run

import 'dart:convert' show utf8;
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  runApp(
    const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: _DetectionDemo(),
    ),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

class _Detection {
  final String label;
  final double score;
  // Normalized [0,1] in source-image space: xmin, ymin, xmax, ymax.
  final List<double> box;

  const _Detection({
    required this.label,
    required this.score,
    required this.box,
  });

  factory _Detection.fromMap(Map m) => _Detection(
    label: m['label'] as String,
    score: (m['score'] as num).toDouble(),
    box: List<double>.from(
      (m['box'] as List).cast<num>().map((v) => v.toDouble()),
    ),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Isolate startup data
// ─────────────────────────────────────────────────────────────────────────────

class _StartupData {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final TransferableTypedData labelsBytes;
  final String performanceModeName;
  final int? numThreads;

  _StartupData({
    required this.sendPort,
    required this.modelBytes,
    required this.labelsBytes,
    required this.performanceModeName,
    required this.numThreads,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// IsolateWorkerBase subclass — wires up the isolate RPC channel
// ─────────────────────────────────────────────────────────────────────────────

class _DetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required Uint8List modelBytes,
    required Uint8List labelsBytes,
    required PerformanceConfig performanceConfig,
  }) async {
    await initWorker(
      (sendPort) => Isolate.spawn(
        _Detector._isolateEntry,
        _StartupData(
          sendPort: sendPort,
          modelBytes: TransferableTypedData.fromList([modelBytes]),
          labelsBytes: TransferableTypedData.fromList([labelsBytes]),
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
        ),
        debugName: 'flutter_litert.detector',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Detector isolate initialization timed out',
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public detector — thin wrapper that delegates to the background isolate
// ─────────────────────────────────────────────────────────────────────────────

class _Detector {
  _DetectorWorker? _worker;

  bool get isReady => _worker?.isReady ?? false;

  Future<void> initialize({
    PerformanceConfig config = const PerformanceConfig(),
  }) async {
    final modelData = await rootBundle.load('assets/efficientdet_lite0.tflite');
    final labelsData = await rootBundle.load('assets/labelmap.txt');
    final worker = _DetectorWorker();
    await worker.initialize(
      modelBytes: modelData.buffer.asUint8List(),
      labelsBytes: labelsData.buffer.asUint8List(),
      performanceConfig: config,
    );
    _worker = worker;
  }

  Future<(List<_Detection>, int)> detect(
    Uint8List imageBytes, {
    double threshold = 0.5,
  }) async {
    if (!isReady) throw StateError('Detector not initialized.');
    final Map<String, dynamic> result = await _worker!
        .sendRequest<Map<String, dynamic>>('detect', {
          'bytes': TransferableTypedData.fromList([imageBytes]),
          'threshold': threshold,
        });
    final detections = (result['detections'] as List)
        .map((m) => _Detection.fromMap(m as Map))
        .toList();
    final inferenceMs = result['inferenceMs'] as int;
    return (detections, inferenceMs);
  }

  Future<void> dispose() async {
    await _worker?.dispose();
    _worker = null;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Isolate entry — all TFLite work runs here, never on the UI thread
  // ───────────────────────────────────────────────────────────────────────────

  @pragma('vm:entry-point')
  static void _isolateEntry(_StartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    Interpreter? interpreter;
    Delegate? delegate;
    TensorFloat32Views? views;
    List<List<double>>? anchors;
    List<String>? labels;
    int inputW = 0, inputH = 0;
    int boxesIdx = 0, classesIdx = 1;

    try {
      final modelBytes = data.modelBytes.materialize().asUint8List();
      final labelsBytes = data.labelsBytes.materialize().asUint8List();

      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );
      final perf = PerformanceConfig(
        mode: performanceMode,
        numThreads: data.numThreads,
      );
      final (options, del) = InterpreterFactory.create(perf);
      delegate = del;

      interpreter = Interpreter.fromBuffer(modelBytes, options: options);
      interpreter.allocateTensors();

      final inputShape = interpreter.getInputTensor(0).shape;
      inputH = inputShape[1];
      inputW = inputShape[2];

      // Discover which output index is boxes (last dim = 4) vs classes (>4).
      final outputs = interpreter.getOutputTensors();
      for (int i = 0; i < outputs.length; i++) {
        final s = outputs[i].shape;
        if (s.length == 3) {
          if (s[2] == 4) boxesIdx = i;
          if (s[2] > 4) classesIdx = i;
        }
      }

      views = TensorFloat32Views.capture(interpreter);
      anchors = _generateAnchors(inputW);
      labels = utf8
          .decode(labelsBytes, allowMalformed: true)
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList(growable: false);

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({'error': 'Detector isolate init failed: $e\n$st'});
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;
      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;
      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            final imgBytes = (message['bytes'] as TransferableTypedData)
                .materialize()
                .asUint8List();
            final double threshold = (message['threshold'] as num).toDouble();

            final cv.Mat src = cv.imdecode(imgBytes, cv.IMREAD_COLOR);
            final int srcW = src.cols, srcH = src.rows;
            try {
              // Letterbox-resize to model input dimensions.
              final lb = computeLetterboxParams(
                srcWidth: srcW,
                srcHeight: srcH,
                targetWidth: inputW,
                targetHeight: inputH,
              );
              final cv.Mat resized = cv.resize(src, (
                lb.newWidth,
                lb.newHeight,
              ), interpolation: cv.INTER_LINEAR);
              final cv.Mat padded = cv.copyMakeBorder(
                resized,
                lb.padTop,
                lb.padBottom,
                lb.padLeft,
                lb.padRight,
                cv.BORDER_CONSTANT,
                value: cv.Scalar.black,
              );
              resized.dispose();

              // BGR→RGB + normalize to [-1, 1] (EfficientDet MediaPipe format).
              final Float32List tensor = bgrBytesToSignedFloat32(
                bytes: padded.data,
                totalPixels: inputW * inputH,
              );
              padded.dispose();

              // Run inference.
              final sw = Stopwatch()..start();
              views!.inputs[0].setAll(0, tensor);
              interpreter!.invoke();
              sw.stop();

              final Float32List boxBuf = views.outputs[boxesIdx];
              final Float32List clsBuf = views.outputs[classesIdx];

              // Decode anchors → raw detections in letterboxed model space.
              final raw = _decodeAnchorsAndScore(
                boxBuf: boxBuf,
                clsBuf: clsBuf,
                anchors: anchors!,
                numClasses: interpreter.getOutputTensor(classesIdx).shape[2],
                threshold: threshold,
              );

              if (raw.isEmpty) {
                mainSendPort.send({
                  'id': id,
                  'result': {
                    'detections': <Map>[],
                    'inferenceMs': sw.elapsedMilliseconds,
                  },
                });
                return;
              }

              // NMS in letterboxed model space.
              final boxes = raw
                  .map((d) => [d.xmin, d.ymin, d.xmax, d.ymax])
                  .toList();
              final scores = raw.map((d) => d.score).toList();
              final kept = weightedNms(
                boxes,
                scores,
                iouThres: 0.45,
                maxDet: 100,
              );

              // Remove letterbox padding and map to source-image [0,1] coords.
              final double pt = lb.padTop / inputH;
              final double pb = lb.padBottom / inputH;
              final double pl = lb.padLeft / inputW;
              final double pr = lb.padRight / inputW;
              final double sx = 1.0 - (pl + pr);
              final double sy = 1.0 - (pt + pb);

              double clamp01(double v) => v.clamp(0.0, 1.0);

              final result = <Map<String, dynamic>>[];
              for (final r in kept) {
                final d = raw[r.index];
                final String name =
                    d.classIdx >= 0 && d.classIdx < labels!.length
                    ? labels[d.classIdx]
                    : '???';
                result.add({
                  'label': name,
                  'score': r.score,
                  'box': [
                    clamp01((r.box[0] - pl) / sx),
                    clamp01((r.box[1] - pt) / sy),
                    clamp01((r.box[2] - pl) / sx),
                    clamp01((r.box[3] - pt) / sy),
                  ],
                });
              }

              mainSendPort.send({
                'id': id,
                'result': {
                  'detections': result,
                  'inferenceMs': sw.elapsedMilliseconds,
                },
              });
            } finally {
              src.dispose();
            }

          case 'dispose':
            interpreter?.close();
            delegate?.delete();
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Helpers (run inside the isolate)
  // ───────────────────────────────────────────────────────────────────────────

  // EfficientDet anchor generator — FPN levels P3-P7, 9 anchors per location.
  static List<List<double>> _generateAnchors(int imageSize) {
    const int minLevel = 3, maxLevel = 7, numScales = 3;
    const List<double> aspectRatios = [1.0, 2.0, 0.5];
    const double anchorScale = 4.0;
    final anchors = <List<double>>[];
    for (int level = minLevel; level <= maxLevel; level++) {
      final int stride = 1 << level;
      final int featureSize = (imageSize / stride).ceil();
      final double baseSize = anchorScale * stride.toDouble();
      for (int y = 0; y < featureSize; y++) {
        for (int x = 0; x < featureSize; x++) {
          final double cy = (y + 0.5) * stride / imageSize;
          final double cx = (x + 0.5) * stride / imageSize;
          for (int s = 0; s < numScales; s++) {
            final double scale = math.pow(2, s / numScales).toDouble();
            for (final aspect in aspectRatios) {
              final double sqA = math.sqrt(aspect);
              final double w = baseSize * scale * sqA / imageSize;
              final double h = baseSize * scale / sqA / imageSize;
              anchors.add([cx, cy, w, h]);
            }
          }
        }
      }
    }
    return anchors;
  }

  static List<_RawDetection> _decodeAnchorsAndScore({
    required Float32List boxBuf,
    required Float32List clsBuf,
    required List<List<double>> anchors,
    required int numClasses,
    required double threshold,
  }) {
    final int n = anchors.length;
    // Pre-compute the minimum logit that can pass sigmoid(threshold).
    final double minLogit = threshold > 0 && threshold < 1
        ? math.log(threshold / (1.0 - threshold))
        : -1e9;

    final out = <_RawDetection>[];
    for (int i = 0; i < n; i++) {
      final int clsBase = i * numClasses;
      // Find the highest-scoring class for this anchor.
      double bestLogit = -double.infinity;
      int bestCls = 0;
      for (int c = 0; c < numClasses; c++) {
        final double v = clsBuf[clsBase + c];
        if (v > bestLogit) {
          bestLogit = v;
          bestCls = c;
        }
      }
      if (bestLogit < minLogit) continue;
      final double score = sigmoid(bestLogit);
      if (score < threshold) continue;

      // Decode box deltas (RetinaNet / EfficientDet [ty, tx, th, tw] format).
      final List<double> a = anchors[i];
      final double cxA = a[0], cyA = a[1], wA = a[2], hA = a[3];
      final int boxBase = i * 4;
      final double cy = boxBuf[boxBase + 0] * hA + cyA;
      final double cx = boxBuf[boxBase + 1] * wA + cxA;
      final double h = math.exp(boxBuf[boxBase + 2]) * hA;
      final double w = math.exp(boxBuf[boxBase + 3]) * wA;

      final double xmin = (cx - w * 0.5).clamp(0.0, 1.0);
      final double ymin = (cy - h * 0.5).clamp(0.0, 1.0);
      final double xmax = (cx + w * 0.5).clamp(0.0, 1.0);
      final double ymax = (cy + h * 0.5).clamp(0.0, 1.0);
      if (xmax - xmin < 1e-3 || ymax - ymin < 1e-3) continue;

      out.add(
        _RawDetection(
          xmin: xmin,
          ymin: ymin,
          xmax: xmax,
          ymax: ymax,
          score: score,
          classIdx: bestCls,
        ),
      );
    }
    return out;
  }
}

class _RawDetection {
  final double xmin, ymin, xmax, ymax, score;
  final int classIdx;
  const _RawDetection({
    required this.xmin,
    required this.ymin,
    required this.xmax,
    required this.ymax,
    required this.score,
    required this.classIdx,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────

const _kSamples = <(String, String)>[
  ('Street', 'assets/samples/street.jpg'),
  ('Cat', 'assets/samples/cat.jpg'),
  ('Dog', 'assets/samples/dog.jpg'),
  ('People', 'assets/samples/people.jpg'),
];

class _DetectionDemo extends StatefulWidget {
  const _DetectionDemo();

  @override
  State<_DetectionDemo> createState() => _DetectionDemoState();
}

class _DetectionDemoState extends State<_DetectionDemo> {
  final _Detector _detector = _Detector();

  ui.Image? _decodedImage;
  List<_Detection> _detections = const [];
  int _inferenceMs = 0;
  bool _busy = false;
  String? _error;
  int _sampleIdx = 0;
  double _threshold = 0.6;

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    setState(() => _busy = true);
    try {
      await _detector.initialize();
      await _runOnSample(_sampleIdx);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _runOnSample(int idx) async {
    if (!_detector.isReady) return;
    setState(() {
      _busy = true;
      _sampleIdx = idx;
      _error = null;
    });
    try {
      final data = await rootBundle.load(_kSamples[idx].$2);
      final bytes = data.buffer.asUint8List();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final (dets, ms) = await _detector.detect(bytes, threshold: _threshold);
      if (!mounted) return;
      setState(() {
        _decodedImage = frame.image;
        _detections = dets;
        _inferenceMs = ms;
      });
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text(
          'flutter_litert · Object Detection',
          style: TextStyle(color: Colors.white),
        ),
        actions: [
          if (_inferenceMs > 0)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              child: Text(
                '${_inferenceMs}ms',
                style: const TextStyle(color: Colors.white70),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // Sample selector
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                for (int i = 0; i < _kSamples.length; i++)
                  Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: ChoiceChip(
                      label: Text(_kSamples[i].$1),
                      selected: _sampleIdx == i,
                      onSelected: _busy ? null : (_) => _runOnSample(i),
                    ),
                  ),
              ],
            ),
          ),
          // Confidence threshold slider
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                const Text(
                  'Confidence',
                  style: TextStyle(color: Colors.white70, fontSize: 12),
                ),
                Expanded(
                  child: Slider(
                    value: _threshold,
                    min: 0.1,
                    max: 0.95,
                    divisions: 17,
                    label: '${(_threshold * 100).round()}%',
                    onChanged: _busy
                        ? null
                        : (v) => setState(() => _threshold = v),
                    onChangeEnd: _busy ? null : (_) => _runOnSample(_sampleIdx),
                  ),
                ),
                Text(
                  '${(_threshold * 100).round()}%',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
          // Image + overlay
          Expanded(
            child: Center(
              child: _busy && _decodedImage == null
                  ? const CircularProgressIndicator(color: Colors.white)
                  : _error != null && _decodedImage == null
                  ? Text(
                      _error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    )
                  : _decodedImage != null
                  ? LayoutBuilder(
                      builder: (context, constraints) {
                        return CustomPaint(
                          size: constraints.biggest,
                          painter: _OverlayPainter(
                            image: _decodedImage!,
                            detections: _detections,
                            busy: _busy,
                          ),
                        );
                      },
                    )
                  : const SizedBox.shrink(),
            ),
          ),
          // Detections list
          if (_detections.isNotEmpty)
            Container(
              color: Colors.black87,
              padding: const EdgeInsets.all(8),
              height: 80,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _detections.length,
                itemBuilder: (context, i) {
                  final d = _detections[i];
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Chip(
                      label: Text(
                        '${d.label} ${(d.score * 100).toStringAsFixed(0)}%',
                        style: const TextStyle(fontSize: 12),
                      ),
                      backgroundColor: Colors.indigo.shade700,
                      labelStyle: const TextStyle(color: Colors.white),
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }
}

class _OverlayPainter extends CustomPainter {
  final ui.Image image;
  final List<_Detection> detections;
  final bool busy;

  const _OverlayPainter({
    required this.image,
    required this.detections,
    required this.busy,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double imgW = image.width.toDouble();
    final double imgH = image.height.toDouble();

    // Fit image into canvas while preserving aspect ratio.
    final double scaleX = size.width / imgW;
    final double scaleY = size.height / imgH;
    final double scale = math.min(scaleX, scaleY);
    final double drawW = imgW * scale;
    final double drawH = imgH * scale;
    final double offsetX = (size.width - drawW) / 2;
    final double offsetY = (size.height - drawH) / 2;

    final Rect dst = Rect.fromLTWH(offsetX, offsetY, drawW, drawH);
    final Rect src = Rect.fromLTWH(0, 0, imgW, imgH);
    canvas.drawImageRect(image, src, dst, Paint());

    if (busy) return;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.cyanAccent;

    final labelBgPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.cyanAccent.withAlpha(200);

    const textStyle = TextStyle(
      color: Colors.black,
      fontSize: 12,
      fontWeight: FontWeight.bold,
    );

    for (final d in detections) {
      // Map normalized [0,1] coords to canvas pixels.
      final double x1 = offsetX + d.box[0] * drawW;
      final double y1 = offsetY + d.box[1] * drawH;
      final double x2 = offsetX + d.box[2] * drawW;
      final double y2 = offsetY + d.box[3] * drawH;

      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      final String labelText =
          '${d.label} ${(d.score * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(text: labelText, style: textStyle),
        textDirection: ui.TextDirection.ltr,
      )..layout();

      final double lx = x1;
      final double ly = y1 - tp.height - 2;
      canvas.drawRect(
        Rect.fromLTWH(lx, ly, tp.width + 4, tp.height + 2),
        labelBgPaint,
      );
      tp.paint(canvas, Offset(lx + 2, ly + 1));
    }
  }

  @override
  bool shouldRepaint(covariant _OverlayPainter old) =>
      old.image != image || old.detections != detections || old.busy != busy;
}
