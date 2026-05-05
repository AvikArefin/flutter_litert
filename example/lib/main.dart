import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  runApp(const _App());
}

class _App extends StatelessWidget {
  const _App();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'flutter_litert · Object Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const _DetectionDemo(),
    );
  }
}

class _DetectionDemo extends StatefulWidget {
  const _DetectionDemo();

  @override
  State<_DetectionDemo> createState() => _DetectionDemoState();
}

class _DetectionDemoState extends State<_DetectionDemo> {
  ObjectDetector? _detector;
  ObjectDetectionModel _model = ObjectDetectionModel.efficientDetLite0;
  double _threshold = 0.63;
  int _maxResults = 10;

  Uint8List? _imageBytes;
  Size? _imageSize;
  List<DetectedObject> _detections = const [];
  int _inferenceMs = 0;
  bool _busy = false;
  String? _error;

  static const _samples = <(String, String)>[
    ('Street', 'assets/samples/street.jpg'),
    ('Cat', 'assets/samples/cat.jpg'),
    ('Dog', 'assets/samples/dog.jpg'),
    ('People', 'assets/samples/people.jpg'),
  ];

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    setState(() => _busy = true);
    try {
      final d = await ObjectDetector.create(model: _model);
      if (!mounted) return;
      setState(() {
        _detector = d;
        _busy = false;
      });
      await _loadSample(_samples.first.$2);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Init failed: $e';
        _busy = false;
      });
    }
  }

  Future<void> _switchModel(ObjectDetectionModel m) async {
    final old = _detector;
    setState(() {
      _detector = null;
      _busy = true;
      _model = m;
    });
    await old?.dispose();
    try {
      final d = await ObjectDetector.create(model: m);
      if (!mounted) return;
      setState(() {
        _detector = d;
        _busy = false;
      });
      if (_imageBytes != null) await _runDetection(_imageBytes!);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Switch failed: $e';
        _busy = false;
      });
    }
  }

  Future<void> _loadSample(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    final bytes = data.buffer.asUint8List();
    final ui = await decodeImageFromList(bytes);
    setState(() {
      _imageBytes = bytes;
      _imageSize = Size(ui.width.toDouble(), ui.height.toDouble());
      _detections = const [];
      _error = null;
    });
    await _runDetection(bytes);
  }

  Future<void> _runDetection(Uint8List bytes) async {
    final det = _detector;
    if (det == null) return;
    setState(() => _busy = true);
    final sw = Stopwatch()..start();
    try {
      final results = await det.detect(
        bytes,
        options: ObjectDetectorOptions(
          scoreThreshold: _threshold,
          maxResults: _maxResults,
        ),
      );
      sw.stop();
      if (!mounted) return;
      setState(() {
        _detections = results;
        _inferenceMs = sw.elapsedMilliseconds;
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Detection failed: $e';
        _busy = false;
      });
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('flutter_litert · Object Detection'),
        actions: [
          if (_inferenceMs > 0)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Center(
                child: TimingBadge(
                  totalMs: _inferenceMs,
                  detectionMs: _inferenceMs,
                ),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          _buildControls(),
          const Divider(height: 1),
          Expanded(child: _buildPreview()),
          if (_detections.isNotEmpty)
            SizedBox(height: 90, child: _buildResultList()),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Padding(
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Wrap(
            spacing: 6,
            runSpacing: 4,
            children: [
              for (final m in ObjectDetectionModel.values)
                ChoiceChip(
                  label: Text(
                    m == ObjectDetectionModel.efficientDetLite0
                        ? 'Lite0'
                        : 'Lite2',
                  ),
                  selected: _model == m,
                  onSelected: _busy ? null : (s) => s ? _switchModel(m) : null,
                ),
              const SizedBox(width: 8),
              for (final s in _samples)
                ActionChip(
                  label: Text(s.$1),
                  onPressed: _busy ? null : () => _loadSample(s.$2),
                ),
            ],
          ),
          const SizedBox(height: 4),
          CompactSlider(
            label: 'Score',
            value: _threshold,
            min: 0,
            max: 1,
            onChanged: (v) {
              setState(() => _threshold = v);
              if (_imageBytes != null) _runDetection(_imageBytes!);
            },
          ),
          CompactSlider(
            label: 'Max',
            value: _maxResults.toDouble(),
            min: 1,
            max: 30,
            onChanged: (v) {
              setState(() => _maxResults = v.round());
              if (_imageBytes != null) _runDetection(_imageBytes!);
            },
          ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.red, fontSize: 12),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildPreview() {
    final bytes = _imageBytes;
    final sz = _imageSize;
    if (bytes == null || sz == null) {
      return Center(
        child: _busy
            ? const CircularProgressIndicator()
            : const Text('Select a sample above.'),
      );
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        final fit = applyBoxFit(
          BoxFit.contain,
          sz,
          Size(constraints.maxWidth, constraints.maxHeight),
        );
        final imgRect = Alignment.center.inscribe(
          fit.destination,
          Offset.zero & Size(constraints.maxWidth, constraints.maxHeight),
        );
        return Stack(
          alignment: Alignment.center,
          children: [
            Image.memory(
              bytes,
              fit: BoxFit.contain,
              gaplessPlayback: true,
              width: constraints.maxWidth,
              height: constraints.maxHeight,
            ),
            Positioned.fill(
              child: CustomPaint(
                painter: DetectionsPainter(
                  detections: _detections,
                  imageRectOnCanvas: imgRect,
                  originalImageSize: sz,
                ),
              ),
            ),
            if (_busy)
              const Positioned(
                bottom: 16,
                right: 16,
                child: SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
          ],
        );
      },
    );
  }

  Widget _buildResultList() {
    return ListView.separated(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      itemCount: _detections.length,
      separatorBuilder: (context, i) => const SizedBox(width: 8),
      itemBuilder: (context, i) {
        final d = _detections[i];
        final color = colorForClass(d.category.index);
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: color.withAlpha(40),
            border: Border.all(color: color, width: 1.5),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                d.categoryName,
                style: TextStyle(fontWeight: FontWeight.bold, color: color),
              ),
              Text(
                '${(d.score * 100).toStringAsFixed(1)}%',
                style: const TextStyle(fontSize: 12),
              ),
            ],
          ),
        );
      },
    );
  }
}
