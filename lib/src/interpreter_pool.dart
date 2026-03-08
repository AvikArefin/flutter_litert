import 'dart:async';

import 'native/interpreter.dart';
import 'native/interpreter_options.dart';
import 'native/delegate.dart';
import 'native/isolate_interpreter.dart';
import 'interpreter_factory.dart';
import 'performance_config.dart';

/// Factory function to create and configure an [Interpreter] for one pool slot.
///
/// Receives [InterpreterOptions] (with delegate pre-embedded) and the raw
/// [Delegate] (nullable, for informational use). The pool handles delegate
/// cleanup and [IsolateInterpreter] creation internally.
typedef InterpreterCreator =
    Future<Interpreter> Function(
      InterpreterOptions options,
      Delegate? delegate,
    );

/// A round-robin pool of [Interpreter] instances with per-slot serialization locks.
///
/// Each slot has its own [Future]-chain lock so concurrent callers are
/// serialized per-interpreter rather than globally, enabling parallel inference
/// across multiple pool slots while preventing XNNPACK thread contention.
///
/// Usage:
/// ```dart
/// final pool = InterpreterPool(poolSize: 3);
/// await pool.initialize(
///   (options, _) async {
///     final interp = await Interpreter.fromAsset('model.tflite', options: options);
///     interp.resizeInputTensor(0, [1, 224, 224, 3]);
///     interp.allocateTensors();
///     return interp;
///   },
///   performanceConfig: PerformanceConfig(),
/// );
///
/// final result = await pool.withInterpreter((interp, iso) async {
///   // run inference ...
/// });
///
/// await pool.dispose();
/// ```
class InterpreterPool {
  final int poolSize;

  final List<Interpreter> _interpreters = [];
  final List<IsolateInterpreter?> _isolates = [];
  final List<Delegate> _delegates = [];
  final List<Future<void>> _locks = [];
  int _counter = 0;
  bool _isInitialized = false;

  InterpreterPool({int poolSize = 1}) : poolSize = poolSize.clamp(1, 10);

  bool get isInitialized => _isInitialized;

  /// All interpreters in the pool, in creation order.
  ///
  /// Useful for consumers that need to associate per-slot state (e.g.,
  /// pre-allocated buffers) with each interpreter via a map.
  List<Interpreter> get interpreters => List.unmodifiable(_interpreters);

  /// Initializes the pool by calling [factory] once per slot.
  ///
  /// If already initialized, disposes all resources first.
  /// [IsolateInterpreter] is created automatically when no delegate is used.
  Future<void> initialize(
    InterpreterCreator factory, {
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    for (int i = 0; i < poolSize; i++) {
      final (options, delegate) = InterpreterFactory.create(performanceConfig);
      if (delegate != null) _delegates.add(delegate);

      final interpreter = await factory(options, delegate);
      final isolate = await InterpreterFactory.createIsolateIfNeeded(
        interpreter,
        delegate,
      );

      _interpreters.add(interpreter);
      _isolates.add(isolate);
      _locks.add(Future.value());
    }

    _isInitialized = true;
  }

  /// Runs [fn] with exclusive access to one interpreter, selected round-robin.
  ///
  /// Callers are serialized per slot — different slots can run concurrently.
  Future<T> withInterpreter<T>(
    Future<T> Function(Interpreter, IsolateInterpreter?) fn,
  ) async {
    if (_interpreters.isEmpty) {
      throw StateError('InterpreterPool is empty. Call initialize() first.');
    }

    final idx = _counter % _interpreters.length;
    _counter = (_counter + 1) % _interpreters.length;

    final prev = _locks[idx];
    final completer = Completer<void>();
    _locks[idx] = completer.future;

    try {
      await prev;
      return await fn(_interpreters[idx], _isolates[idx]);
    } finally {
      completer.complete();
    }
  }

  /// Disposes all interpreters, isolates, and delegates.
  Future<void> dispose() async {
    for (int i = 0; i < _interpreters.length; i++) {
      _isolates[i]?.close();
      _interpreters[i].close();
    }
    _interpreters.clear();
    _isolates.clear();

    for (final d in _delegates) {
      d.delete();
    }
    _delegates.clear();
    _locks.clear();
    _counter = 0;
    _isInitialized = false;
  }
}
