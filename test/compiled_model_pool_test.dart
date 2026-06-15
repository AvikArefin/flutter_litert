import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

const String _model = 'example/assets/simple_model.tflite';

CompiledModel _make() =>
    CompiledModel.fromFile(_model, accelerators: {Accelerator.cpu});

bool _runtimeAvailable() {
  try {
    _make().close();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  test('throws before initialize', () {
    final pool = CompiledModelPool();
    expect(pool.isInitialized, isFalse);
    expect(pool.size, 0);
    expect(() => pool.withModel((m, i) async => 0), throwsStateError);
  });

  test('builds the requested number of slots and disposes them', () {
    if (!_runtimeAvailable()) {
      markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
      return;
    }
    final pool = CompiledModelPool();
    pool.initialize(poolSize: 3, inputFloats: 4, create: _make);
    expect(pool.isInitialized, isTrue);
    expect(pool.size, 3);
    pool.dispose();
    expect(pool.isInitialized, isFalse);
    expect(pool.size, 0);
  });

  test('poolSize is clamped to at least 1', () {
    if (!_runtimeAvailable()) {
      markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
      return;
    }
    final pool = CompiledModelPool();
    pool.initialize(poolSize: 0, inputFloats: 4, create: _make);
    expect(pool.size, 1);
    pool.dispose();
  });

  test('onFirstModel is called once, with the first model', () {
    if (!_runtimeAvailable()) {
      markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
      return;
    }
    final pool = CompiledModelPool();
    int calls = 0;
    pool.initialize(
      poolSize: 3,
      inputFloats: 4,
      create: _make,
      onFirstModel: (_) => calls++,
    );
    expect(calls, 1);
    pool.dispose();
  });

  test('initialize cleans up and rethrows when onFirstModel fails', () {
    if (!_runtimeAvailable()) {
      markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
      return;
    }
    final pool = CompiledModelPool();
    expect(
      () => pool.initialize(
        poolSize: 3,
        inputFloats: 4,
        create: _make,
        onFirstModel: (_) => throw UnsupportedError('bad model'),
      ),
      throwsUnsupportedError,
    );
    // No slots leaked; the pool is usable for a subsequent (valid) init.
    expect(pool.isInitialized, isFalse);
    expect(pool.size, 0);
  });

  test('withModel cycles through slots round-robin', () async {
    if (!_runtimeAvailable()) {
      markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
      return;
    }
    final pool = CompiledModelPool();
    pool.initialize(poolSize: 3, inputFloats: 4, create: _make);

    // Each slot owns a distinct reusable input buffer; over 6 calls we should
    // see exactly 3 distinct buffers, each twice, in cycling order.
    final seen = <int>[];
    for (int i = 0; i < 6; i++) {
      await pool.withModel((model, input) async {
        seen.add(identityHashCode(input));
        return null;
      });
    }
    expect(seen[0], isNot(seen[1]));
    expect(seen[1], isNot(seen[2]));
    expect(seen[0], seen[3]); // wrapped back to slot 0
    expect(seen[1], seen[4]);
    expect(seen[2], seen[5]);
    expect(seen.toSet().length, 3);
    pool.dispose();
  });

  test('a single slot serializes overlapping calls (no interleave)', () async {
    if (!_runtimeAvailable()) {
      markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
      return;
    }
    final pool = CompiledModelPool();
    pool.initialize(poolSize: 1, inputFloats: 4, create: _make);

    final order = <String>[];
    Future<void> task(String tag) => pool.withModel((model, input) async {
      order.add('$tag-start');
      await Future<void>.delayed(const Duration(milliseconds: 20));
      order.add('$tag-end');
    });

    await Future.wait([task('a'), task('b')]);
    // With one slot the two calls must not interleave: a fully completes
    // before b starts (registration order).
    expect(order, ['a-start', 'a-end', 'b-start', 'b-end']);
    pool.dispose();
  });
}
