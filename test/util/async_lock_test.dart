import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('AsyncLock', () {
    test('serializes tasks in registration order with no overlap', () async {
      final lock = AsyncLock();
      final log = <String>[];
      var active = 0;
      var maxActive = 0;

      Future<int> task(String id, int value) => lock.run(() async {
        active++;
        maxActive = active > maxActive ? active : maxActive;
        log.add('start:$id');
        // Yield several times so any concurrency would interleave.
        await Future<void>.delayed(Duration.zero);
        await Future<void>.delayed(Duration.zero);
        log.add('end:$id');
        active--;
        return value;
      });

      // Register synchronously so ordering is determined by call order.
      final futures = [task('a', 1), task('b', 2), task('c', 3)];
      final results = await Future.wait(futures);

      expect(results, [1, 2, 3]);
      expect(maxActive, 1, reason: 'tasks must never overlap');
      expect(log, ['start:a', 'end:a', 'start:b', 'end:b', 'start:c', 'end:c']);
    });

    test('a throwing task surfaces its error to its caller', () async {
      final lock = AsyncLock();
      await expectLater(
        lock.run<int>(() async => throw StateError('boom')),
        throwsA(isA<StateError>()),
      );
    });

    test('a throwing task does not poison later tasks on the chain', () async {
      final lock = AsyncLock();
      final order = <String>[];

      final bad = lock.run<int>(() async {
        order.add('bad');
        throw StateError('boom');
      });
      final good = lock.run<int>(() async {
        order.add('good');
        return 42;
      });

      await expectLater(bad, throwsA(isA<StateError>()));
      expect(await good, 42);
      expect(order, ['bad', 'good']);
    });

    test('the next task waits for the previous one to settle', () async {
      final lock = AsyncLock();
      var firstDone = false;

      final first = lock.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 20));
        firstDone = true;
      });
      final second = lock.run(() async {
        expect(
          firstDone,
          isTrue,
          reason: 'second must not start until first has finished',
        );
      });

      await Future.wait([first, second]);
    });
  });
}
