import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('throwDecodeFailure throws a prefixed IsolateRpcExactError', () {
    expect(
      () => throwDecodeFailure('bad input: x'),
      throwsA(
        predicate(
          (e) =>
              e is IsolateRpcExactError &&
              e.message.startsWith(decodeFailurePrefix) &&
              e.message.contains('bad input: x'),
        ),
      ),
    );
  });

  test('rethrowOrFormatException maps a decode-failure StateError', () {
    final wire = '$decodeFailurePrefix Image bytes could not be decoded: oops';
    expect(
      () => rethrowOrFormatException(StateError(wire), Uint8List(3)),
      throwsA(
        predicate((e) => e is FormatException && e.message.contains('oops')),
      ),
    );
  });

  test('rethrowOrFormatException rethrows non-decode errors unchanged', () {
    final other = StateError('something unrelated');
    expect(
      () => rethrowOrFormatException(other),
      throwsA(predicate((e) => identical(e, other))),
    );
  });
}
