import 'dart:async';
import 'dart:isolate';

/// RPC client that sends requests to and receives responses from an isolate.
class IsolateRpcClient {
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;

  SendPort? sendPort;
  Isolate? isolate;
  final ReceivePort receivePort = ReceivePort();

  /// Sends an RPC request to the isolate and returns the typed response future.
  Future<T> sendRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (sendPort == null) {
      throw StateError('IsolateRpcClient not ready: sendPort is null');
    }

    final int id = _nextId++;
    final Completer<T> completer = Completer<T>();
    _pending[id] = completer;

    try {
      sendPort!.send({'id': id, 'op': operation, ...params});
      return await completer.future;
    } catch (e) {
      _pending.remove(id);
      rethrow;
    }
  }

  /// Dispatches an incoming isolate message to the matching pending completer.
  void handleResponse(
    dynamic message, {
    Object Function(String)? errorWrapper,
  }) {
    if (message is! Map) return;

    final int? id = message['id'] as int?;
    if (id == null) return;

    final Completer<dynamic>? completer = _pending.remove(id);
    if (completer == null) return;

    if (message['error'] != null) {
      final errorMsg = message['error'] as String;
      final error = errorWrapper != null
          ? errorWrapper(errorMsg)
          : StateError(errorMsg);
      completer.completeError(error);
    } else {
      completer.complete(message['result']);
    }
  }

  /// Fails all pending requests, optionally sends a dispose op, and kills the isolate.
  void failAllAndDispose({String? disposeOp}) {
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('IsolateRpcClient disposed'));
      }
    }
    _pending.clear();

    if (disposeOp != null && sendPort != null) {
      try {
        sendPort!.send({'id': -1, 'op': disposeOp});
      } catch (_) {}
    }

    isolate?.kill(priority: Isolate.immediate);
    receivePort.close();

    isolate = null;
    sendPort = null;
  }
}

/// Performs the initial SendPort handshake with a newly spawned isolate.
Future<SendPort> setupIsolateHandshake({
  required ReceivePort receivePort,
  required void Function(dynamic) onResponse,
  required Duration timeout,
  required String timeoutMessage,
}) async {
  final Completer<SendPort> initCompleter = Completer<SendPort>();
  late final StreamSubscription<dynamic> subscription;

  subscription = receivePort.listen((message) {
    if (!initCompleter.isCompleted) {
      if (message is SendPort) {
        initCompleter.complete(message);
      } else if (message is Map && message['error'] != null) {
        initCompleter.completeError(StateError(message['error'] as String));
      } else {
        initCompleter.completeError(
          StateError('Expected SendPort, got ${message.runtimeType}'),
        );
      }
      return;
    }
    onResponse(message);
  });

  return initCompleter.future.timeout(
    timeout,
    onTimeout: () {
      subscription.cancel();
      throw TimeoutException(timeoutMessage);
    },
  );
}
