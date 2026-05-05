# flutter_litert_example

Example app and integration-test host for the `flutter_litert` plugin.

The app itself is intentionally small; the useful examples live in:

- `example/example.dart`: minimal native file-based inference snippet.
- `example/integration_test/`: delegate smoke tests for XNNPACK, Metal,
  CoreML, and Flex.
- `example/assets/`: small `.tflite` models used by tests.

Run from the repository root:

```bash
cd example
flutter test
flutter test integration_test
```
