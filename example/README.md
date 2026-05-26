# flutter_litert_example

Example app and integration-test host for the `flutter_litert` plugin.

The app runs an EfficientDet Lite0 object detector against bundled sample
images, with preprocessing, inference, and postprocessing handled from Dart.

Run the app from this directory:

```bash
flutter run
```

Other useful entry points:

- `example.dart`: minimal native file-based inference snippet.
- `integration_test/`: delegate smoke tests for XNNPACK, Metal, and CoreML.
- `flex_test_host/`: addon integration tests that depend on `flutter_litert_flex`.
- `assets/`: models, label maps, and sample images used by the app and tests.

Run the package tests from the repository root:

```bash
cd ..
flutter test
```

Run the optional Flex addon integration host from this directory:

```bash
cd flex_test_host
flutter test -d macos integration_test
```

The `integration_test/` directory contains platform delegate smoke tests for
XNNPACK, Metal, and CoreML. Run those directly only on platforms where the
corresponding delegate libraries are bundled or provided via environment
variables.
