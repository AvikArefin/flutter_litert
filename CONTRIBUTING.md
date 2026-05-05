# Contributing to flutter_litert

Thanks for helping improve `flutter_litert`. This repository is a Flutter
plugin for LiteRT / TensorFlow Lite inference, native delegate loading,
custom ops, web runtimes, and shared ML utility code.

## Before opening a pull request

- Run `flutter pub get`.
- Run `dart format --output=none --set-exit-if-changed .`.
- Run `flutter analyze .`.
- Run `flutter test`.
- Add or update focused tests when changing public API, delegate behavior,
  tensor conversion, model training, custom ops, or web runtime code.

## Native and platform changes

- Keep platform-specific behavior documented in `README.md` and matching
  Dart docs.
- Do not change bundled runtime versions, generated FFI bindings, podspecs,
  or CMake files without updating the related docs and tests.
- When changing TensorFlow Lite / LiteRT C APIs, regenerate bindings with the
  `melos run ffigen` script.
- For custom ops, keep the op name, exported C registration symbol, and Dart
  FFI registration in sync.

## Pull request style

- Keep changes scoped to one behavior or documentation update.
- Call out platform coverage in the PR description.
- Mention any tests that were skipped and why.

## License

New files should use the Apache-2.0 license header used by the surrounding
source files unless the file is generated or follows an existing local pattern.
