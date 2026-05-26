# flutter_litert_flex_test_host

Dedicated integration test host for the optional `flutter_litert_flex` addon.

The main `example/` app keeps a minimal dependency list. This package adds
`flutter_litert_flex` separately so CI can prove the addon is installed,
bundled, and usable without making Flex a dependency of the primary example.

Run from this directory:

```bash
flutter test -d macos integration_test
# or on Linux:
flutter test -d linux integration_test
```
