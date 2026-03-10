import 'dart:ffi';
import 'dart:io';

/// Opens a delegate dylib on macOS using a standard search strategy:
/// 1. Check [getCached] for a previously loaded library
/// 2. Try environment variable override via [envVar]
/// 3. Iterate [bundlePaths] in order
/// 4. Throw UnsupportedError with attempted paths if none found
DynamicLibrary openDelegateLibrary({
  required String envVar,
  required List<String> bundlePaths,
  required String description,
  required DynamicLibrary? Function() getCached,
  required void Function(DynamicLibrary) setCached,
}) {
  final cached = getCached();
  if (cached != null) return cached;

  final List<String> attemptedPaths = [];

  // Strategy 1: Environment variable override
  final envPath = Platform.environment[envVar];
  if (envPath != null && envPath.isNotEmpty) {
    attemptedPaths.add('$envVar: $envPath');
    try {
      final lib = DynamicLibrary.open(envPath);
      setCached(lib);
      return lib;
    } catch (e) {
      // Continue
    }
  }

  // Strategy 2: App bundle paths
  for (final path in bundlePaths) {
    attemptedPaths.add(path);
    try {
      final lib = DynamicLibrary.open(path);
      setCached(lib);
      return lib;
    } catch (e) {
      // Continue
    }
  }

  throw UnsupportedError(
    '$description library not found. Attempted paths:\n'
    '${attemptedPaths.map((p) => '  - $p').join('\n')}\n\n'
    'Solutions:\n'
    '  1. Set $envVar environment variable to the library path\n'
    '  2. The $description requires macOS on Apple Silicon (arm64)\n',
  );
}
