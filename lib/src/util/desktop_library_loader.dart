import 'dart:ffi';
import 'dart:io';

/// Tries to open [libName] from the standard macOS bundle locations used by
/// CocoaPods and SPM. Probes four paths in order:
///
/// 1. `Frameworks/[frameworkName].framework/Versions/A/Resources/[libName]`
/// 2. `Frameworks/[frameworkName].framework/Resources/[libName]`
/// 3. `Resources/[libName]`
/// 4. `Resources/[spmBundleName].bundle/Contents/Resources/[libName]`
///
/// Returns the loaded [DynamicLibrary], or `null` if all paths fail.
/// [attemptedPaths] is populated with tried path strings for error reporting.
DynamicLibrary? tryLoadMacOSBundlePaths(
  String libName, {
  required String frameworkName,
  required String spmBundleName,
  required List<String> attemptedPaths,
}) {
  final appBundle = Directory(Platform.resolvedExecutable).parent.parent;

  final paths = [
    (
      'Framework Resources path',
      '${appBundle.path}/Frameworks/$frameworkName.framework/Versions/A/Resources/$libName',
    ),
    (
      'Framework Resources path (alt)',
      '${appBundle.path}/Frameworks/$frameworkName.framework/Resources/$libName',
    ),
    ('App Resources path', '${appBundle.path}/Resources/$libName'),
    (
      'SPM bundle path',
      '${appBundle.path}/Resources/$spmBundleName.bundle/Contents/Resources/$libName',
    ),
  ];

  for (final (label, path) in paths) {
    attemptedPaths.add('$label: $path');
    try {
      return DynamicLibrary.open(path);
    } catch (_) {
      // Continue
    }
  }

  return null;
}
