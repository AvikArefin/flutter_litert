#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_litert.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_litert'
  s.version          = '0.0.1'
  s.summary          = 'LiteRT (formerly TensorFlow Lite) plugin for Flutter apps.'
  s.description      = <<-DESC
LiteRT (formerly TensorFlow Lite) plugin for Flutter apps.
                       DESC
  s.homepage         = 'https://github.com/hugocornellier/flutter_litert'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Hugo Cornellier' => 'hugo@hugocornellier.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }

  # Include Swift plugin and forwarder C file (which #includes the actual sources)
  s.source_files = 'Classes/**/*'

  # Preserve paths for header includes (these won't be compiled, just available for #include)
  s.preserve_paths = '../src/tensorflow_lite/**/*.h', '../src/custom_ops/**/*.h'

  s.dependency 'Flutter'

  # System frameworks required by TFLite and its delegates
  s.frameworks = 'Metal', 'CoreML', 'Accelerate'
  s.weak_frameworks = 'CoreML'

  s.platform = :ios, '12.0'
  s.static_framework = true

  # Common xcconfig shared between local and published builds
  common_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) TFLITE_USE_FRAMEWORK_HEADERS=1',
    'GCC_SYMBOLS_PRIVATE_EXTERN' => 'NO',
    'HEADER_SEARCH_PATHS' => '"${PODS_TARGET_SRCROOT}/../src" "${PODS_TARGET_SRCROOT}/../src/custom_ops"',
  }

  # Download iOS xcframeworks if not present (pub.dev packages exclude them to
  # stay under the 100 MB size limit; ~85 MB download, cached after first run).
  framework_dir = __dir__
  marker = File.join(framework_dir, 'TensorFlowLiteC.xcframework',
                     'ios-arm64', 'TensorFlowLiteC.framework', 'TensorFlowLiteC')
  unless File.exist?(marker)
    puts '[flutter_litert] Downloading TensorFlow Lite iOS frameworks...'
    zip = File.join(framework_dir, '_tflite_ios.zip')
    system("curl -sL 'https://github.com/hugocornellier/flutter_litert/releases/download/libs-v0.1.8/ios-frameworks.zip' -o '#{zip}'")
    abort '[flutter_litert] ERROR: Failed to download TFLite iOS frameworks. Check your internet connection.' unless $?.success?
    system("unzip -qo '#{zip}' -d '#{framework_dir}'")
    File.delete(zip) if File.exist?(zip)
    puts '[flutter_litert] TensorFlow Lite iOS frameworks installed.'
  end

  # When flutter_litert_flex is present, TensorFlowLiteFlex needs -all_load to
  # pull in C++ static initializers for TF op registration. TFLiteC and TFLiteFlex
  # share 15 symbols (XNNPack delegate + AcquireFlexDelegate). Make those symbols
  # local in TFLiteC so TFLiteFlex's versions are used (important for
  # AcquireFlexDelegate which must return the real flex delegate, not a stub).
  flex_detected = false
  flex_xcfw_mono = File.join(framework_dir, '..', 'flutter_litert_flex',
                             'ios', 'TensorFlowLiteFlex.xcframework')
  flex_detected = File.exist?(flex_xcfw_mono)
  unless flex_detected
    parent_dir = File.expand_path(File.join(framework_dir, '..', '..'))
    Dir.glob(File.join(parent_dir, 'flutter_litert_flex*', 'ios',
                       'TensorFlowLiteFlex.xcframework')).each do |_|
      flex_detected = true
      break
    end
  end
  unless flex_detected
    [Dir.pwd, File.join(Dir.pwd, '..')].each do |dir|
      deps_file = File.join(dir, '.flutter-plugins-dependencies')
      if File.exist?(deps_file)
        flex_detected = File.read(deps_file).include?('flutter_litert_flex')
        break if flex_detected
      end
    end
  end

  if flex_detected
    dedup_marker = File.join(framework_dir, 'TensorFlowLiteC.xcframework', '.flex_deduped')
    unless File.exist?(dedup_marker)
      puts '[flutter_litert] FlexDelegate detected, hiding overlapping symbols in TensorFlowLiteC...'
      syms_file = File.join(framework_dir, '_overlap_syms.txt')
      File.write(syms_file, <<~SYMS)
        _TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider
        _TfLiteXNNPackDelegateCreate
        _TfLiteXNNPackDelegateCreateWithThreadpool
        _TfLiteXNNPackDelegateDelete
        _TfLiteXNNPackDelegateGetFlags
        _TfLiteXNNPackDelegateGetOptions
        _TfLiteXNNPackDelegateGetThreadPool
        _TfLiteXNNPackDelegateInMemoryFilePath
        _TfLiteXNNPackDelegateOptionsDefault
        _TfLiteXNNPackDelegateWeightsCacheCreate
        _TfLiteXNNPackDelegateWeightsCacheCreateWithSize
        _TfLiteXNNPackDelegateWeightsCacheDelete
        _TfLiteXNNPackDelegateWeightsCacheFinalizeHard
        _TfLiteXNNPackDelegateWeightsCacheFinalizeSoft
        __ZN6tflite19AcquireFlexDelegateEv
      SYMS

      ['ios-arm64', 'ios-arm64_x86_64-simulator'].each do |arch|
        tflc_binary = File.join(framework_dir, 'TensorFlowLiteC.xcframework', arch,
                                'TensorFlowLiteC.framework', 'TensorFlowLiteC')
        next unless File.exist?(tflc_binary)
        system("xcrun nmedit -R '#{syms_file}' '#{tflc_binary}'")
      end

      File.delete(syms_file)
      File.write(dedup_marker, 'done')
      puts '[flutter_litert] Symbol deduplication complete.'
    end
  end

  s.vendored_frameworks = 'TensorFlowLiteC.xcframework',
                           'TensorFlowLiteCMetal.xcframework',
                           'TensorFlowLiteCCoreML.xcframework'

  s.pod_target_xcconfig = common_xcconfig.merge({
    'OTHER_LDFLAGS' => '$(inherited) -ObjC -all_load'
  })

  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => '$(inherited) -ObjC'
  }
  s.swift_version = '5.0'
end
