# Building the FlexDelegate xcframework for iOS

This guide builds `TensorFlowLiteFlex.xcframework` from TensorFlow v2.20.0 source. The xcframework provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like `Conv2DBackpropFilter`, `Save`, `Restore`, etc.

The resulting xcframework exports two symbols:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

**Important**: iOS does not allow `dlopen` for third-party dynamic libraries. The flex delegate is built as a **static framework** and linked into the app binary at build time. At runtime, Dart finds the symbols via `DynamicLibrary.process()`.

## Prerequisites

Install these before starting:

1. **macOS** with **Xcode 15+** (with iOS SDK and command-line tools)
2. **Bazelisk** (manages Bazel versions automatically)
   - Install: `brew install bazelisk`
   - TF 2.20.0 requires Bazel 7.4.1. Bazelisk handles this via `USE_BAZEL_VERSION=7.4.1`
3. **Python 3.9–3.12** (with `numpy` installed: `pip install numpy`)

Verify:
```bash
USE_BAZEL_VERSION=7.4.1 bazelisk version  # should show 6.5.0
python3 --version                          # 3.9-3.12
xcodebuild -version                        # 15.x or later
xcrun --show-sdk-path --sdk iphoneos       # should return a valid path
```

## Step 1: Clone TensorFlow

```bash
cd ~
git clone --depth 1 --branch v2.20.0 https://github.com/tensorflow/tensorflow.git tf-2.20.0
cd tf-2.20.0
```

## Step 2: Configure the build

```bash
PYTHON_BIN_PATH=/path/to/python3.10 \
TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_TENSORRT=0 \
TF_CONFIGURE_IOS=1 CC_OPT_FLAGS="-march=native" \
python3 configure.py
```

## Step 3: Add the custom plugin wrapper

Create `tensorflow/lite/delegates/flex/flex_delegate_plugin.cc`:

```cpp
#include "tensorflow/lite/delegates/flex/delegate.h"

extern "C" {

__attribute__((visibility("default")))
TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*)) {
  auto delegate = tflite::FlexDelegate::Create();
  return delegate.release();
}

__attribute__((visibility("default")))
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  delete reinterpret_cast<tflite::FlexDelegate*>(delegate);
}

}  // extern "C"
```

Append to `tensorflow/lite/delegates/flex/BUILD`:

```python
cc_library(
    name = "flex_delegate_plugin",
    srcs = ["flex_delegate_plugin.cc"],
    copts = tflite_copts(),
    features = tf_features_nolayering_check_if_ios(),
    visibility = ["//visibility:public"],
    deps = [
        ":delegate",
    ],
    alwayslink = True,
)
```

Add to `tensorflow/lite/ios/BUILD.apple` (before the CoreML framework target):

```python
ios_static_framework(
    name = "TensorFlowLiteFlex_framework",
    avoid_deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/async/interop/c:types",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
    ],
    bundle_name = "TensorFlowLiteFlex",
    minimum_os_version = TFL_MINIMUM_OS_VERSION,
    deps = [
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/delegates/flex:flex_delegate_plugin",
    ],
)
```

## Step 4: Build for each architecture

### 4a: iOS device (arm64)

```bash
USE_BAZEL_VERSION=7.4.1 bazelisk build -c opt --config=ios_arm64 \
  //tensorflow/lite/ios:TensorFlowLiteFlex_framework
```

### 4b: iOS simulator (arm64, Apple Silicon)

```bash
USE_BAZEL_VERSION=7.4.1 bazelisk build -c opt --config=ios_sim_arm64 \
  //tensorflow/lite/ios:TensorFlowLiteFlex_framework
```

### 4c: iOS simulator (x86_64, Intel)

```bash
USE_BAZEL_VERSION=7.4.1 bazelisk build -c opt --config=ios \
  --cpu=ios_x86_64 \
  //tensorflow/lite/ios:TensorFlowLiteFlex_framework
```

Each build takes 30-90 minutes. The output framework is at:
```
bazel-bin/tensorflow/lite/ios/TensorFlowLiteFlex_framework.zip
```

## Step 5: Extract and package as xcframework

```bash
mkdir -p ~/flex-ios-build/{device,sim-arm64,sim-x86_64}

# After device build:
unzip -o bazel-bin/tensorflow/lite/ios/TensorFlowLiteFlex_framework.zip \
  -d ~/flex-ios-build/device/

# After sim-arm64 build:
unzip -o bazel-bin/tensorflow/lite/ios/TensorFlowLiteFlex_framework.zip \
  -d ~/flex-ios-build/sim-arm64/

# After sim-x86_64 build:
unzip -o bazel-bin/tensorflow/lite/ios/TensorFlowLiteFlex_framework.zip \
  -d ~/flex-ios-build/sim-x86_64/
```

Create fat simulator binary:
```bash
mkdir -p ~/flex-ios-build/sim-universal/TensorFlowLiteFlex.framework
cp -R ~/flex-ios-build/sim-arm64/TensorFlowLiteFlex.framework/* \
  ~/flex-ios-build/sim-universal/TensorFlowLiteFlex.framework/

lipo -create \
  ~/flex-ios-build/sim-arm64/TensorFlowLiteFlex.framework/TensorFlowLiteFlex \
  ~/flex-ios-build/sim-x86_64/TensorFlowLiteFlex.framework/TensorFlowLiteFlex \
  -output ~/flex-ios-build/sim-universal/TensorFlowLiteFlex.framework/TensorFlowLiteFlex
```

Create xcframework:
```bash
xcodebuild -create-xcframework \
  -framework ~/flex-ios-build/device/TensorFlowLiteFlex.framework \
  -framework ~/flex-ios-build/sim-universal/TensorFlowLiteFlex.framework \
  -output ~/flex-ios-build/TensorFlowLiteFlex.xcframework
```

## Step 6: Verify symbols

```bash
nm -gU ~/flex-ios-build/device/TensorFlowLiteFlex.framework/TensorFlowLiteFlex \
  | grep tflite_plugin
```

Expected output:
```
_tflite_plugin_create_delegate
_tflite_plugin_destroy_delegate
```

## Step 7: Upload to GitHub Releases

```bash
cd ~/flex-ios-build
zip -r TensorFlowLiteFlex-ios.xcframework.zip TensorFlowLiteFlex.xcframework

gh release upload flex-v1.0.0 \
  TensorFlowLiteFlex-ios.xcframework.zip \
  --repo hugocornellier/flutter_litert \
  --clobber
```

## Troubleshooting

**Bazel version mismatch**: TF 2.20.0 requires Bazel 6.x. Use `USE_BAZEL_VERSION=7.4.1 bazelisk` instead of `bazel`.

**Xcode SDK not found**: Run `sudo xcode-select -s /Applications/Xcode.app`.

**Build fails with "undeclared inclusion"**: Make sure `configure.py` was run.

**Out of memory**: Add `--local_ram_resources=HOST_RAM*0.5 --jobs=4` to the bazel command.

**`configure.py` fails**: Use Python 3.9-3.12 (not 3.13+). Install `numpy`: `pip3 install numpy`.

**Bitcode warnings**: Ignore, Apple deprecated bitcode in Xcode 14.

## What this enables

Once uploaded, the `flutter_litert_flex` package's iOS podspec downloads the xcframework from:
```
https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.0.0/TensorFlowLiteFlex-ios.xcframework.zip
```

At pod install time, the xcframework is vendored into the app. At runtime, the flex delegate symbols are in the app binary, Dart finds them via `DynamicLibrary.process()`.
