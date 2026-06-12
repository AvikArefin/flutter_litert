#!/bin/bash
# Wraps the prebuilt LiteRT Next iOS dylibs into conventional .framework
# bundles and assembles framework-type xcframeworks.
#
# CocoaPods rejects vendored xcframeworks that contain bare dynamic libraries,
# so the CocoaPods channel ships these framework-wrapped variants. The rename
# (libLiteRt.dylib -> LiteRt.framework/LiteRt) breaks LiteRT's GPU-plugin
# file-name scan; ios/Classes/litert_gpu_accelerator_shim.c compensates by
# exporting LiteRtRegisterGpuAccelerator. The SwiftPM channel keeps bare-dylib
# (library-type) xcframeworks, where the runtime's own scan works.
#
# Usage: wrap_litert_ios_frameworks.sh <prebuilt_dir> <out_dir>
#   <prebuilt_dir> must contain ios_arm64/ and ios_sim_arm64/ with
#   libLiteRt.dylib and libLiteRtMetalAccelerator.dylib in each.
set -euo pipefail

PREBUILT=$1
OUT=$2
VERSION=1.0.0

make_framework() { # dylib name out_dir [universal_sim]
  local dylib=$1 name=$2 out_dir=$3 universal_sim=${4:-}
  local fw="$out_dir/$name.framework"
  mkdir -p "$fw"
  if [ -n "$universal_sim" ]; then
    # Google ships no x86_64 iOS-simulator binaries. CocoaPods' xcframework
    # slice selection matches the build's full ARCHS list ("arm64 x86_64" on
    # simulator), so an arm64-only simulator slice is skipped — and a
    # CocoaPods script bug then reuses the previous pod's slice id, breaking
    # the build. Lipo in an empty x86_64 stub so the slice matches and
    # links; CompiledModel is simply unavailable in x86_64 simulators.
    local stub="$out_dir/$name.x86_64-stub"
    echo "static int litert_unavailable_on_x86_64_simulator;" |
      xcrun --sdk iphonesimulator clang -x c - -dynamiclib \
        -target x86_64-apple-ios14.0-simulator \
        -install_name "@rpath/$name.framework/$name" -o "$stub"
    lipo -create "$dylib" "$stub" -output "$fw/$name"
    rm "$stub"
  else
    cp "$dylib" "$fw/$name"
  fi
  install_name_tool -id "@rpath/$name.framework/$name" "$fw/$name"
  /usr/libexec/PlistBuddy -c "Add :CFBundleDevelopmentRegion string en" \
    -c "Add :CFBundleExecutable string $name" \
    -c "Add :CFBundleIdentifier string dev.flutterlitert.$name" \
    -c "Add :CFBundleInfoDictionaryVersion string 6.0" \
    -c "Add :CFBundleName string $name" \
    -c "Add :CFBundlePackageType string FMWK" \
    -c "Add :CFBundleShortVersionString string $VERSION" \
    -c "Add :CFBundleVersion string $VERSION" \
    -c "Add :MinimumOSVersion string 14.0" \
    "$fw/Info.plist"
  codesign --force --sign - "$fw"
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for name in LiteRt LiteRtMetalAccelerator; do
  mkdir -p "$WORK/ios_arm64" "$WORK/ios_sim_arm64"
  make_framework "$PREBUILT/ios_arm64/lib$name.dylib" "$name" "$WORK/ios_arm64"
  make_framework "$PREBUILT/ios_sim_arm64/lib$name.dylib" "$name" \
    "$WORK/ios_sim_arm64" universal_sim
  rm -rf "$OUT/$name.xcframework"
  xcodebuild -create-xcframework \
    -framework "$WORK/ios_arm64/$name.framework" \
    -framework "$WORK/ios_sim_arm64/$name.framework" \
    -output "$OUT/$name.xcframework"
done
