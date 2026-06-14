cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "flutter_tflite")
project(${PROJECT_NAME} LANGUAGES CXX C)

cmake_policy(VERSION 3.14...3.25)

set(PLUGIN_NAME "flutter_tflite_plugin")

list(APPEND PLUGIN_SOURCES
  "flutter_tflite_plugin.cpp"
  "flutter_tflite_plugin.h"
)

add_library(${PLUGIN_NAME} SHARED
  "include/flutter_tflite/flutter_tflite_plugin_c_api.h"
  "flutter_tflite_plugin_c_api.cpp"
  ${PLUGIN_SOURCES}
)

apply_standard_settings(${PLUGIN_NAME})

# Symbols are hidden by default to reduce the chance of accidental conflicts
# between plugins. Any symbols that should be exported must use FLUTTER_PLUGIN_EXPORT.
set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

# Build the TFLite custom ops library (e.g., Convolution2DTransposeBias for MediaPipe)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/custom_ops")

set(FLUTTER_LITERT_TFLITE_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libtensorflowlite_c-win.dll")
# LiteRT Next runtime (CompiledModel API), x86_64; from the official
# ai-edge-litert 2.1.5 PyPI wheel.
set(FLUTTER_LITERT_LITERT_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.dll")
# WebGPU (Dawn/D3D12) GPU accelerator for the CompiledModel API; from the same
# ai-edge-litert 2.1.5 wheel. The runtime discovers it at startup by scanning
# the RuntimeLibraryDir environment option for this exact file name, so the
# name must not change.
set(FLUTTER_LITERT_GPU_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRtWebGpuAccelerator.dll")
# DirectX Shader Compiler (DXC), from microsoft/DirectXShaderCompiler release
# v1.9.2602.24. Optional but preferred: Dawn uses DXC instead of the OS FXC
# (d3dcompiler_47.dll) when both dxcompiler.dll and dxil.dll are present next
# to the executable.
set(FLUTTER_LITERT_DXCOMPILER_DLL "${CMAKE_CURRENT_SOURCE_DIR}/dxcompiler.dll")
set(FLUTTER_LITERT_DXIL_DLL "${CMAKE_CURRENT_SOURCE_DIR}/dxil.dll")

# The WebGPU GPU accelerator (~20 MB) and DirectX Shader Compiler (~18 MB) are
# fetched at build time from a GitHub release rather than bundled in the
# published pub.dev package, which must stay under the 100 MiB limit. Each is
# downloaded (verified by SHA-256) only if it is not already present.
set(FLUTTER_LITERT_GPU_LIBS_URL
  "https://github.com/hugocornellier/flutter_litert/releases/download/litert-desktop-gpu-v1.0.0")

function(flutter_litert_fetch_lib path url sha)
  if(EXISTS "${path}")
    return()
  endif()
  foreach(attempt RANGE 1 3)
    message(STATUS "flutter_litert: downloading ${url} (attempt ${attempt})")
    file(DOWNLOAD "${url}" "${path}"
      EXPECTED_HASH SHA256=${sha}
      TLS_VERIFY ON
      STATUS dl_status)
    list(GET dl_status 0 dl_code)
    if(dl_code EQUAL 0)
      return()
    endif()
    list(GET dl_status 1 dl_msg)
    message(WARNING "flutter_litert: download failed (${dl_msg}); retrying")
    file(REMOVE "${path}")
  endforeach()
  message(FATAL_ERROR "flutter_litert: could not download ${url}")
endfunction()

flutter_litert_fetch_lib("${FLUTTER_LITERT_GPU_DLL}"
  "${FLUTTER_LITERT_GPU_LIBS_URL}/libLiteRtWebGpuAccelerator.dll"
  "65fb92e4b0c3f1bba389964ffac5544bfebe5edf029c5a94a87864fa3e3ee124")
flutter_litert_fetch_lib("${FLUTTER_LITERT_DXCOMPILER_DLL}"
  "${FLUTTER_LITERT_GPU_LIBS_URL}/dxcompiler.dll"
  "9b5e10ed756c461b4ec2c83a99f1d6ace20e97826e9c0b0e966b7b1cd6f2aec6")
flutter_litert_fetch_lib("${FLUTTER_LITERT_DXIL_DLL}"
  "${FLUTTER_LITERT_GPU_LIBS_URL}/dxil.dll"
  "cbcfe883a09fd0ca1f98abdf3a9553b560895e3283a136da82a8381253a169df")

set(flutter_tflite_bundled_libraries
  "${FLUTTER_LITERT_TFLITE_DLL}"
  "${FLUTTER_LITERT_LITERT_DLL}"
  "${FLUTTER_LITERT_GPU_DLL}"
  "${FLUTTER_LITERT_DXCOMPILER_DLL}"
  "${FLUTTER_LITERT_DXIL_DLL}"
  "$<TARGET_FILE:tflite_custom_ops>"
)

# Flutter's generated_plugins.cmake references plugins by their directory name
# (flutter_litert), not by PROJECT_NAME (flutter_tflite). Set both to ensure
# bundled native libraries are included.
set(flutter_litert_bundled_libraries
  "${FLUTTER_LITERT_TFLITE_DLL}"
  "${FLUTTER_LITERT_LITERT_DLL}"
  "${FLUTTER_LITERT_GPU_DLL}"
  "${FLUTTER_LITERT_DXCOMPILER_DLL}"
  "${FLUTTER_LITERT_DXIL_DLL}"
  "$<TARGET_FILE:tflite_custom_ops>"
)

set(flutter_tflite_bundled_libraries ${flutter_tflite_bundled_libraries} PARENT_SCOPE)
set(flutter_litert_bundled_libraries ${flutter_litert_bundled_libraries} PARENT_SCOPE)

# === Tests ===
if (${include_${PROJECT_NAME}_tests})
set(TEST_RUNNER "${PROJECT_NAME}_test")
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of googletest" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(${TEST_RUNNER}
  test/flutter_tflite_plugin_test.cpp
  ${PLUGIN_SOURCES}
)
apply_standard_settings(${TEST_RUNNER})
target_include_directories(${TEST_RUNNER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TEST_RUNNER} PRIVATE flutter_wrapper_plugin)
target_link_libraries(${TEST_RUNNER} PRIVATE gtest_main gmock)
add_custom_command(TARGET ${TEST_RUNNER} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${FLUTTER_LIBRARY}" $<TARGET_FILE_DIR:${TEST_RUNNER}>
)

include(GoogleTest)
gtest_discover_tests(${TEST_RUNNER})
endif()
