cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "flutter_tflite")
project(${PROJECT_NAME} LANGUAGES CXX C)

set(PLUGIN_NAME "flutter_tflite_plugin")

list(APPEND PLUGIN_SOURCES
  "flutter_tflite_plugin.cc"
)

add_library(${PLUGIN_NAME} SHARED
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
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter)
target_link_libraries(${PLUGIN_NAME} PRIVATE PkgConfig::GTK)

# Build the TFLite custom ops library (e.g., Convolution2DTransposeBias for MediaPipe)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/custom_ops")

set(flutter_tflite_bundled_libraries
  "$<TARGET_FILE:tflite_custom_ops>"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtensorflowlite_c-linux.so"
  # LiteRT Next runtime (CompiledModel API) + WebGPU/Vulkan GPU accelerator,
  # x86_64; from the official ai-edge-litert 2.1.5 PyPI wheel. The runtime
  # discovers the accelerator by scanning its own directory for the bare
  # soname, so both must land in the same bundled lib/ dir.
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRt.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRtWebGpuAccelerator.so"
)

# Flutter's generated_plugins.cmake references plugins by their directory name
# (flutter_litert), not by PROJECT_NAME (flutter_tflite). Set both to ensure
# the custom ops library gets bundled.
set(flutter_litert_bundled_libraries
  "$<TARGET_FILE:tflite_custom_ops>"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtensorflowlite_c-linux.so"
  # LiteRT Next runtime (CompiledModel API) + WebGPU/Vulkan GPU accelerator,
  # x86_64; from the official ai-edge-litert 2.1.5 PyPI wheel. The runtime
  # discovers the accelerator by scanning its own directory for the bare
  # soname, so both must land in the same bundled lib/ dir.
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRt.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRtWebGpuAccelerator.so"
)

set(flutter_tflite_bundled_libraries ${flutter_tflite_bundled_libraries} PARENT_SCOPE)
set(flutter_litert_bundled_libraries ${flutter_litert_bundled_libraries} PARENT_SCOPE)

# === Tests ===
if (${include_${PROJECT_NAME}_tests})
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
message("Unit tests require CMake 3.11.0 or later")
else()
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
  test/flutter_tflite_plugin_test.cc
  ${PLUGIN_SOURCES}
)
apply_standard_settings(${TEST_RUNNER})
target_include_directories(${TEST_RUNNER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TEST_RUNNER} PRIVATE flutter)
target_link_libraries(${TEST_RUNNER} PRIVATE PkgConfig::GTK)
target_link_libraries(${TEST_RUNNER} PRIVATE gtest_main gmock)

include(GoogleTest)
gtest_discover_tests(${TEST_RUNNER})

endif()  # CMake version check
endif()  # include_${PROJECT_NAME}_tests
