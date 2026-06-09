/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'litert_loader.dart';

const int kLiteRtAnyTypeString = 8;
const int kLiteRtEnvOptionTagRuntimeLibraryDir = 22;

/// LiteRtLayout:
/// { uint rank:7; bool has_strides:1; int32 dims[8]; uint32 strides[8]; }
/// ABI size is 68 bytes on the supported Apple/Linux desktop builds.
final class LiteRtLayout extends Struct {
  @Uint32()
  external int bitfields;

  @Array(8)
  external Array<Int32> dimensions;

  @Array(8)
  external Array<Uint32> strides;
}

/// LiteRtRankedTensorType:
/// { int32 element_type; LiteRtLayout layout; }
/// ABI size is 72 bytes on the supported Apple/Linux desktop builds.
final class LiteRtRankedTensorType extends Struct {
  @Int32()
  external int elementType;

  external LiteRtLayout layout;
}

/// LiteRtAny:
/// { int32 type; union { bool; int64; double; const char*; const void*; } }
/// ABI size is 16 bytes on the supported 64-bit desktop builds.
final class LiteRtAny extends Struct {
  @Int32()
  external int type;

  external LiteRtAnyValue value;
}

final class LiteRtAnyValue extends Union {
  @Uint8()
  external int boolValue;

  @Int64()
  external int intValue;

  @Double()
  external double realValue;

  external Pointer<Utf8> strValue;

  external Pointer<Void> ptrValue;
}

/// LiteRtEnvOption:
/// { int32 tag; LiteRtAny value; }
/// ABI size is 24 bytes with [value] at offset 8 on 64-bit builds.
final class LiteRtEnvOption extends Struct {
  @Int32()
  external int tag;

  external LiteRtAny value;
}

typedef LiteRtOpaquePayloadDeleterNative = Void Function(Pointer<Void>);

/// Hand-written bindings for the LiteRT Next C API surface used by
/// [CompiledModel].
final class LiteRtBindings {
  LiteRtBindings(DynamicLibrary dylib)
    : _dylib = dylib,
      createEnvironment = dylib
          .lookupFunction<
            Int32 Function(
              Int32,
              Pointer<LiteRtEnvOption>,
              Pointer<Pointer<Void>>,
            ),
            int Function(int, Pointer<LiteRtEnvOption>, Pointer<Pointer<Void>>)
          >('LiteRtCreateEnvironment'),
      destroyEnvironment = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyEnvironment'),
      createOptions = dylib
          .lookupFunction<
            Int32 Function(Pointer<Pointer<Void>>),
            int Function(Pointer<Pointer<Void>>)
          >('LiteRtCreateOptions'),
      destroyOptions = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyOptions'),
      setOptionsHardwareAccelerators = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Int32),
            int Function(Pointer<Void>, int)
          >('LiteRtSetOptionsHardwareAccelerators'),
      createOpaqueOptions = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Utf8>,
              Pointer<Void>,
              Pointer<NativeFunction<LiteRtOpaquePayloadDeleterNative>>,
              Pointer<Pointer<Void>>,
            ),
            int Function(
              Pointer<Utf8>,
              Pointer<Void>,
              Pointer<NativeFunction<LiteRtOpaquePayloadDeleterNative>>,
              Pointer<Pointer<Void>>,
            )
          >('LiteRtCreateOpaqueOptions'),
      destroyOpaqueOptions = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyOpaqueOptions'),
      addOpaqueOptions = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Void>),
            int Function(Pointer<Void>, Pointer<Void>)
          >('LiteRtAddOpaqueOptions'),
      createModelFromFile = dylib
          .lookupFunction<
            Int32 Function(Pointer<Utf8>, Pointer<Pointer<Void>>),
            int Function(Pointer<Utf8>, Pointer<Pointer<Void>>)
          >('LiteRtCreateModelFromFile'),
      destroyModel = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyModel'),
      createCompiledModel = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              Pointer<Void>,
              Pointer<Void>,
              Pointer<Pointer<Void>>,
            ),
            int Function(
              Pointer<Void>,
              Pointer<Void>,
              Pointer<Void>,
              Pointer<Pointer<Void>>,
            )
          >('LiteRtCreateCompiledModel'),
      destroyCompiledModel = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyCompiledModel'),
      getModelSignature = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, IntPtr, Pointer<Pointer<Void>>),
            int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
          >('LiteRtGetModelSignature'),
      getNumSignatureInputs = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<IntPtr>),
            int Function(Pointer<Void>, Pointer<IntPtr>)
          >('LiteRtGetNumSignatureInputs'),
      getNumSignatureOutputs = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<IntPtr>),
            int Function(Pointer<Void>, Pointer<IntPtr>)
          >('LiteRtGetNumSignatureOutputs'),
      getSignatureInputTensorByIndex = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, IntPtr, Pointer<Pointer<Void>>),
            int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
          >('LiteRtGetSignatureInputTensorByIndex'),
      getSignatureOutputTensorByIndex = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, IntPtr, Pointer<Pointer<Void>>),
            int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
          >('LiteRtGetSignatureOutputTensorByIndex'),
      getRankedTensorType = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<LiteRtRankedTensorType>),
            int Function(Pointer<Void>, Pointer<LiteRtRankedTensorType>)
          >('LiteRtGetRankedTensorType'),
      getCompiledModelInputBufferRequirements = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              IntPtr,
              IntPtr,
              Pointer<Pointer<Void>>,
            ),
            int Function(Pointer<Void>, int, int, Pointer<Pointer<Void>>)
          >('LiteRtGetCompiledModelInputBufferRequirements'),
      getCompiledModelOutputBufferRequirements = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              IntPtr,
              IntPtr,
              Pointer<Pointer<Void>>,
            ),
            int Function(Pointer<Void>, int, int, Pointer<Pointer<Void>>)
          >('LiteRtGetCompiledModelOutputBufferRequirements'),
      getTensorBufferRequirementsBufferSize = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<IntPtr>),
            int Function(Pointer<Void>, Pointer<IntPtr>)
          >('LiteRtGetTensorBufferRequirementsBufferSize'),
      destroyTensorBufferRequirements = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyTensorBufferRequirements'),
      createManagedTensorBufferFromRequirements = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              Pointer<LiteRtRankedTensorType>,
              Pointer<Void>,
              Pointer<Pointer<Void>>,
            ),
            int Function(
              Pointer<Void>,
              Pointer<LiteRtRankedTensorType>,
              Pointer<Void>,
              Pointer<Pointer<Void>>,
            )
          >('LiteRtCreateManagedTensorBufferFromRequirements'),
      destroyTensorBuffer = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyTensorBuffer'),
      getTensorBufferType = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Int32>),
            int Function(Pointer<Void>, Pointer<Int32>)
          >('LiteRtGetTensorBufferType'),
      lockTensorBuffer = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Pointer<Void>>, Int32),
            int Function(Pointer<Void>, Pointer<Pointer<Void>>, int)
          >('LiteRtLockTensorBuffer'),
      unlockTensorBuffer = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)
          >('LiteRtUnlockTensorBuffer'),
      runCompiledModel = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              IntPtr,
              IntPtr,
              Pointer<Pointer<Void>>,
              IntPtr,
              Pointer<Pointer<Void>>,
            ),
            int Function(
              Pointer<Void>,
              int,
              int,
              Pointer<Pointer<Void>>,
              int,
              Pointer<Pointer<Void>>,
            )
          >('LiteRtRunCompiledModel'),
      runCompiledModelAsync = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              IntPtr,
              IntPtr,
              Pointer<Pointer<Void>>,
              IntPtr,
              Pointer<Pointer<Void>>,
              Pointer<Uint8>,
            ),
            int Function(
              Pointer<Void>,
              int,
              int,
              Pointer<Pointer<Void>>,
              int,
              Pointer<Pointer<Void>>,
              Pointer<Uint8>,
            )
          >('LiteRtRunCompiledModelAsync'),
      getCompiledModelInputTensorLayout = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              IntPtr,
              IntPtr,
              Pointer<LiteRtLayout>,
            ),
            int Function(Pointer<Void>, int, int, Pointer<LiteRtLayout>)
          >('LiteRtGetCompiledModelInputTensorLayout'),
      getCompiledModelOutputTensorLayouts = dylib
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              IntPtr,
              IntPtr,
              Pointer<LiteRtLayout>,
              Uint8,
            ),
            int Function(Pointer<Void>, int, int, Pointer<LiteRtLayout>, int)
          >('LiteRtGetCompiledModelOutputTensorLayouts'),
      hasTensorBufferEvent = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Uint8>),
            int Function(Pointer<Void>, Pointer<Uint8>)
          >('LiteRtHasTensorBufferEvent'),
      getTensorBufferEvent = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Pointer<Pointer<Void>>),
            int Function(Pointer<Void>, Pointer<Pointer<Void>>)
          >('LiteRtGetTensorBufferEvent'),
      clearTensorBufferEvent = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)
          >('LiteRtClearTensorBufferEvent'),
      waitEvent = dylib
          .lookupFunction<
            Int32 Function(Pointer<Void>, Int64),
            int Function(Pointer<Void>, int)
          >('LiteRtWaitEvent'),
      destroyEvent = dylib
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('LiteRtDestroyEvent');

  final int Function(int, Pointer<LiteRtEnvOption>, Pointer<Pointer<Void>>)
  createEnvironment;
  final void Function(Pointer<Void>) destroyEnvironment;

  final int Function(Pointer<Pointer<Void>>) createOptions;
  final void Function(Pointer<Void>) destroyOptions;
  final int Function(Pointer<Void>, int) setOptionsHardwareAccelerators;

  final int Function(
    Pointer<Utf8>,
    Pointer<Void>,
    Pointer<NativeFunction<LiteRtOpaquePayloadDeleterNative>>,
    Pointer<Pointer<Void>>,
  )
  createOpaqueOptions;
  final void Function(Pointer<Void>) destroyOpaqueOptions;
  final int Function(Pointer<Void>, Pointer<Void>) addOpaqueOptions;

  final int Function(Pointer<Utf8>, Pointer<Pointer<Void>>) createModelFromFile;
  late final int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
  createModelFromBuffer = _dylib
      .lookupFunction<
        Int32 Function(Pointer<Void>, IntPtr, Pointer<Pointer<Void>>),
        int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
      >('LiteRtCreateModelFromBuffer');
  final void Function(Pointer<Void>) destroyModel;

  final int Function(
    Pointer<Void>,
    Pointer<Void>,
    Pointer<Void>,
    Pointer<Pointer<Void>>,
  )
  createCompiledModel;
  final void Function(Pointer<Void>) destroyCompiledModel;

  final int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
  getModelSignature;
  final int Function(Pointer<Void>, Pointer<IntPtr>) getNumSignatureInputs;
  final int Function(Pointer<Void>, Pointer<IntPtr>) getNumSignatureOutputs;
  final int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
  getSignatureInputTensorByIndex;
  final int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
  getSignatureOutputTensorByIndex;
  final int Function(Pointer<Void>, Pointer<LiteRtRankedTensorType>)
  getRankedTensorType;

  final int Function(Pointer<Void>, int, int, Pointer<Pointer<Void>>)
  getCompiledModelInputBufferRequirements;
  final int Function(Pointer<Void>, int, int, Pointer<Pointer<Void>>)
  getCompiledModelOutputBufferRequirements;
  final int Function(Pointer<Void>, Pointer<IntPtr>)
  getTensorBufferRequirementsBufferSize;
  final void Function(Pointer<Void>) destroyTensorBufferRequirements;

  final int Function(
    Pointer<Void>,
    Pointer<LiteRtRankedTensorType>,
    Pointer<Void>,
    Pointer<Pointer<Void>>,
  )
  createManagedTensorBufferFromRequirements;
  final void Function(Pointer<Void>) destroyTensorBuffer;
  final int Function(Pointer<Void>, Pointer<Int32>) getTensorBufferType;
  final int Function(Pointer<Void>, Pointer<Pointer<Void>>, int)
  lockTensorBuffer;
  final int Function(Pointer<Void>) unlockTensorBuffer;

  final int Function(
    Pointer<Void>,
    int,
    int,
    Pointer<Pointer<Void>>,
    int,
    Pointer<Pointer<Void>>,
  )
  runCompiledModel;
  final int Function(
    Pointer<Void>,
    int,
    int,
    Pointer<Pointer<Void>>,
    int,
    Pointer<Pointer<Void>>,
    Pointer<Uint8>,
  )
  runCompiledModelAsync;

  final int Function(Pointer<Void>, int, int, Pointer<LiteRtLayout>)
  getCompiledModelInputTensorLayout;
  final int Function(Pointer<Void>, int, int, Pointer<LiteRtLayout>, int)
  getCompiledModelOutputTensorLayouts;

  final int Function(Pointer<Void>, Pointer<Uint8>) hasTensorBufferEvent;
  final int Function(Pointer<Void>, Pointer<Pointer<Void>>)
  getTensorBufferEvent;
  final int Function(Pointer<Void>) clearTensorBufferEvent;
  final int Function(Pointer<Void>, int) waitEvent;
  final void Function(Pointer<Void>) destroyEvent;

  final DynamicLibrary _dylib;
}

final litert = LiteRtBindings(litertDynamicLibrary);
