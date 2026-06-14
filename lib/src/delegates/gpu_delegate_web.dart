/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import '../web/delegate.dart';

/// GPU delegate for Android (no-op on web)
@Deprecated(
  'Manual hardware-acceleration delegates are superseded by LiteRT Next. Use '
  'CompiledModel.fromFile / CompiledModel.fromBuffer with '
  'accelerators: {Accelerator.gpu, Accelerator.cpu}, or '
  'CompiledModel.fromBufferWithGpuFallback. The Interpreter API itself remains '
  'supported for CPU inference. '
  'See https://developers.google.com/edge/litert/next/get_started; planned '
  'for removal in flutter_litert 4.0.0.',
)
class GpuDelegateV2 extends Delegate {
  GpuDelegateV2({GpuDelegateOptionsV2? options});

  @override
  void delete() {}
}

/// GPU delegate options for Android (no-op on web)
@Deprecated(
  'Options for a deprecated delegate. Configure acceleration through '
  'CompiledModel (accelerators / Precision) instead. '
  'Planned for removal in flutter_litert 4.0.0.',
)
class GpuDelegateOptionsV2 {
  GpuDelegateOptionsV2({
    bool isPrecisionLossAllowed = false,
    int inferencePreference = 0,
    int inferencePriority1 = 0,
    int inferencePriority2 = 0,
    int inferencePriority3 = 0,
    List<int> experimentalFlags = const [],
    int maxDelegatePartitions = 1,
    String? serializationDir,
    String? modelToken,
  });

  void delete() {}
}
