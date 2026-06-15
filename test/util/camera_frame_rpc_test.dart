import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('cameraFrameRpcFields -> cameraFrameFromRpcMessage round-trips', () {
    final frame = CameraFrame(
      bytes: Uint8List.fromList(List<int>.generate(48, (i) => i)),
      width: 4,
      height: 4,
      strideCols: 4,
      conversion: CameraFrameConversion.yuv2bgrNv21,
      rotation: CameraFrameRotation.cw90,
    );

    final map = cameraFrameRpcFields(frame, {'maxDim': 640});
    expect(map['maxDim'], 640);
    expect(map['conversion'], CameraFrameConversion.yuv2bgrNv21.index);
    expect(map['rotation'], CameraFrameRotation.cw90.index);

    final bytes = (map['bytes'] as TransferableTypedData)
        .materialize()
        .asUint8List();
    final rebuilt = cameraFrameFromRpcMessage(map, bytes);
    expect(rebuilt.width, 4);
    expect(rebuilt.height, 4);
    expect(rebuilt.strideCols, 4);
    expect(rebuilt.conversion, CameraFrameConversion.yuv2bgrNv21);
    expect(rebuilt.rotation, CameraFrameRotation.cw90);
    expect(rebuilt.bytes, frame.bytes);
  });

  test('null rotation round-trips', () {
    final frame = CameraFrame(
      bytes: Uint8List(12),
      width: 2,
      height: 2,
      strideCols: 2,
      conversion: CameraFrameConversion.bgra2bgr,
      rotation: null,
    );
    final map = cameraFrameRpcFields(frame);
    expect(map['rotation'], isNull);
    final bytes = (map['bytes'] as TransferableTypedData)
        .materialize()
        .asUint8List();
    final rebuilt = cameraFrameFromRpcMessage(map, bytes);
    expect(rebuilt.rotation, isNull);
    expect(rebuilt.conversion, CameraFrameConversion.bgra2bgr);
  });
}
