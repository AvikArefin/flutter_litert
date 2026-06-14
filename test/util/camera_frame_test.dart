import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('PackedImageLayout', () {
    test('copyTo validates source and destination byte length', () {
      const layout = PackedImageLayout(
        rows: 2,
        cols: 3,
        channels: 4,
        format: PackedImageFormat.bgra8888,
      );
      final src = Uint8List.fromList(List<int>.generate(24, (i) => i));
      final dst = Uint8List(24);

      layout.copyTo(dst, src);

      expect(dst, src);
      expect(
        () => layout.copyTo(Uint8List(23), src),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => layout.copyTo(dst, Uint8List(23)),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('CameraFrame.decodePlan', () {
    test(
      'BGRA uses stride columns and resize/rotate before colour conversion',
      () {
        final frame = CameraFrame(
          bytes: Uint8List(2 * 5 * 4),
          width: 3,
          height: 2,
          strideCols: 5,
          conversion: CameraFrameConversion.bgra2bgr,
          rotation: CameraFrameRotation.cw90,
        );

        final plan = frame.decodePlan();

        expect(plan.sourceLayout.rows, 2);
        expect(plan.sourceLayout.cols, 5);
        expect(plan.sourceLayout.channels, 4);
        expect(plan.sourceLayout.format, PackedImageFormat.bgra8888);
        expect(plan.sourceLayout.byteLength, frame.bytes.length);
        expect(plan.visibleWidth, 3);
        expect(plan.visibleHeight, 2);
        expect(plan.hasStridePadding, isTrue);
        expect(plan.conversion, CameraFrameConversion.bgra2bgr);
        expect(plan.rotation, CameraFrameRotation.cw90);
        expect(plan.order, CameraFrameDecodeOrder.resizeRotateThenColorConvert);
      },
    );

    test('RGBA reports no stride padding when stride matches width', () {
      final frame = CameraFrame(
        bytes: Uint8List(2 * 3 * 4),
        width: 3,
        height: 2,
        strideCols: 3,
        conversion: CameraFrameConversion.rgba2bgr,
      );

      final plan = frame.decodePlan();

      expect(plan.sourceLayout.format, PackedImageFormat.rgba8888);
      expect(plan.hasStridePadding, isFalse);
    });

    test('YUV uses packed luma/chroma layout and colour conversion first', () {
      final frame = CameraFrame(
        bytes: Uint8List(4 * 4 * 3 ~/ 2),
        width: 4,
        height: 4,
        strideCols: 4,
        conversion: CameraFrameConversion.yuv2bgrNv21,
        rotation: CameraFrameRotation.cw180,
      );

      final plan = frame.decodePlan();

      expect(plan.sourceLayout.rows, 6);
      expect(plan.sourceLayout.cols, 4);
      expect(plan.sourceLayout.channels, 1);
      expect(plan.sourceLayout.format, PackedImageFormat.yuv420Nv21);
      expect(plan.sourceLayout.byteLength, frame.bytes.length);
      expect(plan.visibleWidth, 4);
      expect(plan.visibleHeight, 4);
      expect(plan.hasStridePadding, isFalse);
      expect(plan.order, CameraFrameDecodeOrder.colorConvertThenResizeRotate);
    });
  });

  group('PackedYuv.sourceLayout', () {
    test('matches packed bytes for I420', () {
      final packed = PackedYuv(
        bytes: Uint8List(4 * 4 * 3 ~/ 2),
        layout: YuvLayout.i420,
        width: 4,
        height: 4,
      );

      final layout = packed.sourceLayout;

      expect(layout.rows, 6);
      expect(layout.cols, 4);
      expect(layout.channels, 1);
      expect(layout.format, PackedImageFormat.yuv420I420);
      expect(layout.byteLength, packed.bytes.length);
    });
  });
}
