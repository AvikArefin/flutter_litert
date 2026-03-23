/// Data type of a LiteRT tensor.
///
/// This is a platform-independent enum shared across native, web, and
/// unsupported implementations. The integer [value] matches the TfLiteType
/// constants from the C API.
enum TensorType {
  noType(0),
  float32(1),
  int32(2),
  uint8(3),
  int64(4),
  string(5),
  boolean(6),
  int16(7),
  complex64(8),
  int8(9),
  float16(10),
  float64(11),
  complex128(12),
  uint64(13),
  resource(14),
  variant(15),
  uint32(16),
  uint16(17),
  int4(18);

  const TensorType(this.value);

  /// Looks up a [TensorType] by its integer value.
  ///
  /// Returns [TensorType.noType] if the value is not recognised.
  static TensorType fromValue(int value) {
    for (final t in TensorType.values) {
      if (t.value == value) return t;
    }
    return TensorType.noType;
  }

  final int value;

  @override
  String toString() => name;
}
