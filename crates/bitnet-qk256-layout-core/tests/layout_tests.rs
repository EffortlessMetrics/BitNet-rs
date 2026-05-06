use bitnet_qk256_layout_core::{
    QK256_BLOCK_COLS, QK256_PACKED_BYTES_PER_BLOCK, Qk256Layout, Qk256LayoutError,
    pack_qk256_codes, parse_input_shape, parse_qk256_layout, qk256_packed_len_bytes,
    qk256_row_stride_bytes, unpack_qk256_codes, validate_input_cols,
};

#[test]
fn parses_qk256_layout() {
    let layout = parse_qk256_layout("w", &[32, 64]).expect("layout");
    assert_eq!(layout.rows, 32);
    assert_eq!(layout.row_stride_bytes, 64);
    assert_eq!(layout.cols, 256);
    assert_eq!(layout.blocks_per_row, 1);
    assert_eq!(layout.packed_len_bytes, 32 * 64);
}

#[test]
fn rejects_invalid_qk256_rank() {
    let err = parse_qk256_layout("w", &[1, 2, 3]).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::InvalidQk256Shape { .. }));
}

#[test]
fn rejects_unaligned_row_stride() {
    let err = parse_qk256_layout("w", &[32, 96]).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::InvalidRowStride { .. }));
}

#[test]
fn computes_canonical_geometry_from_rows_cols() {
    let layout = Qk256Layout::from_rows_cols(7, 257).expect("layout");
    assert_eq!(layout.rows, 7);
    assert_eq!(layout.cols, 257);
    assert_eq!(layout.blocks_per_row, 2);
    assert_eq!(layout.row_stride_bytes, 128);
    assert_eq!(layout.packed_len_bytes, 896);
    assert_eq!(qk256_row_stride_bytes(257).expect("stride"), 128);
    assert_eq!(qk256_packed_len_bytes(7, 257).expect("packed len"), 896);
}

#[test]
fn reports_row_and_block_ranges() {
    let layout = Qk256Layout::from_rows_cols(3, 512).expect("layout");
    assert_eq!(layout.row_range(1).expect("row"), 128..256);
    assert_eq!(layout.block_range(1, 0).expect("block"), 128..192);
    assert_eq!(layout.block_range(1, 1).expect("block"), 192..256);

    let rows: Vec<_> = layout.row_ranges().collect();
    assert_eq!(rows, vec![0..128, 128..256, 256..384]);
}

#[test]
fn validates_exact_packed_length() {
    let layout = Qk256Layout::from_rows_cols(2, 512).expect("layout");
    layout.validate_packed_len(256).expect("exact length");

    let err = layout.validate_packed_len(255).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::PackedLengthMismatch { .. }));
}

#[test]
fn pack_unpack_fixture_is_byte_exact() {
    let mut codes = [0u8; QK256_BLOCK_COLS];
    for (offset, code) in codes.iter_mut().enumerate() {
        *code = (offset % 4) as u8;
    }

    let packed = pack_qk256_codes(&codes).expect("pack");
    assert_eq!(packed, [0b11_10_01_00u8; QK256_PACKED_BYTES_PER_BLOCK]);
    assert_eq!(unpack_qk256_codes(&packed), codes);
}

#[test]
fn rejects_invalid_pack_code() {
    let mut codes = [0u8; QK256_BLOCK_COLS];
    codes[17] = 4;

    let err = pack_qk256_codes(&codes).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::InvalidCode { offset: 17, code: 4 }));
}

#[test]
fn parses_2d_input_shape() {
    let shape = parse_input_shape(&[4, 256]).expect("shape");
    assert_eq!(shape.batch_size, 4);
    assert_eq!(shape.seq_len, 1);
    assert_eq!(shape.cols, 256);
}

#[test]
fn rejects_input_shape_other_than_2d_or_3d() {
    let err = parse_input_shape(&[256]).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::UnsupportedInputShape { .. }));
}

#[test]
fn rejects_column_mismatch() {
    let err = validate_input_cols("layer", 255, 256).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::DimensionMismatch { .. }));
}
