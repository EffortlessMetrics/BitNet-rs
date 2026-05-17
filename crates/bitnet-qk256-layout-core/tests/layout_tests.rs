use bitnet_qk256_layout_core::{
    QK256_BLOCK_COLS, QK256_PACKED_BYTES_PER_BLOCK, Qk256InputShape, Qk256Layout, Qk256LayoutError,
    pack_qk256_codes, parse_input_shape, parse_qk256_layout, qk256_blocks_per_row,
    qk256_packed_len_bytes, qk256_row_stride_bytes, unpack_qk256_codes, validate_input_cols,
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
fn rejects_invalid_qk256_rank_with_weight_name_and_dims() {
    let err = parse_qk256_layout("blk.weight", &[1, 2, 3]).expect_err("should fail");

    assert_eq!(
        err,
        Qk256LayoutError::InvalidQk256Shape {
            weight_name: "blk.weight".to_string(),
            dims: vec![1, 2, 3],
        }
    );
}

#[test]
fn rejects_unaligned_row_stride_with_stride_value() {
    let err = parse_qk256_layout("w", &[32, 96]).expect_err("should fail");

    assert_eq!(err, Qk256LayoutError::InvalidRowStride { row_stride_bytes: 96 });
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
fn block_rounding_covers_zero_exact_and_partial_column_counts() -> Result<(), Qk256LayoutError> {
    assert_eq!(qk256_blocks_per_row(0), 0);
    assert_eq!(qk256_blocks_per_row(1), 1);
    assert_eq!(qk256_blocks_per_row(QK256_BLOCK_COLS), 1);
    assert_eq!(qk256_blocks_per_row(QK256_BLOCK_COLS + 1), 2);
    assert_eq!(qk256_row_stride_bytes(0)?, 0);
    assert_eq!(qk256_row_stride_bytes(QK256_BLOCK_COLS + 1)?, 128);
    Ok(())
}

#[test]
fn from_rows_stride_recovers_multi_block_shape() -> Result<(), Qk256LayoutError> {
    let layout = Qk256Layout::from_rows_stride(2, QK256_PACKED_BYTES_PER_BLOCK * 3)?;

    assert_eq!(layout.rows, 2);
    assert_eq!(layout.cols, QK256_BLOCK_COLS * 3);
    assert_eq!(layout.blocks_per_row, 3);
    assert_eq!(layout.row_stride_bytes, QK256_PACKED_BYTES_PER_BLOCK * 3);
    assert_eq!(layout.packed_len_bytes, QK256_PACKED_BYTES_PER_BLOCK * 6);
    Ok(())
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
fn rejects_out_of_bounds_row_and_block_indices() -> Result<(), Qk256LayoutError> {
    let layout = Qk256Layout::from_rows_cols(2, QK256_BLOCK_COLS * 2)?;

    assert_eq!(
        layout.row_range(2).unwrap_err(),
        Qk256LayoutError::RowOutOfBounds { row: 2, rows: 2 }
    );
    assert_eq!(
        layout.block_range(0, 2).unwrap_err(),
        Qk256LayoutError::BlockOutOfBounds { block: 2, blocks_per_row: 2 }
    );
    Ok(())
}

#[test]
fn validates_exact_packed_length() -> Result<(), Qk256LayoutError> {
    let layout = Qk256Layout::from_rows_cols(2, 512)?;
    layout.validate_packed_len(256)?;
    Ok(())
}

#[test]
fn rejects_packed_length_mismatch_with_expected_length() {
    let layout = Qk256Layout::from_rows_cols(2, 512).expect("layout");

    let err = layout.validate_packed_len(255).expect_err("should fail");

    assert_eq!(
        err,
        Qk256LayoutError::PackedLengthMismatch {
            rows: 2,
            cols: 512,
            actual_len: 255,
            expected_len: 256,
        }
    );
}

#[test]
fn pack_unpack_fixture_is_byte_exact() -> Result<(), Qk256LayoutError> {
    let mut codes = [0u8; QK256_BLOCK_COLS];
    for (offset, code) in codes.iter_mut().enumerate() {
        *code = (offset % 4) as u8;
    }

    let packed = pack_qk256_codes(&codes)?;
    assert_eq!(packed, [0b11_10_01_00u8; QK256_PACKED_BYTES_PER_BLOCK]);
    assert_eq!(unpack_qk256_codes(&packed), codes);
    Ok(())
}

#[test]
fn pack_qk256_codes_places_two_bit_values_in_little_endian_slots() -> Result<(), Qk256LayoutError> {
    let mut codes = [0u8; QK256_BLOCK_COLS];
    codes[0] = 3;
    codes[1] = 2;
    codes[2] = 1;
    codes[3] = 0;
    codes[4] = 1;
    codes[5] = 1;
    codes[6] = 2;
    codes[7] = 2;

    let packed = pack_qk256_codes(&codes)?;

    assert_eq!(packed[0], 0b00_01_10_11);
    assert_eq!(packed[1], 0b10_10_01_01);
    assert_eq!(unpack_qk256_codes(&packed), codes);
    Ok(())
}

#[test]
fn rejects_invalid_pack_code_with_first_invalid_offset() {
    let mut codes = [0u8; QK256_BLOCK_COLS];
    codes[17] = 4;
    codes[18] = 5;

    let err = pack_qk256_codes(&codes).expect_err("should fail");
    assert_eq!(err, Qk256LayoutError::InvalidCode { offset: 17, code: 4 });
}

#[test]
fn parses_2d_input_shape() {
    let shape = parse_input_shape(&[4, 256]).expect("shape");
    assert_eq!(shape.batch_size, 4);
    assert_eq!(shape.seq_len, 1);
    assert_eq!(shape.cols, 256);
    assert_eq!(shape.input_rank, 2);
}

#[test]
fn parses_3d_input_shape_with_sequence_length() -> Result<(), Qk256LayoutError> {
    assert_eq!(
        parse_input_shape(&[2, 8, 1024])?,
        Qk256InputShape { batch_size: 2, seq_len: 8, cols: 1024, input_rank: 3 }
    );
    Ok(())
}

#[test]
fn rejects_input_shape_other_than_2d_or_3d_with_dims() {
    let err = parse_input_shape(&[1, 2, 3, 4]).expect_err("should fail");

    assert_eq!(err, Qk256LayoutError::UnsupportedInputShape { dims: vec![1, 2, 3, 4] });
}

#[test]
fn rejects_column_mismatch_with_weight_name_and_dimensions() {
    let err = validate_input_cols("layer", 255, 256).expect_err("should fail");

    assert_eq!(
        err,
        Qk256LayoutError::DimensionMismatch {
            weight_name: "layer".to_string(),
            input_cols: 255,
            expected_cols: 256,
        }
    );
}
