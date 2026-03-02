use bitnet_qk256_layout_core::{
    Qk256LayoutError, parse_input_shape, parse_qk256_layout, validate_input_cols,
};

#[test]
fn parses_qk256_layout() {
    let layout = parse_qk256_layout("w", &[32, 64]).expect("layout");
    assert_eq!(layout.rows, 32);
    assert_eq!(layout.row_stride_bytes, 64);
    assert_eq!(layout.cols, 256);
}

#[test]
fn rejects_invalid_qk256_rank() {
    let err = parse_qk256_layout("w", &[1, 2, 3]).expect_err("should fail");
    assert!(matches!(err, Qk256LayoutError::InvalidQk256Shape { .. }));
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
