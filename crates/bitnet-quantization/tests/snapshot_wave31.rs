//! Wave 31 snapshot tests for bitnet-quantization.
//!
//! Covers: QuantizationType Display/Debug for all variants, I2S block format
//! serialization, QK256 config defaults, TL1/TL2 table dimensions,
//! quantization error messages.

use bitnet_common::{QuantizationError, QuantizationType};
use bitnet_quantization::i2s::I2SLayout;
use bitnet_quantization::tl1::TL1Config;
use bitnet_quantization::tl2::TL2Config;

// ── QuantizationType Display for all variants ───────────────────────────────

#[test]
fn quantization_type_i2s_display() {
    insta::assert_snapshot!(format!("{}", QuantizationType::I2S));
}

#[test]
fn quantization_type_tl1_display() {
    insta::assert_snapshot!(format!("{}", QuantizationType::TL1));
}

#[test]
fn quantization_type_tl2_display() {
    insta::assert_snapshot!(format!("{}", QuantizationType::TL2));
}

// ── QuantizationType Debug for all variants ─────────────────────────────────

#[test]
fn quantization_type_i2s_debug() {
    insta::assert_debug_snapshot!(QuantizationType::I2S);
}

#[test]
fn quantization_type_tl1_debug() {
    insta::assert_debug_snapshot!(QuantizationType::TL1);
}

#[test]
fn quantization_type_tl2_debug() {
    insta::assert_debug_snapshot!(QuantizationType::TL2);
}

// ── I2S block format serialization ──────────────────────────────────────────

#[test]
fn i2s_layout_default_debug() {
    let layout = I2SLayout::default();
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block
    ));
}

#[test]
fn i2s_layout_block_128() {
    let layout = I2SLayout::with_block_size(128);
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block
    ));
}

// ── QK256 config defaults ───────────────────────────────────────────────────

#[test]
fn qk256_block_constants() {
    insta::assert_snapshot!(format!(
        "QK256_BLOCK={} QK256_PACKED_BYTES={} QK256_SIZE_TOLERANCE={}",
        bitnet_quantization::i2s_qk256::QK256_BLOCK,
        bitnet_quantization::i2s_qk256::QK256_PACKED_BYTES,
        bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT,
    ));
}

// ── TL1/TL2 table dimensions ───────────────────────────────────────────────

#[test]
fn tl1_config_default_debug() {
    insta::assert_debug_snapshot!(TL1Config::default());
}

#[test]
fn tl2_config_default_debug() {
    insta::assert_debug_snapshot!(TL2Config::default());
}

#[test]
fn tl1_tl2_dimension_comparison() {
    let tl1 = TL1Config::default();
    let tl2 = TL2Config::default();
    insta::assert_snapshot!(format!(
        "tl1(block={} table={} bits={}) tl2(block={} table={} bits={})",
        tl1.block_size,
        tl1.lookup_table_size,
        tl1.precision_bits,
        tl2.block_size,
        tl2.lookup_table_size,
        tl2.precision_bits
    ));
}

// ── Quantization error messages ─────────────────────────────────────────────

#[test]
fn error_unsupported_type() {
    let err = QuantizationError::UnsupportedType { qtype: "Q8_0".to_string() };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_quantization_failed() {
    let err = QuantizationError::QuantizationFailed {
        reason: "block size mismatch: expected 256, got 128".to_string(),
    };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_invalid_block_size() {
    let err = QuantizationError::InvalidBlockSize { size: 7 };
    insta::assert_snapshot!(err.to_string());
}
