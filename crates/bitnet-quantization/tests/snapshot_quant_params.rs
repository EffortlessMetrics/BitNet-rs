//! Wave 8 snapshot tests — quantization parameters and layouts.
//!
//! Pins the configuration defaults and scale-factor arithmetic so that
//! any silent change to block sizes, byte layouts, or numeric constants
//! is caught at review time.

use bitnet_common::QuantizationType;
use bitnet_quantization::I2SLayout;
use bitnet_quantization::tl1::TL1Config;
use bitnet_quantization::tl2::TL2Config;
use bitnet_quantization::utils::{calculate_grouped_scales, calculate_scale};

// ---------------------------------------------------------------------------
// I2SLayout defaults and custom sizes
// ---------------------------------------------------------------------------

#[test]
fn i2s_layout_default_snapshot() {
    let l = I2SLayout::default();
    insta::assert_snapshot!(
        "i2s_layout_default",
        format!(
            "block_size={} bytes_per_block={} data={} scale={}",
            l.block_size, l.bytes_per_block, l.data_bytes_per_block, l.scale_bytes_per_block,
        )
    );
}

#[test]
fn i2s_layout_block_64_snapshot() {
    let l = I2SLayout::with_block_size(64);
    insta::assert_snapshot!(
        "i2s_layout_block_64",
        format!(
            "block_size={} bytes_per_block={} data={} scale={}",
            l.block_size, l.bytes_per_block, l.data_bytes_per_block, l.scale_bytes_per_block,
        )
    );
}

#[test]
fn i2s_layout_block_256_snapshot() {
    let l = I2SLayout::with_block_size(256);
    insta::assert_snapshot!(
        "i2s_layout_block_256",
        format!(
            "block_size={} bytes_per_block={} data={} scale={}",
            l.block_size, l.bytes_per_block, l.data_bytes_per_block, l.scale_bytes_per_block,
        )
    );
}

// ---------------------------------------------------------------------------
// TL1Config defaults
// ---------------------------------------------------------------------------

#[test]
fn tl1_config_default_snapshot() {
    let config = TL1Config::default();
    insta::assert_debug_snapshot!("tl1_config_default", config);
}

// ---------------------------------------------------------------------------
// TL2Config defaults
// ---------------------------------------------------------------------------

#[test]
fn tl2_config_default_snapshot() {
    let config = TL2Config::default();
    // Redact runtime-detected SIMD fields that differ per machine.
    insta::assert_snapshot!(
        "tl2_config_default",
        format!(
            "TL2Config {{ block_size: {}, lookup_table_size: {}, precision_bits: {}, vectorized_tables: {} }}",
            config.block_size,
            config.lookup_table_size,
            config.precision_bits,
            config.vectorized_tables,
        )
    );
}

// ---------------------------------------------------------------------------
// QuantizationConfig (from bitnet-common)
// ---------------------------------------------------------------------------

#[test]
fn quantization_config_default_snapshot() {
    let config = bitnet_common::config::QuantizationConfig::default();
    insta::assert_debug_snapshot!("quantization_config_default", config);
}

#[test]
fn quantization_config_per_type_snapshot() {
    let configs: Vec<_> = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2]
        .iter()
        .map(|&qt| {
            let cfg = bitnet_common::config::QuantizationConfig {
                quantization_type: qt,
                ..Default::default()
            };
            format!("{}: block_size={}, precision={}", qt, cfg.block_size, cfg.precision)
        })
        .collect();
    insta::assert_debug_snapshot!("quantization_config_per_type", configs);
}

// ---------------------------------------------------------------------------
// Scale factor computation — known inputs
// ---------------------------------------------------------------------------

#[test]
fn scale_factor_uniform_ones_2bit() {
    let data = vec![1.0f32; 32];
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_uniform_ones_2bit", format!("{scale:.6}"));
}

#[test]
fn scale_factor_mixed_values_2bit() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_mixed_values_2bit", format!("{scale:.6}"));
}

#[test]
fn scale_factor_all_zeros() {
    let data = vec![0.0f32; 64];
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_all_zeros", format!("{scale:.6}"));
}

#[test]
fn scale_factor_contains_nan() {
    let mut data = vec![1.0f32; 32];
    data[5] = f32::NAN;
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_with_nan", format!("{scale:.6}"));
}

#[test]
fn scale_factor_contains_inf() {
    let mut data = vec![0.5f32; 32];
    data[10] = f32::INFINITY;
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_with_inf", format!("{scale:.6}"));
}

// ---------------------------------------------------------------------------
// Grouped scale factors
// ---------------------------------------------------------------------------

#[test]
fn grouped_scales_two_blocks() {
    // Two blocks of 4 elements each with distinct max values.
    let data = vec![1.0, 0.5, -1.0, 0.25, 2.0, -2.0, 0.0, 0.5];
    let scales = calculate_grouped_scales(&data, 4, 2);
    let formatted: Vec<String> = scales.iter().map(|s| format!("{s:.6}")).collect();
    insta::assert_debug_snapshot!("grouped_scales_two_blocks", formatted);
}

#[test]
fn grouped_scales_partial_last_block() {
    // 6 elements with block_size=4 → 2 blocks, second is partial.
    let data = vec![3.0, -1.0, 0.0, 0.5, 0.25, -0.25];
    let scales = calculate_grouped_scales(&data, 4, 2);
    let formatted: Vec<String> = scales.iter().map(|s| format!("{s:.6}")).collect();
    insta::assert_debug_snapshot!("grouped_scales_partial_block", formatted);
}

// ---------------------------------------------------------------------------
// Block layout descriptions
// ---------------------------------------------------------------------------

#[test]
fn i2s_layout_byte_budget_description() {
    let layout = I2SLayout::default();
    let description = format!(
        "I2S block: {} elements → {} total bytes (data={}, scale={})",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    );
    insta::assert_snapshot!("i2s_block_byte_budget", description);
}

#[test]
fn i2s_compression_ratio() {
    let layout = I2SLayout::default();
    // Original: block_size elements × 4 bytes (f32)
    let original_bytes = layout.block_size * 4;
    let ratio = original_bytes as f64 / layout.bytes_per_block as f64;
    insta::assert_snapshot!("i2s_compression_ratio", format!("{ratio:.2}x"));
}

#[test]
fn i2s_layout_block_256_byte_budget() {
    let layout = I2SLayout::with_block_size(256);
    let description = format!(
        "I2S block: {} elements → {} total bytes (data={}, scale={})",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    );
    insta::assert_snapshot!("i2s_block_256_byte_budget", description);
}
