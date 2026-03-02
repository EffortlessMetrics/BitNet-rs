//! Wave 28 insta snapshot tests for bitnet-quantization types.

use bitnet_quantization::device_aware_quantizer::{
    CPUQuantizer, QuantizationType as DeviceQuantizationType, ToleranceConfig,
};
use bitnet_quantization::{I2SLayout, QuantizationConfig};

// ── QuantizationConfig ───────────────────────────────────────────────────────

#[test]
fn snapshot_quantization_config_default() {
    let cfg = QuantizationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_quantization_config_tl1() {
    let cfg = QuantizationConfig {
        quantization_type: bitnet_common::QuantizationType::TL1,
        block_size: 64,
        precision: 1e-4,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_quantization_config_tl2() {
    let cfg = QuantizationConfig {
        quantization_type: bitnet_common::QuantizationType::TL2,
        block_size: 128,
        precision: 1e-3,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ── I2SLayout ────────────────────────────────────────────────────────────────

#[test]
fn snapshot_i2s_layout_default() {
    let layout = I2SLayout::default();
    let desc = format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    );
    insta::assert_snapshot!(desc);
}

#[test]
fn snapshot_i2s_layout_block_size_256() {
    let layout = I2SLayout::with_block_size(256);
    let desc = format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    );
    insta::assert_snapshot!(desc);
}

// ── ToleranceConfig ──────────────────────────────────────────────────────────

#[test]
fn snapshot_tolerance_config_default() {
    let cfg = ToleranceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ── Device-aware QuantizationType ────────────────────────────────────────────

#[test]
fn snapshot_device_quantization_type_all_variants() {
    let variants = vec![
        DeviceQuantizationType::I2S,
        DeviceQuantizationType::TL1,
        DeviceQuantizationType::TL2,
        DeviceQuantizationType::IQ2S,
        DeviceQuantizationType::FP32,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snapshot_device_quantization_type_display() {
    let displays: Vec<String> = vec![
        DeviceQuantizationType::I2S,
        DeviceQuantizationType::TL1,
        DeviceQuantizationType::TL2,
        DeviceQuantizationType::IQ2S,
        DeviceQuantizationType::FP32,
    ]
    .into_iter()
    .map(|q| q.to_string())
    .collect();
    insta::assert_debug_snapshot!(displays);
}

// ── Quantize / Dequantize round-trip ─────────────────────────────────────────

#[test]
fn snapshot_cpu_quantize_i2s_known_input() {
    let tol = ToleranceConfig::default();
    let cpu = CPUQuantizer::new(tol);
    let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75];
    let quantized = cpu.quantize_i2s(&input).expect("quantize_i2s");
    insta::assert_debug_snapshot!(quantized);
}

#[test]
fn snapshot_cpu_dequantize_i2s_round_trip() {
    let tol = ToleranceConfig::default();
    let cpu = CPUQuantizer::new(tol);
    let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75];
    let quantized = cpu.quantize_i2s(&input).expect("quantize_i2s");
    let output = cpu.dequantize_i2s(&quantized).expect("dequantize_i2s");
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snapshot_cpu_quantize_tl1_known_input() {
    let tol = ToleranceConfig::default();
    let cpu = CPUQuantizer::new(tol);
    let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75];
    let quantized = cpu.quantize_tl1(&input).expect("quantize_tl1");
    insta::assert_debug_snapshot!(quantized);
}

#[test]
fn snapshot_cpu_dequantize_tl1_round_trip() {
    let tol = ToleranceConfig::default();
    let cpu = CPUQuantizer::new(tol);
    let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75];
    let quantized = cpu.quantize_tl1(&input).expect("quantize_tl1");
    let output = cpu.dequantize_tl1(&quantized).expect("dequantize_tl1");
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snapshot_cpu_quantize_tl2_known_input() {
    let tol = ToleranceConfig::default();
    let cpu = CPUQuantizer::new(tol);
    let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75];
    let quantized = cpu.quantize_tl2(&input).expect("quantize_tl2");
    insta::assert_debug_snapshot!(quantized);
}

// ── Error messages ───────────────────────────────────────────────────────────

#[test]
fn snapshot_quantization_error_invalid_block_size() {
    let err = bitnet_common::QuantizationError::InvalidBlockSize { size: 0 };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snapshot_quantization_error_unsupported_type() {
    let err = bitnet_common::QuantizationError::UnsupportedType { qtype: "Q4_K_M".to_string() };
    insta::assert_snapshot!(err.to_string());
}
