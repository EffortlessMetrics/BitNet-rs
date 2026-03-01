//! Wave 6 snapshot tests for bitnet-quantization.
//!
//! Pins QuantizationType Display output, I2S quantization of known vectors,
//! scale factor computation, and block size configurations.

use bitnet_common::QuantizationType;
use bitnet_quantization::QuantizedTensor;
use bitnet_quantization::i2s::{I2SLayout, I2SQuantizer};
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_scale, pack_2bit_values, unpack_2bit_values,
};

// ── QuantizationType Display for each variant ───────────────────────────────

#[test]
fn snapshot_quantization_type_display_i2s() {
    insta::assert_snapshot!("qtype_display_i2s", QuantizationType::I2S.to_string());
}

#[test]
fn snapshot_quantization_type_display_tl1() {
    insta::assert_snapshot!("qtype_display_tl1", QuantizationType::TL1.to_string());
}

#[test]
fn snapshot_quantization_type_display_tl2() {
    insta::assert_snapshot!("qtype_display_tl2", QuantizationType::TL2.to_string());
}

#[test]
fn snapshot_quantization_type_debug_all() {
    let all = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    insta::assert_json_snapshot!("qtype_debug_all", all);
}

// ── I2S quantization of known input vectors ─────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn snapshot_i2s_quantize_known_vector() {
    let data: Vec<f32> = vec![
        1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.75, -0.75, 1.0, -1.0, 0.5, -0.5, 0.0, 0.25,
        -0.25, 0.75, -0.75, 1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.75, -0.75, 1.0, -1.0, 0.5,
        -0.5, 0.0,
    ];
    let tensor =
        bitnet_common::BitNetTensor::from_slice(&data, &[32], &bitnet_common::Device::Cpu).unwrap();
    let quantizer = I2SQuantizer::new();
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();

    // Snapshot metadata (data bytes are platform-independent for this input)
    insta::assert_snapshot!(
        "i2s_known_vector_meta",
        format!(
            "shape={:?} qtype={} block_size={} numel={} scales_len={} data_len={}",
            quantized.shape,
            quantized.qtype,
            quantized.block_size,
            quantized.numel(),
            quantized.scales.len(),
            quantized.data.len(),
        )
    );
}

#[cfg(feature = "cpu")]
#[test]
fn snapshot_i2s_quantize_zeros() {
    let data = vec![0.0f32; 32];
    let tensor =
        bitnet_common::BitNetTensor::from_slice(&data, &[32], &bitnet_common::Device::Cpu).unwrap();
    let quantizer = I2SQuantizer::new();
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();

    insta::assert_snapshot!(
        "i2s_zeros_meta",
        format!(
            "shape={:?} qtype={} numel={} compression_ratio={:.1}",
            quantized.shape,
            quantized.qtype,
            quantized.numel(),
            quantized.compression_ratio(),
        )
    );
}

// ── Scale factor computation ────────────────────────────────────────────────

#[test]
fn snapshot_scale_factor_uniform() {
    let data: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5];
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_uniform", format!("{scale:.6}"));
}

#[test]
fn snapshot_scale_factor_zeros() {
    let data = vec![0.0f32; 8];
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_zeros", format!("{scale:.6}"));
}

#[test]
fn snapshot_scale_factor_large_range() {
    let data: Vec<f32> = vec![100.0, -50.0, 25.0, -12.5];
    let scale = calculate_scale(&data, 2);
    insta::assert_snapshot!("scale_large_range", format!("{scale:.6}"));
}

#[test]
fn snapshot_grouped_scales() {
    let data: Vec<f32> = vec![
        1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.75, // block 1 (max=1.0)
        2.0, -2.0, 1.5, -1.5, 0.0, 0.5, -0.5, 1.0, // block 2 (max=2.0)
    ];
    let scales = calculate_grouped_scales(&data, 8, 2);
    let formatted: Vec<String> = scales.iter().map(|s| format!("{s:.6}")).collect();
    insta::assert_json_snapshot!("grouped_scales_two_blocks", formatted);
}

// ── Block size configurations ───────────────────────────────────────────────

#[test]
fn snapshot_i2s_layout_default() {
    let layout = I2SLayout::default();
    insta::assert_snapshot!(
        "i2s_layout_default",
        format!(
            "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
            layout.block_size,
            layout.bytes_per_block,
            layout.data_bytes_per_block,
            layout.scale_bytes_per_block,
        )
    );
}

#[test]
fn snapshot_i2s_layout_custom_64() {
    let layout = I2SLayout::with_block_size(64);
    insta::assert_snapshot!(
        "i2s_layout_64",
        format!(
            "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
            layout.block_size,
            layout.bytes_per_block,
            layout.data_bytes_per_block,
            layout.scale_bytes_per_block,
        )
    );
}

#[test]
fn snapshot_i2s_layout_custom_256() {
    let layout = I2SLayout::with_block_size(256);
    insta::assert_snapshot!(
        "i2s_layout_256",
        format!(
            "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
            layout.block_size,
            layout.bytes_per_block,
            layout.data_bytes_per_block,
            layout.scale_bytes_per_block,
        )
    );
}

// ── Pack / unpack round-trip ────────────────────────────────────────────────

#[test]
fn snapshot_pack_2bit_known_values() {
    let values: Vec<i8> = vec![1, 0, -1, -2, 1, -1, 0, -2];
    let packed = pack_2bit_values(&values);
    let unpacked = unpack_2bit_values(&packed, values.len());
    insta::assert_snapshot!(
        "pack_2bit_roundtrip",
        format!("packed={:?} unpacked={:?}", packed, unpacked)
    );
}

// ── QuantizedTensor metadata snapshot ───────────────────────────────────────

#[test]
fn snapshot_quantized_tensor_new() {
    let qt = QuantizedTensor::new(vec![0u8; 8], vec![1.0], vec![32], QuantizationType::I2S);
    insta::assert_snapshot!(
        "quantized_tensor_new_meta",
        format!(
            "shape={:?} qtype={} block_size={} numel={} compression_ratio={:.1}",
            qt.shape,
            qt.qtype,
            qt.block_size,
            qt.numel(),
            qt.compression_ratio(),
        )
    );
}
