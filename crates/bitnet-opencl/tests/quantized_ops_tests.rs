//! Tests for quantized inference operations.
//!
//! Covers I2_S/QK256 dequantization, ternary matmul/matvec, packing,
//! scale factors, round-trips, and edge cases.

use bitnet_opencl::quantized_kernels;
use bitnet_opencl::quantized_ops;

// ── I2_S dequantization ──────────────────────────────────────────────────────

#[test]
fn i2s_dequant_known_bytes() {
    // byte 0b10_00_01_01 → [+1, +1, 0, -1]
    let packed = vec![0b10_00_01_01u8];
    let result = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(result, vec![1.0, 1.0, 0.0, -1.0]);
}

#[test]
fn i2s_dequant_all_zeros() {
    let packed = vec![0x00u8; 4];
    let result = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert!(result.iter().all(|&v| v == 0.0));
    assert_eq!(result.len(), 16);
}

#[test]
fn i2s_dequant_all_plus_one() {
    // 0b01_01_01_01 = 0x55 → [+1,+1,+1,+1]
    let packed = vec![0x55u8; 3];
    let result = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(result.len(), 12);
    assert!(result.iter().all(|&v| v == 1.0));
}

#[test]
fn i2s_dequant_all_minus_one() {
    // 0b10_10_10_10 = 0xAA → [-1,-1,-1,-1]
    let packed = vec![0xAAu8; 2];
    let result = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(result.len(), 8);
    assert!(result.iter().all(|&v| v == -1.0));
}

#[test]
fn i2s_dequant_scale_applied() {
    let packed = vec![0x55u8]; // all +1
    let result = quantized_ops::dequantize_i2s(&packed, 3.0);
    assert_eq!(result, vec![3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn i2s_dequant_negative_scale() {
    let packed = vec![0x55u8]; // all +1
    let result = quantized_ops::dequantize_i2s(&packed, -2.0);
    assert_eq!(result, vec![-2.0, -2.0, -2.0, -2.0]);
}

#[test]
fn i2s_dequant_zero_scale() {
    let packed = vec![0x55u8, 0xAAu8];
    let result = quantized_ops::dequantize_i2s(&packed, 0.0);
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn i2s_dequant_reserved_bits() {
    // 0b11 → treated as 0
    let packed = vec![0xFFu8]; // 0b11_11_11_11 → [0,0,0,0]
    let result = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn i2s_dequant_mixed_pattern() {
    // 0b00_10_01_00 = weights [0, +1, -1, 0] packed LSB-first
    let byte = 0b00_10_01_00u8;
    let result = quantized_ops::dequantize_i2s(&[byte], 1.0);
    assert_eq!(result, vec![0.0, 1.0, -1.0, 0.0]);
}

#[test]
fn i2s_dequant_empty_input() {
    let result = quantized_ops::dequantize_i2s(&[], 1.0);
    assert!(result.is_empty());
}

// ── QK256 dequantization ─────────────────────────────────────────────────────

#[test]
fn qk256_dequant_too_short() {
    let result = quantized_ops::dequantize_qk256(&[0u8; 65]);
    assert!(result.is_empty());
}

#[test]
fn qk256_dequant_all_zeros() {
    let block = vec![0u8; quantized_ops::QK256_BLOCK_SIZE];
    let result = quantized_ops::dequantize_qk256(&block);
    assert_eq!(result.len(), 256);
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn qk256_dequant_scale_one_all_plus() {
    let mut block = vec![0u8; quantized_ops::QK256_BLOCK_SIZE];
    // f16 1.0 = 0x3C00 (little-endian: [0x00, 0x3C])
    block[0] = 0x00;
    block[1] = 0x3C;
    // Fill weight bytes with all +1: 0x55
    for b in &mut block[2..] {
        *b = 0x55;
    }
    let result = quantized_ops::dequantize_qk256(&block);
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
    }
}

#[test]
fn qk256_dequant_negative_scale() {
    let mut block = vec![0u8; quantized_ops::QK256_BLOCK_SIZE];
    // f16 -2.0 = 0xC000 (little-endian: [0x00, 0xC0])
    block[0] = 0x00;
    block[1] = 0xC0;
    // all +1 weights
    for b in &mut block[2..] {
        *b = 0x55;
    }
    let result = quantized_ops::dequantize_qk256(&block);
    for &v in &result {
        assert!((v - (-2.0)).abs() < 1e-3, "expected ~-2.0, got {v}");
    }
}

#[test]
fn qk256_dequant_output_count() {
    let block = vec![0u8; quantized_ops::QK256_BLOCK_SIZE];
    let result = quantized_ops::dequantize_qk256(&block);
    assert_eq!(result.len(), quantized_ops::QK256_WEIGHTS_PER_BLOCK);
}

// ── Pack / unpack round-trip ─────────────────────────────────────────────────

#[test]
fn pack_unpack_roundtrip_simple() {
    let original = vec![1.0, -1.0, 0.0, 1.0];
    let packed = quantized_ops::pack_i2s(&original);
    let unpacked = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(&unpacked[..original.len()], &original[..]);
}

#[test]
fn pack_unpack_roundtrip_longer() {
    let original = vec![1.0, 0.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0];
    let packed = quantized_ops::pack_i2s(&original);
    let unpacked = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(&unpacked[..original.len()], &original[..]);
}

#[test]
fn pack_unpack_roundtrip_non_multiple_of_4() {
    // 5 elements → 2 packed bytes, unpacked → 8 elements (padded)
    let original = vec![1.0, -1.0, 0.0, 1.0, -1.0];
    let packed = quantized_ops::pack_i2s(&original);
    assert_eq!(packed.len(), 2);
    let unpacked = quantized_ops::dequantize_i2s(&packed, 1.0);
    assert_eq!(unpacked.len(), 8);
    assert_eq!(&unpacked[..5], &original[..]);
    // padding should be zero
    assert_eq!(unpacked[5], 0.0);
    assert_eq!(unpacked[6], 0.0);
    assert_eq!(unpacked[7], 0.0);
}

#[test]
fn pack_empty() {
    let packed = quantized_ops::pack_i2s(&[]);
    assert!(packed.is_empty());
}

// ── Ternary matvec ───────────────────────────────────────────────────────────

#[test]
fn matvec_identity_like() {
    // 4×4 "identity" in ternary: diagonal +1, rest 0
    let mut weights = vec![0.0f32; 16];
    for i in 0..4 {
        weights[i * 4 + i] = 1.0;
    }
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 4, 4);
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matvec_all_zero_weights() {
    let packed = vec![0x00u8; 4]; // 4×4, cols_packed=1
    let input = vec![5.0, 6.0, 7.0, 8.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 4, 4);
    assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn matvec_all_plus_one_weights() {
    // 2×4 matrix, all +1
    let weights = vec![1.0f32; 8];
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 2, 4);
    let sum: f32 = input.iter().sum();
    assert_eq!(output, vec![sum, sum]);
}

#[test]
fn matvec_all_minus_one_weights() {
    let weights = vec![-1.0f32; 8];
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 2, 4);
    let neg_sum: f32 = -input.iter().sum::<f32>();
    assert_eq!(output, vec![neg_sum, neg_sum]);
}

#[test]
fn matvec_scale_factor() {
    let weights = vec![1.0f32; 4]; // 1×4
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 2.5, 1, 4);
    assert!((output[0] - 10.0).abs() < 1e-6); // 4 * 2.5
}

#[test]
fn matvec_mixed_weights() {
    // 1×4 weights: [+1, -1, +1, 0]
    let weights = vec![1.0, -1.0, 1.0, 0.0];
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![3.0, 5.0, 7.0, 11.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 1, 4);
    // 3 - 5 + 7 + 0 = 5
    assert!((output[0] - 5.0).abs() < 1e-6);
}

// ── Ternary matmul (batched) ─────────────────────────────────────────────────

#[test]
fn matmul_batch_of_two() {
    let weights = vec![1.0f32; 8]; // 2×4
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let output = quantized_ops::ternary_matmul(&packed, &input, 1.0, 2, 4, 2);
    assert_eq!(output.len(), 4); // 2 batches × 2 rows
    assert!((output[0] - 10.0).abs() < 1e-6); // sum batch0
    assert!((output[1] - 10.0).abs() < 1e-6);
    assert!((output[2] - 26.0).abs() < 1e-6); // sum batch1
    assert!((output[3] - 26.0).abs() < 1e-6);
}

#[test]
fn matmul_single_element() {
    // 1×1 matrix with +1 weight
    let packed = quantized_ops::pack_i2s(&[1.0]);
    let input = vec![42.0];
    let output = quantized_ops::ternary_matmul(&packed, &input, 1.0, 1, 1, 1);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 42.0).abs() < 1e-6);
}

// ── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn i2s_dequant_single_byte() {
    let result = quantized_ops::dequantize_i2s(&[0x55], 1.0);
    assert_eq!(result.len(), 4);
}

#[test]
fn matvec_256_boundary() {
    // 1×256 weights, all +1
    let weights = vec![1.0f32; 256];
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![1.0f32; 256];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 1, 256);
    assert!((output[0] - 256.0).abs() < 1e-3);
}

#[test]
fn matvec_cols_not_multiple_of_4() {
    // 1×5 weights: [+1, +1, +1, +1, +1]
    let weights = vec![1.0f32; 5];
    let packed = quantized_ops::pack_i2s(&weights);
    let input = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let output = quantized_ops::ternary_matvec(&packed, &input, 1.0, 1, 5);
    // 2+3+4+5+6 = 20
    assert!((output[0] - 20.0).abs() < 1e-6);
}

// ── Kernel source validation ─────────────────────────────────────────────────

#[test]
fn kernel_sources_have_kernel_keyword() {
    assert!(quantized_kernels::DEQUANTIZE_I2S_CL.contains("__kernel"));
    assert!(quantized_kernels::TERNARY_MATMUL_CL.contains("__kernel"));
    assert!(quantized_kernels::QK256_DEQUANT_CL.contains("__kernel"));
}

#[test]
fn kernel_all_sources_returns_three() {
    let sources = quantized_kernels::all_kernel_sources();
    assert_eq!(sources.len(), 3);
    for (name, src) in &sources {
        assert!(!name.is_empty());
        assert!(src.contains(name), "kernel source should contain '{name}'");
    }
}

#[test]
fn kernel_workgroup_sizes_positive() {
    assert!(quantized_kernels::DEQUANTIZE_I2S_WORKGROUP > 0);
    assert!(quantized_kernels::TERNARY_MATMUL_WORKGROUP > 0);
    assert!(quantized_kernels::QK256_DEQUANT_WORKGROUP > 0);
}

#[test]
fn kernel_mem_estimate_scales_with_size() {
    let small = quantized_kernels::ternary_matmul_mem_bytes(16, 16);
    let large = quantized_kernels::ternary_matmul_mem_bytes(256, 256);
    assert!(large > small);
}

#[test]
fn qk256_block_size_constant() {
    assert_eq!(quantized_ops::QK256_BLOCK_SIZE, 66);
    assert_eq!(quantized_ops::QK256_WEIGHTS_PER_BLOCK, 256);
}
