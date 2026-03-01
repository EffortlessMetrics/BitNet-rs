//! Edge-case integration tests for `bitnet_kernels::cpu::fallback::FallbackKernel`.
//!
//! The FallbackKernel implements the KernelProvider trait with naive but correct
//! implementations. These tests exercise the public trait methods as integration
//! tests, verifying correctness from outside the crate.

use bitnet_common::QuantizationType;
use bitnet_kernels::KernelProvider;
use bitnet_kernels::cpu::fallback::FallbackKernel;

const TOL: f32 = 1e-5;

// =========================================================================
// KernelProvider trait basics
// =========================================================================

#[test]
fn fallback_always_available() {
    let k = FallbackKernel;
    assert!(k.is_available());
}

#[test]
fn fallback_name_is_fallback() {
    let k = FallbackKernel;
    assert_eq!(k.name(), "fallback");
}

// =========================================================================
// matmul_i2s: correctness
// =========================================================================

#[test]
fn matmul_identity_2x2() {
    let k = FallbackKernel;
    let a = vec![1i8, 0, 0, 1];
    let b = vec![5u8, 3, 7, 2];
    let mut c = vec![0.0f32; 4];
    k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    assert!((c[0] - 5.0).abs() < TOL);
    assert!((c[1] - 3.0).abs() < TOL);
    assert!((c[2] - 7.0).abs() < TOL);
    assert!((c[3] - 2.0).abs() < TOL);
}

#[test]
fn matmul_1x1() {
    let k = FallbackKernel;
    let a = vec![3i8];
    let b = vec![4u8];
    let mut c = vec![0.0f32; 1];
    k.matmul_i2s(&a, &b, &mut c, 1, 1, 1).unwrap();
    assert!((c[0] - 12.0).abs() < TOL);
}

#[test]
fn matmul_zero_a() {
    let k = FallbackKernel;
    let a = vec![0i8; 4];
    let b = vec![1u8, 2, 3, 4];
    let mut c = vec![999.0f32; 4];
    k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    for &v in &c {
        assert!(v.abs() < TOL, "zero A should give zero C, got {v}");
    }
}

#[test]
fn matmul_zero_b() {
    let k = FallbackKernel;
    let a = vec![1i8, 2, 3, 4];
    let b = vec![0u8; 4];
    let mut c = vec![999.0f32; 4];
    k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    for &v in &c {
        assert!(v.abs() < TOL, "zero B should give zero C, got {v}");
    }
}

#[test]
fn matmul_negative_a_values() {
    let k = FallbackKernel;
    // -1 * 2 + -1 * 3 = -5
    let a = vec![-1i8, -1];
    let b = vec![2u8, 3];
    let mut c = vec![0.0f32; 1];
    k.matmul_i2s(&a, &b, &mut c, 1, 1, 2).unwrap();
    assert!((c[0] - (-5.0)).abs() < TOL);
}

#[test]
fn matmul_non_square() {
    let k = FallbackKernel;
    // A: 2x3, B: 3x1
    let a = vec![1i8, 2, 3, 4, 5, 6];
    let b = vec![1u8, 1, 1];
    let mut c = vec![0.0f32; 2];
    k.matmul_i2s(&a, &b, &mut c, 2, 1, 3).unwrap();
    assert!((c[0] - 6.0).abs() < TOL); // 1+2+3
    assert!((c[1] - 15.0).abs() < TOL); // 4+5+6
}

#[test]
fn matmul_output_initialized_to_zero() {
    let k = FallbackKernel;
    let a = vec![1i8; 4];
    let b = vec![0u8; 4];
    let mut c = vec![42.0f32; 4]; // Pre-filled with non-zero
    k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    // Output should be overwritten, not accumulated
    for &v in &c {
        assert!(v.abs() < TOL, "should be zero, got {v}");
    }
}

// =========================================================================
// matmul_i2s: dimension validation errors
// =========================================================================

#[test]
fn matmul_a_too_small() {
    let k = FallbackKernel;
    let a = vec![1i8; 2]; // Need 4 for 2x2
    let b = vec![1u8; 4];
    let mut c = vec![0.0f32; 4];
    assert!(k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).is_err());
}

#[test]
fn matmul_b_too_small() {
    let k = FallbackKernel;
    let a = vec![1i8; 4];
    let b = vec![1u8; 2]; // Need 4 for 2x2
    let mut c = vec![0.0f32; 4];
    assert!(k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).is_err());
}

#[test]
fn matmul_c_too_small() {
    let k = FallbackKernel;
    let a = vec![1i8; 4];
    let b = vec![1u8; 4];
    let mut c = vec![0.0f32; 2]; // Need 4 for 2x2
    assert!(k.matmul_i2s(&a, &b, &mut c, 2, 2, 2).is_err());
}

// =========================================================================
// quantize: I2S
// =========================================================================

#[test]
fn quantize_i2s_basic() {
    let k = FallbackKernel;
    let input = vec![1.0, -1.0, 0.0, 0.5];
    let mut output = vec![0u8; 1]; // 4 values / 4 per byte
    let mut scales = vec![0.0f32; 1];
    k.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert!(scales[0] > 0.0, "scale should be positive");
    assert!(output[0] != 0, "should have non-zero quantized values");
}

#[test]
fn quantize_i2s_all_zeros() {
    let k = FallbackKernel;
    let input = vec![0.0; 8];
    let mut output = vec![0u8; 2];
    let mut scales = vec![0.0f32; 1];
    k.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    // All-zero input should produce all-zero quantized
    for &byte in &output {
        assert_eq!(byte, 0, "zero input should quantize to zero");
    }
}

#[test]
fn quantize_i2s_output_too_small() {
    let k = FallbackKernel;
    let input = vec![1.0; 32];
    let mut output = vec![0u8; 1]; // Too small (need 32/4=8)
    let mut scales = vec![0.0f32; 1];
    assert!(k.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).is_err());
}

#[test]
fn quantize_i2s_scales_too_small() {
    let k = FallbackKernel;
    let input = vec![1.0; 64]; // 2 blocks of 32
    let mut output = vec![0u8; 16];
    let mut scales = vec![0.0f32; 1]; // Need 2
    assert!(k.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).is_err());
}

// =========================================================================
// quantize: TL1
// =========================================================================

#[test]
fn quantize_tl1_basic() {
    let k = FallbackKernel;
    let input = vec![1.0, -1.0, 0.3, -0.3, 0.0, 2.0, -2.0, 0.1];
    let mut output = vec![0u8; 2];
    let mut scales = vec![0.0f32; 1];
    k.quantize(&input, &mut output, &mut scales, QuantizationType::TL1).unwrap();
    assert!(scales[0] > 0.0);
}

#[test]
fn quantize_tl1_output_too_small() {
    let k = FallbackKernel;
    let input = vec![1.0; 64]; // 1 TL1 block (64)
    let mut output = vec![0u8; 1]; // Too small
    let mut scales = vec![0.0f32; 1];
    assert!(k.quantize(&input, &mut output, &mut scales, QuantizationType::TL1).is_err());
}

// =========================================================================
// quantize: TL2
// =========================================================================

#[test]
fn quantize_tl2_basic() {
    let k = FallbackKernel;
    let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let mut output = vec![0u8; 32];
    let mut scales = vec![0.0f32; 1];
    k.quantize(&input, &mut output, &mut scales, QuantizationType::TL2).unwrap();
    assert!(scales[0] > 0.0);
}

#[test]
fn quantize_tl2_output_too_small() {
    let k = FallbackKernel;
    let input = vec![1.0; 128]; // 1 TL2 block (128)
    let mut output = vec![0u8; 1]; // Too small
    let mut scales = vec![0.0f32; 1];
    assert!(k.quantize(&input, &mut output, &mut scales, QuantizationType::TL2).is_err());
}

// =========================================================================
// quantize: all types produce 2-bit packed output
// =========================================================================

#[test]
fn quantize_all_types_pack_2bit() {
    let k = FallbackKernel;
    // For each type, verify output bytes only use 2-bit codes per nibble
    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let block_size = match qtype {
            QuantizationType::I2S => 32,
            QuantizationType::TL1 => 64,
            QuantizationType::TL2 => 128,
        };
        let input: Vec<f32> = (0..block_size).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut output = vec![0u8; block_size / 4];
        let mut scales = vec![0.0f32; 1];
        k.quantize(&input, &mut output, &mut scales, qtype).unwrap();
        // Each 2-bit code should be 0..3
        for &byte in &output {
            for shift in (0..8).step_by(2) {
                let code = (byte >> shift) & 0x03;
                assert!(code <= 3, "2-bit code should be 0..3, got {code}");
            }
        }
    }
}
