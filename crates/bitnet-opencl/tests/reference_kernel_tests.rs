//! Tests for CPU reference kernel implementations.

use bitnet_opencl::reference_kernels::*;
use bitnet_opencl::test_fixtures::*;
use bitnet_opencl::testing::NumericalValidator;

// ── Matmul tests ────────────────────────────────────────────────────

#[test]
fn test_ref_matmul_identity() {
    let n = 4;
    let a =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let eye = identity_matrix(n);
    let mut c = vec![0.0f32; n * n];
    ref_matmul(&a, &eye, &mut c, n, n, n);
    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&a, &c).passed, "A * I should equal A");
}

#[test]
fn test_ref_matmul_known_values() {
    let (a, b, expected) = golden_matmul_4x4();
    let mut c = vec![0.0f32; 16];
    ref_matmul(&a, &b, &mut c, 4, 4, 4);
    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&expected, &c).passed);
}

#[test]
fn test_ref_matmul_non_square() {
    // A(2×3) * B(3×2) = C(2×2)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = vec![0.0f32; 4];
    ref_matmul(&a, &b, &mut c, 2, 2, 3);
    // Row 0: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
    // Row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
    let expected = vec![58.0, 64.0, 139.0, 154.0];
    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&expected, &c).passed);
}

// ── RMSNorm tests ───────────────────────────────────────────────────

#[test]
fn test_ref_rmsnorm_unit_norm() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let mut output = vec![0.0f32; 4];
    ref_rmsnorm(&input, &weight, &mut output, 1e-6);

    // RMS of [1,1,1,1] = 1.0, so output ≈ input
    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&input, &output).passed);
}

#[test]
fn test_ref_rmsnorm_scaling_property() {
    // Scaling input by constant should not change direction after rmsnorm
    let input1 = vec![1.0, 2.0, 3.0, 4.0];
    let input2 = vec![10.0, 20.0, 30.0, 40.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let mut out1 = vec![0.0f32; 4];
    let mut out2 = vec![0.0f32; 4];
    ref_rmsnorm(&input1, &weight, &mut out1, 1e-6);
    ref_rmsnorm(&input2, &weight, &mut out2, 1e-6);

    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&out1, &out2).passed);
}

// ── Softmax tests ───────────────────────────────────────────────────

#[test]
fn test_ref_softmax_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = vec![0.0f32; 5];
    ref_softmax(&input, &mut output, 5);

    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {}", sum);
}

#[test]
fn test_ref_softmax_monotonic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 4];
    ref_softmax(&input, &mut output, 4);

    for i in 1..4 {
        assert!(output[i] > output[i - 1], "softmax should preserve ordering");
    }
}

#[test]
fn test_ref_softmax_numerical_stability() {
    // Very large inputs should not produce NaN or Inf
    let input = vec![1000.0, 1001.0, 1002.0, 999.0];
    let mut output = vec![0.0f32; 4];
    ref_softmax(&input, &mut output, 4);

    for &v in &output {
        assert!(v.is_finite(), "softmax produced non-finite: {}", v);
    }
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ── RoPE tests ──────────────────────────────────────────────────────

#[test]
fn test_ref_rope_pos_zero_no_rotation() {
    let mut q = vec![1.0, 0.0, 0.0, 0.0];
    let mut k = vec![1.0, 0.0, 0.0, 0.0];
    let original_q = q.clone();
    let original_k = k.clone();

    ref_rope(&mut q, &mut k, 0, 4, 10000.0);

    // At position 0, cos(0)=1, sin(0)=0: no rotation
    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&original_q, &q).passed);
    assert!(v.compare_f32(&original_k, &k).passed);
}

#[test]
fn test_ref_rope_preserves_norm() {
    let mut q = vec![1.0, 2.0, 3.0, 4.0];
    let mut k = vec![5.0, 6.0, 7.0, 8.0];

    let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_norm_before: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

    ref_rope(&mut q, &mut k, 42, 4, 10000.0);

    let q_norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_norm_after: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!((q_norm_before - q_norm_after).abs() < 1e-4, "RoPE should preserve q norm");
    assert!((k_norm_before - k_norm_after).abs() < 1e-4, "RoPE should preserve k norm");
}

// ── Embedding tests ─────────────────────────────────────────────────

#[test]
fn test_ref_embedding_lookup() {
    let dim = 3;
    #[rustfmt::skip]
    let table = vec![
        0.1, 0.2, 0.3,  // token 0
        0.4, 0.5, 0.6,  // token 1
        0.7, 0.8, 0.9,  // token 2
    ];
    let tokens = vec![2u32, 0, 1];
    let mut output = vec![0.0f32; 9];

    ref_embedding(&tokens, &table, &mut output, dim);

    let expected = vec![0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&expected, &output).passed);
}

// ── Activation tests ────────────────────────────────────────────────

#[test]
fn test_ref_silu_known_values() {
    let input = vec![0.0, 1.0, -1.0];
    let mut output = vec![0.0f32; 3];
    ref_silu(&input, &mut output);

    // SiLU(0) = 0, SiLU(1) ≈ 0.7311, SiLU(-1) ≈ -0.2689
    assert!((output[0]).abs() < 1e-6);
    assert!((output[1] - 0.7311).abs() < 0.001);
    assert!((output[2] + 0.2689).abs() < 0.001);
}

#[test]
fn test_ref_gelu_known_values() {
    let input = vec![0.0, 1.0, -1.0];
    let mut output = vec![0.0f32; 3];
    ref_gelu(&input, &mut output);

    // GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
    assert!((output[0]).abs() < 1e-6);
    assert!((output[1] - 0.8412).abs() < 0.001);
    assert!((output[2] + 0.1588).abs() < 0.001);
}

// ── LayerNorm test ──────────────────────────────────────────────────

#[test]
fn test_ref_layernorm_identity_transform() {
    // weight=1, bias=0 should just normalize
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0, 0.0];
    let mut output = vec![0.0f32; 4];

    ref_layernorm(&input, &weight, &bias, &mut output, 1e-5);

    // Output should have mean≈0 and var≈1
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    let var: f32 = output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;

    assert!((mean).abs() < 1e-5, "mean should be ~0, got {}", mean);
    assert!((var - 1.0).abs() < 0.01, "var should be ~1, got {}", var);
}

// ── Dequantization tests ────────────────────────────────────────────

#[test]
fn test_ref_dequant_i2s_roundtrip() {
    // Pack known ternary values: +1, -1, 0, +1 (repeated)
    // 0b01=+1, 0b10=-1, 0b00=0
    let packed = vec![0b01_00_10_01u32]; // bits: 01 00 10 01 = +1,-1,0,+1
    let scale = 2.5;
    let mut output = vec![0.0f32; 4];

    ref_dequant_i2s(&packed, scale, &mut output);

    // Bit extraction is LSB first:
    // bits[0:1]=01 → +1, bits[2:3]=10 → -1, bits[4:5]=00 → 0, bits[6:7]=01 → +1
    assert_eq!(output[0], 2.5); // +1 * scale
    assert_eq!(output[1], -2.5); // -1 * scale
    assert_eq!(output[2], 0.0); // 0 * scale
    assert_eq!(output[3], 2.5); // +1 * scale
}

#[test]
fn test_ref_dequant_qk256_known_block() {
    // Single block: 16 words × 16 values/word = 256 values
    let mut packed = vec![0u32; 16];
    // Set first word to all +1 (0b01 repeated)
    packed[0] = 0x55555555; // 0b01_01_01_01... = all +1

    // f16 scale = 1.0 → 0x3C00
    let scales = vec![0x3C00u16];
    let mut output = vec![0.0f32; 256];

    ref_dequant_qk256(&packed, &scales, &mut output);

    // First 16 values should be 1.0 (all +1 * 1.0)
    for i in 0..16 {
        assert!((output[i] - 1.0).abs() < 1e-3, "output[{}] = {}, expected 1.0", i, output[i]);
    }
    // Remaining should be 0.0 (packed words are 0)
    for i in 16..256 {
        assert_eq!(output[i], 0.0, "output[{}] should be 0", i);
    }
}
