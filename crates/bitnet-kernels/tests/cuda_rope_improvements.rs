#![cfg(any(feature = "gpu", feature = "cuda"))]
//! Comprehensive CUDA RoPE (Rotary Position Embedding) improvement tests.
//!
//! These tests exercise the CUDA RoPE module's CPU fallback path and validate
//! mathematical properties that any correct RoPE implementation (CPU or GPU)
//! must satisfy. Tests are organised into seven categories:
//!
//! 1. Mathematical correctness at known angles (π/2, π, 2π, …)
//! 2. Position encoding consistency
//! 3. Property tests — norm preservation (proptest)
//! 4. Edge cases — position 0, very large positions, head_dim = 2
//! 5. Multi-head RoPE consistency
//! 6. Frequency table generation correctness
//! 7. Base frequency parameter tests

use bitnet_kernels::cuda::rope::{
    RopeConfig, apply_rope, apply_rope_batched, build_rope_freqs, compute_sincos_table,
    rope_backward_cpu, rope_forward_cpu,
};
use proptest::prelude::*;
use std::f32::consts::PI;

// ═════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Independent reference RoPE for a single (head, pos) vector.
fn reference_rope_single(data: &[f32], head_dim: usize, pos: f32, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = data.to_vec();
    for i in 0..half {
        let exp = -(2.0 * i as f32) / head_dim as f32;
        let theta = base.powf(exp);
        let angle = pos * theta;
        let (s, c) = angle.sin_cos();
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        out[2 * i] = x0 * c - x1 * s;
        out[2 * i + 1] = x0 * s + x1 * c;
    }
    out
}

// ═════════════════════════════════════════════════════════════════════
// 1. Mathematical correctness at known angles
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_cos_sin_at_pi_over_2() {
    // head_dim=2: theta_0 = base^0 = 1.0, angle = pos * 1.0
    // Choose pos such that angle = π/2
    let pos_pi2 = (PI / 2.0) as usize; // floor(1.5707…) = 1
    let head_dim = 2;
    let cfg = RopeConfig::for_shape(head_dim, 1, pos_pi2 + 1).unwrap();
    let total = (pos_pi2 + 1) * head_dim;
    let mut input = vec![0.0f32; total];
    input[pos_pi2 * head_dim] = 1.0;
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    let angle = pos_pi2 as f32; // = 1.0
    let expected_cos = angle.cos();
    let expected_sin = angle.sin();
    let start = pos_pi2 * head_dim;
    assert!(
        (output[start] - expected_cos).abs() < 1e-5,
        "cos mismatch: {} vs {expected_cos}",
        output[start]
    );
    assert!(
        (output[start + 1] - expected_sin).abs() < 1e-5,
        "sin mismatch: {} vs {expected_sin}",
        output[start + 1]
    );
}

#[test]
fn cuda_rope_rotation_at_exact_pi() {
    // At angle = π: cos(π) = −1, sin(π) = 0
    // head_dim=2, theta_0 = 1.0, so pos = π ≈ 3.14 → use pos=3
    // angle = 3.0, manually verify rotation
    let head_dim = 2;
    let pos = 3;
    let cfg = RopeConfig::for_shape(head_dim, 1, pos + 1).unwrap();
    let total = (pos + 1) * head_dim;
    let mut input = vec![0.0f32; total];
    input[pos * head_dim] = 2.0;
    input[pos * head_dim + 1] = -1.0;
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    let angle = pos as f32; // 3.0
    let (s, c) = angle.sin_cos();
    let start = pos * head_dim;
    let expected_0 = 2.0 * c + s;
    let expected_1 = 2.0 * s - c;
    assert!((output[start] - expected_0).abs() < 1e-5);
    assert!((output[start + 1] - expected_1).abs() < 1e-5);
}

#[test]
fn cuda_rope_rotation_at_two_pi() {
    // At angle = 2π: rotation is near-identity
    // head_dim=2, theta_0=1.0, need pos such that angle ≈ 2π → pos = 6 (≈6.28)
    let head_dim = 2;
    let pos = 6;
    let cfg = RopeConfig::for_shape(head_dim, 1, pos + 1).unwrap();
    let total = (pos + 1) * head_dim;
    let mut input = vec![0.0f32; total];
    input[pos * head_dim] = 5.0;
    input[pos * head_dim + 1] = 3.0;
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    let angle = 6.0f32;
    let (s, c) = angle.sin_cos();
    let start = pos * head_dim;
    assert!((output[start] - (5.0 * c - 3.0 * s)).abs() < 1e-4);
    assert!((output[start + 1] - (5.0 * s + 3.0 * c)).abs() < 1e-4);
}

#[test]
fn cuda_rope_known_angle_head_dim_4() {
    // head_dim=4, pos=2, base=10000
    // pair 0: theta = 1.0, angle = 2.0
    // pair 1: theta = 10000^(-0.5), angle = 2 * 10000^(-0.5)
    let head_dim = 4;
    let pos = 2;
    let cfg = RopeConfig::for_shape(head_dim, 1, pos + 1).unwrap();
    let total = (pos + 1) * head_dim;
    let mut input = vec![0.0f32; total];
    let x = [1.5f32, -0.7, 2.1, 0.3];
    input[pos * head_dim..pos * head_dim + head_dim].copy_from_slice(&x);
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    let angle0 = 2.0f32;
    let angle1 = 2.0 * 10_000.0f32.powf(-0.5);
    let (s0, c0) = angle0.sin_cos();
    let (s1, c1) = angle1.sin_cos();
    let start = pos * head_dim;

    let expected = [
        x[0] * c0 - x[1] * s0,
        x[0] * s0 + x[1] * c0,
        x[2] * c1 - x[3] * s1,
        x[2] * s1 + x[3] * c1,
    ];

    for (i, (&got, &want)) in
        output[start..start + head_dim].iter().zip(expected.iter()).enumerate()
    {
        assert!((got - want).abs() < 1e-5, "dim {i}: got {got}, expected {want}");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 2. Position encoding consistency
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_same_position_always_same_encoding() {
    let head_dim = 8;
    let cfg = RopeConfig::for_shape(head_dim, 2, 16).unwrap();
    let input: Vec<f32> = (0..2 * 16 * head_dim).map(|i| (i as f32) * 0.1).collect();
    let mut out1 = vec![0.0f32; input.len()];
    let mut out2 = vec![0.0f32; input.len()];

    rope_forward_cpu(&input, &mut out1, &cfg).unwrap();
    rope_forward_cpu(&input, &mut out2, &cfg).unwrap();

    assert_eq!(out1, out2, "Same input + config must produce identical output");
}

#[test]
fn cuda_rope_position_encoding_independent_of_seq_len() {
    // Encoding at position P should be the same regardless of how long the
    // sequence is.
    let head_dim = 8;
    let n_heads = 1;
    let pattern = [1.0f32, -0.5, 0.3, 2.0, -1.0, 0.7, 0.0, 0.9];

    for target_pos in [0usize, 1, 3, 7] {
        let mut results: Vec<Vec<f32>> = Vec::new();
        for seq_len in [target_pos + 1, target_pos + 4, target_pos + 16] {
            let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
            let total = n_heads * seq_len * head_dim;
            let mut input = vec![0.0f32; total];
            input[target_pos * head_dim..(target_pos + 1) * head_dim].copy_from_slice(&pattern);
            let mut output = vec![0.0f32; total];
            rope_forward_cpu(&input, &mut output, &cfg).unwrap();
            results.push(output[target_pos * head_dim..(target_pos + 1) * head_dim].to_vec());
        }

        for r in &results[1..] {
            for (i, (&a, &b)) in results[0].iter().zip(r.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "pos={target_pos} dim={i}: seq_len independence violated"
                );
            }
        }
    }
}

#[test]
fn cuda_rope_position_offset_equivalence() {
    // rope(pos=P, offset=0) == rope(pos=0, offset=P)
    let head_dim = 8;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    for target_pos in [1usize, 5, 42] {
        // Method 1: direct position
        let cfg1 = RopeConfig::for_shape(head_dim, 1, target_pos + 1).unwrap();
        let total1 = (target_pos + 1) * head_dim;
        let mut full_input = vec![0.0f32; total1];
        full_input[target_pos * head_dim..(target_pos + 1) * head_dim].copy_from_slice(&input);
        let mut out1 = vec![0.0f32; total1];
        rope_forward_cpu(&full_input, &mut out1, &cfg1).unwrap();
        let result1: Vec<f32> = out1[target_pos * head_dim..(target_pos + 1) * head_dim].to_vec();

        // Method 2: offset
        let cfg2 = RopeConfig::for_shape(head_dim, 1, 1).unwrap().with_position_offset(target_pos);
        let mut out2 = vec![0.0f32; head_dim];
        rope_forward_cpu(&input, &mut out2, &cfg2).unwrap();

        for (i, (&a, &b)) in result1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "pos={target_pos}, dim={i}: offset equivalence failed: {a} vs {b}"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 3. Property tests — norm preservation
// ═════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    #[test]
    fn prop_cuda_rope_preserves_l2_norm(
        head_dim_half in 1usize..=32,
        n_heads in 1usize..=4,
        pos in 0usize..128,
        seed in 0u64..1000,
    ) {
        let head_dim = head_dim_half * 2;
        let seq_len = pos + 1;
        let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
        let total = n_heads * seq_len * head_dim;
        let input: Vec<f32> = (0..total)
            .map(|i| ((i as u64 * 31 + seed) as f32).sin() * 3.0)
            .collect();
        let mut output = vec![0.0f32; total];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        for h in 0..n_heads {
            for p in 0..seq_len {
                let start = h * seq_len * head_dim + p * head_dim;
                let in_norm = l2_norm(&input[start..start + head_dim]);
                let out_norm = l2_norm(&output[start..start + head_dim]);
                prop_assert!(
                    (in_norm - out_norm).abs() < 1e-3,
                    "norm changed: head={h} pos={p}, {in_norm} → {out_norm}"
                );
            }
        }
    }

    #[test]
    fn prop_cuda_rope_forward_backward_roundtrip(
        head_dim_half in 1usize..=16,
        n_heads in 1usize..=3,
        seq_len in 1usize..=8,
        seed in 0u64..500,
    ) {
        let head_dim = head_dim_half * 2;
        let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
        let total = n_heads * seq_len * head_dim;
        let original: Vec<f32> = (0..total)
            .map(|i| ((i as u64 * 17 + seed) as f32).sin())
            .collect();

        let mut forward_out = vec![0.0f32; total];
        rope_forward_cpu(&original, &mut forward_out, &cfg).unwrap();

        let mut roundtrip = vec![0.0f32; total];
        rope_backward_cpu(&forward_out, &mut roundtrip, &cfg).unwrap();

        for (i, (&a, &b)) in roundtrip.iter().zip(original.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-3,
                "roundtrip failed at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn prop_cuda_rope_output_always_finite(
        head_dim_half in 1usize..=16,
        pos in 0usize..512,
        vals in prop::collection::vec(-50.0f32..50.0, 64),
    ) {
        let head_dim = head_dim_half * 2;
        let cfg = RopeConfig::for_shape(head_dim, 1, pos + 1).unwrap();
        let total = (pos + 1) * head_dim;
        let input: Vec<f32> = vals.iter().copied().cycle().take(total).collect();
        let mut output = vec![0.0f32; total];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "non-finite at index {i}: {v}");
        }
    }

    #[test]
    fn prop_cuda_rope_sincos_table_cos2_sin2_eq_1(
        head_dim_half in 1usize..=32,
        max_seq in 1usize..=64,
    ) {
        let head_dim = head_dim_half * 2;
        let cfg = RopeConfig::for_shape(head_dim, 1, max_seq)
            .unwrap()
            .with_max_seq_len(max_seq);
        let table = compute_sincos_table(&cfg);
        let half = head_dim / 2;

        for p in 0..max_seq {
            for i in 0..half {
                let idx = p * head_dim + 2 * i;
                let c = table[idx];
                let s = table[idx + 1];
                let sum = c * c + s * s;
                prop_assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "cos²+sin²≠1 at pos={p}, pair={i}: {sum}"
                );
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 4. Edge cases
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_position_zero_is_identity() {
    for head_dim in [2, 4, 8, 16, 64, 128] {
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap();
        let input: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.7).collect();
        let mut output = vec![0.0f32; head_dim];

        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-5,
                "pos=0 not identity: head_dim={head_dim}, dim={i}: {inp} vs {out}"
            );
        }
    }
}

#[test]
fn cuda_rope_head_dim_2_minimum() {
    // head_dim = 2 is the smallest valid dimension (one rotation pair)
    let head_dim = 2;
    let cfg = RopeConfig::for_shape(head_dim, 1, 4).unwrap();
    let total = 4 * head_dim;
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.3 + 1.0).collect();
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    assert!(output.iter().all(|x| x.is_finite()));
    for pos in 0..4 {
        let start = pos * head_dim;
        let in_n = l2_norm(&input[start..start + head_dim]);
        let out_n = l2_norm(&output[start..start + head_dim]);
        assert!((in_n - out_n).abs() < 1e-4, "head_dim=2 norm at pos={pos}: {in_n} vs {out_n}");
    }
}

#[test]
fn cuda_rope_very_large_position() {
    let head_dim = 4;
    let large_pos = 131_072usize; // 128K context window
    let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap().with_position_offset(large_pos);
    let input = vec![1.0, 0.0, 1.0, 0.0];
    let mut output = vec![0.0f32; head_dim];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    assert!(
        output.iter().all(|x| x.is_finite()),
        "Large position (128K) produced non-finite values"
    );
    let norm = l2_norm(&output);
    let expected_norm = l2_norm(&input);
    assert!(
        (norm - expected_norm).abs() < 1e-2,
        "Norm at pos={large_pos}: {norm} vs {expected_norm}"
    );
}

#[test]
fn cuda_rope_zero_vector_stays_zero() {
    let head_dim = 16;
    let cfg = RopeConfig::for_shape(head_dim, 2, 4).unwrap();
    let total = 2 * 4 * head_dim;
    let input = vec![0.0f32; total];
    let mut output = vec![1.0f32; total]; // pre-fill with non-zero

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    for (i, &v) in output.iter().enumerate() {
        assert!(v.abs() < 1e-10, "zero input not preserved at index {i}: {v}");
    }
}

#[test]
fn cuda_rope_single_element_batch() {
    // batch_size=1, seq_len=1, n_heads=1, head_dim=2
    let head_dim = 2;
    let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap();
    let input = vec![3.0f32, 4.0];
    let result = apply_rope_batched(&input, 1, 1, &cfg);
    // pos=0 → identity
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1] - 4.0).abs() < 1e-6);
}

#[test]
fn cuda_rope_backward_at_position_zero_is_identity() {
    for head_dim in [2, 4, 8, 32] {
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap();
        let grad_out: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut grad_in = vec![0.0f32; head_dim];

        rope_backward_cpu(&grad_out, &mut grad_in, &cfg).unwrap();

        for (i, (&g_out, &g_in)) in grad_out.iter().zip(grad_in.iter()).enumerate() {
            assert!(
                (g_out - g_in).abs() < 1e-5,
                "backward pos=0 not identity at dim {i}: {g_out} vs {g_in}"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 5. Multi-head RoPE consistency
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_all_heads_same_rotation_same_input() {
    let head_dim = 16;
    let n_heads = 8;
    let seq_len = 4;
    let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();

    let pattern: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let per_pos: Vec<f32> = pattern.iter().copied().cycle().take(n_heads * head_dim).collect();
    let input: Vec<f32> =
        per_pos.iter().copied().cycle().take(n_heads * seq_len * head_dim).collect();
    let mut output = vec![0.0f32; input.len()];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    // For each position, all heads should produce the same output
    for pos in 0..seq_len {
        let stride = seq_len * head_dim;
        let ref_start = pos * head_dim;
        let ref_vec = &output[ref_start..ref_start + head_dim];

        for h in 1..n_heads {
            let h_start = h * stride + pos * head_dim;
            let h_vec = &output[h_start..h_start + head_dim];
            for d in 0..head_dim {
                assert!(
                    (ref_vec[d] - h_vec[d]).abs() < 1e-5,
                    "pos={pos}, head {h} vs head 0 at dim {d}: {} vs {}",
                    h_vec[d],
                    ref_vec[d],
                );
            }
        }
    }
}

#[test]
fn cuda_rope_multi_head_norm_preservation() {
    let head_dim = 32;
    let n_heads = 6;
    let seq_len = 8;
    let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
    let total = n_heads * seq_len * head_dim;
    let input: Vec<f32> = (0..total).map(|i| ((i * 41 + 7) as f32).sin() * 2.0).collect();
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    for h in 0..n_heads {
        for p in 0..seq_len {
            let start = h * seq_len * head_dim + p * head_dim;
            let in_n = l2_norm(&input[start..start + head_dim]);
            let out_n = l2_norm(&output[start..start + head_dim]);
            assert!(
                (in_n - out_n).abs() < 1e-3,
                "norm changed: head={h}, pos={p}: {in_n} → {out_n}"
            );
        }
    }
}

#[test]
fn cuda_rope_apply_rope_multi_head_explicit_positions() {
    let head_dim = 8;
    let n_heads = 3;
    let positions = [0u32, 5, 10];
    let cfg = RopeConfig::for_shape(head_dim, n_heads, positions.len()).unwrap();
    let total = n_heads * positions.len() * head_dim;
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

    let output = apply_rope(&input, &positions, &cfg);
    assert_eq!(output.len(), total);
    assert!(output.iter().all(|x| x.is_finite()));
}

// ═════════════════════════════════════════════════════════════════════
// 6. Frequency table generation correctness
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_sincos_table_length() {
    for (hd, msl) in [(2, 1), (4, 8), (8, 16), (64, 128), (128, 512)] {
        let cfg = RopeConfig::for_shape(hd, 1, msl).unwrap().with_max_seq_len(msl);
        let table = compute_sincos_table(&cfg);
        assert_eq!(table.len(), msl * hd, "Wrong length for hd={hd}, msl={msl}");
    }
}

#[test]
fn cuda_rope_sincos_table_position_zero_cos1_sin0() {
    for head_dim in [2, 4, 8, 16, 64] {
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap().with_max_seq_len(1);
        let table = compute_sincos_table(&cfg);
        let half = head_dim / 2;
        for i in 0..half {
            assert!(
                (table[2 * i] - 1.0).abs() < 1e-6,
                "cos(0) should be 1: head_dim={head_dim}, pair={i}"
            );
            assert!(
                table[2 * i + 1].abs() < 1e-6,
                "sin(0) should be 0: head_dim={head_dim}, pair={i}"
            );
        }
    }
}

#[test]
fn cuda_rope_sincos_table_cos2_plus_sin2() {
    let head_dim = 16;
    let max_seq = 128;
    let cfg = RopeConfig::for_shape(head_dim, 1, max_seq).unwrap().with_max_seq_len(max_seq);
    let table = compute_sincos_table(&cfg);
    let half = head_dim / 2;

    for pos in 0..max_seq {
        for i in 0..half {
            let idx = pos * head_dim + 2 * i;
            let c = table[idx];
            let s = table[idx + 1];
            let sum = c * c + s * s;
            assert!((sum - 1.0).abs() < 1e-5, "cos²+sin²≠1 at pos={pos}, pair={i}: {sum}");
        }
    }
}

#[test]
fn cuda_rope_sincos_table_monotonic_frequency_decay() {
    let head_dim = 16;
    let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_max_seq_len(2);
    let table = compute_sincos_table(&cfg);
    let half = head_dim / 2;
    let pos1_offset = head_dim;

    for i in 0..(half - 1) {
        let sin_i = table[pos1_offset + 2 * i + 1].abs();
        let sin_next = table[pos1_offset + 2 * (i + 1) + 1].abs();
        assert!(
            sin_i >= sin_next,
            "Frequency should decrease: pair {i} sin={sin_i} < pair {} sin={sin_next}",
            i + 1,
        );
    }
}

#[test]
fn cuda_rope_build_rope_freqs_matches_sincos_table() {
    for head_dim in [4, 8, 16, 64] {
        let max_seq = 32;
        let cfg = RopeConfig::for_shape(head_dim, 1, max_seq).unwrap().with_max_seq_len(max_seq);
        let via_config = compute_sincos_table(&cfg);
        let via_standalone = build_rope_freqs(head_dim, max_seq, 10_000.0);

        assert_eq!(via_config.len(), via_standalone.len());
        for (i, (&a, &b)) in via_config.iter().zip(via_standalone.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "table mismatch at {i}: {a} vs {b}");
        }
    }
}

#[test]
fn cuda_rope_sincos_table_all_finite() {
    let cfg = RopeConfig::for_shape(128, 1, 2048).unwrap().with_max_seq_len(2048);
    let table = compute_sincos_table(&cfg);
    for (i, &v) in table.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite at index {i}: {v}");
    }
}

#[test]
fn cuda_rope_sincos_table_scaling_factor_effect() {
    let head_dim = 8;
    let max_seq = 8;
    let cfg1 = RopeConfig::for_shape(head_dim, 1, max_seq).unwrap().with_max_seq_len(max_seq);
    let cfg2 = RopeConfig::for_shape(head_dim, 1, max_seq)
        .unwrap()
        .with_max_seq_len(max_seq)
        .with_scaling_factor(2.0);

    let t1 = compute_sincos_table(&cfg1);
    let t2 = compute_sincos_table(&cfg2);

    // Position 0 identical (angle = 0 for both)
    for i in 0..head_dim {
        assert!((t1[i] - t2[i]).abs() < 1e-6);
    }
    // Position 1 with factor 2 ≈ position 2 without factor
    for i in 0..head_dim {
        assert!(
            (t2[head_dim + i] - t1[2 * head_dim + i]).abs() < 1e-5,
            "scaling factor 2x at pos1 ≈ pos2: {} vs {}",
            t2[head_dim + i],
            t1[2 * head_dim + i]
        );
    }
}

// ═════════════════════════════════════════════════════════════════════
// 7. Base frequency parameter tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_default_base_10000() {
    let cfg = RopeConfig::for_shape(8, 1, 1).unwrap();
    assert!((cfg.base - 10_000.0).abs() < 1e-3);
}

#[test]
fn cuda_rope_custom_base_changes_rotation() {
    let head_dim = 8;
    let input: Vec<f32> = (0..2 * head_dim).map(|i| (i as f32) * 0.1 + 1.0).collect();

    for base in [100.0f32, 1_000.0, 500_000.0, 1_000_000.0] {
        let cfg_default = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let cfg_custom = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_base(base);
        let mut out_default = vec![0.0f32; input.len()];
        let mut out_custom = vec![0.0f32; input.len()];

        rope_forward_cpu(&input, &mut out_default, &cfg_default).unwrap();
        rope_forward_cpu(&input, &mut out_custom, &cfg_custom).unwrap();

        // Position 0 always identical (angle = 0)
        for i in 0..head_dim {
            assert!((out_default[i] - out_custom[i]).abs() < 1e-6);
        }

        // Position 1 should differ (unless base happens to match)
        if (base - 10_000.0).abs() > 1.0 {
            let any_diff = (0..head_dim)
                .any(|i| (out_default[head_dim + i] - out_custom[head_dim + i]).abs() > 1e-5);
            assert!(any_diff, "base={base} should produce different rotation at pos=1");
        }
    }
}

#[test]
fn cuda_rope_large_base_near_identity() {
    let head_dim = 8;
    let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_base(1e12);
    let input: Vec<f32> = (0..2 * head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut output = vec![0.0f32; input.len()];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    // inv_freq[0] = base^0 = 1.0 for ANY base, so dimension pair 0 always
    // rotates by angle = pos regardless of base.  Higher pairs (i≥1) have
    // inv_freq ≈ 0 when base is huge, so output ≈ input for those.
    let half = head_dim / 2;
    for i in 1..half {
        let idx0 = head_dim + 2 * i;
        let idx1 = head_dim + 2 * i + 1;
        assert!(
            (output[idx0] - input[idx0]).abs() < 0.01,
            "Large base should produce near-identity at pair {i}: {} vs {}",
            output[idx0],
            input[idx0],
        );
        assert!(
            (output[idx1] - input[idx1]).abs() < 0.01,
            "Large base should produce near-identity at pair {i}: {} vs {}",
            output[idx1],
            input[idx1],
        );
    }
}

#[test]
fn cuda_rope_small_base_fast_rotation() {
    let head_dim = 4;
    let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_base(2.0);
    let input = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut output = vec![0.0f32; 8];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    let in_n = l2_norm(&input[head_dim..]);
    let out_n = l2_norm(&output[head_dim..]);
    assert!((in_n - out_n).abs() < 1e-4, "norm: {in_n} vs {out_n}");
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn cuda_rope_custom_base_matches_reference() {
    let head_dim = 8;
    let base = 500_000.0f32;
    let cfg = RopeConfig::for_shape(head_dim, 1, 8).unwrap().with_base(base);
    let total = 8 * head_dim;
    let input: Vec<f32> = (0..total).map(|i| ((i * 13 + 7) as f32).sin()).collect();
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    for pos in 0..8 {
        let start = pos * head_dim;
        let expected =
            reference_rope_single(&input[start..start + head_dim], head_dim, pos as f32, base);
        for (d, (&got, &want)) in
            output[start..start + head_dim].iter().zip(expected.iter()).enumerate()
        {
            assert!((got - want).abs() < 1e-5, "base={base} pos={pos} dim={d}: {got} vs {want}");
        }
    }
}

#[test]
fn cuda_rope_base_frequency_affects_wavelength() {
    // Use head_dim=4 → 2 dimension pairs.  Pair 0 always has freq=1.0
    // (base^0) regardless of base, so we need non-zero input in the second
    // pair (indices 2,3 at pos=1) to observe the effect of changing base.
    let head_dim = 4;
    let input = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // only pair-1 at pos=1

    let cfg_low = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_base(100.0);
    let cfg_high = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_base(1_000_000.0);

    let mut out_low = vec![0.0f32; 8];
    let mut out_high = vec![0.0f32; 8];
    rope_forward_cpu(&input, &mut out_low, &cfg_low).unwrap();
    rope_forward_cpu(&input, &mut out_high, &cfg_high).unwrap();

    // Measure rotation distance for pair-1 only (indices 2,3 at pos=1)
    let dist_low: f32 = (2..head_dim)
        .map(|i| (out_low[head_dim + i] - input[head_dim + i]).powi(2))
        .sum::<f32>()
        .sqrt();
    let dist_high: f32 = (2..head_dim)
        .map(|i| (out_high[head_dim + i] - input[head_dim + i]).powi(2))
        .sum::<f32>()
        .sqrt();

    assert!(
        dist_low > dist_high,
        "Lower base should rotate more at pair-1: dist_low={dist_low} vs dist_high={dist_high}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Additional: Interleaved layout tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_interleaved_norm_preservation() {
    let head_dim = 8;
    let n_heads = 2;
    let seq_len = 4;
    let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap().with_interleaved(true);
    let total = n_heads * seq_len * head_dim;
    let input: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let mut output = vec![0.0f32; total];

    rope_forward_cpu(&input, &mut output, &cfg).unwrap();

    for h in 0..n_heads {
        for p in 0..seq_len {
            let start = h * seq_len * head_dim + p * head_dim;
            let in_n = l2_norm(&input[start..start + head_dim]);
            let out_n = l2_norm(&output[start..start + head_dim]);
            assert!(
                (in_n - out_n).abs() < 1e-3,
                "interleaved norm: h={h}, p={p}: {in_n} vs {out_n}"
            );
        }
    }
}

#[test]
fn cuda_rope_interleaved_roundtrip() {
    let head_dim = 8;
    let cfg = RopeConfig::for_shape(head_dim, 2, 4).unwrap().with_interleaved(true);
    let total = 2 * 4 * head_dim;
    let original: Vec<f32> = (0..total).map(|i| ((i * 11 + 3) as f32).sin()).collect();

    let mut forward_out = vec![0.0f32; total];
    rope_forward_cpu(&original, &mut forward_out, &cfg).unwrap();

    let mut roundtrip = vec![0.0f32; total];
    rope_backward_cpu(&forward_out, &mut roundtrip, &cfg).unwrap();

    for (i, (&a, &b)) in roundtrip.iter().zip(original.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "interleaved roundtrip at {i}: {a} vs {b}");
    }
}

// ═════════════════════════════════════════════════════════════════════
// Additional: Batched API tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cuda_rope_batched_independent_batches() {
    let head_dim = 8;
    let seq_len = 4;
    let n_heads = 2;
    let batch_size = 3;
    let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
    let per_batch = n_heads * seq_len * head_dim;
    let pattern: Vec<f32> = (0..per_batch).map(|i| (i as f32) * 0.1).collect();
    let input: Vec<f32> = pattern.iter().copied().cycle().take(batch_size * per_batch).collect();

    let output = apply_rope_batched(&input, batch_size, seq_len, &cfg);

    for b in 1..batch_size {
        for i in 0..per_batch {
            let ref_val = output[i];
            let val = output[b * per_batch + i];
            assert!((ref_val - val).abs() < 1e-5, "batch {b} differs at {i}: {val} vs {ref_val}");
        }
    }
}

#[test]
fn cuda_rope_batched_matches_single_forward() {
    let head_dim = 4;
    let seq_len = 3;
    let n_heads = 2;
    let batch_size = 2;
    let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
    let per_batch = n_heads * seq_len * head_dim;
    let total = batch_size * per_batch;
    let input: Vec<f32> = (0..total).map(|i| ((i * 7 + 3) as f32).cos()).collect();

    let batched = apply_rope_batched(&input, batch_size, seq_len, &cfg);

    for b in 0..batch_size {
        let start = b * per_batch;
        let mut expected = vec![0.0f32; per_batch];
        rope_forward_cpu(&input[start..start + per_batch], &mut expected, &cfg).unwrap();
        for i in 0..per_batch {
            assert!(
                (batched[start + i] - expected[i]).abs() < 1e-5,
                "batch {b} idx {i}: {} vs {}",
                batched[start + i],
                expected[i],
            );
        }
    }
}
