//! Comprehensive RoPE (Rotary Position Embedding) validation tests.
//!
//! Categories:
//! 1. Mathematical property tests (magnitude preservation, identity, smoothness)
//! 2. Known-value tests against reference implementation
//! 3. Edge cases (single element, large positions, extreme theta)
//! 4. Cross-validation between CPU reference and kernel dispatch
//! 5. Property tests (proptest) for norm preservation and invertibility
//! 6. Determinism and consistency tests

use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Pure-Rust reference RoPE for a single position and head vector.
/// Intentionally independent of the kernel under test.
fn reference_rope(data: &mut [f32], position: usize, head_dim: usize, base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let exponent = -(2.0 * i as f32) / head_dim as f32;
        let theta = base.powf(exponent);
        let angle = position as f32 * theta;
        let (sin_val, cos_val) = angle.sin_cos();

        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * cos_val - x1 * sin_val;
        data[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Inverse RoPE: rotate by -angle.
fn inverse_rope(data: &mut [f32], position: usize, head_dim: usize, base: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let exponent = -(2.0 * i as f32) / head_dim as f32;
        let theta = base.powf(exponent);
        let angle = position as f32 * theta;
        let (sin_val, cos_val) = angle.sin_cos();

        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        // Inverse rotation: negate the sin component
        data[2 * i] = x0 * cos_val + x1 * sin_val;
        data[2 * i + 1] = -x0 * sin_val + x1 * cos_val;
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Mathematical property tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_preserves_magnitude_various_positions() {
    let head_dim = 16;
    let cfg = RopeConfig::new(head_dim, 256);
    let freqs = compute_frequencies(&cfg);
    let data_template: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.7).collect();

    for pos in [0, 1, 2, 10, 50, 127, 255] {
        let mut data = data_template.clone();
        let norm_before = l2_norm(&data);

        apply_rope(&mut data, pos, head_dim, &freqs);

        let norm_after = l2_norm(&data);
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "Norm not preserved at pos={pos}: {norm_before} vs {norm_after}"
        );
    }
}

#[test]
fn rope_identity_at_position_zero_all_dims() {
    for head_dim in [2, 4, 6, 8, 16, 32, 64, 128] {
        let cfg = RopeConfig::new(head_dim, 1);
        let freqs = compute_frequencies(&cfg);
        let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 1.23 + 0.5).collect();
        let original = data.clone();

        apply_rope(&mut data, 0, head_dim, &freqs);

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (o - d).abs() < 1e-5,
                "Not identity at pos=0, head_dim={head_dim}, dim={i}: {o} vs {d}"
            );
        }
    }
}

#[test]
fn rope_smoothness_small_position_change() {
    let head_dim = 8;
    let cfg = RopeConfig::new(head_dim, 102);
    let freqs = compute_frequencies(&cfg);
    let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

    let mut data_pos100 = original.clone();
    apply_rope(&mut data_pos100, 100, head_dim, &freqs);

    let mut data_pos101 = original.clone();
    apply_rope(&mut data_pos101, 101, head_dim, &freqs);

    let diff: f32 = data_pos100
        .iter()
        .zip(data_pos101.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    // Adjacent positions should produce a small (bounded) change
    let input_norm = l2_norm(&original);
    assert!(
        diff < input_norm * 2.0,
        "Adjacent positions changed too much: diff={diff}, norm={input_norm}"
    );
    assert!(diff > 0.0, "Adjacent positions should differ");
}

#[test]
fn rope_frequency_decreases_with_dimension() {
    let head_dim = 16;
    let cfg = RopeConfig::new(head_dim, 2);
    let freqs = compute_frequencies(&cfg);
    let half_dim = head_dim / 2;

    // At position 1, extract sin values for each dim pair.
    // Lower dimension pairs rotate faster (higher sin magnitude).
    let pos1_offset = head_dim;
    for i in 0..(half_dim - 1) {
        let sin_i = freqs[pos1_offset + 2 * i + 1].abs();
        let sin_next = freqs[pos1_offset + 2 * (i + 1) + 1].abs();
        assert!(
            sin_i >= sin_next,
            "Frequency should decrease: dim_pair {i} sin={sin_i} < dim_pair {} sin={sin_next}",
            i + 1,
        );
    }
}

#[test]
fn rope_cos2_plus_sin2_equals_one() {
    let head_dim = 8;
    let cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&cfg);
    let half_dim = head_dim / 2;

    for pos in 0..64 {
        for i in 0..half_dim {
            let idx = pos * head_dim + 2 * i;
            let cos_val = freqs[idx];
            let sin_val = freqs[idx + 1];
            let sum = cos_val * cos_val + sin_val * sin_val;
            assert!((sum - 1.0).abs() < 1e-5, "cos²+sin²≠1 at pos={pos}, pair={i}: {sum}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Known-value tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_known_value_head_dim4_pos0_identity() {
    let cfg = RopeConfig::new(4, 1);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![3.0, -1.0, 2.5, 0.7];
    let original = data.clone();

    apply_rope(&mut data, 0, 4, &freqs);

    for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
        assert!((o - d).abs() < 1e-6, "dim {i}: {o} vs {d}");
    }
}

#[test]
fn rope_known_value_head_dim4_pos1() {
    // theta_0 = 10000^0 = 1.0,          angle_0 = 1.0
    // theta_1 = 10000^(-2/4) = 0.01,    angle_1 = 0.01
    let cfg = RopeConfig::new(4, 2);
    let freqs = compute_frequencies(&cfg);

    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut data = x.to_vec();

    let angle0 = 1.0f32;
    let angle1 = 10_000.0f32.powf(-0.5);

    let expected = [
        x[0] * angle0.cos() - x[1] * angle0.sin(),
        x[0] * angle0.sin() + x[1] * angle0.cos(),
        x[2] * angle1.cos() - x[3] * angle1.sin(),
        x[2] * angle1.sin() + x[3] * angle1.cos(),
    ];

    apply_rope(&mut data, 1, 4, &freqs);

    for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "dim {i}: got {got}, expected {want}");
    }
}

#[test]
fn rope_known_value_head_dim2_unit_vector() {
    // head_dim=2, pos=1, base=10000
    // theta = 10000^0 = 1.0, angle = 1.0
    // Input: (1, 0) → (cos(1), sin(1))
    let cfg = RopeConfig::new(2, 2);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 0.0];

    apply_rope(&mut data, 1, 2, &freqs);

    assert!((data[0] - 1.0f32.cos()).abs() < 1e-5);
    assert!((data[1] - 1.0f32.sin()).abs() < 1e-5);
}

#[test]
fn rope_known_value_head_dim2_pos5() {
    // angle = 5 * 1.0 = 5.0
    let cfg = RopeConfig::new(2, 6);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![2.0, -1.0];

    let angle = 5.0f32;
    let expected =
        [2.0 * angle.cos() - (-1.0) * angle.sin(), 2.0 * angle.sin() + (-1.0) * angle.cos()];

    apply_rope(&mut data, 5, 2, &freqs);

    for (i, (g, w)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((g - w).abs() < 1e-5, "dim {i}: {g} vs {w}");
    }
}

#[test]
fn rope_known_value_custom_base_500k() {
    let base = 500_000.0f32;
    let cfg = RopeConfig::new(4, 2).with_base(base);
    let freqs = compute_frequencies(&cfg);

    let x = [1.0f32, 0.0, 0.0, 1.0];
    let mut data = x.to_vec();

    let angle0 = 1.0f32 * base.powf(0.0); // = 1.0
    let angle1 = 1.0f32 * base.powf(-0.5);

    let expected = [
        x[0] * angle0.cos() - x[1] * angle0.sin(),
        x[0] * angle0.sin() + x[1] * angle0.cos(),
        x[2] * angle1.cos() - x[3] * angle1.sin(),
        x[2] * angle1.sin() + x[3] * angle1.cos(),
    ];

    apply_rope(&mut data, 1, 4, &freqs);

    for (i, (g, w)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((g - w).abs() < 1e-5, "dim {i}: {g} vs {w}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_single_element_head_dim_2() {
    let cfg = RopeConfig::new(2, 4);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 1.0];
    let norm_before = l2_norm(&data);

    apply_rope(&mut data, 3, 2, &freqs);

    let norm_after = l2_norm(&data);
    assert!((norm_before - norm_after).abs() < 1e-5);
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn rope_large_positions_1m_tokens() {
    let max_pos = 1_048_576;
    let head_dim = 4;
    let cfg = RopeConfig::new(head_dim, max_pos + 1);
    let freqs = compute_frequencies(&cfg);

    let mut data = vec![1.0, 0.0, 1.0, 0.0];
    let norm_before = l2_norm(&data);

    apply_rope(&mut data, max_pos, head_dim, &freqs);

    assert!(data.iter().all(|x| x.is_finite()), "Large position produced non-finite values");
    let norm_after = l2_norm(&data);
    assert!(
        (norm_before - norm_after).abs() < 1e-2,
        "Norm diverged at pos={max_pos}: {norm_before} vs {norm_after}"
    );
}

#[test]
fn rope_very_small_theta_high_frequency() {
    // base=2.0 → very high frequencies; rapid oscillation
    let cfg = RopeConfig::new(4, 10).with_base(2.0);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    let norm_before = l2_norm(&data);

    apply_rope(&mut data, 5, 4, &freqs);

    let norm_after = l2_norm(&data);
    assert!(
        (norm_before - norm_after).abs() < 1e-4,
        "Small theta broke norm: {norm_before} vs {norm_after}"
    );
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn rope_very_large_theta_low_frequency() {
    // base=1e10 → extremely slow rotation
    let cfg = RopeConfig::new(4, 10).with_base(1e10);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    let original = data.clone();

    apply_rope(&mut data, 1, 4, &freqs);

    // With enormous base, higher dimension pairs barely rotate
    // pair 1: angle = 1 * (1e10)^(-0.5) ≈ 3.16e-6 → near identity
    let diff_pair1: f32 = (data[2] - original[2]).powi(2) + (data[3] - original[3]).powi(2);
    assert!(
        diff_pair1.sqrt() < 0.01,
        "Large base should barely rotate higher dims: diff={diff_pair1}"
    );
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
#[should_panic(expected = "head_dim must be even")]
fn rope_odd_head_dim_panics() {
    let _cfg = RopeConfig::new(3, 4);
}

#[test]
#[should_panic(expected = "head_dim must be even")]
fn rope_head_dim_zero_panics() {
    let _cfg = RopeConfig::new(0, 4);
}

#[test]
fn rope_scaling_factor_doubles_angle() {
    let head_dim = 4;
    let cfg1 = RopeConfig::new(head_dim, 4);
    let cfg2 = RopeConfig::new(head_dim, 4).with_scaling_factor(2.0);
    let freqs1 = compute_frequencies(&cfg1);
    let freqs2 = compute_frequencies(&cfg2);

    // At position 1, scaled frequencies should differ from unscaled
    let pos1_offset = head_dim;
    for i in 0..head_dim {
        let f1 = freqs1[pos1_offset + i];
        let f2 = freqs2[pos1_offset + i];
        // scaling_factor=2 means angle is doubled
        // So freqs at pos 1 with factor 2 ≈ freqs at pos 2 with factor 1
        let f1_pos2 = freqs1[2 * head_dim + i];
        assert!(
            (f2 - f1_pos2).abs() < 1e-5,
            "Scaling factor 2×pos1 should ≈ pos2: {f2} vs {f1_pos2} (unscaled pos1={f1})"
        );
    }
}

#[test]
fn rope_zero_vector_stays_zero() {
    let head_dim = 8;
    let cfg = RopeConfig::new(head_dim, 10);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![0.0; head_dim];

    apply_rope(&mut data, 5, head_dim, &freqs);

    for (i, &v) in data.iter().enumerate() {
        assert!(v.abs() < 1e-10, "Zero vector not preserved at dim {i}: {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Cross-validation: reference vs kernel
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_kernel_matches_reference_implementation() {
    let base = 10_000.0f32;
    for head_dim in [2, 4, 8, 16, 32] {
        let cfg = RopeConfig::new(head_dim, 20);
        let freqs = compute_frequencies(&cfg);

        for pos in [0, 1, 5, 10, 19] {
            let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3 - 0.5).collect();

            let mut kernel_data = original.clone();
            apply_rope(&mut kernel_data, pos, head_dim, &freqs);

            let mut ref_data = original.clone();
            reference_rope(&mut ref_data, pos, head_dim, base);

            for (d, (k, r)) in kernel_data.iter().zip(ref_data.iter()).enumerate() {
                assert!(
                    (k - r).abs() < 1e-5,
                    "Kernel/ref mismatch: head_dim={head_dim}, pos={pos}, dim={d}: {k} vs {r}"
                );
            }
        }
    }
}

#[test]
fn rope_batch_matches_reference() {
    let head_dim = 8;
    let num_heads = 3;
    let seq_len = 5;
    let start_pos = 2;
    let base = 10_000.0f32;
    let cfg = RopeConfig::new(head_dim, start_pos + seq_len);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.13) - 3.0).collect();

    // Kernel batch
    let mut batch_data = original.clone();
    apply_rope_batch(&mut batch_data, start_pos, seq_len, num_heads, head_dim, &freqs);

    // Reference per-head
    let mut ref_data = original.clone();
    for s in 0..seq_len {
        let position = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            reference_rope(&mut ref_data[offset..offset + head_dim], position, head_dim, base);
        }
    }

    for (i, (b, r)) in batch_data.iter().zip(ref_data.iter()).enumerate() {
        assert!((b - r).abs() < 1e-4, "Batch/ref mismatch at index {i}: {b} vs {r}");
    }
}

#[test]
fn rope_deterministic_same_input_same_output() {
    let head_dim = 16;
    let cfg = RopeConfig::new(head_dim, 10);
    let freqs = compute_frequencies(&cfg);
    let original: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1).collect();

    let mut run1 = original.clone();
    apply_rope(&mut run1, 7, head_dim, &freqs);

    let mut run2 = original.clone();
    apply_rope(&mut run2, 7, head_dim, &freqs);

    assert_eq!(run1, run2, "RoPE must be deterministic");
}

#[test]
fn rope_batch_deterministic() {
    let head_dim = 8;
    let num_heads = 2;
    let seq_len = 4;
    let cfg = RopeConfig::new(head_dim, seq_len + 3);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| i as f32 * 0.01).collect();

    let mut run1 = original.clone();
    apply_rope_batch(&mut run1, 3, seq_len, num_heads, head_dim, &freqs);

    let mut run2 = original.clone();
    apply_rope_batch(&mut run2, 3, seq_len, num_heads, head_dim, &freqs);

    assert_eq!(run1, run2, "Batch RoPE must be deterministic");
}

#[test]
fn rope_different_seq_lengths_consistent_positions() {
    // Positions encoded the same regardless of surrounding sequence length
    let head_dim = 8;

    let cfg_short = RopeConfig::new(head_dim, 10);
    let freqs_short = compute_frequencies(&cfg_short);

    let cfg_long = RopeConfig::new(head_dim, 100);
    let freqs_long = compute_frequencies(&cfg_long);

    let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

    for pos in [0, 1, 5, 9] {
        let mut data_short = original.clone();
        apply_rope(&mut data_short, pos, head_dim, &freqs_short);

        let mut data_long = original.clone();
        apply_rope(&mut data_long, pos, head_dim, &freqs_long);

        for (d, (s, l)) in data_short.iter().zip(data_long.iter()).enumerate() {
            assert!(
                (s - l).abs() < 1e-6,
                "Seq-length shouldn't matter: pos={pos}, dim={d}: {s} vs {l}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Property tests (proptest)
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_rope_preserves_l2_norm(
        head_dim_half in 1usize..=32,
        pos in 0usize..512,
        data_vals in prop::collection::vec(-10.0f32..10.0, 64),
    ) {
        let head_dim = head_dim_half * 2;
        let actual_len = head_dim.min(data_vals.len());
        let head_dim = if actual_len % 2 == 0 { actual_len } else { actual_len - 1 };
        if head_dim < 2 {
            return Ok(());
        }

        let cfg = RopeConfig::new(head_dim, pos + 1);
        let freqs = compute_frequencies(&cfg);
        let mut data: Vec<f32> = data_vals[..head_dim].to_vec();
        let norm_before = l2_norm(&data);

        apply_rope(&mut data, pos, head_dim, &freqs);

        let norm_after = l2_norm(&data);
        prop_assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "Norm changed: {norm_before} → {norm_after} (head_dim={head_dim}, pos={pos})"
        );
    }

    #[test]
    fn prop_rope_invertible(
        head_dim_half in 1usize..=16,
        pos in 0usize..256,
        data_vals in prop::collection::vec(-5.0f32..5.0, 32),
    ) {
        let head_dim = head_dim_half * 2;
        let actual_len = head_dim.min(data_vals.len());
        let head_dim = if actual_len % 2 == 0 { actual_len } else { actual_len - 1 };
        if head_dim < 2 {
            return Ok(());
        }

        let base = 10_000.0f32;
        let original: Vec<f32> = data_vals[..head_dim].to_vec();
        let mut data = original.clone();

        // Apply forward then inverse
        reference_rope(&mut data, pos, head_dim, base);
        inverse_rope(&mut data, pos, head_dim, base);

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            prop_assert!(
                (o - d).abs() < 1e-4,
                "Not invertible: dim={i}, original={o}, roundtrip={d}"
            );
        }
    }

    #[test]
    fn prop_rope_kernel_matches_reference_random(
        head_dim_half in 1usize..=16,
        pos in 0usize..128,
        data_vals in prop::collection::vec(-10.0f32..10.0, 32),
    ) {
        let head_dim = head_dim_half * 2;
        let actual_len = head_dim.min(data_vals.len());
        let head_dim = if actual_len % 2 == 0 { actual_len } else { actual_len - 1 };
        if head_dim < 2 {
            return Ok(());
        }

        let base = 10_000.0f32;
        let cfg = RopeConfig::new(head_dim, pos + 1);
        let freqs = compute_frequencies(&cfg);
        let original: Vec<f32> = data_vals[..head_dim].to_vec();

        let mut kernel_data = original.clone();
        apply_rope(&mut kernel_data, pos, head_dim, &freqs);

        let mut ref_data = original.clone();
        reference_rope(&mut ref_data, pos, head_dim, base);

        for (i, (k, r)) in kernel_data.iter().zip(ref_data.iter()).enumerate() {
            prop_assert!(
                (k - r).abs() < 1e-4,
                "Mismatch: dim={i}, kernel={k}, ref={r} (head_dim={head_dim}, pos={pos})"
            );
        }
    }

    #[test]
    fn prop_rope_output_always_finite(
        head_dim_half in 1usize..=16,
        pos in 0usize..1024,
        data_vals in prop::collection::vec(-100.0f32..100.0, 32),
    ) {
        let head_dim = head_dim_half * 2;
        let actual_len = head_dim.min(data_vals.len());
        let head_dim = if actual_len % 2 == 0 { actual_len } else { actual_len - 1 };
        if head_dim < 2 {
            return Ok(());
        }

        let cfg = RopeConfig::new(head_dim, pos + 1);
        let freqs = compute_frequencies(&cfg);
        let mut data: Vec<f32> = data_vals[..head_dim].to_vec();

        apply_rope(&mut data, pos, head_dim, &freqs);

        for (i, &v) in data.iter().enumerate() {
            prop_assert!(v.is_finite(), "Non-finite at dim={i}: {v}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Batch-specific edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_batch_single_head_single_seq() {
    let head_dim = 4;
    let cfg = RopeConfig::new(head_dim, 5);
    let freqs = compute_frequencies(&cfg);
    let mut batch = vec![1.0, 2.0, 3.0, 4.0];
    let mut single = batch.clone();

    apply_rope_batch(&mut batch, 3, 1, 1, head_dim, &freqs);
    apply_rope(&mut single, 3, head_dim, &freqs);

    for (b, s) in batch.iter().zip(single.iter()) {
        assert!((b - s).abs() < 1e-6, "Single batch ≠ single apply: {b} vs {s}");
    }
}

#[test]
fn rope_batch_multi_head_all_same_pattern() {
    let head_dim = 8;
    let num_heads = 4;
    let cfg = RopeConfig::new(head_dim, 3);
    let freqs = compute_frequencies(&cfg);

    let pattern: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut data: Vec<f32> = pattern.iter().copied().cycle().take(num_heads * head_dim).collect();

    apply_rope_batch(&mut data, 2, 1, num_heads, head_dim, &freqs);

    // All heads at the same position should get identical results
    for h in 1..num_heads {
        for d in 0..head_dim {
            let ref_val = data[d];
            let val = data[h * head_dim + d];
            assert!(
                (ref_val - val).abs() < 1e-6,
                "Head {h} differs at dim {d}: {val} vs {ref_val}"
            );
        }
    }
}

#[test]
fn rope_batch_preserves_magnitude_all_heads() {
    let head_dim = 16;
    let num_heads = 4;
    let seq_len = 8;
    let cfg = RopeConfig::new(head_dim, seq_len + 1);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let mut data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

    // Compute norms before
    let mut norms_before = Vec::new();
    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = (s * num_heads + h) * head_dim;
            norms_before.push(l2_norm(&data[off..off + head_dim]));
        }
    }

    apply_rope_batch(&mut data, 0, seq_len, num_heads, head_dim, &freqs);

    // Check norms after
    for (idx, (s, h)) in (0..seq_len).flat_map(|s| (0..num_heads).map(move |h| (s, h))).enumerate()
    {
        let off = (s * num_heads + h) * head_dim;
        let norm_after = l2_norm(&data[off..off + head_dim]);
        assert!(
            (norms_before[idx] - norm_after).abs() < 1e-3,
            "Norm changed at seq={s} head={h}: {} vs {norm_after}",
            norms_before[idx]
        );
    }
}

#[test]
fn rope_batch_non_8_aligned_head_dim() {
    // head_dim=6: forces scalar tail in AVX2 path
    let head_dim = 6;
    let num_heads = 2;
    let seq_len = 3;
    let cfg = RopeConfig::new(head_dim, seq_len + 1);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.2).collect();

    // Batch (may use AVX2 + scalar tail)
    let mut batch = original.clone();
    apply_rope_batch(&mut batch, 0, seq_len, num_heads, head_dim, &freqs);

    // Per-head reference
    let mut reference = original.clone();
    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = (s * num_heads + h) * head_dim;
            apply_rope(&mut reference[off..off + head_dim], s, head_dim, &freqs);
        }
    }

    for (i, (b, r)) in batch.iter().zip(reference.iter()).enumerate() {
        assert!((b - r).abs() < 1e-5, "Non-aligned mismatch at {i}: {b} vs {r}");
    }
}

#[test]
fn rope_frequency_table_length_correct() {
    for (hd, msl) in [(2, 1), (4, 8), (8, 16), (64, 128), (128, 256)] {
        let cfg = RopeConfig::new(hd, msl);
        let freqs = compute_frequencies(&cfg);
        assert_eq!(
            freqs.len(),
            msl * hd,
            "Wrong table length for head_dim={hd}, max_seq_len={msl}"
        );
    }
}

#[test]
fn rope_frequency_table_all_finite() {
    let cfg = RopeConfig::new(64, 1024);
    let freqs = compute_frequencies(&cfg);
    for (i, &v) in freqs.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite frequency at index {i}: {v}");
    }
}
