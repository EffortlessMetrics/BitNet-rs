//! Edge-case integration tests for `bitnet_kernels::cpu::rope` module.
//!
//! Covers additional scenarios beyond inline unit tests:
//! - RopeConfig builder patterns and panics
//! - Frequency table invariants (symmetry, period, cos²+sin²)
//! - apply_rope with all-zero data, negative values, mixed signs
//! - Batch processing with varying start_pos, many heads
//! - Double rotation and inverse rotation properties
//! - Long sequences (16K context)
//! - Non-standard bases (Phi-4's 10K, Llama's 500K)

use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};

// =========================================================================
// RopeConfig construction and builders
// =========================================================================

#[test]
fn rope_config_basic_construction() {
    let cfg = RopeConfig::new(8, 128);
    assert_eq!(cfg.head_dim, 8);
    assert_eq!(cfg.max_seq_len, 128);
    assert!((cfg.base - 10_000.0).abs() < 1e-3);
    assert!((cfg.scaling_factor - 1.0).abs() < 1e-6);
}

#[test]
fn rope_config_with_base_chain() {
    let cfg = RopeConfig::new(4, 16).with_base(500_000.0);
    assert!((cfg.base - 500_000.0).abs() < 1e-3);
}

#[test]
fn rope_config_with_scaling_factor_chain() {
    let cfg = RopeConfig::new(4, 16).with_scaling_factor(0.5);
    assert!((cfg.scaling_factor - 0.5).abs() < 1e-6);
}

#[test]
fn rope_config_builder_chaining() {
    let cfg = RopeConfig::new(16, 256).with_base(1_000_000.0).with_scaling_factor(0.25);
    assert!((cfg.base - 1_000_000.0).abs() < 1e-3);
    assert!((cfg.scaling_factor - 0.25).abs() < 1e-6);
}

#[test]
#[should_panic]
fn rope_config_odd_head_dim_panics() {
    let _ = RopeConfig::new(3, 16);
}

#[test]
#[should_panic]
fn rope_config_zero_head_dim_panics() {
    let _ = RopeConfig::new(0, 16);
}

// =========================================================================
// Frequency table invariants
// =========================================================================

#[test]
fn freq_table_cos_sin_unit_circle() {
    let cfg = RopeConfig::new(8, 32);
    let freqs = compute_frequencies(&cfg);
    let half_dim = cfg.head_dim / 2;

    // For every position and dim pair, cos² + sin² should be 1.0
    for pos in 0..cfg.max_seq_len {
        for i in 0..half_dim {
            let idx = (pos * half_dim + i) * 2;
            let cos_val = freqs[idx];
            let sin_val = freqs[idx + 1];
            let sum = cos_val * cos_val + sin_val * sin_val;
            assert!((sum - 1.0).abs() < 1e-5, "cos²+sin²={sum} at pos={pos}, pair={i}");
        }
    }
}

#[test]
fn freq_table_position_zero_all_ones() {
    // At pos=0, all angles are 0 → cos=1, sin=0
    let cfg = RopeConfig::new(64, 1);
    let freqs = compute_frequencies(&cfg);
    let half_dim = cfg.head_dim / 2;
    for i in 0..half_dim {
        assert!((freqs[2 * i] - 1.0).abs() < 1e-6, "cos should be 1 at pos 0, pair {i}");
        assert!(freqs[2 * i + 1].abs() < 1e-6, "sin should be 0 at pos 0, pair {i}");
    }
}

#[test]
fn freq_table_larger_base_slower_rotation() {
    // Larger base → smaller angles → less rotation
    let cfg_small = RopeConfig::new(4, 4);
    let cfg_large = RopeConfig::new(4, 4).with_base(1_000_000.0);
    let f_small = compute_frequencies(&cfg_small);
    let f_large = compute_frequencies(&cfg_large);

    // At position 1, sin values for large base should be smaller
    let half_dim = 2;
    let pos1_offset = half_dim * 2; // = head_dim
    for i in 0..half_dim {
        let sin_small = f_small[pos1_offset + 2 * i + 1].abs();
        let sin_large = f_large[pos1_offset + 2 * i + 1].abs();
        assert!(
            sin_large <= sin_small + 1e-6,
            "larger base should have smaller angle: sin_small={sin_small}, sin_large={sin_large}"
        );
    }
}

#[test]
fn freq_table_scaling_factor_doubles_angle() {
    let cfg1 = RopeConfig::new(4, 4);
    let cfg2 = RopeConfig::new(4, 4).with_scaling_factor(2.0);
    let f1 = compute_frequencies(&cfg1);
    let f2 = compute_frequencies(&cfg2);

    // At position 1 with scaling=2.0, the angle should be 2× the original
    // So sin for scaled at pos 1 should equal sin for unscaled at pos 2
    let hd = cfg1.head_dim;
    for i in 0..hd {
        assert!(
            (f2[hd + i] - f1[2 * hd + i]).abs() < 1e-5,
            "scaled(pos=1) should equal unscaled(pos=2) at idx {i}"
        );
    }
}

// =========================================================================
// apply_rope: all-zero input
// =========================================================================

#[test]
fn apply_rope_zero_input_stays_zero() {
    let cfg = RopeConfig::new(8, 16);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![0.0; 8];

    for pos in [0, 1, 7, 15] {
        data.fill(0.0);
        apply_rope(&mut data, pos, 8, &freqs);
        for &v in &data {
            assert!(v.abs() < 1e-10, "zero input should stay zero, got {v} at pos={pos}");
        }
    }
}

// =========================================================================
// apply_rope: rotation is orthogonal (norm-preserving)
// =========================================================================

#[test]
fn apply_rope_preserves_norm_various_data() {
    let cfg = RopeConfig::new(16, 64);
    let freqs = compute_frequencies(&cfg);

    let test_vecs: Vec<Vec<f32>> = vec![
        vec![1.0; 16],                                                  // uniform
        (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(), // alternating
        (0..16).map(|i| (i as f32) * 0.1).collect(),                    // ramp
        vec![
            100.0, -100.0, 0.01, -0.01, 50.0, -50.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            8.0,
        ], // mixed
    ];

    for (vi, original) in test_vecs.iter().enumerate() {
        let norm_before: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        for pos in [0, 1, 10, 63] {
            let mut data = original.clone();
            apply_rope(&mut data, pos, 16, &freqs);
            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-3,
                "norm not preserved: vec={vi}, pos={pos}, before={norm_before}, after={norm_after}"
            );
        }
    }
}

// =========================================================================
// Double rotation property
// =========================================================================

#[test]
fn apply_rope_double_rotation() {
    // Applying rope at pos=a then pos=b should differ from pos=a+b
    // (they're different because frequency table is indexed by absolute position)
    // But applying rope at pos=0 should be identity, so rope(pos=3) = rope(pos=3) ∘ rope(pos=0)
    let cfg = RopeConfig::new(4, 16);
    let freqs = compute_frequencies(&cfg);
    let original = vec![1.0, 2.0, 3.0, 4.0];

    // Apply at pos=3
    let mut single = original.clone();
    apply_rope(&mut single, 3, 4, &freqs);

    // Apply at pos=0 (identity) then pos=3
    let mut double = original.clone();
    apply_rope(&mut double, 0, 4, &freqs);
    apply_rope(&mut double, 3, 4, &freqs);

    for (i, (s, d)) in single.iter().zip(double.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-5,
            "identity then rotate should equal single rotate at dim {i}: {s} vs {d}"
        );
    }
}

// =========================================================================
// Batch processing edge cases
// =========================================================================

#[test]
fn batch_rope_single_head_single_position() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    let mut batch = vec![1.0, 2.0, 3.0, 4.0];
    let mut single = batch.clone();

    apply_rope_batch(&mut batch, 2, 1, 1, 4, &freqs);
    apply_rope(&mut single, 2, 4, &freqs);

    for (b, s) in batch.iter().zip(single.iter()) {
        assert!((b - s).abs() < 1e-6);
    }
}

#[test]
fn batch_rope_many_heads() {
    let head_dim = 8;
    let num_heads = 32;
    let seq_len = 4;
    let cfg = RopeConfig::new(head_dim, seq_len + 1);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let mut data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 5.0).collect();

    apply_rope_batch(&mut data, 0, seq_len, num_heads, head_dim, &freqs);
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn batch_rope_large_start_pos() {
    let head_dim = 8;
    let seq_len = 4;
    let start_pos = 4000;
    let cfg = RopeConfig::new(head_dim, start_pos + seq_len);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * 2 * head_dim;
    let mut data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

    apply_rope_batch(&mut data, start_pos, seq_len, 2, head_dim, &freqs);
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn batch_rope_heads_same_position_get_same_rotation() {
    let head_dim = 16;
    let num_heads = 8;
    let cfg = RopeConfig::new(head_dim, 4);
    let freqs = compute_frequencies(&cfg);

    // All heads get the same input vector
    let pattern: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
    let mut data: Vec<f32> = pattern.iter().copied().cycle().take(num_heads * head_dim).collect();

    apply_rope_batch(&mut data, 2, 1, num_heads, head_dim, &freqs);

    // All heads at the same position should produce identical results
    for h in 1..num_heads {
        for d in 0..head_dim {
            let expected = data[d];
            let got = data[h * head_dim + d];
            assert!(
                (expected - got).abs() < 1e-5,
                "head {h} dim {d} diverges: {got} vs {expected}"
            );
        }
    }
}

// =========================================================================
// 16K context (Phi-4 requirement)
// =========================================================================

#[test]
fn freq_table_16k_context() {
    let cfg = RopeConfig::new(128, 16384);
    let freqs = compute_frequencies(&cfg);
    assert_eq!(freqs.len(), 16384 * 128);

    // Spot check last position
    let last_pos_offset = 16383 * 128;
    let cos_val = freqs[last_pos_offset];
    let sin_val = freqs[last_pos_offset + 1];
    let sum = cos_val * cos_val + sin_val * sin_val;
    assert!((sum - 1.0).abs() < 1e-4, "cos²+sin² at last position: {sum}");
}

#[test]
fn apply_rope_at_position_16383() {
    let cfg = RopeConfig::new(8, 16384);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 0.0, 0.5, -0.5, 1.0, 0.0, 0.5, -0.5];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    apply_rope(&mut data, 16383, 8, &freqs);

    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(data.iter().all(|x| x.is_finite()));
    assert!((norm_before - norm_after).abs() < 1e-3, "norm not preserved at pos 16383");
}

// =========================================================================
// Non-standard bases (Phi-4, LLaMA 3, etc.)
// =========================================================================

#[test]
fn rope_phi4_base_config() {
    // Phi-4 uses base=10_000 with head_dim=128
    let cfg = RopeConfig::new(128, 16384);
    let freqs = compute_frequencies(&cfg);

    let mut data = vec![1.0; 128];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    apply_rope(&mut data, 1000, 128, &freqs);

    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_before - norm_after).abs() < 0.01, "norm not preserved with Phi-4 config");
}

#[test]
fn rope_llama3_base_config() {
    // LLaMA 3 uses base=500_000 with head_dim=128
    let cfg = RopeConfig::new(128, 8192).with_base(500_000.0);
    let freqs = compute_frequencies(&cfg);

    let mut data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    apply_rope(&mut data, 4096, 128, &freqs);

    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_before - norm_after).abs() < 0.1, "norm not preserved with LLaMA 3 config");
}

// =========================================================================
// Head dim variations (SIMD boundary testing)
// =========================================================================

#[test]
fn batch_rope_head_dim_2() {
    let cfg = RopeConfig::new(2, 8);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 0.0, 0.0, 1.0]; // 2 heads × 2 dims
    apply_rope_batch(&mut data, 3, 1, 2, 2, &freqs);
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn batch_rope_head_dim_6_avx_tail() {
    // head_dim=6 → only scalar tail in AVX2 path (no full 8-float chunks)
    let cfg = RopeConfig::new(6, 8);
    let freqs = compute_frequencies(&cfg);

    let mut batch_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut single_data = batch_data.clone();

    apply_rope_batch(&mut batch_data, 5, 1, 1, 6, &freqs);
    apply_rope(&mut single_data, 5, 6, &freqs);

    for (b, s) in batch_data.iter().zip(single_data.iter()) {
        assert!((b - s).abs() < 1e-6);
    }
}

#[test]
fn batch_rope_head_dim_10_partial_avx() {
    // head_dim=10 → 1 AVX chunk (8 floats) + 1 scalar pair
    let cfg = RopeConfig::new(10, 8);
    let freqs = compute_frequencies(&cfg);

    let mut batch_data: Vec<f32> = (0..10).map(|i| (i as f32) * 0.5).collect();
    let mut single_data = batch_data.clone();

    apply_rope_batch(&mut batch_data, 3, 1, 1, 10, &freqs);
    apply_rope(&mut single_data, 3, 10, &freqs);

    for (b, s) in batch_data.iter().zip(single_data.iter()) {
        assert!((b - s).abs() < 1e-5);
    }
}

#[test]
fn batch_rope_head_dim_128_typical() {
    // head_dim=128 → 16 AVX chunks, no tail
    let cfg = RopeConfig::new(128, 8);
    let freqs = compute_frequencies(&cfg);

    let num_heads = 4;
    let seq_len = 2;
    let total = seq_len * num_heads * 128;
    let original: Vec<f32> = (0..total).map(|i| ((i * 13 + 7) as f32) * 0.001 - 0.5).collect();

    let mut batch = original.clone();
    apply_rope_batch(&mut batch, 0, seq_len, num_heads, 128, &freqs);

    // Verify against single application
    let mut single = original.clone();
    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * 128;
            apply_rope(&mut single[offset..offset + 128], s, 128, &freqs);
        }
    }

    for (i, (b, s)) in batch.iter().zip(single.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "mismatch at idx {i}: batch={b} single={s}");
    }
}

// =========================================================================
// Negative & extreme values
// =========================================================================

#[test]
fn apply_rope_negative_values() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![-1.0, -2.0, -3.0, -4.0];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    apply_rope(&mut data, 2, 4, &freqs);

    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(data.iter().all(|x| x.is_finite()));
    assert!((norm_before - norm_after).abs() < 1e-3);
}

#[test]
fn apply_rope_very_small_values() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1e-30, 2e-30, 3e-30, 4e-30];

    apply_rope(&mut data, 2, 4, &freqs);

    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn apply_rope_large_values() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1e10, -1e10, 5e9, -5e9];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    apply_rope(&mut data, 1, 4, &freqs);

    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(data.iter().all(|x| x.is_finite()));
    assert!((norm_before - norm_after).abs() / norm_before < 1e-5, "relative norm error too large");
}
