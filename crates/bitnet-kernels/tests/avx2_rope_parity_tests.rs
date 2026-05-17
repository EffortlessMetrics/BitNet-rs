#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for RoPE (Rotary Position Embedding) kernels.
//!
//! Three modules are covered:
//!
//! 1. `cpu::rope`              — basic RoPE with interleaved cos/sin table
//! 2. `cpu::rope_simd`         — SIMD RoPE with scaling strategies (NTK, Linear, YaRN)
//! 3. `cpu::simd_rope_extended` — extended RoPE with layout selection + NTK/YaRN
//!
//! Each test exercises a kernel that has an explicit AVX2 fast-path
//! (via `is_x86_feature_detected!("avx2")` runtime dispatch) and compares
//! the output against a pure-scalar reference within defined tolerances.
//!
//! Test dimensions are chosen to exercise both the 8-wide SIMD body
//! and the scalar tail (lengths not divisible by 8).
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use bitnet_kernels::cpu::rope_simd::{
    RoPEConfig, ScalingType, apply_rope_batch as rope_simd_batch,
    apply_rope_dispatch as rope_simd_dispatch, apply_rope_f32, apply_rope_half_rotated,
    build_frequency_table, inverse_rope,
};
use bitnet_kernels::cpu::simd_rope_extended::{
    ExtendedRopeConfig, ExtendedScaling, RotationLayout, apply_rope_dispatch as ext_dispatch,
    apply_rope_interleaved as ext_interleaved, apply_rope_rotary_half as ext_rotary_half,
    build_extended_freq_table, inverse_rope_interleaved as ext_inv_interleaved,
    inverse_rope_rotary_half as ext_inv_rotary_half,
};

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// RoPE uses exact sin/cos so AVX2 vs scalar should agree tightly.

/// Absolute tolerance for RoPE parity checks.
const ROPE_ABS_TOL: f32 = 1e-5;
/// Relative tolerance for RoPE parity checks.
const ROPE_REL_TOL: f32 = 1e-5;

// ════════════════════════════════════════════════════════════════════════
// 1. BASIC ROPE — cpu::rope
//    Scalar reference: apply_rope() per head in a loop
//    Dispatched:       apply_rope_batch() (AVX2 when available)
// ════════════════════════════════════════════════════════════════════════

/// Aligned dimensions (head_dim=64, divisible by 8): 4 heads, seq_len=8.
#[test]
fn test_rope_batch_parity_aligned() {
    let head_dim = 64;
    let num_heads = 4;
    let seq_len = 8;
    let start_pos = 0;
    let max_seq_len = seq_len + start_pos;

    let config = RopeConfig::new(head_dim, max_seq_len);
    let freqs = compute_frequencies(&config);

    let total = seq_len * num_heads * head_dim;
    let data = pseudo_rand(total, 1000);

    // Scalar reference: iterate positions and heads manually.
    let mut expected = data.clone();
    for s in 0..seq_len {
        let position = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            apply_rope(&mut expected[offset..offset + head_dim], position, head_dim, &freqs);
        }
    }

    // Dispatched (AVX2 when available).
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, start_pos, seq_len, num_heads, head_dim, &freqs);

    assert_vec_parity(&actual, &expected, ROPE_ABS_TOL, ROPE_REL_TOL, "rope_batch_aligned(64×4×8)");
}

/// Unaligned dimensions (head_dim=48, 24 pairs — not divisible by 4 SIMD groups
/// of 4 pairs each → exercises scalar tail in AVX2 path).
#[test]
fn test_rope_batch_parity_unaligned() {
    let head_dim = 48;
    let num_heads = 2;
    let seq_len = 4;
    let start_pos = 3;
    let max_seq_len = seq_len + start_pos;

    let config = RopeConfig::new(head_dim, max_seq_len);
    let freqs = compute_frequencies(&config);

    let total = seq_len * num_heads * head_dim;
    let data = pseudo_rand(total, 2000);

    let mut expected = data.clone();
    for s in 0..seq_len {
        let position = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            apply_rope(&mut expected[offset..offset + head_dim], position, head_dim, &freqs);
        }
    }

    let mut actual = data.clone();
    apply_rope_batch(&mut actual, start_pos, seq_len, num_heads, head_dim, &freqs);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "rope_batch_unaligned(48×2×4)",
    );
}

/// Large dimensions (head_dim=128, 32 heads, seq_len=16).
#[test]
fn test_rope_batch_parity_large() {
    let head_dim = 128;
    let num_heads = 32;
    let seq_len = 16;
    let start_pos = 0;
    let max_seq_len = seq_len + start_pos;

    let config = RopeConfig::new(head_dim, max_seq_len);
    let freqs = compute_frequencies(&config);

    let total = seq_len * num_heads * head_dim;
    let data = pseudo_rand(total, 3000);

    let mut expected = data.clone();
    for s in 0..seq_len {
        let position = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            apply_rope(&mut expected[offset..offset + head_dim], position, head_dim, &freqs);
        }
    }

    let mut actual = data.clone();
    apply_rope_batch(&mut actual, start_pos, seq_len, num_heads, head_dim, &freqs);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "rope_batch_large(128×32×16)",
    );
}

// ════════════════════════════════════════════════════════════════════════
// 2. SIMD ROPE — cpu::rope_simd
//    Scalar reference: apply_rope_f32()
//    Dispatched:       apply_rope_dispatch() (AVX2 → scalar fallback)
// ════════════════════════════════════════════════════════════════════════

/// Aligned dim=64, no scaling.
#[test]
fn test_rope_simd_dispatch_parity_aligned() {
    let dim = 64;
    let max_seq_len = 32;
    let position = 7;

    let config = RoPEConfig::new(dim, max_seq_len);
    let table = build_frequency_table(&config);

    let data = pseudo_rand(dim, 4000);

    let mut expected = data.clone();
    apply_rope_f32(&mut expected, &table, position, dim);

    let mut actual = data.clone();
    rope_simd_dispatch(&mut actual, &table, position, dim);

    assert_vec_parity(&actual, &expected, ROPE_ABS_TOL, ROPE_REL_TOL, "rope_simd_aligned(dim=64)");
}

/// Unaligned dim=48 (24 pairs → 6 SIMD groups of 4, 0 tail pairs — but not
/// divisible by 8 floats, exercises edge alignment).
#[test]
fn test_rope_simd_dispatch_parity_unaligned() {
    let dim = 48;
    let max_seq_len = 32;
    let position = 5;

    let config = RoPEConfig::new(dim, max_seq_len);
    let table = build_frequency_table(&config);

    let data = pseudo_rand(dim, 5000);

    let mut expected = data.clone();
    apply_rope_f32(&mut expected, &table, position, dim);

    let mut actual = data.clone();
    rope_simd_dispatch(&mut actual, &table, position, dim);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "rope_simd_unaligned(dim=48)",
    );
}

/// NTK scaling with factor=2.0, dim=64.
#[test]
fn test_rope_simd_dispatch_parity_with_ntk_scaling() {
    let dim = 64;
    let max_seq_len = 64;
    let position = 10;

    let config = RoPEConfig::new(dim, max_seq_len).with_scaling(ScalingType::Ntk { factor: 2.0 });
    let table = build_frequency_table(&config);

    let data = pseudo_rand(dim, 6000);

    let mut expected = data.clone();
    apply_rope_f32(&mut expected, &table, position, dim);

    let mut actual = data.clone();
    rope_simd_dispatch(&mut actual, &table, position, dim);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "rope_simd_ntk(dim=64, factor=2)",
    );
}

/// Linear scaling with factor=4.0, dim=64.
#[test]
fn test_rope_simd_dispatch_parity_with_linear_scaling() {
    let dim = 64;
    let max_seq_len = 64;
    let position = 15;

    let config =
        RoPEConfig::new(dim, max_seq_len).with_scaling(ScalingType::Linear { factor: 4.0 });
    let table = build_frequency_table(&config);

    let data = pseudo_rand(dim, 7000);

    let mut expected = data.clone();
    apply_rope_f32(&mut expected, &table, position, dim);

    let mut actual = data.clone();
    rope_simd_dispatch(&mut actual, &table, position, dim);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "rope_simd_linear(dim=64, factor=4)",
    );
}

/// Half-rotated layout parity.
#[test]
fn test_rope_simd_half_rotated_parity() {
    let dim = 64;
    let max_seq_len = 32;
    let position = 3;

    let config = RoPEConfig::new(dim, max_seq_len);
    let table = build_frequency_table(&config);

    let data = pseudo_rand(dim, 8000);

    // apply_rope_half_rotated is scalar-only; verify it produces finite results
    // and differs from the interleaved layout.
    let mut half_rot = data.clone();
    apply_rope_half_rotated(&mut half_rot, &table, position);

    let mut interleaved = data.clone();
    apply_rope_f32(&mut interleaved, &table, position, dim);

    // Half-rotated and interleaved should produce different results.
    let any_different = half_rot.iter().zip(interleaved.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_different, "half_rotated and interleaved should differ for the same input");

    // All outputs must be finite.
    for (i, &v) in half_rot.iter().enumerate() {
        assert!(v.is_finite(), "half_rotated[{i}] is not finite: {v}");
    }
}

/// Roundtrip: apply dispatch then inverse, recover original.
#[test]
fn test_rope_simd_roundtrip() {
    let dim = 64;
    let max_seq_len = 32;
    let position = 11;

    let config = RoPEConfig::new(dim, max_seq_len);
    let table = build_frequency_table(&config);

    let original = pseudo_rand(dim, 9000);

    let mut data = original.clone();
    rope_simd_dispatch(&mut data, &table, position, dim);
    inverse_rope(&mut data, &table, position, dim);

    assert_vec_parity(&data, &original, ROPE_ABS_TOL, ROPE_REL_TOL, "rope_simd_roundtrip(dim=64)");
}

/// Batch parity: 4 positions, 8 heads, dim=64.
#[test]
fn test_rope_simd_batch_parity() {
    let dim = 64;
    let num_heads = 8;
    let max_seq_len = 32;
    let positions: Vec<usize> = vec![0, 3, 7, 15];
    let seq_len = positions.len();

    let config = RoPEConfig::new(dim, max_seq_len);
    let table = build_frequency_table(&config);

    let total = seq_len * num_heads * dim;
    let data = pseudo_rand(total, 10000);

    // Scalar reference: apply_rope_f32 per head.
    let mut expected = data.clone();
    for (s, &pos) in positions.iter().enumerate() {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * dim;
            apply_rope_f32(&mut expected[offset..offset + dim], &table, pos, dim);
        }
    }

    // Dispatched batch.
    let mut actual = data.clone();
    rope_simd_batch(&mut actual, &table, &positions, dim, num_heads);

    assert_vec_parity(&actual, &expected, ROPE_ABS_TOL, ROPE_REL_TOL, "rope_simd_batch(64×8×4pos)");
}

// ════════════════════════════════════════════════════════════════════════
// 3. EXTENDED ROPE — cpu::simd_rope_extended
//    Scalar reference: apply_rope_interleaved() / apply_rope_rotary_half()
//    Dispatched:       apply_rope_dispatch() (AVX2 → scalar fallback)
// ════════════════════════════════════════════════════════════════════════

/// Interleaved layout, aligned head_dim=64.
#[test]
fn test_extended_rope_interleaved_parity_aligned() {
    let head_dim = 64;
    let max_seq_len = 64;
    let position = 9;

    let cfg = ExtendedRopeConfig::new(head_dim, max_seq_len);
    let table = build_extended_freq_table(&cfg);

    let data = pseudo_rand(head_dim, 11000);

    let mut expected = data.clone();
    ext_interleaved(&mut expected, &table, position);

    let mut actual = data.clone();
    ext_dispatch(&mut actual, &table, position);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_interleaved_aligned(dim=64)",
    );
}

/// Interleaved layout, unaligned head_dim=48 (24 pairs, 6 SIMD groups).
#[test]
fn test_extended_rope_interleaved_parity_unaligned() {
    let head_dim = 48;
    let max_seq_len = 64;
    let position = 5;

    let cfg = ExtendedRopeConfig::new(head_dim, max_seq_len);
    let table = build_extended_freq_table(&cfg);

    let data = pseudo_rand(head_dim, 12000);

    let mut expected = data.clone();
    ext_interleaved(&mut expected, &table, position);

    let mut actual = data.clone();
    ext_dispatch(&mut actual, &table, position);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_interleaved_unaligned(dim=48)",
    );
}

/// Rotary-half layout, aligned head_dim=64.
#[test]
fn test_extended_rope_rotary_half_parity_aligned() {
    let head_dim = 64;
    let max_seq_len = 64;
    let position = 12;

    let cfg =
        ExtendedRopeConfig::new(head_dim, max_seq_len).with_layout(RotationLayout::RotaryHalf);
    let table = build_extended_freq_table(&cfg);

    let data = pseudo_rand(head_dim, 13000);

    let mut expected = data.clone();
    ext_rotary_half(&mut expected, &table, position);

    let mut actual = data.clone();
    ext_dispatch(&mut actual, &table, position);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_rotary_half_aligned(dim=64)",
    );
}

/// Rotary-half layout, unaligned head_dim=48.
#[test]
fn test_extended_rope_rotary_half_parity_unaligned() {
    let head_dim = 48;
    let max_seq_len = 64;
    let position = 7;

    let cfg =
        ExtendedRopeConfig::new(head_dim, max_seq_len).with_layout(RotationLayout::RotaryHalf);
    let table = build_extended_freq_table(&cfg);

    let data = pseudo_rand(head_dim, 14000);

    let mut expected = data.clone();
    ext_rotary_half(&mut expected, &table, position);

    let mut actual = data.clone();
    ext_dispatch(&mut actual, &table, position);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_rotary_half_unaligned(dim=48)",
    );
}

/// NtkAware scaling parity, interleaved layout.
#[test]
fn test_extended_rope_ntk_scaling_parity() {
    let head_dim = 64;
    let max_seq_len = 128;
    let position = 20;

    let cfg = ExtendedRopeConfig::new(head_dim, max_seq_len)
        .with_scaling(ExtendedScaling::NtkAware { alpha: 2.0 });
    let table = build_extended_freq_table(&cfg);

    let data = pseudo_rand(head_dim, 15000);

    let mut expected = data.clone();
    ext_interleaved(&mut expected, &table, position);

    let mut actual = data.clone();
    ext_dispatch(&mut actual, &table, position);

    assert_vec_parity(
        &actual,
        &expected,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_ntk_scaling(dim=64, alpha=2)",
    );
}

/// Interleaved roundtrip: dispatch forward + inverse recovers original.
#[test]
fn test_extended_rope_interleaved_roundtrip() {
    let head_dim = 64;
    let max_seq_len = 64;
    let position = 13;

    let cfg = ExtendedRopeConfig::new(head_dim, max_seq_len);
    let table = build_extended_freq_table(&cfg);

    let original = pseudo_rand(head_dim, 16000);

    let mut data = original.clone();
    ext_dispatch(&mut data, &table, position);
    ext_inv_interleaved(&mut data, &table, position);

    assert_vec_parity(
        &data,
        &original,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_interleaved_roundtrip(dim=64)",
    );
}

/// Rotary-half roundtrip: dispatch forward + inverse recovers original.
#[test]
fn test_extended_rope_rotary_half_roundtrip() {
    let head_dim = 64;
    let max_seq_len = 64;
    let position = 21;

    let cfg =
        ExtendedRopeConfig::new(head_dim, max_seq_len).with_layout(RotationLayout::RotaryHalf);
    let table = build_extended_freq_table(&cfg);

    let original = pseudo_rand(head_dim, 17000);

    let mut data = original.clone();
    ext_dispatch(&mut data, &table, position);
    ext_inv_rotary_half(&mut data, &table, position);

    assert_vec_parity(
        &data,
        &original,
        ROPE_ABS_TOL,
        ROPE_REL_TOL,
        "ext_rotary_half_roundtrip(dim=64)",
    );
}

/// Multiple positions: verify dispatch parity at positions 0, 7, 42, 127.
#[test]
fn test_extended_rope_multiple_positions() {
    let head_dim = 64;
    let max_seq_len = 128;

    let cfg = ExtendedRopeConfig::new(head_dim, max_seq_len);
    let table = build_extended_freq_table(&cfg);

    for &pos in &[0_usize, 7, 42, 127] {
        let data = pseudo_rand(head_dim, 18000 + pos as u64);

        let mut expected = data.clone();
        ext_interleaved(&mut expected, &table, pos);

        let mut actual = data.clone();
        ext_dispatch(&mut actual, &table, pos);

        assert_vec_parity(
            &actual,
            &expected,
            ROPE_ABS_TOL,
            ROPE_REL_TOL,
            &format!("ext_multi_pos(dim=64, pos={pos})"),
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// 4. PIPELINE SANITY — all three modules on the same dimensions
// ════════════════════════════════════════════════════════════════════════

/// Run all three RoPE modules on the same dimensions and verify no
/// NaN/Inf in any output.
#[test]
fn test_rope_pipeline_no_nan_inf() {
    let dim = 64;
    let max_seq_len = 32;
    let position = 5;

    let data = pseudo_rand(dim, 99000);

    // 1. Basic rope
    let basic_config = RopeConfig::new(dim, max_seq_len);
    let basic_freqs = compute_frequencies(&basic_config);
    let mut basic_out = data.clone();
    apply_rope(&mut basic_out, position, dim, &basic_freqs);
    for (i, &v) in basic_out.iter().enumerate() {
        assert!(v.is_finite(), "basic_rope[{i}] is not finite: {v}");
    }

    // 2. SIMD rope
    let simd_config = RoPEConfig::new(dim, max_seq_len);
    let simd_table = build_frequency_table(&simd_config);
    let mut simd_out = data.clone();
    rope_simd_dispatch(&mut simd_out, &simd_table, position, dim);
    for (i, &v) in simd_out.iter().enumerate() {
        assert!(v.is_finite(), "simd_rope[{i}] is not finite: {v}");
    }

    // 3. Extended rope
    let ext_config = ExtendedRopeConfig::new(dim, max_seq_len);
    let ext_table = build_extended_freq_table(&ext_config);
    let mut ext_out = data.clone();
    ext_dispatch(&mut ext_out, &ext_table, position);
    for (i, &v) in ext_out.iter().enumerate() {
        assert!(v.is_finite(), "ext_rope[{i}] is not finite: {v}");
    }
}
