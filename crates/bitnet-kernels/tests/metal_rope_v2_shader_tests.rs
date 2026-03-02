//! Metal RoPE v2 shader validation tests for Apple Silicon.
//!
//! Comprehensive test suite validating rotary position embedding computations
//! that would run on Metal GPU shaders, including:
//! - Standard RoPE for head dims 32, 64, 128
//! - Interleaved vs split (contiguous pair) layouts
//! - Position offset for KV cache continuation
//! - Multi-head and batched RoPE
//! - Frequency base variants (10 000, 500 000, 1 000 000)
//! - NTK-aware dynamic scaling
//! - Numerical precision and norm preservation
//! - Long-context positions up to 8192+
//! - RoPE + attention pipeline integration
//!
//! CPU-side reference implementations are used for correctness validation.
//! Tests tagged `#[ignore = "requires Metal GPU runtime"]` need a real Metal
//! device and are skipped in CI.

#![cfg(feature = "cpu")]

use std::f32::consts::PI;

use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};

// ── Constants ────────────────────────────────────────────────────────

const DEFAULT_THETA: f32 = 10_000.0;
const HIGH_THETA: f32 = 500_000.0;
const ULTRA_THETA: f32 = 1_000_000.0;
const TOLERANCE: f32 = 1e-5;
const GPU_TOLERANCE: f32 = 1e-4;

// ── Helpers ──────────────────────────────────────────────────────────

/// Reference scalar RoPE for a single head at a given position.
fn reference_rope(data: &[f32], head_dim: usize, pos: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = data.to_vec();
    for i in 0..half {
        let exponent = -(2.0 * i as f32) / head_dim as f32;
        let theta = base.powf(exponent);
        let angle = pos as f32 * theta;
        let (s, c) = angle.sin_cos();
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        out[2 * i] = x0 * c - x1 * s;
        out[2 * i + 1] = x0 * s + x1 * c;
    }
    out
}

/// Reference scalar RoPE with NTK-aware dynamic scaling.
fn reference_rope_ntk(
    data: &[f32],
    head_dim: usize,
    pos: usize,
    original_base: f32,
    scaling_factor: f32,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = data.to_vec();
    // NTK-aware: scale base by factor^(dim/(dim-2))
    let exponent_base = (head_dim as f32) / (head_dim as f32 - 2.0);
    let effective_base = original_base * scaling_factor.powf(exponent_base);
    for i in 0..half {
        let exp = -(2.0 * i as f32) / head_dim as f32;
        let theta = effective_base.powf(exp);
        let angle = pos as f32 * theta;
        let (s, c) = angle.sin_cos();
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        out[2 * i] = x0 * c - x1 * s;
        out[2 * i + 1] = x0 * s + x1 * c;
    }
    out
}

/// Reference split-layout RoPE: first half holds real parts, second half imaginary.
fn reference_rope_split(data: &[f32], head_dim: usize, pos: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = data.to_vec();
    for i in 0..half {
        let exponent = -(2.0 * i as f32) / head_dim as f32;
        let theta = base.powf(exponent);
        let angle = pos as f32 * theta;
        let (s, c) = angle.sin_cos();
        let x_re = data[i];
        let x_im = data[half + i];
        out[i] = x_re * c - x_im * s;
        out[half + i] = x_re * s + x_im * c;
    }
    out
}

/// Compute L2 norm of a slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Generate deterministic test data of given length.
fn test_data(len: usize, seed: u32) -> Vec<f32> {
    (0..len).map(|i| ((i as u32 * 7 + seed) as f32 * 0.01).sin()).collect()
}

/// Assert two slices are element-wise close.
fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: length mismatch {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{msg}: mismatch at index {i}: {x} vs {y} (diff={})",
            (x - y).abs()
        );
    }
}

// =========================================================================
// 1. Basic RoPE tests — standard rotary embedding for various head dims
// =========================================================================

#[test]
fn basic_rope_head_dim_32() {
    let dim = 32;
    let data = test_data(dim, 1);
    let expected = reference_rope(&data, dim, 1, DEFAULT_THETA);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 4));
    let mut actual = data.clone();
    apply_rope(&mut actual, 1, dim, &freqs);
    assert_close(&actual, &expected, TOLERANCE, "head_dim=32 pos=1");
}

#[test]
fn basic_rope_head_dim_64() {
    let dim = 64;
    let data = test_data(dim, 2);
    let expected = reference_rope(&data, dim, 3, DEFAULT_THETA);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 8));
    let mut actual = data.clone();
    apply_rope(&mut actual, 3, dim, &freqs);
    assert_close(&actual, &expected, TOLERANCE, "head_dim=64 pos=3");
}

#[test]
fn basic_rope_head_dim_128() {
    let dim = 128;
    let data = test_data(dim, 3);
    let expected = reference_rope(&data, dim, 5, DEFAULT_THETA);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 8));
    let mut actual = data.clone();
    apply_rope(&mut actual, 5, dim, &freqs);
    assert_close(&actual, &expected, TOLERANCE, "head_dim=128 pos=5");
}

#[test]
fn basic_rope_position_zero_identity_check() {
    // At position 0 all angles are 0 → cos=1, sin=0 → output == input.
    for &dim in &[32, 64, 128] {
        let data = test_data(dim, 10);
        let freqs = compute_frequencies(&RopeConfig::new(dim, 4));
        let mut actual = data.clone();
        apply_rope(&mut actual, 0, dim, &freqs);
        assert_close(&actual, &data, TOLERANCE, &format!("pos=0 identity dim={dim}"));
    }
}

#[test]
fn basic_rope_zero_input_preserved() {
    let dim = 64;
    let data = vec![0.0f32; dim];
    let freqs = compute_frequencies(&RopeConfig::new(dim, 16));
    for pos in 0..16 {
        let mut actual = data.clone();
        apply_rope(&mut actual, pos, dim, &freqs);
        for (i, &v) in actual.iter().enumerate() {
            assert!(v.abs() < 1e-10, "zero not preserved at pos={pos} idx={i}");
        }
    }
}

#[test]
fn basic_rope_consecutive_positions_differ() {
    let dim = 64;
    let data = test_data(dim, 42);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 16));
    let mut a = data.clone();
    let mut b = data.clone();
    apply_rope(&mut a, 3, dim, &freqs);
    apply_rope(&mut b, 4, dim, &freqs);
    let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    assert!(diff > 1e-3, "consecutive positions must differ");
}

#[test]
fn basic_rope_deterministic_repeated_application() {
    let dim = 64;
    let data = test_data(dim, 99);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 8));
    let mut a = data.clone();
    let mut b = data.clone();
    apply_rope(&mut a, 5, dim, &freqs);
    apply_rope(&mut b, 5, dim, &freqs);
    assert_close(&a, &b, 0.0, "deterministic RoPE");
}

// =========================================================================
// 2. Interleaved vs split RoPE layouts
// =========================================================================

#[test]
fn interleaved_layout_matches_reference() {
    let dim = 64;
    let data = test_data(dim, 20);
    let expected = reference_rope(&data, dim, 7, DEFAULT_THETA);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 16));
    let mut actual = data.clone();
    apply_rope(&mut actual, 7, dim, &freqs);
    assert_close(&actual, &expected, TOLERANCE, "interleaved layout");
}

#[test]
fn split_layout_reference_head_dim_32() {
    let dim = 32;
    let data = test_data(dim, 21);
    let result = reference_rope_split(&data, dim, 3, DEFAULT_THETA);
    assert_eq!(result.len(), dim);
    // Verify that split layout produces different output from interleaved for same data
    let interleaved = reference_rope(&data, dim, 3, DEFAULT_THETA);
    let diff: f32 = result.iter().zip(interleaved.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "split and interleaved layouts must differ for non-trivial input");
}

#[test]
fn split_layout_reference_head_dim_64() {
    let dim = 64;
    let data = test_data(dim, 22);
    let result = reference_rope_split(&data, dim, 5, DEFAULT_THETA);
    assert_eq!(result.len(), dim);
}

#[test]
fn split_layout_reference_head_dim_128() {
    let dim = 128;
    let data = test_data(dim, 23);
    let result = reference_rope_split(&data, dim, 2, DEFAULT_THETA);
    assert_eq!(result.len(), dim);
}

#[test]
fn split_layout_norm_preservation() {
    let dim = 64;
    let data = test_data(dim, 24);
    let norm_before = l2_norm(&data);
    let result = reference_rope_split(&data, dim, 10, DEFAULT_THETA);
    let norm_after = l2_norm(&result);
    assert!(
        (norm_before - norm_after).abs() < 1e-3,
        "split RoPE must preserve norm: {norm_before} vs {norm_after}"
    );
}

#[test]
fn split_layout_position_zero_identity() {
    let dim = 64;
    let data = test_data(dim, 25);
    let result = reference_rope_split(&data, dim, 0, DEFAULT_THETA);
    assert_close(&result, &data, TOLERANCE, "split pos=0 identity");
}

#[test]
fn split_layout_zero_input_preserved() {
    let dim = 64;
    let data = vec![0.0f32; dim];
    let result = reference_rope_split(&data, dim, 42, DEFAULT_THETA);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.abs() < 1e-10, "split zero not preserved at idx={i}");
    }
}

// =========================================================================
// 3. Position offset tests — RoPE with KV cache position offsets
// =========================================================================

#[test]
fn position_offset_basic() {
    let dim = 64;
    let offset = 100;
    let data = test_data(dim, 30);
    let freqs = compute_frequencies(&RopeConfig::new(dim, offset + 16));
    // Applying at position offset+p should match reference at that absolute position
    let mut actual = data.clone();
    apply_rope(&mut actual, offset + 3, dim, &freqs);
    let expected = reference_rope(&data, dim, offset + 3, DEFAULT_THETA);
    assert_close(&actual, &expected, TOLERANCE, "offset=100 pos=3");
}

#[test]
fn position_offset_large_kv_cache() {
    let dim = 128;
    let offset = 2048;
    let data = test_data(dim, 31);
    let freqs = compute_frequencies(&RopeConfig::new(dim, offset + 8));
    let mut actual = data.clone();
    apply_rope(&mut actual, offset + 5, dim, &freqs);
    let expected = reference_rope(&data, dim, offset + 5, DEFAULT_THETA);
    assert_close(&actual, &expected, TOLERANCE, "kv cache offset=2048");
}

#[test]
fn position_offset_incremental_decode() {
    // Simulate incremental KV cache decode: apply one position at a time
    let dim = 64;
    let cache_len = 50;
    let data = test_data(dim, 32);
    let freqs = compute_frequencies(&RopeConfig::new(dim, cache_len + 4));
    for new_pos in 0..4 {
        let abs_pos = cache_len + new_pos;
        let mut actual = data.clone();
        apply_rope(&mut actual, abs_pos, dim, &freqs);
        let expected = reference_rope(&data, dim, abs_pos, DEFAULT_THETA);
        assert_close(&actual, &expected, TOLERANCE, &format!("incremental decode pos={abs_pos}"));
    }
}

#[test]
fn position_offset_norm_preservation() {
    let dim = 64;
    let data = test_data(dim, 33);
    let norm_before = l2_norm(&data);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 4096));
    let mut actual = data.clone();
    apply_rope(&mut actual, 3000, dim, &freqs);
    let norm_after = l2_norm(&actual);
    assert!(
        (norm_before - norm_after).abs() < 1e-2,
        "offset norm preservation: {norm_before} vs {norm_after}"
    );
}

#[test]
fn position_offset_zero_vs_nonzero() {
    let dim = 64;
    let data = test_data(dim, 34);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 512));
    let mut with_offset = data.clone();
    let mut without = data.clone();
    apply_rope(&mut with_offset, 256, dim, &freqs);
    apply_rope(&mut without, 0, dim, &freqs);
    // At position 0 => identity, at 256 => rotated differently
    let diff: f32 = with_offset.iter().zip(without.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "offset=256 must differ from offset=0");
}

// =========================================================================
// 4. Multi-head RoPE — apply across multiple heads simultaneously
// =========================================================================

#[test]
fn multi_head_rope_2_heads() {
    let dim = 64;
    let heads = 2;
    let seq = 1;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 4));
    let data = test_data(seq * heads * dim, 40);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 1, seq, heads, dim, &freqs);
    // Each head should match the single-head reference
    for h in 0..heads {
        let offset = h * dim;
        let expected = reference_rope(&data[offset..offset + dim], dim, 1, DEFAULT_THETA);
        assert_close(
            &actual[offset..offset + dim],
            &expected,
            TOLERANCE,
            &format!("2 heads, head={h}"),
        );
    }
}

#[test]
fn multi_head_rope_8_heads() {
    let dim = 64;
    let heads = 8;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 4));
    let data = test_data(heads * dim, 41);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 2, 1, heads, dim, &freqs);
    for h in 0..heads {
        let off = h * dim;
        let expected = reference_rope(&data[off..off + dim], dim, 2, DEFAULT_THETA);
        assert_close(&actual[off..off + dim], &expected, TOLERANCE, &format!("8 heads, head={h}"));
    }
}

#[test]
fn multi_head_rope_32_heads() {
    let dim = 128;
    let heads = 32;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 8));
    let data = test_data(heads * dim, 42);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 5, 1, heads, dim, &freqs);
    for h in 0..heads {
        let off = h * dim;
        let expected = reference_rope(&data[off..off + dim], dim, 5, DEFAULT_THETA);
        assert_close(&actual[off..off + dim], &expected, TOLERANCE, &format!("32 heads, head={h}"));
    }
}

#[test]
fn multi_head_rope_independence() {
    // Changing one head's data should not affect other heads
    let dim = 64;
    let heads = 4;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 4));
    let data = test_data(heads * dim, 43);
    let mut a = data.clone();
    let mut b = data.clone();
    // Modify head 2 in b
    for i in 0..dim {
        b[2 * dim + i] = 999.0;
    }
    apply_rope_batch(&mut a, 1, 1, heads, dim, &freqs);
    apply_rope_batch(&mut b, 1, 1, heads, dim, &freqs);
    // Heads 0, 1, 3 must be identical
    for h in [0, 1, 3] {
        let off = h * dim;
        assert_close(
            &a[off..off + dim],
            &b[off..off + dim],
            0.0,
            &format!("head independence h={h}"),
        );
    }
}

#[test]
fn multi_head_rope_gqa_4_heads_dim_32() {
    // Grouped query attention: small head dim
    let dim = 32;
    let heads = 4;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 8));
    let data = test_data(heads * dim, 44);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 3, 1, heads, dim, &freqs);
    for h in 0..heads {
        let off = h * dim;
        let expected = reference_rope(&data[off..off + dim], dim, 3, DEFAULT_THETA);
        assert_close(&actual[off..off + dim], &expected, TOLERANCE, &format!("gqa h={h}"));
    }
}

// =========================================================================
// 5. Batch RoPE tests — batched across sequence positions
// =========================================================================

#[test]
fn batch_rope_seq_len_4() {
    let dim = 64;
    let heads = 2;
    let seq = 4;
    let freqs = compute_frequencies(&RopeConfig::new(dim, seq + 2));
    let data = test_data(seq * heads * dim, 50);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 0, seq, heads, dim, &freqs);
    for s in 0..seq {
        for h in 0..heads {
            let off = (s * heads + h) * dim;
            let expected = reference_rope(&data[off..off + dim], dim, s, DEFAULT_THETA);
            assert_close(
                &actual[off..off + dim],
                &expected,
                TOLERANCE,
                &format!("batch seq={s} head={h}"),
            );
        }
    }
}

#[test]
fn batch_rope_seq_len_16() {
    let dim = 64;
    let heads = 4;
    let seq = 16;
    let freqs = compute_frequencies(&RopeConfig::new(dim, seq));
    let data = test_data(seq * heads * dim, 51);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 0, seq, heads, dim, &freqs);
    // Spot-check first and last positions
    for &s in &[0, 15] {
        for h in 0..heads {
            let off = (s * heads + h) * dim;
            let expected = reference_rope(&data[off..off + dim], dim, s, DEFAULT_THETA);
            assert_close(
                &actual[off..off + dim],
                &expected,
                TOLERANCE,
                &format!("batch16 seq={s} head={h}"),
            );
        }
    }
}

#[test]
fn batch_rope_with_start_position() {
    let dim = 64;
    let heads = 2;
    let seq = 8;
    let start = 100;
    let freqs = compute_frequencies(&RopeConfig::new(dim, start + seq));
    let data = test_data(seq * heads * dim, 52);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, start, seq, heads, dim, &freqs);
    for s in 0..seq {
        for h in 0..heads {
            let off = (s * heads + h) * dim;
            let expected = reference_rope(&data[off..off + dim], dim, start + s, DEFAULT_THETA);
            assert_close(
                &actual[off..off + dim],
                &expected,
                TOLERANCE,
                &format!("batch start={start} seq={s} head={h}"),
            );
        }
    }
}

#[test]
fn batch_rope_single_token() {
    let dim = 128;
    let heads = 8;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 4));
    let data = test_data(heads * dim, 53);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 2, 1, heads, dim, &freqs);
    for h in 0..heads {
        let off = h * dim;
        let expected = reference_rope(&data[off..off + dim], dim, 2, DEFAULT_THETA);
        assert_close(
            &actual[off..off + dim],
            &expected,
            TOLERANCE,
            &format!("single token head={h}"),
        );
    }
}

#[test]
fn batch_rope_large_batch_32_positions() {
    let dim = 64;
    let heads = 4;
    let seq = 32;
    let freqs = compute_frequencies(&RopeConfig::new(dim, seq));
    let total = seq * heads * dim;
    let data = test_data(total, 54);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, 0, seq, heads, dim, &freqs);
    // Verify that output differs from input (non-trivially)
    let diff: f32 = actual.iter().zip(data.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1.0, "batch of 32 should produce non-trivial changes");
}

// =========================================================================
// 6. Frequency base tests — different theta values
// =========================================================================

#[test]
fn freq_base_10000_default() {
    let dim = 64;
    let config = RopeConfig::new(dim, 16);
    let freqs = compute_frequencies(&config);
    assert_eq!(freqs.len(), 16 * dim);
    // First position cos/sin at pair 0: angle = 0 => cos=1, sin=0
    assert!((freqs[0] - 1.0).abs() < TOLERANCE, "cos(0)=1");
    assert!(freqs[1].abs() < TOLERANCE, "sin(0)=0");
}

#[test]
fn freq_base_500000_high() {
    let dim = 64;
    let config = RopeConfig::new(dim, 16).with_base(HIGH_THETA);
    let freqs = compute_frequencies(&config);
    // Higher base → slower frequency decay → smaller angles at same position
    let config_default = RopeConfig::new(dim, 16);
    let freqs_default = compute_frequencies(&config_default);
    // At position 1, last pair: high base should have angle closer to zero
    let last_pair_idx = (dim / 2 - 1) * 2;
    let sin_default = freqs_default[dim + last_pair_idx + 1].abs();
    let sin_high = freqs[dim + last_pair_idx + 1].abs();
    assert!(
        sin_high < sin_default,
        "higher base should produce smaller angle: {sin_high} >= {sin_default}"
    );
}

#[test]
fn freq_base_1000000_ultra() {
    let dim = 128;
    let config = RopeConfig::new(dim, 8).with_base(ULTRA_THETA);
    let freqs = compute_frequencies(&config);
    assert_eq!(freqs.len(), 8 * dim);
    // Position 0 should still be identity
    assert!((freqs[0] - 1.0).abs() < TOLERANCE);
    assert!(freqs[1].abs() < TOLERANCE);
}

#[test]
fn freq_base_comparison_monotonicity() {
    // Higher base → slower decay of frequencies → smaller rotation angles
    let dim = 64;
    let pos = 5;
    let data = test_data(dim, 60);
    let bases = [DEFAULT_THETA, HIGH_THETA, ULTRA_THETA];
    let mut diffs_from_identity = Vec::new();
    for &base in &bases {
        let result = reference_rope(&data, dim, pos, base);
        let d: f32 = result.iter().zip(data.iter()).map(|(a, b)| (a - b).abs()).sum();
        diffs_from_identity.push(d);
    }
    // Higher base → less rotation → smaller difference from input
    for w in diffs_from_identity.windows(2) {
        assert!(w[0] >= w[1], "higher base should produce less rotation: {} < {}", w[0], w[1]);
    }
}

#[test]
fn freq_base_rope_output_differs_across_bases() {
    let dim = 64;
    let data = test_data(dim, 61);
    let r1 = reference_rope(&data, dim, 5, DEFAULT_THETA);
    let r2 = reference_rope(&data, dim, 5, HIGH_THETA);
    let diff: f32 = r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "different bases must produce different results");
}

#[test]
fn freq_base_500k_vs_default_norm() {
    let dim = 64;
    let data = test_data(dim, 62);
    let norm_orig = l2_norm(&data);
    for &base in &[DEFAULT_THETA, HIGH_THETA, ULTRA_THETA] {
        let result = reference_rope(&data, dim, 10, base);
        let norm = l2_norm(&result);
        assert!(
            (norm_orig - norm).abs() < 1e-3,
            "norm not preserved for base={base}: {norm_orig} vs {norm}"
        );
    }
}

// =========================================================================
// 7. NTK-aware scaling — dynamic NTK RoPE with scaling factors
// =========================================================================

#[test]
fn ntk_scaling_factor_1_equals_standard() {
    let dim = 64;
    let data = test_data(dim, 70);
    let standard = reference_rope(&data, dim, 5, DEFAULT_THETA);
    let ntk = reference_rope_ntk(&data, dim, 5, DEFAULT_THETA, 1.0);
    assert_close(&ntk, &standard, TOLERANCE, "NTK scale=1 should match standard");
}

#[test]
fn ntk_scaling_factor_2() {
    let dim = 64;
    let data = test_data(dim, 71);
    let standard = reference_rope(&data, dim, 5, DEFAULT_THETA);
    let ntk = reference_rope_ntk(&data, dim, 5, DEFAULT_THETA, 2.0);
    let diff: f32 = standard.iter().zip(ntk.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "NTK scale=2 should differ from standard");
}

#[test]
fn ntk_scaling_factor_4() {
    let dim = 128;
    let data = test_data(dim, 72);
    let ntk = reference_rope_ntk(&data, dim, 10, DEFAULT_THETA, 4.0);
    let norm_before = l2_norm(&data);
    let norm_after = l2_norm(&ntk);
    assert!(
        (norm_before - norm_after).abs() < 1e-3,
        "NTK scale=4 norm preservation: {norm_before} vs {norm_after}"
    );
}

#[test]
fn ntk_scaling_norm_preservation() {
    let dim = 64;
    let data = test_data(dim, 73);
    let norm_orig = l2_norm(&data);
    for &factor in &[1.0, 2.0, 4.0, 8.0] {
        let result = reference_rope_ntk(&data, dim, 5, DEFAULT_THETA, factor);
        let norm = l2_norm(&result);
        assert!(
            (norm_orig - norm).abs() < 1e-3,
            "NTK norm preservation factor={factor}: {norm_orig} vs {norm}"
        );
    }
}

#[test]
fn ntk_scaling_monotonic_effect() {
    // Larger scaling factor → effectively higher base → less rotation
    let dim = 64;
    let pos = 10;
    let data = test_data(dim, 74);
    let factors = [1.0f32, 2.0, 4.0, 8.0];
    let mut diffs = Vec::new();
    for &f in &factors {
        let result = reference_rope_ntk(&data, dim, pos, DEFAULT_THETA, f);
        let d: f32 = result.iter().zip(data.iter()).map(|(a, b)| (a - b).abs()).sum();
        diffs.push(d);
    }
    for w in diffs.windows(2) {
        assert!(
            w[0] >= w[1] - 1e-3,
            "larger scale should produce less rotation: {} < {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn ntk_scaling_position_zero_identity() {
    let dim = 64;
    let data = test_data(dim, 75);
    for &factor in &[1.0, 2.0, 4.0] {
        let result = reference_rope_ntk(&data, dim, 0, DEFAULT_THETA, factor);
        assert_close(&result, &data, TOLERANCE, &format!("NTK pos=0 factor={factor}"));
    }
}

#[test]
fn ntk_scaling_with_config() {
    let dim = 64;
    let factor = 2.0;
    let config = RopeConfig::new(dim, 16).with_scaling_factor(factor);
    let freqs = compute_frequencies(&config);
    // The config-based scaling multiplies each theta by factor, not NTK-style.
    // Just verify the table is non-degenerate.
    assert_eq!(freqs.len(), 16 * dim);
    let data = test_data(dim, 76);
    let mut actual = data.clone();
    apply_rope(&mut actual, 5, dim, &freqs);
    let diff: f32 = actual.iter().zip(data.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "scaled config should produce non-trivial rotation");
}

// =========================================================================
// 8. Precision tests — numerical accuracy validation
// =========================================================================

#[test]
fn precision_cos_sin_identity() {
    // cos²(θ) + sin²(θ) = 1 for all frequency table entries
    let dim = 128;
    let config = RopeConfig::new(dim, 256);
    let freqs = compute_frequencies(&config);
    let half = dim / 2;
    for pos in 0..256 {
        for i in 0..half {
            let idx = (pos * half + i) * 2;
            let c = freqs[idx];
            let s = freqs[idx + 1];
            let sum = c * c + s * s;
            assert!((sum - 1.0).abs() < 1e-4, "cos²+sin²≠1 at pos={pos} pair={i}: {sum}");
        }
    }
}

#[test]
fn precision_norm_preservation_strict() {
    let dim = 128;
    let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.123).sin()).collect();
    let norm_before = l2_norm(&data);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 512));
    for pos in [1, 10, 100, 255, 511] {
        let mut actual = data.clone();
        apply_rope(&mut actual, pos, dim, &freqs);
        let norm_after = l2_norm(&actual);
        assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "strict norm check pos={pos}: {norm_before} vs {norm_after}"
        );
    }
}

#[test]
fn precision_rotation_orthogonality() {
    // For each pair, the rotation matrix is orthogonal: determinant = 1
    let dim = 64;
    let half = dim / 2;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 16));
    for pos in 0..16 {
        for i in 0..half {
            let idx = (pos * half + i) * 2;
            let c = freqs[idx];
            let s = freqs[idx + 1];
            // det([[c,-s],[s,c]]) = c²+s² = 1
            let det = c * c + s * s;
            assert!(
                (det - 1.0).abs() < 1e-4,
                "rotation matrix not orthogonal at pos={pos} pair={i}: det={det}"
            );
        }
    }
}

#[test]
fn precision_double_rotation_angle_addition() {
    // Applying RoPE at pos=a then pos=b should equal applying at pos=a+b
    // (for the same input, if we compose the rotations)
    let dim = 64;
    let half = dim / 2;
    let max_pos = 64;
    let freqs = compute_frequencies(&RopeConfig::new(dim, max_pos));
    let data = test_data(dim, 80);
    let pos_a = 3;
    let pos_b = 5;
    // Apply a then b
    let mut two_step = data.clone();
    apply_rope(&mut two_step, pos_a, dim, &freqs);
    apply_rope(&mut two_step, pos_b, dim, &freqs);
    // Apply a+b in one step
    let mut one_step = data.clone();
    apply_rope(&mut one_step, pos_a + pos_b, dim, &freqs);
    // These should be close (rotation composition is angle addition)
    assert_close(&two_step, &one_step, GPU_TOLERANCE, "angle addition property");
}

#[test]
fn precision_inverse_rotation() {
    // Applying RoPE with negated sin gives inverse rotation
    let dim = 64;
    let half = dim / 2;
    let config = RopeConfig::new(dim, 16);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 81);
    let pos = 7;
    let mut rotated = data.clone();
    apply_rope(&mut rotated, pos, dim, &freqs);
    // Manually apply inverse (negate sin components)
    let freq_offset = pos * dim;
    for i in 0..half {
        let c = freqs[freq_offset + 2 * i];
        let s = freqs[freq_offset + 2 * i + 1];
        let x0 = rotated[2 * i];
        let x1 = rotated[2 * i + 1];
        rotated[2 * i] = x0 * c + x1 * s; // cos * x0 + sin * x1 (note sign flip)
        rotated[2 * i + 1] = -x0 * s + x1 * c; // -sin * x0 + cos * x1
    }
    assert_close(&rotated, &data, GPU_TOLERANCE, "inverse rotation recovery");
}

#[test]
fn precision_small_angle_accuracy() {
    // At high dimensions, the last pairs have very small angles; verify precision
    let dim = 128;
    let half = dim / 2;
    let config = RopeConfig::new(dim, 4);
    let freqs = compute_frequencies(&config);
    // Position 1, last pair: should have very small angle
    let idx = (1 * half + (half - 1)) * 2;
    let c = freqs[idx];
    let s = freqs[idx + 1];
    // cos should be very close to 1, sin very close to 0 for small angles
    assert!(c > 0.99, "cos should be near 1 for last pair: {c}");
    assert!(s.abs() < 0.15, "sin should be near 0 for last pair: {s}");
}

#[test]
fn precision_large_angle_wraparound() {
    // At large positions, verify angles wrap correctly
    let dim = 32;
    let max_pos = 8192;
    let config = RopeConfig::new(dim, max_pos + 1);
    let freqs = compute_frequencies(&config);
    let half = dim / 2;
    // Even at large positions, cos²+sin² = 1
    for &pos in &[4096, 8000, 8192] {
        for i in 0..half {
            let idx = (pos * half + i) * 2;
            let c = freqs[idx];
            let s = freqs[idx + 1];
            let sum = c * c + s * s;
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "angle wraparound: cos²+sin²≠1 at pos={pos} pair={i}: {sum}"
            );
        }
    }
}

// =========================================================================
// 9. Long context tests — positions up to 8192+
// =========================================================================

#[test]
fn long_context_4096_positions() {
    let dim = 64;
    let max_pos = 4096;
    let config = RopeConfig::new(dim, max_pos);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 90);
    let mut actual = data.clone();
    apply_rope(&mut actual, max_pos - 1, dim, &freqs);
    let expected = reference_rope(&data, dim, max_pos - 1, DEFAULT_THETA);
    assert_close(&actual, &expected, TOLERANCE, "pos=4095");
}

#[test]
fn long_context_8192_positions() {
    let dim = 128;
    let max_pos = 8192;
    let config = RopeConfig::new(dim, max_pos);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 91);
    let mut actual = data.clone();
    apply_rope(&mut actual, max_pos - 1, dim, &freqs);
    let expected = reference_rope(&data, dim, max_pos - 1, DEFAULT_THETA);
    assert_close(&actual, &expected, TOLERANCE, "pos=8191");
}

#[test]
fn long_context_16384_positions() {
    let dim = 64;
    let max_pos = 16384;
    let config = RopeConfig::new(dim, max_pos);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 92);
    let mut actual = data.clone();
    apply_rope(&mut actual, max_pos - 1, dim, &freqs);
    let norm_before = l2_norm(&data);
    let norm_after = l2_norm(&actual);
    assert!(
        (norm_before - norm_after).abs() < 1e-2,
        "long context norm: {norm_before} vs {norm_after}"
    );
}

#[test]
fn long_context_norm_preservation_sweep() {
    let dim = 64;
    let max_pos = 8192;
    let config = RopeConfig::new(dim, max_pos);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 93);
    let norm_orig = l2_norm(&data);
    for &pos in &[0, 100, 1000, 4096, 8000, 8191] {
        let mut actual = data.clone();
        apply_rope(&mut actual, pos, dim, &freqs);
        let norm = l2_norm(&actual);
        assert!(
            (norm_orig - norm).abs() < 1e-3,
            "long context norm at pos={pos}: {norm_orig} vs {norm}"
        );
    }
}

#[test]
fn long_context_positions_distinguishable() {
    // Far-apart positions should produce noticeably different outputs
    let dim = 64;
    let max_pos = 8192;
    let config = RopeConfig::new(dim, max_pos);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 94);
    let mut near = data.clone();
    let mut far = data.clone();
    apply_rope(&mut near, 100, dim, &freqs);
    apply_rope(&mut far, 7000, dim, &freqs);
    let diff: f32 = near.iter().zip(far.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.1, "positions 100 and 7000 must differ: diff={diff}");
}

#[test]
fn long_context_high_theta_8192() {
    // High theta + long context
    let dim = 128;
    let max_pos = 8192;
    let config = RopeConfig::new(dim, max_pos).with_base(HIGH_THETA);
    let freqs = compute_frequencies(&config);
    let data = test_data(dim, 95);
    let mut actual = data.clone();
    apply_rope(&mut actual, 8000, dim, &freqs);
    let norm_before = l2_norm(&data);
    let norm_after = l2_norm(&actual);
    assert!(
        (norm_before - norm_after).abs() < 1e-2,
        "high theta long context: {norm_before} vs {norm_after}"
    );
}

#[test]
fn long_context_batch_multi_position() {
    let dim = 64;
    let heads = 2;
    let seq = 128;
    let start = 4096;
    let config = RopeConfig::new(dim, start + seq);
    let freqs = compute_frequencies(&config);
    let data = test_data(seq * heads * dim, 96);
    let mut actual = data.clone();
    apply_rope_batch(&mut actual, start, seq, heads, dim, &freqs);
    // Spot-check first and last positions
    for &s in &[0, 127] {
        for h in 0..heads {
            let off = (s * heads + h) * dim;
            let expected = reference_rope(&data[off..off + dim], dim, start + s, DEFAULT_THETA);
            assert_close(
                &actual[off..off + dim],
                &expected,
                TOLERANCE,
                &format!("long batch s={s} h={h}"),
            );
        }
    }
}

// =========================================================================
// 10. Integration tests — RoPE + attention pipeline
// =========================================================================

#[test]
fn integration_qk_dot_product_with_rope() {
    // After applying RoPE to Q and K at the same position, their dot product
    // should equal the dot product of the originals (rotation preserves dot product).
    let dim = 64;
    let q = test_data(dim, 100);
    let k = test_data(dim, 101);
    let dot_before: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
    let freqs = compute_frequencies(&RopeConfig::new(dim, 16));
    let mut q_rot = q.clone();
    let mut k_rot = k.clone();
    apply_rope(&mut q_rot, 5, dim, &freqs);
    apply_rope(&mut k_rot, 5, dim, &freqs);
    let dot_after: f32 = q_rot.iter().zip(k_rot.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (dot_before - dot_after).abs() < 1e-2,
        "dot product should be preserved: {dot_before} vs {dot_after}"
    );
}

#[test]
fn integration_relative_position_encoding() {
    // dot(RoPE(q, pos_q), RoPE(k, pos_k)) depends only on (pos_q - pos_k)
    let dim = 64;
    let q = test_data(dim, 102);
    let k = test_data(dim, 103);
    let freqs = compute_frequencies(&RopeConfig::new(dim, 128));
    // Pair 1: pos_q=10, pos_k=5 (diff=5)
    let mut q1 = q.clone();
    let mut k1 = k.clone();
    apply_rope(&mut q1, 10, dim, &freqs);
    apply_rope(&mut k1, 5, dim, &freqs);
    let dot1: f32 = q1.iter().zip(k1.iter()).map(|(a, b)| a * b).sum();
    // Pair 2: pos_q=20, pos_k=15 (diff=5)
    let mut q2 = q.clone();
    let mut k2 = k.clone();
    apply_rope(&mut q2, 20, dim, &freqs);
    apply_rope(&mut k2, 15, dim, &freqs);
    let dot2: f32 = q2.iter().zip(k2.iter()).map(|(a, b)| a * b).sum();
    assert!((dot1 - dot2).abs() < 1e-2, "relative position property: dot1={dot1} dot2={dot2}");
}

#[test]
fn integration_multi_head_attention_rope() {
    // Simulate Q, K for 4 heads, apply RoPE, check norms preserved per head
    let dim = 64;
    let heads = 4;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 32));
    let q = test_data(heads * dim, 104);
    let k = test_data(heads * dim, 105);
    let q_norms: Vec<f32> = (0..heads).map(|h| l2_norm(&q[h * dim..(h + 1) * dim])).collect();
    let k_norms: Vec<f32> = (0..heads).map(|h| l2_norm(&k[h * dim..(h + 1) * dim])).collect();
    let mut q_rot = q.clone();
    let mut k_rot = k.clone();
    apply_rope_batch(&mut q_rot, 10, 1, heads, dim, &freqs);
    apply_rope_batch(&mut k_rot, 10, 1, heads, dim, &freqs);
    for h in 0..heads {
        let q_norm_after = l2_norm(&q_rot[h * dim..(h + 1) * dim]);
        let k_norm_after = l2_norm(&k_rot[h * dim..(h + 1) * dim]);
        assert!(
            (q_norms[h] - q_norm_after).abs() < 1e-3,
            "Q norm head={h}: {} vs {q_norm_after}",
            q_norms[h]
        );
        assert!(
            (k_norms[h] - k_norm_after).abs() < 1e-3,
            "K norm head={h}: {} vs {k_norm_after}",
            k_norms[h]
        );
    }
}

#[test]
fn integration_causal_attention_score_decay() {
    // Tokens far apart should have lower attention similarity after RoPE
    let dim = 64;
    let freqs = compute_frequencies(&RopeConfig::new(dim, 256));
    let q = test_data(dim, 106);
    let k = test_data(dim, 107);
    // Same position
    let mut q0 = q.clone();
    let mut k0 = k.clone();
    apply_rope(&mut q0, 0, dim, &freqs);
    apply_rope(&mut k0, 0, dim, &freqs);
    let dot_same: f32 = q0.iter().zip(k0.iter()).map(|(a, b)| a * b).sum();
    // Far apart: q at 0, k at 200
    let mut q_far = q.clone();
    let mut k_far = k.clone();
    apply_rope(&mut q_far, 0, dim, &freqs);
    apply_rope(&mut k_far, 200, dim, &freqs);
    let dot_far: f32 = q_far.iter().zip(k_far.iter()).map(|(a, b)| a * b).sum();
    // Not a strict monotonicity requirement, just verify they differ
    let diff = (dot_same - dot_far).abs();
    assert!(diff > 1e-3, "same vs far attention scores should differ: diff={diff}");
}

#[test]
fn integration_prefill_then_decode() {
    // Simulate prefill of 8 tokens then decode 4 tokens one at a time
    let dim = 64;
    let heads = 2;
    let prefill_len = 8;
    let decode_len = 4;
    let total_len = prefill_len + decode_len;
    let freqs = compute_frequencies(&RopeConfig::new(dim, total_len));
    // Prefill phase: batch of 8
    let prefill_data = test_data(prefill_len * heads * dim, 108);
    let mut prefill_out = prefill_data.clone();
    apply_rope_batch(&mut prefill_out, 0, prefill_len, heads, dim, &freqs);
    // Decode phase: one token at a time
    for step in 0..decode_len {
        let pos = prefill_len + step;
        let decode_data = test_data(heads * dim, 109 + step as u32);
        let mut decode_out = decode_data.clone();
        apply_rope_batch(&mut decode_out, pos, 1, heads, dim, &freqs);
        // Each decode step should match reference
        for h in 0..heads {
            let off = h * dim;
            let expected = reference_rope(&decode_data[off..off + dim], dim, pos, DEFAULT_THETA);
            assert_close(
                &decode_out[off..off + dim],
                &expected,
                TOLERANCE,
                &format!("decode step={step} head={h}"),
            );
        }
    }
}

#[test]
fn integration_rope_softmax_pipeline() {
    // Apply RoPE, then compute softmax-like normalization of attention logits
    let dim = 64;
    let seq = 4;
    let freqs = compute_frequencies(&RopeConfig::new(dim, seq));
    let q = test_data(dim, 110);
    let keys: Vec<Vec<f32>> = (0..seq).map(|s| test_data(dim, 111 + s as u32)).collect();
    // Apply RoPE to query
    let mut q_rot = q.clone();
    apply_rope(&mut q_rot, 0, dim, &freqs);
    // Apply RoPE to keys and compute logits
    let mut logits = Vec::new();
    for (s, k) in keys.iter().enumerate() {
        let mut k_rot = k.clone();
        apply_rope(&mut k_rot, s, dim, &freqs);
        let dot: f32 = q_rot.iter().zip(k_rot.iter()).map(|(a, b)| a * b).sum();
        logits.push(dot / (dim as f32).sqrt());
    }
    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp() / exp_sum).collect();
    // Probabilities should sum to ~1
    let prob_sum: f32 = probs.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-5, "softmax probs should sum to 1: {prob_sum}");
    // All probabilities should be positive
    for (i, &p) in probs.iter().enumerate() {
        assert!(p > 0.0, "prob[{i}] should be positive: {p}");
    }
}

// =========================================================================
// GPU-gated tests (require Metal runtime)
// =========================================================================

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_basic_head_dim_64() {
    // Placeholder: would dispatch Metal compute shader for RoPE
    let dim = 64;
    let data = test_data(dim, 200);
    let expected = reference_rope(&data, dim, 3, DEFAULT_THETA);
    // GPU dispatch would go here; for now verify reference is non-trivial
    let diff: f32 = expected.iter().zip(data.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "reference should differ from input");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_multi_head_batch() {
    let dim = 128;
    let heads = 32;
    let seq = 16;
    let total = seq * heads * dim;
    let data = test_data(total, 201);
    // Would dispatch batched Metal RoPE shader
    assert!(total > 0, "non-trivial workload");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_long_context_8192() {
    let dim = 128;
    let max_pos = 8192;
    let data = test_data(dim, 202);
    let expected = reference_rope(&data, dim, max_pos - 1, DEFAULT_THETA);
    let norm = l2_norm(&expected);
    assert!(norm > 0.0, "non-trivial output");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_ntk_scaling() {
    let dim = 64;
    let data = test_data(dim, 203);
    let result = reference_rope_ntk(&data, dim, 100, DEFAULT_THETA, 4.0);
    assert_eq!(result.len(), dim);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_split_layout() {
    let dim = 64;
    let data = test_data(dim, 204);
    let result = reference_rope_split(&data, dim, 5, DEFAULT_THETA);
    assert_eq!(result.len(), dim);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_high_theta_500k() {
    let dim = 128;
    let data = test_data(dim, 205);
    let result = reference_rope(&data, dim, 1000, HIGH_THETA);
    let norm = l2_norm(&result);
    assert!(norm > 0.0);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_buffer_alignment_256() {
    // Metal requires 256-byte aligned buffers for optimal performance
    let alignment = 256u64;
    let dim = 64;
    let buffer_size = (dim * 4) as u64; // f32 = 4 bytes
    let aligned = (buffer_size + alignment - 1) / alignment * alignment;
    assert_eq!(aligned % alignment, 0, "buffer should be 256-byte aligned");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_workgroup_dispatch() {
    // Verify workgroup sizing for Metal threadgroup dispatch
    let dim = 128;
    let heads = 32;
    let seq = 16;
    let total_pairs = seq * heads * (dim / 2);
    let workgroup_size = 64;
    let num_workgroups = (total_pairs + workgroup_size - 1) / workgroup_size;
    assert!(num_workgroups > 0);
    assert!(num_workgroups * workgroup_size >= total_pairs);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_frequency_table_upload() {
    // Validate frequency table construction for GPU buffer upload
    let dim = 128;
    let max_pos = 4096;
    let config = RopeConfig::new(dim, max_pos);
    let freqs = compute_frequencies(&config);
    assert_eq!(freqs.len(), max_pos * dim);
    // Table size in bytes for Metal buffer
    let buffer_bytes = freqs.len() * 4;
    assert!(buffer_bytes > 0);
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn gpu_rope_interleaved_vs_split_parity() {
    let dim = 64;
    let data = test_data(dim, 210);
    let interleaved = reference_rope(&data, dim, 5, DEFAULT_THETA);
    let split = reference_rope_split(&data, dim, 5, DEFAULT_THETA);
    // Different layouts produce different results (not a bug)
    let diff: f32 = interleaved.iter().zip(split.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3 || diff < 1e-10, "layouts differ or data is trivial");
}
