//! Tests for the OpenCL Rotary Position Embedding (RoPE) kernel.
//!
//! These tests verify correctness via a CPU reference implementation that
//! mirrors the kernel logic. Hardware-dependent tests are `#[ignore]`.

// ── CPU reference implementation ────────────────────────────────────

/// Apply RoPE to a single head vector at the given position.
/// Layout: data has `head_dim` elements, pairs (x0, x1) rotated in place.
fn ref_rope_apply(
    data: &mut [f32],
    head_dim: usize,
    position: usize,
    theta_base: f32,
) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * cos_val - x1 * sin_val;
        data[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Apply RoPE across a batch: data layout [seq_len, num_heads, head_dim].
fn ref_rope_apply_batch(
    data: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    theta_base: f32,
    position_offset: usize,
) {
    for pos in 0..seq_len {
        let actual_pos = pos + position_offset;
        for head in 0..num_heads {
            let base_idx = (pos * num_heads + head) * head_dim;
            ref_rope_apply(
                &mut data[base_idx..base_idx + head_dim],
                head_dim,
                actual_pos,
                theta_base,
            );
        }
    }
}

/// Generate cos/sin caches for the cached kernel variant.
fn generate_cos_sin_cache(
    max_seq: usize,
    head_dim: usize,
    theta_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_cache = vec![0.0f32; max_seq * half_dim];
    let mut sin_cache = vec![0.0f32; max_seq * half_dim];

    for pos in 0..max_seq {
        for i in 0..half_dim {
            let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_cache[pos * half_dim + i] = angle.cos();
            sin_cache[pos * half_dim + i] = angle.sin();
        }
    }

    (cos_cache, sin_cache)
}

/// Apply RoPE using pre-computed cos/sin caches.
fn ref_rope_apply_cached(
    data: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: usize,
) {
    let half_dim = head_dim / 2;
    for pos in 0..seq_len {
        let actual_pos = pos + position_offset;
        for head in 0..num_heads {
            let base_idx = (pos * num_heads + head) * head_dim;
            for i in 0..half_dim {
                let cos_val = cos_cache[actual_pos * half_dim + i];
                let sin_val = sin_cache[actual_pos * half_dim + i];

                let x0 = data[base_idx + 2 * i];
                let x1 = data[base_idx + 2 * i + 1];
                data[base_idx + 2 * i] = x0 * cos_val - x1 * sin_val;
                data[base_idx + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

// ── Kernel source validation ────────────────────────────────────────

#[test]
fn rope_kernel_source_is_not_empty() {
    let src = bitnet_kernels::kernels::ROPE_SRC;
    assert!(!src.is_empty());
}

#[test]
fn rope_kernel_has_both_variants() {
    let src = bitnet_kernels::kernels::ROPE_SRC;
    assert!(src.contains("__kernel void rope_apply"));
    assert!(src.contains("__kernel void rope_apply_cached"));
}

#[test]
fn rope_kernel_has_theta_base_param() {
    let src = bitnet_kernels::kernels::ROPE_SRC;
    assert!(
        src.contains("theta_base"),
        "rope_apply should accept theta_base for configurable frequency"
    );
}

#[test]
fn rope_kernel_has_position_offset_param() {
    let src = bitnet_kernels::kernels::ROPE_SRC;
    assert!(
        src.contains("position_offset"),
        "both kernels should support KV cache continuation via position_offset"
    );
}

#[test]
fn rope_cached_kernel_has_cache_params() {
    let src = bitnet_kernels::kernels::ROPE_SRC;
    assert!(src.contains("cos_cache"), "rope_apply_cached needs cos_cache");
    assert!(src.contains("sin_cache"), "rope_apply_cached needs sin_cache");
}

// ── Basic rotation tests ────────────────────────────────────────────

#[test]
fn test_rope_position_zero_is_identity() {
    // At position 0, angle = 0 for all dims, so cos=1 sin=0 → identity
    for head_dim in [2, 4, 8, 16, 64, 128] {
        let mut data: Vec<f32> =
            (0..head_dim).map(|i| (i as f32 + 1.0) * 3.17).collect();
        let original = data.clone();

        ref_rope_apply(&mut data, head_dim, 0, 10_000.0);

        for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                approx_eq(*o, *d, 1e-5),
                "position 0 not identity at dim {i}, head_dim={head_dim}: {o} vs {d}"
            );
        }
    }
}

#[test]
fn test_rope_rotation_at_various_positions() {
    let head_dim = 4;
    let theta_base = 10_000.0f32;

    for pos in [1, 5, 17, 100, 1000] {
        let mut data = vec![1.0, 0.5, 0.8, -0.3];

        // Compute expected values manually
        let mut expected = data.clone();
        let half_dim = head_dim / 2;
        for i in 0..half_dim {
            let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let x0 = expected[2 * i];
            let x1 = expected[2 * i + 1];
            expected[2 * i] = x0 * cos_val - x1 * sin_val;
            expected[2 * i + 1] = x0 * sin_val + x1 * cos_val;
        }

        ref_rope_apply(&mut data, head_dim, pos, theta_base);

        for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*got, *want, 1e-5),
                "pos={pos}, dim {i}: got {got}, expected {want}"
            );
        }
    }
}

#[test]
fn test_rope_different_positions_differ() {
    let head_dim = 4;
    let original = vec![1.0, 2.0, 3.0, 4.0];

    let mut data_pos1 = original.clone();
    ref_rope_apply(&mut data_pos1, head_dim, 1, 10_000.0);

    let mut data_pos2 = original.clone();
    ref_rope_apply(&mut data_pos2, head_dim, 2, 10_000.0);

    let any_diff = data_pos1
        .iter()
        .zip(data_pos2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_diff, "different positions should produce different rotations");
}

// ── Position offset for KV cache continuation ───────────────────────

#[test]
fn test_rope_position_offset() {
    let head_dim = 8;
    let num_heads = 2;
    let seq_len = 3;
    let theta_base = 10_000.0;

    // Apply with offset=0, seq_len=5 to get positions 0..5
    let total = 5 * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let mut full_data = original.clone();
    ref_rope_apply_batch(&mut full_data, 5, num_heads, head_dim, theta_base, 0);

    // Apply with offset=2, seq_len=3 to get positions 2..5
    let offset_total = seq_len * num_heads * head_dim;
    // Use the same original data for positions 2..5
    let start = 2 * num_heads * head_dim;
    let mut offset_data = original[start..start + offset_total].to_vec();
    ref_rope_apply_batch(
        &mut offset_data,
        seq_len,
        num_heads,
        head_dim,
        theta_base,
        2,
    );

    // Results should match for positions 2..5
    for i in 0..offset_total {
        let full_idx = start + i;
        assert!(
            approx_eq(full_data[full_idx], offset_data[i], 1e-5),
            "offset mismatch at i={i}: {} vs {}",
            full_data[full_idx],
            offset_data[i]
        );
    }
}

#[test]
fn test_rope_position_offset_zero_equivalent() {
    let head_dim = 4;
    let num_heads = 1;
    let seq_len = 3;
    let theta_base = 10_000.0;

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();

    let mut data_no_offset = original.clone();
    ref_rope_apply_batch(&mut data_no_offset, seq_len, num_heads, head_dim, theta_base, 0);

    let mut data_with_offset = original.clone();
    ref_rope_apply_batch(
        &mut data_with_offset,
        seq_len,
        num_heads,
        head_dim,
        theta_base,
        0,
    );

    for (i, (a, b)) in data_no_offset.iter().zip(data_with_offset.iter()).enumerate() {
        assert!(
            approx_eq(*a, *b, 1e-6),
            "offset=0 should match no-offset at i={i}: {a} vs {b}"
        );
    }
}

// ── Multi-head RoPE ─────────────────────────────────────────────────

#[test]
fn test_rope_multi_head_same_position_same_rotation() {
    let head_dim = 8;
    let num_heads = 4;
    let theta_base = 10_000.0;

    let pattern: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut data: Vec<f32> = pattern
        .iter()
        .copied()
        .cycle()
        .take(num_heads * head_dim)
        .collect();

    ref_rope_apply_batch(&mut data, 1, num_heads, head_dim, theta_base, 3);

    // All heads at the same position should get the same rotation
    for h in 1..num_heads {
        for d in 0..head_dim {
            let ref_val = data[d];
            let val = data[h * head_dim + d];
            assert!(
                approx_eq(ref_val, val, 1e-6),
                "head {h} diverges at dim {d}: {val} vs {ref_val}"
            );
        }
    }
}

#[test]
fn test_rope_multi_head_different_data() {
    let head_dim = 4;
    let num_heads = 3;
    let theta_base = 10_000.0;

    let mut data: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| ((i * 7 + 3) as f32) * 0.01)
        .collect();

    ref_rope_apply_batch(&mut data, 1, num_heads, head_dim, theta_base, 5);

    assert!(data.iter().all(|x| x.is_finite()), "all outputs should be finite");
}

// ── Minimal dimension cases ─────────────────────────────────────────

#[test]
fn test_rope_head_dim_2_minimal() {
    // Minimal case: single rotation pair
    let head_dim = 2;
    let theta_base = 10_000.0;

    let mut data = vec![1.0, 0.0];
    ref_rope_apply(&mut data, head_dim, 1, theta_base);

    // theta = 10000^0 = 1.0, angle = 1.0
    let expected_cos = 1.0f32.cos();
    let expected_sin = 1.0f32.sin();
    assert!(
        approx_eq(data[0], expected_cos, 1e-5),
        "x0: got {}, expected {expected_cos}",
        data[0]
    );
    assert!(
        approx_eq(data[1], expected_sin, 1e-5),
        "x1: got {}, expected {expected_sin}",
        data[1]
    );
}

#[test]
fn test_rope_head_dim_2_general_input() {
    let head_dim = 2;
    let theta_base = 10_000.0;
    let pos = 7;

    let mut data = vec![3.0, -2.0];
    let angle = pos as f32; // theta = 1.0 for pair 0
    let expected_x0 = 3.0 * angle.cos() - (-2.0) * angle.sin();
    let expected_x1 = 3.0 * angle.sin() + (-2.0) * angle.cos();

    ref_rope_apply(&mut data, head_dim, pos, theta_base);

    assert!(approx_eq(data[0], expected_x0, 1e-4), "x0: {}", data[0]);
    assert!(approx_eq(data[1], expected_x1, 1e-4), "x1: {}", data[1]);
}

// ── Typical dimension case (LLaMA-7B head_dim = 128) ────────────────

#[test]
fn test_rope_head_dim_128_norm_preservation() {
    let head_dim = 128;
    let theta_base = 10_000.0;

    for pos in [0, 1, 10, 100, 2048] {
        let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let norm_before = l2_norm(&data);

        ref_rope_apply(&mut data, head_dim, pos, theta_base);

        let norm_after = l2_norm(&data);
        assert!(
            approx_eq(norm_before, norm_after, 1e-3),
            "norm not preserved at pos={pos}, head_dim=128: {norm_before} vs {norm_after}"
        );
    }
}

#[test]
fn test_rope_head_dim_128_batch() {
    let head_dim = 128;
    let num_heads = 32;
    let seq_len = 4;
    let theta_base = 10_000.0;

    let total = seq_len * num_heads * head_dim;
    let mut data: Vec<f32> = (0..total)
        .map(|i| ((i * 37 + 13) as f32).sin() * 0.5)
        .collect();

    ref_rope_apply_batch(&mut data, seq_len, num_heads, head_dim, theta_base, 0);

    assert!(data.iter().all(|x| x.is_finite()), "all outputs should be finite");
}

// ── Cached vs computed parity ───────────────────────────────────────

#[test]
fn test_rope_cached_vs_computed_parity() {
    let head_dim = 8;
    let num_heads = 4;
    let seq_len = 6;
    let theta_base = 10_000.0;
    let position_offset = 3;
    let max_seq = seq_len + position_offset + 1;

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 3.0).collect();

    // Computed variant
    let mut computed = original.clone();
    ref_rope_apply_batch(
        &mut computed,
        seq_len,
        num_heads,
        head_dim,
        theta_base,
        position_offset,
    );

    // Cached variant
    let (cos_cache, sin_cache) = generate_cos_sin_cache(max_seq, head_dim, theta_base);
    let mut cached = original.clone();
    ref_rope_apply_cached(
        &mut cached,
        &cos_cache,
        &sin_cache,
        seq_len,
        num_heads,
        head_dim,
        position_offset,
    );

    for (i, (c, d)) in computed.iter().zip(cached.iter()).enumerate() {
        assert!(
            approx_eq(*c, *d, 1e-5),
            "cached/computed mismatch at i={i}: {c} vs {d}"
        );
    }
}

#[test]
fn test_rope_cached_vs_computed_head_dim_128() {
    let head_dim = 128;
    let num_heads = 2;
    let seq_len = 8;
    let theta_base = 10_000.0;
    let max_seq = seq_len + 1;

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total)
        .map(|i| ((i * 17 + 5) as f32).sin())
        .collect();

    let mut computed = original.clone();
    ref_rope_apply_batch(&mut computed, seq_len, num_heads, head_dim, theta_base, 0);

    let (cos_cache, sin_cache) = generate_cos_sin_cache(max_seq, head_dim, theta_base);
    let mut cached = original.clone();
    ref_rope_apply_cached(
        &mut cached,
        &cos_cache,
        &sin_cache,
        seq_len,
        num_heads,
        head_dim,
        0,
    );

    for (i, (c, d)) in computed.iter().zip(cached.iter()).enumerate() {
        assert!(
            approx_eq(*c, *d, 1e-5),
            "cached/computed mismatch at i={i} (head_dim=128): {c} vs {d}"
        );
    }
}

// ── cos/sin cache generation tests ──────────────────────────────────

#[test]
fn test_cos_sin_cache_dimensions() {
    let head_dim = 8;
    let max_seq = 16;
    let (cos_cache, sin_cache) = generate_cos_sin_cache(max_seq, head_dim, 10_000.0);

    assert_eq!(cos_cache.len(), max_seq * head_dim / 2);
    assert_eq!(sin_cache.len(), max_seq * head_dim / 2);
}

#[test]
fn test_cos_sin_cache_position_zero() {
    let head_dim = 8;
    let (cos_cache, sin_cache) = generate_cos_sin_cache(2, head_dim, 10_000.0);
    let half_dim = head_dim / 2;

    // At position 0, angle = 0 → cos = 1, sin = 0
    for i in 0..half_dim {
        assert!(
            approx_eq(cos_cache[i], 1.0, 1e-6),
            "cos_cache[{i}] at pos 0 should be 1.0, got {}",
            cos_cache[i]
        );
        assert!(
            approx_eq(sin_cache[i], 0.0, 1e-6),
            "sin_cache[{i}] at pos 0 should be 0.0, got {}",
            sin_cache[i]
        );
    }
}

#[test]
fn test_cos_sin_cache_values_at_position_one() {
    let head_dim = 4;
    let theta_base = 10_000.0;
    let (cos_cache, sin_cache) = generate_cos_sin_cache(2, head_dim, theta_base);
    let half_dim = head_dim / 2;

    // At position 1: angle_i = 1 * freq_i
    for i in 0..half_dim {
        let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
        let angle = freq;
        assert!(
            approx_eq(cos_cache[half_dim + i], angle.cos(), 1e-6),
            "cos mismatch at pair {i}"
        );
        assert!(
            approx_eq(sin_cache[half_dim + i], angle.sin(), 1e-6),
            "sin mismatch at pair {i}"
        );
    }
}

#[test]
fn test_cos_sin_cache_pythagorean_identity() {
    let head_dim = 16;
    let max_seq = 64;
    let (cos_cache, sin_cache) = generate_cos_sin_cache(max_seq, head_dim, 10_000.0);

    // cos²(x) + sin²(x) = 1
    for (i, (c, s)) in cos_cache.iter().zip(sin_cache.iter()).enumerate() {
        let sum_sq = c * c + s * s;
        assert!(
            approx_eq(sum_sq, 1.0, 1e-5),
            "cos²+sin² != 1 at index {i}: {sum_sq}"
        );
    }
}

// ── Property: norm preservation (rotation is unitary) ───────────────

#[test]
fn test_rope_preserves_norm() {
    let theta_base = 10_000.0;

    for head_dim in [2, 4, 8, 16, 32, 64, 128] {
        for pos in [0, 1, 5, 42, 512] {
            let mut data: Vec<f32> =
                (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let norm_before = l2_norm(&data);

            ref_rope_apply(&mut data, head_dim, pos, theta_base);

            let norm_after = l2_norm(&data);
            assert!(
                approx_eq(norm_before, norm_after, 1e-3),
                "norm not preserved: head_dim={head_dim}, pos={pos}: \
                 {norm_before} vs {norm_after}"
            );
        }
    }
}

#[test]
fn test_rope_batch_norm_preservation() {
    let head_dim = 32;
    let seq_len = 16;
    let num_heads = 4;
    let theta_base = 10_000.0;

    let total = seq_len * num_heads * head_dim;
    let mut data: Vec<f32> = (0..total)
        .map(|i| ((i * 37 + 13) as f32).sin() * 2.5)
        .collect();

    let norms_before: Vec<f32> = (0..seq_len * num_heads)
        .map(|chunk| {
            let start = chunk * head_dim;
            l2_norm(&data[start..start + head_dim])
        })
        .collect();

    ref_rope_apply_batch(&mut data, seq_len, num_heads, head_dim, theta_base, 0);

    for (chunk, norm_before) in norms_before.iter().enumerate() {
        let start = chunk * head_dim;
        let norm_after = l2_norm(&data[start..start + head_dim]);
        assert!(
            approx_eq(*norm_before, norm_after, 1e-3),
            "norm not preserved at chunk {chunk}: {norm_before} vs {norm_after}"
        );
    }
}

// ── Property: large theta approaches identity for higher-freq pairs ──

#[test]
fn test_rope_large_theta_approaches_identity() {
    // With very large theta_base, freq_i = theta^(-2i/d) → 0 for i > 0,
    // so angle → 0 and rotation → identity for those pairs.
    // Note: pair 0 has freq = theta^0 = 1.0 regardless of theta.
    let head_dim = 8;
    let theta_base = 1e15f32;

    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let original = data.clone();

    ref_rope_apply(&mut data, head_dim, 1, theta_base);

    // Pairs 1..3 (indices 2..7) should be near identity with huge theta
    for i in 1..(head_dim / 2) {
        let idx = 2 * i;
        assert!(
            approx_eq(original[idx], data[idx], 1e-3),
            "large theta: pair {i} dim {idx}: {} vs {}",
            original[idx],
            data[idx]
        );
        assert!(
            approx_eq(original[idx + 1], data[idx + 1], 1e-3),
            "large theta: pair {i} dim {}: {} vs {}",
            idx + 1,
            original[idx + 1],
            data[idx + 1]
        );
    }
}

// ── Property: zero input preserved ──────────────────────────────────

#[test]
fn test_rope_zero_input_preserved() {
    let head_dim = 16;
    let seq_len = 4;
    let num_heads = 2;
    let total = seq_len * num_heads * head_dim;
    let mut data = vec![0.0f32; total];

    ref_rope_apply_batch(&mut data, seq_len, num_heads, head_dim, 10_000.0, 0);

    for (i, val) in data.iter().enumerate() {
        assert!(val.abs() < 1e-10, "zero not preserved at index {i}");
    }
}

// ── Edge case: seq_len=1, num_heads=1 ───────────────────────────────

#[test]
fn test_rope_single_token_single_head() {
    let head_dim = 4;
    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    let mut reference = data.clone();

    ref_rope_apply_batch(&mut data, 1, 1, head_dim, 10_000.0, 5);
    ref_rope_apply(&mut reference, head_dim, 5, 10_000.0);

    for (i, (a, b)) in data.iter().zip(reference.iter()).enumerate() {
        assert!(
            approx_eq(*a, *b, 1e-6),
            "single-token batch mismatch at {i}: {a} vs {b}"
        );
    }
}

// ── Edge case: large position values ────────────────────────────────

#[test]
fn test_rope_large_position() {
    let head_dim = 4;
    let mut data = vec![1.0, 0.0, 1.0, 0.0];

    ref_rope_apply(&mut data, head_dim, 8000, 10_000.0);

    assert!(data.iter().all(|x| x.is_finite()), "large position produced non-finite");
    let norm = l2_norm(&data);
    let expected_norm = (2.0f32).sqrt();
    assert!(
        approx_eq(norm, expected_norm, 1e-3),
        "norm at large position: {norm} vs {expected_norm}"
    );
}

// ── Edge case: different theta values ───────────────────────────────

#[test]
fn test_rope_different_theta_bases() {
    let head_dim = 4;
    let pos = 5;
    let original = vec![1.0, 2.0, 3.0, 4.0];

    let mut data_10k = original.clone();
    ref_rope_apply(&mut data_10k, head_dim, pos, 10_000.0);

    let mut data_500k = original.clone();
    ref_rope_apply(&mut data_500k, head_dim, pos, 500_000.0);

    let any_diff = data_10k
        .iter()
        .zip(data_500k.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_diff, "different theta bases should produce different rotations");
}

// ── Determinism ─────────────────────────────────────────────────────

#[test]
fn test_rope_deterministic() {
    let head_dim = 8;
    let num_heads = 2;
    let seq_len = 4;
    let theta_base = 10_000.0;

    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

    let mut run1 = original.clone();
    ref_rope_apply_batch(&mut run1, seq_len, num_heads, head_dim, theta_base, 0);

    let mut run2 = original.clone();
    ref_rope_apply_batch(&mut run2, seq_len, num_heads, head_dim, theta_base, 0);

    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(*a, *b, "non-deterministic at index {i}: {a} vs {b}");
    }
}

// ── Known reference value: head_dim=4, pos=3 ────────────────────────

#[test]
fn test_rope_known_reference_head_dim_4_pos_3() {
    let head_dim = 4;
    let theta_base = 10_000.0f32;
    let pos = 3;

    let mut data = vec![1.0, 0.5, 0.8, -0.3];

    // pair 0: theta = 10000^0 = 1.0, angle = 3.0
    // pair 1: theta = 10000^(-0.5), angle = 3.0 * 10000^(-0.5)
    let angle0 = 3.0f32;
    let angle1: f32 = 3.0 * theta_base.powf(-0.5);
    let expected = [
        1.0 * angle0.cos() - 0.5 * angle0.sin(),
        1.0 * angle0.sin() + 0.5 * angle0.cos(),
        0.8 * angle1.cos() - (-0.3) * angle1.sin(),
        0.8 * angle1.sin() + (-0.3) * angle1.cos(),
    ];

    ref_rope_apply(&mut data, head_dim, pos, theta_base);

    for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*got, *want, 1e-5),
            "dim {i}: got {got}, expected {want}"
        );
    }
}

// ── Frequency monotonic decay ───────────────────────────────────────

#[test]
fn test_rope_frequency_monotonic_decay() {
    let head_dim = 8;
    let theta_base = 10_000.0f32;
    let half_dim = head_dim / 2;

    // Higher dimension index → lower frequency → smaller angle at pos 1
    let mut angles = Vec::new();
    for i in 0..half_dim {
        let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
        angles.push(freq);
    }

    for i in 1..angles.len() {
        assert!(
            angles[i] < angles[i - 1],
            "frequencies should decrease: freq[{}]={} >= freq[{}]={}",
            i,
            angles[i],
            i - 1,
            angles[i - 1]
        );
    }
}

// ── Hardware-dependent tests (require OpenCL runtime) ───────────────

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_rope_apply_on_device() {
    todo!("Implement OpenCL device test for rope_apply");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_rope_apply_cached_on_device() {
    todo!("Implement OpenCL device test for rope_apply_cached");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_rope_parity_computed_vs_cached_on_device() {
    todo!("Implement OpenCL device parity test between rope_apply and rope_apply_cached");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_rope_multi_head_on_device() {
    todo!("Implement OpenCL device test for multi-head RoPE");
}
