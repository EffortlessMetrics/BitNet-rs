//! CPU Rotary Position Embedding (RoPE) kernel with SIMD acceleration.
//!
//! Encodes absolute position information into query/key vectors by rotating
//! pairs of dimensions using sinusoidal functions:
//!
//!   For each pair `(x[2i], x[2i+1])` at position `pos`:
//!     `theta_i = base^(-2i / head_dim) * scaling_factor`
//!     `angle   = pos * theta_i`
//!     `y[2i]   = x[2i]   * cos(angle) - x[2i+1] * sin(angle)`
//!     `y[2i+1] = x[2i]   * sin(angle) + x[2i+1] * cos(angle)`
//!
//! The default rotation base is `10_000.0` (following the original RoPE paper).
//!
//! The implementation precomputes sin/cos frequency tables for all
//! `(position, dim_pair)` combinations, then applies the rotation in a
//! single pass — optionally accelerated with AVX2 SIMD intrinsics.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for RoPE frequency computation and application.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Per-head embedding dimension (must be even).
    pub head_dim: usize,
    /// Maximum sequence length the frequency table covers.
    pub max_seq_len: usize,
    /// Rotation base frequency (default `10_000.0`).
    pub base: f32,
    /// Scaling factor applied to frequencies (default `1.0`).
    pub scaling_factor: f32,
}

impl RopeConfig {
    /// Create a new `RopeConfig` with sensible defaults.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is zero or odd.
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "head_dim must be even and non-zero");
        Self { head_dim, max_seq_len, base: 10_000.0, scaling_factor: 1.0 }
    }

    /// Override the rotation base frequency.
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Override the frequency scaling factor.
    #[must_use]
    pub fn with_scaling_factor(mut self, factor: f32) -> Self {
        self.scaling_factor = factor;
        self
    }
}

// ── Frequency table ─────────────────────────────────────────────────

/// Precompute interleaved `[cos, sin, cos, sin, …]` table.
///
/// Layout: for position `p` and dimension pair `i`, the cos value is at
/// `frequencies[(p * half_dim + i) * 2]` and the sin value at
/// `frequencies[(p * half_dim + i) * 2 + 1]`.
///
/// Total length: `max_seq_len * head_dim` (= `max_seq_len * half_dim * 2`).
pub fn compute_frequencies(config: &RopeConfig) -> Vec<f32> {
    let half_dim = config.head_dim / 2;
    let mut freqs = Vec::with_capacity(config.max_seq_len * config.head_dim);

    for pos in 0..config.max_seq_len {
        for i in 0..half_dim {
            let exponent = -(2.0 * i as f32) / config.head_dim as f32;
            let theta = config.base.powf(exponent) * config.scaling_factor;
            let angle = pos as f32 * theta;
            freqs.push(angle.cos());
            freqs.push(angle.sin());
        }
    }

    freqs
}

// ── Scalar implementation (all platforms) ────────────────────────────

/// Apply RoPE rotation to a single position's head vector **in-place**.
///
/// `data` must have length ≥ `head_dim`, and `frequencies` must contain
/// the interleaved cos/sin values for the given `position` (i.e. the
/// slice starting at `position * head_dim` with length `head_dim`).
pub fn apply_rope(data: &mut [f32], position: usize, head_dim: usize, frequencies: &[f32]) {
    let half_dim = head_dim / 2;
    let freq_offset = position * head_dim;

    for i in 0..half_dim {
        let cos_val = frequencies[freq_offset + 2 * i];
        let sin_val = frequencies[freq_offset + 2 * i + 1];

        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];

        data[2 * i] = x0 * cos_val - x1 * sin_val;
        data[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Apply RoPE across a batch of positions and heads **in-place**.
///
/// `data` layout: `[seq_len, num_heads, head_dim]` — each contiguous
/// `head_dim` block gets the rotation for its position.
pub fn apply_rope_batch(
    data: &mut [f32],
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    frequencies: &[f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && head_dim >= 8 {
            // Safety: AVX2 confirmed available by runtime check.
            unsafe {
                apply_rope_batch_avx2(data, start_pos, seq_len, num_heads, head_dim, frequencies);
            }
            return;
        }
    }
    apply_rope_batch_scalar(data, start_pos, seq_len, num_heads, head_dim, frequencies);
}

/// Scalar batch RoPE — always available on every platform.
fn apply_rope_batch_scalar(
    data: &mut [f32],
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    frequencies: &[f32],
) {
    for s in 0..seq_len {
        let position = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            apply_rope(&mut data[offset..offset + head_dim], position, head_dim, frequencies);
        }
    }
}

// ── AVX2 implementation (x86_64 only) ───────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rope_batch_avx2(
    data: &mut [f32],
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    frequencies: &[f32],
) {
    for s in 0..seq_len {
        let position = start_pos + s;
        let freq_offset = position * head_dim;

        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let head_data = &mut data[offset..offset + head_dim];

            // Process 8 floats (4 rotation pairs) per AVX2 iteration.
            let chunks = head_dim / 8;
            for c in 0..chunks {
                let base = c * 8;

                unsafe {
                    // Load 4 interleaved (cos, sin) pairs from frequency table.
                    let freq = _mm256_loadu_ps(frequencies.as_ptr().add(freq_offset + base));
                    // freq = [c0, s0, c1, s1, c2, s2, c3, s3]

                    let vals = _mm256_loadu_ps(head_data.as_ptr().add(base));
                    // vals = [x0, x1, x2, x3, x4, x5, x6, x7]

                    // Deinterleave cos/sin: even lanes = cos, odd lanes = sin.
                    // cos_vals = [c0, c0, c1, c1, c2, c2, c3, c3]
                    // sin_vals = [s0, s0, s1, s1, s2, s2, s3, s3]
                    let cos_vals =
                        _mm256_permutevar8x32_ps(freq, _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6));
                    let sin_vals =
                        _mm256_permutevar8x32_ps(freq, _mm256_setr_epi32(1, 1, 3, 3, 5, 5, 7, 7));

                    // Swapped pairs: [x1, x0, x3, x2, x5, x4, x7, x6]
                    let swapped =
                        _mm256_permutevar8x32_ps(vals, _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6));

                    // Sign mask: [-1, +1, -1, +1, -1, +1, -1, +1]
                    let sign = _mm256_setr_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

                    // result = vals * cos + swapped * sign * sin
                    //   even lanes: x0*cos - x1*sin
                    //   odd  lanes: x1*cos + x0*sin
                    let rotated = _mm256_add_ps(
                        _mm256_mul_ps(vals, cos_vals),
                        _mm256_mul_ps(_mm256_mul_ps(swapped, sign), sin_vals),
                    );

                    _mm256_storeu_ps(head_data.as_mut_ptr().add(base), rotated);
                }
            }

            // Scalar tail for remaining pairs not covered by AVX2 chunks.
            let processed = chunks * 8;
            let half_remaining = (head_dim - processed) / 2;
            for i in 0..half_remaining {
                let idx = processed + 2 * i;
                let cos_val = frequencies[freq_offset + idx];
                let sin_val = frequencies[freq_offset + idx + 1];

                let x0 = head_data[idx];
                let x1 = head_data[idx + 1];

                head_data[idx] = x0 * cos_val - x1 * sin_val;
                head_data[idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Frequency computation ───────────────────────────────────────

    #[test]
    fn test_frequency_table_length() {
        let cfg = RopeConfig::new(8, 16);
        let freqs = compute_frequencies(&cfg);
        assert_eq!(freqs.len(), 16 * 8); // max_seq_len * head_dim
    }

    #[test]
    fn test_frequency_values_position_zero() {
        let cfg = RopeConfig::new(4, 2);
        let freqs = compute_frequencies(&cfg);
        // At position 0 all angles are 0 → cos=1, sin=0
        assert!((freqs[0] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(freqs[1].abs() < 1e-6, "sin(0) should be 0");
        assert!((freqs[2] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(freqs[3].abs() < 1e-6, "sin(0) should be 0");
    }

    #[test]
    fn test_frequency_monotonic_decay() {
        // Inverse frequencies should decrease with dimension index,
        // so at position 1 the angle for pair 0 should be larger than pair 1.
        let cfg = RopeConfig::new(8, 2);
        let freqs = compute_frequencies(&cfg);
        let half_dim = 4;
        // Position 1 starts at index head_dim = 8
        let pos1_base = cfg.head_dim;
        let angle_0 = freqs[pos1_base + 1]; // sin for pair 0
        let angle_last = freqs[pos1_base + (half_dim - 1) * 2 + 1]; // sin for last pair
        // Higher-frequency pair should have larger sin at position 1
        assert!(
            angle_0.abs() > angle_last.abs(),
            "pair 0 angle should be larger: {angle_0} vs {angle_last}"
        );
    }

    #[test]
    fn test_frequency_custom_base() {
        let cfg_default = RopeConfig::new(4, 2);
        let cfg_custom = RopeConfig::new(4, 2).with_base(500_000.0);
        let f1 = compute_frequencies(&cfg_default);
        let f2 = compute_frequencies(&cfg_custom);
        // Position 0 identical (angle = 0)
        assert!((f1[0] - f2[0]).abs() < 1e-6);
        // Position 1 should differ
        let any_diff = (0..4).any(|i| (f1[4 + i] - f2[4 + i]).abs() > 1e-6);
        assert!(any_diff, "different base should produce different frequencies");
    }

    #[test]
    fn test_frequency_scaling_factor() {
        let cfg1 = RopeConfig::new(4, 4);
        let cfg2 = RopeConfig::new(4, 4).with_scaling_factor(2.0);
        let f1 = compute_frequencies(&cfg1);
        let f2 = compute_frequencies(&cfg2);
        // At position 1, scaled frequencies should differ
        let pos1 = cfg1.head_dim;
        let any_diff = (0..cfg1.head_dim).any(|i| (f1[pos1 + i] - f2[pos1 + i]).abs() > 1e-6);
        assert!(any_diff, "scaling_factor should change frequencies");
    }

    // ── Single-position apply_rope ──────────────────────────────────

    #[test]
    fn test_apply_rope_identity_at_position_zero() {
        let cfg = RopeConfig::new(4, 2);
        let freqs = compute_frequencies(&cfg);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();

        apply_rope(&mut data, 0, 4, &freqs);

        for (o, d) in original.iter().zip(data.iter()) {
            assert!((o - d).abs() < 1e-6, "position 0 should be identity: {o} vs {d}");
        }
    }

    #[test]
    fn test_apply_rope_preserves_norm() {
        let cfg = RopeConfig::new(8, 32);
        let freqs = compute_frequencies(&cfg);

        for pos in [0, 1, 5, 17, 31] {
            let mut data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

            apply_rope(&mut data, pos, 8, &freqs);

            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "norm not preserved at pos={pos}: {norm_before} vs {norm_after}"
            );
        }
    }

    #[test]
    fn test_apply_rope_different_positions_differ() {
        let cfg = RopeConfig::new(4, 4);
        let freqs = compute_frequencies(&cfg);
        let original = vec![1.0, 2.0, 3.0, 4.0];

        let mut data_pos1 = original.clone();
        apply_rope(&mut data_pos1, 1, 4, &freqs);

        let mut data_pos2 = original.clone();
        apply_rope(&mut data_pos2, 2, 4, &freqs);

        let any_diff = data_pos1.iter().zip(data_pos2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "different positions should produce different rotations");
    }

    #[test]
    fn test_apply_rope_known_reference_head_dim_2() {
        // head_dim=2, position=1, base=10000
        // theta_0 = 10000^0 = 1.0, angle = 1.0
        // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
        let cfg = RopeConfig::new(2, 2);
        let freqs = compute_frequencies(&cfg);
        let mut data = vec![1.0, 0.0];

        apply_rope(&mut data, 1, 2, &freqs);

        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!(
            (data[0] - expected_cos).abs() < 1e-5,
            "x0: got {}, expected {expected_cos}",
            data[0]
        );
        assert!(
            (data[1] - expected_sin).abs() < 1e-5,
            "x1: got {}, expected {expected_sin}",
            data[1]
        );
    }

    #[test]
    fn test_apply_rope_known_reference_general() {
        // head_dim=4, position=3, base=10000
        // pair 0: theta = 10000^(0/4) = 1.0, angle = 3.0
        // pair 1: theta = 10000^(-2/4) = 10000^(-0.5), angle = 3 * 10000^(-0.5)
        let cfg = RopeConfig::new(4, 4);
        let freqs = compute_frequencies(&cfg);
        let mut data = vec![1.0, 0.5, 0.8, -0.3];

        let angle0 = 3.0f32;
        let angle1 = 3.0 * 10_000.0f32.powf(-0.5);
        let expected = [
            1.0 * angle0.cos() - 0.5 * angle0.sin(),
            1.0 * angle0.sin() + 0.5 * angle0.cos(),
            0.8 * angle1.cos() - (-0.3) * angle1.sin(),
            0.8 * angle1.sin() + (-0.3) * angle1.cos(),
        ];

        apply_rope(&mut data, 3, 4, &freqs);

        for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5, "dim {i}: got {got}, expected {want}");
        }
    }

    // ── Batch application ───────────────────────────────────────────

    #[test]
    fn test_batch_vs_single_parity() {
        let head_dim = 8;
        let num_heads = 4;
        let seq_len = 6;
        let start_pos = 2;
        let cfg = RopeConfig::new(head_dim, start_pos + seq_len + 1);
        let freqs = compute_frequencies(&cfg);

        let total = seq_len * num_heads * head_dim;
        let original: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 5.0).collect();

        // Batch version
        let mut batch_data = original.clone();
        apply_rope_batch(&mut batch_data, start_pos, seq_len, num_heads, head_dim, &freqs);

        // Single version applied per-head, per-position
        let mut single_data = original.clone();
        for s in 0..seq_len {
            let position = start_pos + s;
            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;
                apply_rope(&mut single_data[offset..offset + head_dim], position, head_dim, &freqs);
            }
        }

        for (i, (b, s)) in batch_data.iter().zip(single_data.iter()).enumerate() {
            assert!((b - s).abs() < 1e-5, "batch/single mismatch at index {i}: {b} vs {s}");
        }
    }

    #[test]
    fn test_batch_start_pos_zero() {
        let head_dim = 4;
        let num_heads = 2;
        let seq_len = 3;
        let cfg = RopeConfig::new(head_dim, seq_len);
        let freqs = compute_frequencies(&cfg);

        let total = seq_len * num_heads * head_dim;
        let mut data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.5).collect();

        apply_rope_batch(&mut data, 0, seq_len, num_heads, head_dim, &freqs);

        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batch_single_element() {
        let head_dim = 4;
        let cfg = RopeConfig::new(head_dim, 2);
        let freqs = compute_frequencies(&cfg);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let mut reference = data.clone();

        apply_rope_batch(&mut data, 1, 1, 1, head_dim, &freqs);
        apply_rope(&mut reference, 1, head_dim, &freqs);

        for (d, r) in data.iter().zip(reference.iter()) {
            assert!((d - r).abs() < 1e-6, "single-element batch mismatch: {d} vs {r}");
        }
    }

    // ── Head dimension handling ──────────────────────────────────────

    #[test]
    fn test_various_head_dims() {
        for head_dim in [2, 4, 8, 16, 32, 64, 128] {
            let cfg = RopeConfig::new(head_dim, 8);
            let freqs = compute_frequencies(&cfg);

            let mut data: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

            apply_rope(&mut data, 3, head_dim, &freqs);

            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "norm not preserved for head_dim={head_dim}"
            );
            assert!(data.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_head_dim_not_multiple_of_8() {
        // head_dim=6 forces the scalar tail in AVX2 path
        let head_dim = 6;
        let cfg = RopeConfig::new(head_dim, 4);
        let freqs = compute_frequencies(&cfg);

        let mut batch = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut single = batch.clone();

        apply_rope_batch(&mut batch, 2, 1, 1, head_dim, &freqs);
        apply_rope(&mut single, 2, head_dim, &freqs);

        for (b, s) in batch.iter().zip(single.iter()) {
            assert!((b - s).abs() < 1e-6, "non-8-aligned mismatch: {b} vs {s}");
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_position_zero_is_identity() {
        for head_dim in [2, 4, 8, 64] {
            let cfg = RopeConfig::new(head_dim, 1);
            let freqs = compute_frequencies(&cfg);
            let mut data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 3.17).collect();
            let original = data.clone();

            apply_rope(&mut data, 0, head_dim, &freqs);

            for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
                assert!(
                    (o - d).abs() < 1e-5,
                    "position 0 not identity at dim {i}, head_dim={head_dim}"
                );
            }
        }
    }

    #[test]
    fn test_large_position() {
        let cfg = RopeConfig::new(4, 8192);
        let freqs = compute_frequencies(&cfg);
        let mut data = vec![1.0, 0.0, 1.0, 0.0];

        apply_rope(&mut data, 8000, 4, &freqs);

        assert!(data.iter().all(|x| x.is_finite()), "large position produced non-finite values");
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected_norm = (2.0f32).sqrt();
        assert!(
            (norm - expected_norm).abs() < 1e-3,
            "norm at large position: {norm} vs {expected_norm}"
        );
    }

    #[test]
    fn test_batch_multi_head_consistency() {
        // Each head at the same position should get the same rotation
        let head_dim = 8;
        let num_heads = 4;
        let cfg = RopeConfig::new(head_dim, 2);
        let freqs = compute_frequencies(&cfg);

        let pattern: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut data: Vec<f32> =
            pattern.iter().copied().cycle().take(num_heads * head_dim).collect();

        apply_rope_batch(&mut data, 1, 1, num_heads, head_dim, &freqs);

        // All heads should produce identical results
        for h in 1..num_heads {
            for d in 0..head_dim {
                let ref_val = data[d];
                let val = data[h * head_dim + d];
                assert!(
                    (ref_val - val).abs() < 1e-6,
                    "head {h} diverges at dim {d}: {val} vs {ref_val}"
                );
            }
        }
    }

    // ── SIMD vs scalar parity ───────────────────────────────────────

    #[test]
    fn test_avx2_scalar_parity() {
        // The batch function dispatches to AVX2 when available.
        // Verify it matches the scalar single-apply path for various sizes.
        for head_dim in [8, 16, 32, 64, 128] {
            let num_heads = 4;
            let seq_len = 8;
            let cfg = RopeConfig::new(head_dim, seq_len + 4);
            let freqs = compute_frequencies(&cfg);

            let total = seq_len * num_heads * head_dim;
            let original: Vec<f32> =
                (0..total).map(|i| ((i * 7 + 3) as f32) * 0.01 - 2.0).collect();

            // Batch (may use AVX2)
            let mut batch = original.clone();
            apply_rope_batch(&mut batch, 2, seq_len, num_heads, head_dim, &freqs);

            // Scalar reference
            let mut scalar = original.clone();
            apply_rope_batch_scalar(&mut scalar, 2, seq_len, num_heads, head_dim, &freqs);

            for (i, (b, s)) in batch.iter().zip(scalar.iter()).enumerate() {
                assert!(
                    (b - s).abs() < 1e-5,
                    "SIMD/scalar mismatch at index {i} (head_dim={head_dim}): {b} vs {s}"
                );
            }
        }
    }

    #[test]
    fn test_apply_rope_zero_input_preserved() {
        let head_dim = 64;
        let seq_len = 4;
        let num_heads = 2;
        let total = seq_len * num_heads * head_dim;
        let mut data = vec![0.0f32; total];
        let freqs = compute_frequencies(&RopeConfig::new(head_dim, seq_len));
        apply_rope_batch_scalar(&mut data, 0, seq_len, num_heads, head_dim, &freqs);
        for (i, val) in data.iter().enumerate() {
            assert!(val.abs() < 1e-10, "zero not preserved at index {i}");
        }
    }

    #[test]
    fn test_batch_rope_norm_preservation_many_positions() {
        let head_dim = 32;
        let seq_len = 32;
        let num_heads = 4;
        let total = seq_len * num_heads * head_dim;
        let mut data: Vec<f32> = (0..total).map(|i| ((i * 37 + 13) as f32).sin() * 2.5).collect();
        let norms_before: Vec<f32> = (0..seq_len * num_heads)
            .map(|chunk| {
                let start = chunk * head_dim;
                data[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();
        let freqs = compute_frequencies(&RopeConfig::new(head_dim, seq_len));
        apply_rope_batch_scalar(&mut data, 0, seq_len, num_heads, head_dim, &freqs);
        for (chunk, norm_before) in norms_before.iter().enumerate() {
            let start = chunk * head_dim;
            let norm_after: f32 =
                data[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-3,
                "norm not preserved at chunk {chunk}: {norm_before} vs {norm_after}"
            );
        }
    }
}
