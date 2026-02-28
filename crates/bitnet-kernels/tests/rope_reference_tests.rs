//! CPU reference tests for the RoPE OpenCL kernel logic.
//!
//! Validates the rotary position embedding math using a CPU reference
//! implementation that mirrors the rope.cl kernel logic.

/// Pre-compute RoPE frequency tables (mirrors host-side setup).
fn build_freq_tables(
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut freq_cos = vec![0.0f32; max_seq_len * half_dim];
    let mut freq_sin = vec![0.0f32; max_seq_len * half_dim];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let theta = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * theta;
            freq_cos[pos * half_dim + i] = angle.cos();
            freq_sin[pos * half_dim + i] = angle.sin();
        }
    }
    (freq_cos, freq_sin)
}

/// CPU reference implementation of rope_apply (mirrors rope.cl).
fn rope_apply_reference(
    x: &mut [f32],
    freq_cos: &[f32],
    freq_sin: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    pos_offset: usize,
) {
    let half_dim = head_dim / 2;

    for seq_pos in 0..seq_len {
        for head_idx in 0..num_heads {
            for pair_idx in 0..half_dim {
                let base_idx =
                    (seq_pos * num_heads + head_idx) * head_dim + 2 * pair_idx;
                let pos = pos_offset + seq_pos;
                let freq_idx = pos * half_dim + pair_idx;

                let x_even = x[base_idx];
                let x_odd = x[base_idx + 1];
                let fc = freq_cos[freq_idx];
                let fs = freq_sin[freq_idx];

                x[base_idx] = x_even * fc - x_odd * fs;
                x[base_idx + 1] = x_even * fs + x_odd * fc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_identity_at_position_zero() {
        // At position 0, all angles are 0 → cos=1, sin=0 → identity
        let head_dim = 4;
        let num_heads = 1;
        let seq_len = 1;
        let (freq_cos, freq_sin) = build_freq_tables(1, head_dim, 10000.0);

        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let original = x.clone();

        rope_apply_reference(
            &mut x, &freq_cos, &freq_sin,
            seq_len, num_heads, head_dim, 0,
        );

        for i in 0..x.len() {
            assert!(
                (x[i] - original[i]).abs() < 1e-6,
                "at pos 0, RoPE should be identity: x[{}] = {} vs {}",
                i, x[i], original[i],
            );
        }
    }

    #[test]
    fn test_rope_preserves_vector_norm() {
        // RoPE is a rotation — it should preserve the L2 norm of each (even,odd) pair
        let head_dim = 8;
        let num_heads = 2;
        let seq_len = 3;
        let max_seq = 16;
        let (freq_cos, freq_sin) = build_freq_tables(max_seq, head_dim, 10000.0);

        let mut x: Vec<f32> = (0..(seq_len * num_heads * head_dim))
            .map(|i| (i as f32) * 0.1 - 1.0)
            .collect();

        // Compute pair norms before
        let half_dim = head_dim / 2;
        let total_pairs = seq_len * num_heads * half_dim;
        let mut norms_before = Vec::with_capacity(total_pairs);
        for s in 0..seq_len {
            for h in 0..num_heads {
                for p in 0..half_dim {
                    let idx = (s * num_heads + h) * head_dim + 2 * p;
                    let norm = (x[idx] * x[idx] + x[idx + 1] * x[idx + 1]).sqrt();
                    norms_before.push(norm);
                }
            }
        }

        rope_apply_reference(
            &mut x, &freq_cos, &freq_sin,
            seq_len, num_heads, head_dim, 0,
        );

        // Verify norms preserved
        let mut pair_idx = 0;
        for s in 0..seq_len {
            for h in 0..num_heads {
                for p in 0..half_dim {
                    let idx = (s * num_heads + h) * head_dim + 2 * p;
                    let norm_after =
                        (x[idx] * x[idx] + x[idx + 1] * x[idx + 1]).sqrt();
                    assert!(
                        (norm_after - norms_before[pair_idx]).abs() < 1e-4,
                        "norm not preserved at pair {}: before={} after={}",
                        pair_idx, norms_before[pair_idx], norm_after,
                    );
                    pair_idx += 1;
                }
            }
        }
    }

    #[test]
    fn test_rope_different_positions_give_different_results() {
        let head_dim = 4;
        let num_heads = 1;
        let max_seq = 16;
        let (freq_cos, freq_sin) = build_freq_tables(max_seq, head_dim, 10000.0);

        let input = vec![1.0f32, 0.5, -1.0, 0.5];

        let mut x0 = input.clone();
        rope_apply_reference(&mut x0, &freq_cos, &freq_sin, 1, num_heads, head_dim, 0);

        let mut x5 = input.clone();
        rope_apply_reference(&mut x5, &freq_cos, &freq_sin, 1, num_heads, head_dim, 5);

        // At least one element should differ
        let any_diff = x0
            .iter()
            .zip(x5.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "different positions should produce different embeddings");
    }

    #[test]
    fn test_rope_pos_offset_matches_sequential() {
        // Applying RoPE with pos_offset=3 to a single token should match
        // the 4th token of a length-4 sequence starting at offset 0
        let head_dim = 4;
        let num_heads = 1;
        let max_seq = 8;
        let (freq_cos, freq_sin) = build_freq_tables(max_seq, head_dim, 10000.0);

        let token_data = vec![1.0f32, 2.0, 3.0, 4.0];

        // Method 1: single token at offset 3
        let mut x1 = token_data.clone();
        rope_apply_reference(&mut x1, &freq_cos, &freq_sin, 1, num_heads, head_dim, 3);

        // Method 2: 4-token sequence, take token at index 3
        let mut x2 = vec![0.0f32; 4 * head_dim];
        x2[3 * head_dim..4 * head_dim].copy_from_slice(&token_data);
        rope_apply_reference(&mut x2, &freq_cos, &freq_sin, 4, num_heads, head_dim, 0);

        let start = 3 * head_dim;
        for i in 0..head_dim {
            assert!(
                (x1[i] - x2[start + i]).abs() < 1e-5,
                "pos_offset mismatch at {}: {} vs {}",
                i, x1[i], x2[start + i],
            );
        }
    }

    #[test]
    fn test_rope_multi_head() {
        // Each head should be rotated independently with the same frequencies
        let head_dim = 4;
        let num_heads = 3;
        let max_seq = 4;
        let (freq_cos, freq_sin) = build_freq_tables(max_seq, head_dim, 10000.0);

        // All heads start with the same data
        let head_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut x = Vec::new();
        for _ in 0..num_heads {
            x.extend_from_slice(&head_data);
        }

        rope_apply_reference(&mut x, &freq_cos, &freq_sin, 1, num_heads, head_dim, 1);

        // All heads should get the same result (same position, same freq)
        for h in 1..num_heads {
            for i in 0..head_dim {
                assert!(
                    (x[i] - x[h * head_dim + i]).abs() < 1e-6,
                    "head {} differs from head 0 at dim {}: {} vs {}",
                    h, i, x[h * head_dim + i], x[i],
                );
            }
        }
    }

    #[test]
    fn test_rope_known_rotation_90_degrees() {
        // Construct a frequency table where position 1 gives exactly 90° rotation
        // for the first pair: cos=0, sin=1
        let head_dim = 2;
        // half_dim = 1, freq tables have 2 entries: pos=0 and pos=1
        // freq_cos[pos=1][i=0] = cos(pi/2) = 0
        // freq_sin[pos=1][i=0] = sin(pi/2) = 1
        let freq_cos = vec![1.0, 0.0]; // pos=0: cos=1, pos=1: cos=0
        let freq_sin = vec![0.0, 1.0]; // pos=0: sin=0, pos=1: sin=1

        let mut x = vec![3.0f32, 4.0]; // one head, one pair

        rope_apply_reference(&mut x, &freq_cos, &freq_sin, 1, 1, head_dim, 1);

        // [3,4] rotated by 90°: [3*0 - 4*1, 3*1 + 4*0] = [-4, 3]
        assert!(
            (x[0] - (-4.0)).abs() < 1e-6,
            "expected -4.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 3.0).abs() < 1e-6,
            "expected 3.0, got {}",
            x[1]
        );
    }
}
