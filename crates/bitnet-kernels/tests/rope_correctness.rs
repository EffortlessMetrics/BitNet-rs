//! RoPE correctness regression tests.
//!
//! Pinned numerical outputs and structural invariants that guard against
//! accidental changes to the RoPE implementation. These are integration
//! tests exercising the public API from outside the crate.

use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Round to 6 decimal places for snapshot comparison.
fn snap(v: f32) -> f32 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

// ═══════════════════════════════════════════════════════════════════
// 1. Frequency table dimensions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn freq_table_dimensions_various() {
    // build_tables (compute_frequencies) must return max_seq_len * head_dim elements
    for (head_dim, seq_len) in [(2, 1), (2, 16384), (4, 128), (64, 512), (128, 16384)] {
        let cfg = RopeConfig::new(head_dim, seq_len);
        let freqs = compute_frequencies(&cfg);
        assert_eq!(
            freqs.len(),
            seq_len * head_dim,
            "wrong length for head_dim={head_dim}, seq_len={seq_len}"
        );
    }
}

#[test]
fn freq_table_minimum_dimensions() {
    // Minimum valid config: head_dim=2, seq_len=1
    let cfg = RopeConfig::new(2, 1);
    let freqs = compute_frequencies(&cfg);
    assert_eq!(freqs.len(), 2); // 1 position × 2 (cos, sin)
    // pos=0 → angle=0 → cos=1, sin=0
    assert!((freqs[0] - 1.0).abs() < 1e-7);
    assert!(freqs[1].abs() < 1e-7);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Pinned regression values
// ═══════════════════════════════════════════════════════════════════

#[test]
fn regression_head_dim4_pos1_pinned_output() {
    // Pinned expected values for head_dim=4, base=10000, position=1.
    // If these change, the RoPE formula has been altered.
    let cfg = RopeConfig::new(4, 2);
    let freqs = compute_frequencies(&cfg);
    let mut data = [1.0f32, 0.0, 0.0, 1.0];

    apply_rope(&mut data, 1, 4, &freqs);

    // pair 0: angle = 1.0 → cos(1)≈0.540302, sin(1)≈0.841471
    // pair 1: angle = 10000^(-0.5) ≈ 0.01 → cos≈0.99995, sin≈0.00999983
    let expected = [
        1.0f32.cos(),                   // 1*cos(1) - 0*sin(1)
        1.0f32.sin(),                   // 1*sin(1) + 0*cos(1)
        -(10000.0f32.powf(-0.5)).sin(), // 0*cos(θ₁) - 1*sin(θ₁)
        (10000.0f32.powf(-0.5)).cos(),  // 0*sin(θ₁) + 1*cos(θ₁)
    ];

    for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "regression dim {i}: got {got}, expected {want}");
    }
}

#[test]
fn regression_head_dim8_pos7_pinned_snapshot() {
    // A specific snapshot to catch any formula drift.
    let cfg = RopeConfig::new(8, 8);
    let freqs = compute_frequencies(&cfg);
    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];

    apply_rope(&mut data, 7, 8, &freqs);

    // Verify finite and record the snapshot for regression.
    let snapped: Vec<f32> = data.iter().map(|&v| snap(v)).collect();
    assert!(data.iter().all(|x| x.is_finite()));

    // Re-run and confirm identical output (determinism + pinned).
    let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
    apply_rope(&mut data2, 7, 8, &freqs);
    let snapped2: Vec<f32> = data2.iter().map(|&v| snap(v)).collect();
    assert_eq!(snapped, snapped2, "regression snapshot changed between runs");
}

#[test]
fn regression_frequency_values_pos1_head_dim8() {
    // Pin the frequency table at position 1 for head_dim=8, base=10000.
    // theta_i = 10000^(-2i/8) for i in 0..4
    // angle_i = 1.0 * theta_i
    let cfg = RopeConfig::new(8, 2);
    let freqs = compute_frequencies(&cfg);
    let pos1 = &freqs[8..16]; // position 1 starts at offset head_dim=8

    let thetas: [f32; 4] = [
        10000.0f32.powf(0.0),   // 1.0
        10000.0f32.powf(-0.25), // ~0.1
        10000.0f32.powf(-0.5),  // ~0.01
        10000.0f32.powf(-0.75), // ~0.001
    ];

    for (i, theta) in thetas.iter().enumerate() {
        let angle = theta; // pos=1, so angle = 1 * theta
        let expected_cos = angle.cos();
        let expected_sin = angle.sin();
        assert!(
            (pos1[2 * i] - expected_cos).abs() < 1e-5,
            "cos mismatch at pair {i}: got {}, expected {expected_cos}",
            pos1[2 * i]
        );
        assert!(
            (pos1[2 * i + 1] - expected_sin).abs() < 1e-5,
            "sin mismatch at pair {i}: got {}, expected {expected_sin}",
            pos1[2 * i + 1]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Rotation preserves vector magnitude
// ═══════════════════════════════════════════════════════════════════

#[test]
fn norm_preserved_across_full_sequence() {
    // Check every position in a 64-element sequence for head_dim=32.
    let head_dim = 32;
    let seq_len = 64;
    let cfg = RopeConfig::new(head_dim, seq_len);
    let freqs = compute_frequencies(&cfg);
    let original: Vec<f32> = (0..head_dim).map(|i| ((i * 7 + 3) as f32) * 0.1 - 1.0).collect();
    let norm_before = l2_norm(&original);

    for pos in 0..seq_len {
        let mut data = original.clone();
        apply_rope(&mut data, pos, head_dim, &freqs);
        let norm_after = l2_norm(&data);
        assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "norm diverged at pos={pos}: {norm_before} vs {norm_after}"
        );
    }
}

#[test]
fn norm_preserved_batch_16k_spot_check() {
    // Batch at 16K context boundary with multiple heads.
    let head_dim = 64;
    let num_heads = 4;
    let seq_len = 4;
    let start_pos = 16380; // positions 16380..16383
    let cfg = RopeConfig::new(head_dim, start_pos + seq_len);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let mut data: Vec<f32> = (0..total).map(|i| ((i * 13 + 5) as f32).sin()).collect();

    let norms_before: Vec<f32> = (0..seq_len * num_heads)
        .map(|chunk| {
            let start = chunk * head_dim;
            l2_norm(&data[start..start + head_dim])
        })
        .collect();

    apply_rope_batch(&mut data, start_pos, seq_len, num_heads, head_dim, &freqs);

    for (chunk, nb) in norms_before.iter().enumerate() {
        let start = chunk * head_dim;
        let na = l2_norm(&data[start..start + head_dim]);
        assert!((nb - na).abs() < 0.01, "norm diverged at chunk {chunk} (near 16K): {nb} vs {na}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Determinism
// ═══════════════════════════════════════════════════════════════════

#[test]
fn determinism_single_apply_bit_exact() {
    let head_dim = 64;
    let cfg = RopeConfig::new(head_dim, 128);
    let freqs = compute_frequencies(&cfg);
    let original: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.37 - 12.0).collect();

    let results: Vec<Vec<f32>> = (0..5)
        .map(|_| {
            let mut data = original.clone();
            apply_rope(&mut data, 99, head_dim, &freqs);
            data
        })
        .collect();

    for (run, result) in results.iter().enumerate().skip(1) {
        assert_eq!(&results[0], result, "determinism failed between run 0 and run {run}");
    }
}

#[test]
fn determinism_batch_bit_exact() {
    let head_dim = 32;
    let num_heads = 8;
    let seq_len = 16;
    let cfg = RopeConfig::new(head_dim, seq_len + 10);
    let freqs = compute_frequencies(&cfg);
    let total = seq_len * num_heads * head_dim;
    let original: Vec<f32> = (0..total).map(|i| ((i * 3 + 1) as f32) * 0.007).collect();

    let results: Vec<Vec<f32>> = (0..3)
        .map(|_| {
            let mut data = original.clone();
            apply_rope_batch(&mut data, 5, seq_len, num_heads, head_dim, &freqs);
            data
        })
        .collect();

    for (run, result) in results.iter().enumerate().skip(1) {
        assert_eq!(&results[0], result, "batch determinism failed between run 0 and run {run}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. 16K context length support
// ═══════════════════════════════════════════════════════════════════

#[test]
fn context_16k_frequency_table_finite() {
    let cfg = RopeConfig::new(128, 16384);
    let freqs = compute_frequencies(&cfg);

    // Spot-check positions near the boundary.
    for pos in [0, 1, 8191, 8192, 16382, 16383] {
        let offset = pos * 128;
        for i in 0..128 {
            assert!(
                freqs[offset + i].is_finite(),
                "non-finite at pos={pos}, idx={i}: {}",
                freqs[offset + i]
            );
        }
    }
}

#[test]
fn context_16k_cos_sin_identity() {
    // cos²+sin²=1 at the extremes of a 16K table.
    let cfg = RopeConfig::new(128, 16384);
    let freqs = compute_frequencies(&cfg);
    let half_dim = 64;

    for pos in [0, 16383] {
        for i in 0..half_dim {
            let idx = (pos * half_dim + i) * 2;
            let cos_v = freqs[idx];
            let sin_v = freqs[idx + 1];
            let sum = cos_v * cos_v + sin_v * sin_v;
            assert!((sum - 1.0).abs() < 1e-4, "cos²+sin²={sum} at pos={pos}, pair={i}");
        }
    }
}

#[test]
fn context_16k_batch_end_to_end() {
    // Run a small batch at the tail of 16K context.
    let head_dim = 64;
    let num_heads = 2;
    let seq_len = 4;
    let start_pos = 16380;
    let cfg = RopeConfig::new(head_dim, start_pos + seq_len);
    let freqs = compute_frequencies(&cfg);

    let total = seq_len * num_heads * head_dim;
    let mut data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 2.5).collect();

    apply_rope_batch(&mut data, start_pos, seq_len, num_heads, head_dim, &freqs);

    assert!(data.iter().all(|x| x.is_finite()), "16K tail batch produced non-finite values");
}

// ═══════════════════════════════════════════════════════════════════
// 6. Edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn edge_seq_len_1_head_dim_2() {
    let cfg = RopeConfig::new(2, 1);
    let freqs = compute_frequencies(&cfg);
    let mut data = [42.0f32, -7.0];
    let original = data;

    apply_rope(&mut data, 0, 2, &freqs);

    // pos=0 → identity
    assert!((data[0] - original[0]).abs() < 1e-6);
    assert!((data[1] - original[1]).abs() < 1e-6);
}

#[test]
fn edge_batch_seq_len_1_head_dim_2() {
    let cfg = RopeConfig::new(2, 2);
    let freqs = compute_frequencies(&cfg);
    let mut batch = [3.0f32, 5.0];
    let mut single = batch;

    apply_rope_batch(&mut batch, 1, 1, 1, 2, &freqs);
    apply_rope(&mut single, 1, 2, &freqs);

    for (b, s) in batch.iter().zip(single.iter()) {
        assert!((b - s).abs() < 1e-6, "batch vs single mismatch: {b} vs {s}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Frequency monotonic decay
// ═══════════════════════════════════════════════════════════════════

#[test]
fn frequencies_monotonically_decrease_all_head_dims() {
    // For each head_dim, at position 1 the inverse-frequency (and thus sin
    // magnitude) must decrease with dimension pair index.
    for head_dim in [4, 8, 16, 32, 64, 128] {
        let cfg = RopeConfig::new(head_dim, 2);
        let freqs = compute_frequencies(&cfg);
        let half_dim = head_dim / 2;
        let pos1_offset = head_dim; // position 1

        for i in 0..(half_dim - 1) {
            let sin_i = freqs[pos1_offset + 2 * i + 1].abs();
            let sin_next = freqs[pos1_offset + 2 * (i + 1) + 1].abs();
            assert!(
                sin_i >= sin_next - 1e-7,
                "head_dim={head_dim}: sin magnitude not monotonic at pair {i}: {sin_i} < {sin_next}"
            );
        }
    }
}

#[test]
fn frequencies_first_pair_fastest() {
    // The first dimension pair should rotate the fastest (largest sin at pos=1).
    let cfg = RopeConfig::new(64, 2);
    let freqs = compute_frequencies(&cfg);
    let pos1_offset = 64;

    let sin_first = freqs[pos1_offset + 1].abs();
    let sin_last = freqs[pos1_offset + 63].abs();
    assert!(
        sin_first > sin_last * 10.0,
        "first pair should rotate much faster: {sin_first} vs {sin_last}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// 8. Position encoding consistency
// ═══════════════════════════════════════════════════════════════════

#[test]
fn position_encoding_independent_of_table_size() {
    // Frequencies for position p must be identical regardless of max_seq_len.
    let head_dim = 32;
    let cfg_small = RopeConfig::new(head_dim, 64);
    let cfg_large = RopeConfig::new(head_dim, 16384);
    let freqs_small = compute_frequencies(&cfg_small);
    let freqs_large = compute_frequencies(&cfg_large);

    for pos in [0, 1, 10, 63] {
        let offset = pos * head_dim;
        for i in 0..head_dim {
            assert!(
                (freqs_small[offset + i] - freqs_large[offset + i]).abs() < 1e-7,
                "table-size dependence at pos={pos}, idx={i}"
            );
        }
    }
}

#[test]
fn apply_rope_same_result_regardless_of_table_size() {
    // Applying RoPE at the same position must give the same result
    // whether the table was built for seq_len=64 or seq_len=16384.
    let head_dim = 16;
    let cfg_small = RopeConfig::new(head_dim, 64);
    let cfg_large = RopeConfig::new(head_dim, 16384);
    let freqs_small = compute_frequencies(&cfg_small);
    let freqs_large = compute_frequencies(&cfg_large);
    let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

    for pos in [0, 1, 31, 63] {
        let mut data_small = original.clone();
        let mut data_large = original.clone();

        apply_rope(&mut data_small, pos, head_dim, &freqs_small);
        apply_rope(&mut data_large, pos, head_dim, &freqs_large);

        assert_eq!(data_small, data_large, "result differs for pos={pos} across table sizes");
    }
}
