//! ARM64-specific property-based tests for NEON kernels on Apple Silicon.
//!
//! Validates numerical invariants of NEON-accelerated kernels using proptest:
//!
//! - **LayerNorm (NEON)**: zero-mean, unit-variance, NEON-vs-scalar parity
//! - **RMSNorm (NEON)**: unit RMS, sign preservation, gamma scaling
//! - **RoPE (NEON)**: norm preservation, identity at pos 0, batch consistency
//! - **Embedding**: determinism, batch-vs-sequential parity, OOB errors
//! - **MatMul (I2_S)**: zero-vector identity, scaling linearity
//! - **Cross-path**: NEON matches scalar fallback for all tested sizes

#![cfg(all(target_arch = "aarch64", feature = "cpu"))]

use proptest::prelude::*;

use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm as cpu_layer_norm, rms_norm};
use bitnet_kernels::cpu::neon_layernorm::{layernorm_neon, rmsnorm_neon};
use bitnet_kernels::cpu::neon_rope::{
    apply_rope_batch_neon, apply_rope_neon, build_cos_sin_tables_neon,
};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};

// ── Helpers ────────────────────────────────────────────────────────

fn vec_f32_range(max_len: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(lo..hi, 1..=max_len)
}

fn proptest_cfg() -> ProptestConfig {
    ProptestConfig::with_cases(200)
}

const EPS: f32 = 1e-5;

// ═══════════════════════════════════════════════════════════════════
// 1. LayerNorm NEON properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_cfg())]

    /// NEON LayerNorm output has approximately zero mean (gamma=1, beta=0).
    #[test]
    fn prop_neon_layernorm_zero_mean(
        input in prop::collection::vec(-10.0f32..10.0f32, 2..=256),
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };

        let mean: f32 = output.iter().sum::<f32>() / n as f32;
        prop_assert!(
            mean.abs() < 1e-4,
            "LayerNorm output mean = {mean}, expected ~0"
        );
    }

    /// NEON LayerNorm output has approximately unit variance (gamma=1, beta=0).
    #[test]
    fn prop_neon_layernorm_unit_variance(
        input in prop::collection::vec(-10.0f32..10.0f32, 4..=256),
    ) {
        let n = input.len();
        // Skip near-constant inputs.
        let var_in: f32 = {
            let m = input.iter().sum::<f32>() / n as f32;
            input.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n as f32
        };
        prop_assume!(var_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };

        let out_mean: f32 = output.iter().sum::<f32>() / n as f32;
        let out_var: f32 =
            output.iter().map(|x| (x - out_mean) * (x - out_mean)).sum::<f32>() / n as f32;
        prop_assert!(
            (out_var - 1.0).abs() < 0.05,
            "LayerNorm output variance = {out_var}, expected ~1.0"
        );
    }

    /// NEON LayerNorm is deterministic (same input → identical output).
    #[test]
    fn prop_neon_layernorm_deterministic(
        input in prop::collection::vec(-10.0f32..10.0f32, 1..=128),
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut out1 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];

        unsafe {
            layernorm_neon(&input, &mut out1, &gamma, &beta, EPS);
            layernorm_neon(&input, &mut out2, &gamma, &beta, EPS);
        }

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            prop_assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "non-deterministic at [{i}]: {a} vs {b}"
            );
        }
    }

    /// NEON LayerNorm matches the scalar `cpu_layer_norm` path.
    #[test]
    fn prop_neon_layernorm_matches_scalar(
        input in prop::collection::vec(-10.0f32..10.0f32, 2..=256),
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];

        // Scalar reference.
        let config = LayerNormConfig::new(vec![n]);
        let scalar_out = cpu_layer_norm(&input, &gamma, &beta, &config).unwrap();

        // NEON path.
        let mut neon_out = vec![0.0f32; n];
        unsafe { layernorm_neon(&input, &mut neon_out, &gamma, &beta, EPS) };

        for (i, (&s, &ne)) in scalar_out.iter().zip(neon_out.iter()).enumerate() {
            prop_assert!(
                (s - ne).abs() < 1e-4,
                "LayerNorm parity at [{i}]: scalar={s}, neon={ne}"
            );
        }
    }

    /// NEON LayerNorm produces no NaN/Inf for finite inputs.
    #[test]
    fn prop_neon_layernorm_no_nan_inf(
        input in prop::collection::vec(-1e6f32..1e6f32, 1..=256),
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };

        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "LayerNorm output[{i}] is not finite: {v}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. RMSNorm NEON properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_cfg())]

    /// NEON RMSNorm with gamma=1 produces output with approximately unit RMS.
    #[test]
    fn prop_neon_rmsnorm_unit_rms(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=256
        ),
    ) {
        let n = input.len();
        let rms_in: f32 =
            (input.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };

        let rms_out: f32 =
            (output.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assert!(
            (rms_out - 1.0).abs() < 0.05,
            "RMSNorm output RMS = {rms_out}, expected ~1.0"
        );
    }

    /// NEON RMSNorm preserves sign when gamma is positive.
    #[test]
    fn prop_neon_rmsnorm_preserves_sign(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=256
        ),
    ) {
        let n = input.len();
        let rms_in: f32 =
            (input.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp.abs() > 1e-6 {
                prop_assert!(
                    inp.signum() == out.signum(),
                    "RMSNorm sign mismatch at [{i}]: input={inp}, output={out}"
                );
            }
        }
    }

    /// NEON RMSNorm gamma scaling: scale(gamma) → scale(output).
    #[test]
    fn prop_neon_rmsnorm_gamma_scaling(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=128
        ),
        scale in 0.5f32..5.0f32,
    ) {
        let n = input.len();
        let rms_in: f32 =
            (input.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma1 = vec![1.0f32; n];
        let gamma2 = vec![scale; n];
        let mut out1 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];

        unsafe {
            rmsnorm_neon(&input, &mut out1, &gamma1, EPS);
            rmsnorm_neon(&input, &mut out2, &gamma2, EPS);
        }

        for (i, (&v1, &v2)) in out1.iter().zip(out2.iter()).enumerate() {
            let expected = v1 * scale;
            prop_assert!(
                (v2 - expected).abs() < 1e-4,
                "RMSNorm gamma scaling at [{i}]: {v2} != {expected}"
            );
        }
    }

    /// NEON RMSNorm matches the scalar `rms_norm` path.
    #[test]
    fn prop_neon_rmsnorm_matches_scalar(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=256
        ),
    ) {
        let n = input.len();
        let rms_in: f32 =
            (input.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let scalar_out = rms_norm(&input, &gamma, &config).unwrap();

        let mut neon_out = vec![0.0f32; n];
        unsafe { rmsnorm_neon(&input, &mut neon_out, &gamma, EPS) };

        for (i, (&s, &ne)) in scalar_out.iter().zip(neon_out.iter()).enumerate() {
            prop_assert!(
                (s - ne).abs() < 1e-4,
                "RMSNorm parity at [{i}]: scalar={s}, neon={ne}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. RoPE NEON properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_cfg())]

    /// NEON RoPE preserves vector norm (rotation is unitary).
    #[test]
    fn prop_neon_rope_preserves_norm(
        dim_half in 1usize..=32,
        pos in 0usize..64,
    ) {
        let dim = dim_half * 2;
        let max_seq = 65;
        let (cos_t, sin_t) =
            unsafe { build_cos_sin_tables_neon(dim, max_seq, 10_000.0) };

        let data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let norm_before: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut out = data.clone();
        unsafe { apply_rope_neon(&mut out, &cos_t, &sin_t, dim, pos) };

        let norm_after: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        prop_assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "RoPE norm: before={norm_before}, after={norm_after}"
        );
    }

    /// NEON RoPE at position 0 is the identity transform.
    #[test]
    fn prop_neon_rope_identity_at_pos_zero(
        dim_half in 1usize..=32,
    ) {
        let dim = dim_half * 2;
        let (cos_t, sin_t) =
            unsafe { build_cos_sin_tables_neon(dim, 1, 10_000.0) };

        let data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.7).collect();
        let mut out = data.clone();
        unsafe { apply_rope_neon(&mut out, &cos_t, &sin_t, dim, 0) };

        for (i, (&orig, &rot)) in data.iter().zip(out.iter()).enumerate() {
            prop_assert!(
                (orig - rot).abs() < 1e-5,
                "RoPE pos 0 not identity at [{i}]: {orig} vs {rot}"
            );
        }
    }

    /// NEON RoPE batch application equals per-head sequential application.
    #[test]
    fn prop_neon_rope_batch_equals_sequential(
        dim_half in 1usize..=16,
        num_heads in 1usize..=8,
        pos in 0usize..32,
    ) {
        let dim = dim_half * 2;
        let max_seq = 33;
        let (cos_t, sin_t) =
            unsafe { build_cos_sin_tables_neon(dim, max_seq, 10_000.0) };

        let data: Vec<f32> = (0..num_heads * dim)
            .map(|i| (i as f32 * 0.13) - 2.0)
            .collect();

        // Batch path.
        let mut batch = data.clone();
        unsafe {
            apply_rope_batch_neon(&mut batch, &cos_t, &sin_t, dim, num_heads, pos);
        }

        // Sequential per-head path.
        let mut sequential = data.clone();
        for h in 0..num_heads {
            let off = h * dim;
            unsafe {
                apply_rope_neon(
                    &mut sequential[off..off + dim],
                    &cos_t,
                    &sin_t,
                    dim,
                    pos,
                );
            }
        }

        for (i, (&b, &s)) in batch.iter().zip(sequential.iter()).enumerate() {
            prop_assert!(
                (b - s).abs() < 1e-5,
                "RoPE batch/sequential mismatch at [{i}]: {b} vs {s}"
            );
        }
    }

    /// NEON RoPE matches the scalar `apply_rope` path from the rope module.
    #[test]
    fn prop_neon_rope_matches_scalar(
        dim_half in 2usize..=32,
        pos in 0usize..32,
    ) {
        let dim = dim_half * 2;
        let max_seq = 33;

        // NEON tables.
        let (cos_t, sin_t) =
            unsafe { build_cos_sin_tables_neon(dim, max_seq, 10_000.0) };

        // Scalar frequencies (interleaved [cos, sin] per pair).
        let config = RopeConfig::new(dim, max_seq);
        let freqs = compute_frequencies(&config);

        let data: Vec<f32> = (0..dim)
            .map(|i| ((i * 7 + 3) as f32) * 0.01 - 2.0)
            .collect();

        let mut neon_data = data.clone();
        unsafe { apply_rope_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

        let mut scalar_data = data.clone();
        apply_rope(&mut scalar_data, pos, dim, &freqs);

        for (i, (&n, &s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
            prop_assert!(
                (n - s).abs() < 1e-4,
                "RoPE NEON/scalar mismatch at [{i}] (dim={dim}, pos={pos}): {n} vs {s}"
            );
        }
    }

    /// NEON RoPE produces no NaN/Inf for finite inputs.
    #[test]
    fn prop_neon_rope_no_nan_inf(
        dim_half in 1usize..=32,
        pos in 0usize..64,
    ) {
        let dim = dim_half * 2;
        let max_seq = 65;
        let (cos_t, sin_t) =
            unsafe { build_cos_sin_tables_neon(dim, max_seq, 10_000.0) };

        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 100.0 - 5000.0).collect();
        unsafe { apply_rope_neon(&mut data, &cos_t, &sin_t, dim, pos) };

        for (i, &v) in data.iter().enumerate() {
            prop_assert!(v.is_finite(), "RoPE output[{i}] is not finite: {v}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Embedding properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_cfg())]

    /// Embedding lookup is deterministic.
    #[test]
    fn prop_embedding_lookup_deterministic(
        vocab in 2usize..=64,
        dim in 1usize..=32,
        num_idx in 1usize..=16,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<u32> = (0..num_idx).map(|i| (i % vocab) as u32).collect();

        let out1 = embedding_lookup(&table, &indices, dim).unwrap();
        let out2 = embedding_lookup(&table, &indices, dim).unwrap();

        prop_assert_eq!(out1, out2, "embedding lookup not deterministic");
    }

    /// Batch embedding lookup equals sequential single lookups.
    #[test]
    fn prop_embedding_batch_equals_sequential(
        vocab in 2usize..=64,
        dim in 1usize..=32,
        num_idx in 1usize..=16,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<u32> = (0..num_idx).map(|i| (i % vocab) as u32).collect();

        let batch = embedding_lookup(&table, &indices, dim).unwrap();

        for (j, &idx) in indices.iter().enumerate() {
            let single = embedding_lookup(&table, &[idx], dim).unwrap();
            let batch_row = &batch[j * dim..(j + 1) * dim];
            prop_assert_eq!(
                batch_row,
                single.as_slice(),
                "batch/single mismatch at index {j}"
            );
        }
    }

    /// Out-of-bounds index returns an error.
    #[test]
    fn prop_embedding_oob_returns_error(
        vocab in 2usize..=64,
        dim in 1usize..=32,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let oob_idx = vocab as u32; // one past the last valid index
        let result = embedding_lookup(&table, &[oob_idx], dim);
        prop_assert!(result.is_err(), "expected error for OOB index {oob_idx}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. MatMul (I2_S) properties
// ═══════════════════════════════════════════════════════════════════

/// Pack ternary weights and run a small I2_S matmul.
fn i2s_matmul_helper(
    activations: &[f32],
    weights: &[Vec<i8>],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let block_size = k; // single block
    let packed_k = k.div_ceil(4);
    let mut packed = vec![0u8; n * packed_k];
    let scales = vec![1.0f32; n]; // unit scale

    for col in 0..n {
        for row_start in (0..k).step_by(4) {
            let mut vals = [0i8; 4];
            for off in 0..4 {
                if row_start + off < k {
                    vals[off] = weights[col][row_start + off];
                }
            }
            packed[col * packed_k + row_start / 4] = pack_i2s(vals);
        }
    }

    let mut out = vec![0.0f32; m * n];
    i2s_matmul_f32(activations, &packed, &scales, &mut out, m, n, k, block_size).unwrap();
    out
}

proptest! {
    #![proptest_config(proptest_cfg())]

    /// I2_S matmul with zero activations produces zero output.
    #[test]
    fn prop_i2s_matmul_zero_activations(
        n in 1usize..=8,
        k_raw in 1usize..=16,
    ) {
        let k = (k_raw / 4).max(1) * 4; // align to 4
        let m = 1;
        let activations = vec![0.0f32; m * k];
        let weights: Vec<Vec<i8>> = (0..n)
            .map(|col| (0..k).map(|r| ((r + col) % 3) as i8 - 1).collect())
            .collect();

        let out = i2s_matmul_helper(&activations, &weights, m, n, k);

        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v.abs() < 1e-6,
                "zero * W should be 0 at [{i}]: got {v}"
            );
        }
    }

    /// I2_S matmul with zero weights produces zero output.
    #[test]
    fn prop_i2s_matmul_zero_weights(
        n in 1usize..=8,
        k_raw in 1usize..=16,
    ) {
        let k = (k_raw / 4).max(1) * 4;
        let m = 1;
        let activations: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<i8>> = (0..n).map(|_| vec![0i8; k]).collect();

        let out = i2s_matmul_helper(&activations, &weights, m, n, k);

        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v.abs() < 1e-6,
                "A * 0 should be 0 at [{i}]: got {v}"
            );
        }
    }

    /// Scaling activations by α scales the output by α.
    #[test]
    fn prop_i2s_matmul_scaling(
        n in 1usize..=4,
        k_raw in 1usize..=8,
        alpha in 0.5f32..5.0f32,
    ) {
        let k = (k_raw / 4).max(1) * 4;
        let m = 1;
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1) - 0.5).collect();
        let scaled: Vec<f32> = activations.iter().map(|&v| v * alpha).collect();
        let weights: Vec<Vec<i8>> = (0..n)
            .map(|col| (0..k).map(|r| ((r + col) % 3) as i8 - 1).collect())
            .collect();

        let out1 = i2s_matmul_helper(&activations, &weights, m, n, k);
        let out2 = i2s_matmul_helper(&scaled, &weights, m, n, k);

        for (i, (&v1, &v2)) in out1.iter().zip(out2.iter()).enumerate() {
            let expected = v1 * alpha;
            let tol = expected.abs() * 1e-4 + 1e-5;
            prop_assert!(
                (v2 - expected).abs() < tol,
                "scaling at [{i}]: {v2} != {expected} (alpha={alpha})"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. General NEON cross-path properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_cfg())]

    /// NEON LayerNorm handles non-4-aligned lengths correctly
    /// (exercises the scalar tail path inside NEON intrinsics).
    #[test]
    fn prop_neon_layernorm_alignment_independent(
        extra in 0usize..=3,
        base_len in 1usize..=32,
    ) {
        let n = base_len * 4 + extra; // test both aligned and unaligned
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];

        let config = LayerNormConfig::new(vec![n]);
        let scalar_out = cpu_layer_norm(&input, &gamma, &beta, &config).unwrap();

        let mut neon_out = vec![0.0f32; n];
        unsafe { layernorm_neon(&input, &mut neon_out, &gamma, &beta, EPS) };

        for (i, (&s, &ne)) in scalar_out.iter().zip(neon_out.iter()).enumerate() {
            prop_assert!(
                (s - ne).abs() < 1e-4,
                "alignment parity at [{i}] (n={n}): scalar={s}, neon={ne}"
            );
        }
    }

    /// NEON RoPE handles non-8-aligned dimensions correctly
    /// (exercises the scalar tail path inside NEON RoPE).
    #[test]
    fn prop_neon_rope_alignment_independent(
        dim_half in 1usize..=16,
        pos in 0usize..16,
    ) {
        let dim = dim_half * 2;
        let max_seq = 17;
        let (cos_t, sin_t) =
            unsafe { build_cos_sin_tables_neon(dim, max_seq, 10_000.0) };

        let config = RopeConfig::new(dim, max_seq);
        let freqs = compute_frequencies(&config);

        let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 1.5).collect();

        let mut neon_data = data.clone();
        unsafe { apply_rope_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

        let mut scalar_data = data.clone();
        apply_rope(&mut scalar_data, pos, dim, &freqs);

        for (i, (&n, &s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
            prop_assert!(
                (n - s).abs() < 1e-4,
                "RoPE alignment at [{i}] (dim={dim}, pos={pos}): neon={n}, scalar={s}"
            );
        }
    }
}
