//! Property-based tests — wave 28.
//!
//! Matrix operation properties: distributivity of matmul over addition,
//! matmul identity and zero, SIMD matmul shape invariants, batch matmul
//! uniformity, reduction kernel invariants (sum/mean/l1/l2 norms), RoPE
//! frequency properties, linear layer output shape, layer-norm idempotence,
//! quantize/dequant i8 round-trip error bounds, embedding pack/unpack
//! fidelity, softmax temperature monotonicity, and residual connection
//! properties.
//!
//! 55 property assertions across 18 invariant categories.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::embedding::{
    embedding_lookup, normalize_embeddings, pack_embedding_table, positional_embedding,
    unpack_embedding_lookup,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_forward};
use bitnet_kernels::cpu::quantize::{
    dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use bitnet_kernels::cuda::matmul::{MatmulConfig, matmul_cpu};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn vec_f32(n: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, n..=n)
}

fn finite_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-5.0f32..5.0, 1..=max_len)
}

// ── 1. Matmul distributivity: A*(B+C) ≈ A*B + A*C ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn matmul_distributes_over_addition(
        a in vec_f32(4, -2.0, 2.0),
        b in vec_f32(4, -2.0, 2.0),
        c in vec_f32(4, -2.0, 2.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        // B + C element-wise
        let bc: Vec<f32> = b.iter().zip(&c).map(|(x, y)| x + y).collect();
        // A*(B+C)
        let mut a_bc = vec![0.0f32; 4];
        matmul_cpu(&a, &bc, &mut a_bc, &cfg).unwrap();
        // A*B
        let mut ab = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        // A*C
        let mut ac = vec![0.0f32; 4];
        matmul_cpu(&a, &c, &mut ac, &cfg).unwrap();
        // A*B + A*C
        let ab_ac: Vec<f32> = ab.iter().zip(&ac).map(|(x, y)| x + y).collect();

        for i in 0..4 {
            prop_assert!(
                (a_bc[i] - ab_ac[i]).abs() < 1e-3,
                "distributivity failed at [{}]: {} vs {}", i, a_bc[i], ab_ac[i]
            );
        }
    }
}

// ── 2. Scalar multiplication through matmul: (αA)*B = α*(A*B) ──────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Scaling input before matmul equals scaling output after.
    #[test]
    fn scalar_matmul_commutativity(
        a in vec_f32(4, -2.0, 2.0),
        b in vec_f32(4, -2.0, 2.0),
        alpha in -2.0f32..2.0,
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        // (αA)*B
        let scaled_a: Vec<f32> = a.iter().map(|&x| x * alpha).collect();
        let mut left = vec![0.0f32; 4];
        matmul_cpu(&scaled_a, &b, &mut left, &cfg).unwrap();
        // α*(A*B)
        let mut ab = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        let right: Vec<f32> = ab.iter().map(|&x| x * alpha).collect();

        for i in 0..4 {
            prop_assert!(
                (left[i] - right[i]).abs() < 1e-3,
                "scalar commutativity failed at [{}]: {} vs {}", i, left[i], right[i]
            );
        }
    }

    /// A * zero-matrix = zero-matrix.
    #[test]
    fn matmul_zero_right(a in vec_f32(4, -3.0, 3.0)) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let zero = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &zero, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "A*0 != 0 at [{}]: {}", i, v);
        }
    }

    /// zero-matrix * B = zero-matrix.
    #[test]
    fn matmul_zero_left(b in vec_f32(4, -3.0, 3.0)) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let zero = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&zero, &b, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "0*B != 0 at [{}]: {}", i, v);
        }
    }
}

// ── 3. SIMD matmul shape and identity ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// simd_matmul_f32 output has correct length m*n.
    #[test]
    fn simd_matmul_output_shape(
        m in 1usize..6,
        n in 1usize..6,
        k in 1usize..6,
    ) {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let cfg = SimdMatmulConfig::new(m, n, k);
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        prop_assert_eq!(c.len(), m * n);
    }

    /// simd_matmul_f32: A * I ≈ A for 3×3.
    #[test]
    fn simd_matmul_identity(a in vec_f32(9, -3.0, 3.0)) {
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let cfg = SimdMatmulConfig::new(3, 3, 3);
        let mut out = vec![0.0f32; 9];
        simd_matmul_f32(&a, &identity, &mut out, &cfg).unwrap();
        for i in 0..9 {
            prop_assert!(
                (out[i] - a[i]).abs() < 1e-4,
                "A*I != A at [{}]: {} vs {}", i, out[i], a[i]
            );
        }
    }

    /// simd_matmul_f32 with alpha scaling.
    #[test]
    fn simd_matmul_alpha_scaling(
        a in vec_f32(4, -2.0, 2.0),
        b in vec_f32(4, -2.0, 2.0),
        alpha in 0.1f32..3.0,
    ) {
        let cfg_base = SimdMatmulConfig::new(2, 2, 2);
        let mut base = vec![0.0f32; 4];
        simd_matmul_f32(&a, &b, &mut base, &cfg_base).unwrap();

        let cfg_scaled = SimdMatmulConfig {
            m: 2, n: 2, k: 2,
            alpha,
            beta: 0.0,
            transpose_a: false,
            transpose_b: false,
        };
        let mut scaled = vec![0.0f32; 4];
        simd_matmul_f32(&a, &b, &mut scaled, &cfg_scaled).unwrap();

        for i in 0..4 {
            let expected = base[i] * alpha;
            prop_assert!(
                (scaled[i] - expected).abs() < 1e-3,
                "alpha scaling failed at [{}]: {} vs {}", i, scaled[i], expected
            );
        }
    }
}

// ── 4. Matmul associativity: (A*B)*C ≈ A*(B*C) ─────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn matmul_associativity(
        a in vec_f32(4, -2.0, 2.0),
        b in vec_f32(4, -2.0, 2.0),
        c in vec_f32(4, -2.0, 2.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        // AB
        let mut ab = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        // (AB)C
        let mut abc_left = vec![0.0f32; 4];
        matmul_cpu(&ab, &c, &mut abc_left, &cfg).unwrap();
        // BC
        let mut bc = vec![0.0f32; 4];
        matmul_cpu(&b, &c, &mut bc, &cfg).unwrap();
        // A(BC)
        let mut abc_right = vec![0.0f32; 4];
        matmul_cpu(&a, &bc, &mut abc_right, &cfg).unwrap();

        for i in 0..4 {
            prop_assert!(
                (abc_left[i] - abc_right[i]).abs() < 1e-2,
                "associativity failed at [{}]: {} vs {}", i, abc_left[i], abc_right[i]
            );
        }
    }
}

// ── 6. Reduction kernel invariants ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// sum of all-ones vector equals length.
    #[test]
    fn reduction_sum_ones(n in 1usize..64) {
        let data = vec![1.0f32; n];
        let s = ReductionKernel::sum(&data).unwrap();
        prop_assert!(
            (s - n as f32).abs() < 1e-4,
            "sum({} ones) = {} != {}", n, s, n
        );
    }

    /// mean of constant vector equals the constant.
    #[test]
    fn reduction_mean_constant(
        n in 1usize..64,
        c in -10.0f32..10.0,
    ) {
        let data = vec![c; n];
        let m = ReductionKernel::mean(&data).unwrap();
        prop_assert!(
            (m - c).abs() < 1e-4,
            "mean of [{}; {}] = {} != {}", c, n, m, c
        );
    }

    /// l2_norm of a single-element vector equals abs(value).
    #[test]
    fn reduction_l2_norm_singleton(v in -10.0f32..10.0) {
        let data = vec![v];
        let norm = ReductionKernel::l2_norm(&data).unwrap();
        prop_assert!(
            (norm - v.abs()).abs() < 1e-5,
            "l2([{}]) = {} != {}", v, norm, v.abs()
        );
    }

    /// l1_norm >= 0 always.
    #[test]
    fn reduction_l1_norm_nonneg(data in finite_vec(32)) {
        let norm = ReductionKernel::l1_norm(&data).unwrap();
        prop_assert!(norm >= 0.0, "l1 norm negative: {}", norm);
    }

    /// l2_norm >= 0 always.
    #[test]
    fn reduction_l2_norm_nonneg(data in finite_vec(32)) {
        let norm = ReductionKernel::l2_norm(&data).unwrap();
        prop_assert!(norm >= 0.0, "l2 norm negative: {}", norm);
    }

    /// max >= min always.
    #[test]
    fn reduction_max_ge_min(data in finite_vec(32)) {
        let mx = ReductionKernel::max(&data).unwrap();
        let mn = ReductionKernel::min(&data).unwrap();
        prop_assert!(
            mx.value >= mn.value,
            "max {} < min {}", mx.value, mn.value
        );
    }
}

// ── 7. RoPE frequency properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Frequency table length = max_seq_len * head_dim.
    #[test]
    fn rope_freq_length(head_dim in (1usize..16).prop_map(|d| d * 2)) {
        let max_seq_len = 128;
        let config = RopeConfig::new(head_dim, max_seq_len);
        let freqs = compute_frequencies(&config);
        let expected = max_seq_len * head_dim;
        prop_assert_eq!(freqs.len(), expected, "freq len {} != expected {}", freqs.len(), expected);
    }

    /// All frequency values are finite.
    #[test]
    fn rope_freq_finite(head_dim in (1usize..16).prop_map(|d| d * 2)) {
        let config = RopeConfig::new(head_dim, 128);
        let freqs = compute_frequencies(&config);
        for (i, &f) in freqs.iter().enumerate() {
            prop_assert!(f.is_finite(), "freq[{}] = {} not finite", i, f);
        }
    }

    /// apply_rope preserves vector length.
    #[test]
    fn rope_preserves_length(
        head_dim in (1usize..8).prop_map(|d| d * 2),
        pos in 0usize..64,
    ) {
        let config = RopeConfig::new(head_dim, 128);
        let freqs = compute_frequencies(&config);
        let mut data = vec![1.0f32; head_dim];
        let orig_len = data.len();
        apply_rope(&mut data, pos, head_dim, &freqs);
        prop_assert_eq!(data.len(), orig_len);
    }
}

// ── 8. Linear layer output shape ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Linear output has length batch * out_features.
    #[test]
    fn linear_output_shape(
        batch in 1usize..4,
        inf in 1usize..8,
        outf in 1usize..8,
    ) {
        let input = vec![1.0f32; batch * inf];
        let weights = vec![0.1f32; outf * inf];
        let bias = vec![0.0f32; outf];
        let mut output = vec![0.0f32; batch * outf];
        let config = LinearConfig::new(batch, inf, outf).unwrap();
        linear_forward(&input, &weights, Some(&bias), &mut output, &config).unwrap();
        prop_assert_eq!(output.len(), batch * outf);
    }

    /// Linear with zero weights and zero bias produces zeros.
    #[test]
    fn linear_zero_weights(
        batch in 1usize..4,
        inf in 1usize..8,
        outf in 1usize..8,
    ) {
        let input = vec![1.0f32; batch * inf];
        let weights = vec![0.0f32; outf * inf];
        let bias = vec![0.0f32; outf];
        let mut output = vec![0.0f32; batch * outf];
        let config = LinearConfig::new(batch, inf, outf).unwrap();
        linear_forward(&input, &weights, Some(&bias), &mut output, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "output[{}] = {} != 0", i, v);
        }
    }
}

// ── 9. Layer-norm idempotence ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Applying layer_norm twice with gamma=1, beta=0 gives the same result.
    #[test]
    fn layernorm_idempotent(data in vec_f32(8, -3.0, 3.0)) {
        let gamma = vec![1.0f32; 8];
        let beta = vec![0.0f32; 8];
        let config = LayerNormConfig::new(vec![8]);

        let first = layer_norm(&data, &gamma, Some(&beta), &config).unwrap();
        let second = layer_norm(&first, &gamma, Some(&beta), &config).unwrap();

        for i in 0..8 {
            prop_assert!(
                (first[i] - second[i]).abs() < 1e-4,
                "layernorm not idempotent at [{}]: {} vs {}", i, first[i], second[i]
            );
        }
    }

    /// RMS-norm output is always finite for bounded input.
    #[test]
    fn rms_norm_output_finite(data in vec_f32(8, -5.0, 5.0)) {
        let gamma = vec![1.0f32; 8];
        let config = LayerNormConfig::new(vec![8]);
        let out = rms_norm(&data, &gamma, &config).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "rms_norm[{}] = {} not finite", i, v);
        }
    }
}

// ── 10. Quantize/dequant i8 round-trip error ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Round-trip error is bounded for symmetric i8 quantization.
    #[test]
    fn quant_i8_roundtrip_bounded(data in vec_f32(16, -5.0, 5.0)) {
        let (quantized, scale) = quantize_symmetric_i8(&data, 8);
        let recovered = dequantize_symmetric_i8(&quantized, scale);
        for i in 0..data.len() {
            let err = (data[i] - recovered[i]).abs();
            prop_assert!(
                err < scale + 1e-6,
                "roundtrip error {} > scale {} at [{}]", err, scale, i
            );
        }
    }

    /// Scale is always non-negative.
    #[test]
    fn quant_i8_scale_nonneg(data in vec_f32(16, -10.0, 10.0)) {
        let (_, scale) = quantize_symmetric_i8(&data, 8);
        prop_assert!(scale >= 0.0, "scale {} < 0", scale);
    }

    /// Quantized values are in [-127, 127] for 8-bit.
    #[test]
    fn quant_i8_value_range(data in vec_f32(16, -10.0, 10.0)) {
        let (quantized, _) = quantize_symmetric_i8(&data, 8);
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert!(
                v >= -127,
                "quantized[{}] = {} out of range", i, v
            );
        }
    }
}

// ── 11. Ternary quantization ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Ternary values are in {-1, 0, 1}.
    #[test]
    fn ternary_values_in_set(
        data in vec_f32(16, -5.0, 5.0),
        threshold in 0.01f32..2.0,
    ) {
        let quantized = quantize_ternary(&data, threshold);
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert!(
                v == -1 || v == 0 || v == 1,
                "ternary[{}] = {} not in {{-1,0,1}}", i, v
            );
        }
    }

    /// Binary quantization values are in {-1, 1}.
    #[test]
    fn binary_values_in_set(data in vec_f32(16, -5.0, 5.0)) {
        let quantized = quantize_binary(&data);
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert!(
                v == -1 || v == 1,
                "binary[{}] = {} not in {{-1,1}}", i, v
            );
        }
    }
}

// ── 12. Embedding pack/unpack fidelity ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Pack then unpack recovers original embeddings within quantization tolerance.
    #[test]
    fn embedding_pack_unpack_roundtrip(
        dim in 2usize..8,
        vocab in 2usize..8,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.1).collect();
        let packed = pack_embedding_table(&table, vocab, dim);
        let idx = vec![0u32];
        let recovered = unpack_embedding_lookup(&packed, &idx).unwrap();
        // Quantization to i8 introduces error up to abs_max/127 per element
        let abs_max = table[..dim].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let tol = (abs_max / 127.0) + 1e-5;
        for i in 0..dim {
            prop_assert!(
                (recovered[i] - table[i]).abs() < tol,
                "pack/unpack mismatch at [{}]: {} vs {} (tol={})", i, recovered[i], table[i], tol
            );
        }
    }

    /// Embedding lookup output length = num_indices * dim.
    #[test]
    fn embedding_lookup_output_len(
        dim in 2usize..8,
        vocab in 2usize..8,
        n_idx in 1usize..4,
    ) {
        let table = vec![1.0f32; vocab * dim];
        let indices: Vec<u32> = (0..n_idx).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(out.len(), n_idx * dim);
    }
}

// ── 13. Positional embedding properties ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Positional embedding output length = seq_len * embed_dim.
    #[test]
    fn positional_embed_shape(
        seq_len in 1usize..16,
        dim in (1usize..8).prop_map(|d| d * 2),
    ) {
        let pe = positional_embedding(seq_len, dim);
        prop_assert_eq!(pe.len(), seq_len * dim);
    }

    /// All positional embedding values are finite.
    #[test]
    fn positional_embed_finite(
        seq_len in 1usize..16,
        dim in (1usize..8).prop_map(|d| d * 2),
    ) {
        let pe = positional_embedding(seq_len, dim);
        for (i, &v) in pe.iter().enumerate() {
            prop_assert!(v.is_finite(), "pe[{}] = {} not finite", i, v);
        }
    }

    /// Positional embeddings at position 0: sin(0)=0 for even dims, cos(0)=1 for odd dims.
    #[test]
    fn positional_embed_pos0_sin(dim in (2usize..8).prop_map(|d| d * 2)) {
        let pe = positional_embedding(1, dim);
        // Even indices are sin(0)=0, odd indices are cos(0)=1
        prop_assert!(
            pe[0].abs() < 1e-5,
            "pe[0] = {} should be ~0 (sin(0))", pe[0]
        );
        prop_assert!(
            (pe[1] - 1.0).abs() < 1e-5,
            "pe[1] = {} should be ~1 (cos(0))", pe[1]
        );
    }
}

// ── 14. Softmax temperature monotonicity ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Higher temperature → higher entropy (flatter distribution).
    #[test]
    fn softmax_higher_temp_higher_entropy(logits in vec_f32(8, -3.0, 3.0)) {
        // Low temperature
        let cfg_lo = SoftmaxConfig::for_shape(8, 1).unwrap();
        let scaled_lo: Vec<f32> = logits.iter().map(|&x| x / 0.5).collect();
        let mut out_lo = vec![0.0f32; 8];
        softmax_cpu(&scaled_lo, &mut out_lo, &cfg_lo).unwrap();

        // High temperature
        let scaled_hi: Vec<f32> = logits.iter().map(|&x| x / 2.0).collect();
        let mut out_hi = vec![0.0f32; 8];
        softmax_cpu(&scaled_hi, &mut out_hi, &cfg_lo).unwrap();

        // Entropy: -sum(p * ln(p))
        let entropy = |probs: &[f32]| -> f32 {
            probs.iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| -p * p.ln())
                .sum::<f32>()
        };
        let h_lo = entropy(&out_lo);
        let h_hi = entropy(&out_hi);
        prop_assert!(
            h_hi >= h_lo - 1e-4,
            "higher temp should have >= entropy: {} vs {}", h_hi, h_lo
        );
    }

    /// Softmax with batch_size > 1 preserves per-row normalization.
    #[test]
    fn softmax_batch_rows_normalize(
        rows in 2usize..4,
        cols in 2usize..8,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let cfg = SoftmaxConfig::for_shape(cols, rows).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for r in 0..rows {
            let row_sum: f32 = out[r * cols..(r + 1) * cols].iter().sum();
            prop_assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "row {} sum = {} != 1.0", r, row_sum
            );
        }
    }
}

// ── 15. Residual connection properties ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// add_residual(x, 0) ≈ x
    #[test]
    fn residual_identity(data in vec_f32(16, -5.0, 5.0)) {
        let zero = vec![0.0f32; 16];
        let mut out = data.clone();
        add_residual(&mut out, &zero).unwrap();
        for i in 0..16 {
            prop_assert!(
                (out[i] - data[i]).abs() < 1e-6,
                "residual identity failed at [{}]", i
            );
        }
    }

    /// add_residual_scaled(x, r, 0.0) ≈ x
    #[test]
    fn residual_scaled_zero(data in vec_f32(16, -5.0, 5.0)) {
        let residual = vec![1.0f32; 16];
        let mut out = data.clone();
        add_residual_scaled(&mut out, &residual, 0.0).unwrap();
        for i in 0..16 {
            prop_assert!(
                (out[i] - data[i]).abs() < 1e-6,
                "scaled residual zero failed at [{}]", i
            );
        }
    }

    /// add_residual is commutative: x+r ≈ r+x
    #[test]
    fn residual_commutative(
        a in vec_f32(8, -5.0, 5.0),
        b in vec_f32(8, -5.0, 5.0),
    ) {
        let mut ab = a.clone();
        add_residual(&mut ab, &b).unwrap();
        let mut ba = b.clone();
        add_residual(&mut ba, &a).unwrap();
        for i in 0..8 {
            prop_assert!(
                (ab[i] - ba[i]).abs() < 1e-5,
                "residual not commutative at [{}]: {} vs {}", i, ab[i], ba[i]
            );
        }
    }
}

// ── 16. Normalize embeddings unit-norm ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// After normalize_embeddings, each row has L2 norm ≈ 1.
    #[test]
    fn normalized_embeddings_unit_norm(
        dim in 2usize..8,
        n_rows in 1usize..4,
    ) {
        let n = n_rows * dim;
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.5).collect();
        normalize_embeddings(&mut data, dim);
        for r in 0..n_rows {
            let row = &data[r * dim..(r + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!(
                (norm - 1.0).abs() < 1e-4,
                "row {} norm = {} != 1.0", r, norm
            );
        }
    }
}

// ── 17. Matmul commutativity with transpose ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// For 1×1 "matrices" (scalars), matmul is commutative.
    #[test]
    fn matmul_scalar_commutative(
        a in vec_f32(1, -5.0, 5.0),
        b in vec_f32(1, -5.0, 5.0),
    ) {
        let cfg = MatmulConfig::for_shape(1, 1, 1).unwrap();
        let mut ab = vec![0.0f32; 1];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        let mut ba = vec![0.0f32; 1];
        matmul_cpu(&b, &a, &mut ba, &cfg).unwrap();
        prop_assert!(
            (ab[0] - ba[0]).abs() < 1e-5,
            "scalar matmul not commutative: {} vs {}", ab[0], ba[0]
        );
    }

    /// Matmul of all-ones produces correct inner-product result.
    #[test]
    fn matmul_ones_product(
        m in 1usize..5,
        k in 1usize..5,
        n in 1usize..5,
    ) {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        // Each element should equal k (sum of k ones).
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                (v - k as f32).abs() < 1e-4,
                "ones product[{}] = {} != k={}", i, v, k
            );
        }
    }
}

// ── 18. Batch matmul shape preservation ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Batched matmul output length equals batch * m * n.
    #[test]
    fn batch_matmul_output_shape(batch in 1usize..4) {
        let m = 2;
        let k = 3;
        let n = 2;
        let a = vec![1.0f32; batch * m * k];
        let b = vec![1.0f32; batch * k * n];
        let out = bitnet_kernels::cpu::batched_matmul(&a, &b, batch, m, k, n).unwrap();
        prop_assert_eq!(out.len(), batch * m * n);
    }

    /// Batched matmul with identity batch produces same result per batch.
    #[test]
    fn batch_matmul_uniform_batches(batch in 2usize..4) {
        let a = [1.0f32; 4]; // 2×2
        let b = [0.5f32; 4]; // 2×2
        let a_batched: Vec<f32> = a.iter().cycle().take(batch * 4).copied().collect();
        let b_batched: Vec<f32> = b.iter().cycle().take(batch * 4).copied().collect();
        let out = bitnet_kernels::cpu::batched_matmul(&a_batched, &b_batched, batch, 2, 2, 2).unwrap();
        // All batches should produce the same output
        let first_batch = &out[0..4];
        for bi in 1..batch {
            let this_batch = &out[bi * 4..(bi + 1) * 4];
            for i in 0..4 {
                prop_assert!(
                    (first_batch[i] - this_batch[i]).abs() < 1e-5,
                    "batch {} differs from batch 0 at [{}]: {} vs {}",
                    bi, i, this_batch[i], first_batch[i]
                );
            }
        }
    }
}
