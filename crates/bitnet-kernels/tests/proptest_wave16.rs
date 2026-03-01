//! Property-based tests — wave 16.
//!
//! Activation function bounds, layer norm shape preservation, causal mask
//! invariants, RoPE frequency properties, embedding lookup shapes, and
//! reduction kernel consistency.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{
    elu, gelu, gelu_tanh, hard_sigmoid, hard_swish, leaky_relu, mish, quick_gelu, relu, selu,
    sigmoid, silu, softplus, swish, tanh_act,
};
use bitnet_kernels::cpu::attention::causal_mask;
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::rope::{RopeConfig, compute_frequencies};
use proptest::prelude::*;

// ── Activation function bounds ──────────────────────────────────────────────

proptest! {
    /// relu(x) >= 0 for all x.
    #[test]
    fn relu_non_negative(x in -100.0f32..100.0) {
        prop_assert!(relu(x) >= 0.0);
    }

    /// relu is idempotent: relu(relu(x)) == relu(x).
    #[test]
    fn relu_idempotent(x in -100.0f32..100.0) {
        let r = relu(x);
        prop_assert_eq!(relu(r), r);
    }

    /// sigmoid output in (0, 1) for moderate inputs.
    #[test]
    fn sigmoid_bounded(x in -10.0f32..10.0) {
        let s = sigmoid(x);
        prop_assert!(s > 0.0 && s < 1.0,
            "sigmoid({}) = {} not in (0,1)", x, s);
    }

    /// sigmoid is monotonically non-decreasing.
    #[test]
    fn sigmoid_monotone(a in -50.0f32..50.0, b in -50.0f32..50.0) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(sigmoid(lo) <= sigmoid(hi) + 1e-6);
    }

    /// sigmoid(0) ≈ 0.5.
    #[test]
    fn sigmoid_zero_is_half(_dummy in 0u8..1) {
        prop_assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    /// tanh output in (-1, 1) for moderate inputs.
    #[test]
    fn tanh_bounded(x in -8.0f32..8.0) {
        let t = tanh_act(x);
        prop_assert!(t > -1.0 && t < 1.0,
            "tanh({}) = {} not in (-1,1)", x, t);
    }

    /// hard_sigmoid output in [0, 1].
    #[test]
    fn hard_sigmoid_bounded(x in -100.0f32..100.0) {
        let hs = hard_sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&hs),
            "hard_sigmoid({}) = {} not in [0,1]", x, hs);
    }

    /// silu(0) == 0.
    #[test]
    fn silu_zero(_dummy in 0u8..1) {
        prop_assert!((silu(0.0)).abs() < 1e-6);
    }

    /// gelu(x) >= -0.2 (known lower bound approximation).
    #[test]
    fn gelu_lower_bound(x in -100.0f32..100.0) {
        let g = gelu(x);
        prop_assert!(g >= -0.2,
            "gelu({}) = {} below -0.2", x, g);
    }

    /// gelu_tanh(x) >= -0.2.
    #[test]
    fn gelu_tanh_lower_bound(x in -100.0f32..100.0) {
        let g = gelu_tanh(x);
        prop_assert!(g >= -0.2,
            "gelu_tanh({}) = {} below -0.2", x, g);
    }

    /// leaky_relu(x, alpha) for alpha >= 0: positive x unchanged, negative scaled.
    #[test]
    fn leaky_relu_lower_bound(x in -100.0f32..100.0, alpha in 0.0f32..1.0) {
        let lr = leaky_relu(x, alpha);
        if x >= 0.0 {
            prop_assert_eq!(lr, x);
        } else {
            prop_assert!((lr - alpha * x).abs() < 1e-4,
                "leaky_relu({}, {}) = {} != {}", x, alpha, lr, alpha * x);
        }
    }

    /// softplus(x) > 0 for moderate finite x.
    #[test]
    fn softplus_positive(x in -10.0f32..50.0) {
        prop_assert!(softplus(x) > 0.0);
    }

    /// softplus is monotonically non-decreasing.
    #[test]
    fn softplus_monotone(a in -50.0f32..50.0, b in -50.0f32..50.0) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(softplus(lo) <= softplus(hi) + 1e-6);
    }

    /// elu(x, alpha) == x for x >= 0.
    #[test]
    fn elu_positive_identity(x in 0.0f32..100.0, alpha in 0.01f32..2.0) {
        prop_assert_eq!(elu(x, alpha), x);
    }

    /// selu output is finite for bounded input.
    #[test]
    fn selu_finite(x in -50.0f32..50.0) {
        let s = selu(x);
        prop_assert!(s.is_finite(), "selu({}) = {} not finite", x, s);
    }

    /// quick_gelu(0) == 0.
    #[test]
    fn quick_gelu_zero(_dummy in 0u8..1) {
        prop_assert!((quick_gelu(0.0)).abs() < 1e-6);
    }

    /// swish(x, 1.0) == silu(x).
    #[test]
    fn swish_beta1_equals_silu(x in -50.0f32..50.0) {
        let s = swish(x, 1.0);
        let si = silu(x);
        prop_assert!((s - si).abs() < 1e-5,
            "swish({}, 1.0)={} != silu({})={}", x, s, x, si);
    }

    /// mish output is finite for bounded input.
    #[test]
    fn mish_finite(x in -50.0f32..50.0) {
        let m = mish(x);
        prop_assert!(m.is_finite(), "mish({}) = {} not finite", x, m);
    }

    /// hard_swish output is finite.
    #[test]
    fn hard_swish_finite(x in -100.0f32..100.0) {
        let hs = hard_swish(x);
        prop_assert!(hs.is_finite(), "hard_swish({}) not finite", x);
    }
}

// ── Causal mask properties ──────────────────────────────────────────────────

proptest! {
    /// Causal mask has size seq_len².
    #[test]
    fn causal_mask_size(seq_len in 1usize..32) {
        let mask = causal_mask(seq_len);
        prop_assert_eq!(mask.len(), seq_len * seq_len);
    }

    /// Diagonal is always zero (token can attend to itself).
    #[test]
    fn causal_mask_diagonal_zero(seq_len in 1usize..32) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            prop_assert_eq!(mask[i * seq_len + i], 0.0,
                "diagonal position ({},{}) should be 0", i, i);
        }
    }

    /// Lower triangle (i >= j) is always zero.
    #[test]
    fn causal_mask_lower_triangle_zero(seq_len in 1usize..16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in 0..=i {
                prop_assert_eq!(mask[i * seq_len + j], 0.0,
                    "lower triangle ({},{}) should be 0", i, j);
            }
        }
    }

    /// Upper triangle (j > i) is -inf.
    #[test]
    fn causal_mask_upper_triangle_neg_inf(seq_len in 2usize..16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                prop_assert_eq!(mask[i * seq_len + j], f32::NEG_INFINITY,
                    "upper triangle ({},{}) should be -inf", i, j);
            }
        }
    }

    /// Causal mask is idempotent under element-wise min.
    #[test]
    fn causal_mask_idempotent_min(seq_len in 1usize..16) {
        let mask = causal_mask(seq_len);
        let min_mask: Vec<f32> = mask.iter().zip(&mask)
            .map(|(&a, &b)| a.min(b))
            .collect();
        prop_assert_eq!(mask, min_mask);
    }
}

// ── Layer norm shape preservation ───────────────────────────────────────────

proptest! {
    /// layer_norm output has same length as input.
    #[test]
    fn layer_norm_preserves_length(
        batch in 1usize..4,
        dim in 2usize..32,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let output = layer_norm(&input, &gamma, None, &config).unwrap();
        prop_assert_eq!(output.len(), input.len());
    }

    /// layer_norm output is finite for finite input.
    #[test]
    fn layer_norm_output_finite(
        dim in 2usize..16,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 - dim as f32 / 2.0) * 0.5).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let output = layer_norm(&input, &gamma, None, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "output[{}] = {} not finite", i, v);
        }
    }

    /// rms_norm output has same length as input.
    #[test]
    fn rms_norm_preserves_length(
        batch in 1usize..4,
        dim in 2usize..32,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.01).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let output = rms_norm(&input, &gamma, &config).unwrap();
        prop_assert_eq!(output.len(), input.len());
    }

    /// rms_norm with gamma=1 does not change zero input.
    #[test]
    fn rms_norm_zero_input(dim in 2usize..16) {
        let input = vec![0.0f32; dim];
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let output = rms_norm(&input, &gamma, &config).unwrap();
        for &v in &output {
            prop_assert!(v.abs() < 1e-6, "rms_norm(zeros) should be ~0, got {}", v);
        }
    }
}

// ── RoPE frequency properties ───────────────────────────────────────────────

proptest! {
    /// Frequency table length = max_seq_len * head_dim.
    #[test]
    fn rope_freq_length(head_dim in (1usize..8).prop_map(|x| x * 2)) {
        let max_seq_len = 32;
        let config = RopeConfig::new(head_dim, max_seq_len);
        let freqs = compute_frequencies(&config);
        prop_assert_eq!(freqs.len(), max_seq_len * head_dim);
    }

    /// All frequencies are finite.
    #[test]
    fn rope_freq_finite(head_dim in (1usize..8).prop_map(|x| x * 2)) {
        let config = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&config);
        for (i, &f) in freqs.iter().enumerate() {
            prop_assert!(f.is_finite(),
                "freq[{}] = {} not finite", i, f);
        }
    }

    /// Position 0 cosines are 1.0 and sines are 0.0.
    #[test]
    fn rope_freq_position_zero(head_dim in (1usize..8).prop_map(|x| x * 2)) {
        let config = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&config);
        let half_dim = head_dim / 2;
        for i in 0..half_dim {
            let cos_val = freqs[i * 2];
            let sin_val = freqs[i * 2 + 1];
            prop_assert!((cos_val - 1.0).abs() < 1e-5,
                "position 0 cos[{}] = {}, expected 1.0", i, cos_val);
            prop_assert!(sin_val.abs() < 1e-5,
                "position 0 sin[{}] = {}, expected 0.0", i, sin_val);
        }
    }
}

// ── Embedding lookup shape ──────────────────────────────────────────────────

proptest! {
    /// Output length = num_indices * embedding_dim.
    #[test]
    fn embedding_lookup_output_shape(
        vocab in 10usize..100,
        dim in 2usize..16,
        n_idx in 1usize..8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..n_idx).map(|i| (i % vocab) as u32).collect();
        let output = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(output.len(), n_idx * dim);
    }

    /// Out-of-bounds indices produce an error.
    #[test]
    fn embedding_lookup_oob_error(
        vocab in 10usize..50,
        dim in 2usize..8,
    ) {
        let table = vec![0.0f32; vocab * dim];
        let indices = vec![vocab as u32]; // OOB
        prop_assert!(embedding_lookup(&table, &indices, dim).is_err());
    }
}

// ── Reduction kernel properties ─────────────────────────────────────────────

proptest! {
    /// sum of all-ones = length.
    #[test]
    fn reduction_sum_ones(n in 1usize..64) {
        let data = vec![1.0f32; n];
        let sum = ReductionKernel::sum(&data).unwrap();
        prop_assert!((sum - n as f32).abs() < 1e-4,
            "sum of {} ones = {}", n, sum);
    }

    /// mean is bounded by min and max of data.
    #[test]
    fn reduction_mean_bounded(
        data in prop::collection::vec(-100.0f32..100.0, 1..64),
    ) {
        let mean = ReductionKernel::mean(&data).unwrap();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(mean >= min - 1e-4 && mean <= max + 1e-4,
            "mean {} not in [{}, {}]", mean, min, max);
    }

    /// max value >= all elements.
    #[test]
    fn reduction_max_is_upper_bound(
        data in prop::collection::vec(-100.0f32..100.0, 1..64),
    ) {
        let result = ReductionKernel::max(&data).unwrap();
        for &v in &data {
            prop_assert!(result.value >= v - 1e-6);
        }
    }

    /// min value <= all elements.
    #[test]
    fn reduction_min_is_lower_bound(
        data in prop::collection::vec(-100.0f32..100.0, 1..64),
    ) {
        let result = ReductionKernel::min(&data).unwrap();
        for &v in &data {
            prop_assert!(result.value <= v + 1e-6);
        }
    }

    /// l2_norm is non-negative.
    #[test]
    fn reduction_l2_norm_non_negative(
        data in prop::collection::vec(-100.0f32..100.0, 1..64),
    ) {
        let norm = ReductionKernel::l2_norm(&data).unwrap();
        prop_assert!(norm >= 0.0);
    }

    /// l1_norm is non-negative.
    #[test]
    fn reduction_l1_norm_non_negative(
        data in prop::collection::vec(-100.0f32..100.0, 1..64),
    ) {
        let norm = ReductionKernel::l1_norm(&data).unwrap();
        prop_assert!(norm >= 0.0);
    }

    /// l2_norm of zeros is zero.
    #[test]
    fn reduction_l2_norm_zeros(n in 1usize..32) {
        let data = vec![0.0f32; n];
        let norm = ReductionKernel::l2_norm(&data).unwrap();
        prop_assert!(norm.abs() < 1e-6);
    }
}
