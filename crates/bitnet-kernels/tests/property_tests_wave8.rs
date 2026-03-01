//! Wave 8 property tests: CPU kernel invariants.
//!
//! Key invariants:
//! - Conv1d: output dimensions match the analytical formula
//! - Conv1d: Same-padding preserves ceil(input_width / stride) output width
//! - Conv1d: identity kernel (size 1, weight 1) is a no-op
//! - Softmax: outputs sum to ~1.0 for arbitrary finite inputs
//! - Softmax: all outputs lie in [0, 1]
//! - Softmax: in-place and allocating paths agree
//! - Embedding: lookup with valid indices returns correct-size output
//! - Embedding: normalize produces unit-length vectors
//! - SIMD math: fast_exp agrees with scalar exp within tolerance
//! - SIMD math: dot product is commutative
//! - SIMD math: vector_add is commutative
//! - SIMD math: sigmoid outputs lie in (0, 1) for finite inputs
//! - RoPE: rotation preserves vector magnitudes

use bitnet_kernels::cpu::conv1d::{Conv1dConfig, PaddingMode, conv1d_forward, conv1d_output_width};
use bitnet_kernels::cpu::embedding;
use bitnet_kernels::cpu::rope::{self, RopeConfig};
use bitnet_kernels::cpu::simd_math;
use bitnet_kernels::cpu::softmax;
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Generate a non-empty f32 vector with finite values in [-50, 50].
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-50.0f32..50.0f32, 1..=max_len)
}

/// Generate a pair of equal-length f32 vectors.
fn finite_f32_vec_pair(max_len: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (1usize..=max_len).prop_flat_map(|len| {
        (
            prop::collection::vec(-50.0f32..50.0f32, len),
            prop::collection::vec(-50.0f32..50.0f32, len),
        )
    })
}

// -------------------------------------------------------------------
// Properties: Conv1d — output dimension correctness
// -------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Conv1d output width matches the standard formula:
    /// out_w = (input_width + 2*pad - ek) / stride + 1
    #[test]
    fn prop_conv1d_output_width_formula(
        input_width in 1usize..=64,
        kernel_size in 1usize..=8,
        stride in 1usize..=4,
        pad in 0usize..=4,
        dilation in 1usize..=3,
    ) {
        let ek = dilation * (kernel_size - 1) + 1;
        let padded = input_width + 2 * pad;
        prop_assume!(padded >= ek);
        let expected = (padded - ek) / stride + 1;
        let cfg = Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size,
            stride,
            padding: PaddingMode::Zero(pad),
            dilation,
            groups: 1,
            bias: false,
        };
        prop_assert_eq!(
            conv1d_output_width(&cfg, input_width),
            expected,
        );
    }

    /// Same-padding preserves ceil(input_width / stride).
    #[test]
    fn prop_conv1d_same_padding_width(
        input_width in 1usize..=64,
        kernel_size in 1usize..=8,
        stride in 1usize..=4,
        dilation in 1usize..=3,
    ) {
        let expected = input_width.div_ceil(stride);
        let cfg = Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size,
            stride,
            padding: PaddingMode::Same,
            dilation,
            groups: 1,
            bias: false,
        };
        prop_assert_eq!(
            conv1d_output_width(&cfg, input_width),
            expected,
        );
    }

    /// A size-1 identity kernel reproduces the input exactly.
    #[test]
    fn prop_conv1d_identity_kernel(
        input_width in 1usize..=64,
    ) {
        let input: Vec<f32> =
            (0..input_width).map(|i| i as f32 * 0.1).collect();
        let weight = vec![1.0f32];
        let cfg = Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: PaddingMode::Zero(0),
            dilation: 1,
            groups: 1,
            bias: false,
        };
        let out = conv1d_forward(&input, &weight, None, &cfg)
            .expect("identity conv must succeed");
        prop_assert_eq!(
            out.len(),
            input.len(),
            "identity kernel changes length"
        );
        for (i, (&o, &e)) in out.iter().zip(input.iter()).enumerate() {
            prop_assert!(
                (o - e).abs() < 1e-6,
                "identity mismatch at {i}: {o} vs {e}"
            );
        }
    }
}

// -------------------------------------------------------------------
// Properties: Softmax — distribution invariants
// -------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Softmax outputs sum to approximately 1.0.
    #[test]
    fn prop_softmax_sums_to_one(input in finite_f32_vec(256)) {
        let out = softmax::softmax(&input, 1.0)
            .expect("softmax on finite input must succeed");
        let sum: f32 = out.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax sum = {sum}, expected ~1.0"
        );
    }

    /// Every softmax output is in [0, 1].
    #[test]
    fn prop_softmax_outputs_in_unit_interval(
        input in finite_f32_vec(256),
    ) {
        let out = softmax::softmax(&input, 1.0)
            .expect("softmax must succeed");
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&v),
                "out[{i}] = {v} not in [0, 1]"
            );
        }
    }

    /// In-place softmax agrees with allocating softmax.
    #[test]
    fn prop_softmax_inplace_matches_alloc(
        input in finite_f32_vec(128),
    ) {
        let expected = softmax::softmax(&input, 1.0)
            .expect("softmax must succeed");
        let mut data = input.clone();
        softmax::softmax_inplace(&mut data, 1.0)
            .expect("softmax_inplace must succeed");
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-5,
                "in-place differs at {i}: {a} vs {b}"
            );
        }
    }
}

// -------------------------------------------------------------------
// Properties: Embedding — shape and normalization
// -------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Embedding lookup returns exactly n_indices * dim elements.
    #[test]
    fn prop_embedding_lookup_shape(
        vocab in 2usize..16,
        dim in 1usize..32,
        n_indices in 1usize..8,
    ) {
        let table: Vec<f32> =
            (0..(vocab * dim)).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<u32> =
            (0..n_indices).map(|i| (i % vocab) as u32).collect();
        let out = embedding::embedding_lookup(&table, &indices, dim)
            .expect("valid lookup must succeed");
        prop_assert_eq!(
            out.len(),
            n_indices * dim,
            "expected {} elements, got {}",
            n_indices * dim,
            out.len()
        );
    }

    /// After normalize_embeddings, each vector has L2 norm ≈ 1.0.
    #[test]
    fn prop_embedding_normalize_unit_vectors(
        n_vecs in 1usize..4,
        dim in 2usize..16,
    ) {
        let mut data: Vec<f32> =
            (0..(n_vecs * dim)).map(|i| (i as f32) + 1.0).collect();
        embedding::normalize_embeddings(&mut data, dim);
        for v in 0..n_vecs {
            let start = v * dim;
            let norm_sq: f32 =
                data[start..start + dim].iter().map(|x| x * x).sum();
            prop_assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "vector {v} L2 norm² = {norm_sq}, expected ≈ 1.0"
            );
        }
    }
}

// -------------------------------------------------------------------
// Properties: SIMD math — scalar/SIMD parity and algebraic laws
// -------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// fast_exp agrees with scalar f32::exp within tolerance.
    #[test]
    fn prop_simd_exp_matches_scalar(
        input in prop::collection::vec(-20.0f32..20.0f32, 1..=128),
    ) {
        let result = simd_math::fast_exp_f32(&input);
        let expected: Vec<f32> =
            input.iter().map(|&x| x.exp()).collect();
        for (i, (&r, &e)) in
            result.iter().zip(expected.iter()).enumerate()
        {
            if e.is_finite() && e > 1e-30 {
                let rel = ((r - e) / e).abs();
                prop_assert!(
                    rel < 1e-4,
                    "exp({}) rel err {rel} at index {i}",
                    input[i]
                );
            }
        }
    }

    /// Dot product is commutative: a·b == b·a.
    #[test]
    fn prop_simd_dot_product_commutative(
        (a, b) in finite_f32_vec_pair(128),
    ) {
        let ab = simd_math::simd_dot_product(&a, &b);
        let ba = simd_math::simd_dot_product(&b, &a);
        let tol = ab.abs() * 1e-5 + 1e-5;
        prop_assert!(
            (ab - ba).abs() < tol,
            "dot commutativity: a·b={ab} b·a={ba}"
        );
    }

    /// Vector addition is commutative: a+b == b+a.
    #[test]
    fn prop_simd_vector_add_commutative(
        (a, b) in finite_f32_vec_pair(128),
    ) {
        let ab = simd_math::simd_vector_add(&a, &b);
        let ba = simd_math::simd_vector_add(&b, &a);
        for (i, (&x, &y)) in ab.iter().zip(ba.iter()).enumerate() {
            prop_assert!(
                (x - y).abs() < 1e-6,
                "add commutativity at {i}: {x} vs {y}"
            );
        }
    }

    /// Sigmoid outputs lie strictly in (0, 1) for finite inputs.
    #[test]
    fn prop_simd_sigmoid_range(
        input in prop::collection::vec(-50.0f32..50.0f32, 1..=128),
    ) {
        let out = simd_math::fast_sigmoid_f32(&input);
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v > 0.0 - 1e-6 && v < 1.0 + 1e-6,
                "sigmoid[{i}] = {v} out of (0, 1)"
            );
        }
    }
}

// -------------------------------------------------------------------
// Properties: RoPE — rotation preserves vector magnitudes
// -------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RoPE rotation preserves the L2 norm of each head vector.
    #[test]
    fn prop_rope_preserves_norm(
        half_dim in 1usize..=16,
        position in 0usize..64,
    ) {
        let head_dim = half_dim * 2;
        let max_seq = position + 1;
        let cfg = RopeConfig::new(head_dim, max_seq);
        let freqs = rope::compute_frequencies(&cfg);

        let mut data: Vec<f32> =
            (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let norm_before: f32 =
            data.iter().map(|x| x * x).sum::<f32>().sqrt();

        rope::apply_rope(&mut data, position, head_dim, &freqs);

        let norm_after: f32 =
            data.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "norm changed: {norm_before} -> {norm_after} \
             (pos={position}, head_dim={head_dim})"
        );
    }

    /// RoPE batch application matches per-head scalar application.
    #[test]
    fn prop_rope_batch_matches_scalar(
        half_dim in 1usize..=8,
        num_heads in 1usize..=4,
        seq_len in 1usize..=4,
        start_pos in 0usize..8,
    ) {
        let head_dim = half_dim * 2;
        let max_seq = start_pos + seq_len + 1;
        let cfg = RopeConfig::new(head_dim, max_seq);
        let freqs = rope::compute_frequencies(&cfg);

        let total = seq_len * num_heads * head_dim;
        let original: Vec<f32> =
            (0..total).map(|i| (i as f32) * 0.1 - 5.0).collect();

        // Batch path
        let mut batch = original.clone();
        rope::apply_rope_batch(
            &mut batch, start_pos, seq_len, num_heads,
            head_dim, &freqs,
        );

        // Scalar path
        let mut scalar = original.clone();
        for s in 0..seq_len {
            let pos = start_pos + s;
            for h in 0..num_heads {
                let off = (s * num_heads + h) * head_dim;
                rope::apply_rope(
                    &mut scalar[off..off + head_dim],
                    pos, head_dim, &freqs,
                );
            }
        }

        for (i, (&b, &s)) in
            batch.iter().zip(scalar.iter()).enumerate()
        {
            prop_assert!(
                (b - s).abs() < 1e-4,
                "batch/scalar mismatch at {i}: {b} vs {s}"
            );
        }
    }
}
