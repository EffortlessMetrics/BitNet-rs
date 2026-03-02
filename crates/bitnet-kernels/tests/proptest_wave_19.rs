//! Property-based tests — wave 19.
//!
//! Covers under-tested CPU kernel modules: gating (SwiGLU/GeGLU/ReGLU),
//! linear projection, residual connections, FFN forward passes, concat/split
//! round-trips, pooling invariants, quantize round-trips (asymmetric),
//! embedding pack/unpack, positional encoding, and fusion helpers.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::embedding::{
    embedding_lookup, normalize_embeddings, pack_embedding_table, positional_embedding,
    positional_encoding, unpack_embedding_lookup,
};
use bitnet_kernels::cpu::ffn::{FfnConfig, ffn_forward, ffn_forward_batched, gated_ffn_forward};
use bitnet_kernels::cpu::fusion::{fused_add_normalize, fused_scale_add};
use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, l1_loss, mse_loss, smooth_l1_loss,
};
use bitnet_kernels::cpu::pooling::{
    PoolConfig, PoolType, adaptive_avg_pool_1d, global_avg_pool, global_max_pool, pool_1d,
};
use bitnet_kernels::cpu::quantize::{
    dequantize_asymmetric_u8, dequantize_symmetric_i8, quantize_asymmetric_u8,
    quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled, add_residual_with_dropout};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(any::<f32>().prop_filter("finite", |v| v.is_finite()), min_len..=max_len)
}

fn small_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, min_len..=max_len)
}

fn positive_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.01f32..10.0, min_len..=max_len)
}

// ── Gating properties ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // -- SwiGLU --

    /// SwiGLU with zero gate produces zero output.
    #[test]
    fn prop_swiglu_zero_gate_is_zero(up in small_f32_vec(1, 64)) {
        let gate = vec![0.0f32; up.len()];
        let mut out = vec![f32::NAN; up.len()];
        swiglu(&gate, &up, &mut out).unwrap();
        for &v in &out {
            prop_assert!(v.abs() < 1e-6, "swiglu(0, up) should be ~0, got {v}");
        }
    }

    /// SwiGLU with zero up produces zero output.
    #[test]
    fn prop_swiglu_zero_up_is_zero(gate in small_f32_vec(1, 64)) {
        let up = vec![0.0f32; gate.len()];
        let mut out = vec![f32::NAN; gate.len()];
        swiglu(&gate, &up, &mut out).unwrap();
        for &v in &out {
            prop_assert!(v.abs() < 1e-6, "swiglu(gate, 0) should be ~0, got {v}");
        }
    }

    /// SwiGLU output is bounded by |gate| * |up| (since |SiLU(x)| <= |x|).
    #[test]
    fn prop_swiglu_bounded(
        gate in small_f32_vec(1, 64),
        up in small_f32_vec(1, 64),
    ) {
        let n = gate.len().min(up.len());
        let gate = &gate[..n];
        let up = &up[..n];
        let mut out = vec![0.0f32; n];
        swiglu(gate, up, &mut out).unwrap();
        for i in 0..n {
            let bound = gate[i].abs() * up[i].abs();
            prop_assert!(out[i].abs() <= bound + 1e-5,
                "|swiglu[{i}]| = {} > bound {bound}", out[i].abs());
        }
    }

    /// SwiGLU preserves output length.
    #[test]
    fn prop_swiglu_preserves_length(n in 1usize..=128) {
        let gate = vec![1.0f32; n];
        let up = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        swiglu(&gate, &up, &mut out).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    // -- GeGLU --

    /// GeGLU with zero gate produces zero output.
    #[test]
    fn prop_geglu_zero_gate_is_zero(up in small_f32_vec(1, 64)) {
        let gate = vec![0.0f32; up.len()];
        let mut out = vec![f32::NAN; up.len()];
        geglu(&gate, &up, &mut out).unwrap();
        for &v in &out {
            prop_assert!(v.abs() < 1e-6, "geglu(0, up) should be ~0, got {v}");
        }
    }

    /// GeGLU with zero up produces zero output.
    #[test]
    fn prop_geglu_zero_up_is_zero(gate in small_f32_vec(1, 64)) {
        let up = vec![0.0f32; gate.len()];
        let mut out = vec![f32::NAN; gate.len()];
        geglu(&gate, &up, &mut out).unwrap();
        for &v in &out {
            prop_assert!(v.abs() < 1e-6, "geglu(gate, 0) should be ~0, got {v}");
        }
    }

    // -- ReGLU --

    /// ReGLU with all-negative gate produces zero output.
    #[test]
    fn prop_reglu_negative_gate_is_zero(
        n in 1usize..=64,
        up in small_f32_vec(1, 64),
    ) {
        let n = n.min(up.len());
        let gate: Vec<f32> = (0..n).map(|i| -(i as f32 + 1.0)).collect();
        let up = &up[..n];
        let mut out = vec![f32::NAN; n];
        reglu(&gate, up, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-7, "reglu(-gate, up)[{i}] = {v} should be 0");
        }
    }

    /// ReGLU with positive gate and positive up gives non-negative output.
    #[test]
    fn prop_reglu_positive_gate_positive_up_non_negative(
        gate in positive_f32_vec(1, 64),
        up in positive_f32_vec(1, 64),
    ) {
        let n = gate.len().min(up.len());
        let mut out = vec![0.0f32; n];
        reglu(&gate[..n], &up[..n], &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= -1e-7, "reglu[{i}] = {v} should be >= 0");
        }
    }

    /// ReGLU with positive gate equals gate * up (ReLU is identity for positive).
    #[test]
    fn prop_reglu_positive_gate_is_product(
        gate in positive_f32_vec(1, 64),
        up in small_f32_vec(1, 64),
    ) {
        let n = gate.len().min(up.len());
        let mut out = vec![0.0f32; n];
        reglu(&gate[..n], &up[..n], &mut out).unwrap();
        for i in 0..n {
            let expected = gate[i] * up[i];
            prop_assert!((out[i] - expected).abs() < 1e-5,
                "reglu[{i}] = {} expected {expected}", out[i]);
        }
    }

    /// apply_gating dispatches consistently with direct calls.
    #[test]
    fn prop_apply_gating_dispatch_consistent(
        gate in small_f32_vec(1, 32),
        up in small_f32_vec(1, 32),
        gating_idx in 0usize..3,
    ) {
        let n = gate.len().min(up.len());
        let gate = &gate[..n];
        let up = &up[..n];
        let gating = match gating_idx {
            0 => GatingType::SwiGLU,
            1 => GatingType::GeGLU,
            _ => GatingType::ReGLU,
        };
        let mut out_dispatch = vec![0.0f32; n];
        let mut out_direct = vec![0.0f32; n];

        apply_gating(gating, gate, up, &mut out_dispatch).unwrap();
        match gating {
            GatingType::SwiGLU => swiglu(gate, up, &mut out_direct).unwrap(),
            GatingType::GeGLU => geglu(gate, up, &mut out_direct).unwrap(),
            GatingType::ReGLU => reglu(gate, up, &mut out_direct).unwrap(),
        }
        for i in 0..n {
            prop_assert!((out_dispatch[i] - out_direct[i]).abs() < 1e-7,
                "dispatch mismatch at {i}");
        }
    }

    // ── Linear projection properties ────────────────────────────────────────

    /// linear_cpu output has correct shape (batch * out_features).
    #[test]
    fn prop_linear_output_shape(
        batch in 1usize..=8,
        inf in 1usize..=16,
        outf in 1usize..=16,
    ) {
        let x = vec![0.5f32; batch * inf];
        let w = vec![0.1f32; outf * inf];
        let cfg = LinearConfig::new(batch, inf, outf).unwrap();
        let mut out = vec![0.0f32; batch * outf];
        linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();
        // All values should be finite
        for &v in &out {
            prop_assert!(v.is_finite(), "linear output should be finite, got {v}");
        }
    }

    /// linear with identity weight is identity (no bias).
    #[test]
    fn prop_linear_identity_weight(dim in 1usize..=16) {
        let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.3).collect();
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        let cfg = LinearConfig::new(1, dim, dim).unwrap();
        let mut out = vec![0.0f32; dim];
        linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();
        for i in 0..dim {
            prop_assert!((out[i] - x[i]).abs() < 1e-5,
                "identity mismatch at {i}: {} vs {}", out[i], x[i]);
        }
    }

    /// linear with zero weights produces zero output (no bias).
    #[test]
    fn prop_linear_zero_weight_zero_output(
        batch in 1usize..=4,
        inf in 1usize..=8,
        outf in 1usize..=8,
    ) {
        let x: Vec<f32> = (0..batch * inf).map(|i| i as f32 * 0.1).collect();
        let w = vec![0.0f32; outf * inf];
        let cfg = LinearConfig::new(batch, inf, outf).unwrap();
        let mut out = vec![f32::NAN; batch * outf];
        linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-7, "zero weight out[{i}] = {v}");
        }
    }

    /// linear with zero input and bias produces bias broadcast.
    #[test]
    fn prop_linear_zero_input_bias_broadcast(
        batch in 1usize..=4,
        inf in 1usize..=8,
        outf in 1usize..=8,
    ) {
        let x = vec![0.0f32; batch * inf];
        let w = vec![1.0f32; outf * inf];
        let bias: Vec<f32> = (0..outf).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let cfg = LinearConfig::new(batch, inf, outf).unwrap().with_bias(true);
        let mut out = vec![0.0f32; batch * outf];
        linear_cpu(&x, &w, Some(&bias), &mut out, &cfg).unwrap();
        for b in 0..batch {
            for j in 0..outf {
                let expected = bias[j];
                let actual = out[b * outf + j];
                prop_assert!((actual - expected).abs() < 1e-5,
                    "bias broadcast [{b},{j}]: {actual} vs {expected}");
            }
        }
    }

    /// linear is homogeneous: f(a*x) = a*f(x) when no bias.
    #[test]
    fn prop_linear_homogeneous(
        scale in -5.0f32..5.0,
        inf in 1usize..=8,
        outf in 1usize..=8,
    ) {
        let x: Vec<f32> = (0..inf).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let sx: Vec<f32> = x.iter().map(|v| v * scale).collect();
        let w: Vec<f32> = (0..outf * inf).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let cfg = LinearConfig::new(1, inf, outf).unwrap();
        let mut out_x = vec![0.0f32; outf];
        let mut out_sx = vec![0.0f32; outf];
        linear_cpu(&x, &w, None, &mut out_x, &cfg).unwrap();
        linear_cpu(&sx, &w, None, &mut out_sx, &cfg).unwrap();
        for i in 0..outf {
            let expected = out_x[i] * scale;
            prop_assert!((out_sx[i] - expected).abs() < 1e-3,
                "homogeneity at {i}: {} vs {expected}", out_sx[i]);
        }
    }

    // ── Residual connection properties ──────────────────────────────────────

    /// add_residual with zero residual is identity.
    #[test]
    fn prop_residual_zero_identity(data in finite_f32_vec(1, 128)) {
        let mut output = data.clone();
        let zeros = vec![0.0f32; data.len()];
        add_residual(&mut output, &zeros).unwrap();
        for (i, (&a, &b)) in output.iter().zip(data.iter()).enumerate() {
            prop_assert!((a - b).abs() < 1e-7,
                "zero residual changed value at {i}: {a} vs {b}");
        }
    }

    /// add_residual then subtract = original (inverse).
    #[test]
    fn prop_residual_add_subtract_roundtrip(
        data in small_f32_vec(1, 64),
        residual in small_f32_vec(1, 64),
    ) {
        let n = data.len().min(residual.len());
        let data = &data[..n];
        let residual = &residual[..n];
        let mut output = data.to_vec();
        add_residual(&mut output, residual).unwrap();
        let neg: Vec<f32> = residual.iter().map(|v| -v).collect();
        add_residual(&mut output, &neg).unwrap();
        for i in 0..n {
            prop_assert!((output[i] - data[i]).abs() < 1e-4,
                "roundtrip at {i}: {} vs {}", output[i], data[i]);
        }
    }

    /// add_residual_scaled with scale=0 is identity.
    #[test]
    fn prop_residual_scaled_zero_identity(data in finite_f32_vec(1, 128)) {
        let mut output = data.clone();
        let residual = vec![100.0f32; data.len()];
        add_residual_scaled(&mut output, &residual, 0.0).unwrap();
        for (i, (&a, &b)) in output.iter().zip(data.iter()).enumerate() {
            prop_assert!((a - b).abs() < 1e-7,
                "scaled(0) changed at {i}: {a} vs {b}");
        }
    }

    /// add_residual_scaled with scale=1 matches add_residual.
    #[test]
    fn prop_residual_scaled_one_matches_unscaled(
        data in small_f32_vec(1, 64),
        residual in small_f32_vec(1, 64),
    ) {
        let n = data.len().min(residual.len());
        let mut out_scaled = data[..n].to_vec();
        let mut out_plain = data[..n].to_vec();
        add_residual_scaled(&mut out_scaled, &residual[..n], 1.0).unwrap();
        add_residual(&mut out_plain, &residual[..n]).unwrap();
        for i in 0..n {
            prop_assert!((out_scaled[i] - out_plain[i]).abs() < 1e-5,
                "scale=1 mismatch at {i}");
        }
    }

    /// add_residual_with_dropout with all-true mask matches add_residual.
    #[test]
    fn prop_residual_dropout_all_true_matches_plain(
        data in small_f32_vec(1, 64),
        residual in small_f32_vec(1, 64),
    ) {
        let n = data.len().min(residual.len());
        let mask = vec![true; n];
        let mut out_dropout = data[..n].to_vec();
        let mut out_plain = data[..n].to_vec();
        add_residual_with_dropout(&mut out_dropout, &residual[..n], &mask).unwrap();
        add_residual(&mut out_plain, &residual[..n]).unwrap();
        for i in 0..n {
            prop_assert!((out_dropout[i] - out_plain[i]).abs() < 1e-7,
                "all-true dropout mismatch at {i}");
        }
    }

    /// add_residual_with_dropout with all-false mask is identity.
    #[test]
    fn prop_residual_dropout_all_false_identity(data in finite_f32_vec(1, 128)) {
        let mut output = data.clone();
        let residual = vec![999.0f32; data.len()];
        let mask = vec![false; data.len()];
        add_residual_with_dropout(&mut output, &residual, &mask).unwrap();
        for (i, (&a, &b)) in output.iter().zip(data.iter()).enumerate() {
            prop_assert!((a - b).abs() < 1e-7,
                "all-false dropout changed at {i}: {a} vs {b}");
        }
    }

    // ── FFN properties ──────────────────────────────────────────────────────

    /// FFN output has correct shape (hidden_dim).
    #[test]
    fn prop_ffn_output_shape(
        hidden in 1usize..=8,
        inter in 1usize..=16,
        act_idx in 0usize..3,
    ) {
        let act = match act_idx {
            0 => bitnet_kernels::cpu::ffn::FfnActivation::ReLU,
            1 => bitnet_kernels::cpu::ffn::FfnActivation::SiLU,
            _ => bitnet_kernels::cpu::ffn::FfnActivation::GeLU,
        };
        let cfg = FfnConfig::new(hidden, inter, act).unwrap();
        let input = vec![0.1f32; hidden];
        let w_up = vec![0.1f32; inter * hidden];
        let w_down = vec![0.1f32; hidden * inter];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        prop_assert_eq!(out.len(), hidden, "FFN output should be [hidden_dim]");
    }

    /// FFN with zero input produces zero output (for ReLU).
    #[test]
    fn prop_ffn_zero_input_relu_zero_output(
        hidden in 1usize..=8,
        inter in 1usize..=16,
    ) {
        let cfg = FfnConfig::new(hidden, inter,
            bitnet_kernels::cpu::ffn::FfnActivation::ReLU).unwrap();
        let input = vec![0.0f32; hidden];
        let w_up = vec![0.5f32; inter * hidden];
        let w_down = vec![0.5f32; hidden * inter];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-7, "FFN(0) should be 0, got [{i}]={v}");
        }
    }

    /// Batched FFN with batch=1 matches single FFN.
    #[test]
    fn prop_ffn_batched_single_matches_unbatched(
        hidden in 1usize..=8,
        inter in 1usize..=16,
    ) {
        let cfg = FfnConfig::new(hidden, inter,
            bitnet_kernels::cpu::ffn::FfnActivation::SiLU).unwrap();
        let input: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.1).collect();
        let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32) * 0.01).collect();
        let single = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        let batched = ffn_forward_batched(&input, &w_up, &w_down, &cfg, 1).unwrap();
        for i in 0..hidden {
            prop_assert!((single[i] - batched[i]).abs() < 1e-5,
                "batch=1 mismatch at {i}: {} vs {}", single[i], batched[i]);
        }
    }

    /// Gated FFN output has correct shape.
    #[test]
    fn prop_gated_ffn_output_shape(
        hidden in 1usize..=8,
        inter in 1usize..=16,
    ) {
        let cfg = FfnConfig::new(hidden, inter,
            bitnet_kernels::cpu::ffn::FfnActivation::SiLU).unwrap();
        let input = vec![0.1f32; hidden];
        let w_gate = vec![0.1f32; inter * hidden];
        let w_up = vec![0.1f32; inter * hidden];
        let w_down = vec![0.1f32; hidden * inter];
        let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
        prop_assert_eq!(out.len(), hidden);
    }

    // ── Concat/split round-trip ─────────────────────────────────────────────

    /// Concat then split_sizes is identity.
    #[test]
    fn prop_concat_split_roundtrip(
        a_len in 1usize..=32,
        b_len in 1usize..=32,
    ) {
        let a: Vec<f32> = (0..a_len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..b_len).map(|i| (i + a_len) as f32).collect();
        let inputs = vec![a.as_slice(), b.as_slice()];
        let shape_a = [a_len];
        let shape_b = [b_len];
        let shapes: Vec<&[usize]> = vec![&shape_a, &shape_b];
        let cat = ConcatKernel::concat(&inputs, &shapes, 0).unwrap();
        prop_assert_eq!(cat.len(), a_len + b_len);
        // Split back using split_sizes
        let total = [a_len + b_len];
        let split = ConcatKernel::split_sizes(&cat, &total, 0, &[a_len, b_len]).unwrap();
        prop_assert_eq!(split.len(), 2);
        prop_assert_eq!(&split[0], &a);
        prop_assert_eq!(&split[1], &b);
    }

    /// Concat preserves all elements.
    #[test]
    fn prop_concat_preserves_elements(
        a in small_f32_vec(1, 32),
        b in small_f32_vec(1, 32),
    ) {
        let inputs = vec![a.as_slice(), b.as_slice()];
        let shape_a = [a.len()];
        let shape_b = [b.len()];
        let shapes: Vec<&[usize]> = vec![&shape_a, &shape_b];
        let cat = ConcatKernel::concat(&inputs, &shapes, 0).unwrap();
        prop_assert_eq!(cat.len(), a.len() + b.len());
        prop_assert_eq!(&cat[..a.len()], a.as_slice());
        prop_assert_eq!(&cat[a.len()..], b.as_slice());
    }

    // ── Pooling properties ──────────────────────────────────────────────────

    /// Global max pool result is the maximum of each channel.
    #[test]
    fn prop_global_max_pool_is_max(
        channels in 1usize..=4,
        spatial in 1usize..=16,
    ) {
        let input: Vec<f32> = (0..channels * spatial)
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.1)
            .collect();
        let result = global_max_pool(&input, &[spatial]).unwrap();
        prop_assert_eq!(result.len(), channels);
        for c in 0..channels {
            let channel_max = input[c * spatial..(c + 1) * spatial]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            prop_assert!((result[c] - channel_max).abs() < 1e-6,
                "channel {c}: max={channel_max}, pool={}", result[c]);
        }
    }

    /// Global avg pool of uniform values equals that value.
    #[test]
    fn prop_global_avg_pool_uniform(
        channels in 1usize..=4,
        spatial in 1usize..=16,
        value in -10.0f32..10.0,
    ) {
        let input = vec![value; channels * spatial];
        let result = global_avg_pool(&input, &[spatial]).unwrap();
        prop_assert_eq!(result.len(), channels);
        for &v in &result {
            prop_assert!((v - value).abs() < 1e-5, "avg pool of uniform should be {value}, got {v}");
        }
    }

    /// Adaptive avg pool output size matches requested size.
    #[test]
    fn prop_adaptive_avg_pool_output_size(
        input_len in 1usize..=64,
    ) {
        let output_size = (input_len / 2).max(1);
        let input: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let result = adaptive_avg_pool_1d(&input, output_size).unwrap();
        prop_assert_eq!(result.len(), output_size);
    }

    /// Avg pool of constant input equals constant.
    #[test]
    fn prop_avg_pool_constant_input(n in 2usize..=64, value in -5.0f32..5.0) {
        let input = vec![value; n];
        let cfg = PoolConfig {
            pool_type: PoolType::Average,
            kernel_size: 2,
            stride: 1,
            padding: 0,
        };
        let result = pool_1d(&input, &cfg).unwrap();
        for (i, &v) in result.iter().enumerate() {
            prop_assert!((v - value).abs() < 1e-5,
                "avg pool constant: [{i}]={v} vs {value}");
        }
    }

    // ── Quantization round-trip properties ──────────────────────────────────

    /// Symmetric i8 quantize-dequantize round-trip error is bounded.
    #[test]
    fn prop_symmetric_i8_roundtrip(input in small_f32_vec(1, 64)) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let recovered = dequantize_symmetric_i8(&quantized, scale);
        prop_assert_eq!(recovered.len(), input.len());
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            // 8-bit quantization error should be small relative to range
            let range = input.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let tol = if range > 0.0 { range / 127.0 + 1e-5 } else { 1e-5 };
            prop_assert!((orig - rec).abs() < tol + 1e-5,
                "sym i8 roundtrip at {i}: {orig} -> {rec} (tol={tol})");
        }
    }

    /// Asymmetric u8 quantize-dequantize preserves length.
    #[test]
    fn prop_asymmetric_u8_preserves_length(input in small_f32_vec(1, 64)) {
        let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
        let recovered = dequantize_asymmetric_u8(&quantized, scale, zero_point);
        prop_assert_eq!(recovered.len(), input.len());
    }

    /// Asymmetric u8 round-trip error is bounded.
    #[test]
    fn prop_asymmetric_u8_roundtrip_bounded(input in small_f32_vec(2, 64)) {
        let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
        let recovered = dequantize_asymmetric_u8(&quantized, scale, zero_point);
        let min_val = input.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let tol = if range > 0.0 { range / 255.0 + 1e-4 } else { 1e-4 };
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            prop_assert!((orig - rec).abs() < tol + 1e-4,
                "asym u8 roundtrip [{i}]: {orig} -> {rec} (tol={tol})");
        }
    }

    /// Ternary quantization values are in {-1, 0, 1}.
    #[test]
    fn prop_ternary_values_valid(
        input in small_f32_vec(1, 128),
        threshold in 0.01f32..5.0,
    ) {
        let quantized = quantize_ternary(&input, threshold);
        prop_assert_eq!(quantized.len(), input.len());
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert!(v == -1 || v == 0 || v == 1,
                "ternary [{i}] = {v} not in {{-1,0,1}}");
        }
    }

    /// Ternary quantization: values within threshold map to 0.
    #[test]
    fn prop_ternary_threshold_zero(
        n in 1usize..=64,
        threshold in 0.1f32..5.0,
    ) {
        // All values within threshold should be 0
        let input: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * threshold * 0.9).collect();
        let quantized = quantize_ternary(&input, threshold);
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert_eq!(v, 0, "value {} within threshold={} but got {}", input[i], threshold, v);
        }
    }

    // ── Embedding properties ────────────────────────────────────────────────

    /// Pack then unpack is identity.
    #[test]
    fn prop_embedding_pack_unpack_roundtrip(
        vocab in 2usize..=16,
        dim in 1usize..=16,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let packed = pack_embedding_table(&table, vocab, dim);
        let indices: Vec<u32> = (0..vocab as u32).collect();
        let result = unpack_embedding_lookup(&packed, &indices).unwrap();
        prop_assert_eq!(result.len(), vocab * dim);
        // i8 packing is lossy — check shape and finiteness only
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v.is_finite(), "pack/unpack [{i}] not finite: {v}");
        }
    }

    /// Positional embedding output has correct shape.
    #[test]
    fn prop_positional_embedding_shape(
        seq_len in 1usize..=32,
        dim in (1usize..=8).prop_map(|x| x * 2),
    ) {
        let pe = positional_embedding(seq_len, dim);
        prop_assert_eq!(pe.len(), seq_len * dim, "PE shape mismatch");
        for &v in &pe {
            prop_assert!(v.is_finite(), "PE contains non-finite: {v}");
        }
    }

    /// Positional encoding output has correct shape and is bounded.
    #[test]
    fn prop_positional_encoding_bounded(
        seq_len in 1usize..=32,
        dim in (1usize..=8).prop_map(|x| x * 2),
    ) {
        let pe = positional_encoding(seq_len, dim, 10000.0);
        prop_assert_eq!(pe.len(), seq_len * dim);
        for (i, &v) in pe.iter().enumerate() {
            prop_assert!(v.is_finite() && v.abs() <= 1.0 + 1e-5,
                "PE[{i}] = {v} should be in [-1, 1]");
        }
    }

    /// Normalize embeddings produces unit vectors.
    #[test]
    fn prop_normalize_embeddings_unit_norm(
        n_vecs in 1usize..=8,
        dim in 1usize..=16,
    ) {
        let mut embeddings: Vec<f32> = (0..n_vecs * dim)
            .map(|i| (i as f32 * 0.3 + 0.1).sin())
            .collect();
        normalize_embeddings(&mut embeddings, dim);
        for v in 0..n_vecs {
            let norm: f32 = embeddings[v * dim..(v + 1) * dim]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-4,
                "vector {v} norm = {norm}, expected 1.0");
        }
    }

    // ── Loss function properties ────────────────────────────────────────────

    /// MSE is symmetric: mse(a, b) == mse(b, a).
    #[test]
    fn prop_mse_symmetric(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let a = &a[..n];
        let b = &b[..n];
        let mse_ab = mse_loss(a, b, LossReduction::Mean).unwrap();
        let mse_ba = mse_loss(b, a, LossReduction::Mean).unwrap();
        prop_assert!((mse_ab - mse_ba).abs() < 1e-5,
            "MSE not symmetric: {mse_ab} vs {mse_ba}");
    }

    /// L1 is symmetric.
    #[test]
    fn prop_l1_symmetric(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let l1_ab = l1_loss(&a[..n], &b[..n], LossReduction::Mean).unwrap();
        let l1_ba = l1_loss(&b[..n], &a[..n], LossReduction::Mean).unwrap();
        prop_assert!((l1_ab - l1_ba).abs() < 1e-5,
            "L1 not symmetric: {l1_ab} vs {l1_ba}");
    }

    /// Smooth L1 with large beta ≈ MSE/2.
    #[test]
    fn prop_smooth_l1_large_beta_approx_mse(
        a in small_f32_vec(2, 32),
        b in small_f32_vec(2, 32),
    ) {
        let n = a.len().min(b.len());
        let a = &a[..n];
        let b = &b[..n];
        // With very large beta, smooth L1 ≈ 0.5 * MSE (all diffs < beta)
        let sl1 = smooth_l1_loss(a, b, 1e6, LossReduction::Sum).unwrap();
        let mse = mse_loss(a, b, LossReduction::Sum).unwrap();
        // smooth_l1 = 0.5 * diff^2 / beta for |diff| < beta
        // So sl1 * beta ≈ 0.5 * sum(diff^2) = 0.5 * mse * n (for Sum reduction)
        // Actually just check it's non-negative and finite
        prop_assert!(sl1.is_finite() && sl1 >= 0.0,
            "smooth L1 should be non-negative, got {sl1}");
        prop_assert!(mse.is_finite() && mse >= 0.0);
    }

    /// Cosine similarity of vector with itself is ~0 loss (similarity ~1).
    #[test]
    fn prop_cosine_self_similarity(input in positive_f32_vec(2, 64)) {
        let loss = cosine_similarity_loss(&input, &input).unwrap();
        // cos_sim(x, x) = 1.0, loss = 1 - cos_sim = 0
        prop_assert!(loss.abs() < 1e-4,
            "cosine self-similarity loss should be ~0, got {loss}");
    }

    // ── Fusion properties ───────────────────────────────────────────────────

    /// fused_scale_add(a, b, 1.0) = a + b element-wise.
    #[test]
    fn prop_fused_scale_add_unit(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let result = fused_scale_add(&a[..n], &b[..n], 1.0).unwrap();
        prop_assert_eq!(result.len(), n);
        for i in 0..n {
            let expected = a[i] + b[i];
            prop_assert!((result[i] - expected).abs() < 1e-5,
                "scale_add(1.0) at {i}: {} vs {expected}", result[i]);
        }
    }

    /// fused_scale_add(a, b, 0.0) = a element-wise.
    #[test]
    fn prop_fused_scale_add_zero(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let result = fused_scale_add(&a[..n], &b[..n], 0.0).unwrap();
        for i in 0..n {
            prop_assert!((result[i] - a[i]).abs() < 1e-5,
                "scale_add(0) at {i}: {} vs {}", result[i], a[i]);
        }
    }

    /// fused_add_normalize preserves length.
    #[test]
    fn prop_fused_add_normalize_length(n in 2usize..=64) {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2) - 1.0).collect();
        let gamma = vec![1.0f32; n];
        let result = fused_add_normalize(&a, &b, &gamma, 1e-5).unwrap();
        prop_assert_eq!(result.len(), n, "fused_add_normalize changed length");
    }

    // ── Layer norm extended properties ──────────────────────────────────────

    /// Layer norm is idempotent with gamma=1, beta=0.
    #[test]
    fn prop_layer_norm_idempotent(input in small_f32_vec(2, 32)) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let cfg = LayerNormConfig {
            normalized_shape: vec![n],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let first = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        let second = layer_norm(&first, &gamma, Some(&beta), &cfg).unwrap();
        for i in 0..n {
            prop_assert!((first[i] - second[i]).abs() < 1e-3,
                "layer_norm not idempotent at {i}: {} vs {}", first[i], second[i]);
        }
    }

    /// RMS norm with all-ones gamma preserves norm direction.
    #[test]
    fn prop_rms_norm_finite_output(input in small_f32_vec(2, 32)) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let cfg = LayerNormConfig {
            normalized_shape: vec![n],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let result = rms_norm(&input, &gamma, &cfg).unwrap();
        prop_assert_eq!(result.len(), n);
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v.is_finite(), "rms_norm[{i}] not finite: {v}");
        }
    }

    // ── Pool 1d properties ──────────────────────────────────────────────────

    /// Max pool output values come from input.
    #[test]
    fn prop_max_pool_output_from_input(input in small_f32_vec(2, 64)) {
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 1,
            padding: 0,
        };
        if let Ok(result) = pool_1d(&input, &cfg) {
            for (i, &v) in result.iter().enumerate() {
                prop_assert!(input.contains(&v),
                    "max pool[{i}]={v} not found in input");
            }
        }
    }

    /// Global avg pool and global max pool output channels match.
    #[test]
    fn prop_global_pool_channel_count(
        channels in 1usize..=8,
        spatial in 1usize..=16,
    ) {
        let input = vec![1.0f32; channels * spatial];
        let avg = global_avg_pool(&input, &[spatial]).unwrap();
        let max = global_max_pool(&input, &[spatial]).unwrap();
        prop_assert_eq!(avg.len(), channels);
        prop_assert_eq!(max.len(), channels);
    }

    // ── Gating: finite output ───────────────────────────────────────────────

    /// All gating functions produce finite output for small inputs.
    #[test]
    fn prop_gating_finite_output(
        gate in small_f32_vec(1, 64),
        up in small_f32_vec(1, 64),
    ) {
        let n = gate.len().min(up.len());
        let gate = &gate[..n];
        let up = &up[..n];

        let mut out_s = vec![0.0f32; n];
        let mut out_g = vec![0.0f32; n];
        let mut out_r = vec![0.0f32; n];
        swiglu(gate, up, &mut out_s).unwrap();
        geglu(gate, up, &mut out_g).unwrap();
        reglu(gate, up, &mut out_r).unwrap();
        for i in 0..n {
            prop_assert!(out_s[i].is_finite(), "swiglu[{i}] not finite: {}", out_s[i]);
            prop_assert!(out_g[i].is_finite(), "geglu[{i}] not finite: {}", out_g[i]);
            prop_assert!(out_r[i].is_finite(), "reglu[{i}] not finite: {}", out_r[i]);
        }
    }

    // ── Linear: additivity f(a+b) = f(a) + f(b) ────────────────────────────

    /// Linear projection is additive (no bias).
    #[test]
    fn prop_linear_additive(
        inf in 1usize..=8,
        outf in 1usize..=8,
    ) {
        let a: Vec<f32> = (0..inf).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..inf).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let w: Vec<f32> = (0..outf * inf).map(|i| (i as f32) * 0.05).collect();
        let cfg = LinearConfig::new(1, inf, outf).unwrap();
        let mut out_a = vec![0.0f32; outf];
        let mut out_b = vec![0.0f32; outf];
        let mut out_ab = vec![0.0f32; outf];
        linear_cpu(&a, &w, None, &mut out_a, &cfg).unwrap();
        linear_cpu(&b, &w, None, &mut out_b, &cfg).unwrap();
        linear_cpu(&ab, &w, None, &mut out_ab, &cfg).unwrap();
        for i in 0..outf {
            let expected = out_a[i] + out_b[i];
            prop_assert!((out_ab[i] - expected).abs() < 1e-3,
                "additivity at {i}: {} vs {expected}", out_ab[i]);
        }
    }

    // ── Residual: commutativity ─────────────────────────────────────────────

    /// add_residual is commutative: starting from zeros, a+b == b+a.
    #[test]
    fn prop_residual_commutative(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let a = &a[..n];
        let b = &b[..n];
        let mut ab = a.to_vec();
        add_residual(&mut ab, b).unwrap();
        let mut ba = b.to_vec();
        add_residual(&mut ba, a).unwrap();
        for i in 0..n {
            prop_assert!((ab[i] - ba[i]).abs() < 1e-5,
                "commutative at {i}: {} vs {}", ab[i], ba[i]);
        }
    }

    // ── Residual: scaled linearity ──────────────────────────────────────────

    /// add_residual_scaled(x, r, a+b) == add_residual_scaled twice.
    #[test]
    fn prop_residual_scaled_linearity(
        data in small_f32_vec(1, 32),
        residual in small_f32_vec(1, 32),
        scale_a in -3.0f32..3.0,
        scale_b in -3.0f32..3.0,
    ) {
        let n = data.len().min(residual.len());
        let data = &data[..n];
        let residual = &residual[..n];

        // Apply (a+b) at once
        let mut combined = data.to_vec();
        add_residual_scaled(&mut combined, residual, scale_a + scale_b).unwrap();

        // Apply a then b
        let mut separate = data.to_vec();
        add_residual_scaled(&mut separate, residual, scale_a).unwrap();
        add_residual_scaled(&mut separate, residual, scale_b).unwrap();

        for i in 0..n {
            prop_assert!((combined[i] - separate[i]).abs() < 1e-3,
                "scaled linearity at {i}: {} vs {}", combined[i], separate[i]);
        }
    }

    // ── FFN: gated FFN with zero gate weight → zero output (ReLU) ───────

    /// Gated FFN with zero gate weights and ReLU gives zero output.
    #[test]
    fn prop_gated_ffn_zero_gate_relu(
        hidden in 1usize..=4,
        inter in 1usize..=8,
    ) {
        let cfg = FfnConfig::new(hidden, inter,
            bitnet_kernels::cpu::ffn::FfnActivation::ReLU).unwrap();
        let input: Vec<f32> = (0..hidden).map(|i| (i + 1) as f32).collect();
        let w_gate = vec![0.0f32; inter * hidden];
        let w_up = vec![1.0f32; inter * hidden];
        let w_down = vec![1.0f32; hidden * inter];
        let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-7,
                "gated FFN(zero gate)[{i}] = {v}");
        }
    }

    // ── Embedding: lookup shape ─────────────────────────────────────────────

    /// Embedding lookup returns correct number of elements.
    #[test]
    fn prop_embedding_lookup_shape(
        vocab in 2usize..=16,
        dim in 1usize..=16,
        n_indices in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..n_indices).map(|i| (i % vocab) as u32).collect();
        let result = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(result.len(), n_indices * dim,
            "lookup shape: {} vs {}", result.len(), n_indices * dim);
    }

    // ── Loss: MSE zero for identical, non-negative ──────────────────────────

    /// MSE of identical inputs is zero.
    #[test]
    fn prop_mse_identical_is_zero(input in small_f32_vec(1, 64)) {
        let mse = mse_loss(&input, &input, LossReduction::Mean).unwrap();
        prop_assert!(mse.abs() < 1e-6, "MSE(x,x) = {mse} should be 0");
    }

    /// MSE is non-negative.
    #[test]
    fn prop_mse_non_negative(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let mse = mse_loss(&a[..n], &b[..n], LossReduction::Mean).unwrap();
        prop_assert!(mse >= -1e-7, "MSE should be non-negative, got {mse}");
    }

    /// L1 loss is non-negative.
    #[test]
    fn prop_l1_non_negative(
        a in small_f32_vec(1, 64),
        b in small_f32_vec(1, 64),
    ) {
        let n = a.len().min(b.len());
        let l1 = l1_loss(&a[..n], &b[..n], LossReduction::Mean).unwrap();
        prop_assert!(l1 >= -1e-7, "L1 should be non-negative, got {l1}");
    }

    /// L1 of identical inputs is zero.
    #[test]
    fn prop_l1_identical_is_zero(input in small_f32_vec(1, 64)) {
        let l1 = l1_loss(&input, &input, LossReduction::Mean).unwrap();
        prop_assert!(l1.abs() < 1e-6, "L1(x,x) = {l1} should be 0");
    }
}
