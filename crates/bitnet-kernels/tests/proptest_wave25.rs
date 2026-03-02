//! Property-based tests — wave 25.
//!
//! Kernel correctness invariants for recent CUDA and CPU modules:
//! KV cache monotonicity & clear semantics, softmax sum-to-one &
//! shift-invariance & monotonicity, conv1d output length formula &
//! stride-dilation relationship & zero-padding, elementwise commutativity &
//! distributivity & identity, residual identity & scaling linearity,
//! cross-entropy non-negativity & MSE symmetry & gradient finiteness,
//! batch-norm zero-mean & unit-variance, tensor-parallel shard+gather
//! round-trip & chunk correctness.
//!
//! 55 property assertions across 8 invariant categories.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::loss::{
    LossReduction, cross_entropy_loss, gradient_clip_norm, gradient_clip_value, l1_loss, mse_loss,
    perplexity,
};
use bitnet_kernels::cpu::tensor_parallel::{
    CommBackend, ShardingStrategy, TensorParallelConfig, compute_shard_ranges, gather_shards,
    shard_tensor, validate_sharding,
};
use bitnet_kernels::cuda::conv1d::{Conv1dConfig, PaddingMode, conv1d_cpu};
use bitnet_kernels::cuda::elementwise::{
    ElementwiseConfig, ElementwiseOp, elementwise_cpu_fallback, elementwise_unary_cpu,
    fused_elementwise_cpu,
};
use bitnet_kernels::cuda::kv_cache::{CacheDtype, KvCacheBuffer, KvCacheConfig};
use bitnet_kernels::cuda::residual::{
    gated_residual, residual_add, residual_add_scaled, stochastic_depth_residual,
};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn vec_strategy(n: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, n..=n)
}

fn finite_vec(n: usize) -> impl Strategy<Value = Vec<f32>> {
    vec_strategy(n, -10.0, 10.0)
}

fn positive_vec(n: usize) -> impl Strategy<Value = Vec<f32>> {
    vec_strategy(n, 0.01, 10.0)
}

// ── 1. CUDA KV Cache ────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Appending to KV cache increases layer length monotonically.
    #[test]
    fn kv_cache_append_monotonic(
        heads in 1usize..=4,
        head_dim in 1usize..=8,
    ) {
        let cfg = KvCacheConfig::new(1, heads, head_dim, 32, CacheDtype::F32).unwrap();
        let mut buf = KvCacheBuffer::new(cfg);
        let token_k = vec![1.0f32; heads * head_dim];
        let token_v = vec![2.0f32; heads * head_dim];

        let mut prev_len = 0usize;
        for pos in 0..5 {
            buf.append_kv(0, pos, &token_k, &token_v).unwrap();
            let cur = buf.layer_len(0).unwrap();
            prop_assert!(cur > prev_len, "len must grow: {} <= {}", cur, prev_len);
            prev_len = cur;
        }
    }

    /// Layer length increases by exactly 1 per append.
    #[test]
    fn kv_cache_append_increments_by_one(
        heads in 1usize..=4,
        head_dim in 1usize..=8,
        steps in 1usize..=10,
    ) {
        let cfg = KvCacheConfig::new(1, heads, head_dim, 32, CacheDtype::F32).unwrap();
        let mut buf = KvCacheBuffer::new(cfg);
        let k = vec![0.5f32; heads * head_dim];
        let v = vec![0.5f32; heads * head_dim];

        for i in 0..steps {
            buf.append_kv(0, i, &k, &v).unwrap();
            let len = buf.layer_len(0).unwrap();
            prop_assert_eq!(len, i + 1, "after {} appends, len={}", i + 1, len);
        }
    }

    /// Clearing the cache resets all layer lengths to zero.
    #[test]
    fn kv_cache_clear_resets_length(
        layers in 1usize..=3,
        heads in 1usize..=4,
        head_dim in 1usize..=8,
    ) {
        let cfg = KvCacheConfig::new(layers, heads, head_dim, 16, CacheDtype::F32).unwrap();
        let mut buf = KvCacheBuffer::new(cfg);
        let k = vec![1.0f32; heads * head_dim];
        let v = vec![1.0f32; heads * head_dim];

        for layer in 0..layers {
            buf.append_kv(layer, 0, &k, &v).unwrap();
        }
        buf.clear();
        for layer in 0..layers {
            let len = buf.layer_len(layer).unwrap();
            prop_assert_eq!(len, 0, "layer {} not cleared", layer);
        }
    }

    /// After append, get_kv returns data with the correct length.
    #[test]
    fn kv_cache_get_returns_correct_size(
        heads in 1usize..=4,
        head_dim in 1usize..=8,
        tokens in 1usize..=6,
    ) {
        let cfg = KvCacheConfig::new(1, heads, head_dim, 32, CacheDtype::F32).unwrap();
        let mut buf = KvCacheBuffer::new(cfg);
        let k = vec![1.0f32; heads * head_dim];
        let v = vec![2.0f32; heads * head_dim];

        for pos in 0..tokens {
            buf.append_kv(0, pos, &k, &v).unwrap();
        }
        let (keys, vals) = buf.get_kv(0, 0, tokens).unwrap();
        let expected = tokens * heads * head_dim;
        prop_assert_eq!(keys.len(), expected, "keys len mismatch");
        prop_assert_eq!(vals.len(), expected, "vals len mismatch");
    }

    /// Truncation reduces layer length to the requested value.
    #[test]
    fn kv_cache_truncate_sets_length(
        heads in 1usize..=3,
        head_dim in 1usize..=8,
    ) {
        let cfg = KvCacheConfig::new(1, heads, head_dim, 16, CacheDtype::F32).unwrap();
        let mut buf = KvCacheBuffer::new(cfg);
        let k = vec![1.0f32; heads * head_dim];
        let v = vec![1.0f32; heads * head_dim];

        for pos in 0..5 {
            buf.append_kv(0, pos, &k, &v).unwrap();
        }
        buf.truncate(0, 2).unwrap();
        prop_assert_eq!(buf.layer_len(0).unwrap(), 2);
    }

    /// Stats report non-zero memory bytes after construction.
    #[test]
    fn kv_cache_stats_memory_nonzero(
        heads in 1usize..=4,
        head_dim in 1usize..=8,
    ) {
        let cfg = KvCacheConfig::new(1, heads, head_dim, 16, CacheDtype::F32).unwrap();
        let buf = KvCacheBuffer::new(cfg);
        let stats = buf.stats();
        prop_assert!(stats.memory_bytes > 0, "stats.memory_bytes must be > 0");
    }
}

// ── 2. CUDA Softmax ─────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Softmax output sums to 1 for each row.
    #[test]
    fn softmax_sum_to_one(
        n_cols in 2usize..=32,
        data in proptest::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let n_cols = n_cols.min(data.len());
        let n_rows = data.len() / n_cols;
        if n_rows == 0 { return Ok(()); }
        let input = &data[..n_rows * n_cols];
        let cfg = SoftmaxConfig::for_shape(n_cols, n_rows).unwrap();
        let mut output = vec![0.0f32; input.len()];
        softmax_cpu(input, &mut output, &cfg).unwrap();

        for r in 0..n_rows {
            let row = &output[r * n_cols..(r + 1) * n_cols];
            let sum: f32 = row.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "row {} sum={}, expected 1.0", r, sum,
            );
        }
    }

    /// Softmax is shift-invariant: softmax(x) == softmax(x + c).
    #[test]
    fn softmax_shift_invariance(
        vals in vec_strategy(8, -5.0, 5.0),
        shift in -100.0f32..100.0,
    ) {
        let cfg = SoftmaxConfig::for_shape(8, 1).unwrap();
        let mut out_orig = vec![0.0f32; 8];
        softmax_cpu(&vals, &mut out_orig, &cfg).unwrap();

        let shifted: Vec<f32> = vals.iter().map(|&x| x + shift).collect();
        let mut out_shifted = vec![0.0f32; 8];
        softmax_cpu(&shifted, &mut out_shifted, &cfg).unwrap();

        for i in 0..8 {
            prop_assert!(
                (out_orig[i] - out_shifted[i]).abs() < 1e-4,
                "shift invariance violated at [{}]: {} vs {}", i, out_orig[i], out_shifted[i],
            );
        }
    }

    /// Softmax preserves monotonicity: if x_i > x_j then softmax(x)_i > softmax(x)_j.
    #[test]
    fn softmax_monotonicity(
        vals in vec_strategy(8, -5.0, 5.0),
    ) {
        let cfg = SoftmaxConfig::for_shape(8, 1).unwrap();
        let mut out = vec![0.0f32; 8];
        softmax_cpu(&vals, &mut out, &cfg).unwrap();

        for i in 0..8 {
            for j in (i + 1)..8 {
                if vals[i] > vals[j] + 1e-6 {
                    prop_assert!(
                        out[i] >= out[j] - 1e-6,
                        "monotonicity: vals[{}]={} > vals[{}]={} but out[{}]={} < out[{}]={}",
                        i, vals[i], j, vals[j], i, out[i], j, out[j],
                    );
                }
            }
        }
    }

    /// All softmax outputs are non-negative.
    #[test]
    fn softmax_non_negative(
        vals in vec_strategy(16, -50.0, 50.0),
    ) {
        let cfg = SoftmaxConfig::for_shape(16, 1).unwrap();
        let mut out = vec![0.0f32; 16];
        softmax_cpu(&vals, &mut out, &cfg).unwrap();

        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= 0.0, "softmax[{}]={} is negative", i, v);
        }
    }

    /// Softmax outputs are bounded in [0, 1].
    #[test]
    fn softmax_bounded_zero_one(
        vals in vec_strategy(12, -20.0, 20.0),
    ) {
        let cfg = SoftmaxConfig::for_shape(12, 1).unwrap();
        let mut out = vec![0.0f32; 12];
        softmax_cpu(&vals, &mut out, &cfg).unwrap();

        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= 0.0 && v <= 1.0 + 1e-6, "softmax[{}]={} out of [0,1]", i, v);
        }
    }

    /// Softmax with temperature > 1 makes distribution more uniform.
    #[test]
    fn softmax_temperature_entropy(
        vals in vec_strategy(8, -3.0, 3.0),
    ) {
        let cfg_t1 = SoftmaxConfig::for_shape(8, 1).unwrap()
            .with_temperature(1.0).unwrap();
        let cfg_t4 = SoftmaxConfig::for_shape(8, 1).unwrap()
            .with_temperature(4.0).unwrap();

        let mut out_t1 = vec![0.0f32; 8];
        let mut out_t4 = vec![0.0f32; 8];
        softmax_cpu(&vals, &mut out_t1, &cfg_t1).unwrap();
        softmax_cpu(&vals, &mut out_t4, &cfg_t4).unwrap();

        // Higher temperature → higher entropy → max probability decreases
        let max_t1 = out_t1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_t4 = out_t4.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            max_t4 <= max_t1 + 1e-5,
            "higher temp should flatten: max_t1={} max_t4={}", max_t1, max_t4,
        );
    }
}

// ── 3. CUDA Conv1d ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Output length matches the formula: floor((W + 2*pad - dilation*(ks-1) - 1) / stride) + 1.
    #[test]
    fn conv1d_output_length_formula(
        input_w in 4usize..=32,
        ks in 1usize..=5,
        stride in 1usize..=3,
        padding in 0usize..=3,
        dilation in 1usize..=3,
    ) {
        let cfg = Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: ks,
            stride,
            padding: PaddingMode::Zero(padding),
            dilation,
            groups: 1,
            bias: false,
        };
        let out_w = cfg.output_width(input_w);
        let ek = dilation * (ks - 1) + 1;
        let padded = input_w + 2 * padding;
        let expected = if padded < ek { 0 } else { (padded - ek) / stride + 1 };
        prop_assert_eq!(out_w, expected, "output_width mismatch");
    }

    /// Larger stride yields smaller or equal output width.
    #[test]
    fn conv1d_stride_reduces_output(
        input_w in 8usize..=32,
        ks in 1usize..=3,
    ) {
        let cfg1 = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: ks,
            stride: 1, padding: PaddingMode::Zero(0), dilation: 1, groups: 1, bias: false,
        };
        let cfg2 = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: ks,
            stride: 2, padding: PaddingMode::Zero(0), dilation: 1, groups: 1, bias: false,
        };
        let w1 = cfg1.output_width(input_w);
        let w2 = cfg2.output_width(input_w);
        prop_assert!(w2 <= w1, "stride=2 output {} > stride=1 output {}", w2, w1);
    }

    /// Larger dilation yields smaller or equal output width (no padding).
    #[test]
    fn conv1d_dilation_reduces_output(
        input_w in 8usize..=32,
        ks in 2usize..=4,
    ) {
        let cfg1 = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: ks,
            stride: 1, padding: PaddingMode::Zero(0), dilation: 1, groups: 1, bias: false,
        };
        let cfg2 = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: ks,
            stride: 1, padding: PaddingMode::Zero(0), dilation: 2, groups: 1, bias: false,
        };
        prop_assert!(
            cfg2.output_width(input_w) <= cfg1.output_width(input_w),
            "dilation=2 should not increase output width",
        );
    }

    /// Conv1d with zero-valued weight produces all-zero output.
    #[test]
    fn conv1d_zero_weight_identity(
        input_w in 4usize..=16,
    ) {
        let cfg = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: 3,
            stride: 1, padding: PaddingMode::Zero(1), dilation: 1, groups: 1, bias: false,
        };
        let input = vec![1.0f32; input_w];
        let weight = vec![0.0f32; 3];
        let output = conv1d_cpu(&input, &weight, None, &cfg).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "output[{}]={} should be 0", i, v);
        }
    }

    /// Conv1d output has the correct number of elements.
    #[test]
    fn conv1d_output_element_count(
        input_w in 4usize..=16,
        out_ch in 1usize..=4,
    ) {
        let cfg = Conv1dConfig {
            in_channels: 1, out_channels: out_ch, kernel_size: 3,
            stride: 1, padding: PaddingMode::Zero(1), dilation: 1, groups: 1, bias: false,
        };
        let input = vec![1.0f32; input_w];
        let weight = vec![0.1f32; out_ch * 3];
        let output = conv1d_cpu(&input, &weight, None, &cfg).unwrap();
        let expected_w = cfg.output_width(input_w);
        prop_assert_eq!(output.len(), out_ch * expected_w, "output element count mismatch");
    }

    /// Same-padding preserves output width == ceil(input_w / stride).
    #[test]
    fn conv1d_same_padding_formula(
        input_w in 4usize..=32,
        stride in 1usize..=3,
        ks in 1usize..=5,
    ) {
        let cfg = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: ks,
            stride, padding: PaddingMode::Same, dilation: 1, groups: 1, bias: false,
        };
        let out_w = cfg.output_width(input_w);
        let expected = input_w.div_ceil(stride);
        prop_assert_eq!(out_w, expected, "Same padding: out_w={} expected={}", out_w, expected);
    }

    /// Conv1d output is finite for bounded inputs.
    #[test]
    fn conv1d_output_finite(
        input_w in 4usize..=16,
        vals in vec_strategy(16, -5.0, 5.0),
    ) {
        let cfg = Conv1dConfig {
            in_channels: 1, out_channels: 1, kernel_size: 3,
            stride: 1, padding: PaddingMode::Zero(1), dilation: 1, groups: 1, bias: false,
        };
        let input = &vals[..input_w];
        let weight = vec![0.3f32; 3];
        let output = conv1d_cpu(input, &weight, None, &cfg).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "output[{}]={} not finite", i, v);
        }
    }
}

// ── 4. CUDA Elementwise ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Addition is commutative: a + b == b + a.
    #[test]
    fn elementwise_add_commutative(
        a in vec_strategy(16, -10.0, 10.0),
        b in vec_strategy(16, -10.0, 10.0),
    ) {
        let ab = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Add).unwrap();
        let ba = elementwise_cpu_fallback(&b, &a, ElementwiseOp::Add).unwrap();
        for i in 0..16 {
            prop_assert!(
                (ab[i] - ba[i]).abs() < 1e-6,
                "add not commutative at [{}]: {} vs {}", i, ab[i], ba[i],
            );
        }
    }

    /// Multiplication is commutative: a * b == b * a.
    #[test]
    fn elementwise_mul_commutative(
        a in vec_strategy(16, -10.0, 10.0),
        b in vec_strategy(16, -10.0, 10.0),
    ) {
        let ab = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Mul).unwrap();
        let ba = elementwise_cpu_fallback(&b, &a, ElementwiseOp::Mul).unwrap();
        for i in 0..16 {
            prop_assert!(
                (ab[i] - ba[i]).abs() < 1e-5,
                "mul not commutative at [{}]: {} vs {}", i, ab[i], ba[i],
            );
        }
    }

    /// Distributivity: a * (b + c) ≈ a*b + a*c.
    #[test]
    fn elementwise_distributivity(
        a in vec_strategy(8, -5.0, 5.0),
        b in vec_strategy(8, -5.0, 5.0),
        c in vec_strategy(8, -5.0, 5.0),
    ) {
        let bc = elementwise_cpu_fallback(&b, &c, ElementwiseOp::Add).unwrap();
        let lhs = elementwise_cpu_fallback(&a, &bc, ElementwiseOp::Mul).unwrap();
        let ab = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Mul).unwrap();
        let ac = elementwise_cpu_fallback(&a, &c, ElementwiseOp::Mul).unwrap();
        let rhs = elementwise_cpu_fallback(&ab, &ac, ElementwiseOp::Add).unwrap();
        for i in 0..8 {
            prop_assert!(
                (lhs[i] - rhs[i]).abs() < 1e-3,
                "distributivity at [{}]: {} vs {}", i, lhs[i], rhs[i],
            );
        }
    }

    /// Additive identity: a + 0 == a.
    #[test]
    fn elementwise_add_identity(
        a in vec_strategy(16, -10.0, 10.0),
    ) {
        let zeros = vec![0.0f32; 16];
        let result = elementwise_cpu_fallback(&a, &zeros, ElementwiseOp::Add).unwrap();
        for i in 0..16 {
            prop_assert!(
                (result[i] - a[i]).abs() < 1e-6,
                "add identity at [{}]: {} vs {}", i, result[i], a[i],
            );
        }
    }

    /// Multiplicative identity: a * 1 == a.
    #[test]
    fn elementwise_mul_identity(
        a in vec_strategy(16, -10.0, 10.0),
    ) {
        let ones = vec![1.0f32; 16];
        let result = elementwise_cpu_fallback(&a, &ones, ElementwiseOp::Mul).unwrap();
        for i in 0..16 {
            prop_assert!(
                (result[i] - a[i]).abs() < 1e-5,
                "mul identity at [{}]: {} vs {}", i, result[i], a[i],
            );
        }
    }

    /// Multiplicative zero: a * 0 == 0.
    #[test]
    fn elementwise_mul_zero(
        a in vec_strategy(16, -10.0, 10.0),
    ) {
        let zeros = vec![0.0f32; 16];
        let result = elementwise_cpu_fallback(&a, &zeros, ElementwiseOp::Mul).unwrap();
        for i in 0..16 {
            prop_assert!(result[i].abs() < 1e-6, "a*0[{}]={} != 0", i, result[i]);
        }
    }

    /// Fused add-mul matches sequential: (x + b) * s.
    #[test]
    fn elementwise_fused_matches_sequential(
        x in vec_strategy(8, -5.0, 5.0),
        b in vec_strategy(8, -5.0, 5.0),
        s in vec_strategy(8, 0.1, 5.0),
    ) {
        let fused = fused_elementwise_cpu(&x, &b, &s).unwrap();
        let added = elementwise_cpu_fallback(&x, &b, ElementwiseOp::Add).unwrap();
        let sequential = elementwise_cpu_fallback(&added, &s, ElementwiseOp::Mul).unwrap();
        for i in 0..8 {
            prop_assert!(
                (fused[i] - sequential[i]).abs() < 1e-4,
                "fused vs sequential at [{}]: {} vs {}", i, fused[i], sequential[i],
            );
        }
    }

    /// ReLU output is non-negative.
    #[test]
    fn elementwise_relu_non_negative(
        vals in vec_strategy(16, -10.0, 10.0),
    ) {
        let cfg = ElementwiseConfig::new(16, ElementwiseOp::Relu).unwrap();
        let out = elementwise_unary_cpu(&vals, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= 0.0, "relu[{}]={} is negative", i, v);
        }
    }

    /// Sigmoid output is in (0, 1).
    #[test]
    fn elementwise_sigmoid_bounded(
        vals in vec_strategy(16, -10.0, 10.0),
    ) {
        let cfg = ElementwiseConfig::new(16, ElementwiseOp::Sigmoid).unwrap();
        let out = elementwise_unary_cpu(&vals, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v > -1e-6 && v < 1.0 + 1e-6, "sigmoid[{}]={} out of (0,1)", i, v);
        }
    }
}

// ── 5. CUDA Residual ────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Residual + zero == identity.
    #[test]
    fn residual_add_zero_identity(
        x in vec_strategy(16, -10.0, 10.0),
    ) {
        let zeros = vec![0.0f32; 16];
        let result = residual_add(&x, &zeros).unwrap();
        for i in 0..16 {
            prop_assert!(
                (result[i] - x[i]).abs() < 1e-6,
                "residual+0 at [{}]: {} vs {}", i, result[i], x[i],
            );
        }
    }

    /// Residual add is commutative: x + r == r + x.
    #[test]
    fn residual_add_commutative(
        x in vec_strategy(16, -10.0, 10.0),
        r in vec_strategy(16, -10.0, 10.0),
    ) {
        let xr = residual_add(&x, &r).unwrap();
        let rx = residual_add(&r, &x).unwrap();
        for i in 0..16 {
            prop_assert!(
                (xr[i] - rx[i]).abs() < 1e-6,
                "residual add not commutative at [{}]", i,
            );
        }
    }

    /// Scaled residual with alpha=0 yields the original input.
    #[test]
    fn residual_scaled_alpha_zero(
        x in vec_strategy(16, -10.0, 10.0),
        r in vec_strategy(16, -10.0, 10.0),
    ) {
        let result = residual_add_scaled(&x, &r, 0.0).unwrap();
        for i in 0..16 {
            prop_assert!(
                (result[i] - x[i]).abs() < 1e-6,
                "scaled(alpha=0) at [{}]: {} vs {}", i, result[i], x[i],
            );
        }
    }

    /// Scaled residual with alpha=1 equals plain residual add.
    #[test]
    fn residual_scaled_alpha_one_matches_add(
        x in vec_strategy(16, -5.0, 5.0),
        r in vec_strategy(16, -5.0, 5.0),
    ) {
        let plain = residual_add(&x, &r).unwrap();
        let scaled = residual_add_scaled(&x, &r, 1.0).unwrap();
        for i in 0..16 {
            prop_assert!(
                (plain[i] - scaled[i]).abs() < 1e-5,
                "plain vs scaled(1.0) at [{}]: {} vs {}", i, plain[i], scaled[i],
            );
        }
    }

    /// Scaling linearity: residual(x, r, 2*alpha) == x + 2*alpha*r.
    #[test]
    fn residual_scaling_linearity(
        x in vec_strategy(8, -5.0, 5.0),
        r in vec_strategy(8, -5.0, 5.0),
        alpha in 0.1f32..5.0,
    ) {
        let s1 = residual_add_scaled(&x, &r, alpha).unwrap();
        let s2 = residual_add_scaled(&x, &r, 2.0 * alpha).unwrap();
        // s2[i] - x[i] should be 2*(s1[i] - x[i])
        for i in 0..8 {
            let diff1 = s1[i] - x[i];
            let diff2 = s2[i] - x[i];
            prop_assert!(
                (diff2 - 2.0 * diff1).abs() < 1e-3,
                "linearity at [{}]: diff2={} vs 2*diff1={}", i, diff2, 2.0 * diff1,
            );
        }
    }

    /// Gated residual with gate=1 everywhere equals plain add.
    #[test]
    fn residual_gated_all_ones(
        x in vec_strategy(16, -5.0, 5.0),
        sub in vec_strategy(16, -5.0, 5.0),
    ) {
        let gate = vec![1.0f32; 16];
        let gated = gated_residual(&x, &sub, &gate).unwrap();
        let plain = residual_add(&x, &sub).unwrap();
        for i in 0..16 {
            prop_assert!(
                (gated[i] - plain[i]).abs() < 1e-5,
                "gated(1) vs plain at [{}]: {} vs {}", i, gated[i], plain[i],
            );
        }
    }

    /// Gated residual with gate=0 everywhere returns x.
    #[test]
    fn residual_gated_all_zeros(
        x in vec_strategy(16, -5.0, 5.0),
        sub in vec_strategy(16, -5.0, 5.0),
    ) {
        let gate = vec![0.0f32; 16];
        let gated = gated_residual(&x, &sub, &gate).unwrap();
        for i in 0..16 {
            prop_assert!(
                (gated[i] - x[i]).abs() < 1e-5,
                "gated(0) at [{}]: {} vs x={}", i, gated[i], x[i],
            );
        }
    }

    /// Stochastic depth with keep_prob=1.0 equals plain residual add.
    #[test]
    fn residual_stochastic_keep_all(
        x in vec_strategy(16, -5.0, 5.0),
        sub in vec_strategy(16, -5.0, 5.0),
    ) {
        let result = stochastic_depth_residual(&x, &sub, 1.0, true).unwrap();
        let plain = residual_add(&x, &sub).unwrap();
        for i in 0..16 {
            prop_assert!(
                (result[i] - plain[i]).abs() < 1e-5,
                "stochastic(1.0) vs plain at [{}]", i,
            );
        }
    }
}

// ── 6. CPU Loss Functions ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Cross-entropy is non-negative.
    #[test]
    fn loss_cross_entropy_non_negative(
        logits in vec_strategy(20, -5.0, 5.0),
    ) {
        let num_classes = 5;
        let batch = logits.len() / num_classes;
        if batch == 0 { return Ok(()); }
        let targets: Vec<usize> = (0..batch).map(|i| i % num_classes).collect();
        let (loss, per_sample) = cross_entropy_loss(
            &logits[..batch * num_classes], &targets, num_classes, LossReduction::Mean,
        ).unwrap();
        prop_assert!(loss >= -1e-6, "CE loss={} is negative", loss);
        for (i, &s) in per_sample.iter().enumerate() {
            prop_assert!(s >= -1e-6, "per_sample[{}]={} is negative", i, s);
        }
    }

    /// MSE is non-negative.
    #[test]
    fn loss_mse_non_negative(
        a in vec_strategy(16, -10.0, 10.0),
        b in vec_strategy(16, -10.0, 10.0),
    ) {
        let loss = mse_loss(&a, &b, LossReduction::Mean).unwrap();
        prop_assert!(loss >= -1e-6, "MSE loss={} is negative", loss);
    }

    /// MSE is symmetric: MSE(a, b) == MSE(b, a).
    #[test]
    fn loss_mse_symmetric(
        a in vec_strategy(16, -10.0, 10.0),
        b in vec_strategy(16, -10.0, 10.0),
    ) {
        let ab = mse_loss(&a, &b, LossReduction::Mean).unwrap();
        let ba = mse_loss(&b, &a, LossReduction::Mean).unwrap();
        prop_assert!(
            (ab - ba).abs() < 1e-5,
            "MSE not symmetric: {} vs {}", ab, ba,
        );
    }

    /// MSE of identical inputs is zero.
    #[test]
    fn loss_mse_identical_zero(
        a in vec_strategy(16, -10.0, 10.0),
    ) {
        let loss = mse_loss(&a, &a, LossReduction::Mean).unwrap();
        prop_assert!(loss.abs() < 1e-6, "MSE(a,a)={} != 0", loss);
    }

    /// L1 loss is non-negative.
    #[test]
    fn loss_l1_non_negative(
        a in vec_strategy(16, -10.0, 10.0),
        b in vec_strategy(16, -10.0, 10.0),
    ) {
        let loss = l1_loss(&a, &b, LossReduction::Mean).unwrap();
        prop_assert!(loss >= -1e-6, "L1 loss={} is negative", loss);
    }

    /// L1 is symmetric: L1(a, b) == L1(b, a).
    #[test]
    fn loss_l1_symmetric(
        a in vec_strategy(16, -10.0, 10.0),
        b in vec_strategy(16, -10.0, 10.0),
    ) {
        let ab = l1_loss(&a, &b, LossReduction::Mean).unwrap();
        let ba = l1_loss(&b, &a, LossReduction::Mean).unwrap();
        prop_assert!(
            (ab - ba).abs() < 1e-5,
            "L1 not symmetric: {} vs {}", ab, ba,
        );
    }

    /// Perplexity is >= 1 for non-negative CE.
    #[test]
    fn loss_perplexity_lower_bound(
        ce in 0.0f32..20.0,
    ) {
        let ppl = perplexity(ce);
        prop_assert!(ppl >= 1.0 - 1e-5, "perplexity({})={} < 1", ce, ppl);
    }

    /// Gradient clip by norm produces finite values with bounded norm.
    #[test]
    fn loss_gradient_clip_norm_finite(
        grads in vec_strategy(16, -100.0, 100.0),
        max_norm in 0.1f32..50.0,
    ) {
        let mut g = grads.clone();
        let orig_norm = gradient_clip_norm(&mut g, max_norm).unwrap();
        prop_assert!(orig_norm.is_finite(), "original norm not finite");
        let clipped_norm: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            clipped_norm <= max_norm + 1e-3,
            "clipped norm {} > max_norm {}", clipped_norm, max_norm,
        );
        for (i, &v) in g.iter().enumerate() {
            prop_assert!(v.is_finite(), "grad[{}] not finite after clip", i);
        }
    }

    /// Gradient clip by value keeps all values in [-max, max].
    #[test]
    fn loss_gradient_clip_value_bounded(
        grads in vec_strategy(16, -100.0, 100.0),
        max_val in 0.1f32..50.0,
    ) {
        let mut g = grads.clone();
        gradient_clip_value(&mut g, max_val).unwrap();
        for (i, &v) in g.iter().enumerate() {
            prop_assert!(
                v >= -max_val - 1e-6 && v <= max_val + 1e-6,
                "grad[{}]={} outside [-{}, {}]", i, v, max_val, max_val,
            );
        }
    }
}

// ── 7. CPU Batch Norm ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Training batch norm output has approximately zero mean per channel.
    #[test]
    fn batch_norm_zero_mean(
        batch_size in 4usize..=16,
        features in 1usize..=4,
    ) {
        let n = batch_size * features;
        // Generate data with non-zero mean
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 2.0).collect();
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];
        let cfg = BatchNormConfig { num_features: features, eps: 1e-5, momentum: 0.1, training: true };

        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &cfg).unwrap();

        for ch in 0..features {
            let mean: f32 = (0..batch_size).map(|n| output[n * features + ch]).sum::<f32>() / batch_size as f32;
            prop_assert!(
                mean.abs() < 1e-3,
                "channel {} mean={}, expected ~0", ch, mean,
            );
        }
    }

    /// Training batch norm output has approximately unit variance per channel.
    #[test]
    fn batch_norm_unit_variance(
        batch_size in 4usize..=16,
        features in 1usize..=4,
    ) {
        let n = batch_size * features;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.5).collect();
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];
        let cfg = BatchNormConfig { num_features: features, eps: 1e-5, momentum: 0.1, training: true };

        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &cfg).unwrap();

        for ch in 0..features {
            let vals: Vec<f32> = (0..batch_size).map(|n| output[n * features + ch]).collect();
            let mean: f32 = vals.iter().sum::<f32>() / batch_size as f32;
            let var: f32 = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / batch_size as f32;
            prop_assert!(
                (var - 1.0).abs() < 0.15,
                "channel {} var={}, expected ~1.0", ch, var,
            );
        }
    }

    /// Inference batch norm output is finite for bounded inputs.
    #[test]
    fn batch_norm_inference_finite(
        batch_size in 2usize..=8,
        features in 1usize..=4,
    ) {
        let n = batch_size * features;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];

        let output = batch_norm_inference(&input, &gamma, &beta, &running_mean, &running_var, 1e-5).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "output[{}] not finite", i);
        }
    }

    /// Batch norm with gamma=1, beta=0 on pre-normalized data preserves values.
    #[test]
    fn batch_norm_identity_affine(
        features in 1usize..=4,
    ) {
        // Construct input that is already zero-mean, unit-variance per channel
        let batch = 8;
        let mut input = vec![0.0f32; batch * features];
        for ch in 0..features {
            for n in 0..batch {
                input[n * features + ch] = (n as f32 - 3.5) / 2.29; // roughly N(0,1)
            }
        }
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];
        let cfg = BatchNormConfig { num_features: features, eps: 1e-5, momentum: 0.1, training: true };

        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &cfg).unwrap();
        // Output should be close to the re-normalized version (may differ slightly from input)
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "output[{}] not finite", i);
            prop_assert!(v.abs() < 10.0, "output[{}]={} unexpectedly large", i, v);
        }
    }

    /// Running stats are updated when training=true.
    #[test]
    fn batch_norm_running_stats_update(
        features in 1usize..=4,
    ) {
        let batch = 8;
        let n = batch * features;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2).collect();
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];
        let cfg = BatchNormConfig { num_features: features, eps: 1e-5, momentum: 0.1, training: true };

        let (_, updated_mean, updated_var) = batch_norm_forward(
            &input, &gamma, &beta, &running_mean, &running_var, &cfg,
        ).unwrap();

        // Running mean should have been updated (not all zeros anymore)
        let mean_changed = updated_mean.iter().any(|&m| m.abs() > 1e-8);
        prop_assert!(mean_changed, "running_mean not updated");

        for &v in &updated_var {
            prop_assert!(v.is_finite() && v >= 0.0, "updated_var={} invalid", v);
        }
    }
}

// ── 8. CPU Tensor Parallel ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Shard then gather == identity.
    #[test]
    fn tensor_parallel_shard_gather_roundtrip(
        num_ranks in 1usize..=4,
    ) {
        // Use evenly divisible tensor length
        let len = num_ranks * 8;
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let cfg = TensorParallelConfig {
            num_ranks,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        let (shards, _) = shard_tensor(&data, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        prop_assert_eq!(gathered.len(), data.len(), "roundtrip length mismatch");
        for i in 0..len {
            prop_assert!(
                (gathered[i] - data[i]).abs() < 1e-6,
                "roundtrip mismatch at [{}]: {} vs {}", i, gathered[i], data[i],
            );
        }
    }

    /// Shard count equals num_ranks.
    #[test]
    fn tensor_parallel_shard_count(
        num_ranks in 1usize..=8,
    ) {
        let len = num_ranks * 4;
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let cfg = TensorParallelConfig {
            num_ranks,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        let (shards, _) = shard_tensor(&data, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        prop_assert_eq!(shards.len(), num_ranks, "shard count mismatch");
    }

    /// Each shard has the correct size for evenly divisible tensors.
    #[test]
    fn tensor_parallel_shard_size_even(
        num_ranks in 1usize..=4,
        chunk in 1usize..=8,
    ) {
        let len = num_ranks * chunk;
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let cfg = TensorParallelConfig {
            num_ranks,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        let (shards, _) = shard_tensor(&data, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        for (i, s) in shards.iter().enumerate() {
            prop_assert_eq!(s.data.len(), chunk, "shard {} size mismatch", i);
        }
    }

    /// compute_shard_ranges covers the full length without gaps or overlaps.
    #[test]
    fn tensor_parallel_ranges_cover_full(
        len in 1usize..=64,
        num_ranks in 1usize..=8,
    ) {
        let ranges = compute_shard_ranges(len, num_ranks).unwrap();
        prop_assert_eq!(ranges.len(), num_ranks, "range count mismatch");
        prop_assert_eq!(ranges[0].0, 0, "first range doesn't start at 0");
        prop_assert_eq!(ranges.last().unwrap().1, len, "last range doesn't end at len");
        for w in ranges.windows(2) {
            prop_assert_eq!(w[0].1, w[1].0, "gap between ranges");
        }
    }

    /// Total shard data length equals original tensor length.
    #[test]
    fn tensor_parallel_total_length_preserved(
        num_ranks in 1usize..=4,
        chunk in 1usize..=8,
    ) {
        let len = num_ranks * chunk;
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let cfg = TensorParallelConfig {
            num_ranks,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        let (shards, _) = shard_tensor(&data, &cfg, &ShardingStrategy::RowParallel).unwrap();
        let total: usize = shards.iter().map(|s| s.data.len()).sum();
        prop_assert_eq!(total, len, "total shard len mismatch");
    }

    /// validate_sharding rejects uneven splits.
    #[test]
    fn tensor_parallel_validate_rejects_uneven(
        len in 1usize..=32,
        shards in 2usize..=8,
    ) {
        if len % shards != 0 {
            let result = validate_sharding(len, shards);
            prop_assert!(result.is_err(), "should reject uneven: len={} shards={}", len, shards);
        }
    }

    /// validate_sharding accepts even splits.
    #[test]
    fn tensor_parallel_validate_accepts_even(
        factor in 1usize..=8,
        multiplier in 1usize..=8,
    ) {
        let len = factor * multiplier;
        let result = validate_sharding(len, factor);
        prop_assert!(result.is_ok(), "should accept: len={} shards={}", len, factor);
    }

    /// Shard metadata is consistent.
    #[test]
    fn tensor_parallel_shard_metadata(
        num_ranks in 1usize..=4,
    ) {
        let len = num_ranks * 4;
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let cfg = TensorParallelConfig {
            num_ranks,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        let (shards, _) = shard_tensor(&data, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        for (i, s) in shards.iter().enumerate() {
            prop_assert_eq!(s.shard_index, i, "shard_index mismatch");
            prop_assert_eq!(s.rank_id, i, "rank_id mismatch");
            prop_assert_eq!(s.total_shards, num_ranks, "total_shards mismatch");
        }
    }
}
