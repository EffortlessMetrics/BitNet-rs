//! Wave 5 snapshot tests for `bitnet-kernels` kernel output regression.
//!
//! Pins the kernel provider listing, SIMD detection, capability summaries,
//! and — most importantly — the numerical outputs of core kernel operations
//! (reduction, conv1d, RoPE, pooling, scatter/gather, softmax, embedding,
//! shaped reduction, SIMD math, and fused ops) so that unintentional changes
//! are caught at review time.

use bitnet_kernels::KernelManager;
use bitnet_kernels::cpu::conv1d::{Conv1dConfig, PaddingMode, conv1d_forward};
use bitnet_kernels::cpu::fusion::{fused_gelu_linear, fused_rmsnorm_linear, fused_scale_add};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::softmax::{softmax, softmax_batch};
use bitnet_kernels::cpu::{fast_exp_f32, fast_sigmoid_f32, fast_tanh_f32, simd_dot_product};
use bitnet_kernels::device_features;
use bitnet_kernels::embedding::sinusoidal_position_encoding;
use bitnet_kernels::reduction::{ReductionOp, reduce_f32, reduce_rows_f32};
use bitnet_kernels::scatter_gather::{
    GatherConfig, ScatterGatherKernel, ScatterMode, gather_cpu, scatter_cpu,
};
use bitnet_kernels::shaped_reduction::{ShapedReductionConfig, reduce_f32 as shaped_reduce_f32};

/// Format a float slice to 6 decimal places for stable snapshots.
fn fmt6(v: &[f32]) -> String {
    v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(", ")
}

// =========================================================================
// Section 1 — Original API-surface tests
// =========================================================================

#[test]
fn kernel_provider_list_non_empty() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    insta::assert_snapshot!(format!(
        "provider_count={} providers={:?}",
        providers.len(),
        providers
    ));
}

#[test]
fn kernel_provider_list_always_has_cpu_fallback() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    let has_cpu = providers
        .iter()
        .any(|p| p.contains("cpu") || p.contains("fallback") || p.contains("Fallback"));
    insta::assert_snapshot!(format!("has_cpu_fallback={has_cpu}"));
}

#[test]
fn detect_simd_level_is_stable() {
    let level_a = device_features::detect_simd_level();
    let level_b = device_features::detect_simd_level();
    assert_eq!(level_a, level_b);
    insta::with_settings!({filters => vec![(r"(?:avx512|avx2|sse4\.2|neon|scalar)", "[SIMD]")]}, {
        insta::assert_snapshot!(format!("simd_level={level_a}"));
    });
}

#[test]
fn device_capability_summary_format() {
    let summary = device_features::device_capability_summary();
    insta::with_settings!({filters => vec![
        (r"(?:avx512|avx2|sse4\.2|neon|scalar)", "[SIMD]"),
        (r"CUDA \d+\.\d+", "CUDA [VERSION]"),
    ]}, {
        insta::assert_snapshot!(summary);
    });
}

#[test]
fn current_kernel_capabilities_summary() {
    let caps = device_features::current_kernel_capabilities();
    insta::with_settings!({filters => vec![(r"simd=\w+", "simd=[SIMD]")]}, {
        insta::assert_snapshot!(caps.summary());
    });
}

#[test]
fn current_kernel_capabilities_cpu_is_compiled() {
    let caps = device_features::current_kernel_capabilities();
    insta::assert_snapshot!(format!("cpu_rust={}", caps.cpu_rust));
}

// =========================================================================
// Section 2 — Reduction operation outputs
// =========================================================================

#[test]
fn reduction_sum_known_input() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = reduce_f32(&data, ReductionOp::Sum);
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_max_known_input() {
    let data = [-3.0_f32, 0.5, 2.0, -1.0, 4.0];
    let result = reduce_f32(&data, ReductionOp::Max);
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_min_known_input() {
    let data = [10.0_f32, -7.0, 3.0, 0.0, 5.0];
    let result = reduce_f32(&data, ReductionOp::Min);
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_mean_known_input() {
    let data = [2.0_f32, 4.0, 6.0, 8.0];
    let result = reduce_f32(&data, ReductionOp::Mean);
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_l2norm_known_input() {
    let data = [3.0_f32, 4.0];
    let result = reduce_f32(&data, ReductionOp::L2Norm);
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_rows_sum_2x3() {
    let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = reduce_rows_f32(&matrix, 2, 3, ReductionOp::Sum).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

// =========================================================================
// Section 3 — Shaped reduction outputs
// =========================================================================

#[test]
fn shaped_reduction_global_mean() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cfg = ShapedReductionConfig::global(ReductionOp::Mean);
    let result = shaped_reduce_f32(&data, &[2, 3], &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

#[test]
fn shaped_reduction_axis0_max() {
    let data = [1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0];
    let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(0), false);
    let result = shaped_reduce_f32(&data, &[2, 3], &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&result));
}

// =========================================================================
// Section 4 — Conv1d outputs
// =========================================================================

#[test]
fn conv1d_small_kernel_no_padding() {
    let config = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: PaddingMode::Zero(0),
        dilation: 1,
        groups: 1,
        bias: false,
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let weight = [1.0_f32, 0.0, -1.0];
    let output = conv1d_forward(&input, &weight, None, &config).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn conv1d_with_bias_and_same_padding() {
    let config = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: PaddingMode::Same,
        dilation: 1,
        groups: 1,
        bias: true,
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let weight = [0.5_f32, 1.0, 0.5];
    let bias = [0.1_f32];
    let output = conv1d_forward(&input, &weight, Some(&bias), &config).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

// =========================================================================
// Section 5 — RoPE embedding tables
// =========================================================================

#[test]
fn rope_frequencies_dim4_len4() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    insta::assert_snapshot!(fmt6(&freqs));
}

#[test]
fn rope_apply_known_vector() {
    let cfg = RopeConfig::new(4, 8);
    let freqs = compute_frequencies(&cfg);
    let mut data = [1.0_f32, 0.0, 0.0, 1.0];
    apply_rope(&mut data, 1, 4, &freqs);
    insta::assert_snapshot!(fmt6(&data));
}

// =========================================================================
// Section 6 — Pooling outputs
// =========================================================================

#[test]
fn pool_avg_known_input() {
    let cfg = PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 0 };
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let output = PoolingKernel::apply(&input, &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn pool_max_known_input() {
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    let input = [1.0_f32, 3.0, 2.0, 5.0, 4.0, 6.0];
    let output = PoolingKernel::apply(&input, &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn pool_global_average() {
    let cfg =
        PoolConfig { pool_type: PoolType::GlobalAverage, kernel_size: 0, stride: 0, padding: 0 };
    let input = [2.0_f32, 4.0, 6.0, 8.0];
    let output = PoolingKernel::apply(&input, &cfg).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

// =========================================================================
// Section 7 — Scatter/gather outputs
// =========================================================================

#[test]
fn gather_axis0_known_indices() {
    let src = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // 3×2
    let kernel = ScatterGatherKernel::new(3, 2).unwrap();
    let config = GatherConfig::new(0, (2, 2), true).unwrap();
    let indices: [usize; 4] = [2, 0, 1, 2];
    let mut output = [0.0_f32; 4];
    gather_cpu(&src, &indices, &mut output, &kernel, &config).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn scatter_add_axis0() {
    let src = [1.0_f32, 2.0, 3.0, 4.0]; // 2×2 source
    let indices: [usize; 4] = [0, 1, 1, 0];
    let config = GatherConfig::new(0, (2, 2), true).unwrap();
    let mut dst = [0.0_f32; 6]; // 3×2 destination
    scatter_cpu(&src, &indices, &mut dst, (3, 2), &config, ScatterMode::Add).unwrap();
    insta::assert_snapshot!(fmt6(&dst));
}

// =========================================================================
// Section 8 — Softmax outputs
// =========================================================================

#[test]
fn softmax_known_logits() {
    let logits = [1.0_f32, 2.0, 3.0];
    let probs = softmax(&logits, 1.0).unwrap();
    insta::assert_snapshot!(fmt6(&probs));
}

#[test]
fn softmax_batch_2x3() {
    let logits = [1.0_f32, 2.0, 3.0, 3.0, 2.0, 1.0];
    let probs = softmax_batch(&logits, 3, 1.0).unwrap();
    insta::assert_snapshot!(fmt6(&probs));
}

// =========================================================================
// Section 9 — Sinusoidal position encoding
// =========================================================================

#[test]
fn sinusoidal_encoding_pos0_dim8() {
    let mut out = vec![0.0_f32; 8];
    sinusoidal_position_encoding(0, 8, &mut out);
    insta::assert_snapshot!(fmt6(&out));
}

#[test]
fn sinusoidal_encoding_pos5_dim8() {
    let mut out = vec![0.0_f32; 8];
    sinusoidal_position_encoding(5, 8, &mut out);
    insta::assert_snapshot!(fmt6(&out));
}

// =========================================================================
// Section 10 — SIMD math functions (exp, tanh, sigmoid, dot)
// =========================================================================

#[test]
fn fast_exp_known_values() {
    let input = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let output = fast_exp_f32(&input);
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn fast_tanh_known_values() {
    let input = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let output = fast_tanh_f32(&input);
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn fast_sigmoid_known_values() {
    let input = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let output = fast_sigmoid_f32(&input);
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn simd_dot_product_known() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [4.0_f32, 3.0, 2.0, 1.0];
    let result = simd_dot_product(&a, &b);
    insta::assert_snapshot!(format!("{result:.6}"));
}

// =========================================================================
// Section 11 — Fused operation outputs
// =========================================================================

#[test]
fn fused_rmsnorm_linear_known() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let gamma = [1.0_f32; 4];
    // weight is [2×4] row-major → 2 outputs
    let weight = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let output = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn fused_gelu_linear_known() {
    let input = [0.0_f32, 1.0, -1.0, 2.0];
    let weight = [1.0_f32, 1.0, 1.0, 1.0]; // [1×4] → 1 output
    let bias = [0.0_f32];
    let output = fused_gelu_linear(&input, &weight, &bias).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}

#[test]
fn fused_scale_add_known() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let output = fused_scale_add(&a, &b, 0.5).unwrap();
    insta::assert_snapshot!(fmt6(&output));
}
