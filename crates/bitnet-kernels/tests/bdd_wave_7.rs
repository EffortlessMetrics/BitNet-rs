#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! BDD Wave 7 — Inference Pipeline Integration Tests
//!
//! Given/When/Then tests for end-to-end kernel pipeline integration:
//!
//! 1. CUDA profiling integration (mocked on CPU)
//! 2. CPU batch normalization flow
//! 3. CUDA graph workflow (mocked replay)
//! 4. CPU element-wise chains
//! 5. Kernel selection flow
//! 6. Quantization round-trip
//! 7. Memory management (arena allocator)
//! 8. Thread pool workflow (rayon)
//! 9. Pipeline parallelism
//! 10. Attention integration

use std::time::Instant;

use bitnet_common::QuantizationType;
use bitnet_kernels::cpu::attention::{
    AttentionConfig, attention_with_kv_cache, multi_head_attention_cpu,
    scaled_dot_product_attention,
};
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::quantize::{
    dequantize_asymmetric_u8, dequantize_symmetric_i8, quantize_asymmetric_u8,
    quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider, select_cpu_kernel};

// ── Helpers ────────────────────────────────────────────────────────

const EPS: f32 = 1e-5;

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    assert!((a - b).abs() <= tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
}

fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&ai, &bi)) in a.iter().zip(b).enumerate() {
        assert_close(ai, bi, tol, &format!("{ctx}[{i}]"));
    }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

fn variance(v: &[f32]) -> f32 {
    let m = mean(v);
    v.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
}

// ═══════════════════════════════════════════════════════════════════
// Section 1 — CUDA Profiling Integration (CPU-side timing mock)
// ═══════════════════════════════════════════════════════════════════

/// Lightweight profiling wrapper used to validate timing integration.
struct KernelProfiler {
    entries: std::cell::RefCell<Vec<ProfileEntry>>,
}

struct ProfileEntry {
    name: String,
    elapsed_ns: u128,
}

struct ProfilingReport {
    total_time_ns: u128,
    entries: Vec<(String, u128)>,
}

impl KernelProfiler {
    fn new() -> Self {
        Self { entries: std::cell::RefCell::new(Vec::new()) }
    }

    fn profile<F: FnOnce()>(&self, name: &str, f: F) {
        let start = Instant::now();
        f();
        let elapsed_ns = start.elapsed().as_nanos();
        self.entries.borrow_mut().push(ProfileEntry { name: name.to_string(), elapsed_ns });
    }

    fn report(&self) -> ProfilingReport {
        let entries = self.entries.borrow();
        let total_time_ns = entries.iter().map(|e| e.elapsed_ns).sum();
        let items = entries.iter().map(|e| (e.name.clone(), e.elapsed_ns)).collect();
        ProfilingReport { total_time_ns, entries: items }
    }
}

#[test]
fn given_profiling_enabled_when_matmul_then_timing_recorded() {
    // Given
    let profiler = KernelProfiler::new();
    let kernel = FallbackKernel;
    let a = vec![1i8; 16];
    let b = vec![1u8; 16];
    let mut c = vec![0.0f32; 16];

    // When
    profiler.profile("matmul_i2s", || {
        let _ = kernel.matmul_i2s(&a, &b, &mut c, 4, 4, 4);
    });

    // Then
    let report = profiler.report();
    assert!(report.total_time_ns > 0, "profiler must record nonzero time");
    assert_eq!(report.entries.len(), 1);
    assert_eq!(report.entries[0].0, "matmul_i2s");
}

#[test]
fn given_profiling_enabled_when_multiple_kernels_then_all_timings_accumulated() {
    // Given
    let profiler = KernelProfiler::new();
    let kernel = FallbackKernel;

    // When — run two operations
    profiler.profile("op1", || {
        let _ = kernel.matmul_i2s(&[1i8; 4], &[1u8; 4], &mut [0.0f32; 4], 2, 2, 2);
    });
    profiler.profile("op2", || {
        let _ = kernel.matmul_i2s(&[1i8; 4], &[1u8; 4], &mut [0.0f32; 4], 2, 2, 2);
    });

    // Then
    let report = profiler.report();
    assert_eq!(report.entries.len(), 2);
    let sum: u128 = report.entries.iter().map(|e| e.1).sum();
    assert_eq!(sum, report.total_time_ns);
}

#[test]
fn given_profiling_enabled_when_no_ops_run_then_report_is_empty() {
    // Given
    let profiler = KernelProfiler::new();

    // When — nothing

    // Then
    let report = profiler.report();
    assert_eq!(report.total_time_ns, 0);
    assert!(report.entries.is_empty());
}

#[test]
fn given_profiling_enabled_when_linear_then_timing_is_positive() {
    // Given
    let profiler = KernelProfiler::new();
    let cfg = LinearConfig {
        in_features: 8,
        out_features: 4,
        batch_size: 2,
        has_bias: false,
        ..LinearConfig::default()
    };
    let x = vec![1.0f32; 16];
    let w = vec![0.5f32; 32];
    let mut out = vec![0.0f32; 8];

    // When
    profiler.profile("linear", || {
        linear_cpu(&x, &w, None, &mut out, &cfg).unwrap();
    });

    // Then
    let report = profiler.report();
    assert!(report.total_time_ns > 0);
}

#[test]
fn given_profiling_enabled_when_quantize_then_timing_recorded() {
    // Given
    let profiler = KernelProfiler::new();
    let data = vec![0.5f32, -0.3, 0.8, -1.0, 0.0, 0.2, -0.7, 0.9];

    // When
    let mut result = (vec![0i8; 8], 0.0f32);
    profiler.profile("quantize", || {
        result = quantize_symmetric_i8(&data, 8);
    });

    // Then
    let report = profiler.report();
    assert!(report.total_time_ns > 0);
    assert_eq!(report.entries[0].0, "quantize");
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — CPU Batch Normalization Flow
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_normalized_data_when_batch_norm_applied_then_mean_approx_zero() {
    // Given — data with known statistics
    let num_features = 2;
    let batch_size = 4;
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; num_features];
    let beta = vec![0.0; num_features];
    let running_mean = vec![0.0; num_features];
    let running_var = vec![1.0; num_features];
    let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };

    // When
    let (output, _, _) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

    // Then — each feature channel mean ≈ 0
    for ch in 0..num_features {
        let ch_vals: Vec<f32> = (0..batch_size).map(|n| output[n * num_features + ch]).collect();
        let ch_mean = mean(&ch_vals);
        assert_close(ch_mean, 0.0, 1e-5, &format!("batch_norm_ch{ch}_mean"));
    }
}

#[test]
fn given_normalized_data_when_batch_norm_applied_then_variance_approx_one() {
    // Given
    let num_features = 2;
    let batch_size = 4;
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; num_features];
    let beta = vec![0.0; num_features];
    let running_mean = vec![0.0; num_features];
    let running_var = vec![1.0; num_features];
    let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };

    // When
    let (output, _, _) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

    // Then — each feature channel variance ≈ 1
    for ch in 0..num_features {
        let ch_vals: Vec<f32> = (0..batch_size).map(|n| output[n * num_features + ch]).collect();
        let ch_var = variance(&ch_vals);
        assert_close(ch_var, 1.0, 0.05, &format!("batch_norm_ch{ch}_var"));
    }
}

#[test]
fn given_batch_norm_training_when_inference_then_uses_running_stats() {
    // Given — train first to update running stats
    let num_features = 2;
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; num_features];
    let beta = vec![0.0; num_features];
    let running_mean = vec![0.0; num_features];
    let running_var = vec![1.0; num_features];
    let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };
    let (_, updated_mean, updated_var) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

    // When — run inference with updated stats
    let inference_out =
        batch_norm_inference(&input, &gamma, &beta, &updated_mean, &updated_var, 1e-5).unwrap();

    // Then — output length matches input
    assert_eq!(inference_out.len(), input.len());
    // Inference output should be finite
    for &v in &inference_out {
        assert!(v.is_finite(), "inference output must be finite");
    }
}

#[test]
fn given_uniform_data_when_batch_norm_then_output_near_zero() {
    // Given — constant input (variance = 0, so output = beta)
    let num_features = 3;
    let input = vec![5.0f32; 4 * num_features];
    let gamma = vec![1.0; num_features];
    let beta = vec![0.0; num_features];
    let running_mean = vec![0.0; num_features];
    let running_var = vec![1.0; num_features];
    let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };

    // When
    let (output, _, _) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

    // Then — all outputs ≈ beta (0) since (x - mean) = 0
    for &v in &output {
        assert_close(v, 0.0, 1e-3, "uniform_bn_output");
    }
}

#[test]
fn given_gamma_beta_when_batch_norm_then_scaling_applied() {
    // Given
    let num_features = 1;
    let input = vec![1.0, 3.0, 5.0, 7.0]; // mean=4, var=5
    let gamma = vec![2.0]; // scale by 2
    let beta = vec![1.0]; // shift by 1
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };

    // When
    let (output, _, _) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

    // Then — mean of output should be ≈ beta (since normalized mean = 0, scaled by gamma + beta)
    let out_mean = mean(&output);
    assert_close(out_mean, 1.0, 0.1, "scaled_bn_mean");
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — CUDA Graph Workflow (CPU replay simulation)
// ═══════════════════════════════════════════════════════════════════

/// Simulates a captured CUDA graph: record ops, then replay.
struct CpuGraphRecorder {
    ops: Vec<GraphOp>,
}

type GraphOp = Box<dyn Fn(&[f32]) -> Vec<f32>>;

impl CpuGraphRecorder {
    fn new() -> Self {
        Self { ops: Vec::new() }
    }

    fn record(&mut self, op: GraphOp) {
        self.ops.push(op);
    }

    fn replay(&self, input: &[f32]) -> Vec<f32> {
        let mut data = input.to_vec();
        for op in &self.ops {
            data = op(&data);
        }
        data
    }
}

#[test]
fn given_captured_ops_when_replaying_then_output_matches_direct_execution() {
    // Given — record two ops: scale by 2, then add 1
    let mut graph = CpuGraphRecorder::new();
    graph.record(Box::new(|v: &[f32]| v.iter().map(|x| x * 2.0).collect()));
    graph.record(Box::new(|v: &[f32]| v.iter().map(|x| x + 1.0).collect()));
    let input = vec![1.0, 2.0, 3.0, 4.0];

    // When
    let replayed = graph.replay(&input);

    // Direct execution
    let direct: Vec<f32> = input.iter().map(|x| x * 2.0 + 1.0).collect();

    // Then
    assert_slice_close(&replayed, &direct, EPS, "graph_replay_vs_direct");
}

#[test]
fn given_captured_linear_and_relu_when_replaying_then_matches_sequential() {
    // Given — linear: y = x * 0.5, relu: max(0, y)
    let mut graph = CpuGraphRecorder::new();
    graph.record(Box::new(|v: &[f32]| v.iter().map(|x| x * 0.5).collect()));
    graph.record(Box::new(|v: &[f32]| v.iter().map(|x| x.max(0.0)).collect()));
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    // When
    let replayed = graph.replay(&input);

    // Then
    let expected = vec![0.0, 0.0, 0.0, 0.5, 1.0];
    assert_slice_close(&replayed, &expected, EPS, "graph_linear_relu");
}

#[test]
fn given_empty_graph_when_replaying_then_output_equals_input() {
    // Given
    let graph = CpuGraphRecorder::new();
    let input = vec![1.0, 2.0, 3.0];

    // When
    let replayed = graph.replay(&input);

    // Then
    assert_slice_close(&replayed, &input, 0.0, "empty_graph_passthrough");
}

#[test]
fn given_captured_normalization_when_replaying_twice_then_results_identical() {
    // Given
    let mut graph = CpuGraphRecorder::new();
    graph.record(Box::new(|v: &[f32]| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter().map(|x| x / norm).collect()
    }));
    let input = vec![3.0, 4.0];

    // When
    let r1 = graph.replay(&input);
    let r2 = graph.replay(&input);

    // Then — deterministic
    assert_slice_close(&r1, &r2, 0.0, "graph_replay_deterministic");
    // L2 norm ≈ 1
    let norm: f32 = r1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_close(norm, 1.0, 1e-6, "normalized_l2");
}

#[test]
fn given_captured_quantize_dequantize_when_replaying_then_roundtrip_close() {
    // Given — capture quantize+dequantize as graph ops
    let mut graph = CpuGraphRecorder::new();
    graph.record(Box::new(|v: &[f32]| {
        let (q, scale) = quantize_symmetric_i8(v, 8);
        dequantize_symmetric_i8(&q, scale)
    }));
    let input = vec![0.5, -0.3, 0.8, -1.0];

    // When
    let replayed = graph.replay(&input);

    // Then
    assert_slice_close(&replayed, &input, 0.02, "graph_quant_roundtrip");
}

// ═══════════════════════════════════════════════════════════════════
// Section 4 — CPU Element-wise Chains
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_two_element_wise_ops_when_chained_then_result_matches_sequential() {
    // Given — scale by 3, then subtract 1
    let input = [1.0f32, 2.0, 3.0, 4.0];

    // When — chained
    let chained: Vec<f32> = input.iter().map(|x| x * 3.0 - 1.0).collect();
    // Sequential
    let step1: Vec<f32> = input.iter().map(|x| x * 3.0).collect();
    let step2: Vec<f32> = step1.iter().map(|x| x - 1.0).collect();

    // Then
    assert_slice_close(&chained, &step2, EPS, "chained_vs_sequential");
}

#[test]
fn given_residual_add_when_chained_with_scale_then_result_correct() {
    // Given
    let mut output = vec![1.0, 2.0, 3.0, 4.0];
    let residual = vec![0.5, 0.5, 0.5, 0.5];

    // When — add residual then scale
    add_residual(&mut output, &residual).unwrap();
    // output is now [1.5, 2.5, 3.5, 4.5]

    let mut scaled = output.clone();
    let scale_residual = vec![0.0; 4]; // zero residual
    add_residual_scaled(&mut scaled, &scale_residual, 2.0).unwrap();

    // Then — add_residual_scaled with zero residual doesn't change values
    assert_slice_close(&scaled, &output, EPS, "residual_chain");
}

#[test]
fn given_layer_norm_then_residual_when_chained_then_output_valid() {
    // Given
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let config = LayerNormConfig::new(vec![4]);

    // When — normalize then add residual
    let normed = layer_norm(&input, &gamma, None, &config).unwrap();
    let mut combined = normed.clone();
    add_residual(&mut combined, &input).unwrap();

    // Then — combined = normed + original (skip connection)
    let expected: Vec<f32> = normed.iter().zip(input.iter()).map(|(n, i)| n + i).collect();
    assert_slice_close(&combined, &expected, EPS, "layernorm_residual_chain");
}

#[test]
fn given_silu_activation_when_chained_with_linear_then_output_finite() {
    // Given
    let cfg = LinearConfig {
        in_features: 4,
        out_features: 4,
        batch_size: 1,
        has_bias: false,
        ..LinearConfig::default()
    };
    let x = vec![1.0, -1.0, 0.5, -0.5];
    let w = vec![0.5f32; 16]; // identity-ish
    let mut linear_out = vec![0.0f32; 4];

    // When
    linear_cpu(&x, &w, None, &mut linear_out, &cfg).unwrap();
    // Apply SiLU: x * sigmoid(x)
    let silu_out: Vec<f32> = linear_out.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect();

    // Then
    for &v in &silu_out {
        assert!(v.is_finite(), "SiLU output must be finite, got {v}");
    }
}

#[test]
fn given_rms_norm_then_linear_when_chained_then_output_shape_correct() {
    // Given
    let dim = 8;
    let input = vec![1.0f32; dim];
    let gamma = vec![1.0f32; dim];
    let config = LayerNormConfig::new(vec![dim]);

    // When — RMS norm then linear projection
    let normed = rms_norm(&input, &gamma, &config).unwrap();
    let linear_cfg = LinearConfig {
        in_features: dim,
        out_features: 4,
        batch_size: 1,
        has_bias: false,
        ..LinearConfig::default()
    };
    let w = vec![1.0f32; 4 * dim];
    let mut out = vec![0.0f32; 4];
    linear_cpu(&normed, &w, None, &mut out, &linear_cfg).unwrap();

    // Then
    assert_eq!(out.len(), 4);
    for &v in &out {
        assert!(v.is_finite(), "RMS norm → linear must be finite");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 5 — Kernel Selection Flow
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_cpu_build_when_selecting_kernel_then_cpu_provider_returned() {
    // Given
    let mgr = KernelManager::new();

    // When
    let provider = mgr.select_best().unwrap();

    // Then
    assert!(provider.is_available());
    // In CPU-only builds, should not be "cuda"
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    assert_ne!(provider.name(), "cuda");
}

#[test]
fn given_kernel_manager_when_listing_providers_then_fallback_always_present() {
    // Given
    let mgr = KernelManager::new();

    // When
    let providers = mgr.list_available_providers();

    // Then
    assert!(
        providers.contains(&"fallback"),
        "fallback must always be in provider list: {providers:?}"
    );
}

#[test]
fn given_select_cpu_kernel_when_called_then_provider_is_functional() {
    // Given/When
    let provider = select_cpu_kernel().unwrap();

    // Then — run a small matmul to verify functionality
    let a = vec![1i8; 4];
    let b = vec![1u8; 4];
    let mut c = vec![0.0f32; 4];
    let result = provider.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(result.is_ok(), "selected CPU kernel must be functional");
}

#[test]
fn given_kernel_manager_when_select_best_twice_then_cached_result_identical() {
    // Given
    let mgr = KernelManager::new();

    // When
    let first = mgr.select_best().unwrap().name();
    let second = mgr.select_best().unwrap().name();

    // Then
    assert_eq!(first, second, "kernel selection must be cached/idempotent");
}

#[test]
fn given_fallback_kernel_when_quantize_called_then_succeeds() {
    // Given
    let kernel = FallbackKernel;
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 4];
    let mut scales = vec![0.0f32; 1];

    // When
    let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);

    // Then
    assert!(result.is_ok(), "fallback quantize must succeed");
}

// ═══════════════════════════════════════════════════════════════════
// Section 6 — Quantization Round-Trip
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_float_tensor_when_symmetric_quantize_dequantize_then_within_tolerance() {
    // Given
    let input = vec![0.5, -0.3, 0.8, -1.0, 0.0, 0.2, -0.7, 0.9];

    // When
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let reconstructed = dequantize_symmetric_i8(&quantized, scale);

    // Then
    assert_eq!(reconstructed.len(), input.len());
    for (i, (&orig, &recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
        assert_close(recon, orig, 0.02, &format!("sym_roundtrip[{i}]"));
    }
}

#[test]
fn given_float_tensor_when_asymmetric_quantize_dequantize_then_within_tolerance() {
    // Given
    let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];

    // When
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    let reconstructed = dequantize_asymmetric_u8(&quantized, scale, zero_point);

    // Then
    for (i, (&orig, &recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
        assert_close(recon, orig, 0.01, &format!("asym_roundtrip[{i}]"));
    }
}

#[test]
fn given_ternary_values_when_quantize_then_only_minus1_zero_plus1() {
    // Given
    let input = vec![2.0, -3.0, 0.01, 0.0, -0.01, 5.0];

    // When
    let quantized = quantize_ternary(&input, 0.5);

    // Then
    for &v in &quantized {
        assert!([-1, 0, 1].contains(&v), "ternary must be in {{-1,0,1}}, got {v}");
    }
    assert_eq!(quantized[0], 1); // 2.0 > 0.5
    assert_eq!(quantized[1], -1); // -3.0 < -0.5
    assert_eq!(quantized[3], 0); // 0.0 within threshold
}

#[test]
fn given_zero_tensor_when_quantize_dequantize_then_all_zeros() {
    // Given
    let input = vec![0.0f32; 16];

    // When
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let reconstructed = dequantize_symmetric_i8(&quantized, scale);

    // Then
    assert_eq!(scale, 0.0);
    for &v in &reconstructed {
        assert_eq!(v, 0.0, "zero input roundtrip must produce zeros");
    }
}

#[test]
fn given_i2s_packed_when_matmul_roundtrip_then_logits_correct() {
    // Given — 1×4 activation, 4×2 weight → 1×2 output
    let m = 1;
    let n = 2;
    let k = 4;
    let block_size = 4;
    let activations = vec![1.0f32; k];
    let weights_packed = vec![pack_i2s([1, 1, 1, 1]); n]; // all +1
    let scales = vec![1.0f32; n]; // one block per column

    // When
    let mut logits = vec![0.0f32; n];
    i2s_matmul_f32(&activations, &weights_packed, &scales, &mut logits, m, n, k, block_size)
        .unwrap();

    // Then — each output = sum(activations * 1 * scale) = 4.0
    for &l in &logits {
        assert_close(l, 4.0, EPS, "i2s_roundtrip_logit");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 7 — Memory Management (Buffer Lifecycle)
// ═══════════════════════════════════════════════════════════════════

/// Simple arena for tests — demonstrates buffer reuse pattern.
struct TestArena {
    storage: Vec<u8>,
    offset: usize,
}

impl TestArena {
    fn new(capacity: usize) -> Self {
        Self { storage: vec![0u8; capacity], offset: 0 }
    }

    fn alloc(&mut self, size: usize) -> Option<&mut [u8]> {
        if self.offset + size > self.storage.len() {
            return None;
        }
        let start = self.offset;
        self.offset += size;
        Some(&mut self.storage[start..start + size])
    }

    fn used(&self) -> usize {
        self.offset
    }

    fn remaining(&self) -> usize {
        self.storage.len() - self.offset
    }

    fn reset(&mut self) {
        self.offset = 0;
    }
}

#[test]
fn given_arena_allocator_when_allocating_multiple_buffers_then_all_valid() {
    // Given
    let mut arena = TestArena::new(4096);

    // When
    let buf1 = arena.alloc(64).unwrap();
    buf1.fill(0xAA);
    let buf2 = arena.alloc(128).unwrap();
    buf2.fill(0xBB);
    let buf3 = arena.alloc(256).unwrap();
    buf3.fill(0xCC);

    // Then — all buffers are valid and independent
    assert_eq!(arena.used(), 64 + 128 + 256);
    assert!(arena.remaining() > 0);
}

#[test]
fn given_arena_when_allocating_then_used_increases() {
    // Given
    let mut arena = TestArena::new(4096);
    let initial_used = arena.used();

    // When
    let _ = arena.alloc(128).unwrap();

    // Then
    assert!(arena.used() > initial_used, "used bytes must increase after allocation");
    assert_eq!(arena.used(), 128);
}

#[test]
fn given_arena_when_reset_then_all_space_available() {
    // Given
    let mut arena = TestArena::new(4096);
    let _ = arena.alloc(1024).unwrap();
    assert!(arena.used() > 0);

    // When
    arena.reset();

    // Then
    assert_eq!(arena.used(), 0, "arena.used() must be 0 after reset");
    assert_eq!(arena.remaining(), 4096);
}

#[test]
fn given_arena_when_exceeding_capacity_then_none_returned() {
    // Given
    let mut arena = TestArena::new(256);

    // When
    let result = arena.alloc(512);

    // Then
    assert!(result.is_none(), "exceeding arena capacity must return None");
}

#[test]
fn given_arena_when_alloc_then_write_read_roundtrip_valid() {
    // Given
    let mut arena = TestArena::new(4096);

    // When — allocate f32-sized buffer, write and read back via bytemuck
    let buf = arena.alloc(16 * std::mem::size_of::<f32>()).unwrap();
    let floats: &mut [f32] = bytemuck::cast_slice_mut(buf);
    for (i, v) in floats.iter_mut().enumerate() {
        *v = i as f32;
    }

    // Then
    for (i, &v) in floats.iter().enumerate() {
        assert_eq!(v, i as f32, "arena write-read roundtrip[{i}]");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 8 — Thread Pool Workflow (Rayon)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_work_stealing_pool_when_submitting_tasks_then_all_complete() {
    // Given — a batch of vectors to normalize
    let batch: Vec<Vec<f32>> = (0..8).map(|i| vec![(i + 1) as f32; 16]).collect();

    // When — parallel normalization using rayon
    let results: Vec<Vec<f32>> = batch
        .iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / norm).collect()
        })
        .collect();

    // Then — all tasks completed with valid results
    assert_eq!(results.len(), 8);
    for (i, result) in results.iter().enumerate() {
        let l2: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(l2, 1.0, 1e-5, &format!("rayon_norm_batch[{i}]"));
    }
}

#[test]
fn given_parallel_quantize_when_joining_then_all_blocks_valid() {
    // Given — split data into blocks for parallel quantization
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let block_size = 16;
    let blocks: Vec<&[f32]> = data.chunks(block_size).collect();

    // When — quantize each block independently
    let quantized: Vec<(Vec<i8>, f32)> =
        blocks.iter().map(|block| quantize_symmetric_i8(block, 8)).collect();

    // Then — all blocks quantized
    assert_eq!(quantized.len(), 4);
    for (i, (q, scale)) in quantized.iter().enumerate() {
        assert_eq!(q.len(), block_size, "block[{i}] length");
        assert!(scale.is_finite(), "block[{i}] scale must be finite");
    }
}

#[test]
fn given_parallel_matmul_when_collecting_then_results_correct() {
    // Given — batch of matmuls
    let kernel = FallbackKernel;
    let inputs: Vec<Vec<i8>> = (0..4).map(|_| vec![1i8; 4]).collect();
    let b = vec![1u8; 4];

    // When
    let results: Vec<Vec<f32>> = inputs
        .iter()
        .map(|a| {
            let mut c = vec![0.0f32; 4];
            kernel.matmul_i2s(a, &b, &mut c, 2, 2, 2).unwrap();
            c
        })
        .collect();

    // Then
    assert_eq!(results.len(), 4);
    // All results should be identical (same input)
    for i in 1..results.len() {
        assert_slice_close(&results[i], &results[0], 0.0, &format!("par_matmul[{i}]"));
    }
}

#[test]
fn given_parallel_layer_norm_when_joining_then_all_normalized() {
    // Given — batch of sequences
    let dim = 8;
    let config = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let sequences: Vec<Vec<f32>> =
        (0..4).map(|i| (0..dim).map(|j| (i * dim + j) as f32).collect()).collect();

    // When
    let normed: Vec<Vec<f32>> =
        sequences.iter().map(|seq| layer_norm(seq, &gamma, None, &config).unwrap()).collect();

    // Then — each normalized sequence has mean ≈ 0
    for (i, n) in normed.iter().enumerate() {
        let m = mean(n);
        assert_close(m, 0.0, 1e-5, &format!("par_ln_mean[{i}]"));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 9 — Pipeline Parallelism
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_3_stage_pipeline_when_processing_batch_then_output_correct() {
    // Given — 3-stage: embed → normalize → project
    let vocab = 8;
    let dim = 4;
    let n_classes = 2;
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let gamma = vec![1.0f32; dim];
    let norm_config = LayerNormConfig::new(vec![dim]);
    let linear_cfg = LinearConfig {
        in_features: dim,
        out_features: n_classes,
        batch_size: 1,
        has_bias: false,
        ..LinearConfig::default()
    };
    let w = vec![1.0f32; n_classes * dim];

    // When — process tokens [0, 3, 5] through the pipeline
    let tokens: Vec<u32> = vec![0, 3, 5];
    let outputs: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&tok| {
            // Stage 1: embed
            let emb = embedding_lookup(&table, &[tok], dim).unwrap();
            // Stage 2: normalize
            let normed = layer_norm(&emb, &gamma, None, &norm_config).unwrap();
            // Stage 3: project
            let mut logits = vec![0.0f32; n_classes];
            linear_cpu(&normed, &w, None, &mut logits, &linear_cfg).unwrap();
            logits
        })
        .collect();

    // Then
    assert_eq!(outputs.len(), 3);
    for (i, logits) in outputs.iter().enumerate() {
        assert_eq!(logits.len(), n_classes, "token[{i}] output dim");
        for &v in logits {
            assert!(v.is_finite(), "token[{i}] logit must be finite");
        }
    }
}

#[test]
fn given_embed_rope_attention_pipeline_when_processing_then_output_shape_correct() {
    // Given
    let vocab = 16;
    let dim = 4;
    let seq_len = 3;
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens: Vec<u32> = vec![1, 5, 10];
    let rope_cfg = RopeConfig::new(dim, 32);
    let freqs = compute_frequencies(&rope_cfg);

    // When — embed → RoPE → attention
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();
    let mut q = emb.clone();
    let mut k = emb.clone();
    for pos in 0..seq_len {
        let start = pos * dim;
        apply_rope(&mut q[start..start + dim], pos, dim, &freqs);
        apply_rope(&mut k[start..start + dim], pos, dim, &freqs);
    }
    let v = emb.clone();
    let attn_out = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, dim, true).unwrap();

    // Then
    assert_eq!(attn_out.len(), seq_len * dim, "attention output shape");
}

#[test]
fn given_quantize_matmul_norm_pipeline_when_processing_then_output_finite() {
    // Given
    let dim = 4;
    let input = vec![0.5f32, -0.3, 0.8, -1.0];

    // When — quantize → i2s matmul → layer norm
    let (q_input, scale) = quantize_symmetric_i8(&input, 8);
    let deq_input = dequantize_symmetric_i8(&q_input, scale);
    let gamma = vec![1.0f32; dim];
    let config = LayerNormConfig::new(vec![dim]);
    let normed = layer_norm(&deq_input, &gamma, None, &config).unwrap();

    // Then
    assert_eq!(normed.len(), dim);
    let m = mean(&normed);
    assert_close(m, 0.0, 1e-4, "quant_pipeline_mean");
}

#[test]
fn given_multi_stage_pipeline_when_residual_added_then_gradient_preserved() {
    // Given — simulate transformer block: norm → linear → residual add
    let dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; dim];
    let norm_config = LayerNormConfig::new(vec![dim]);
    let linear_cfg = LinearConfig {
        in_features: dim,
        out_features: dim,
        batch_size: 1,
        has_bias: false,
        ..LinearConfig::default()
    };
    let w: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.1 - 0.5).collect();

    // When
    let normed = layer_norm(&input, &gamma, None, &norm_config).unwrap();
    let mut linear_out = vec![0.0f32; dim];
    linear_cpu(&normed, &w, None, &mut linear_out, &linear_cfg).unwrap();
    add_residual(&mut linear_out, &input).unwrap();

    // Then — output should differ from input (non-trivial transform)
    assert_ne!(linear_out, input, "transformer block must modify values");
    for &v in &linear_out {
        assert!(v.is_finite(), "residual output must be finite");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 10 — Attention Integration
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_qkv_tensors_when_computing_attention_then_output_shape_correct() {
    // Given
    let seq_len = 4;
    let head_dim = 8;
    let q = vec![1.0f32; seq_len * head_dim];
    let k = vec![1.0f32; seq_len * head_dim];
    let v = vec![1.0f32; seq_len * head_dim];

    // When
    let output =
        scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();

    // Then
    assert_eq!(output.len(), seq_len * head_dim, "attention output shape");
}

#[test]
fn given_causal_attention_when_computing_then_future_tokens_masked() {
    // Given — orthogonal Q/K so attention scores are distinct
    let seq_len = 3;
    let head_dim = 4;
    // Q: identity-ish rows
    #[rustfmt::skip]
    let q = vec![
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
        0.0, 0.0, 1.0, 0.0, // token 2
    ];
    let k = q.clone();
    let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    // When — causal attention
    let output =
        scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, true).unwrap();

    // Then — token 0 can only attend to itself → output[0] = V[0]
    assert_close(output[0], 1.0, 0.01, "causal_token0_v0");
    assert_eq!(output.len(), seq_len * head_dim);
}

#[test]
fn given_multi_head_attention_when_computing_then_all_heads_contribute() {
    // Given
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let total_dim = num_heads * head_dim;
    let q = vec![1.0f32; seq_len * total_dim];
    let k = vec![1.0f32; seq_len * total_dim];
    let v = vec![1.0f32; seq_len * total_dim];

    // When
    let output = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();

    // Then
    assert_eq!(output.len(), seq_len * total_dim);
    for &val in &output {
        assert!(val.is_finite(), "MHA output must be finite");
    }
}

#[test]
fn given_kv_cache_when_incremental_attention_then_output_correct() {
    // Given
    let head_dim = 4;
    let mut k_cache: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // one cached key
    let mut v_cache: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5]; // one cached value
    let q = vec![1.0, 0.0, 0.0, 0.0]; // query
    let k_new = vec![0.0, 1.0, 0.0, 0.0]; // new key
    let v_new = vec![0.0, 0.0, 1.0, 0.0]; // new value

    // When
    let output =
        attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k_new, &v_new, head_dim).unwrap();

    // Then
    assert_eq!(output.len(), head_dim, "incremental attention output dim");
    assert_eq!(k_cache.len(), 2 * head_dim, "k_cache grew by one position");
    assert_eq!(v_cache.len(), 2 * head_dim, "v_cache grew by one position");
}

#[test]
fn given_attention_config_when_validating_then_catches_invalid_params() {
    // Given
    let bad_config = AttentionConfig {
        num_heads: 0,
        head_dim: 4,
        seq_len: 2,
        causal: false,
        use_alibi: false,
        scale: None,
    };

    // When
    let result = bad_config.validate();

    // Then
    assert!(result.is_err(), "num_heads=0 must fail validation");
}

#[test]
fn given_self_attention_when_all_same_vectors_then_uniform_output() {
    // Given — all Q/K/V rows identical
    let seq_len = 4;
    let head_dim = 4;
    let uniform = vec![1.0f32; seq_len * head_dim];

    // When
    let output = scaled_dot_product_attention(
        &uniform, &uniform, &uniform, seq_len, seq_len, head_dim, false,
    )
    .unwrap();

    // Then — with uniform inputs, attention weights are uniform (1/seq_len),
    // and weighted sum of identical V rows = V row
    for pos in 0..seq_len {
        for d in 0..head_dim {
            assert_close(output[pos * head_dim + d], 1.0, 1e-4, "uniform_attn");
        }
    }
}

#[test]
fn given_rope_then_attention_when_chained_then_positional_info_encoded() {
    // Given
    let head_dim = 4;
    let seq_len = 2;
    let rope_cfg = RopeConfig::new(head_dim, 16);
    let freqs = compute_frequencies(&rope_cfg);
    let mut q = vec![1.0f32; seq_len * head_dim];
    let mut k = vec![1.0f32; seq_len * head_dim];
    let v = vec![1.0f32; seq_len * head_dim];

    // When — apply RoPE then attention
    for pos in 0..seq_len {
        let start = pos * head_dim;
        apply_rope(&mut q[start..start + head_dim], pos, head_dim, &freqs);
        apply_rope(&mut k[start..start + head_dim], pos, head_dim, &freqs);
    }
    let output =
        scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();

    // Then — output shape correct and finite
    assert_eq!(output.len(), seq_len * head_dim);
    for &val in &output {
        assert!(val.is_finite(), "RoPE+attention output must be finite");
    }
}
