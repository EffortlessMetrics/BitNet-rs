//! BDD Wave 12 — Kernel Lifecycle Integration Tests
//!
//! Given/When/Then scenarios covering:
//! 1. Kernel creation → configuration → execution → teardown
//! 2. Performance tracking across multiple kernel executions
//! 3. Memory pool allocation → use → deallocation lifecycle
//! 4. Error recovery scenarios (OOM, invalid config, timeout)

use std::time::Instant;

use bitnet_common::memory_pool::TensorPool;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::quantize::{
    dequantize_symmetric_i8, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, compute_frequencies};
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider, select_cpu_kernel};

// ── Helpers ────────────────────────────────────────────────────────

const TOL: f32 = 1e-5;

/// Lightweight profiler for tracking kernel execution timings.
struct LifecycleProfiler {
    entries: Vec<(String, u128)>,
}

impl LifecycleProfiler {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn record<F: FnOnce()>(&mut self, name: &str, f: F) {
        let start = Instant::now();
        f();
        self.entries.push((name.to_string(), start.elapsed().as_nanos()));
    }

    fn total_ns(&self) -> u128 {
        self.entries.iter().map(|(_, t)| t).sum()
    }

    fn count(&self) -> usize {
        self.entries.len()
    }

    fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(n, _)| n.as_str()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 1 — Kernel creation → configuration → execution → teardown
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_kernel_manager_when_created_then_providers_available() {
    // Given a fresh KernelManager
    let mgr = KernelManager::new();

    // When listing providers
    let providers = mgr.list_available_providers();

    // Then at least the fallback provider exists
    assert!(!providers.is_empty());
    assert!(providers.contains(&"fallback"));
}

#[test]
fn test_given_kernel_manager_when_best_selected_then_provider_is_available() {
    // Given a KernelManager
    let mgr = KernelManager::new();

    // When selecting the best kernel
    let provider = mgr.select_best().expect("select_best must succeed");

    // Then the provider reports itself as available
    assert!(provider.is_available());
    assert!(!provider.name().is_empty());
}

#[test]
fn test_given_fallback_kernel_when_matmul_executed_then_output_populated() {
    // Given a FallbackKernel and identity-like inputs
    let kernel = FallbackKernel;
    let a = vec![1i8; 16];
    let b = vec![1u8; 16];
    let mut c = vec![0.0f32; 16];

    // When matmul_i2s is executed
    let result = kernel.matmul_i2s(&a, &b, &mut c, 4, 4, 4);

    // Then the operation succeeds and output is non-zero
    assert!(result.is_ok());
    assert!(c.iter().any(|&v| v != 0.0), "output must be non-zero");
}

#[test]
fn test_given_cpu_kernel_when_selected_then_name_is_not_empty() {
    // Given a CPU-only kernel selection
    let kernel = select_cpu_kernel().expect("CPU kernel must be available");

    // When we inspect the name
    let name = kernel.name();

    // Then name is non-empty
    assert!(!name.is_empty());
}

#[test]
fn test_given_linear_config_when_executed_then_output_has_correct_shape() {
    // Given a linear layer configuration
    let cfg = LinearConfig {
        in_features: 8,
        out_features: 4,
        batch_size: 2,
        has_bias: false,
        ..LinearConfig::default()
    };
    let x = vec![1.0f32; 16]; // 2 × 8
    let w = vec![0.5f32; 32]; // 4 × 8
    let mut out = vec![0.0f32; 8]; // 2 × 4

    // When linear_cpu is called
    let result = linear_cpu(&x, &w, None, &mut out, &cfg);

    // Then the operation succeeds with populated output
    assert!(result.is_ok());
    assert!(out.iter().all(|&v| v != 0.0));
}

#[test]
fn test_given_linear_with_bias_when_executed_then_bias_is_applied() {
    // Given a linear layer with bias
    let cfg = LinearConfig {
        in_features: 4,
        out_features: 2,
        batch_size: 1,
        has_bias: true,
        ..LinearConfig::default()
    };
    let x = vec![1.0f32; 4];
    let w = vec![1.0f32; 8]; // 2 × 4
    let bias = vec![10.0f32; 2];
    let mut out_biased = vec![0.0f32; 2];
    let mut out_no_bias = vec![0.0f32; 2];

    // When executed with and without bias
    let cfg_no = LinearConfig { has_bias: false, ..cfg };
    linear_cpu(&x, &w, None, &mut out_no_bias, &cfg_no).unwrap();
    linear_cpu(&x, &w, Some(&bias), &mut out_biased, &cfg).unwrap();

    // Then biased output differs from unbiased by the bias amount
    for i in 0..2 {
        assert!(
            (out_biased[i] - out_no_bias[i] - 10.0).abs() < TOL,
            "bias not applied correctly at index {i}"
        );
    }
}

#[test]
fn test_given_layer_norm_when_applied_then_output_is_normalized() {
    // Given input data and layer norm config
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let cfg = LayerNormConfig { eps: 1e-5, normalized_shape: vec![4], elementwise_affine: true };

    // When layer norm is applied
    let output = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();

    // Then output mean is approximately zero
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 0.01, "mean should be ~0, got {mean}");
}

#[test]
fn test_given_rms_norm_when_applied_then_output_scale_is_bounded() {
    // Given input data and RMS norm params
    let input = vec![3.0f32, 4.0, 0.0, -1.0];
    let weight = vec![1.0f32; 4];
    let cfg = LayerNormConfig { eps: 1e-5, normalized_shape: vec![4], elementwise_affine: true };

    // When RMS norm is applied
    let output = rms_norm(&input, &weight, &cfg).unwrap();

    // Then output values are bounded (unit RMS)
    let rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 0.1, "RMS should be ~1.0, got {rms}");
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — Performance tracking across multiple kernel executions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_kernel_config_when_executed_then_timing_recorded() {
    // Given a profiler and a kernel
    let mut profiler = LifecycleProfiler::new();
    let kernel = FallbackKernel;

    // When a matmul is profiled
    profiler.record("matmul", || {
        let mut c = vec![0.0f32; 4];
        let _ = kernel.matmul_i2s(&[1i8; 4], &[1u8; 4], &mut c, 2, 2, 2);
    });

    // Then timing is recorded and positive
    assert_eq!(profiler.count(), 1);
    assert!(profiler.total_ns() > 0);
}

#[test]
fn test_given_multiple_kernels_when_profiled_then_all_timings_accumulated() {
    // Given a profiler
    let mut profiler = LifecycleProfiler::new();
    let kernel = FallbackKernel;

    // When multiple operations are profiled
    for i in 0..5 {
        profiler.record(&format!("op_{i}"), || {
            let mut c = vec![0.0f32; 4];
            let _ = kernel.matmul_i2s(&[1i8; 4], &[1u8; 4], &mut c, 2, 2, 2);
        });
    }

    // Then all entries are recorded
    assert_eq!(profiler.count(), 5);
    assert_eq!(profiler.names().len(), 5);
    assert!(profiler.total_ns() > 0);
}

#[test]
fn test_given_quantize_and_linear_when_profiled_then_separate_timings() {
    // Given a profiler
    let mut profiler = LifecycleProfiler::new();
    let data = vec![0.5f32, -0.3, 0.8, -1.0];

    // When quantize and linear are profiled separately
    profiler.record("quantize", || {
        let _ = quantize_symmetric_i8(&data, 8);
    });

    let cfg = LinearConfig {
        in_features: 2,
        out_features: 2,
        batch_size: 1,
        has_bias: false,
        ..LinearConfig::default()
    };
    profiler.record("linear", || {
        let mut out = vec![0.0f32; 2];
        let _ = linear_cpu(&[1.0, 2.0], &[1.0, 0.0, 0.0, 1.0], None, &mut out, &cfg);
    });

    // Then both entries exist with names preserved
    assert_eq!(profiler.count(), 2);
    assert_eq!(profiler.names(), vec!["quantize", "linear"]);
}

#[test]
fn test_given_profiler_when_no_ops_then_totals_are_zero() {
    // Given an empty profiler
    let profiler = LifecycleProfiler::new();

    // When nothing is executed
    // Then totals are zero
    assert_eq!(profiler.count(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_given_repeated_executions_when_profiled_then_total_grows_monotonically() {
    // Given a profiler
    let mut profiler = LifecycleProfiler::new();
    let kernel = FallbackKernel;
    let mut prev_total = 0u128;

    // When we add operations one by one
    for i in 0..3 {
        profiler.record(&format!("iter_{i}"), || {
            let mut c = vec![0.0f32; 4];
            let _ = kernel.matmul_i2s(&[1i8; 4], &[1u8; 4], &mut c, 2, 2, 2);
        });

        // Then total grows monotonically
        assert!(profiler.total_ns() >= prev_total);
        prev_total = profiler.total_ns();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — Memory pool allocation → use → deallocation lifecycle
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_memory_pool_when_created_then_stats_are_zero() {
    // Given a freshly created pool
    let pool = TensorPool::new(1024 * 1024);

    // When we query stats
    let stats = pool.stats();

    // Then all counters are zero
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.pooled_bytes, 0);
    assert_eq!(stats.active_bytes, 0);
}

#[test]
fn test_given_pool_when_buffer_allocated_then_miss_counted() {
    // Given an empty pool
    let pool = TensorPool::new(1024 * 1024);

    // When a buffer is allocated for the first time
    let _buf = pool.allocate(256);

    // Then it's a cache miss
    let stats = pool.stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 0);
}

#[test]
fn test_given_pool_when_buffer_dropped_and_reallocated_then_hit_counted() {
    // Given a pool with a previously allocated+dropped buffer
    let pool = TensorPool::new(1024 * 1024);
    let buf = pool.allocate(256);
    drop(buf); // Return to pool

    // When we allocate the same size again
    let _buf2 = pool.allocate(256);

    // Then we get a cache hit
    let stats = pool.stats();
    assert!(stats.hits >= 1, "expected at least 1 hit, got {}", stats.hits);
}

#[test]
fn test_given_pool_when_buffer_used_for_f32_then_data_is_accessible() {
    // Given a pool allocation
    let pool = TensorPool::new(1024 * 1024);
    let mut buf = pool.allocate(16 * std::mem::size_of::<f32>());

    // When we write f32 data into it
    let slice = buf.as_f32_mut_slice();
    for (i, v) in slice.iter_mut().enumerate() {
        *v = i as f32;
    }

    // Then the data is readable back
    let read = buf.as_f32_slice();
    for (i, &v) in read.iter().enumerate() {
        assert!((v - i as f32).abs() < TOL, "mismatch at {i}");
    }
}

#[test]
fn test_given_pool_when_cleared_then_pooled_bytes_reset() {
    // Given a pool with returned buffers
    let pool = TensorPool::new(1024 * 1024);
    let buf = pool.allocate(512);
    drop(buf);
    assert!(pool.stats().pooled_bytes > 0);

    // When the pool is cleared
    pool.clear();

    // Then pooled bytes are zero
    assert_eq!(pool.stats().pooled_bytes, 0);
}

#[test]
fn test_given_pool_when_multiple_sizes_allocated_then_total_allocations_correct() {
    // Given a pool
    let pool = TensorPool::new(1024 * 1024);

    // When allocating different sizes
    let _a = pool.allocate(64);
    let _b = pool.allocate(128);
    let _c = pool.allocate(256);

    // Then total allocations == 3
    let stats = pool.stats();
    assert_eq!(stats.total_allocations(), 3);
}

// ═══════════════════════════════════════════════════════════════════
// Section 4 — Error recovery scenarios
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_linear_with_mismatched_dims_when_executed_then_error_returned() {
    // Given mismatched dimensions (weight cols != input features)
    let cfg = LinearConfig {
        in_features: 8,
        out_features: 4,
        batch_size: 1,
        has_bias: false,
        ..LinearConfig::default()
    };
    // x has 4 elements but in_features says 8
    let x = vec![1.0f32; 4];
    let w = vec![1.0f32; 32]; // 4 × 8
    let mut out = vec![0.0f32; 4];

    // When executing linear_cpu
    let result = linear_cpu(&x, &w, None, &mut out, &cfg);

    // Then an error is returned (not a panic)
    assert!(result.is_err(), "mismatched dims should produce an error");
}

#[test]
fn test_given_quantize_with_empty_input_then_graceful_result() {
    // Given an empty input slice
    let data: Vec<f32> = vec![];

    // When quantizing
    let (quantized, scale) = quantize_symmetric_i8(&data, 8);

    // Then result is empty (no panic)
    assert!(quantized.is_empty());
    assert!(scale == 0.0 || scale.is_finite());
}

#[test]
fn test_given_ternary_quantize_with_all_zeros_then_output_is_all_zeros() {
    // Given all-zero input
    let data = vec![0.0f32; 16];

    // When ternary-quantized
    let packed = quantize_ternary(&data, 0.05);

    // Then all packed values are zero
    assert!(packed.iter().all(|&b| b == 0), "all-zero input => zero packed");
}

#[test]
fn test_given_rope_with_zero_dim_when_applied_then_no_panic() {
    // Given a minimal RoPE config
    let cfg = RopeConfig { head_dim: 4, max_seq_len: 4, base: 10000.0, scaling_factor: 1.0 };

    // When computing frequencies
    let freqs = compute_frequencies(&cfg);

    // Then frequencies are finite and non-empty
    assert!(!freqs.is_empty());
    assert!(freqs.iter().all(|f| f.is_finite()));
}

#[test]
fn test_given_residual_add_with_matched_lengths_then_summed() {
    // Given matched-length vectors
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    let residual = vec![0.1, 0.2, 0.3, 0.4];

    // When adding residual (in-place)
    add_residual(&mut x, &residual).unwrap();

    // Then elements are summed correctly
    assert!((x[0] - 1.1).abs() < TOL);
    assert!((x[3] - 4.4).abs() < TOL);
}

#[test]
fn test_given_residual_scaled_when_applied_then_scale_factor_honored() {
    // Given inputs and a scale factor
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    let residual = vec![10.0, 20.0, 30.0, 40.0];
    let scale = 0.1;

    // When adding scaled residual (in-place)
    add_residual_scaled(&mut x, &residual, scale).unwrap();

    // Then residual is scaled before addition
    assert!((x[0] - 2.0).abs() < TOL); // 1.0 + 10.0 * 0.1
    assert!((x[3] - 8.0).abs() < TOL); // 4.0 + 40.0 * 0.1
}

#[test]
fn test_given_quantize_roundtrip_when_dequantized_then_error_bounded() {
    // Given arbitrary values
    let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();

    // When quantized then dequantized
    let (quantized, scale) = quantize_symmetric_i8(&data, 8);
    let recovered = dequantize_symmetric_i8(&quantized, scale);

    // Then error is bounded by scale
    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(err <= scale + TOL, "roundtrip error at {i}: {err} > scale={scale}");
    }
}

#[test]
fn test_given_pool_stress_when_many_allocations_then_no_leak() {
    // Given a small pool
    let pool = TensorPool::new(4096);

    // When allocating and dropping many buffers
    for _ in 0..100 {
        let buf = pool.allocate(64);
        drop(buf);
    }

    // Then stats reflect correct accounting
    let stats = pool.stats();
    assert_eq!(stats.total_allocations(), 100);
    assert_eq!(stats.active_bytes, 0, "no active allocations after drops");
}
