//! Integration Wave 8: Cross-crate kernel pipeline tests.
//!
//! Validates interactions between `bitnet-kernels`, `bitnet-common`,
//! `bitnet-quantization`, `bitnet-device-probe`, and `bitnet-sampling`
//! to ensure kernel managers, CPU fallback chains, quantization round-trips,
//! device feature detection, SIMD dispatch, and multi-kernel pipelines
//! compose correctly across crate boundaries.

// ─── Kernel Manager & Provider Registration ──────────────────────────────────

#[cfg(test)]
mod kernel_manager_tests {
    use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

    #[test]
    fn kernel_manager_discovers_at_least_one_provider() {
        let mgr = KernelManager::new();
        let best = mgr.select_best();
        assert!(best.is_ok(), "KernelManager must find at least one provider");
    }

    #[test]
    fn kernel_manager_lists_available_providers_non_empty() {
        let mgr = KernelManager::new();
        let names = mgr.list_available_providers();
        assert!(!names.is_empty(), "available providers must be non-empty");
    }

    #[test]
    fn kernel_manager_selected_name_matches_best() {
        let mgr = KernelManager::new();
        let best = mgr.select_best().unwrap();
        let best_name = best.name();
        let cached_name = mgr.selected_provider_name().unwrap();
        assert_eq!(best_name, cached_name);
    }

    #[test]
    fn fallback_kernel_is_always_available() {
        let fb = FallbackKernel;
        assert!(fb.is_available());
        assert_eq!(fb.name(), "fallback");
    }

    #[test]
    fn kernel_provider_trait_object_safety() {
        let provider: Box<dyn KernelProvider> = Box::new(FallbackKernel);
        assert!(provider.is_available());
        assert!(!provider.name().is_empty());
    }
}

// ─── CPU Kernel Fallback Chain ───────────────────────────────────────────────

#[cfg(test)]
mod cpu_fallback_tests {
    use bitnet_kernels::select_cpu_kernel;

    #[test]
    fn select_cpu_kernel_succeeds() {
        let provider = select_cpu_kernel().expect("CPU kernel selection must succeed");
        assert!(provider.is_available());
    }

    #[test]
    fn cpu_kernel_name_is_not_empty() {
        let provider = select_cpu_kernel().unwrap();
        assert!(!provider.name().is_empty());
    }

    #[test]
    fn gpu_kernel_selection_fails_without_gpu_feature() {
        let result = bitnet_kernels::select_gpu_kernel(0);
        assert!(result.is_err(), "GPU kernel selection should fail without gpu feature");
    }
}

// ─── Quantization Round-Trip Through Kernel Pipeline ─────────────────────────

#[cfg(test)]
mod quantization_round_trip_tests {
    use bitnet_common::QuantizationType;
    use bitnet_kernels::{FallbackKernel, KernelProvider};

    #[test]
    fn fallback_kernel_quantize_produces_output() {
        let fb = FallbackKernel;
        let input = vec![1.0f32, -1.0, 0.5, -0.5];
        let mut packed = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];
        let result = fb.quantize(&input, &mut packed, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok(), "quantize must succeed: {result:?}");
    }

    #[test]
    fn fallback_matmul_i2s_does_not_panic() {
        let fb = FallbackKernel;
        let a = vec![1i8, 0, 0, 1];
        let b = vec![0u8; 1];
        let mut c = vec![0.0f32; 4];
        let _ = fb.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    }

    #[test]
    fn quantize_dequantize_round_trip_preserves_sign() {
        use bitnet_quantization::I2SQuantizer;

        let quantizer = I2SQuantizer::new();
        // 32 values to fill at least one I2S block
        let values: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let quantized = quantizer.quantize_weights(&values).unwrap();
        let device = candle_core::Device::Cpu;
        let restored = quantizer.dequantize(&quantized, &device).unwrap();

        let restored_data = restored.to_vec().unwrap();
        for (orig, rest) in values.iter().zip(restored_data.iter()) {
            if *orig > 0.0 {
                assert!(*rest >= 0.0, "positive {orig} became {rest}");
            } else if *orig < 0.0 {
                assert!(*rest <= 0.0, "negative {orig} became {rest}");
            }
        }
    }
}

// ─── Device Feature Detection Consistency ────────────────────────────────────

#[cfg(test)]
mod device_feature_tests {
    use bitnet_common::kernel_registry::{KernelBackend, SimdLevel};
    use bitnet_kernels::device_features;

    #[test]
    fn device_features_simd_matches_capabilities() {
        let simd = device_features::detect_simd_level();
        let caps = device_features::current_kernel_capabilities();
        assert_eq!(
            simd, caps.simd_level,
            "detect_simd_level() and capabilities.simd_level must agree"
        );
    }

    #[test]
    fn kernel_capabilities_cpu_compiled_with_cpu_feature() {
        let caps = device_features::current_kernel_capabilities();
        assert!(caps.cpu_rust, "cpu_rust must be true when cpu feature is on");
    }

    #[test]
    fn device_probe_agrees_with_kernels() {
        let probe = bitnet_device_probe::probe_cpu();
        let simd = device_features::detect_simd_level();

        if probe.has_avx2 {
            assert!(simd >= SimdLevel::Avx2, "probe has AVX2 but SIMD level is {simd:?}");
        }
    }

    #[test]
    fn gpu_not_compiled_without_feature() {
        assert!(
            !device_features::gpu_compiled(),
            "gpu should not be compiled with only --features cpu"
        );
    }

    #[test]
    fn kernel_backend_cpu_does_not_require_gpu() {
        assert!(!KernelBackend::CpuRust.requires_gpu());
    }
}

// ─── Backend Capability Reporting ────────────────────────────────────────────

#[cfg(test)]
mod backend_capability_tests {
    use bitnet_common::{BackendRequest, BackendStartupSummary, select_backend};
    use bitnet_kernels::device_features;

    #[test]
    fn backend_startup_summary_format() {
        let summary = BackendStartupSummary::new("cpu", vec!["cpu-rust".to_string()], "cpu-rust");
        let line = summary.log_line();
        assert!(line.contains("cpu"), "log line: {line}");
        assert!(line.contains("cpu-rust"), "log line: {line}");
    }

    #[test]
    fn select_backend_cpu_request_succeeds() {
        let caps = device_features::current_kernel_capabilities();
        let result = select_backend(BackendRequest::Cpu, &caps);
        assert!(result.is_ok(), "CPU backend selection should succeed: {result:?}");
    }

    #[test]
    fn capabilities_summary_contains_simd_info() {
        let caps = device_features::current_kernel_capabilities();
        let s = caps.summary();
        assert!(s.contains("simd="), "summary: {s}");
        assert!(s.contains("backends="), "summary: {s}");
    }

    #[test]
    fn compiled_backends_includes_cpu() {
        let caps = device_features::current_kernel_capabilities();
        let backends = caps.compiled_backends();
        assert!(
            backends.contains(&bitnet_common::KernelBackend::CpuRust),
            "compiled backends must include CpuRust"
        );
    }
}

// ─── SimdLevel Detection & Kernel Dispatch ───────────────────────────────────

#[cfg(test)]
mod simd_dispatch_tests {
    use bitnet_common::kernel_registry::SimdLevel;
    use bitnet_kernels::device_features;

    #[test]
    fn simd_level_is_at_least_scalar() {
        let level = device_features::detect_simd_level();
        assert!(level >= SimdLevel::Scalar);
    }

    #[test]
    fn simd_level_display_is_non_empty() {
        let level = device_features::detect_simd_level();
        let display = format!("{level}");
        assert!(!display.is_empty());
    }

    #[test]
    fn simd_level_ordering_is_consistent() {
        assert!(SimdLevel::Scalar < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
        assert!(SimdLevel::Neon < SimdLevel::Avx2);
    }
}

// ─── Reduction Operations Chain ──────────────────────────────────────────────

#[cfg(test)]
mod reduction_chain_tests {
    use bitnet_kernels::reduction::{self, ReductionOp};

    #[test]
    fn reduce_sum_then_mean_chain() {
        let data = vec![2.0f32, 4.0, 6.0, 8.0];
        let sum = reduction::reduce_f32(&data, ReductionOp::Sum);
        assert!((sum - 20.0).abs() < 1e-5);

        let mean = reduction::reduce_f32(&data, ReductionOp::Mean);
        assert!((mean - 5.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_max_min_l2norm_chain() {
        let data = vec![3.0f32, -1.0, 4.0, -1.5, 2.0];
        let max_val = reduction::reduce_f32(&data, ReductionOp::Max);
        assert!((max_val - 4.0).abs() < 1e-5);

        let min_val = reduction::reduce_f32(&data, ReductionOp::Min);
        assert!((min_val - (-1.5)).abs() < 1e-5);

        let l2 = reduction::reduce_f32(&data, ReductionOp::L2Norm);
        let expected = (9.0 + 1.0 + 16.0 + 2.25 + 4.0f32).sqrt();
        assert!((l2 - expected).abs() < 1e-4, "l2={l2} expected={expected}");
    }

    #[test]
    fn shaped_reduction_along_axis() {
        use bitnet_kernels::shaped_reduction::{
            ShapedReductionConfig, reduce_f32 as shaped_reduce,
        };

        // 2×3 matrix, reduce along axis 1 (columns → row sums)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
        let result = shaped_reduce(&data, &shape, &config).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-5);
        assert!((result[1] - 15.0).abs() < 1e-5);
    }
}

// ─── Conv1d → Activation → Pooling Pipeline ─────────────────────────────────

#[cfg(test)]
mod conv_activation_pooling_tests {
    use bitnet_kernels::cpu::conv1d::{Conv1dConfig, PaddingMode, conv1d_forward};
    use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};

    #[test]
    fn conv1d_then_global_avg_pool() {
        let config = Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: PaddingMode::Same,
            dilation: 1,
            groups: 1,
            bias: false,
        };
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0f32, 0.0, -1.0];
        let result = conv1d_forward(&input, &weight, None, &config).unwrap();

        let pool_cfg = PoolConfig {
            pool_type: PoolType::GlobalAverage,
            kernel_size: 1,
            stride: 1,
            padding: 0,
        };
        let pooled = PoolingKernel::apply(&result, &pool_cfg).unwrap();
        assert_eq!(pooled.len(), 1, "global avg pool should produce 1 value");
    }

    #[test]
    fn conv1d_then_max_pool() {
        let config = Conv1dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: PaddingMode::Zero(1),
            dilation: 1,
            groups: 1,
            bias: false,
        };
        let input = vec![1.0f32, 3.0, 2.0, 4.0];
        let weight = vec![0.5f32, 1.0, 0.5];
        let conv_out = conv1d_forward(&input, &weight, None, &config).unwrap();

        let pool_cfg =
            PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
        let pooled = PoolingKernel::apply(&conv_out, &pool_cfg).unwrap();
        assert!(!pooled.is_empty(), "max pool output should be non-empty");
    }
}

// ─── Attention → Softmax Pipeline ────────────────────────────────────────────

#[cfg(test)]
mod attention_softmax_tests {
    use bitnet_kernels::cpu::attention::{AttentionConfig, AttentionKernel};
    use bitnet_kernels::cpu::softmax;

    #[test]
    fn multi_head_attention_output_is_valid() {
        let config =
            AttentionConfig { num_heads: 1, head_dim: 4, seq_len: 2, causal: false, scale: None };
        // q, k, v: [seq_len, num_heads * head_dim] = [2, 4] flattened
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = AttentionKernel::multi_head_attention(&q, &k, &v, &config);
        assert!(result.is_ok(), "attention failed: {result:?}");
        let output = result.unwrap();
        assert_eq!(output.len(), 8, "output should match Q dims");
    }

    #[test]
    fn softmax_output_sums_to_one() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = softmax::softmax(&input, 1.0).unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {sum}");
    }

    #[test]
    fn attention_then_softmax_pipeline() {
        let config =
            AttentionConfig { num_heads: 1, head_dim: 4, seq_len: 3, causal: true, scale: None };
        let q = vec![0.1f32; 12];
        let k = vec![0.2f32; 12];
        let v = vec![0.3f32; 12];
        let attn_out = AttentionKernel::multi_head_attention(&q, &k, &v, &config).unwrap();

        let logits_slice = &attn_out[..4];
        let probs = softmax::softmax(logits_slice, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax over attention output should sum to 1");
    }
}

// ─── Deterministic Seed Propagation ──────────────────────────────────────────

#[cfg(test)]
mod deterministic_seed_tests {
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    #[test]
    fn same_seed_same_token_across_calls() {
        let logits = vec![1.0f32, 2.0, 1.5, 0.8, 2.5, 1.2, 0.9, 3.0];
        let cfg = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
        let mut s1 = SamplingStrategy::new(cfg.clone());
        let mut s2 = SamplingStrategy::new(cfg);
        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        assert_eq!(t1, t2, "same seed must produce same token");
    }

    #[test]
    fn greedy_sampling_is_deterministic_without_seed() {
        let logits = vec![0.1f32, 5.0, 0.3, 0.05, 4.9];
        let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        let mut s = SamplingStrategy::new(cfg);
        let token = s.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1, "greedy must pick index of max logit");
    }
}

// ─── Error Propagation ──────────────────────────────────────────────────────

#[cfg(test)]
mod error_propagation_tests {
    use bitnet_common::{BitNetError, KernelError};
    use bitnet_kernels::matmul::GemmConfig;

    #[test]
    fn gemm_config_rejects_zero_dimensions() {
        let result = GemmConfig::new(0, 4, 4);
        assert!(result.is_err(), "zero m dimension should error");
    }

    #[test]
    fn kernel_error_converts_to_bitnet_error() {
        let kerr = KernelError::NoProvider;
        let berr: BitNetError = kerr.into();
        let msg = format!("{berr}");
        assert!(msg.contains("Kernel"), "BitNetError should wrap kernel error: {msg}");
    }

    #[test]
    fn gemm_forward_rejects_mismatched_slices() {
        let config = GemmConfig::new(2, 2, 2).unwrap();
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 4];
        let mut c = vec![0.0f32; 1]; // wrong size
        let result = bitnet_kernels::matmul::gemm_forward(&a, &b, &mut c, &config);
        assert!(result.is_err(), "mismatched c buffer should error");
    }
}

// ─── Fused Kernel Operations ─────────────────────────────────────────────────

#[cfg(test)]
mod fused_kernel_tests {
    use bitnet_kernels::cpu::fusion;

    #[test]
    fn fused_rmsnorm_linear_produces_output() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let weight = vec![1.0f32; 8]; // 2 output × 4 input
        let result = fusion::fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn fused_gelu_linear_produces_finite_output() {
        let input = vec![0.5f32, -0.5, 1.0, -1.0];
        let weight = vec![1.0f32; 8]; // 2 output × 4 input
        let bias = vec![0.0f32; 2];
        let result = fusion::fused_gelu_linear(&input, &weight, &bias);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
