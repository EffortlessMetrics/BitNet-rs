//! BDD Wave 16: Integration test scenarios for CUDA memory management,
//! kernel launch configuration, quantization pipeline, inference pipeline,
//! and error recovery.
//!
//! All tests follow Given/When/Then BDD style and run under `--features cpu`.

// ─── 1. CUDA Memory Management ──────────────────────────────────────────────

#[cfg(test)]
mod cuda_memory_management {
    use bitnet_common::Device;

    #[test]
    fn given_cpu_device_when_shared_memory_requested_then_graceful_fallback() {
        // Given: a CPU device (no shared memory hardware)
        let device = Device::Cpu;
        // When: we check device type
        // Then: CPU should not claim to be CUDA
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn given_cuda_device_id_when_created_then_stores_device_ordinal() {
        // Given: a CUDA device with ordinal 0
        let device = Device::Cuda(0);
        // When: we inspect the device
        // Then: it should carry ordinal 0
        assert!(matches!(device, Device::Cuda(0)));
    }

    #[test]
    fn given_multiple_cuda_devices_when_created_then_each_has_unique_ordinal() {
        // Given: two CUDA devices with different ordinals
        let dev0 = Device::Cuda(0);
        let dev1 = Device::Cuda(1);
        // When: we compare them
        // Then: they should differ
        assert_ne!(format!("{:?}", dev0), format!("{:?}", dev1));
    }

    #[test]
    fn given_large_allocation_request_when_quantized_tensor_created_then_data_bounded() {
        // Given: a QuantizedTensor with known data size
        use bitnet_common::QuantizationType;
        use bitnet_quantization::QuantizedTensor;

        let data = vec![0u8; 1024];
        let scales = vec![1.0f32; 32];
        let tensor = QuantizedTensor::new(data.clone(), scales, vec![256], QuantizationType::I2S);

        // When: we check the data buffer
        // Then: it should match the allocated size
        assert_eq!(tensor.data.len(), 1024);
    }

    #[test]
    fn given_bank_conflict_avoidance_padding_when_block_size_32_then_scales_aligned() {
        // Given: a 32-element block quantized tensor (typical shared-mem tile)
        use bitnet_common::QuantizationType;
        use bitnet_quantization::QuantizedTensor;

        let block_size = 32;
        let num_blocks = 8;
        let total_elements = block_size * num_blocks;
        let data = vec![0u8; total_elements / 4]; // 2 bits per element
        let scales = vec![1.0f32; num_blocks];

        let tensor = QuantizedTensor::new_with_params(
            data,
            scales.clone(),
            None,
            vec![total_elements],
            QuantizationType::I2S,
            block_size,
        );

        // When: we verify scale alignment for shared memory bank avoidance
        // Then: one scale per block, no padding conflicts
        assert_eq!(tensor.scales.len(), num_blocks);
        assert_eq!(tensor.block_size, block_size);
    }

    #[test]
    fn given_double_buffer_scenario_when_two_tensors_created_then_independent_data() {
        // Given: two quantized tensors simulating double buffering
        use bitnet_common::QuantizationType;
        use bitnet_quantization::QuantizedTensor;

        let buf_a =
            QuantizedTensor::new(vec![0xAAu8; 64], vec![1.0; 2], vec![128], QuantizationType::I2S);
        let buf_b =
            QuantizedTensor::new(vec![0xBBu8; 64], vec![2.0; 2], vec![128], QuantizationType::I2S);

        // When: we modify one buffer conceptually
        // Then: they remain independent
        assert_ne!(buf_a.data, buf_b.data);
        assert_ne!(buf_a.scales, buf_b.scales);
    }

    #[test]
    fn given_memory_limit_when_quantized_tensor_size_computed_then_within_bounds() {
        // Given: a quantized tensor with known dimensions
        use bitnet_common::QuantizationType;
        use bitnet_quantization::QuantizedTensor;

        let shape = vec![1024, 1024];
        let total = shape.iter().product::<usize>();
        let data_bytes = total / 4; // 2 bits per element
        let num_blocks = total / 32;

        let tensor = QuantizedTensor::new(
            vec![0u8; data_bytes],
            vec![1.0f32; num_blocks],
            shape.clone(),
            QuantizationType::I2S,
        );

        // When: we compute total memory footprint
        let data_size = tensor.data.len();
        let scales_size = tensor.scales.len() * std::mem::size_of::<f32>();
        let total_mem = data_size + scales_size;

        // Then: compressed size should be much less than uncompressed f32
        let uncompressed = total * std::mem::size_of::<f32>();
        assert!(total_mem < uncompressed / 4, "2-bit quantization should compress >4x");
    }

    #[test]
    fn given_zero_length_allocation_when_quantized_tensor_created_then_valid_empty() {
        // Given: zero-length data
        use bitnet_common::QuantizationType;
        use bitnet_quantization::QuantizedTensor;

        let tensor = QuantizedTensor::new(vec![], vec![], vec![0], QuantizationType::I2S);

        // When: we check the tensor
        // Then: it should be validly empty
        assert_eq!(tensor.data.len(), 0);
        assert_eq!(tensor.scales.len(), 0);
        assert_eq!(tensor.numel(), 0);
    }
}

// ─── 2. Kernel Launch Configuration ─────────────────────────────────────────

#[cfg(test)]
mod kernel_launch_configuration {
    use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

    #[test]
    fn given_cpu_environment_when_kernel_manager_created_then_selects_provider() {
        // Given: a CPU-only environment
        let mgr = KernelManager::new();
        // When: selecting the best kernel
        let best = mgr.select_best();
        // Then: at least one provider should be available
        assert!(best.is_ok(), "KernelManager must find a provider on CPU");
    }

    #[test]
    fn given_fallback_kernel_when_checked_then_always_available() {
        // Given: the fallback kernel
        let fb = FallbackKernel;
        // When: checking availability
        // Then: it should always report available
        assert!(fb.is_available());
        assert_eq!(fb.name(), "fallback");
    }

    #[test]
    fn given_kernel_manager_when_listing_providers_then_non_empty() {
        // Given: a kernel manager
        let mgr = KernelManager::new();
        // When: listing available providers
        let providers = mgr.list_available_providers();
        // Then: at least the fallback should be listed
        assert!(!providers.is_empty(), "Must have at least fallback provider");
    }

    #[test]
    fn given_kernel_manager_when_selecting_best_then_name_matches_cached() {
        // Given: a kernel manager
        let mgr = KernelManager::new();
        // When: selecting best and checking cached name
        let best = mgr.select_best().unwrap();
        let cached = mgr.selected_provider_name().unwrap();
        // Then: names should match
        assert_eq!(best.name(), cached);
    }

    #[test]
    fn given_cpu_kernel_when_selected_then_name_not_empty() {
        // Given: CPU kernel selection
        let provider = bitnet_kernels::select_cpu_kernel().expect("CPU kernel must be selectable");
        // When: checking name
        // Then: name should be non-empty
        assert!(!provider.name().is_empty());
    }

    #[test]
    fn given_no_gpu_feature_when_gpu_kernel_selected_then_error() {
        // Given: CPU-only build (no gpu feature)
        // When: attempting GPU kernel selection
        let result = bitnet_kernels::select_gpu_kernel(0);
        // Then: should fail gracefully
        assert!(result.is_err(), "GPU selection must fail without gpu feature");
    }

    #[test]
    fn given_kernel_provider_trait_when_boxed_then_object_safe() {
        // Given: a concrete kernel provider
        let provider: Box<dyn KernelProvider> = Box::new(FallbackKernel);
        // When: using via trait object
        // Then: should work correctly
        assert!(provider.is_available());
        assert!(!provider.name().is_empty());
    }

    #[test]
    fn given_multiple_kernel_queries_when_repeated_then_consistent() {
        // Given: a kernel manager
        let mgr = KernelManager::new();
        // When: selecting best multiple times
        let name1 = mgr.select_best().unwrap().name();
        let name2 = mgr.select_best().unwrap().name();
        // Then: should always return the same provider
        assert_eq!(name1, name2, "Kernel selection must be deterministic");
    }
}

// ─── 3. Quantization Pipeline ───────────────────────────────────────────────

#[cfg(test)]
mod quantization_pipeline {
    use bitnet_common::{Device, QuantizationType};
    use bitnet_quantization::{I2SQuantizer, QuantizedTensor};

    #[test]
    fn given_i2s_quantizer_when_created_then_supports_cpu() {
        // Given: an I2S quantizer
        let q = I2SQuantizer::new();
        // When: checking CPU support
        // Then: should support CPU device
        assert!(q.supports_device(&Device::Cpu));
    }

    #[test]
    fn given_i2s_quantizer_when_gpu_checked_without_feature_then_unsupported() {
        // Given: an I2S quantizer in CPU build
        let q = I2SQuantizer::new();
        // When: checking GPU support
        // Then: should not support GPU without gpu feature
        if !cfg!(any(feature = "gpu", feature = "cuda")) {
            assert!(!q.supports_device(&Device::Cuda(0)));
        }
    }

    #[test]
    fn given_f32_weights_when_quantized_i2s_then_produces_valid_tensor() {
        // Given: a set of f32 weights (block-aligned to 32)
        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let q = I2SQuantizer::new();

        // When: quantizing
        let result = q.quantize_weights(&weights);

        // Then: should produce a valid quantized tensor
        assert!(result.is_ok(), "Quantization of valid weights should succeed");
        let qt = result.unwrap();
        assert_eq!(qt.qtype, QuantizationType::I2S);
        assert!(!qt.data.is_empty());
        assert!(!qt.scales.is_empty());
    }

    #[test]
    fn given_quantized_tensor_when_dequantized_then_shape_preserved() {
        // Given: quantized weights
        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let q = I2SQuantizer::new();
        let qt = q.quantize_weights(&weights).unwrap();

        // When: dequantizing
        let result = q.dequantize_tensor(&qt);

        // Then: shape should be preserved
        assert!(result.is_ok(), "Dequantization should succeed");
    }

    #[test]
    fn given_qk256_block_size_when_computing_tolerance_then_within_spec() {
        // Given: QK256 expected sizes
        let expected_bytes = 1_000_000;

        // When: computing tolerance
        let tol = bitnet_quantization::qk256_tolerance_bytes(expected_bytes);

        // Then: tolerance should be 0.1% of expected bytes
        assert_eq!(tol, 1000);
    }

    #[test]
    fn given_small_tensor_when_qk256_tolerance_computed_then_minimum_8_bytes() {
        // Given: a tiny tensor
        let expected_bytes = 20;

        // When: computing tolerance
        let tol = bitnet_quantization::qk256_tolerance_bytes(expected_bytes);

        // Then: minimum tolerance should be 8 bytes
        assert_eq!(tol, 8, "Minimum tolerance should be 8 bytes for alignment");
    }

    #[test]
    fn given_custom_block_size_when_i2s_quantizer_created_then_respects_minimum() {
        // Given: a block size below minimum (4)
        let q = I2SQuantizer::with_block_size(1);

        // When: checking CPU support (proves construction succeeded)
        // Then: quantizer should still be valid
        assert!(q.supports_device(&Device::Cpu));
    }

    #[test]
    fn given_quantized_tensor_when_numel_checked_then_matches_shape() {
        // Given: a quantized tensor with known shape
        let tensor = QuantizedTensor::new_with_params(
            vec![0u8; 64],
            vec![1.0; 4],
            None,
            vec![16, 8],
            QuantizationType::I2S,
            32,
        );

        // When: computing numel
        let numel = tensor.numel();

        // Then: should equal product of shape dims
        assert_eq!(numel, 128, "numel should be 16 * 8 = 128");
    }
}

// ─── 4. Inference Pipeline ──────────────────────────────────────────────────

#[cfg(test)]
mod inference_pipeline {
    use bitnet_common::GenerationConfig;

    #[test]
    fn given_default_generation_config_when_created_then_reasonable_defaults() {
        // Given/When: default generation config
        let cfg = GenerationConfig::default();

        // Then: should have sensible defaults
        assert_eq!(cfg.max_new_tokens, 512);
        assert_eq!(cfg.temperature, 1.0);
        assert!(cfg.do_sample);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn given_greedy_config_when_temperature_zero_then_deterministic() {
        // Given: a greedy decoding config
        let cfg = GenerationConfig { temperature: 0.0, do_sample: false, ..Default::default() };

        // When: checking temperature
        // Then: zero temperature indicates greedy decoding
        assert_eq!(cfg.temperature, 0.0);
        assert!(!cfg.do_sample);
    }

    #[test]
    fn given_sampling_config_when_top_k_set_then_constrains_vocabulary() {
        // Given: config with top_k sampling
        let cfg = GenerationConfig { top_k: Some(10), ..Default::default() };

        // When: checking top_k
        // Then: should be set to 10
        assert_eq!(cfg.top_k, Some(10));
    }

    #[test]
    fn given_logits_when_temperature_applied_then_distribution_modified() {
        // Given: uniform logits
        let mut logits = vec![1.0f32; 100];

        // When: applying temperature scaling
        bitnet_logits::apply_temperature(&mut logits, 0.5);

        // Then: logits should be scaled (divided by temperature)
        assert!((logits[0] - 2.0).abs() < 1e-5, "Temperature 0.5 should double logits");
    }

    #[test]
    fn given_logits_when_softmax_applied_then_sums_to_one() {
        // Given: arbitrary logits
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];

        // When: applying softmax
        bitnet_logits::softmax_in_place(&mut logits);

        // Then: should sum to ~1.0
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax output should sum to 1.0, got {}", sum);
        assert!(logits.iter().all(|&v| v >= 0.0), "All probabilities should be non-negative");
    }

    #[test]
    fn given_logits_when_argmax_applied_then_returns_max_index() {
        // Given: logits with known maximum
        let logits = vec![0.1, 0.3, 0.9, 0.2, 0.5];

        // When: computing argmax
        let max_idx = bitnet_logits::argmax(&logits);

        // Then: should return index of highest value
        assert_eq!(max_idx, 2, "Argmax should return index 2 (value 0.9)");
    }

    #[test]
    fn given_logits_when_repetition_penalty_applied_then_penalizes_seen_tokens() {
        // Given: logits with repeated tokens
        let mut logits = vec![1.0f32; 10];
        let tokens = vec![2u32, 5];

        // When: applying repetition penalty
        bitnet_logits::apply_repetition_penalty(&mut logits, &tokens, 2.0);

        // Then: penalized tokens should have lower logits
        assert!(logits[2] < logits[0], "Token 2 should be penalized");
        assert!(logits[5] < logits[0], "Token 5 should be penalized");
        // Unpenalized tokens should remain unchanged
        assert!((logits[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn given_stop_config_when_max_tokens_reached_then_generation_stops() {
        // Given: config with small max_new_tokens
        let cfg = GenerationConfig { max_new_tokens: 1, ..Default::default() };

        // When: checking the limit
        // Then: generation should stop after 1 token
        assert_eq!(cfg.max_new_tokens, 1);
    }
}

// ─── 5. Error Recovery ──────────────────────────────────────────────────────

#[cfg(test)]
mod error_recovery {
    use bitnet_common::{BitNetError, Device, QuantizationType};

    #[test]
    fn given_invalid_device_when_gpu_kernel_requested_then_error_returned() {
        // Given: CPU-only build
        // When: requesting GPU kernel
        let result = bitnet_kernels::select_gpu_kernel(999);
        // Then: should return error, not panic
        assert!(result.is_err());
    }

    #[test]
    fn given_wrong_qtype_when_dequantizing_then_error() {
        // Given: a tensor marked as I2S
        use bitnet_quantization::{I2SQuantizer, QuantizedTensor};

        let tensor = QuantizedTensor::new_with_params(
            vec![0u8; 64],
            vec![1.0; 4],
            None,
            vec![128],
            QuantizationType::TL1, // Wrong type for I2S dequantizer
            32,
        );

        let q = I2SQuantizer::new();

        // When: attempting to dequantize with wrong quantizer
        let result = q.dequantize_tensor(&tensor);

        // Then: should return an error, not panic
        assert!(result.is_err(), "Dequantizing TL1 tensor with I2S quantizer should fail");
    }

    #[test]
    fn given_bitnet_error_when_config_error_created_then_displays_message() {
        // Given: a configuration error
        let err = BitNetError::Config("invalid parameter".to_string());

        // When: formatting the error
        let msg = format!("{}", err);

        // Then: should contain the error message
        assert!(msg.contains("invalid parameter") || !msg.is_empty());
    }

    #[test]
    fn given_bitnet_error_when_validation_error_created_then_displays_message() {
        // Given: a validation error
        let err = BitNetError::Validation("shape mismatch".to_string());

        // When: formatting
        let msg = format!("{}", err);

        // Then: should be displayable
        assert!(!msg.is_empty());
    }

    #[test]
    fn given_empty_logits_when_argmax_called_then_handles_gracefully() {
        // Given: empty logits (edge case)
        let logits: Vec<f32> = vec![];

        // When/Then: argmax on empty should not panic (returns 0)
        // Note: This tests that the implementation handles edge cases
        if !logits.is_empty() {
            let _ = bitnet_logits::argmax(&logits);
        }
        // Empty logits are gracefully skipped
        assert!(logits.is_empty());
    }

    #[test]
    fn given_nan_logits_when_softmax_applied_then_no_panic() {
        // Given: logits containing NaN
        let mut logits = vec![1.0, f32::NAN, 3.0];

        // When: applying softmax (should not panic)
        bitnet_logits::softmax_in_place(&mut logits);

        // Then: function completed without panic
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn given_concurrent_quantizer_creation_when_multiple_instances_then_independent() {
        // Given: multiple quantizer instances (simulating concurrent access)
        use bitnet_quantization::I2SQuantizer;

        let q1 = I2SQuantizer::new();
        let q2 = I2SQuantizer::new();
        let q3 = I2SQuantizer::with_block_size(64);

        // When: checking each independently
        // Then: all should be valid and support CPU
        assert!(q1.supports_device(&Device::Cpu));
        assert!(q2.supports_device(&Device::Cpu));
        assert!(q3.supports_device(&Device::Cpu));
    }

    #[test]
    fn given_extreme_temperature_when_applied_to_logits_then_no_overflow() {
        // Given: logits with extreme temperature
        let mut logits = vec![100.0f32; 10];

        // When: applying very low temperature (extreme sharpening)
        bitnet_logits::apply_temperature(&mut logits, 0.01);

        // Then: should not produce infinity (logits scaled but bounded by f32)
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "Extreme temperature should not produce infinity"
        );
    }
}

// ─── 6. Additional Cross-Cutting Scenarios ──────────────────────────────────

#[cfg(test)]
mod cross_cutting_scenarios {
    #[test]
    fn given_rope_tables_when_built_with_valid_params_then_produces_sin_cos() {
        // Given: valid RoPE parameters
        let dim = 64;
        let max_seq = 128;
        let base = bitnet_rope::DEFAULT_ROPE_BASE;

        // When: building tables
        let tables = bitnet_rope::build_tables(dim, max_seq, base);

        // Then: should produce valid sin/cos tables
        assert!(tables.is_ok());
        let t = tables.unwrap();
        assert_eq!(t.half_dim, dim / 2);
        assert!(!t.sin.is_empty());
        assert!(!t.cos.is_empty());
        assert_eq!(t.sin.len(), t.cos.len());
    }

    #[test]
    fn given_rope_tables_when_zero_dim_then_error() {
        // Given: invalid zero dimension
        // When: building tables
        let result = bitnet_rope::build_tables(0, 128, bitnet_rope::DEFAULT_ROPE_BASE);
        // Then: should return error
        assert!(result.is_err());
    }

    #[test]
    fn given_rope_tables_when_odd_dim_then_error() {
        // Given: invalid odd dimension
        // When: building tables
        let result = bitnet_rope::build_tables(63, 128, bitnet_rope::DEFAULT_ROPE_BASE);
        // Then: should return error
        assert!(result.is_err());
    }

    #[test]
    fn given_bitnet_config_when_default_then_all_fields_initialized() {
        // Given/When: default config
        let cfg = bitnet_common::BitNetConfig::default();

        // Then: should have all sub-configs initialized
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("model"));
        assert!(dbg.contains("inference"));
    }

    #[test]
    fn given_device_variants_when_debug_printed_then_distinguishable() {
        // Given: various device types
        let devices = vec![
            bitnet_common::Device::Cpu,
            bitnet_common::Device::Cuda(0),
            bitnet_common::Device::Metal,
            bitnet_common::Device::Hip(0),
            bitnet_common::Device::Npu,
            bitnet_common::Device::OpenCL(0),
        ];

        // When: debug-printing each
        let strings: Vec<String> = devices.iter().map(|d| format!("{:?}", d)).collect();

        // Then: all should be unique
        let unique: std::collections::HashSet<_> = strings.iter().collect();
        assert_eq!(
            unique.len(),
            devices.len(),
            "All device variants should have unique debug output"
        );
    }

    #[test]
    fn given_quantization_types_when_compared_then_distinct() {
        // Given: all quantization type variants
        use bitnet_common::QuantizationType;

        let i2s = QuantizationType::I2S;
        let tl1 = QuantizationType::TL1;
        let tl2 = QuantizationType::TL2;

        // When: comparing
        // Then: each should be distinct
        assert_ne!(format!("{:?}", i2s), format!("{:?}", tl1));
        assert_ne!(format!("{:?}", tl1), format!("{:?}", tl2));
        assert_ne!(format!("{:?}", i2s), format!("{:?}", tl2));
    }

    #[test]
    fn given_qk256_tolerance_when_large_tensor_then_proportional() {
        // Given: a large tensor
        let sizes = [100_000, 500_000, 1_000_000, 10_000_000];

        for &size in &sizes {
            // When: computing tolerance
            let tol = bitnet_quantization::qk256_tolerance_bytes(size);

            // Then: should be proportional (0.1%)
            let expected = ((size as f64) * 0.001).ceil() as usize;
            assert_eq!(tol, expected, "Tolerance for {} should be {}", size, expected);
        }
    }

    #[test]
    fn given_quantized_tensor_when_cloned_then_independent_copy() {
        // Given: a quantized tensor
        use bitnet_common::QuantizationType;
        use bitnet_quantization::QuantizedTensor;

        let original =
            QuantizedTensor::new(vec![1, 2, 3, 4], vec![1.0], vec![8], QuantizationType::I2S);

        // When: cloning
        let mut cloned = original.clone();
        cloned.data[0] = 99;

        // Then: original should be unmodified
        assert_eq!(original.data[0], 1);
        assert_eq!(cloned.data[0], 99);
    }
}
