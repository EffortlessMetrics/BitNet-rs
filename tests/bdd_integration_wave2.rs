//! BDD Integration Wave 2: End-to-end behavioral validation.
//!
//! Validates kernel dispatch, configuration validation, memory safety,
//! device selection, and error propagation across the inference pipeline.
//! Each test follows Given/When/Then structure.

// ─── Kernel Dispatch ─────────────────────────────────────────────────────────

#[cfg(test)]
mod kernel_dispatch {
    use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

    #[test]
    fn given_cpu_backend_when_selecting_kernel_then_returns_available_provider() {
        // Given: A kernel manager with CPU backend (no GPU feature)
        let mgr = KernelManager::new();

        // When: We select the best kernel
        let best = mgr.select_best();

        // Then: A valid provider is returned and it is available
        let provider = best.expect("CPU backend must yield a provider");
        assert!(provider.is_available());
    }

    #[test]
    fn given_cpu_backend_when_listing_providers_then_includes_fallback() {
        // Given: A kernel manager built on CPU feature
        let mgr = KernelManager::new();

        // When: We list available providers
        let names = mgr.list_available_providers();

        // Then: The fallback kernel is always present
        assert!(
            names.contains(&"fallback"),
            "fallback kernel must always be in the provider list, got: {names:?}"
        );
    }

    #[test]
    fn given_fallback_kernel_when_checking_availability_then_always_true() {
        // Given: A fallback kernel instance
        let fb = FallbackKernel;

        // When: We check if it's available
        // Then: It must be available on any platform
        assert!(fb.is_available());
        assert_eq!(fb.name(), "fallback");
    }

    #[test]
    fn given_cpu_kernel_when_performing_matmul_then_produces_correct_dimensions() {
        // Given: A CPU kernel and correctly sized I2S inputs
        let kernel = bitnet_kernels::select_cpu_kernel().expect("CPU kernel must be available");
        let m = 2;
        let n = 2;
        let k = 4;
        let a = vec![1i8; m * k]; // 2x4
        // Packed 2-bit: each byte holds 4 values. Need k*n/4 = 4*2/4 = 2 bytes
        // but kernel may index up to k*n so provide generous buffer
        let b = vec![0u8; k * n];
        let mut c = vec![0.0f32; m * n];

        // When: We perform matmul
        let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);

        // Then: The result is Ok and output has correct size
        assert!(result.is_ok(), "matmul_i2s must not error: {result:?}");
        assert_eq!(c.len(), m * n);
    }

    #[test]
    fn given_cpu_kernel_when_quantizing_i2s_then_produces_packed_output() {
        // Given: A CPU kernel and float input
        let kernel = bitnet_kernels::select_cpu_kernel().expect("CPU kernel must be available");
        let input = vec![1.0f32, -1.0, 0.5, -0.5];
        let mut packed = vec![0u8; 2]; // 4 values * 2 bits / 8 = 1 byte, pad to 2
        let mut scales = vec![0.0f32; 1];

        // When: We quantize
        let result =
            kernel.quantize(&input, &mut packed, &mut scales, bitnet_common::QuantizationType::I2S);

        // Then: Quantization succeeds
        assert!(result.is_ok(), "I2S quantization must succeed: {result:?}");
    }

    #[test]
    fn given_kernel_manager_when_selecting_twice_then_returns_same_provider() {
        // Given: A kernel manager
        let mgr = KernelManager::new();

        // When: We select best twice
        let name1 = mgr.select_best().unwrap().name();
        let name2 = mgr.select_best().unwrap().name();

        // Then: Same provider is returned (cached selection)
        assert_eq!(name1, name2, "cached selection must be stable");
    }

    #[test]
    fn given_cpu_only_build_when_selecting_gpu_kernel_then_returns_error() {
        // Given: A build without GPU feature
        // When: We attempt to select a GPU kernel
        let result = bitnet_kernels::select_gpu_kernel(0);

        // Then: It returns an error (no GPU provider)
        assert!(result.is_err(), "GPU kernel selection must fail without gpu feature");
    }

    #[test]
    fn given_kernel_provider_when_used_as_trait_object_then_dispatches_correctly() {
        // Given: A KernelProvider behind a trait object
        let provider: Box<dyn KernelProvider> = Box::new(FallbackKernel);

        // When: We call trait methods through the box
        // Then: They work correctly
        assert!(provider.is_available());
        assert!(!provider.name().is_empty());
    }
}

// ─── Configuration Validation ────────────────────────────────────────────────

#[cfg(test)]
mod configuration_validation {
    use bitnet_engine_core::{ConfigError, SessionConfig, VALID_BACKENDS};

    #[test]
    fn given_valid_config_when_validating_then_succeeds() {
        // Given: A fully valid session configuration
        let cfg = SessionConfig {
            model_path: "model.gguf".into(),
            tokenizer_path: "tokenizer.json".into(),
            backend: "cpu".into(),
            max_context: 2048,
            seed: None,
        };

        // When: We validate
        // Then: No error
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn given_empty_model_path_when_validating_then_returns_empty_model_error() {
        // Given: A config with empty model path
        let cfg = SessionConfig {
            model_path: String::new(),
            tokenizer_path: "tokenizer.json".into(),
            backend: "cpu".into(),
            max_context: 512,
            seed: None,
        };

        // When: We validate
        let err = cfg.validate().unwrap_err();

        // Then: EmptyModelPath error is returned
        assert_eq!(err, ConfigError::EmptyModelPath);
    }

    #[test]
    fn given_empty_tokenizer_path_when_validating_then_returns_empty_tokenizer_error() {
        // Given: A config with empty tokenizer path
        let cfg = SessionConfig {
            model_path: "model.gguf".into(),
            tokenizer_path: String::new(),
            backend: "cpu".into(),
            max_context: 512,
            seed: None,
        };

        // When: We validate
        let err = cfg.validate().unwrap_err();

        // Then: EmptyTokenizerPath error
        assert_eq!(err, ConfigError::EmptyTokenizerPath);
    }

    #[test]
    fn given_unsupported_backend_when_validating_then_returns_backend_error() {
        // Given: A config with an unknown backend
        let cfg = SessionConfig {
            model_path: "model.gguf".into(),
            tokenizer_path: "tokenizer.json".into(),
            backend: "quantum".into(),
            max_context: 512,
            seed: None,
        };

        // When: We validate
        let err = cfg.validate().unwrap_err();

        // Then: UnsupportedBackend with the offending value
        assert_eq!(err, ConfigError::UnsupportedBackend("quantum".into()));
    }

    #[test]
    fn given_zero_context_window_when_validating_then_returns_zero_context_error() {
        // Given: A config with max_context = 0
        let cfg = SessionConfig {
            model_path: "model.gguf".into(),
            tokenizer_path: "tokenizer.json".into(),
            backend: "cpu".into(),
            max_context: 0,
            seed: None,
        };

        // When: We validate
        let err = cfg.validate().unwrap_err();

        // Then: ZeroContextWindow error
        assert_eq!(err, ConfigError::ZeroContextWindow);
    }

    #[test]
    fn given_all_valid_backends_when_validating_then_all_accepted() {
        // Given: Each known-valid backend identifier
        for &backend in VALID_BACKENDS {
            let cfg = SessionConfig {
                model_path: "m.gguf".into(),
                tokenizer_path: "t.json".into(),
                backend: backend.into(),
                max_context: 128,
                seed: None,
            };

            // When/Then: Validation succeeds for every valid backend
            assert!(cfg.validate().is_ok(), "backend {backend:?} should be valid");
        }
    }

    #[test]
    fn given_config_error_when_displayed_then_message_is_descriptive() {
        // Given: Each config error variant
        let errors = vec![
            ConfigError::EmptyModelPath,
            ConfigError::EmptyTokenizerPath,
            ConfigError::UnsupportedBackend("test".into()),
            ConfigError::ZeroContextWindow,
        ];

        // When/Then: Each error has a non-empty display message
        for err in &errors {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "error message should be non-empty for {err:?}");
        }
    }
}

// ─── Memory Safety ───────────────────────────────────────────────────────────

#[cfg(test)]
mod memory_safety {
    use bitnet_kernels::cpu::{
        embedding,
        layer_norm::{self, LayerNormConfig},
        quantize, rope,
    };

    #[test]
    fn given_empty_input_when_quantizing_symmetric_then_returns_empty_output() {
        // Given: An empty float slice
        let input: Vec<f32> = vec![];

        // When: We quantize symmetrically
        let (quantized, scale) = quantize::quantize_symmetric_i8(&input, 8);

        // Then: Output is empty and scale is zero
        assert!(quantized.is_empty());
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn given_empty_input_when_quantizing_asymmetric_then_returns_empty_output() {
        // Given: An empty float slice
        let input: Vec<f32> = vec![];

        // When: We quantize asymmetrically
        let (quantized, scale, zp) = quantize::quantize_asymmetric_u8(&input);

        // Then: Output is empty with zero scale
        assert!(quantized.is_empty());
        assert_eq!(scale, 0.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn given_empty_input_when_quantizing_ternary_then_returns_empty_output() {
        // Given: An empty float slice
        let input: Vec<f32> = vec![];

        // When: We quantize to ternary values
        let quantized = quantize::quantize_ternary(&input, 0.5);

        // Then: Output is empty
        assert!(quantized.is_empty());
    }

    #[test]
    fn given_empty_input_when_quantizing_binary_then_returns_empty_output() {
        // Given: An empty float slice
        let input: Vec<f32> = vec![];

        // When: We quantize to binary
        let quantized = quantize::quantize_binary(&input);

        // Then: Output is empty
        assert!(quantized.is_empty());
    }

    #[test]
    fn given_all_zeros_when_quantizing_then_does_not_panic() {
        // Given: An all-zero input
        let input = vec![0.0f32; 16];

        // When: We quantize with various methods
        let (q_sym, scale_sym) = quantize::quantize_symmetric_i8(&input, 8);
        let (q_asym, _, _) = quantize::quantize_asymmetric_u8(&input);

        // Then: No panics, outputs are zero-filled
        assert!(q_sym.iter().all(|&v| v == 0));
        assert_eq!(scale_sym, 0.0);
        assert!(q_asym.iter().all(|&v| v == 0));
    }

    #[test]
    fn given_constant_input_when_computing_quantization_error_then_reports_zero() {
        // Given: Identical original and quantized slices
        let original = vec![1.0f32; 4];
        let quantized = vec![1.0f32; 4];

        // When: We compute the error metrics
        let err = quantize::compute_quantization_error(&original, &quantized);

        // Then: MSE is zero (perfect reconstruction)
        assert_eq!(err.mse, 0.0);
        assert_eq!(err.max_abs_error, 0.0);
    }

    #[test]
    fn given_single_element_when_layer_norming_then_returns_valid_result() {
        // Given: A single-element input with matching gamma
        let input = vec![3.0f32];
        let gamma = vec![1.0f32];
        let config = LayerNormConfig::new(vec![1]);

        // When: We apply layer norm
        let result = layer_norm::layer_norm(&input, &gamma, None, &config);

        // Then: Result is Ok (single-element normalizes to 0)
        assert!(result.is_ok(), "single-element layer_norm must not error: {result:?}");
    }

    #[test]
    fn given_empty_indices_when_looking_up_embedding_then_returns_empty() {
        // Given: A valid embedding table but no indices
        let table = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 embeddings of dim 3
        let indices: Vec<u32> = vec![];

        // When: We do lookup with empty indices
        let result = embedding::embedding_lookup(&table, &indices, 3);

        // Then: Returns Ok with empty output
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn given_rope_config_when_computing_frequencies_then_no_panic() {
        // Given: A valid RoPE config with small dimensions
        let config = rope::RopeConfig::new(8, 32);

        // When: We compute frequencies
        let freqs = rope::compute_frequencies(&config);

        // Then: Frequencies are computed without panic, length = head_dim/2 * max_seq
        assert!(!freqs.is_empty());
    }
}

// ─── Device Selection ────────────────────────────────────────────────────────

#[cfg(test)]
mod device_selection {
    use bitnet_common::Device;
    use bitnet_kernels::device_features;

    #[test]
    fn given_cpu_build_when_checking_gpu_compiled_then_returns_false() {
        // Given: A binary built with --features cpu (no gpu)
        // When: We check gpu_compiled
        let compiled = device_features::gpu_compiled();

        // Then: Returns false (CPU-only build)
        assert!(!compiled, "gpu_compiled() must be false in cpu-only build");
    }

    #[test]
    fn given_cpu_build_when_checking_gpu_runtime_then_returns_false() {
        // Given: A binary built without GPU features
        // When: We check gpu_available_runtime
        let available = device_features::gpu_available_runtime();

        // Then: Returns false
        assert!(!available, "gpu_available_runtime() must be false in cpu-only build");
    }

    #[test]
    fn given_device_cpu_when_checking_predicates_then_is_cpu_true() {
        // Given: A CPU device
        let device = Device::Cpu;

        // When: We check device predicates
        // Then: is_cpu is true, is_cuda is false
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
    }

    #[test]
    fn given_device_cuda_when_checking_predicates_then_is_cuda_true() {
        // Given: A CUDA device (logical, not requiring hardware)
        let device = Device::Cuda(0);

        // When: We check device predicates
        // Then: is_cuda is true, is_cpu is false
        assert!(device.is_cuda());
        assert!(!device.is_cpu());
    }

    #[test]
    fn given_default_device_when_created_then_is_cpu() {
        // Given/When: Default device
        let device = Device::default();

        // Then: Defaults to CPU
        assert!(device.is_cpu());
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn given_device_probe_when_probing_cpu_then_returns_valid_capabilities() {
        // Given: The current CPU
        // When: We probe capabilities
        let caps = bitnet_device_probe::probe_cpu();

        // Then: At least one core is reported
        assert!(caps.core_count >= 1, "must detect at least 1 core");
    }

    #[test]
    fn given_no_gpu_feature_when_selecting_gpu_kernel_then_falls_back_to_error() {
        // Given: No GPU feature compiled
        // When: We try to select GPU kernel
        let result = bitnet_kernels::select_gpu_kernel(0);

        // Then: Error is returned (graceful fallback, not panic)
        assert!(result.is_err());
        let err_msg = format!("{}", result.as_ref().err().unwrap());
        assert!(
            err_msg.contains("provider") || err_msg.contains("kernel") || err_msg.contains("GPU"),
            "error should mention kernel/provider issue: {err_msg}"
        );
    }
}

// ─── Error Propagation ───────────────────────────────────────────────────────

#[cfg(test)]
mod error_propagation {
    use bitnet_common::{BitNetError, InferenceError, KernelError, QuantizationError};

    #[test]
    fn given_kernel_error_when_converted_to_bitnet_error_then_preserves_variant() {
        // Given: A kernel error
        let kernel_err = KernelError::NoProvider;

        // When: Converted to BitNetError
        let err: BitNetError = kernel_err.into();

        // Then: It's a Kernel variant
        assert!(matches!(err, BitNetError::Kernel(KernelError::NoProvider)));
    }

    #[test]
    fn given_quantization_error_when_converted_then_preserves_message() {
        // Given: A quantization error with detail
        let qerr = QuantizationError::InvalidBlockSize { size: 7 };

        // When: Converted and displayed
        let err: BitNetError = qerr.into();
        let msg = err.to_string();

        // Then: The message contains the block size detail
        assert!(msg.contains("7"), "error message should include block size: {msg}");
    }

    #[test]
    fn given_inference_error_when_converted_then_preserves_reason() {
        // Given: An inference error
        let ierr = InferenceError::InvalidInput { reason: "empty prompt".into() };

        // When: Converted to BitNetError
        let err: BitNetError = ierr.into();
        let msg = err.to_string();

        // Then: Reason is preserved in the display
        assert!(msg.contains("empty prompt"), "should contain reason: {msg}");
    }

    #[test]
    fn given_config_validation_error_when_displayed_then_is_actionable() {
        // Given: A configuration error
        let err = BitNetError::Config("backend 'tpu' is not supported".into());

        // When: We display it
        let msg = err.to_string();

        // Then: Message is actionable (mentions what went wrong)
        assert!(msg.contains("tpu"), "config error should mention offending value: {msg}");
    }

    #[test]
    fn given_kernel_execution_failure_when_propagated_then_contains_reason() {
        // Given: A kernel execution failure
        let err = KernelError::ExecutionFailed { reason: "dimension mismatch".into() };
        let bitnet_err: BitNetError = err.into();

        // When: We display it
        let msg = bitnet_err.to_string();

        // Then: The root cause is visible
        assert!(msg.contains("dimension mismatch"), "should preserve root cause: {msg}");
    }

    #[test]
    fn given_valid_matmul_when_kernel_executes_then_output_is_populated() {
        // Given: Correctly sized inputs for a 1x1 matmul (k=4)
        let kernel = bitnet_kernels::select_cpu_kernel().unwrap();
        let a = vec![1i8; 4]; // 1x4
        let b = vec![0u8; 4]; // packed 2-bit: generously sized for 4x1
        let mut c = vec![f32::NAN; 1]; // 1x1 output

        // When: We perform matmul
        let result = kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 4);

        // Then: Output is filled (no NaN remaining)
        assert!(result.is_ok(), "valid matmul should succeed");
        assert!(c[0].is_finite(), "output should be finite, got {}", c[0]);
    }
}

// ─── Engine State Machine ────────────────────────────────────────────────────

#[cfg(test)]
mod engine_state_machine {
    use bitnet_engine_core::{EngineState, EngineStateTracker};

    #[test]
    fn given_new_engine_when_created_then_state_is_idle() {
        // Given/When: A new engine state tracker
        let tracker = EngineStateTracker::new();

        // Then: Initial state is Idle
        assert_eq!(tracker.state(), &EngineState::Idle);
    }

    #[test]
    fn given_idle_engine_when_started_then_transitions_to_running() {
        // Given: An idle engine
        let mut tracker = EngineStateTracker::new();

        // When: We start it
        tracker.start().unwrap();

        // Then: State is Running
        assert_eq!(tracker.state(), &EngineState::Running);
    }

    #[test]
    fn given_running_engine_when_finished_then_transitions_to_done() {
        // Given: A running engine
        let mut tracker = EngineStateTracker::new();
        tracker.start().unwrap();

        // When: We finish it
        tracker.finish().unwrap();

        // Then: State is Done
        assert_eq!(tracker.state(), &EngineState::Done);
    }

    #[test]
    fn given_idle_engine_when_finished_then_returns_error() {
        // Given: An idle engine (not yet started)
        let mut tracker = EngineStateTracker::new();

        // When: We try to finish without starting
        let result = tracker.finish();

        // Then: Returns error about invalid transition
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("cannot"), "error should explain invalid transition: {msg}");
    }

    #[test]
    fn given_done_engine_when_started_then_returns_error() {
        // Given: A finished engine
        let mut tracker = EngineStateTracker::new();
        tracker.start().unwrap();
        tracker.finish().unwrap();

        // When: We try to restart
        let result = tracker.start();

        // Then: Returns error (Done → Running not allowed)
        assert!(result.is_err());
    }
}

// ─── Sampling Pipeline ───────────────────────────────────────────────────────

#[cfg(test)]
mod sampling_pipeline {
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    #[test]
    fn given_greedy_config_when_sampling_then_picks_argmax() {
        // Given: Greedy sampling (temperature 0)
        let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        let mut strategy = SamplingStrategy::new(cfg);
        let logits = vec![0.1f32, 0.9, 0.3, 0.05];

        // When: We sample
        let token = strategy.sample(&logits, &[]).unwrap();

        // Then: The highest logit index is selected
        assert_eq!(token, 1);
    }

    #[test]
    fn given_empty_logits_when_sampling_then_returns_error() {
        // Given: Empty logits
        let cfg = SamplingConfig::default();
        let mut strategy = SamplingStrategy::new(cfg);

        // When: We attempt to sample
        let result = strategy.sample(&[], &[]);

        // Then: Error is returned (not a panic)
        assert!(result.is_err());
    }

    #[test]
    fn given_seeded_config_when_sampling_twice_then_results_are_identical() {
        // Given: Two strategies with the same seed
        let cfg = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
        let mut a = SamplingStrategy::new(cfg.clone());
        let mut b = SamplingStrategy::new(cfg);
        let logits = vec![1.0f32, 2.0, 1.5, 0.8, 2.5, 0.3, 1.1, 0.9];

        // When: Both sample from the same logits
        let token_a = a.sample(&logits, &[]).unwrap();
        let token_b = b.sample(&logits, &[]).unwrap();

        // Then: Results are deterministic
        assert_eq!(token_a, token_b, "same seed must produce same token");
    }

    #[test]
    fn given_high_repetition_penalty_when_token_in_history_then_avoids_repeat() {
        // Given: Very high repetition penalty with token 0 in history
        let cfg = SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 50.0,
            seed: Some(0),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(cfg);
        // Token 0 has highest logit but is penalized due to history
        let logits = vec![10.0f32, 5.0, 0.1];

        // When: We sample with token 0 in history
        let token = strategy.sample(&logits, &[0]).unwrap();

        // Then: Token 0 is avoided, token 1 is selected
        assert_eq!(token, 1);
    }
}

// ─── Stop Criteria ───────────────────────────────────────────────────────────

#[cfg(test)]
mod stop_criteria {
    use bitnet_generation::{StopCriteria, StopReason, check_stop};

    #[test]
    fn given_stop_token_id_when_token_matches_then_stops_immediately() {
        // Given: Stop criteria with token ID 128009
        let criteria =
            StopCriteria { stop_token_ids: vec![128009], max_tokens: 100, ..Default::default() };

        // When: We check with the matching token
        let result = check_stop(&criteria, 128009, &[], "");

        // Then: Stops with StopTokenId reason
        assert_eq!(result, Some(StopReason::StopTokenId(128009)));
    }

    #[test]
    fn given_eos_token_when_produced_then_stops() {
        // Given: Criteria with EOS token
        let criteria =
            StopCriteria { eos_token_id: Some(2), max_tokens: 100, ..Default::default() };

        // When: EOS token is generated
        let result = check_stop(&criteria, 2, &[], "");

        // Then: Stops with EosToken reason
        assert_eq!(result, Some(StopReason::EosToken));
    }

    #[test]
    fn given_max_tokens_reached_when_checked_then_stops() {
        // Given: Criteria with max_tokens = 3
        let criteria = StopCriteria { max_tokens: 3, ..Default::default() };
        let generated = vec![10u32, 20, 30]; // 3 tokens already

        // When: We check with generated count at limit
        let result = check_stop(&criteria, 99, &generated, "");

        // Then: Stops with MaxTokens reason
        assert_eq!(result, Some(StopReason::MaxTokens));
    }

    #[test]
    fn given_stop_string_when_found_in_tail_then_stops() {
        // Given: Stop on "</s>"
        let criteria = StopCriteria {
            stop_strings: vec!["</s>".to_string()],
            max_tokens: 100,
            ..Default::default()
        };

        // When: The decoded tail contains the stop string
        let result = check_stop(&criteria, 99, &[], "Hello world</s>");

        // Then: Stops with StopString reason
        assert_eq!(result, Some(StopReason::StopString("</s>".to_string())));
    }

    #[test]
    fn given_no_stop_conditions_met_when_checked_then_continues() {
        // Given: Criteria that won't trigger
        let criteria = StopCriteria {
            stop_token_ids: vec![128009],
            max_tokens: 100,
            eos_token_id: Some(2),
            stop_strings: vec!["</s>".to_string()],
        };

        // When: No condition is met
        let result = check_stop(&criteria, 42, &[1, 2, 3], "Hello");

        // Then: Returns None (continue generating)
        // Note: EOS check for token_id=2 won't fire since we're checking token 42
        // But generated contains [1,2,3] which is < max_tokens=100
        // The check_stop checks the current token_id, not the history
        // So EOS fires only when current token matches eos_token_id
        assert!(result.is_none(), "no stop condition should trigger for token 42");
    }
}

// ─── Concurrency Config ─────────────────────────────────────────────────────

#[cfg(test)]
mod concurrency_config {
    use bitnet_engine_core::ConcurrencyConfig;

    #[test]
    fn given_default_concurrency_when_checked_then_allows_up_to_four() {
        // Given: Default concurrency config
        let cfg = ConcurrencyConfig::default();

        // When/Then: Allows 0..3, rejects 4+
        assert!(cfg.allows(0));
        assert!(cfg.allows(3));
        assert!(!cfg.allows(4));
        assert!(!cfg.allows(100));
    }

    #[test]
    fn given_custom_concurrency_when_at_limit_then_rejects() {
        // Given: Concurrency limit of 1
        let cfg = ConcurrencyConfig { max_concurrent: 1 };

        // When: One session is active
        // Then: No more allowed
        assert!(cfg.allows(0));
        assert!(!cfg.allows(1));
    }
}

// ─── KV Cache Validation ─────────────────────────────────────────────────────

#[cfg(test)]
mod kv_cache_validation {
    use bitnet_kernels::cpu::kv_cache::{KvCache, KvCacheConfig, KvDtype};

    #[test]
    fn given_valid_kv_config_when_creating_cache_then_succeeds() {
        // Given: A valid KV cache configuration
        let config = KvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 64,
            max_seq_len: 128,
            dtype: KvDtype::F32,
        };

        // When: We create the cache
        let result = KvCache::new(config);

        // Then: Cache is created successfully
        assert!(result.is_ok());
    }

    #[test]
    fn given_zero_layers_when_validating_kv_config_then_returns_error() {
        // Given: A KV config with 0 layers
        let config = KvCacheConfig {
            num_layers: 0,
            num_heads: 4,
            head_dim: 64,
            max_seq_len: 128,
            dtype: KvDtype::F32,
        };

        // When: We validate
        let result = config.validate();

        // Then: Returns error
        assert!(result.is_err());
    }

    #[test]
    fn given_zero_heads_when_validating_kv_config_then_returns_error() {
        // Given: A KV config with 0 heads
        let config = KvCacheConfig {
            num_layers: 2,
            num_heads: 0,
            head_dim: 64,
            max_seq_len: 128,
            dtype: KvDtype::F32,
        };

        // When: We validate
        let result = config.validate();

        // Then: Returns error
        assert!(result.is_err());
    }

    #[test]
    fn given_valid_cache_when_querying_seq_len_then_starts_at_zero() {
        // Given: A freshly created cache
        let config = KvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 64,
            max_seq_len: 128,
            dtype: KvDtype::F32,
        };
        let cache = KvCache::new(config).unwrap();

        // When: We check sequence length for layer 0
        let seq_len = cache.seq_len(0).unwrap();

        // Then: It starts at zero (no tokens cached yet)
        assert_eq!(seq_len, 0);
    }
}

// ─── Logits Transforms ──────────────────────────────────────────────────────

#[cfg(test)]
mod logits_transforms {
    use bitnet_logits::{apply_temperature, apply_top_k, argmax, softmax_in_place};

    #[test]
    fn given_logits_when_applying_temperature_then_scales_correctly() {
        // Given: Logits and temperature = 2.0
        let mut logits = vec![2.0f32, 4.0, 6.0];

        // When: We apply temperature
        apply_temperature(&mut logits, 2.0);

        // Then: Each logit is divided by temperature
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn given_logits_when_softmax_then_sums_to_one() {
        // Given: Some logits
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];

        // When: We apply softmax
        softmax_in_place(&mut logits);

        // Then: They sum to ~1.0
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be ~1.0, got {sum}");
    }

    #[test]
    fn given_logits_when_argmax_then_returns_index_of_max() {
        // Given: Logits with known maximum
        let logits = vec![0.1f32, 0.3, 0.9, 0.2];

        // When: We find argmax
        let idx = argmax(&logits);

        // Then: Index 2 (value 0.9)
        assert_eq!(idx, 2);
    }

    #[test]
    fn given_logits_when_top_k_applied_then_only_k_survive() {
        // Given: 5 logits
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];

        // When: We apply top-k with k=2
        apply_top_k(&mut logits, 2);

        // Then: Only the top 2 values remain; others are -inf
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 2, "only top-2 logits should remain finite");
    }
}

// ─── Session ID Generation ──────────────────────────────────────────────────

#[cfg(test)]
mod session_id_generation {
    use bitnet_engine_core::SessionId;

    #[test]
    fn given_session_id_when_generated_then_is_nonempty() {
        // Given/When: A generated session ID
        let id = SessionId::generate();

        // Then: Non-empty string
        assert!(!id.as_str().is_empty());
    }

    #[test]
    fn given_two_session_ids_when_generated_then_are_unique() {
        // Given/When: Two generated session IDs
        let id1 = SessionId::generate();
        let id2 = SessionId::generate();

        // Then: They are distinct
        assert_ne!(id1.as_str(), id2.as_str(), "session IDs must be unique");
    }
}
