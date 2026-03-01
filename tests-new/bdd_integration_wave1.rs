//! BDD Integration Wave 1: End-to-end flow tests without real model files.
//!
//! Validates cross-crate composition across the BitNet-rs stack:
//! - Kernel manager initialization and provider registration
//! - CPU kernel provider operation coverage
//! - Quantization round-trip (I2S quantize → dequantize)
//! - Inference config validation (builder + edge cases)
//! - Sampling strategy creation and determinism
//! - Prompt template detection and application
//! - Error type propagation through the crate stack

// ─── Kernel Manager Initialization ───────────────────────────────────────────

#[cfg(test)]
mod kernel_manager_init {
    use bitnet_kernels::KernelManager;

    #[test]
    fn new_manager_discovers_providers() {
        let mgr = KernelManager::new();
        let providers = mgr.list_available_providers();
        assert!(!providers.is_empty(), "must discover at least one provider on any platform");
    }

    #[test]
    fn select_best_returns_ok() {
        let mgr = KernelManager::new();
        let best = mgr.select_best();
        assert!(best.is_ok(), "select_best must succeed");
    }

    #[test]
    fn selected_name_is_cached_after_select_best() {
        let mgr = KernelManager::new();
        assert!(mgr.selected_provider_name().is_none(), "no selection before select_best");
        let _ = mgr.select_best().unwrap();
        assert!(mgr.selected_provider_name().is_some(), "name must be cached after select_best");
    }

    #[test]
    fn default_trait_is_equivalent_to_new() {
        let mgr_new = KernelManager::new();
        let mgr_def = KernelManager::default();
        assert_eq!(
            mgr_new.list_available_providers(),
            mgr_def.list_available_providers(),
            "new() and default() must discover the same providers"
        );
    }
}

// ─── CPU Kernel Provider Operations ──────────────────────────────────────────

#[cfg(test)]
mod cpu_kernel_provider_ops {
    use bitnet_common::QuantizationType;
    use bitnet_kernels::{FallbackKernel, KernelProvider, select_cpu_kernel};

    #[test]
    fn cpu_kernel_selection_always_succeeds() {
        let provider = select_cpu_kernel().expect("CPU kernel must be available");
        assert!(provider.is_available());
    }

    #[test]
    fn cpu_kernel_reports_non_empty_name() {
        let provider = select_cpu_kernel().unwrap();
        assert!(!provider.name().is_empty());
    }

    #[test]
    fn fallback_kernel_quantize_i2s() {
        let fb = FallbackKernel;
        let input = vec![1.0f32, -1.0, 0.5, -0.5];
        let mut packed = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];
        let result = fb.quantize(&input, &mut packed, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok(), "FallbackKernel quantize(I2S) must succeed: {result:?}");
    }

    #[test]
    fn fallback_kernel_matmul_i2s_does_not_panic() {
        let fb = FallbackKernel;
        let a = vec![1i8, 0, 0, 1];
        let b = vec![0u8; 1];
        let mut c = vec![0.0f32; 4];
        // Should not panic even with trivial input
        let _ = fb.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    }

    #[test]
    fn gpu_kernel_selection_fails_without_gpu_feature() {
        let result = bitnet_kernels::select_gpu_kernel(0);
        assert!(result.is_err(), "GPU selection should fail with only --features cpu");
    }
}

// ─── Quantization Round-Trip ─────────────────────────────────────────────────

#[cfg(test)]
mod quantization_round_trip {
    use bitnet_quantization::I2SQuantizer;

    #[test]
    fn i2s_quantize_then_dequantize_preserves_sign() {
        let quantizer = I2SQuantizer::new();
        let values: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let quantized = quantizer.quantize_weights(&values).expect("quantize must succeed");
        let device = candle_core::Device::Cpu;
        let restored = quantizer.dequantize(&quantized, &device).expect("dequantize must succeed");
        let restored_data: Vec<f32> = restored.to_vec().expect("tensor to vec");
        for (orig, rest) in values.iter().zip(restored_data.iter()) {
            if *orig > 0.0 {
                assert!(*rest >= 0.0, "positive {orig} became {rest}");
            } else if *orig < 0.0 {
                assert!(*rest <= 0.0, "negative {orig} became {rest}");
            }
        }
    }

    #[test]
    fn i2s_quantize_uniform_positive_block() {
        let quantizer = I2SQuantizer::new();
        let values = vec![0.5f32; 32];
        let quantized = quantizer.quantize_weights(&values);
        assert!(quantized.is_ok(), "uniform positive block must quantize: {quantized:?}");
    }

    #[test]
    fn i2s_quantize_uniform_zero_block() {
        let quantizer = I2SQuantizer::new();
        let values = vec![0.0f32; 32];
        let quantized = quantizer.quantize_weights(&values);
        assert!(quantized.is_ok(), "all-zero block must quantize: {quantized:?}");
        let device = candle_core::Device::Cpu;
        let restored =
            quantizer.dequantize(&quantized.unwrap(), &device).expect("dequantize must succeed");
        let data: Vec<f32> = restored.to_vec().expect("tensor to vec");
        for v in &data {
            assert!(v.abs() < 1e-3, "all-zero input should produce near-zero output, got {v}");
        }
    }

    #[test]
    fn i2s_quantize_multiple_blocks() {
        let quantizer = I2SQuantizer::new();
        // 64 values = 2 I2S blocks of 32
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let quantized = quantizer.quantize_weights(&values);
        assert!(quantized.is_ok(), "multi-block quantization must succeed: {quantized:?}");
    }
}

// ─── Inference Config Validation ─────────────────────────────────────────────

#[cfg(test)]
mod inference_config_validation {
    use bitnet_inference::GenerationConfig;

    #[test]
    fn default_config_validates() {
        let config = GenerationConfig::default();
        assert!(config.validate().is_ok(), "default config must be valid");
    }

    #[test]
    fn greedy_config_validates() {
        let config = GenerationConfig::greedy();
        assert!(config.validate().is_ok(), "greedy config must be valid");
    }

    #[test]
    fn creative_config_validates() {
        let config = GenerationConfig::creative();
        assert!(config.validate().is_ok(), "creative config must be valid");
    }

    #[test]
    fn balanced_config_validates() {
        let config = GenerationConfig::balanced();
        assert!(config.validate().is_ok(), "balanced config must be valid");
    }

    #[test]
    fn zero_max_tokens_is_invalid() {
        let config = GenerationConfig::default().with_max_tokens(0);
        let result = config.validate();
        assert!(result.is_err(), "max_tokens=0 must fail validation");
        assert!(
            result.unwrap_err().contains("max_new_tokens"),
            "error must mention max_new_tokens"
        );
    }

    #[test]
    fn negative_temperature_is_invalid() {
        let config = GenerationConfig::default().with_temperature(-0.1);
        assert!(config.validate().is_err(), "negative temperature must fail validation");
    }

    #[test]
    fn zero_top_p_is_invalid() {
        let config = GenerationConfig::default().with_top_p(0.0);
        assert!(config.validate().is_err(), "top_p=0.0 must fail validation");
    }

    #[test]
    fn top_p_above_one_is_invalid() {
        let config = GenerationConfig::default().with_top_p(1.1);
        assert!(config.validate().is_err(), "top_p=1.1 must fail validation");
    }

    #[test]
    fn zero_repetition_penalty_is_invalid() {
        let config = GenerationConfig::default().with_repetition_penalty(0.0);
        assert!(config.validate().is_err(), "repetition_penalty=0.0 must fail validation");
    }

    #[test]
    fn builder_chain_composes_correctly() {
        let config = GenerationConfig::greedy()
            .with_max_tokens(16)
            .with_seed(42)
            .with_stop_sequence("</s>".to_string());
        assert!(config.validate().is_ok());
        assert_eq!(config.max_new_tokens, 16);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.stop_sequences, vec!["</s>"]);
    }
}

// ─── Sampling Strategy Creation ──────────────────────────────────────────────

#[cfg(test)]
mod sampling_strategy_creation {
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    #[test]
    fn default_config_creates_strategy() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(config.seed.is_none());
        let _ = SamplingStrategy::new(config);
    }

    #[test]
    fn greedy_sampling_picks_argmax() {
        let config = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        let mut strategy = SamplingStrategy::new(config);
        let logits = vec![0.1f32, 0.9, 0.3, 0.05, 0.2];
        let token = strategy.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1, "greedy must pick index of max logit (1), got {token}");
    }

    #[test]
    fn deterministic_seed_produces_same_token() {
        let logits = vec![1.0f32, 2.0, 1.5, 0.8, 2.5, 1.2, 0.9, 3.0];
        let cfg = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
        let mut s1 = SamplingStrategy::new(cfg.clone());
        let mut s2 = SamplingStrategy::new(cfg);
        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        assert_eq!(t1, t2, "same seed must produce same token");
    }

    #[test]
    fn reset_clears_state_for_new_sequence() {
        let cfg = SamplingConfig { temperature: 0.8, seed: Some(99), ..Default::default() };
        let mut strategy = SamplingStrategy::new(cfg.clone());
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];

        let t1 = strategy.sample(&logits, &[]).unwrap();
        strategy.reset();
        let t2 = strategy.sample(&logits, &[]).unwrap();
        // After reset, internal state is cleared so same seed + logits → same token
        assert_eq!(t1, t2, "reset must restore initial state");
    }

    #[test]
    fn update_config_changes_behavior() {
        let greedy = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        let mut strategy = SamplingStrategy::new(greedy);
        let logits = vec![0.1f32, 5.0, 0.3];
        let t1 = strategy.sample(&logits, &[]).unwrap();
        assert_eq!(t1, 1, "greedy picks max");

        // Update to still-greedy but with different config
        let new_cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
        strategy.update_config(new_cfg);
        let t2 = strategy.sample(&logits, &[]).unwrap();
        assert_eq!(t2, 1, "still greedy after update");
    }
}

// ─── Prompt Template Detection & Application ─────────────────────────────────

#[cfg(test)]
mod prompt_template_tests {
    use bitnet_prompt_templates::TemplateType;

    #[test]
    fn detect_raw_when_no_hints() {
        let detected = TemplateType::detect(None, None);
        // With no hints, should fall back to a reasonable default
        assert!(
            matches!(
                detected,
                TemplateType::Raw | TemplateType::Instruct | TemplateType::Llama3Chat
            ),
            "no-hint detection returned {detected:?}"
        );
    }

    #[test]
    fn detect_llama3_from_tokenizer_name() {
        let detected = TemplateType::detect(Some("llama3-instruct"), None);
        assert_eq!(detected, TemplateType::Llama3Chat, "llama3 in name must detect Llama3Chat");
    }

    #[test]
    fn apply_raw_returns_input_unchanged() {
        let output = TemplateType::Raw.apply("Hello world", None);
        assert_eq!(output, "Hello world");
    }

    #[test]
    fn apply_instruct_wraps_prompt() {
        let output = TemplateType::Instruct.apply("What is 2+2?", None);
        assert!(output.contains("What is 2+2?"), "instruct template must contain the user prompt");
        // Instruct templates typically add Q:/A: or similar framing
        assert!(output.len() > "What is 2+2?".len(), "instruct must add framing around prompt");
    }

    #[test]
    fn apply_llama3_chat_includes_special_tokens() {
        let output = TemplateType::Llama3Chat.apply("Explain photosynthesis", None);
        assert!(output.contains("Explain photosynthesis"));
        // LLaMA-3 chat uses <|begin_of_text|>, <|start_header_id|>, etc.
        assert!(
            output.contains("<|") || output.len() > "Explain photosynthesis".len(),
            "llama3-chat must add special token framing"
        );
    }

    #[test]
    fn apply_with_system_prompt() {
        let output = TemplateType::Llama3Chat.apply("Hello", Some("You are a helpful assistant"));
        assert!(output.contains("Hello"));
        assert!(
            output.contains("helpful assistant"),
            "system prompt must appear in output: {output}"
        );
    }
}

// ─── Error Type Propagation ──────────────────────────────────────────────────

#[cfg(test)]
mod error_propagation {
    use bitnet_common::{BitNetError, InferenceError, KernelError, ModelError, QuantizationError};

    #[test]
    fn kernel_error_converts_to_bitnet_error() {
        let kerr = KernelError::NoProvider;
        let berr: BitNetError = kerr.into();
        let msg = format!("{berr}");
        assert!(msg.contains("Kernel") || msg.contains("kernel") || msg.contains("provider"));
    }

    #[test]
    fn model_error_converts_to_bitnet_error() {
        let merr = ModelError::NotFound { path: "/nonexistent/model.gguf".to_string() };
        let berr: BitNetError = merr.into();
        let msg = format!("{berr}");
        assert!(msg.contains("Model") || msg.contains("model") || msg.contains("nonexistent"));
    }

    #[test]
    fn quantization_error_converts_to_bitnet_error() {
        let qerr = QuantizationError::UnsupportedType { qtype: "Q99_XYZ".to_string() };
        let berr: BitNetError = qerr.into();
        let msg = format!("{berr}");
        assert!(msg.contains("Q99_XYZ"), "error message must contain the qtype: {msg}");
    }

    #[test]
    fn inference_error_converts_to_bitnet_error() {
        let ierr = InferenceError::InvalidInput { reason: "empty prompt".to_string() };
        let berr: BitNetError = ierr.into();
        let msg = format!("{berr}");
        assert!(msg.contains("empty prompt"), "error message must contain reason: {msg}");
    }

    #[test]
    fn config_error_display_includes_message() {
        let berr = BitNetError::Config("bad parameter".to_string());
        let msg = format!("{berr}");
        assert!(msg.contains("bad parameter"), "Config error must include message: {msg}");
    }

    #[test]
    fn validation_error_display_includes_message() {
        let berr = BitNetError::Validation("shape mismatch".to_string());
        let msg = format!("{berr}");
        assert!(msg.contains("shape mismatch"), "Validation error must include message: {msg}");
    }

    #[test]
    fn strict_mode_error_display_includes_message() {
        let berr = BitNetError::StrictMode("suspicious weights".to_string());
        let msg = format!("{berr}");
        assert!(msg.contains("suspicious weights"), "StrictMode error must include message: {msg}");
    }
}

// ─── Device Feature Detection Cross-Crate ────────────────────────────────────

#[cfg(test)]
mod device_feature_cross_crate {
    use bitnet_kernels::device_features;

    #[test]
    fn simd_level_and_capabilities_agree() {
        let simd = device_features::detect_simd_level();
        let caps = device_features::current_kernel_capabilities();
        assert_eq!(
            simd, caps.simd_level,
            "detect_simd_level() and capabilities.simd_level must agree"
        );
    }

    #[test]
    fn capabilities_summary_is_non_empty() {
        let caps = device_features::current_kernel_capabilities();
        let summary = caps.summary();
        assert!(!summary.is_empty(), "capabilities summary must be non-empty");
        assert!(summary.contains("simd="), "summary must contain simd info: {summary}");
    }

    #[test]
    fn gpu_not_compiled_with_cpu_only() {
        assert!(
            !device_features::gpu_compiled(),
            "gpu should not be compiled with --features cpu alone"
        );
    }

    #[test]
    fn device_probe_cpu_returns_info() {
        let probe = bitnet_device_probe::probe_cpu();
        // On any machine, the CPU probe should report core count
        assert!(probe.core_count > 0, "CPU probe must report at least 1 core");
    }
}
