//! Integration tests for bitnet-common foundational types.
//!
//! Covers Device, QuantizationType, error types, StrictModeEnforcer,
//! warn_once!, ModelConfig, ArchitectureRegistry, NormType, ActivationType,
//! and serialization roundtrips.
#![allow(clippy::field_reassign_with_default)]

use bitnet_common::arch_registry::ArchitectureRegistry;
use bitnet_common::config::{ActivationType, ModelConfig, NormType};
use bitnet_common::error::*;
use bitnet_common::strict_mode::{
    ComputationType, MissingKernelScenario, MockInferencePath, PerformanceMetrics,
    StrictModeConfig, StrictModeEnforcer,
};
use bitnet_common::types::{Device, GenerationConfig, QuantizationType};
use bitnet_common::warn_once;

// ── Device enum ─────────────────────────────────────────────────────────

mod device_variants {
    use super::*;

    #[test]
    fn cpu_is_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn cpu_predicate() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_cuda());
        assert!(!Device::Cpu.is_opencl());
        assert!(!Device::Cpu.is_hip());
        assert!(!Device::Cpu.is_npu());
    }

    #[test]
    fn cuda_predicate() {
        let dev = Device::Cuda(0);
        assert!(dev.is_cuda());
        assert!(!dev.is_cpu());
        assert!(!dev.is_opencl());
    }

    #[test]
    fn cuda_new() {
        let dev = Device::new_cuda(0).unwrap();
        assert_eq!(dev, Device::Cuda(0));
        assert!(dev.is_cuda());
    }

    #[test]
    fn opencl_predicate() {
        let dev = Device::OpenCL(1);
        assert!(dev.is_opencl());
        assert!(!dev.is_cpu());
        assert!(!dev.is_cuda());
    }

    #[test]
    fn opencl_new() {
        let dev = Device::new_opencl(2).unwrap();
        assert_eq!(dev, Device::OpenCL(2));
    }

    #[test]
    fn hip_predicate() {
        let dev = Device::Hip(0);
        assert!(dev.is_hip());
        assert!(!dev.is_cpu());
        assert!(!dev.is_cuda());
    }

    #[test]
    fn npu_predicate() {
        assert!(Device::Npu.is_npu());
        assert!(!Device::Npu.is_cpu());
    }

    #[test]
    fn metal_predicates() {
        let dev = Device::Metal;
        assert!(!dev.is_cpu());
        assert!(!dev.is_cuda());
        assert!(!dev.is_opencl());
        assert!(!dev.is_hip());
        assert!(!dev.is_npu());
    }

    #[test]
    fn device_ordering() {
        // Device derives Ord — Cpu < Cuda < Hip < Npu < Metal < OpenCL
        assert!(Device::Cpu < Device::Cuda(0));
        assert!(Device::Cuda(0) < Device::Cuda(1));
    }

    #[test]
    fn device_equality() {
        assert_eq!(Device::Cuda(0), Device::Cuda(0));
        assert_ne!(Device::Cuda(0), Device::Cuda(1));
        assert_ne!(Device::Cpu, Device::Cuda(0));
    }

    #[test]
    fn device_clone() {
        let dev = Device::Cuda(3);
        let cloned = dev;
        assert_eq!(dev, cloned);
    }

    #[test]
    fn device_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Device::Cpu);
        set.insert(Device::Cuda(0));
        set.insert(Device::Cuda(0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn device_debug_format() {
        let debug = format!("{:?}", Device::Cuda(0));
        assert!(debug.contains("Cuda"));
        assert!(debug.contains("0"));
    }

    #[test]
    fn device_to_candle_cpu() {
        let dev = Device::Cpu;
        let candle_dev = dev.to_candle().unwrap();
        assert!(matches!(candle_dev, candle_core::Device::Cpu));
    }

    #[test]
    fn device_opencl_to_candle_fallback() {
        let dev = Device::OpenCL(0);
        let candle_dev = dev.to_candle().unwrap();
        assert!(matches!(candle_dev, candle_core::Device::Cpu));
    }
}

// ── Device serialization ────────────────────────────────────────────────

mod device_serialization {
    use super::*;

    #[test]
    fn cpu_roundtrip() {
        let dev = Device::Cpu;
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }

    #[test]
    fn cuda_roundtrip() {
        let dev = Device::Cuda(7);
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }

    #[test]
    fn opencl_roundtrip() {
        let dev = Device::OpenCL(3);
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }

    #[test]
    fn hip_roundtrip() {
        let dev = Device::Hip(0);
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }

    #[test]
    fn metal_roundtrip() {
        let dev = Device::Metal;
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }

    #[test]
    fn npu_roundtrip() {
        let dev = Device::Npu;
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }
}

// ── QuantizationType ────────────────────────────────────────────────────

mod quantization_type {
    use super::*;

    #[test]
    fn all_variants_exist() {
        let _i2s = QuantizationType::I2S;
        let _tl1 = QuantizationType::TL1;
        let _tl2 = QuantizationType::TL2;
    }

    #[test]
    fn display_i2s() {
        assert_eq!(format!("{}", QuantizationType::I2S), "I2_S");
    }

    #[test]
    fn display_tl1() {
        assert_eq!(format!("{}", QuantizationType::TL1), "TL1");
    }

    #[test]
    fn display_tl2() {
        assert_eq!(format!("{}", QuantizationType::TL2), "TL2");
    }

    #[test]
    fn equality_and_copy() {
        let a = QuantizationType::I2S;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(QuantizationType::I2S, QuantizationType::TL1);
    }

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QuantizationType::I2S);
        set.insert(QuantizationType::TL1);
        set.insert(QuantizationType::TL2);
        set.insert(QuantizationType::I2S); // duplicate
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn serialization_roundtrip() {
        for qt in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let json = serde_json::to_string(&qt).unwrap();
            let back: QuantizationType = serde_json::from_str(&json).unwrap();
            assert_eq!(qt, back, "roundtrip failed for {:?}", qt);
        }
    }

    #[test]
    fn debug_format() {
        let dbg = format!("{:?}", QuantizationType::I2S);
        assert_eq!(dbg, "I2S");
    }
}

// ── Error types ─────────────────────────────────────────────────────────

mod error_types {
    use super::*;

    #[test]
    fn model_error_not_found_display() {
        let err = ModelError::NotFound { path: "/tmp/missing.gguf".into() };
        let msg = format!("{}", err);
        assert!(msg.contains("not found"), "got: {}", msg);
        assert!(msg.contains("/tmp/missing.gguf"));
    }

    #[test]
    fn model_error_invalid_format() {
        let err = ModelError::InvalidFormat { format: "xyz".into() };
        assert!(format!("{}", err).contains("xyz"));
    }

    #[test]
    fn model_error_loading_failed() {
        let err = ModelError::LoadingFailed { reason: "corrupt header".into() };
        assert!(format!("{}", err).contains("corrupt header"));
    }

    #[test]
    fn model_error_unsupported_version() {
        let err = ModelError::UnsupportedVersion { version: "99".into() };
        assert!(format!("{}", err).contains("99"));
    }

    #[test]
    fn quantization_error_unsupported_type() {
        let err = QuantizationError::UnsupportedType { qtype: "Q4_0".into() };
        assert!(format!("{}", err).contains("Q4_0"));
    }

    #[test]
    fn quantization_error_invalid_block_size() {
        let err = QuantizationError::InvalidBlockSize { size: 7 };
        assert!(format!("{}", err).contains("7"));
    }

    #[test]
    fn kernel_error_no_provider() {
        let err = KernelError::NoProvider;
        assert!(format!("{}", err).contains("No available kernel provider"));
    }

    #[test]
    fn kernel_error_execution_failed() {
        let err = KernelError::ExecutionFailed { reason: "timeout".into() };
        assert!(format!("{}", err).contains("timeout"));
    }

    #[test]
    fn kernel_error_unsupported_hardware() {
        let err = KernelError::UnsupportedHardware {
            required: "AVX2".into(),
            available: "SSE4.2".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("AVX2"));
        assert!(msg.contains("SSE4.2"));
    }

    #[test]
    fn inference_error_generation_failed() {
        let err = InferenceError::GenerationFailed { reason: "OOM".into() };
        assert!(format!("{}", err).contains("OOM"));
    }

    #[test]
    fn inference_error_context_exceeded() {
        let err = InferenceError::ContextLengthExceeded { length: 4096 };
        assert!(format!("{}", err).contains("4096"));
    }

    #[test]
    fn security_error_resource_limit() {
        let err = SecurityError::ResourceLimit {
            resource: "tensor_elements".into(),
            value: 2_000_000_000,
            limit: 1_000_000_000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("tensor_elements"));
        assert!(msg.contains("2000000000"));
    }

    #[test]
    fn security_error_memory_bomb() {
        let err = SecurityError::MemoryBomb { reason: "excessive allocation".into() };
        assert!(format!("{}", err).contains("excessive allocation"));
    }

    #[test]
    fn bitnet_error_from_model_error() {
        let model_err = ModelError::NotFound { path: "missing.gguf".into() };
        let bitnet_err: BitNetError = model_err.into();
        assert!(format!("{}", bitnet_err).contains("missing.gguf"));
    }

    #[test]
    fn bitnet_error_from_quantization_error() {
        let q_err = QuantizationError::QuantizationFailed { reason: "bad data".into() };
        let bitnet_err: BitNetError = q_err.into();
        assert!(format!("{}", bitnet_err).contains("bad data"));
    }

    #[test]
    fn bitnet_error_from_kernel_error() {
        let k_err = KernelError::NoProvider;
        let bitnet_err: BitNetError = k_err.into();
        assert!(format!("{}", bitnet_err).contains("kernel"));
    }

    #[test]
    fn bitnet_error_from_inference_error() {
        let i_err = InferenceError::InvalidInput { reason: "empty prompt".into() };
        let bitnet_err: BitNetError = i_err.into();
        assert!(format!("{}", bitnet_err).contains("empty prompt"));
    }

    #[test]
    fn bitnet_error_config_variant() {
        let err = BitNetError::Config("bad config".into());
        assert!(format!("{}", err).contains("bad config"));
    }

    #[test]
    fn bitnet_error_validation_variant() {
        let err = BitNetError::Validation("shape mismatch".into());
        assert!(format!("{}", err).contains("shape mismatch"));
    }

    #[test]
    fn bitnet_error_strict_mode_variant() {
        let err = BitNetError::StrictMode("mock detected".into());
        assert!(format!("{}", err).contains("mock detected"));
    }

    #[test]
    fn security_limits_defaults() {
        let limits = SecurityLimits::default();
        assert_eq!(limits.max_tensor_elements, 1_000_000_000);
        assert_eq!(limits.max_memory_allocation, 4 * 1024 * 1024 * 1024);
        assert_eq!(limits.max_metadata_size, 100 * 1024 * 1024);
        assert_eq!(limits.max_string_length, 1024 * 1024);
        assert_eq!(limits.max_array_length, 1_000_000);
    }

    #[test]
    fn validation_error_details_construction() {
        let details = ValidationErrorDetails {
            errors: vec!["bad shape".into()],
            warnings: vec!["deprecated field".into()],
            recommendations: vec!["use v2 format".into()],
        };
        assert_eq!(details.errors.len(), 1);
        assert_eq!(details.warnings.len(), 1);
        assert_eq!(details.recommendations.len(), 1);
    }
}

// ── StrictModeEnforcer ──────────────────────────────────────────────────

mod strict_mode {
    use super::*;

    fn make_enforcer(enabled: bool) -> StrictModeEnforcer {
        StrictModeEnforcer::with_config(Some(StrictModeConfig {
            enabled,
            fail_on_mock: enabled,
            require_quantization: enabled,
            enforce_quantized_inference: enabled,
            validate_performance: enabled,
            ci_enhanced_mode: false,
            log_all_validations: false,
            fail_fast_on_any_mock: false,
        }))
    }

    #[test]
    fn enforcer_disabled_by_default() {
        let enforcer = make_enforcer(false);
        assert!(!enforcer.is_enabled());
    }

    #[test]
    fn enforcer_enabled_via_config() {
        let enforcer = make_enforcer(true);
        assert!(enforcer.is_enabled());
        assert!(enforcer.get_config().fail_on_mock);
        assert!(enforcer.get_config().require_quantization);
    }

    #[test]
    fn disabled_enforcer_allows_mock_path() {
        let enforcer = make_enforcer(false);
        let path = MockInferencePath {
            description: "test mock".into(),
            uses_mock_computation: true,
            fallback_reason: "testing".into(),
        };
        assert!(enforcer.validate_inference_path(&path).is_ok());
    }

    #[test]
    fn enabled_enforcer_rejects_mock_path() {
        let enforcer = make_enforcer(true);
        let path = MockInferencePath {
            description: "test mock".into(),
            uses_mock_computation: true,
            fallback_reason: "testing".into(),
        };
        assert!(enforcer.validate_inference_path(&path).is_err());
    }

    #[test]
    fn enabled_enforcer_allows_real_path() {
        let enforcer = make_enforcer(true);
        let path = MockInferencePath {
            description: "real path".into(),
            uses_mock_computation: false,
            fallback_reason: String::new(),
        };
        assert!(enforcer.validate_inference_path(&path).is_ok());
    }

    #[test]
    fn enabled_enforcer_rejects_mock_performance() {
        let enforcer = make_enforcer(true);
        let metrics = PerformanceMetrics {
            tokens_per_second: 10.0,
            computation_type: ComputationType::Mock,
            ..Default::default()
        };
        assert!(enforcer.validate_performance_metrics(&metrics).is_err());
    }

    #[test]
    fn enabled_enforcer_allows_real_performance() {
        let enforcer = make_enforcer(true);
        let metrics = PerformanceMetrics {
            tokens_per_second: 10.0,
            computation_type: ComputationType::Real,
            ..Default::default()
        };
        assert!(enforcer.validate_performance_metrics(&metrics).is_ok());
    }

    #[test]
    fn enabled_enforcer_rejects_suspicious_performance() {
        let enforcer = make_enforcer(true);
        let metrics = PerformanceMetrics {
            tokens_per_second: 200.0, // suspiciously high
            computation_type: ComputationType::Real,
            ..Default::default()
        };
        assert!(enforcer.validate_performance_metrics(&metrics).is_err());
    }

    #[test]
    fn disabled_enforcer_allows_suspicious_performance() {
        let enforcer = make_enforcer(false);
        let metrics = PerformanceMetrics {
            tokens_per_second: 200.0,
            computation_type: ComputationType::Mock,
            ..Default::default()
        };
        assert!(enforcer.validate_performance_metrics(&metrics).is_ok());
    }

    #[test]
    fn enabled_enforcer_rejects_kernel_fallback() {
        let enforcer = make_enforcer(true);
        let scenario = MissingKernelScenario {
            quantization_type: QuantizationType::I2S,
            device: Device::Cpu,
            fallback_available: true,
        };
        assert!(enforcer.validate_kernel_availability(&scenario).is_err());
    }

    #[test]
    fn enabled_enforcer_rejects_quantization_fallback() {
        let enforcer = make_enforcer(true);
        let result = enforcer.validate_quantization_fallback(
            QuantizationType::I2S,
            Device::Cpu,
            &[4096, 4096],
            "no kernel",
        );
        assert!(result.is_err());
    }

    #[test]
    fn disabled_enforcer_allows_quantization_fallback() {
        let enforcer = make_enforcer(false);
        let result = enforcer.validate_quantization_fallback(
            QuantizationType::I2S,
            Device::Cpu,
            &[4096, 4096],
            "no kernel",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn computation_type_default_is_real() {
        assert_eq!(ComputationType::default(), ComputationType::Real);
    }

    #[test]
    fn performance_metrics_default() {
        let m = PerformanceMetrics::default();
        assert_eq!(m.tokens_per_second, 0.0);
        assert_eq!(m.latency_ms, 0.0);
        assert_eq!(m.memory_usage_mb, 0.0);
        assert_eq!(m.computation_type, ComputationType::Real);
        assert!(m.gpu_utilization.is_none());
    }

    #[test]
    fn computation_type_serialization_roundtrip() {
        for ct in [ComputationType::Real, ComputationType::Mock] {
            let json = serde_json::to_string(&ct).unwrap();
            let back: ComputationType = serde_json::from_str(&json).unwrap();
            assert_eq!(ct, back);
        }
    }
}

// ── warn_once! macro ────────────────────────────────────────────────────

mod warn_once_macro {
    use super::*;

    #[test]
    fn warn_once_does_not_panic() {
        warn_once!("test_key_1", "This should not panic");
    }

    #[test]
    fn warn_once_repeated_calls_safe() {
        for _ in 0..100 {
            warn_once!("test_key_repeated", "repeated call");
        }
    }
}

// ── ModelConfig ─────────────────────────────────────────────────────────

mod model_config {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 0);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.max_position_embeddings, 2048);
        assert!(cfg.path.is_none());
        assert_eq!(cfg.norm_type, NormType::LayerNorm);
        assert_eq!(cfg.activation_type, ActivationType::Silu);
    }

    #[test]
    fn apply_architecture_defaults_llama() {
        let mut cfg = ModelConfig::default();
        cfg.apply_architecture_defaults("llama");
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        assert_eq!(cfg.activation_type, ActivationType::Silu);
        // llama has no default context, so max_position_embeddings stays at 2048
        assert_eq!(cfg.max_position_embeddings, 2048);
    }

    #[test]
    fn apply_architecture_defaults_phi() {
        let mut cfg = ModelConfig::default();
        cfg.apply_architecture_defaults("phi");
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        assert_eq!(cfg.activation_type, ActivationType::Silu);
        assert_eq!(cfg.max_position_embeddings, 16384);
    }

    #[test]
    fn apply_architecture_defaults_bitnet() {
        let mut cfg = ModelConfig::default();
        cfg.apply_architecture_defaults("bitnet");
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        assert_eq!(cfg.activation_type, ActivationType::Relu2);
    }

    #[test]
    fn apply_architecture_defaults_gpt() {
        let mut cfg = ModelConfig::default();
        cfg.apply_architecture_defaults("gpt");
        assert_eq!(cfg.norm_type, NormType::LayerNorm);
        assert_eq!(cfg.activation_type, ActivationType::Gelu);
    }

    #[test]
    fn apply_architecture_defaults_unknown_noop() {
        let mut cfg = ModelConfig::default();
        let original_norm = cfg.norm_type;
        let original_act = cfg.activation_type;
        cfg.apply_architecture_defaults("totally_unknown_arch");
        assert_eq!(cfg.norm_type, original_norm);
        assert_eq!(cfg.activation_type, original_act);
    }

    #[test]
    fn apply_architecture_defaults_case_insensitive() {
        let mut cfg = ModelConfig::default();
        cfg.apply_architecture_defaults("PHI");
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
    }

    #[test]
    fn apply_architecture_defaults_preserves_custom_context() {
        let mut cfg = ModelConfig::default();
        cfg.max_position_embeddings = 8192; // custom value != default 2048
        cfg.apply_architecture_defaults("phi"); // phi has ctx 16384
        // Should NOT override because max_position_embeddings != 2048
        assert_eq!(cfg.max_position_embeddings, 8192);
    }

    #[test]
    fn serialization_roundtrip() {
        let cfg = ModelConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.vocab_size, cfg.vocab_size);
        assert_eq!(back.hidden_size, cfg.hidden_size);
        assert_eq!(back.norm_type, cfg.norm_type);
    }
}

// ── ArchitectureRegistry ────────────────────────────────────────────────

mod architecture_registry {
    use super::*;

    #[test]
    fn known_architectures_non_empty() {
        let archs = ArchitectureRegistry::known_architectures();
        assert!(!archs.is_empty());
        assert!(archs.len() >= 40);
    }

    #[test]
    fn lookup_known_returns_some() {
        assert!(ArchitectureRegistry::lookup("llama").is_some());
        assert!(ArchitectureRegistry::lookup("phi").is_some());
        assert!(ArchitectureRegistry::lookup("bitnet").is_some());
        assert!(ArchitectureRegistry::lookup("gemma").is_some());
        assert!(ArchitectureRegistry::lookup("gpt").is_some());
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(ArchitectureRegistry::lookup("unknown_model_xyz").is_none());
        assert!(ArchitectureRegistry::lookup("").is_none());
    }

    #[test]
    fn is_known_agrees_with_lookup() {
        for arch in ArchitectureRegistry::known_architectures() {
            assert_eq!(
                ArchitectureRegistry::is_known(arch),
                ArchitectureRegistry::lookup(arch).is_some(),
                "disagreement for '{}'",
                arch
            );
        }
        assert!(!ArchitectureRegistry::is_known("nonexistent"));
    }

    #[test]
    fn case_insensitive() {
        assert!(ArchitectureRegistry::lookup("LLAMA").is_some());
        assert!(ArchitectureRegistry::lookup("Phi").is_some());
        assert!(ArchitectureRegistry::lookup("BitNet-B1.58").is_some());
    }

    #[test]
    fn all_known_have_valid_context_lengths() {
        for arch in ArchitectureRegistry::known_architectures() {
            let defaults = ArchitectureRegistry::lookup(arch).unwrap();
            if let Some(ctx) = defaults.default_context_length {
                assert!(ctx > 0, "context length for '{}' must be > 0", arch);
            }
        }
    }
}

// ── NormType enum ───────────────────────────────────────────────────────

mod norm_type {
    use super::*;

    #[test]
    fn default_is_layer_norm() {
        assert_eq!(NormType::default(), NormType::LayerNorm);
    }

    #[test]
    fn variants_exist() {
        let _ln = NormType::LayerNorm;
        let _rms = NormType::RmsNorm;
    }

    #[test]
    fn equality() {
        assert_eq!(NormType::LayerNorm, NormType::LayerNorm);
        assert_ne!(NormType::LayerNorm, NormType::RmsNorm);
    }

    #[test]
    fn serialization_roundtrip() {
        for nt in [NormType::LayerNorm, NormType::RmsNorm] {
            let json = serde_json::to_string(&nt).unwrap();
            let back: NormType = serde_json::from_str(&json).unwrap();
            assert_eq!(nt, back);
        }
    }

    #[test]
    fn debug_format() {
        assert!(format!("{:?}", NormType::LayerNorm).contains("LayerNorm"));
        assert!(format!("{:?}", NormType::RmsNorm).contains("RmsNorm"));
    }
}

// ── ActivationType enum ─────────────────────────────────────────────────

mod activation_type {
    use super::*;

    #[test]
    fn default_is_silu() {
        assert_eq!(ActivationType::default(), ActivationType::Silu);
    }

    #[test]
    fn variants_exist() {
        let _silu = ActivationType::Silu;
        let _relu2 = ActivationType::Relu2;
        let _gelu = ActivationType::Gelu;
    }

    #[test]
    fn equality() {
        assert_eq!(ActivationType::Silu, ActivationType::Silu);
        assert_ne!(ActivationType::Silu, ActivationType::Gelu);
        assert_ne!(ActivationType::Gelu, ActivationType::Relu2);
    }

    #[test]
    fn serialization_roundtrip() {
        for at in [ActivationType::Silu, ActivationType::Relu2, ActivationType::Gelu] {
            let json = serde_json::to_string(&at).unwrap();
            let back: ActivationType = serde_json::from_str(&json).unwrap();
            assert_eq!(at, back);
        }
    }
}

// ── GenerationConfig ────────────────────────────────────────────────────

mod generation_config {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 512);
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_k, Some(50));
        assert_eq!(cfg.top_p, Some(0.9));
        assert_eq!(cfg.repetition_penalty, 1.1);
        assert!(cfg.do_sample);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn serialization_roundtrip() {
        let cfg = GenerationConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GenerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_new_tokens, cfg.max_new_tokens);
        assert_eq!(back.temperature, cfg.temperature);
        assert_eq!(back.top_k, cfg.top_k);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────────

mod edge_cases {
    use super::*;

    #[test]
    fn empty_arch_string_returns_none() {
        assert!(ArchitectureRegistry::lookup("").is_none());
        assert!(!ArchitectureRegistry::is_known(""));
    }

    #[test]
    fn whitespace_arch_returns_none() {
        assert!(ArchitectureRegistry::lookup(" ").is_none());
        assert!(ArchitectureRegistry::lookup("\t").is_none());
    }

    #[test]
    fn very_long_arch_string_returns_none() {
        let long = "x".repeat(10_000);
        assert!(ArchitectureRegistry::lookup(&long).is_none());
    }

    #[test]
    fn model_config_apply_empty_arch_noop() {
        let mut cfg = ModelConfig::default();
        let orig = cfg.norm_type;
        cfg.apply_architecture_defaults("");
        assert_eq!(cfg.norm_type, orig);
    }

    #[test]
    fn strict_mode_config_explicit_fields() {
        let config = StrictModeConfig {
            enabled: true,
            fail_on_mock: false,
            require_quantization: true,
            enforce_quantized_inference: false,
            validate_performance: true,
            ci_enhanced_mode: true,
            log_all_validations: true,
            fail_fast_on_any_mock: true,
        };
        assert!(config.enabled);
        assert!(!config.fail_on_mock);
        assert!(config.require_quantization);
        assert!(!config.enforce_quantized_inference);
        assert!(config.ci_enhanced_mode);
    }

    #[test]
    fn device_cuda_max_index() {
        let dev = Device::Cuda(usize::MAX);
        assert!(dev.is_cuda());
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }
}
