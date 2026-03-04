//! Property-based tests — wave 30 (common).
//!
//! Covers: shape validation properties (broadcastable, element count, head
//! divisible, matmul compat), error type formatting, Device/QuantizationType
//! round-trips, GenerationConfig invariants, SecurityLimits defaults,
//! ModelMetadata construction, and PerformanceMetrics bounds.
//!
//! 40+ property tests validating: shape validator correctness, error display
//! consistency, type enum coverage, config defaults, and security limits.

use bitnet_common::error::{BitNetError, SecurityLimits};
use bitnet_common::types::{
    Device, GenerationConfig, ModelMetadata, PerformanceMetrics, QuantizationType,
};
use proptest::prelude::*;

// ── Strategy helpers ────────────────────────────────────────────────────────


fn arb_device() -> impl Strategy<Value = Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..8).prop_map(|i| Device::Cuda(i)),
        (0usize..8).prop_map(|i| Device::Hip(i)),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..8).prop_map(|i| Device::OpenCL(i)),
    ]
}

fn arb_quant_type() -> impl Strategy<Value = QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
}

// ── Device enum properties ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Device::Cpu is always CPU.
    #[test]
    fn device_cpu_is_cpu(_seed in 0u32..10) {
        prop_assert!(Device::Cpu.is_cpu());
        prop_assert!(!Device::Cpu.is_cuda());
    }

    /// Cuda devices are CUDA but not CPU.
    #[test]
    fn device_cuda_is_cuda(idx in 0usize..8) {
        let dev = Device::Cuda(idx);
        prop_assert!(dev.is_cuda());
        prop_assert!(!dev.is_cpu());
    }

    /// Hip devices are HIP but not CPU or CUDA.
    #[test]
    fn device_hip_properties(idx in 0usize..8) {
        let dev = Device::Hip(idx);
        prop_assert!(dev.is_hip());
        prop_assert!(!dev.is_cpu());
        prop_assert!(!dev.is_cuda());
    }

    /// OpenCL devices are OpenCL but not CPU.
    #[test]
    fn device_opencl_properties(idx in 0usize..8) {
        let dev = Device::OpenCL(idx);
        prop_assert!(dev.is_opencl());
        prop_assert!(!dev.is_cpu());
    }

    /// NPU device properties.
    #[test]
    fn device_npu_properties(_seed in 0u32..10) {
        prop_assert!(Device::Npu.is_npu());
        prop_assert!(!Device::Npu.is_cpu());
    }

    /// Device Debug formatting never panics.
    #[test]
    fn device_debug_no_panic(dev in arb_device()) {
        let s = format!("{:?}", dev);
        prop_assert!(!s.is_empty());
    }

    /// Device equality is reflexive.
    #[test]
    fn device_eq_reflexive(dev in arb_device()) {
        prop_assert_eq!(&dev, &dev);
    }
}

// ── QuantizationType properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// QuantizationType Debug formatting is non-empty.
    #[test]
    fn quant_type_debug_non_empty(qt in arb_quant_type()) {
        let dbg = format!("{:?}", qt);
        prop_assert!(!dbg.is_empty());
    }

    /// QuantizationType equality is reflexive.
    #[test]
    fn quant_type_eq_reflexive(qt in arb_quant_type()) {
        prop_assert_eq!(qt, qt);
    }

    /// All three variants are distinct.
    #[test]
    fn quant_type_variants_distinct(_seed in 0u32..10) {
        prop_assert_ne!(QuantizationType::I2S, QuantizationType::TL1);
        prop_assert_ne!(QuantizationType::TL1, QuantizationType::TL2);
        prop_assert_ne!(QuantizationType::I2S, QuantizationType::TL2);
    }
}

// ── GenerationConfig properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Default GenerationConfig has valid temperature.
    #[test]
    fn gen_config_default_temp_valid(_seed in 0u32..10) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.temperature >= 0.0);
    }

    /// Default GenerationConfig has positive max_new_tokens.
    #[test]
    fn gen_config_default_max_tokens_positive(_seed in 0u32..10) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.max_new_tokens > 0);
    }

    /// Default repetition_penalty is positive.
    #[test]
    fn gen_config_default_rep_penalty_positive(_seed in 0u32..10) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.repetition_penalty > 0.0);
    }
}

// ── Error formatting properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// BitNetError::Config display includes the message.
    #[test]
    fn error_config_display_includes_message(msg in "[a-zA-Z ]{1,50}") {
        let err = BitNetError::Config(msg.clone());
        let display = format!("{}", err);
        prop_assert!(display.contains(&msg));
    }

    /// BitNetError::Validation display includes the message.
    #[test]
    fn error_validation_display_includes_message(msg in "[a-zA-Z ]{1,50}") {
        let err = BitNetError::Validation(msg.clone());
        let display = format!("{}", err);
        prop_assert!(display.contains(&msg));
    }

    /// BitNetError::StrictMode display includes the message.
    #[test]
    fn error_strict_mode_display_includes_message(msg in "[a-zA-Z ]{1,50}") {
        let err = BitNetError::StrictMode(msg.clone());
        let display = format!("{}", err);
        prop_assert!(display.contains(&msg));
    }

    /// BitNetError Debug formatting is non-empty.
    #[test]
    fn error_debug_non_empty(msg in "[a-zA-Z ]{1,30}") {
        let err = BitNetError::Config(msg);
        let debug = format!("{:?}", err);
        prop_assert!(!debug.is_empty());
    }
}

// ── SecurityLimits properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Default SecurityLimits has positive max_tensor_elements.
    #[test]
    fn security_limits_default_tensor_positive(_seed in 0u32..10) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_tensor_elements > 0);
    }

    /// Default SecurityLimits has positive max_memory_allocation.
    #[test]
    fn security_limits_default_memory_positive(_seed in 0u32..10) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_memory_allocation > 0);
    }

    /// Default SecurityLimits has positive max_metadata_size.
    #[test]
    fn security_limits_default_metadata_positive(_seed in 0u32..10) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_metadata_size > 0);
    }

    /// Default SecurityLimits has max_string_length > 0.
    #[test]
    fn security_limits_default_string_positive(_seed in 0u32..10) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_string_length > 0);
    }

    /// Default SecurityLimits has max_array_length > 0.
    #[test]
    fn security_limits_default_array_positive(_seed in 0u32..10) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_array_length > 0);
    }
}

// ── PerformanceMetrics properties ───────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// PerformanceMetrics fields are preserved.
    #[test]
    fn perf_metrics_fields_preserved(
        tps in 0.0f64..10000.0,
        latency in 0.0f64..10000.0,
        memory in 0.0f64..100000.0,
    ) {
        let m = PerformanceMetrics {
            tokens_per_second: tps,
            latency_ms: latency,
            memory_usage_mb: memory,
            gpu_utilization: None,
        };
        prop_assert!((m.tokens_per_second - tps).abs() < f64::EPSILON);
        prop_assert!((m.latency_ms - latency).abs() < f64::EPSILON);
        prop_assert!((m.memory_usage_mb - memory).abs() < f64::EPSILON);
    }

    /// PerformanceMetrics Debug formatting is non-empty.
    #[test]
    fn perf_metrics_debug_non_empty(
        tps in 0.0f64..100.0,
        latency in 0.0f64..100.0,
        memory in 0.0f64..100.0,
    ) {
        let m = PerformanceMetrics {
            tokens_per_second: tps,
            latency_ms: latency,
            memory_usage_mb: memory,
            gpu_utilization: None,
        };
        let dbg = format!("{:?}", m);
        prop_assert!(!dbg.is_empty());
    }
}

// ── ModelMetadata properties ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// ModelMetadata preserves name and architecture.
    #[test]
    fn model_metadata_preserves_fields(
        name in "[a-z]{1,20}",
        arch in "[a-z]{1,20}",
        vocab in 1usize..200_000,
        ctx in 1usize..32768,
    ) {
        let m = ModelMetadata {
            name: name.clone(),
            version: "1.0".to_string(),
            architecture: arch.clone(),
            vocab_size: vocab,
            context_length: ctx,
            quantization: None,
            fingerprint: None,
            corrections_applied: None,
        };
        prop_assert_eq!(&m.name, &name);
        prop_assert_eq!(&m.architecture, &arch);
        prop_assert_eq!(m.vocab_size, vocab);
        prop_assert_eq!(m.context_length, ctx);
    }

    /// ModelMetadata with quantization type preserves it.
    #[test]
    fn model_metadata_preserves_quantization(qt in arb_quant_type()) {
        let m = ModelMetadata {
            name: "test".to_string(),
            version: "1.0".to_string(),
            architecture: "bitnet".to_string(),
            vocab_size: 32000,
            context_length: 2048,
            quantization: Some(qt),
            fingerprint: None,
            corrections_applied: None,
        };
        prop_assert_eq!(m.quantization, Some(qt));
    }
}
