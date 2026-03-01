//! Comprehensive property-based tests for `bitnet-common` foundational types.
//!
//! Complements `property_tests.rs` with deeper coverage of:
//! - `BitNetConfig` / `ConfigBuilder` validation invariants
//! - Sub-error type message formatting
//! - `BackendStartupSummary` field preservation and log-line format
//! - `BackendRequest` display
//! - `Device` serialization round-trip
//! - `SimdLevel` and `KernelBackend` display
//! - `SecurityLimits` structural invariants
//! - `ConcreteTensor` mock construction
//! - `GenerationConfig` serde round-trip
//! - `QuantizationConfig` power-of-two block-size validation

use bitnet_common::{
    BitNetConfig,
    backend_selection::{BackendRequest, BackendStartupSummary},
    error::{
        BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
        SecurityLimits,
    },
    kernel_registry::{KernelBackend, SimdLevel},
    tensor::{ConcreteTensor, Tensor},
    types::{Device, GenerationConfig, QuantizationType},
};
use proptest::prelude::*;

// ── BitNetConfig validation invariants ───────────────────────────────────────

proptest! {
    /// A config built with all-positive, power-of-two-aligned values must pass validation.
    #[test]
    fn prop_valid_config_passes_validation(
        // num_heads must divide hidden_size; choose hidden_size as a multiple.
        num_heads  in prop_oneof![Just(1usize), Just(2), Just(4), Just(8), Just(16), Just(32)],
        hidden_mul in 1usize..=64,   // hidden_size = num_heads * hidden_mul
        num_layers in 1usize..=64,
        vocab_size in 1usize..=65536,
        temperature in 0.01f32..=4.0,
        top_k in proptest::option::of(1usize..=200),
        top_p in proptest::option::of(0.01f32..=1.0f32),
        rep_penalty in 0.01f32..=3.0,
    ) {
        let hidden_size = num_heads * hidden_mul;
        let config = BitNetConfig::builder()
            .vocab_size(vocab_size)
            .hidden_size(hidden_size)
            .num_layers(num_layers)
            .num_heads(num_heads)
            .temperature(temperature)
            .top_k(top_k)
            .top_p(top_p)
            .build();

        // If top_p is out of (0, 1], validation should reject it.
        #[allow(clippy::collapsible_if)]
        #[allow(clippy::collapsible_if)]
        #[allow(clippy::collapsible_if)]
        #[allow(clippy::collapsible_if)]
        if let Some(p) = top_p {
            if p <= 0.0 || p > 1.0 {
                prop_assert!(config.is_err());
                return Ok(());
            }
        }
        prop_assert!(
            config.is_ok(),
            "expected valid config but got: {:?}", config.err()
        );
        let _ = rep_penalty; // consumed to avoid unused-variable warning
    }

    /// Zero vocab_size must always fail validation.
    #[test]
    fn prop_config_zero_vocab_fails(
        num_heads in prop_oneof![Just(1usize), Just(2), Just(4)],
    ) {
        let result = BitNetConfig::builder()
            .vocab_size(0)
            .num_heads(num_heads)
            .build();
        prop_assert!(result.is_err());
    }

    /// Zero num_layers must always fail validation.
    #[test]
    fn prop_config_zero_num_layers_fails(_dummy in 0u8..1) {
        let result = BitNetConfig::builder()
            .num_layers(0)
            .build();
        prop_assert!(result.is_err());
    }

    /// Temperature <= 0 must always fail validation.
    #[test]
    fn prop_config_non_positive_temperature_fails(
        temp in prop_oneof![Just(0.0f32), Just(-0.001f32), Just(-10.0f32)],
    ) {
        let result = BitNetConfig::builder()
            .temperature(temp)
            .build();
        prop_assert!(result.is_err());
    }

    /// QuantizationConfig block_size must be a power of two to pass validation.
    #[test]
    fn prop_config_non_power_of_two_block_size_fails(
        // Odd numbers greater than 1 are never powers of two.
        k in 1usize..=500,
    ) {
        let block_size = 2 * k + 1; // guaranteed odd and > 1 → not a power of two
        let mut config = BitNetConfig::default();
        config.quantization.block_size = block_size;
        prop_assert!(config.validate().is_err(),
            "expected failure for odd block_size={block_size}");
    }

    /// A BitNetConfig serialises to JSON and deserialises to an equivalent config.
    #[test]
    fn prop_default_config_serde_roundtrip(_dummy in 0u8..1) {
        let config = BitNetConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let back: BitNetConfig = serde_json::from_str(&json).expect("deserialize");
        // Spot-check key structural fields.
        prop_assert_eq!(back.model.vocab_size,       config.model.vocab_size);
        prop_assert_eq!(back.model.num_heads,        config.model.num_heads);
        prop_assert_eq!(back.quantization.block_size, config.quantization.block_size);
        prop_assert!(back.validate().is_ok());
    }
}

// ── Sub-error type message invariants ────────────────────────────────────────

proptest! {
    /// ModelError::NotFound display contains the path.
    #[test]
    fn prop_model_error_not_found_contains_path(
        path in "[a-z/._-]{1,60}",
    ) {
        let err = ModelError::NotFound { path: path.clone() };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&path), "expected path in '{s}'");
    }

    /// ModelError::LoadingFailed display contains the reason.
    #[test]
    fn prop_model_error_loading_failed_contains_reason(
        reason in "[a-zA-Z0-9 _-]{1,64}",
    ) {
        let err = ModelError::LoadingFailed { reason: reason.clone() };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&reason));
    }

    /// QuantizationError::UnsupportedType display contains qtype.
    #[test]
    fn prop_quantization_error_unsupported_type_contains_qtype(
        qtype in "[A-Z0-9_]{1,16}",
    ) {
        let err = QuantizationError::UnsupportedType { qtype: qtype.clone() };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&qtype));
    }

    /// QuantizationError::InvalidBlockSize display mentions the size.
    #[test]
    fn prop_quantization_error_invalid_block_size(size in 0usize..=1024) {
        let err = QuantizationError::InvalidBlockSize { size };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&size.to_string()));
    }

    /// KernelError::ExecutionFailed display contains the reason.
    #[test]
    fn prop_kernel_error_execution_failed_contains_reason(
        reason in "[a-zA-Z0-9 _-]{1,64}",
    ) {
        let err = KernelError::ExecutionFailed { reason: reason.clone() };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&reason));
    }

    /// KernelError::NoProvider always produces a non-empty display string.
    #[test]
    fn prop_kernel_error_no_provider_non_empty(_dummy in 0u8..1) {
        let err = KernelError::NoProvider;
        prop_assert!(!err.to_string().is_empty());
    }

    /// InferenceError::ContextLengthExceeded display mentions the length.
    #[test]
    fn prop_inference_error_context_length_exceeded(length in 0usize..=1_000_000) {
        let err = InferenceError::ContextLengthExceeded { length };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&length.to_string()));
    }

    /// SecurityError::ResourceLimit display contains resource, value, and limit.
    #[test]
    fn prop_security_error_resource_limit(
        resource in "[a-z_]{1,20}",
        value  in 1u64..=u64::MAX / 2,
        limit  in 0u64..=u64::MAX / 2,
    ) {
        let err = SecurityError::ResourceLimit {
            resource: resource.clone(),
            value,
            limit,
        };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&resource));
        prop_assert!(s.contains(&value.to_string()));
        prop_assert!(s.contains(&limit.to_string()));
    }

    /// SecurityError::InputValidation display contains the reason.
    #[test]
    fn prop_security_error_input_validation_contains_reason(
        reason in "[a-zA-Z0-9 _-]{1,64}",
    ) {
        let err = SecurityError::InputValidation { reason: reason.clone() };
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&reason));
    }

    /// BitNetError wrapping a ModelError is always non-empty on display.
    #[test]
    fn prop_bitnet_error_wraps_model_error(
        path in "[a-z/._-]{1,40}",
    ) {
        let inner = ModelError::NotFound { path };
        let err = BitNetError::Model(inner);
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        let dbg = format!("{err:?}");
        prop_assert!(!dbg.is_empty());
    }
}

// ── BackendStartupSummary field preservation and log-line format ──────────────

proptest! {
    /// BackendStartupSummary::new preserves all fields.
    #[test]
    fn prop_backend_startup_summary_fields_preserved(
        requested in "[a-z]{1,8}",
        selected  in "[a-z-]{1,16}",
        n_detected in 0usize..=4,
    ) {
        let detected: Vec<String> = (0..n_detected)
            .map(|i| format!("backend-{i}"))
            .collect();
        let summary = BackendStartupSummary::new(&requested, detected.clone(), &selected);
        prop_assert_eq!(&summary.requested, &requested);
        prop_assert_eq!(&summary.selected,  &selected);
        prop_assert_eq!(summary.detected,   detected);
    }

    /// BackendStartupSummary::log_line contains "requested=", "selected=".
    #[test]
    fn prop_backend_startup_summary_log_line_format(
        requested in "[a-z]{1,8}",
        selected  in "[a-z-]{1,16}",
    ) {
        let summary = BackendStartupSummary::new(&requested, vec![], &selected);
        let line = summary.log_line();
        prop_assert!(line.contains("requested="), "missing 'requested=' in '{line}'");
        prop_assert!(line.contains("selected="),  "missing 'selected=' in '{line}'");
        prop_assert!(line.contains(&requested),   "missing requested value in '{line}'");
        prop_assert!(line.contains(&selected),    "missing selected value in '{line}'");
    }

    /// BackendStartupSummary round-trips through JSON.
    #[test]
    fn prop_backend_startup_summary_serde_roundtrip(
        requested in "[a-z]{1,8}",
        selected  in "[a-z-]{1,16}",
    ) {
        let summary = BackendStartupSummary::new(&requested, vec!["cpu-rust".to_string()], &selected);
        let json = serde_json::to_string(&summary).expect("serialize");
        let back: BackendStartupSummary = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back.requested, summary.requested);
        prop_assert_eq!(back.selected,  summary.selected);
        prop_assert_eq!(back.detected,  summary.detected);
    }
}

// ── BackendRequest display ────────────────────────────────────────────────────

proptest! {
    /// Display output for every BackendRequest variant is non-empty.
    #[test]
    fn prop_backend_request_display_non_empty(
        req in prop_oneof![
            Just(BackendRequest::Auto),
            Just(BackendRequest::Cpu),
            Just(BackendRequest::Gpu),
            Just(BackendRequest::Cuda),
        ]
    ) {
        let s = req.to_string();
        prop_assert!(!s.is_empty());
    }

    /// BackendRequest variants have distinct display strings.
    #[test]
    fn prop_backend_request_display_distinct(_dummy in 0u8..1) {
        use std::collections::HashSet;
        let displays: HashSet<_> = [
            BackendRequest::Auto.to_string(),
            BackendRequest::Cpu.to_string(),
            BackendRequest::Gpu.to_string(),
            BackendRequest::Cuda.to_string(),
        ]
        .into_iter()
        .collect();
        prop_assert_eq!(displays.len(), 4, "some BackendRequest variants share display strings");
    }
}

// ── Device JSON serialisation round-trip ─────────────────────────────────────

proptest! {
    /// Device::Cpu round-trips through JSON.
    #[test]
    fn prop_device_cpu_serde_roundtrip(_dummy in 0u8..1) {
        let d = Device::Cpu;
        let json = serde_json::to_string(&d).expect("serialize");
        let back: Device = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back, d);
    }

    /// Device::Cuda(idx) round-trips through JSON for arbitrary ordinals.
    #[test]
    fn prop_device_cuda_serde_roundtrip(idx in 0usize..=255) {
        let d = Device::Cuda(idx);
        let json = serde_json::to_string(&d).expect("serialize");
        let back: Device = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back, d);
    }

    /// Device::Metal round-trips through JSON.
    #[test]
    fn prop_device_metal_serde_roundtrip(_dummy in 0u8..1) {
        let d = Device::Metal;
        let json = serde_json::to_string(&d).expect("serialize");
        let back: Device = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back, d);
    }
}

// ── SimdLevel and KernelBackend display invariants ────────────────────────────

proptest! {
    /// SimdLevel display is non-empty for every variant.
    #[test]
    fn prop_simd_level_display_non_empty(
        lvl in prop_oneof![
            Just(SimdLevel::Scalar),
            Just(SimdLevel::Neon),
            Just(SimdLevel::Sse42),
            Just(SimdLevel::Avx2),
            Just(SimdLevel::Avx512),
        ]
    ) {
        prop_assert!(!lvl.to_string().is_empty());
    }

    /// All SimdLevel variants have distinct display strings.
    #[test]
    fn prop_simd_level_display_distinct(_dummy in 0u8..1) {
        use std::collections::HashSet;
        let displays: HashSet<_> = [
            SimdLevel::Scalar.to_string(),
            SimdLevel::Neon.to_string(),
            SimdLevel::Sse42.to_string(),
            SimdLevel::Avx2.to_string(),
            SimdLevel::Avx512.to_string(),
        ]
        .into_iter()
        .collect();
        prop_assert_eq!(displays.len(), 5);
    }

    /// KernelBackend display is non-empty for every variant.
    #[test]
    fn prop_kernel_backend_display_non_empty(
        backend in prop_oneof![
            Just(KernelBackend::CpuRust),
            Just(KernelBackend::Cuda),
            Just(KernelBackend::CppFfi),
        ]
    ) {
        prop_assert!(!backend.to_string().is_empty());
    }

    /// All KernelBackend variants have distinct display strings.
    #[test]
    fn prop_kernel_backend_display_distinct(_dummy in 0u8..1) {
        use std::collections::HashSet;
        let displays: HashSet<_> = [
            KernelBackend::CpuRust.to_string(),
            KernelBackend::Cuda.to_string(),
            KernelBackend::CppFfi.to_string(),
        ]
        .into_iter()
        .collect();
        prop_assert_eq!(displays.len(), 3);
    }
}

// ── SecurityLimits structural invariants ─────────────────────────────────────

proptest! {
    /// Default SecurityLimits has all non-zero fields.
    #[test]
    fn prop_security_limits_defaults_non_zero(_dummy in 0u8..1) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_tensor_elements    > 0);
        prop_assert!(limits.max_memory_allocation  > 0);
        prop_assert!(limits.max_metadata_size      > 0);
        prop_assert!(limits.max_string_length      > 0);
        prop_assert!(limits.max_array_length       > 0);
    }

    /// Default SecurityLimits: max_memory_allocation >= max_metadata_size.
    #[test]
    fn prop_security_limits_memory_gte_metadata(_dummy in 0u8..1) {
        let limits = SecurityLimits::default();
        prop_assert!(limits.max_memory_allocation >= limits.max_metadata_size);
    }
}

// ── ConcreteTensor::mock shape invariants ────────────────────────────────────

proptest! {
    /// ConcreteTensor::mock preserves the shape it was constructed with.
    #[test]
    fn prop_concrete_tensor_mock_shape_preserved(
        dims in proptest::collection::vec(1usize..=64, 1..=4),
    ) {
        let t = ConcreteTensor::mock(dims.clone());
        prop_assert_eq!(t.shape(), dims.as_slice());
    }

    /// Element count in ConcreteTensor::mock equals the product of dimensions.
    #[test]
    fn prop_concrete_tensor_mock_element_count(
        dims in proptest::collection::vec(1usize..=16, 1..=3),
    ) {
        let expected: usize = dims.iter().product();
        let t = ConcreteTensor::mock(dims);
        let actual: usize = t.shape().iter().product();
        prop_assert_eq!(actual, expected);
    }
}

// ── GenerationConfig JSON round-trip ─────────────────────────────────────────

proptest! {
    /// GenerationConfig survives a JSON serialize → deserialize round-trip.
    #[test]
    fn prop_generation_config_serde_roundtrip(
        max_new_tokens in 1usize..=4096,
        temperature    in 0.01f32..=4.0,
        do_sample      in any::<bool>(),
        seed           in proptest::option::of(any::<u64>()),
    ) {
        let cfg = GenerationConfig {
            max_new_tokens,
            temperature,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            do_sample,
            seed,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: GenerationConfig = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back.max_new_tokens, max_new_tokens);
        prop_assert!((back.temperature - temperature).abs() < 1e-5);
        prop_assert_eq!(back.do_sample, do_sample);
        prop_assert_eq!(back.seed, seed);
    }
}

// ── QuantizationType display strings are valid ASCII identifiers ──────────────

proptest! {
    /// Display strings of all QuantizationType variants contain only ASCII printable chars.
    #[test]
    fn prop_quantization_type_display_is_ascii(
        qt in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ]
    ) {
        let s = qt.to_string();
        prop_assert!(s.is_ascii(), "non-ASCII in display string '{s}'");
        prop_assert!(s.chars().all(|c| !c.is_control()),
            "control character in display string '{s}'");
    }
}
