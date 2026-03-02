//! Property-based tests for bitnet-common foundational types.
//!
//! Verifies invariants across all possible inputs — catching edge cases
//! that unit tests may miss.

use bitnet_common::{
    backend_selection::{BackendRequest, select_backend},
    error::BitNetError,
    kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel},
    tensor::{MockTensor, Tensor},
    types::{Device, GenerationConfig, ModelMetadata, PerformanceMetrics, QuantizationType},
    warn_once_fn,
};
use proptest::prelude::*;

// ── QuantizationType ──────────────────────────────────────────────────────────

proptest! {
    /// Display output is non-empty and stable for every variant.
    #[test]
    fn prop_quantization_type_display_non_empty(
        qt in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ]
    ) {
        let s = qt.to_string();
        prop_assert!(!s.is_empty());
        prop_assert_eq!(qt.to_string(), s);
    }

    /// Equality is reflexive for all variants.
    #[test]
    fn prop_quantization_type_eq_reflexive(
        qt in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ]
    ) {
        prop_assert_eq!(qt, qt);
    }

    /// Copy semantics: a copied value compares equal to the original.
    #[test]
    fn prop_quantization_type_copy(
        qt in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ]
    ) {
        let copy = qt;
        prop_assert_eq!(qt, copy);
    }
}

// ── Device ────────────────────────────────────────────────────────────────────

proptest! {
    /// CPU device is always identified as CPU, never CUDA.
    #[test]
    fn prop_device_cpu_predicates(_dummy in 0u8..1) {
        let d = Device::Cpu;
        prop_assert!(d.is_cpu());
        prop_assert!(!d.is_cuda());
    }

    /// CUDA device is always identified as CUDA, never CPU.
    #[test]
    fn prop_device_cuda_predicates(idx in 0usize..8) {
        let d = Device::Cuda(idx);
        prop_assert!(!d.is_cpu());
        prop_assert!(d.is_cuda());
    }

    /// Device equality is reflexive.
    #[test]
    fn prop_device_eq_reflexive(idx in 0usize..8) {
        let d = Device::Cuda(idx);
        prop_assert_eq!(d, d);
    }

    /// Device ordering: Cpu < Cuda(_) — CPU always sorts before CUDA.
    #[test]
    fn prop_device_ord_cpu_lt_cuda(idx in 0usize..8) {
        prop_assert!(Device::Cpu < Device::Cuda(idx));
    }

    /// new_cuda returns a Cuda device with the expected index.
    #[test]
    fn prop_device_new_cuda_roundtrip(idx in 0usize..8) {
        let d = Device::new_cuda(idx).expect("new_cuda should succeed");
        prop_assert_eq!(d, Device::Cuda(idx));
        prop_assert!(d.is_cuda());
    }
}

// ── GenerationConfig ──────────────────────────────────────────────────────────

proptest! {
    /// Default config has non-zero max_new_tokens and valid numeric fields.
    #[test]
    fn prop_generation_config_default_valid(_dummy in 0u8..1) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.max_new_tokens > 0);
        prop_assert!(cfg.temperature > 0.0);
        prop_assert!(cfg.repetition_penalty >= 1.0);
    }

    /// Constructed config preserves all fields without mutation.
    #[test]
    fn prop_generation_config_fields_preserved(
        max_tokens in 1usize..4096,
        temp in 0.01f32..2.0,
        top_k in proptest::option::of(1usize..100),
        rep_penalty in 1.0f32..2.0,
        seed in proptest::option::of(any::<u64>()),
    ) {
        let cfg = GenerationConfig {
            max_new_tokens: max_tokens,
            temperature: temp,
            top_k,
            top_p: None,
            repetition_penalty: rep_penalty,
            do_sample: true,
            seed,
        };
        prop_assert_eq!(cfg.max_new_tokens, max_tokens);
        prop_assert!((cfg.temperature - temp).abs() < 1e-6);
        prop_assert_eq!(cfg.top_k, top_k);
        prop_assert_eq!(cfg.seed, seed);
    }
}

// ── ModelMetadata ─────────────────────────────────────────────────────────────

proptest! {
    /// ModelMetadata fields round-trip through JSON without data loss.
    #[test]
    fn prop_model_metadata_json_roundtrip(
        name in "[a-z][a-z0-9-]{0,30}",
        vocab_size in 100usize..50000,
        context_len in 128usize..8192,
    ) {
        let meta = ModelMetadata {
            name: name.clone(),
            version: "0.1.0".to_string(),
            architecture: "bitnet".to_string(),
            vocab_size,
            context_length: context_len,
            quantization: Some(QuantizationType::I2S),
            fingerprint: None,
            corrections_applied: None,
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        let back: ModelMetadata = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back.name, name);
        prop_assert_eq!(back.vocab_size, vocab_size);
        prop_assert_eq!(back.context_length, context_len);
    }
}

// ── PerformanceMetrics ────────────────────────────────────────────────────────

proptest! {
    /// PerformanceMetrics fields are preserved exactly after construction.
    #[test]
    fn prop_perf_metrics_fields_preserved(
        tps in 0.0f64..1000.0,
        lat_ms in 0.0f64..60_000.0,
        mem_mb in 0.0f64..100_000.0,
    ) {
        let m = PerformanceMetrics {
            tokens_per_second: tps,
            latency_ms: lat_ms,
            memory_usage_mb: mem_mb,
            gpu_utilization: None,
        };
        prop_assert!((m.tokens_per_second - tps).abs() < 1e-10);
        prop_assert!((m.latency_ms - lat_ms).abs() < 1e-10);
        prop_assert!((m.memory_usage_mb - mem_mb).abs() < 1e-10);
    }
}

// ── KernelCapabilities + BackendSelection ─────────────────────────────────────

proptest! {
    /// KernelBackend::requires_gpu semantics are correct for each variant.
    #[test]
    fn prop_kernel_backend_requires_gpu(_dummy in 0u8..1) {
        prop_assert!(!KernelBackend::CpuRust.requires_gpu());
        prop_assert!(KernelBackend::Cuda.requires_gpu());
        prop_assert!(!KernelBackend::CppFfi.requires_gpu());
    }

    /// SimdLevel ordering: Scalar < Neon < Sse42 < Avx2 < Avx512.
    #[test]
    fn prop_simd_level_ordering(_dummy in 0u8..1) {
        prop_assert!(SimdLevel::Scalar < SimdLevel::Neon);
        prop_assert!(SimdLevel::Neon < SimdLevel::Sse42);
        prop_assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
        prop_assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    /// CPU-only caps select CPU backend successfully.
    #[test]
    fn prop_backend_select_cpu_succeeds(_dummy in 0u8..1) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Scalar,
        };
        let result = select_backend(BackendRequest::Cpu, &caps);
        prop_assert!(result.is_ok());
        let sel = result.unwrap();
        let summary = sel.summary();
        prop_assert!(!summary.is_empty());
        prop_assert!(summary.contains("selected="));
    }

    /// Auto selection with CPU-only caps always picks cpu-rust.
    #[test]
    fn prop_backend_select_auto_cpu_only(_dummy in 0u8..1) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        };
        let result = select_backend(BackendRequest::Auto, &caps);
        prop_assert!(result.is_ok());
        let sel = result.unwrap();
        prop_assert!(sel.summary().contains("cpu-rust"));
    }

    /// Requesting CUDA when not compiled always fails.
    #[test]
    fn prop_backend_select_cuda_unavailable(_dummy in 0u8..1) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Scalar,
        };
        let result = select_backend(BackendRequest::Cuda, &caps);
        prop_assert!(result.is_err());
    }
}

// ── MockTensor shape invariants ───────────────────────────────────────────────

proptest! {
    /// Element count equals the product of all dimensions for positive-dimension shapes.
    #[test]
    fn prop_mock_tensor_element_count_is_product(
        dims in proptest::collection::vec(1usize..32, 1..=4),
    ) {
        let expected: usize = dims.iter().product();
        let t = MockTensor::new(dims.clone());
        prop_assert_eq!(t.shape(), dims.as_slice());
        let count: usize = t.shape().iter().product();
        prop_assert_eq!(count, expected);
    }

    /// A shape containing a zero dimension has zero total elements.
    #[test]
    fn prop_mock_tensor_zero_dim_means_zero_elements(
        pre  in proptest::collection::vec(1usize..32, 0..=3),
        post in proptest::collection::vec(1usize..32, 0..=3),
    ) {
        let mut dims = pre;
        dims.push(0);
        dims.extend(post);
        let t = MockTensor::new(dims);
        let count: usize = t.shape().iter().product();
        prop_assert_eq!(count, 0usize);
    }
}

// ── QuantizationType round-trips ──────────────────────────────────────────────

proptest! {
    /// Every QuantizationType survives a JSON serialize → deserialize round-trip.
    #[test]
    fn prop_quantization_type_serde_roundtrip(
        qt in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ]
    ) {
        let json = serde_json::to_string(&qt).expect("serialize");
        let back: QuantizationType = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(qt, back);
    }
}

// ── BitNetError message invariants ────────────────────────────────────────────

proptest! {
    /// BitNetError::Config display is non-empty and echoes the original message.
    #[test]
    fn prop_bitnet_error_config_display_non_empty(
        msg in "[a-zA-Z0-9 _-]{1,64}",
    ) {
        let err = BitNetError::Config(msg.clone());
        let s = err.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.contains(&msg));
    }

    /// BitNetError::Validation display and Debug are always non-empty.
    #[test]
    fn prop_bitnet_error_validation_non_empty(
        msg in "[a-zA-Z0-9 _-]{1,64}",
    ) {
        let err = BitNetError::Validation(msg);
        prop_assert!(!err.to_string().is_empty());
        let dbg = format!("{err:?}");
        prop_assert!(!dbg.is_empty());
    }

    /// BitNetError::StrictMode display and Debug are always non-empty.
    #[test]
    fn prop_bitnet_error_strict_mode_non_empty(
        msg in "[a-zA-Z0-9 _-]{1,64}",
    ) {
        let err = BitNetError::StrictMode(msg);
        prop_assert!(!err.to_string().is_empty());
        let dbg = format!("{err:?}");
        prop_assert!(!dbg.is_empty());
    }
}

// ── Device debug invariants ───────────────────────────────────────────────────

proptest! {
    /// Debug output for every Device variant is non-empty.
    #[test]
    fn prop_device_debug_non_empty(idx in 0usize..256) {
        let cpu_dbg = format!("{:?}", Device::Cpu);
        let cuda_dbg = format!("{:?}", Device::Cuda(idx));
        let metal_dbg = format!("{:?}", Device::Metal);
        prop_assert!(!cpu_dbg.is_empty());
        prop_assert!(!cuda_dbg.is_empty());
        prop_assert!(!metal_dbg.is_empty());
    }
}

// ── KernelCapabilities: cpu_rust implies CpuRust reachable ───────────────────

proptest! {
    /// When cpu_rust is true, CpuRust appears in compiled_backends and best_available is Some.
    #[test]
    fn prop_caps_cpu_rust_implies_cpu_reachable(
        cuda_compiled in any::<bool>(),
        cpp_ffi      in any::<bool>(),
    ) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled,
            cuda_runtime: false, // no GPU at runtime; CPU must still be reachable
            hip_compiled: false,
            hip_runtime: false,
            oneapi_compiled: false,
            oneapi_runtime: false,
            cpp_ffi,
            simd_level: SimdLevel::Scalar,
        };
        let backends = caps.compiled_backends();
        prop_assert!(
            backends.contains(&KernelBackend::CpuRust),
            "cpu_rust=true but CpuRust absent from {:?}", backends
        );
        prop_assert!(caps.best_available().is_some());
    }
}

// ── warn_once! macro — key-based deduplication ───────────────────────────────

proptest! {
    /// warn_once_fn never panics for arbitrary valid string keys and messages.
    /// Calling twice with the same key must also be safe (second call is rate-limited).
    #[test]
    fn prop_warn_once_fn_no_panic(
        key in "[a-z][a-z0-9_]{0,31}",
        msg in "[a-zA-Z0-9 ]{1,64}",
    ) {
        warn_once_fn(&key, &msg);
        // Second call with same key: rate-limited to DEBUG, must not panic.
        warn_once_fn(&key, &msg);
    }
}
