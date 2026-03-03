//! Edge-case tests for bitnet-common types focused on CPU inference path coverage gaps.
//!
//! Covers: Device::to_candle(), apply_architecture_defaults(), ConfigLoader multi-source,
//! ModelConfig field interactions, KernelCapabilities, SimdLevel, BackendStartupSummary,
//! and strict mode validation through the config pipeline.

use bitnet_common::{
    BackendRequest, BackendStartupSummary, BitNetConfig, ConfigSource, Device, KernelBackend,
    KernelCapabilities, ModelConfig, QuantizationType, SimdLevel, StrictModeConfig,
    StrictModeEnforcer,
    config::{ActivationType, ConfigLoader, NormType, RopeScaling},
    select_backend,
    strict_mode::{ComputationType, MockInferencePath, PerformanceMetrics},
};

// ── Device::to_candle() ────────────────────────────────────────────────

#[test]
fn device_cpu_to_candle_returns_cpu() {
    let candle = Device::Cpu.to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

#[test]
fn device_npu_to_candle_falls_back_to_cpu() {
    let candle = Device::Npu.to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

#[test]
fn device_hip_to_candle_falls_back_to_cpu() {
    let candle = Device::Hip(0).to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

#[test]
fn device_metal_to_candle_falls_back_to_cpu_on_non_macos() {
    // On non-macOS (or without metal feature), Metal falls back to CPU
    let candle = Device::Metal.to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

#[test]
fn device_opencl_to_candle_falls_back_to_cpu() {
    let candle = Device::OpenCL(3).to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

#[test]
fn device_cuda_usize_max_to_candle_falls_back_to_cpu() {
    // Unknown ordinal (usize::MAX) should fall back to CPU in CPU-only builds
    let candle = Device::Cuda(usize::MAX).to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

#[test]
fn device_cuda_zero_to_candle_falls_back_to_cpu_without_gpu() {
    // Without GPU feature, Cuda(0) falls back to CPU
    let candle = Device::Cuda(0).to_candle().unwrap();
    assert!(matches!(candle, candle_core::Device::Cpu));
}

// ── Device predicates ──────────────────────────────────────────────────

#[test]
fn device_predicate_exhaustive() {
    assert!(Device::Cpu.is_cpu());
    assert!(!Device::Cpu.is_cuda());
    assert!(!Device::Cpu.is_hip());
    assert!(!Device::Cpu.is_npu());
    assert!(!Device::Cpu.is_opencl());

    assert!(Device::Cuda(0).is_cuda());
    assert!(!Device::Cuda(0).is_cpu());

    assert!(Device::Hip(0).is_hip());
    assert!(!Device::Hip(0).is_cuda());

    assert!(Device::Npu.is_npu());
    assert!(!Device::Npu.is_cpu());

    assert!(Device::OpenCL(0).is_opencl());
    assert!(!Device::OpenCL(0).is_cpu());

    // Metal has no is_metal(), just check it's not CPU/CUDA
    assert!(!Device::Metal.is_cpu());
    assert!(!Device::Metal.is_cuda());
}

#[test]
fn device_default_is_cpu() {
    assert_eq!(Device::default(), Device::Cpu);
}

#[test]
fn device_ordering() {
    // Device derives Ord; verify CPU < Cuda < Hip < Npu < Metal < OpenCL
    assert!(Device::Cpu < Device::Cuda(0));
    assert!(Device::Cuda(0) < Device::Cuda(1));
    assert!(Device::Cuda(usize::MAX) < Device::Hip(0));
}

// ── From<&candle_core::Device> for Device ──────────────────────────────

#[test]
fn device_from_candle_cpu() {
    let candle_cpu = candle_core::Device::Cpu;
    let dev: Device = (&candle_cpu).into();
    assert_eq!(dev, Device::Cpu);
}

// ── apply_architecture_defaults() ──────────────────────────────────────

#[test]
fn apply_architecture_defaults_llama_sets_rmsnorm_silu() {
    let mut cfg = ModelConfig::default();
    assert_eq!(cfg.norm_type, NormType::LayerNorm); // default
    cfg.apply_architecture_defaults("llama");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn apply_architecture_defaults_phi2_sets_layernorm_gelu() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-2");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

#[test]
fn apply_architecture_defaults_phi4_overrides_context_length() {
    let mut cfg = ModelConfig::default();
    assert_eq!(cfg.max_position_embeddings, 2048);
    cfg.apply_architecture_defaults("phi-4");
    assert_eq!(cfg.max_position_embeddings, 16384);
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
}

#[test]
fn apply_architecture_defaults_preserves_custom_context_length() {
    let mut cfg = ModelConfig::default();
    cfg.max_position_embeddings = 4096; // custom, not default 2048
    cfg.apply_architecture_defaults("phi-4");
    // phi-4 default is 16384, but should NOT override since 4096 != 2048
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn apply_architecture_defaults_unknown_is_noop() {
    let mut cfg = ModelConfig::default();
    let original_norm = cfg.norm_type;
    let original_activation = cfg.activation_type;
    let original_ctx = cfg.max_position_embeddings;
    cfg.apply_architecture_defaults("totally_unknown_arch");
    assert_eq!(cfg.norm_type, original_norm);
    assert_eq!(cfg.activation_type, original_activation);
    assert_eq!(cfg.max_position_embeddings, original_ctx);
}

#[test]
fn apply_architecture_defaults_case_insensitive() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("LLAMA");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
}

#[test]
fn apply_architecture_defaults_bitnet_keeps_layernorm() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("bitnet");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    // bitnet has no default context length
    assert_eq!(cfg.max_position_embeddings, 2048);
}

#[test]
fn apply_architecture_defaults_gpt_sets_gelu() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("gpt");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
}

#[test]
fn apply_architecture_defaults_deepseek_v3_large_context() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("deepseek-v3");
    assert_eq!(cfg.max_position_embeddings, 65536);
}

// ── ModelConfig field interactions (rope_theta, rope_scaling) ──────────

#[test]
fn model_config_default_rope_fields_are_none() {
    let cfg = ModelConfig::default();
    assert!(cfg.rope_theta.is_none());
    assert!(cfg.rope_scaling.is_none());
    assert!(cfg.rms_norm_eps.is_none());
}

#[test]
fn model_config_rope_theta_roundtrip() {
    let mut cfg = ModelConfig::default();
    cfg.rope_theta = Some(10000.0);
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.rope_theta, Some(10000.0));
}

#[test]
fn model_config_rope_scaling_roundtrip() {
    let mut cfg = ModelConfig::default();
    cfg.rope_scaling = Some(RopeScaling { scaling_type: "linear".to_string(), factor: 2.0 });
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: ModelConfig = serde_json::from_str(&json).unwrap();
    let scaling = restored.rope_scaling.unwrap();
    assert_eq!(scaling.scaling_type, "linear");
    assert!((scaling.factor - 2.0).abs() < f32::EPSILON);
}

#[test]
fn model_config_gqa_num_kv_heads_zero_means_mha() {
    let cfg = ModelConfig::default();
    assert_eq!(cfg.num_key_value_heads, 0);
    // 0 means "use num_heads" (MHA)
}

// ── ConfigLoader multi-source loading ──────────────────────────────────

#[test]
fn config_loader_empty_sources_returns_defaults() {
    let cfg = ConfigLoader::load_from_sources(&[]).unwrap();
    assert_eq!(cfg.model.vocab_size, 32000);
    assert_eq!(cfg.inference.temperature, 1.0);
}

#[test]
fn config_loader_inline_source_overrides_defaults() {
    let mut override_cfg = BitNetConfig::default();
    override_cfg.model.vocab_size = 64000;
    override_cfg.model.hidden_size = 2048;
    override_cfg.model.num_heads = 16;

    let cfg =
        ConfigLoader::load_from_sources(&[ConfigSource::Inline(Box::new(override_cfg))]).unwrap();
    assert_eq!(cfg.model.vocab_size, 64000);
    assert_eq!(cfg.model.hidden_size, 2048);
}

#[test]
fn config_loader_multiple_inline_sources_last_wins() {
    let mut first = BitNetConfig::default();
    first.model.vocab_size = 50000;
    first.model.hidden_size = 1024;
    first.model.num_heads = 8;

    let mut second = BitNetConfig::default();
    second.model.vocab_size = 64000;
    second.model.hidden_size = 1024;
    second.model.num_heads = 8;

    let cfg = ConfigLoader::load_from_sources(&[
        ConfigSource::Inline(Box::new(first)),
        ConfigSource::Inline(Box::new(second)),
    ])
    .unwrap();
    assert_eq!(cfg.model.vocab_size, 64000);
}

#[test]
fn config_loader_file_source_missing_file_errors() {
    let result =
        ConfigLoader::load_from_sources(&[ConfigSource::File("/nonexistent/config.toml".into())]);
    assert!(result.is_err());
}

#[test]
fn config_loader_file_source_unsupported_extension_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(&path, "key: value").unwrap();

    let result = ConfigLoader::load_from_sources(&[ConfigSource::File(path)]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Unsupported config file format"));
}

// ── Config merge_with ──────────────────────────────────────────────────

#[test]
fn config_merge_preserves_base_when_other_is_default() {
    let mut base = BitNetConfig::default();
    base.model.vocab_size = 50000;
    base.model.hidden_size = 2048;
    base.model.num_heads = 16;

    let other = BitNetConfig::default();
    base.merge_with(other);

    // Other has default vocab_size (32000), so base's 50000 should be preserved
    // UNLESS other.vocab_size != default → it IS default, so base keeps 50000
    assert_eq!(base.model.vocab_size, 50000);
}

#[test]
fn config_merge_overrides_non_default_fields() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.inference.seed = Some(42);
    other.performance.num_threads = Some(8);
    other.performance.memory_limit = Some(1024 * 1024 * 1024);

    base.merge_with(other);
    assert_eq!(base.inference.seed, Some(42));
    assert_eq!(base.performance.num_threads, Some(8));
    assert_eq!(base.performance.memory_limit, Some(1024 * 1024 * 1024));
}

#[test]
fn config_merge_model_path_from_other() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.model.path = Some("/models/test.gguf".into());

    base.merge_with(other);
    assert_eq!(base.model.path, Some("/models/test.gguf".into()));
}

// ── Config validation edge cases ───────────────────────────────────────

#[test]
fn config_validation_rejects_zero_vocab_size() {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = 0;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("vocab_size"));
}

#[test]
fn config_validation_rejects_non_divisible_hidden_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 100;
    cfg.model.num_heads = 7;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("divisible"));
}

#[test]
fn config_validation_rejects_kv_heads_greater_than_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_key_value_heads = 64;
    cfg.model.num_heads = 32;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("num_key_value_heads"));
}

#[test]
fn config_validation_rejects_non_divisible_kv_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 5; // 32 not divisible by 5
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("divisible by num_key_value_heads"));
}

#[test]
fn config_validation_rejects_zero_temperature() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.temperature = 0.0;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("temperature"));
}

#[test]
fn config_validation_rejects_top_p_out_of_range() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_p = Some(1.5);
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("top_p"));
}

#[test]
fn config_validation_rejects_non_power_of_two_block_size() {
    let mut cfg = BitNetConfig::default();
    cfg.quantization.block_size = 65;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("power of 2"));
}

#[test]
fn config_validation_accepts_valid_gqa_config() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 8; // GQA: 32/8 = 4 groups
    assert!(cfg.validate().is_ok());
}

#[test]
fn config_validation_accumulates_multiple_errors() {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = 0;
    cfg.model.hidden_size = 0;
    cfg.model.num_layers = 0;
    cfg.model.num_heads = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("vocab_size"));
    assert!(msg.contains("hidden_size"));
    assert!(msg.contains("num_layers"));
    assert!(msg.contains("num_heads"));
}

// ── ConfigBuilder ──────────────────────────────────────────────────────

#[test]
fn config_builder_basic_roundtrip() {
    let cfg = BitNetConfig::builder()
        .vocab_size(50000)
        .hidden_size(2048)
        .num_layers(24)
        .num_heads(16)
        .temperature(0.7)
        .build()
        .unwrap();
    assert_eq!(cfg.model.vocab_size, 50000);
    assert_eq!(cfg.model.hidden_size, 2048);
    assert_eq!(cfg.inference.temperature, 0.7);
}

#[test]
fn config_builder_rejects_invalid() {
    let result = BitNetConfig::builder().vocab_size(0).build();
    assert!(result.is_err());
}

#[test]
fn config_builder_gqa_config() {
    let cfg = BitNetConfig::builder().num_heads(32).num_key_value_heads(8).build().unwrap();
    assert_eq!(cfg.model.num_key_value_heads, 8);
}

// ── KernelCapabilities and SimdLevel ───────────────────────────────────

#[test]
fn simd_level_display_all_variants() {
    assert_eq!(SimdLevel::Scalar.to_string(), "scalar");
    assert_eq!(SimdLevel::Neon.to_string(), "neon");
    assert_eq!(SimdLevel::Sse42.to_string(), "sse4.2");
    assert_eq!(SimdLevel::Avx2.to_string(), "avx2");
    assert_eq!(SimdLevel::Avx512.to_string(), "avx512");
}

#[test]
fn simd_level_total_ordering() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    for i in 0..levels.len() - 1 {
        assert!(levels[i] < levels[i + 1], "{:?} should be < {:?}", levels[i], levels[i + 1]);
    }
}

#[test]
fn kernel_backend_display_all_variants() {
    assert_eq!(KernelBackend::CpuRust.to_string(), "cpu-rust");
    assert_eq!(KernelBackend::Cuda.to_string(), "cuda");
    assert_eq!(KernelBackend::Hip.to_string(), "hip");
    assert_eq!(KernelBackend::OneApi.to_string(), "oneapi");
    assert_eq!(KernelBackend::OpenCL.to_string(), "opencl");
    assert_eq!(KernelBackend::CppFfi.to_string(), "cpp-ffi");
}

#[test]
fn kernel_backend_requires_gpu_classification() {
    assert!(!KernelBackend::CpuRust.requires_gpu());
    assert!(KernelBackend::Cuda.requires_gpu());
    assert!(KernelBackend::Hip.requires_gpu());
    assert!(KernelBackend::OneApi.requires_gpu());
    assert!(KernelBackend::OpenCL.requires_gpu());
    assert!(!KernelBackend::CppFfi.requires_gpu());
}

#[test]
fn kernel_backend_is_compiled_cpu_feature() {
    // With --features cpu, CpuRust should be compiled
    assert!(KernelBackend::CpuRust.is_compiled());
    // Without cuda feature, Cuda should not be compiled
    #[cfg(not(feature = "cuda"))]
    assert!(!KernelBackend::Cuda.is_compiled());
}

#[test]
fn kernel_capabilities_from_compile_time_cpu_only() {
    let caps = KernelCapabilities::from_compile_time();
    #[cfg(feature = "cpu")]
    assert!(caps.cpu_rust);
    // No runtime probing done, so cuda_runtime is always false
    assert!(!caps.cuda_runtime);
}

#[test]
fn kernel_capabilities_with_runtime_builders_chain() {
    let caps = KernelCapabilities::from_compile_time()
        .with_cuda_runtime(false)
        .with_hip_runtime(false)
        .with_oneapi_runtime(false)
        .with_opencl_runtime(false)
        .with_cpp_ffi(false);

    assert!(!caps.cuda_runtime);
    assert!(!caps.hip_runtime);
    assert!(!caps.oneapi_runtime);
    assert!(!caps.opencl_runtime);
    assert!(!caps.cpp_ffi);
}

#[test]
fn kernel_capabilities_best_available_empty() {
    let caps = KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    assert!(caps.best_available().is_none());
}

#[test]
fn kernel_capabilities_best_available_prefers_cuda_over_hip() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: true,
        hip_runtime: true,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::Cuda));
}

#[test]
fn kernel_capabilities_best_available_prefers_ffi_over_cpu() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Scalar,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CppFfi));
}

#[test]
fn kernel_capabilities_compiled_backends_order() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        hip_compiled: true,
        hip_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: false,
        opencl_compiled: true,
        opencl_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx2,
    };
    let backends = caps.compiled_backends();
    // Order: Cuda, Hip, OneApi, OpenCL, CppFfi, CpuRust
    assert_eq!(backends[0], KernelBackend::Cuda);
    assert_eq!(backends[1], KernelBackend::Hip);
    assert_eq!(backends[2], KernelBackend::OneApi);
    assert_eq!(backends[3], KernelBackend::OpenCL);
    assert_eq!(backends[4], KernelBackend::CppFfi);
    assert_eq!(backends[5], KernelBackend::CpuRust);
}

#[test]
fn kernel_capabilities_summary_format() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    let summary = caps.summary();
    assert!(summary.starts_with("simd=avx2"), "got: {summary}");
    assert!(summary.contains("cpu-rust"), "got: {summary}");
}

// ── BackendStartupSummary ──────────────────────────────────────────────

#[test]
fn backend_startup_summary_new_and_log_line() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".to_string()], "cpu-rust");
    assert_eq!(summary.requested, "auto");
    assert_eq!(summary.selected, "cpu-rust");
    assert_eq!(summary.detected, vec!["cpu-rust".to_string()]);

    let log = summary.log_line();
    assert!(log.contains("requested=auto"), "got: {log}");
    assert!(log.contains("detected=[cpu-rust]"), "got: {log}");
    assert!(log.contains("selected=cpu-rust"), "got: {log}");
}

#[test]
fn backend_startup_summary_multiple_detected() {
    let summary =
        BackendStartupSummary::new("gpu", vec!["cuda".to_string(), "cpu-rust".to_string()], "cuda");
    let log = summary.log_line();
    assert!(log.contains("detected=[cuda, cpu-rust]"), "got: {log}");
}

#[test]
fn backend_startup_summary_empty_detected() {
    let summary = BackendStartupSummary::new("auto", vec![], "cpu-rust");
    let log = summary.log_line();
    assert!(log.contains("detected=[]"), "got: {log}");
}

#[test]
fn backend_startup_summary_serialization_roundtrip() {
    let summary = BackendStartupSummary::new("cpu", vec!["cpu-rust".to_string()], "cpu-rust");
    let json = serde_json::to_string(&summary).unwrap();
    let restored: BackendStartupSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(summary, restored);
}

// ── select_backend edge cases ──────────────────────────────────────────

fn cpu_only_caps() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

#[test]
fn select_backend_hip_request_without_hip_fails() {
    let err = select_backend(BackendRequest::Hip, &cpu_only_caps()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not available"), "got: {msg}");
}

#[test]
fn select_backend_oneapi_request_without_oneapi_fails() {
    let err = select_backend(BackendRequest::OneApi, &cpu_only_caps()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not available"), "got: {msg}");
}

#[test]
fn select_backend_cpu_request_without_cpu_fails() {
    let caps = KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    let err = select_backend(BackendRequest::Cpu, &caps).unwrap_err();
    assert!(err.to_string().contains("not available"));
}

#[test]
fn select_backend_gpu_prefers_hip_when_no_cuda_runtime() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: true,
        hip_runtime: true,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    let result = select_backend(BackendRequest::Gpu, &caps).unwrap();
    assert_eq!(result.selected, KernelBackend::Hip);
}

#[test]
fn select_backend_gpu_prefers_oneapi_when_no_cuda_no_hip() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: true,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    let result = select_backend(BackendRequest::Gpu, &caps).unwrap();
    assert_eq!(result.selected, KernelBackend::OneApi);
}

#[test]
fn select_backend_result_summary_format() {
    let result = select_backend(BackendRequest::Auto, &cpu_only_caps()).unwrap();
    let summary = result.summary();
    assert!(summary.contains("requested=auto"));
    assert!(summary.contains("selected=cpu-rust"));
}

#[test]
fn backend_request_display_all_variants() {
    assert_eq!(BackendRequest::Auto.to_string(), "auto");
    assert_eq!(BackendRequest::Cpu.to_string(), "cpu");
    assert_eq!(BackendRequest::Gpu.to_string(), "gpu");
    assert_eq!(BackendRequest::Cuda.to_string(), "cuda");
    assert_eq!(BackendRequest::Hip.to_string(), "hip");
    assert_eq!(BackendRequest::OneApi.to_string(), "oneapi");
}

// ── StrictModeConfig through config pipeline ───────────────────────────

#[test]
fn strict_mode_config_disabled_allows_mock() {
    let config = StrictModeConfig {
        enabled: false,
        fail_on_mock: false,
        require_quantization: false,
        enforce_quantized_inference: false,
        validate_performance: false,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let mock_path = MockInferencePath {
        description: "test mock".to_string(),
        uses_mock_computation: true,
        fallback_reason: "testing".to_string(),
    };
    assert!(config.validate_inference_path(&mock_path).is_ok());
}

#[test]
fn strict_mode_config_enabled_rejects_mock() {
    let config = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let mock_path = MockInferencePath {
        description: "test mock".to_string(),
        uses_mock_computation: true,
        fallback_reason: "testing".to_string(),
    };
    let err = config.validate_inference_path(&mock_path).unwrap_err();
    assert!(format!("{err}").contains("Strict mode"));
}

#[test]
fn strict_mode_performance_rejects_suspicious_tps() {
    let config = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let metrics = PerformanceMetrics {
        tokens_per_second: 200.0, // above 150.0 threshold
        latency_ms: 5.0,
        memory_usage_mb: 100.0,
        computation_type: ComputationType::Real,
        gpu_utilization: None,
    };
    let err = config.validate_performance_metrics(&metrics).unwrap_err();
    assert!(format!("{err}").contains("Suspicious performance"));
}

#[test]
fn strict_mode_performance_accepts_normal_tps() {
    let config = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let metrics = PerformanceMetrics {
        tokens_per_second: 30.0,
        latency_ms: 33.0,
        memory_usage_mb: 500.0,
        computation_type: ComputationType::Real,
        gpu_utilization: None,
    };
    assert!(config.validate_performance_metrics(&metrics).is_ok());
}

#[test]
fn strict_mode_rejects_mock_computation_type() {
    let config = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let metrics = PerformanceMetrics {
        tokens_per_second: 10.0,
        computation_type: ComputationType::Mock,
        ..Default::default()
    };
    let err = config.validate_performance_metrics(&metrics).unwrap_err();
    assert!(format!("{err}").contains("Mock computation"));
}

#[test]
fn strict_mode_disabled_skips_performance_validation() {
    let config = StrictModeConfig {
        enabled: false,
        fail_on_mock: false,
        require_quantization: false,
        enforce_quantized_inference: false,
        validate_performance: false,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let metrics = PerformanceMetrics {
        tokens_per_second: 999999.0,
        computation_type: ComputationType::Mock,
        ..Default::default()
    };
    assert!(config.validate_performance_metrics(&metrics).is_ok());
}

#[test]
fn strict_mode_quantization_fallback_rejected_when_enabled() {
    let config = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    let err = config
        .validate_quantization_fallback(
            QuantizationType::I2S,
            Device::Cpu,
            &[4096, 4096],
            "no kernel available",
        )
        .unwrap_err();
    assert!(format!("{err}").contains("FP32 fallback rejected"));
}

#[test]
fn strict_mode_quantization_fallback_allowed_when_disabled() {
    let config = StrictModeConfig {
        enabled: false,
        fail_on_mock: false,
        require_quantization: false,
        enforce_quantized_inference: false,
        validate_performance: false,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };

    assert!(
        config
            .validate_quantization_fallback(
                QuantizationType::I2S,
                Device::Cpu,
                &[4096, 4096],
                "no kernel",
            )
            .is_ok()
    );
}

// ── StrictModeEnforcer with explicit config ────────────────────────────

#[test]
fn strict_mode_enforcer_with_explicit_enabled_config() {
    let config = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    assert!(enforcer.is_enabled());

    let mock_path = MockInferencePath {
        description: "mock test".to_string(),
        uses_mock_computation: true,
        fallback_reason: "test".to_string(),
    };
    assert!(enforcer.validate_inference_path(&mock_path).is_err());
}

#[test]
fn strict_mode_enforcer_with_explicit_disabled_config() {
    let config = StrictModeConfig {
        enabled: false,
        fail_on_mock: false,
        require_quantization: false,
        enforce_quantized_inference: false,
        validate_performance: false,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    assert!(!enforcer.is_enabled());

    let mock_path = MockInferencePath {
        description: "mock test".to_string(),
        uses_mock_computation: true,
        fallback_reason: "test".to_string(),
    };
    assert!(enforcer.validate_inference_path(&mock_path).is_ok());
}

// ── QuantizationType ───────────────────────────────────────────────────

#[test]
fn quantization_type_display() {
    assert_eq!(QuantizationType::I2S.to_string(), "I2_S");
    assert_eq!(QuantizationType::TL1.to_string(), "TL1");
    assert_eq!(QuantizationType::TL2.to_string(), "TL2");
}

#[test]
fn quantization_type_serialization_roundtrip() {
    for qt in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let json = serde_json::to_string(&qt).unwrap();
        let restored: QuantizationType = serde_json::from_str(&json).unwrap();
        assert_eq!(qt, restored);
    }
}

// ── Config TOML/JSON loading ───────────────────────────────────────────

#[test]
fn config_from_toml_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_config.toml");
    std::fs::write(
        &path,
        r#"
[model]
vocab_size = 50000
hidden_size = 2048
num_layers = 24
num_heads = 16

[inference]
temperature = 0.8

[quantization]
block_size = 128
"#,
    )
    .unwrap();

    let cfg = BitNetConfig::from_file(&path).unwrap();
    assert_eq!(cfg.model.vocab_size, 50000);
    assert_eq!(cfg.model.hidden_size, 2048);
    assert_eq!(cfg.inference.temperature, 0.8);
    assert_eq!(cfg.quantization.block_size, 128);
}

#[test]
fn config_from_json_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_config.json");
    std::fs::write(
        &path,
        r#"{
  "model": { "vocab_size": 64000, "hidden_size": 4096, "num_layers": 32, "num_heads": 32 },
  "inference": { "temperature": 0.5 }
}"#,
    )
    .unwrap();

    let cfg = BitNetConfig::from_file(&path).unwrap();
    assert_eq!(cfg.model.vocab_size, 64000);
    assert_eq!(cfg.inference.temperature, 0.5);
}

#[test]
fn config_from_invalid_toml_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.toml");
    std::fs::write(&path, "this is not valid toml {{{{").unwrap();

    let err = BitNetConfig::from_file(&path).unwrap_err();
    assert!(format!("{err}").contains("TOML"));
}

#[test]
fn config_from_invalid_json_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.json");
    std::fs::write(&path, "not json at all").unwrap();

    let err = BitNetConfig::from_file(&path).unwrap_err();
    assert!(format!("{err}").contains("JSON"));
}
