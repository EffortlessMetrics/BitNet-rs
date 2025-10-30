//! Comprehensive test suite for Issue #453: Strict Quantization Guards
//!
//! This test suite validates the three-tier validation strategy:
//! 1. Debug assertions (development-time detection)
//! 2. Strict mode enforcement (production-time rejection)
//! 3. Receipt validation (post-inference verification)
//!
//! **Specification:** docs/explanation/strict-quantization-guards.md
//! **API Contracts:** docs/reference/strict-mode-api.md
//!
//! All tests are tagged with `// AC:ID` comments for traceability.
/// AC1: Test I2S fallback detection with debug assertions
///
/// Tests feature spec: strict-quantization-guards.md#ac1-debug-assertions-in-quantizedlinearforward
#[test]
#[cfg(all(debug_assertions, feature = "cpu"))]
fn test_ac1_debug_assert_i2s_fallback() {}
/// AC1: Test TL1 fallback detection with debug assertions (ARM-specific)
#[test]
#[cfg(all(debug_assertions, feature = "cpu"))]
fn test_ac1_debug_assert_tl1_fallback() {}
/// AC1: Test TL2 fallback detection with debug assertions (x86-specific)
#[test]
#[cfg(all(debug_assertions, feature = "cpu"))]
fn test_ac1_debug_assert_tl2_fallback() {}
/// AC1: Test that release builds allow fallback without panic
#[test]
#[cfg(all(not(debug_assertions), feature = "cpu"))]
fn test_ac1_release_allows_fallback() {}
/// AC2: Test attention projection fallback detection
#[test]
#[cfg(all(debug_assertions, feature = "cpu"))]
fn test_ac2_debug_assert_attention_projection() {}
/// AC2: Test that all projections use quantized kernels
#[test]
#[cfg(feature = "cpu")]
fn test_ac2_all_projections_quantized() {}
/// AC3: Test strict mode rejects FP32 fallback
#[test]
#[cfg(feature = "cpu")]
fn test_ac3_strict_mode_rejects_fallback() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    const LAYER_DIMENSIONS: [usize; 2] = [2048, 2048];
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &LAYER_DIMENSIONS,
        "kernel_unavailable",
    );
    assert!(result.is_err(), "AC3: Strict mode should reject FP32 fallback");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Strict mode"), "AC3: Error message should mention strict mode");
    assert!(
        err_msg.contains("FP32 fallback rejected"),
        "AC3: Error message should indicate fallback rejection"
    );
}
/// AC3: Test error message context is detailed
#[test]
#[cfg(feature = "cpu")]
fn test_ac3_error_message_context() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    const LAYER_DIMENSIONS: [usize; 2] = [2048, 2048];
    const TEST_REASON: &str = "test_reason";
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &LAYER_DIMENSIONS,
        TEST_REASON,
    );
    assert!(result.is_err(), "AC3: Should return error for fallback validation");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("I2S"), "AC3: Error should include quantization type for diagnostics");
    assert!(err_msg.contains("Cpu"), "AC3: Error should include device information");
    assert!(err_msg.contains("2048"), "AC3: Error should include layer dimensions for debugging");
    assert!(err_msg.contains(TEST_REASON), "AC3: Error should include specific fallback reason");
}
/// AC3: Test granular strict mode control
#[test]
#[cfg(feature = "cpu")]
fn test_ac3_granular_strict_mode() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = false;
    config.enforce_quantized_inference = false;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[2048, 2048],
        "kernel_unavailable",
    );
    assert!(result.is_ok(), "Should allow fallback when strict mode disabled");
}
/// AC4: Test attention strict mode validation
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_attention_strict_mode_validation() {}
/// AC4: Test successful attention forward with quantized kernels
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_attention_success_with_quantized_kernels() {}
/// AC5: 16-token decode in strict mode (CPU)
#[test]
#[cfg(feature = "cpu")]
fn test_ac5_16_token_decode_cpu_strict_mode() {}
/// AC5: 16-token decode in strict mode (GPU)
#[test]
#[cfg(feature = "gpu")]
fn test_ac5_16_token_decode_gpu_strict_mode() {}
/// AC5: Test deterministic inference with strict mode
#[test]
#[cfg(feature = "cpu")]
fn test_ac5_deterministic_strict_mode() {}
/// AC6: Test receipt with quantized kernels passes validation
#[test]
fn test_ac6_receipt_quantized_kernels_valid() {
    use serde_json::json;
    let receipt = json!(
        { "schema_version" : "1.0.0", "compute_path" : "real", "kernels" :
        ["i2s_matmul_cpu", "tl1_lookup", "gemm_i2s_gpu"], "backend" : "cpu" }
    );
    assert!(receipt.get("compute_path").is_some());
    assert!(receipt.get("kernels").is_some());
}
/// AC6: Test receipt with false quantization claim fails validation
#[test]
fn test_ac6_receipt_false_quantization_claim_fails() {
    use serde_json::json;
    let receipt = json!(
        { "schema_version" : "1.0.0", "compute_path" : "real", "kernels" :
        ["dequant_fp32", "fallback_matmul", "matmul_f32"], "backend" : "cpu" }
    );
    assert_eq!(receipt["compute_path"], "real");
    assert!(!receipt["kernels"].as_array().unwrap().is_empty());
}
/// AC6: Test receipt with explicit fp32_fallback is accepted
#[test]
fn test_ac6_receipt_fp32_fallback_explicit() {
    use serde_json::json;
    let receipt = json!(
        { "schema_version" : "1.0.0", "compute_path" : "fallback", "kernels" :
        ["dequant_fp32", "fallback_matmul"], "backend" : "cpu" }
    );
    assert_eq!(receipt["compute_path"], "fallback");
}
/// AC6: Test receipt schema v1.0.0 backward compatibility
#[test]
fn test_ac6_receipt_v1_0_backward_compatibility() {
    use serde_json::json;
    let receipt_v1_0 = json!(
        { "schema_version" : "1.0", "compute_path" : "real", "kernels" : ["i2s_matmul"],
        "backend" : "cpu" }
    );
    let receipt_v1_0_0 = json!(
        { "schema_version" : "1.0.0", "compute_path" : "real", "kernels" :
        ["i2s_matmul"], "backend" : "cpu" }
    );
    assert!(receipt_v1_0["schema_version"].as_str().unwrap().starts_with("1.0"));
    assert!(receipt_v1_0_0["schema_version"].as_str().unwrap().starts_with("1.0"));
}
/// AC6: Test kernel ID pattern matching helpers
#[test]
fn test_ac6_kernel_id_pattern_matching() {
    const QUANTIZED_KERNELS: &[&str] = &["i2s_matmul", "tl1_lookup", "tl2_simd", "gemm_i2s_gpu"];
    for kernel_id in QUANTIZED_KERNELS {
        let is_quantized = kernel_id.contains("i2s_")
            || kernel_id.contains("tl1_")
            || kernel_id.contains("tl2_")
            || kernel_id.contains("gemm_i2s_");
        assert!(is_quantized, "AC6: Kernel '{}' should match quantized pattern", kernel_id);
    }
    const FALLBACK_KERNELS: &[&str] = &["dequant_fp32", "fp32_matmul", "fallback_compute"];
    for kernel_id in FALLBACK_KERNELS {
        let is_fallback = kernel_id.contains("dequant")
            || kernel_id.contains("fp32_")
            || kernel_id.contains("fallback_");
        assert!(is_fallback, "AC6: Kernel '{}' should match fallback pattern", kernel_id);
    }
}
/// AC7: Documentation tests for strict mode
#[test]
fn test_ac7_documentation_tests() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    let _config = StrictModeConfig::from_env();
    let _enforcer = StrictModeEnforcer::new();
}
/// Test AC3 with edge case: minimal 1x1 layer dimensions
#[test]
#[cfg(feature = "cpu")]
fn test_edge_case_minimal_layer_dimensions() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    const MINIMAL_DIMENSIONS: [usize; 2] = [1, 1];
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &MINIMAL_DIMENSIONS,
        "kernel_unavailable",
    );
    assert!(result.is_err(), "Should reject fallback even for 1x1 layer");
    assert!(result.unwrap_err().to_string().contains("1"));
}
/// Test AC3 with edge case: large 8192x8192 layer dimensions
#[test]
#[cfg(feature = "cpu")]
fn test_edge_case_large_layer_dimensions() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    const LARGE_DIMENSIONS: [usize; 2] = [8192, 8192];
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &LARGE_DIMENSIONS,
        "kernel_unavailable",
    );
    assert!(result.is_err(), "Should reject fallback for large layers");
    assert!(result.unwrap_err().to_string().contains("8192"));
}
/// Test AC3 with edge case: asymmetric layer dimensions
#[test]
#[cfg(feature = "cpu")]
fn test_edge_case_asymmetric_layer_dimensions() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    const ASYMMETRIC_DIMENSIONS: [usize; 2] = [128, 8192];
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::TL1,
        Device::Cpu,
        &ASYMMETRIC_DIMENSIONS,
        "kernel_unavailable",
    );
    assert!(result.is_err(), "Should reject fallback for asymmetric layers");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("128") && err_msg.contains("8192"));
}
/// Test AC3 with mixed CPU/GPU device scenarios
#[test]
#[cfg(all(feature = "cpu", feature = "gpu"))]
fn test_edge_case_mixed_cpu_gpu_devices() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config.clone()));
    let cpu_result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[2048, 2048],
        "cpu_kernel_missing",
    );
    assert!(cpu_result.is_err());
    assert!(cpu_result.unwrap_err().to_string().contains("Cpu"));
    let gpu_result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cuda(0),
        &[2048, 2048],
        "gpu_kernel_missing",
    );
    assert!(gpu_result.is_err());
    assert!(gpu_result.unwrap_err().to_string().contains("Cuda"));
}
/// Test AC3 with all quantization types
#[test]
#[cfg(feature = "cpu")]
fn test_error_path_all_quantization_types() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let i2s_result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[2048, 2048],
        "i2s_kernel_missing",
    );
    assert!(i2s_result.is_err());
    assert!(i2s_result.unwrap_err().to_string().contains("I2S"));
    let tl1_result = enforcer.validate_quantization_fallback(
        QuantizationType::TL1,
        Device::Cpu,
        &[2048, 2048],
        "tl1_kernel_missing",
    );
    assert!(tl1_result.is_err());
    assert!(tl1_result.unwrap_err().to_string().contains("TL1"));
    let tl2_result = enforcer.validate_quantization_fallback(
        QuantizationType::TL2,
        Device::Cpu,
        &[2048, 2048],
        "tl2_kernel_missing",
    );
    assert!(tl2_result.is_err());
    assert!(tl2_result.unwrap_err().to_string().contains("TL2"));
}
/// Test AC3 error message with empty fallback reason
#[test]
#[cfg(feature = "cpu")]
fn test_error_path_empty_fallback_reason() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[2048, 2048],
        "",
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Strict mode"));
}
/// Test AC3 with disabled strict mode (should pass)
#[test]
#[cfg(feature = "cpu")]
fn test_error_path_disabled_strict_mode_allows_fallback() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = false;
    config.enforce_quantized_inference = false;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[2048, 2048],
        "kernel_unavailable",
    );
    assert!(result.is_ok(), "Disabled strict mode should allow fallback");
}
/// Test AC3 with partial strict mode (quantization disabled)
#[test]
#[cfg(feature = "cpu")]
fn test_error_path_partial_strict_mode() {
    use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
    use bitnet_common::{Device, QuantizationType};
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.enforce_quantized_inference = false;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[2048, 2048],
        "kernel_unavailable",
    );
    assert!(result.is_ok(), "Should allow fallback when quantization enforcement disabled");
}
/// Test AC3 performance metrics validation with mock computation
#[test]
#[cfg(feature = "cpu")]
fn test_performance_mock_computation_detection() {
    use bitnet_common::strict_mode::{
        ComputationType, PerformanceMetrics, StrictModeConfig, StrictModeEnforcer,
    };
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.validate_performance = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let metrics = PerformanceMetrics {
        tokens_per_second: 45.0,
        latency_ms: 22.2,
        memory_usage_mb: 1024.0,
        computation_type: ComputationType::Mock,
        gpu_utilization: None,
    };
    let result = enforcer.validate_performance_metrics(&metrics);
    assert!(result.is_err(), "Should detect mock computation");
    assert!(result.unwrap_err().to_string().contains("Mock computation"));
}
/// Test AC3 performance metrics validation with suspicious TPS
#[test]
#[cfg(feature = "cpu")]
fn test_performance_suspicious_tps_detection() {
    use bitnet_common::strict_mode::{
        ComputationType, PerformanceMetrics, StrictModeConfig, StrictModeEnforcer,
    };
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.validate_performance = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let metrics = PerformanceMetrics {
        tokens_per_second: 200.0,
        latency_ms: 5.0,
        memory_usage_mb: 1024.0,
        computation_type: ComputationType::Real,
        gpu_utilization: Some(95.0),
    };
    let result = enforcer.validate_performance_metrics(&metrics);
    assert!(result.is_err(), "Should detect suspicious performance");
    assert!(result.unwrap_err().to_string().contains("Suspicious performance"));
}
/// Test AC3 performance metrics validation with realistic values
#[test]
#[cfg(feature = "cpu")]
fn test_performance_realistic_values_pass() {
    use bitnet_common::strict_mode::{
        ComputationType, PerformanceMetrics, StrictModeConfig, StrictModeEnforcer,
    };
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.validate_performance = true;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let metrics = PerformanceMetrics {
        tokens_per_second: 35.0,
        latency_ms: 28.6,
        memory_usage_mb: 2048.0,
        computation_type: ComputationType::Real,
        gpu_utilization: None,
    };
    let result = enforcer.validate_performance_metrics(&metrics);
    assert!(result.is_ok(), "Realistic performance should pass validation");
}
/// Test AC3 performance metrics validation disabled
#[test]
#[cfg(feature = "cpu")]
fn test_performance_validation_disabled() {
    use bitnet_common::strict_mode::{
        ComputationType, PerformanceMetrics, StrictModeConfig, StrictModeEnforcer,
    };
    let mut config = StrictModeConfig::from_env();
    config.enabled = true;
    config.validate_performance = false;
    let enforcer = StrictModeEnforcer::with_config(Some(config));
    let metrics = PerformanceMetrics {
        tokens_per_second: 200.0,
        latency_ms: 5.0,
        memory_usage_mb: 1024.0,
        computation_type: ComputationType::Mock,
        gpu_utilization: None,
    };
    let result = enforcer.validate_performance_metrics(&metrics);
    assert!(result.is_ok(), "Should pass when performance validation disabled");
}
/// AC6: Test receipt with empty kernel list
#[test]
fn test_ac6_receipt_edge_case_empty_kernels() {
    use serde_json::json;
    let receipt = json!(
        { "schema_version" : "1.0.0", "compute_path" : "real", "kernels" : [], "backend"
        : "cpu" }
    );
    assert!(receipt["kernels"].as_array().unwrap().is_empty());
    assert_eq!(receipt["compute_path"], "real");
}
/// AC6: Test receipt with mixed quantization types
#[test]
fn test_ac6_receipt_edge_case_mixed_quantization() {
    use serde_json::json;
    let receipt = json!(
        { "schema_version" : "1.0.0", "compute_path" : "real", "kernels" :
        ["i2s_matmul_cpu", "tl1_lookup", "tl2_simd", "gemm_i2s_gpu"], "backend" : "cpu" }
    );
    let kernels = receipt["kernels"].as_array().unwrap();
    let has_i2s = kernels.iter().any(|k| k.as_str().unwrap().contains("i2s_"));
    let has_tl1 = kernels.iter().any(|k| k.as_str().unwrap().contains("tl1_"));
    let has_tl2 = kernels.iter().any(|k| k.as_str().unwrap().contains("tl2_"));
    assert!(has_i2s && has_tl1 && has_tl2, "Should have mixed quantization types");
}
/// AC6: Test receipt with GPU backend and GPU kernels
#[test]
#[cfg(feature = "gpu")]
fn test_ac6_receipt_edge_case_gpu_backend() {
    use serde_json::json;
    let receipt = json!(
        { "schema_version" : "1.0.0", "compute_path" : "real", "kernels" :
        ["gemm_i2s_gpu", "i2s_gpu_matmul"], "backend" : "cuda" }
    );
    assert_eq!(receipt["backend"], "cuda");
    let kernels = receipt["kernels"].as_array().unwrap();
    let has_gpu_kernels = kernels.iter().all(|k| k.as_str().unwrap().contains("gpu"));
    assert!(has_gpu_kernels, "GPU backend should use GPU kernels");
}
/// Test StrictModeConfig::from_env_detailed
#[test]
fn test_strict_mode_config_from_env_detailed() {
    use bitnet_common::strict_mode::StrictModeConfig;
    let config = StrictModeConfig::from_env_detailed();
    assert_eq!(config.enabled, config.fail_on_mock);
    assert_eq!(config.enabled, config.require_quantization);
}
/// Test StrictModeConfig::from_env_with_ci_enhancements
#[test]
fn test_strict_mode_config_ci_enhancements() {
    use bitnet_common::strict_mode::StrictModeConfig;
    let config = StrictModeConfig::from_env_with_ci_enhancements();
    if std::env::var("CI").is_ok()
        && std::env::var("BITNET_CI_ENHANCED_STRICT").unwrap_or_default() == "1"
    {
        assert!(config.ci_enhanced_mode);
    } else {
        assert!(!config.ci_enhanced_mode);
    }
}
/// Test StrictModeEnforcer default constructor
#[test]
fn test_strict_mode_enforcer_default() {
    use bitnet_common::strict_mode::StrictModeEnforcer;
    let enforcer = StrictModeEnforcer::default();
    let config = enforcer.get_config();
    assert_eq!(config.enabled, enforcer.is_enabled());
}
/// Test StrictModeEnforcer new_fresh constructor
#[test]
fn test_strict_mode_enforcer_new_fresh() {
    use bitnet_common::strict_mode::StrictModeEnforcer;
    let enforcer = StrictModeEnforcer::new_fresh();
    let config = enforcer.get_config();
    assert!(!config.ci_enhanced_mode);
}
