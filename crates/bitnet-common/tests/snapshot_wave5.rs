//! Wave 5 snapshot tests for `bitnet-common` public API surface.
//!
//! Pins Debug/Display formats and serialization of core types so that
//! unintentional changes are caught at review time.

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_common::types::{Device, ModelMetadata, PerformanceMetrics, QuantizationType};
use bitnet_common::{BackendRequest, BackendStartupSummary};

// ---------------------------------------------------------------------------
// Device Debug
// ---------------------------------------------------------------------------

#[test]
fn device_all_variants_debug() {
    let devices = [
        Device::Cpu,
        Device::Cuda(0),
        Device::Hip(1),
        Device::Npu,
        Device::Metal,
        Device::OpenCL(2),
    ];
    let output: Vec<String> = devices.iter().map(|d| format!("{d:?}")).collect();
    insta::assert_debug_snapshot!(output);
}

#[test]
fn device_default_is_cpu() {
    let dev = Device::default();
    insta::assert_snapshot!(format!("{dev:?}"));
}

// ---------------------------------------------------------------------------
// QuantizationType Display
// ---------------------------------------------------------------------------

#[test]
fn quantization_type_display_all_variants() {
    let qtypes = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let displays: Vec<String> = qtypes.iter().map(|q| q.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn quantization_type_debug_all_variants() {
    let qtypes = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let debugs: Vec<String> = qtypes.iter().map(|q| format!("{q:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

// ---------------------------------------------------------------------------
// KernelCapabilities summary (with SIMD filter)
// ---------------------------------------------------------------------------

#[test]
fn kernel_capabilities_summary_cpu_only() {
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
    insta::with_settings!({filters => vec![(r"simd=\w+", "simd=[SIMD]")]}, {
        insta::assert_snapshot!(caps.summary());
    });
}

#[test]
fn kernel_capabilities_summary_full_stack() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx512,
    };
    insta::with_settings!({filters => vec![(r"simd=\w+", "simd=[SIMD]")]}, {
        insta::assert_snapshot!(caps.summary());
    });
}

#[test]
fn kernel_capabilities_compiled_backends_priority_order() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        hip_compiled: true,
        hip_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Scalar,
    };
    let backends: Vec<String> = caps.compiled_backends().iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!(backends);
}

// ---------------------------------------------------------------------------
// BackendStartupSummary
// ---------------------------------------------------------------------------

#[test]
fn backend_startup_summary_auto_cpu() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".to_string()], "cpu-rust");
    insta::assert_snapshot!(summary.log_line());
}

#[test]
fn backend_startup_summary_auto_gpu() {
    let summary = BackendStartupSummary::new(
        "auto",
        vec!["cuda".to_string(), "cpu-rust".to_string()],
        "cuda",
    );
    insta::assert_snapshot!(summary.log_line());
}

// ---------------------------------------------------------------------------
// BackendRequest Display
// ---------------------------------------------------------------------------

#[test]
fn backend_request_display_all_variants() {
    let requests = [
        BackendRequest::Auto,
        BackendRequest::Cpu,
        BackendRequest::Gpu,
        BackendRequest::Cuda,
        BackendRequest::Hip,
        BackendRequest::OneApi,
    ];
    let displays: Vec<String> = requests.iter().map(|r| r.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// ---------------------------------------------------------------------------
// ModelMetadata default Debug
// ---------------------------------------------------------------------------

#[test]
fn model_metadata_debug_snapshot() {
    let meta = ModelMetadata {
        name: "test-model".to_string(),
        version: "1.0.0".to_string(),
        architecture: "bitnet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::I2S),
        fingerprint: None,
        corrections_applied: None,
    };
    insta::assert_debug_snapshot!(meta);
}

// ---------------------------------------------------------------------------
// PerformanceMetrics default Debug
// ---------------------------------------------------------------------------

#[test]
fn performance_metrics_default_debug() {
    let metrics = PerformanceMetrics::default();
    insta::assert_debug_snapshot!(metrics);
}

// ---------------------------------------------------------------------------
// KernelBackend Display — all variants including Hip/OneApi
// ---------------------------------------------------------------------------

#[test]
fn kernel_backend_display_all_five_variants() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::CppFfi,
    ];
    let displays: Vec<String> = backends.iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}
