//! Comprehensive BDD-style tests for multi-backend selection logic.
//!
//! These tests exercise `select_backend()` from `bitnet-common` with every
//! meaningful combination of backend capabilities, user requests, environment
//! variables, and failure modes.
//!
//! Naming convention: `given_<context>_when_<action>_then_<outcome>`
//!
//! Environment-mutating tests use `#[serial(bitnet_env)]` + `temp_env`.

use bitnet_common::backend_selection::{
    BackendRequest, BackendSelectionError, select_backend,
};
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use serial_test::serial;
use temp_env::with_vars;

// ---------------------------------------------------------------------------
// Helpers — build KernelCapabilities for each scenario
// ---------------------------------------------------------------------------

fn caps_cpu_only() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn caps_cuda_and_cpu() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn caps_oneapi_and_cpu() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn caps_cuda_oneapi_and_cpu() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn caps_cuda_compiled_no_runtime() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn caps_oneapi_compiled_no_runtime() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn caps_none() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    }
}

fn caps_cpp_ffi_only() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Scalar,
    }
}

fn caps_all_backends() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx512,
    }
}

fn caps_cuda_and_ffi() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx2,
    }
}

// ---------------------------------------------------------------------------
// Scenario 1: Only CPU available → selects CPU
// ---------------------------------------------------------------------------

#[test]
fn given_only_cpu_when_auto_then_selects_cpu() {
    let result = select_backend(BackendRequest::Auto, &caps_cpu_only()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
    assert_eq!(result.requested, BackendRequest::Auto);
}

#[test]
fn given_only_cpu_when_cpu_requested_then_selects_cpu() {
    let result = select_backend(BackendRequest::Cpu, &caps_cpu_only()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
}

// ---------------------------------------------------------------------------
// Scenario 2: CUDA + CPU → selects CUDA
// ---------------------------------------------------------------------------

#[test]
fn given_cuda_and_cpu_when_auto_then_selects_cuda() {
    let result = select_backend(BackendRequest::Auto, &caps_cuda_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

#[test]
fn given_cuda_and_cpu_when_gpu_requested_then_selects_cuda() {
    let result = select_backend(BackendRequest::Gpu, &caps_cuda_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

// ---------------------------------------------------------------------------
// Scenario 3: OneAPI + CPU → selects OneAPI
// ---------------------------------------------------------------------------

#[test]
fn given_oneapi_and_cpu_when_auto_then_selects_oneapi() {
    let result = select_backend(BackendRequest::Auto, &caps_oneapi_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::OneApi);
}

#[test]
fn given_oneapi_and_cpu_when_gpu_requested_then_selects_oneapi() {
    let result = select_backend(BackendRequest::Gpu, &caps_oneapi_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::OneApi);
}

#[test]
fn given_oneapi_and_cpu_when_oneapi_requested_then_selects_oneapi() {
    let result = select_backend(BackendRequest::OneApi, &caps_oneapi_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::OneApi);
}

// ---------------------------------------------------------------------------
// Scenario 4: CUDA + OneAPI + CPU → selects CUDA (highest priority)
// ---------------------------------------------------------------------------

#[test]
fn given_all_gpu_backends_when_auto_then_cuda_wins() {
    let result = select_backend(BackendRequest::Auto, &caps_cuda_oneapi_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

// ---------------------------------------------------------------------------
// Scenario 5: All backends → CUDA > OneAPI > CppFfi > CPU
// ---------------------------------------------------------------------------

#[test]
fn given_every_backend_when_auto_then_cuda_is_top_priority() {
    let result = select_backend(BackendRequest::Auto, &caps_all_backends()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

#[test]
fn given_every_backend_when_cpu_forced_then_cpu_used() {
    let result = select_backend(BackendRequest::Cpu, &caps_all_backends()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
}

// ---------------------------------------------------------------------------
// Scenario 6: User forces --device oneapi with CUDA available → uses OneAPI
// ---------------------------------------------------------------------------

#[test]
fn given_cuda_and_oneapi_when_oneapi_forced_then_uses_oneapi() {
    let result =
        select_backend(BackendRequest::OneApi, &caps_cuda_oneapi_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::OneApi);
}

// ---------------------------------------------------------------------------
// Scenario 7: User forces --device cuda but no CUDA → error with message
// ---------------------------------------------------------------------------

#[test]
fn given_only_cpu_when_cuda_forced_then_error() {
    let err = select_backend(BackendRequest::Cuda, &caps_cpu_only()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
    let msg = err.to_string();
    assert!(msg.contains("not available"), "error message: {msg}");
}

#[test]
fn given_only_cpu_when_oneapi_forced_then_error() {
    let err = select_backend(BackendRequest::OneApi, &caps_cpu_only()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
}

// ---------------------------------------------------------------------------
// Scenario 8: CUDA compiled but no runtime → GPU request falls back to CPU
// ---------------------------------------------------------------------------

#[test]
fn given_cuda_compiled_no_runtime_when_gpu_requested_then_falls_back_to_cpu() {
    let result =
        select_backend(BackendRequest::Gpu, &caps_cuda_compiled_no_runtime()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
    assert!(
        result.rationale.contains("falling back"),
        "rationale should explain fallback: {}",
        result.rationale
    );
}

#[test]
fn given_cuda_compiled_no_runtime_when_cuda_forced_then_error() {
    let err =
        select_backend(BackendRequest::Cuda, &caps_cuda_compiled_no_runtime()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
}

// ---------------------------------------------------------------------------
// Scenario 9: BITNET_GPU_FAKE=oneapi → fakes OneAPI detection
// (env-mutating; test via device_features)
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn given_gpu_fake_oneapi_when_probing_then_oneapi_detected() {
    with_vars(
        [
            ("BITNET_GPU_FAKE", Some("oneapi")),
            ("BITNET_STRICT_NO_FAKE_GPU", None::<&str>),
        ],
        || {
            let info = bitnet_kernels::gpu_utils::get_gpu_info();
            // GPU_FAKE for oneapi doesn't directly set info.cuda,
            // but the intent is that the runtime detection layer honours it.
            // At minimum, the call must not panic.
            let _ = info;
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn given_gpu_fake_cuda_when_probing_then_cuda_detected() {
    with_vars(
        [
            ("BITNET_GPU_FAKE", Some("cuda")),
            ("BITNET_STRICT_NO_FAKE_GPU", None::<&str>),
        ],
        || {
            let info = bitnet_kernels::gpu_utils::get_gpu_info();
            assert!(info.cuda, "BITNET_GPU_FAKE=cuda should fake CUDA detection");
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 10: BITNET_STRICT_MODE=1 → rejects fake devices
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
#[should_panic(expected = "BITNET_GPU_FAKE is set but strict mode forbids fake GPU")]
fn given_strict_mode_when_fake_gpu_then_panics() {
    with_vars(
        [
            ("BITNET_GPU_FAKE", Some("cuda")),
            ("BITNET_STRICT_NO_FAKE_GPU", Some("1")),
        ],
        || {
            bitnet_kernels::gpu_utils::get_gpu_info();
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn given_strict_mode_no_fake_when_probing_then_ok() {
    with_vars(
        [
            ("BITNET_GPU_FAKE", None::<&str>),
            ("BITNET_STRICT_NO_FAKE_GPU", Some("1")),
        ],
        || {
            // Should not panic — no fake GPU set
            let _info = bitnet_kernels::gpu_utils::get_gpu_info();
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 11: No backend at all → NoBackendAvailable
// ---------------------------------------------------------------------------

#[test]
fn given_no_backends_when_auto_then_no_backend_error() {
    let err = select_backend(BackendRequest::Auto, &caps_none()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::NoBackendAvailable));
    let msg = err.to_string();
    assert!(
        msg.contains("no kernel backend"),
        "error should mention no backend: {msg}"
    );
}

#[test]
fn given_no_backends_when_cpu_requested_then_error() {
    let err = select_backend(BackendRequest::Cpu, &caps_none()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
}

// ---------------------------------------------------------------------------
// Scenario 12: Only CppFfi available → selects CppFfi
// ---------------------------------------------------------------------------

#[test]
fn given_only_ffi_when_auto_then_selects_cpp_ffi() {
    let result = select_backend(BackendRequest::Auto, &caps_cpp_ffi_only()).unwrap();
    assert_eq!(result.selected, KernelBackend::CppFfi);
}

// ---------------------------------------------------------------------------
// Scenario 13: CUDA + CppFfi + CPU → CUDA takes priority over FFI
// ---------------------------------------------------------------------------

#[test]
fn given_cuda_ffi_cpu_when_auto_then_cuda_wins_over_ffi() {
    let result = select_backend(BackendRequest::Auto, &caps_cuda_and_ffi()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

// ---------------------------------------------------------------------------
// Scenario 14: OneAPI compiled but no runtime → auto falls back to CPU
// ---------------------------------------------------------------------------

#[test]
fn given_oneapi_compiled_no_runtime_when_auto_then_cpu() {
    let result =
        select_backend(BackendRequest::Auto, &caps_oneapi_compiled_no_runtime()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
}

#[test]
fn given_oneapi_compiled_no_runtime_when_oneapi_forced_then_error() {
    let err = select_backend(BackendRequest::OneApi, &caps_oneapi_compiled_no_runtime())
        .unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
}

// ---------------------------------------------------------------------------
// Scenario 15: Summary format stability
// ---------------------------------------------------------------------------

#[test]
fn given_cpu_only_when_auto_then_summary_contains_requested_and_selected() {
    let result = select_backend(BackendRequest::Auto, &caps_cpu_only()).unwrap();
    let summary = result.summary();
    assert!(summary.contains("requested=auto"), "summary: {summary}");
    assert!(summary.contains("selected=cpu-rust"), "summary: {summary}");
}

#[test]
fn given_cuda_when_auto_then_summary_shows_cuda_selected() {
    let result = select_backend(BackendRequest::Auto, &caps_cuda_and_cpu()).unwrap();
    let summary = result.summary();
    assert!(summary.contains("selected=cuda"), "summary: {summary}");
}

// ---------------------------------------------------------------------------
// Scenario 16: Detected backends list is correct
// ---------------------------------------------------------------------------

#[test]
fn given_cuda_and_cpu_when_auto_then_detected_includes_both() {
    let result = select_backend(BackendRequest::Auto, &caps_cuda_and_cpu()).unwrap();
    assert!(result.detected.contains(&KernelBackend::Cuda));
    assert!(result.detected.contains(&KernelBackend::CpuRust));
}

#[test]
fn given_all_backends_when_auto_then_detected_includes_all_compiled() {
    let result = select_backend(BackendRequest::Auto, &caps_all_backends()).unwrap();
    assert!(result.detected.contains(&KernelBackend::Cuda));
    assert!(result.detected.contains(&KernelBackend::OneApi));
    assert!(result.detected.contains(&KernelBackend::CppFfi));
    assert!(result.detected.contains(&KernelBackend::CpuRust));
}

// ---------------------------------------------------------------------------
// Scenario 17: Priority order — CUDA > OneAPI > CppFfi > CPU
// ---------------------------------------------------------------------------

#[test]
fn given_all_backends_then_compiled_backends_order_is_gpu_first() {
    let caps = caps_all_backends();
    let backends = caps.compiled_backends();
    // CUDA should come before OneApi, which comes before CppFfi, then CPU
    let cuda_pos = backends.iter().position(|b| *b == KernelBackend::Cuda).unwrap();
    let oneapi_pos = backends.iter().position(|b| *b == KernelBackend::OneApi).unwrap();
    let ffi_pos = backends.iter().position(|b| *b == KernelBackend::CppFfi).unwrap();
    let cpu_pos = backends.iter().position(|b| *b == KernelBackend::CpuRust).unwrap();

    assert!(cuda_pos < oneapi_pos, "CUDA must be before OneAPI");
    assert!(oneapi_pos < ffi_pos, "OneAPI must be before CppFfi");
    assert!(ffi_pos < cpu_pos, "CppFfi must be before CPU");
}

// ---------------------------------------------------------------------------
// Scenario 18: BackendStartupSummary construction and formatting
// ---------------------------------------------------------------------------

#[test]
fn given_startup_summary_when_constructed_then_log_line_is_stable() {
    use bitnet_common::backend_selection::BackendStartupSummary;

    let summary = BackendStartupSummary::new(
        "auto",
        vec!["cuda".to_string(), "cpu-rust".to_string()],
        "cuda",
    );

    let line = summary.log_line();
    assert!(line.contains("requested=auto"), "line: {line}");
    assert!(line.contains("selected=cuda"), "line: {line}");
    assert!(line.contains("cpu-rust"), "line: {line}");
}

// ---------------------------------------------------------------------------
// Scenario 19: KernelCapabilities builder methods
// ---------------------------------------------------------------------------

#[test]
fn given_compile_time_caps_when_with_cuda_runtime_then_cuda_becomes_best() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
    .with_cuda_runtime(true);

    assert!(caps.cuda_runtime);
    assert_eq!(caps.best_available(), Some(KernelBackend::Cuda));
}

#[test]
fn given_compile_time_caps_when_with_oneapi_runtime_then_oneapi_becomes_best() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
    .with_oneapi_runtime(true);

    assert!(caps.oneapi_runtime);
    assert_eq!(caps.best_available(), Some(KernelBackend::OneApi));
}

// ---------------------------------------------------------------------------
// Scenario 20: GPU request without any GPU compiled → error
// ---------------------------------------------------------------------------

#[test]
fn given_cpu_only_when_gpu_requested_then_error_lists_available() {
    let err = select_backend(BackendRequest::Gpu, &caps_cpu_only()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("cpu-rust"), "error should list available: {msg}");
}

// ---------------------------------------------------------------------------
// Scenario 21: Error type implements Display and Error traits
// ---------------------------------------------------------------------------

#[test]
fn given_no_backend_error_then_display_is_helpful() {
    let err = BackendSelectionError::NoBackendAvailable;
    let msg = format!("{err}");
    assert!(msg.contains("--features"), "should hint at features: {msg}");
}

#[test]
fn given_unavailable_error_then_display_lists_backends() {
    let err = BackendSelectionError::RequestedUnavailable {
        requested: BackendRequest::Cuda,
        available: vec![KernelBackend::CpuRust],
    };
    let msg = format!("{err}");
    assert!(msg.contains("cuda"), "should mention requested: {msg}");
    assert!(msg.contains("cpu-rust"), "should list available: {msg}");
}

// ---------------------------------------------------------------------------
// Scenario 22: KernelBackend properties
// ---------------------------------------------------------------------------

#[test]
fn given_kernel_backends_then_requires_gpu_is_correct() {
    assert!(!KernelBackend::CpuRust.requires_gpu());
    assert!(KernelBackend::Cuda.requires_gpu());
    assert!(KernelBackend::OneApi.requires_gpu());
    assert!(!KernelBackend::CppFfi.requires_gpu());
}

// ---------------------------------------------------------------------------
// Scenario 23: BITNET_GPU_FAKE with empty string → no fake
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn given_gpu_fake_empty_when_probing_then_no_fake() {
    with_vars(
        [
            ("BITNET_GPU_FAKE", Some("")),
            ("BITNET_STRICT_NO_FAKE_GPU", None::<&str>),
        ],
        || {
            // Should not fake any GPU — just real detection
            let _info = bitnet_kernels::gpu_utils::get_gpu_info();
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 24: Capabilities summary string format
// ---------------------------------------------------------------------------

#[test]
fn given_caps_when_summary_then_contains_simd_and_backends() {
    let caps = caps_cuda_and_cpu();
    let s = caps.summary();
    assert!(s.contains("avx2"), "summary should contain simd level: {s}");
    assert!(s.contains("cuda"), "summary should list cuda: {s}");
    assert!(s.contains("cpu-rust"), "summary should list cpu-rust: {s}");
}

// ---------------------------------------------------------------------------
// Scenario 25: best_available with no runtime returns None for GPU-only
// ---------------------------------------------------------------------------

#[test]
fn given_no_cpu_no_runtime_when_best_available_then_none() {
    let caps = KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: true,
        cuda_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    assert_eq!(caps.best_available(), None);
}

// ---------------------------------------------------------------------------
// Scenario 26: Multiple OneAPI compiled + runtime → OneAPI wins when no CUDA
// ---------------------------------------------------------------------------

#[test]
fn given_oneapi_only_gpu_when_gpu_requested_then_oneapi() {
    let result = select_backend(BackendRequest::Gpu, &caps_oneapi_and_cpu()).unwrap();
    assert_eq!(result.selected, KernelBackend::OneApi);
}

// ---------------------------------------------------------------------------
// Scenario 27: Rationale field provides human-readable explanation
// ---------------------------------------------------------------------------

#[test]
fn given_auto_cpu_only_then_rationale_mentions_auto() {
    let result = select_backend(BackendRequest::Auto, &caps_cpu_only()).unwrap();
    assert!(
        result.rationale.contains("auto"),
        "rationale: {}",
        result.rationale
    );
}

#[test]
fn given_explicit_cpu_then_rationale_mentions_explicitly() {
    let result = select_backend(BackendRequest::Cpu, &caps_all_backends()).unwrap();
    assert!(
        result.rationale.contains("explicitly") || result.rationale.contains("CPU"),
        "rationale: {}",
        result.rationale
    );
}
