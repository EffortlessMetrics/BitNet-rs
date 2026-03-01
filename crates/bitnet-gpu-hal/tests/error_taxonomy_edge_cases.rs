//! Edge-case tests for GPU HAL error taxonomy.
//!
//! Tests all error type construction, Display formatting, From conversions,
//! error source chaining, context attachment, transient classification,
//! and recovery strategies — all without GPU hardware.

use bitnet_gpu_hal::error_taxonomy::*;
use std::error::Error;

// ── DeviceError Display ─────────────────────────────────────────────────────

#[test]
fn device_error_not_found_display() {
    let e = DeviceError::NotFound { query: "A770".into() };
    assert!(e.to_string().contains("A770"));
    assert!(e.to_string().contains("no compatible device"));
}

#[test]
fn device_error_unavailable_display() {
    let e = DeviceError::Unavailable { device_id: 0, reason: "busy".into() };
    let s = e.to_string();
    assert!(s.contains("0"));
    assert!(s.contains("busy"));
}

#[test]
fn device_error_driver_mismatch_display() {
    let e = DeviceError::DriverError { expected: "535.0".into(), found: "530.0".into() };
    let s = e.to_string();
    assert!(s.contains("535.0"));
    assert!(s.contains("530.0"));
}

#[test]
fn device_error_capability_missing_display() {
    let e = DeviceError::CapabilityMissing { device_id: 1, capability: "fp16".into() };
    assert!(e.to_string().contains("fp16"));
}

#[test]
fn device_error_init_failed_display() {
    let e = DeviceError::InitFailed { device_id: 2, message: "timeout".into() };
    assert!(e.to_string().contains("init failed"));
}

#[test]
fn device_error_device_lost_display() {
    let e = DeviceError::DeviceLost { device_id: 3 };
    assert!(e.to_string().contains("3"));
    assert!(e.to_string().contains("lost"));
}

#[test]
fn device_error_clone_eq() {
    let e1 = DeviceError::NotFound { query: "test".into() };
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}

#[test]
fn device_error_implements_std_error() {
    let e = DeviceError::NotFound { query: "gpu".into() };
    let _: &dyn Error = &e;
}

// ── MemoryError Display ─────────────────────────────────────────────────────

#[test]
fn memory_error_oom_display() {
    let e = MemoryError::OutOfMemory { requested: 1_000_000, available: 500_000 };
    let s = e.to_string();
    assert!(s.contains("1000000"));
    assert!(s.contains("500000"));
}

#[test]
fn memory_error_out_of_bounds_display() {
    let e = MemoryError::OutOfBounds { offset: 100, length: 200, buffer_size: 150 };
    let s = e.to_string();
    assert!(s.contains("100"));
    assert!(s.contains("200"));
    assert!(s.contains("150"));
}

#[test]
fn memory_error_corruption_display() {
    let e = MemoryError::Corruption { buffer_id: 42, details: "checksum mismatch".into() };
    assert!(e.to_string().contains("42"));
    assert!(e.to_string().contains("checksum"));
}

#[test]
fn memory_error_fragmentation_display() {
    let e = MemoryError::Fragmentation { requested: 1024, total_free: 2048, largest_block: 512 };
    let s = e.to_string();
    assert!(s.contains("1024"));
    assert!(s.contains("2048"));
    assert!(s.contains("512"));
}

#[test]
fn memory_error_invalid_buffer_display() {
    let e = MemoryError::InvalidBuffer { buffer_id: 99 };
    assert!(e.to_string().contains("99"));
}

#[test]
fn memory_error_mapping_failed_display() {
    let e = MemoryError::MappingFailed { reason: "permissions".into() };
    assert!(e.to_string().contains("permissions"));
}

#[test]
fn memory_error_alignment_display() {
    let e = MemoryError::AlignmentError { required: 16, actual: 8 };
    let s = e.to_string();
    assert!(s.contains("16"));
    assert!(s.contains("8"));
}

// ── KernelError Display ─────────────────────────────────────────────────────

#[test]
fn kernel_error_compilation_failed() {
    let e =
        KernelError::CompilationFailed { kernel_name: "matmul".into(), log: "syntax error".into() };
    let s = e.to_string();
    assert!(s.contains("matmul"));
    assert!(s.contains("syntax error"));
}

#[test]
fn kernel_error_launch_config_invalid() {
    let e = KernelError::LaunchConfigInvalid {
        kernel_name: "softmax".into(),
        reason: "workgroup too large".into(),
    };
    assert!(e.to_string().contains("softmax"));
}

#[test]
fn kernel_error_timeout() {
    let e =
        KernelError::Timeout { kernel_name: "attention".into(), elapsed_ms: 5000, limit_ms: 3000 };
    let s = e.to_string();
    assert!(s.contains("5000"));
    assert!(s.contains("3000"));
}

#[test]
fn kernel_error_invalid_argument() {
    let e = KernelError::InvalidArgument {
        kernel_name: "rope".into(),
        index: 3,
        reason: "null pointer".into(),
    };
    assert!(e.to_string().contains("3"));
}

#[test]
fn kernel_error_not_found() {
    let e = KernelError::NotFound { kernel_name: "missing_kernel".into() };
    assert!(e.to_string().contains("missing_kernel"));
}

#[test]
fn kernel_error_execution_failed() {
    let e = KernelError::ExecutionFailed { kernel_name: "silu".into(), error_code: -1 };
    let s = e.to_string();
    assert!(s.contains("silu"));
    assert!(s.contains("-1"));
}

// ── TransferDirection ───────────────────────────────────────────────────────

#[test]
fn transfer_direction_display() {
    assert_eq!(TransferDirection::HostToDevice.to_string(), "host→device");
    assert_eq!(TransferDirection::DeviceToHost.to_string(), "device→host");
    assert_eq!(TransferDirection::DeviceToDevice.to_string(), "device→device");
}

#[test]
fn transfer_direction_clone_eq() {
    let d = TransferDirection::HostToDevice;
    assert_eq!(d, d.clone());
}

// ── TransferError Display ───────────────────────────────────────────────────

#[test]
fn transfer_error_failed() {
    let e = TransferError::Failed {
        direction: TransferDirection::HostToDevice,
        size_bytes: 4096,
        reason: "bus error".into(),
    };
    let s = e.to_string();
    assert!(s.contains("host→device"));
    assert!(s.contains("4096"));
}

#[test]
fn transfer_error_timeout() {
    let e =
        TransferError::Timeout { direction: TransferDirection::DeviceToHost, elapsed_ms: 10_000 };
    assert!(e.to_string().contains("10000"));
}

#[test]
fn transfer_error_invalid_buffer() {
    let e = TransferError::InvalidBuffer {
        direction: TransferDirection::DeviceToDevice,
        buffer_id: 77,
    };
    assert!(e.to_string().contains("77"));
}

#[test]
fn transfer_error_size_mismatch() {
    let e = TransferError::SizeMismatch { source_size: 100, dest_size: 200 };
    let s = e.to_string();
    assert!(s.contains("100"));
    assert!(s.contains("200"));
}

#[test]
fn transfer_error_dma() {
    let e = TransferError::DmaError { channel: 2, message: "underrun".into() };
    assert!(e.to_string().contains("2"));
    assert!(e.to_string().contains("underrun"));
}

// ── BackendKind Display ─────────────────────────────────────────────────────

#[test]
fn backend_kind_display_all_variants() {
    assert_eq!(BackendKind::CUDA.to_string(), "CUDA");
    assert_eq!(BackendKind::OpenCL.to_string(), "OpenCL");
    assert_eq!(BackendKind::Vulkan.to_string(), "Vulkan");
    assert_eq!(BackendKind::Metal.to_string(), "Metal");
    assert_eq!(BackendKind::ROCm.to_string(), "ROCm");
    assert_eq!(BackendKind::LevelZero.to_string(), "LevelZero");
    assert_eq!(BackendKind::WebGPU.to_string(), "WebGPU");
    assert_eq!(BackendKind::Other.to_string(), "Other");
}

#[test]
fn backend_kind_clone_eq_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(BackendKind::CUDA);
    set.insert(BackendKind::OpenCL);
    set.insert(BackendKind::CUDA);
    assert_eq!(set.len(), 2);
}

// ── BackendError ────────────────────────────────────────────────────────────

#[test]
fn backend_error_new() {
    let e = BackendError::new(BackendKind::CUDA, 700, "driver crash");
    assert_eq!(e.kind, BackendKind::CUDA);
    assert_eq!(e.native_code, 700);
    assert_eq!(e.message, "driver crash");
    assert!(e.api_call.is_none());
}

#[test]
fn backend_error_with_api_call() {
    let e = BackendError::new(BackendKind::OpenCL, -1, "invalid context")
        .with_api_call("clCreateBuffer");
    assert_eq!(e.api_call.as_deref(), Some("clCreateBuffer"));
}

#[test]
fn backend_error_display_with_api_call() {
    let e = BackendError::new(BackendKind::Metal, 42, "pipeline fail")
        .with_api_call("newComputePipelineState");
    let s = e.to_string();
    assert!(s.contains("[Metal]"));
    assert!(s.contains("newComputePipelineState"));
    assert!(s.contains("42"));
}

#[test]
fn backend_error_display_without_api_call() {
    let e = BackendError::new(BackendKind::Vulkan, 0, "ok");
    let s = e.to_string();
    assert!(s.contains("[Vulkan]"));
    assert!(!s.contains(":"));
}

// ── GpuHalError ─────────────────────────────────────────────────────────────

#[test]
fn gpu_hal_error_category_device() {
    let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
    assert_eq!(e.category(), "device");
}

#[test]
fn gpu_hal_error_category_memory() {
    let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 });
    assert_eq!(e.category(), "memory");
}

#[test]
fn gpu_hal_error_category_kernel() {
    let e = GpuHalError::Kernel(KernelError::NotFound { kernel_name: "x".into() });
    assert_eq!(e.category(), "kernel");
}

#[test]
fn gpu_hal_error_category_transfer() {
    let e = GpuHalError::Transfer(TransferError::SizeMismatch { source_size: 1, dest_size: 2 });
    assert_eq!(e.category(), "transfer");
}

#[test]
fn gpu_hal_error_category_backend() {
    let e = GpuHalError::Backend(BackendError::new(BackendKind::Other, 0, ""));
    assert_eq!(e.category(), "backend");
}

#[test]
fn gpu_hal_error_category_other() {
    let e = GpuHalError::Other("misc".into());
    assert_eq!(e.category(), "other");
}

#[test]
fn gpu_hal_error_is_transient_oom() {
    let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 });
    assert!(e.is_transient());
}

#[test]
fn gpu_hal_error_is_transient_kernel_timeout() {
    let e = GpuHalError::Kernel(KernelError::Timeout {
        kernel_name: "k".into(),
        elapsed_ms: 100,
        limit_ms: 50,
    });
    assert!(e.is_transient());
}

#[test]
fn gpu_hal_error_is_transient_transfer_timeout() {
    let e = GpuHalError::Transfer(TransferError::Timeout {
        direction: TransferDirection::HostToDevice,
        elapsed_ms: 100,
    });
    assert!(e.is_transient());
}

#[test]
fn gpu_hal_error_is_transient_device_unavailable() {
    let e = GpuHalError::Device(DeviceError::Unavailable { device_id: 0, reason: "busy".into() });
    assert!(e.is_transient());
}

#[test]
fn gpu_hal_error_not_transient_device_lost() {
    let e = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
    assert!(!e.is_transient());
}

#[test]
fn gpu_hal_error_not_transient_compilation() {
    let e = GpuHalError::Kernel(KernelError::CompilationFailed {
        kernel_name: "k".into(),
        log: "".into(),
    });
    assert!(!e.is_transient());
}

#[test]
fn gpu_hal_error_not_transient_other() {
    let e = GpuHalError::Other("something".into());
    assert!(!e.is_transient());
}

// ── From conversions ────────────────────────────────────────────────────────

#[test]
fn from_device_error() {
    let de = DeviceError::DeviceLost { device_id: 1 };
    let e: GpuHalError = de.into();
    assert_eq!(e.category(), "device");
}

#[test]
fn from_memory_error() {
    let me = MemoryError::InvalidBuffer { buffer_id: 5 };
    let e: GpuHalError = me.into();
    assert_eq!(e.category(), "memory");
}

#[test]
fn from_kernel_error() {
    let ke = KernelError::NotFound { kernel_name: "x".into() };
    let e: GpuHalError = ke.into();
    assert_eq!(e.category(), "kernel");
}

#[test]
fn from_transfer_error() {
    let te = TransferError::SizeMismatch { source_size: 1, dest_size: 2 };
    let e: GpuHalError = te.into();
    assert_eq!(e.category(), "transfer");
}

#[test]
fn from_backend_error() {
    let be = BackendError::new(BackendKind::CUDA, -1, "fail");
    let e: GpuHalError = be.into();
    assert_eq!(e.category(), "backend");
}

// ── Error source chaining ───────────────────────────────────────────────────

#[test]
fn error_source_device() {
    let e = GpuHalError::Device(DeviceError::NotFound { query: "q".into() });
    assert!(e.source().is_some());
}

#[test]
fn error_source_other_is_none() {
    let e = GpuHalError::Other("misc".into());
    assert!(e.source().is_none());
}

// ── ErrorContext ─────────────────────────────────────────────────────────────

#[test]
fn error_context_new() {
    let ctx = ErrorContext::new("matmul");
    assert_eq!(ctx.operation, "matmul");
    assert!(ctx.device_id.is_none());
    assert!(ctx.backend.is_none());
    assert!(ctx.metadata.is_empty());
    assert!(ctx.location.is_none());
    assert!(ctx.timestamp_ms > 0);
}

#[test]
fn error_context_builder_chain() {
    let ctx = ErrorContext::new("inference")
        .with_device(0)
        .with_backend(BackendKind::CUDA)
        .with_metadata("layer", "12")
        .with_location("engine.rs:42");
    assert_eq!(ctx.device_id, Some(0));
    assert_eq!(ctx.backend, Some(BackendKind::CUDA));
    assert_eq!(ctx.metadata.get("layer").unwrap(), "12");
    assert_eq!(ctx.location.as_deref(), Some("engine.rs:42"));
}

#[test]
fn error_context_display_minimal() {
    let ctx = ErrorContext::new("test_op");
    let s = ctx.to_string();
    assert!(s.contains("op=test_op"));
}

#[test]
fn error_context_display_full() {
    let ctx = ErrorContext::new("fwd")
        .with_device(1)
        .with_backend(BackendKind::OpenCL)
        .with_location("hal.rs:10");
    let s = ctx.to_string();
    assert!(s.contains("device=1"));
    assert!(s.contains("backend=OpenCL"));
    assert!(s.contains("at hal.rs:10"));
}

#[test]
fn error_context_display_with_metadata() {
    let ctx = ErrorContext::new("op").with_metadata("key", "val");
    let s = ctx.to_string();
    assert!(s.contains("key=val"));
}

// ── GpuHalError with_context ────────────────────────────────────────────────

#[test]
fn gpu_hal_error_with_context() {
    let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1024, available: 0 });
    let ctx = ErrorContext::new("alloc").with_device(0);
    let ce = e.with_context(ctx);
    assert_eq!(ce.category(), "memory");
    let s = ce.to_string();
    assert!(s.contains("context"));
    assert!(s.contains("alloc"));
}

#[test]
fn gpu_hal_error_contextualized_source_chains() {
    let inner = GpuHalError::Device(DeviceError::DeviceLost { device_id: 0 });
    let ctx = ErrorContext::new("probe");
    let e = inner.with_context(ctx);
    // source should be the inner GpuHalError
    assert!(e.source().is_some());
}

// ── Display formatting smoke tests ──────────────────────────────────────────

#[test]
fn gpu_hal_error_display_device() {
    let e = GpuHalError::Device(DeviceError::NotFound { query: "x".into() });
    assert!(e.to_string().starts_with("device error:"));
}

#[test]
fn gpu_hal_error_display_memory() {
    let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1, available: 0 });
    assert!(e.to_string().starts_with("memory error:"));
}

#[test]
fn gpu_hal_error_display_kernel() {
    let e = GpuHalError::Kernel(KernelError::NotFound { kernel_name: "x".into() });
    assert!(e.to_string().starts_with("kernel error:"));
}

#[test]
fn gpu_hal_error_display_transfer() {
    let e = GpuHalError::Transfer(TransferError::SizeMismatch { source_size: 1, dest_size: 2 });
    assert!(e.to_string().starts_with("transfer error:"));
}

#[test]
fn gpu_hal_error_display_backend() {
    let e = GpuHalError::Backend(BackendError::new(BackendKind::CUDA, 0, "ok"));
    assert!(e.to_string().starts_with("backend error:"));
}

#[test]
fn gpu_hal_error_display_other() {
    let e = GpuHalError::Other("catch-all".into());
    assert!(e.to_string().contains("catch-all"));
}

// ── ErrorRecovery (if available) ────────────────────────────────────────────

#[test]
fn error_recovery_suggest_for_oom() {
    let e = GpuHalError::Memory(MemoryError::OutOfMemory { requested: 1_000_000, available: 0 });
    let recovery = ErrorRecovery {
        max_retries: 3,
        retry_delay: std::time::Duration::from_secs(1),
        fallback_backend: None,
    };
    let action = recovery.suggest(&e);
    // OOM should suggest a recovery action
    let _ = action;
}

#[test]
fn error_recovery_default_fields() {
    let r = ErrorRecovery {
        max_retries: 3,
        retry_delay: std::time::Duration::from_secs(1),
        fallback_backend: None,
    };
    assert_eq!(r.max_retries, 3);
    assert!(r.fallback_backend.is_none());
}

// ── ErrorSeverity ───────────────────────────────────────────────────────────

#[test]
fn error_severity_all_variants_exist() {
    let _info = ErrorSeverity::Info;
    let _warn = ErrorSeverity::Warning;
    let _err = ErrorSeverity::Error;
    let _crit = ErrorSeverity::Critical;
}

// ── ErrorReport ─────────────────────────────────────────────────────────────

#[test]
fn error_report_construction() {
    let report = ErrorReport {
        id: 1,
        severity: ErrorSeverity::Error,
        category: "memory".into(),
        message: "OOM".into(),
        context: None,
        recovery: None,
        timestamp_ms: 12345,
    };
    assert_eq!(report.id, 1);
    assert_eq!(report.category, "memory");
}

#[test]
fn error_report_with_context() {
    let ctx = ErrorContext::new("test");
    let report = ErrorReport {
        id: 2,
        severity: ErrorSeverity::Warning,
        category: "kernel".into(),
        message: "slow".into(),
        context: Some(ctx),
        recovery: Some("retry".into()),
        timestamp_ms: 0,
    };
    assert!(report.context.is_some());
    assert_eq!(report.recovery.as_deref(), Some("retry"));
}

// ── RecoveryAction ──────────────────────────────────────────────────────────

#[test]
fn recovery_action_all_variants() {
    let _ = RecoveryAction::Retry { max_attempts: 3, delay: std::time::Duration::from_millis(100) };
    let _ = RecoveryAction::Fallback { target: "cpu".into() };
    let _ = RecoveryAction::Abort { reason: "fatal".into() };
    let _ = RecoveryAction::Degrade { description: "reduce batch".into() };
    let _ = RecoveryAction::ResetAndRetry { device_id: 0 };
}
