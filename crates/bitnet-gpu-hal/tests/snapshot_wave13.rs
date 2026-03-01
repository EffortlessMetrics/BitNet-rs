//! Snapshot wave 13 — stabilize bitnet-gpu-hal public API surface.
//!
//! Pins Debug and Display representations of core HAL types so that
//! accidental formatting or variant changes are caught by CI.

use bitnet_gpu_hal::HalError;
use bitnet_gpu_hal::hal_traits::{AllocatorStats, ComputeCapabilities, MemoryType};

// ── HalError Display ─────────────────────────────────────────────────────────

#[test]
fn hal_error_device_not_found_display() {
    let err = HalError::DeviceNotFound("cuda:0".into());
    insta::assert_snapshot!("hal_error_device_not_found", err.to_string());
}

#[test]
fn hal_error_out_of_memory_display() {
    let err = HalError::OutOfMemory { requested: 1_073_741_824, available: 536_870_912 };
    insta::assert_snapshot!("hal_error_out_of_memory", err.to_string());
}

#[test]
fn hal_error_compilation_failed_display() {
    let err = HalError::CompilationFailed("syntax error at line 42".into());
    insta::assert_snapshot!("hal_error_compilation_failed", err.to_string());
}

#[test]
fn hal_error_kernel_launch_failed_display() {
    let err = HalError::KernelLaunchFailed("invalid grid dimensions".into());
    insta::assert_snapshot!("hal_error_kernel_launch_failed", err.to_string());
}

#[test]
fn hal_error_invalid_argument_display() {
    let err = HalError::InvalidArgument { index: 3, reason: "expected f32 buffer".into() };
    insta::assert_snapshot!("hal_error_invalid_argument", err.to_string());
}

#[test]
fn hal_error_timeout_display() {
    let err = HalError::Timeout { operation: "matmul_kernel".into(), elapsed_ms: 5000 };
    insta::assert_snapshot!("hal_error_timeout", err.to_string());
}

#[test]
fn hal_error_backend_error_display() {
    let err = HalError::BackendError { backend: "cuda".into(), message: "driver mismatch".into() };
    insta::assert_snapshot!("hal_error_backend_error", err.to_string());
}

#[test]
fn hal_error_all_variants_debug() {
    let variants: Vec<HalError> = vec![
        HalError::DeviceNotFound("gpu:0".into()),
        HalError::OutOfMemory { requested: 1024, available: 512 },
        HalError::CompilationFailed("error".into()),
        HalError::KernelLaunchFailed("failed".into()),
        HalError::InvalidArgument { index: 0, reason: "bad type".into() },
        HalError::BufferAccessError("out of bounds".into()),
        HalError::QueueError("stalled".into()),
        HalError::Timeout { operation: "op".into(), elapsed_ms: 100 },
        HalError::Unsupported("fp64".into()),
        HalError::BackendError { backend: "vulkan".into(), message: "init failed".into() },
    ];
    insta::assert_debug_snapshot!("hal_error_all_variants", variants);
}

// ── MemoryType ───────────────────────────────────────────────────────────────

#[test]
fn memory_type_all_variants_debug() {
    let variants = [MemoryType::Device, MemoryType::Shared, MemoryType::Pinned];
    insta::assert_debug_snapshot!("memory_type_variants", variants);
}

// ── ComputeCapabilities ──────────────────────────────────────────────────────

#[test]
fn compute_capabilities_debug() {
    let caps = ComputeCapabilities {
        max_workgroup_size: [1024, 1024, 64],
        max_grid_size: [2_147_483_647, 65535, 65535],
        max_shared_memory_bytes: 49152,
        compute_units: 80,
        supports_fp16: true,
        supports_int8: true,
        supports_subgroups: true,
    };
    insta::assert_debug_snapshot!("compute_capabilities_typical_gpu", caps);
}

#[test]
fn compute_capabilities_minimal_device() {
    let caps = ComputeCapabilities {
        max_workgroup_size: [256, 1, 1],
        max_grid_size: [1024, 1, 1],
        max_shared_memory_bytes: 16384,
        compute_units: 1,
        supports_fp16: false,
        supports_int8: false,
        supports_subgroups: false,
    };
    insta::assert_debug_snapshot!("compute_capabilities_minimal", caps);
}

// ── AllocatorStats ───────────────────────────────────────────────────────────

#[test]
fn allocator_stats_debug() {
    let stats = AllocatorStats {
        total_allocated: 268_435_456,
        allocation_count: 42,
        peak_allocated: 536_870_912,
    };
    insta::assert_debug_snapshot!("allocator_stats_active", stats);
}

#[test]
fn allocator_stats_empty() {
    let stats = AllocatorStats { total_allocated: 0, allocation_count: 0, peak_allocated: 0 };
    insta::assert_debug_snapshot!("allocator_stats_empty", stats);
}
