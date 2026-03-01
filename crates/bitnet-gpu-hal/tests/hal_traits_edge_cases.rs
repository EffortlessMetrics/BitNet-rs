//! Edge-case tests for GPU HAL traits and public types.
//!
//! Tests HalError construction, Display formatting, MemoryType, ComputeCapabilities,
//! ProgramSource, and AllocatorStats — all without GPU hardware.

use bitnet_gpu_hal::hal_traits::*;
use std::error::Error;

// ── HalError construction and Display ───────────────────────────────────────

#[test]
fn hal_error_device_not_found() {
    let e = HalError::DeviceNotFound("no GPU".into());
    assert!(e.to_string().contains("no GPU"));
    assert!(e.to_string().contains("device not found"));
}

#[test]
fn hal_error_out_of_memory() {
    let e = HalError::OutOfMemory { requested: 8_000_000_000, available: 4_000_000_000 };
    let s = e.to_string();
    assert!(s.contains("8000000000"));
    assert!(s.contains("4000000000"));
}

#[test]
fn hal_error_compilation_failed() {
    let e = HalError::CompilationFailed("syntax error line 42".into());
    assert!(e.to_string().contains("syntax error"));
}

#[test]
fn hal_error_kernel_launch_failed() {
    let e = HalError::KernelLaunchFailed("grid too large".into());
    assert!(e.to_string().contains("grid too large"));
}

#[test]
fn hal_error_invalid_argument() {
    let e = HalError::InvalidArgument { index: 5, reason: "null pointer".into() };
    let s = e.to_string();
    assert!(s.contains("5"));
    assert!(s.contains("null pointer"));
}

#[test]
fn hal_error_buffer_access_error() {
    let e = HalError::BufferAccessError("unmapped buffer".into());
    assert!(e.to_string().contains("unmapped"));
}

#[test]
fn hal_error_queue_error() {
    let e = HalError::QueueError("command buffer overflow".into());
    assert!(e.to_string().contains("command buffer overflow"));
}

#[test]
fn hal_error_timeout() {
    let e = HalError::Timeout { operation: "matmul".into(), elapsed_ms: 5000 };
    let s = e.to_string();
    assert!(s.contains("matmul"));
    assert!(s.contains("5000"));
}

#[test]
fn hal_error_unsupported() {
    let e = HalError::Unsupported("int4 compute".into());
    assert!(e.to_string().contains("int4"));
}

#[test]
fn hal_error_backend_error() {
    let e = HalError::BackendError { backend: "CUDA".into(), message: "driver crash".into() };
    let s = e.to_string();
    assert!(s.contains("[CUDA]"));
    assert!(s.contains("driver crash"));
}

// ── HalError trait impls ────────────────────────────────────────────────────

#[test]
fn hal_error_implements_std_error() {
    let e = HalError::DeviceNotFound("x".into());
    let _: &dyn Error = &e;
}

#[test]
fn hal_error_clone() {
    let e = HalError::OutOfMemory { requested: 1, available: 0 };
    let e2 = e.clone();
    assert_eq!(e, e2);
}

#[test]
fn hal_error_eq() {
    let e1 = HalError::Unsupported("fp8".into());
    let e2 = HalError::Unsupported("fp8".into());
    assert_eq!(e1, e2);
}

#[test]
fn hal_error_ne() {
    let e1 = HalError::Unsupported("fp8".into());
    let e2 = HalError::Unsupported("int4".into());
    assert_ne!(e1, e2);
}

#[test]
fn hal_error_debug() {
    let e = HalError::DeviceNotFound("test".into());
    let s = format!("{e:?}");
    assert!(s.contains("DeviceNotFound"));
}

// ── HalResult ───────────────────────────────────────────────────────────────

#[test]
fn hal_result_ok() {
    let r: HalResult<i32> = Ok(42);
    assert_eq!(r.unwrap(), 42);
}

#[test]
fn hal_result_err() {
    let r: HalResult<i32> = Err(HalError::DeviceNotFound("x".into()));
    assert!(r.is_err());
}

// ── MemoryType ──────────────────────────────────────────────────────────────

#[test]
fn memory_type_device() {
    let mt = MemoryType::Device;
    assert_eq!(mt, MemoryType::Device);
}

#[test]
fn memory_type_shared() {
    let mt = MemoryType::Shared;
    assert_eq!(mt, MemoryType::Shared);
}

#[test]
fn memory_type_pinned() {
    let mt = MemoryType::Pinned;
    assert_eq!(mt, MemoryType::Pinned);
}

#[test]
fn memory_type_ne() {
    assert_ne!(MemoryType::Device, MemoryType::Shared);
    assert_ne!(MemoryType::Shared, MemoryType::Pinned);
    assert_ne!(MemoryType::Pinned, MemoryType::Device);
}

#[test]
fn memory_type_clone_copy() {
    let mt = MemoryType::Device;
    let mt2 = mt;
    let mt3 = mt.clone();
    assert_eq!(mt, mt2);
    assert_eq!(mt, mt3);
}

#[test]
fn memory_type_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(MemoryType::Device);
    set.insert(MemoryType::Shared);
    set.insert(MemoryType::Pinned);
    set.insert(MemoryType::Device); // duplicate
    assert_eq!(set.len(), 3);
}

#[test]
fn memory_type_debug() {
    assert_eq!(format!("{:?}", MemoryType::Device), "Device");
    assert_eq!(format!("{:?}", MemoryType::Shared), "Shared");
    assert_eq!(format!("{:?}", MemoryType::Pinned), "Pinned");
}

// ── ComputeCapabilities ─────────────────────────────────────────────────────

#[test]
fn compute_capabilities_basic() {
    let cc = ComputeCapabilities {
        max_workgroup_size: [1024, 1024, 64],
        max_grid_size: [65535, 65535, 65535],
        max_shared_memory_bytes: 49152,
        compute_units: 80,
        supports_fp16: true,
        supports_int8: true,
        supports_subgroups: true,
    };
    assert_eq!(cc.compute_units, 80);
    assert!(cc.supports_fp16);
}

#[test]
fn compute_capabilities_minimal() {
    let cc = ComputeCapabilities {
        max_workgroup_size: [1, 1, 1],
        max_grid_size: [1, 1, 1],
        max_shared_memory_bytes: 0,
        compute_units: 1,
        supports_fp16: false,
        supports_int8: false,
        supports_subgroups: false,
    };
    assert_eq!(cc.compute_units, 1);
    assert!(!cc.supports_fp16);
}

#[test]
fn compute_capabilities_clone_eq() {
    let cc = ComputeCapabilities {
        max_workgroup_size: [256, 256, 64],
        max_grid_size: [65535, 65535, 65535],
        max_shared_memory_bytes: 32768,
        compute_units: 40,
        supports_fp16: true,
        supports_int8: false,
        supports_subgroups: true,
    };
    let cc2 = cc.clone();
    assert_eq!(cc, cc2);
}

#[test]
fn compute_capabilities_debug() {
    let cc = ComputeCapabilities {
        max_workgroup_size: [1, 1, 1],
        max_grid_size: [1, 1, 1],
        max_shared_memory_bytes: 0,
        compute_units: 1,
        supports_fp16: false,
        supports_int8: false,
        supports_subgroups: false,
    };
    let s = format!("{cc:?}");
    assert!(s.contains("ComputeCapabilities"));
}

// ── ProgramSource ───────────────────────────────────────────────────────────

#[test]
fn program_source_text() {
    let src = ProgramSource::Source("__kernel void test() {}");
    match src {
        ProgramSource::Source(s) => assert!(s.contains("__kernel")),
        _ => panic!("expected Source"),
    }
}

#[test]
fn program_source_spirv() {
    let spirv_magic: &[u8] = &[0x03, 0x02, 0x23, 0x07];
    let src = ProgramSource::SpirV(spirv_magic);
    match src {
        ProgramSource::SpirV(bytes) => assert_eq!(bytes.len(), 4),
        _ => panic!("expected SpirV"),
    }
}

#[test]
fn program_source_clone_eq() {
    let src1 = ProgramSource::Source("test");
    let src2 = src1;
    assert_eq!(src1, src2);
}

#[test]
fn program_source_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ProgramSource::Source("a"));
    set.insert(ProgramSource::Source("b"));
    set.insert(ProgramSource::Source("a")); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn program_source_debug() {
    let src = ProgramSource::Source("test");
    let s = format!("{src:?}");
    assert!(s.contains("Source"));
}

// ── AllocatorStats ──────────────────────────────────────────────────────────

#[test]
fn allocator_stats_construction() {
    let stats =
        AllocatorStats { total_allocated: 1024, peak_allocated: 2048, allocation_count: 10 };
    assert_eq!(stats.total_allocated, 1024);
    assert_eq!(stats.peak_allocated, 2048);
    assert_eq!(stats.allocation_count, 10);
}

#[test]
fn allocator_stats_zero() {
    let stats = AllocatorStats { total_allocated: 0, peak_allocated: 0, allocation_count: 0 };
    assert_eq!(stats.total_allocated, 0);
}

#[test]
fn allocator_stats_clone_eq() {
    let s1 = AllocatorStats { total_allocated: 100, peak_allocated: 200, allocation_count: 3 };
    let s2 = s1.clone();
    assert_eq!(s1, s2);
}

// ── Edge cases with empty strings ───────────────────────────────────────────

#[test]
fn hal_error_empty_device_message() {
    let e = HalError::DeviceNotFound(String::new());
    assert!(e.to_string().contains("device not found:"));
}

#[test]
fn hal_error_empty_backend() {
    let e = HalError::BackendError { backend: String::new(), message: String::new() };
    let s = e.to_string();
    assert!(s.contains("[]"));
}

#[test]
fn hal_error_zero_timeout() {
    let e = HalError::Timeout { operation: "noop".into(), elapsed_ms: 0 };
    assert!(e.to_string().contains("0 ms"));
}

#[test]
fn hal_error_zero_oom() {
    let e = HalError::OutOfMemory { requested: 0, available: 0 };
    let s = e.to_string();
    assert!(s.contains("0 B"));
}

// ── Unicode in error messages ───────────────────────────────────────────────

#[test]
fn hal_error_unicode_message() {
    let e = HalError::DeviceNotFound("設備未找到 🎮".into());
    assert!(e.to_string().contains("🎮"));
}

#[test]
fn hal_error_long_message() {
    let long_msg = "x".repeat(10_000);
    let e = HalError::CompilationFailed(long_msg.clone());
    assert!(e.to_string().len() > 10_000);
}
