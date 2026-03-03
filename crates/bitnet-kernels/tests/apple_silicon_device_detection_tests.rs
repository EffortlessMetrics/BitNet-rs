#![cfg(target_os = "macos")]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    clippy::manual_div_ceil,
    clippy::useless_vec,
    clippy::approx_constant,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::assertions_on_constants,
    clippy::manual_is_multiple_of
)]
//! Apple Silicon device detection and capability reporting tests.
//!
//! Validates that the runtime correctly detects CPU features, thread counts,
//! memory configuration, and architecture strings on macOS / Apple Silicon.
//! Non-GPU tests run on any macOS aarch64 host; GPU tests are `#[ignore]`.

use std::ffi::CStr;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Query an integer sysctl by name. Returns `None` if the key doesn't exist or
/// the call fails.
fn sysctl_u64(name: &str) -> Option<u64> {
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            c_name.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value) } else { None }
}

/// Query a string sysctl by name.
fn sysctl_string(name: &str) -> Option<String> {
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut size: usize = 0;
    // First call: get required buffer size.
    let ret = unsafe {
        libc::sysctlbyname(
            c_name.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || size == 0 {
        return None;
    }
    let mut buf = vec![0u8; size];
    let ret = unsafe {
        libc::sysctlbyname(
            c_name.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return None;
    }
    // Strip trailing NUL.
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf).ok()
}

// ---------------------------------------------------------------------------
// Tests — NEON capability detection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod aarch64_detection {
    use super::*;

    /// NEON is architecturally guaranteed on AArch64; confirm runtime agrees.
    #[test]
    fn test_neon_always_available() {
        assert!(
            std::arch::is_aarch64_feature_detected!("neon"),
            "NEON must be available on every AArch64 target"
        );
    }

    /// Floating-point extension is mandatory on AArch64.
    #[test]
    fn test_fp_feature_detected() {
        assert!(
            std::arch::is_aarch64_feature_detected!("fp"),
            "FP extension must be available on AArch64"
        );
    }

    /// Apple Silicon supports CRC32 instructions.
    #[test]
    fn test_crc32_feature_detected() {
        assert!(
            std::arch::is_aarch64_feature_detected!("crc"),
            "CRC32 should be available on Apple Silicon"
        );
    }

    /// Validate that multiple NEON-related features are all reported together.
    #[test]
    fn test_neon_feature_bundle() {
        let neon = std::arch::is_aarch64_feature_detected!("neon");
        let fp = std::arch::is_aarch64_feature_detected!("fp");
        // On Apple Silicon both must be true simultaneously.
        assert!(neon && fp, "NEON and FP must both be detected on AArch64");
    }
}

// ---------------------------------------------------------------------------
// Tests — thread / parallelism detection
// ---------------------------------------------------------------------------

/// `available_parallelism` should report at least 1 core on any macOS host.
#[test]
fn test_available_parallelism_positive() {
    let par = std::thread::available_parallelism()
        .expect("available_parallelism should succeed on macOS")
        .get();
    assert!(par >= 1, "expected at least 1 thread, got {par}");
}

/// hw.ncpu sysctl should agree with `available_parallelism` (or at least be
/// non-zero).
#[test]
fn test_hw_ncpu_sysctl() {
    let ncpu = sysctl_u64("hw.ncpu").expect("hw.ncpu sysctl must exist on macOS");
    assert!(ncpu >= 1, "hw.ncpu should be >= 1, got {ncpu}");
    // Cross-check with std.
    let std_par = std::thread::available_parallelism().unwrap().get() as u64;
    assert_eq!(ncpu, std_par, "hw.ncpu ({ncpu}) should match available_parallelism ({std_par})");
}

// ---------------------------------------------------------------------------
// Tests — memory / alignment
// ---------------------------------------------------------------------------

/// Apple Silicon NEON registers are 128-bit (16 bytes); verify heap
/// allocations are at least 16-byte aligned.
#[test]
fn test_neon_alignment_requirements() {
    let v: Vec<f32> = vec![0.0f32; 256];
    let ptr = v.as_ptr() as usize;
    assert_eq!(
        ptr % 16,
        0,
        "Vec<f32> allocation should be 16-byte aligned for NEON, got ptr {ptr:#x}"
    );
}

/// hw.memsize should report a positive physical memory value.
#[test]
fn test_hw_memsize_sysctl() {
    let memsize = sysctl_u64("hw.memsize").expect("hw.memsize sysctl must exist on macOS");
    // Any real Mac has at least 1 GiB.
    let one_gib = 1u64 << 30;
    assert!(memsize >= one_gib, "expected >= 1 GiB physical memory, got {memsize}");
}

// ---------------------------------------------------------------------------
// Tests — architecture string detection
// ---------------------------------------------------------------------------

/// hw.machine should return "arm64" on Apple Silicon.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_hw_machine_arm64() {
    let machine = sysctl_string("hw.machine").expect("hw.machine sysctl must exist on macOS");
    assert_eq!(machine, "arm64", "expected hw.machine=\"arm64\", got \"{machine}\"");
}

/// machdep.cpu.brand_string (or hw.model) should be non-empty.
#[test]
fn test_cpu_brand_string_non_empty() {
    // On Apple Silicon hw.model is typically "MacXX,Y"; try it first.
    let brand = sysctl_string("hw.model")
        .or_else(|| sysctl_string("machdep.cpu.brand_string"))
        .expect("could not read any CPU brand/model string");
    assert!(!brand.is_empty(), "CPU brand/model string must not be empty");
}

// ---------------------------------------------------------------------------
// Tests — GPU / Metal (ignored: require Metal runtime)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_metal_device_available() {
    // Would use `metal::Device::system_default()` to verify a Metal GPU is
    // present. Ignored because CI runners may lack a GPU.
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_metal_device_name_non_empty() {
    // Would verify `device.name()` returns a non-empty string identifying the
    // Apple GPU family.
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn test_metal_unified_memory_reported() {
    // Would verify `device.has_unified_memory()` returns `true` on Apple
    // Silicon, confirming shared CPU/GPU address space.
}
