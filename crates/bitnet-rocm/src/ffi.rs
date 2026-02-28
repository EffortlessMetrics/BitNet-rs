//! Basic HIP FFI definitions (hip_runtime_api types).
//!
//! These are the minimal type definitions needed to call into the HIP runtime
//! via dynamic loading. We avoid a build-time dependency on the ROCm SDK.

use std::ffi::c_void;

/// Opaque HIP stream handle.
pub type HipStream = *mut c_void;

/// Opaque HIP module handle.
pub type HipModule = *mut c_void;

/// Opaque HIP function (kernel) handle.
pub type HipFunction = *mut c_void;

/// Opaque device pointer.
pub type HipDevicePtr = *mut c_void;

/// Null stream constant.
pub const HIP_STREAM_DEFAULT: HipStream = std::ptr::null_mut();

/// HIP device properties (simplified subset).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HipDeviceProperties {
    pub name: [u8; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub multi_processor_count: i32,
    pub compute_units: i32,
}

impl Default for HipDeviceProperties {
    fn default() -> Self {
        Self {
            name: [0u8; 256],
            total_global_mem: 0,
            shared_mem_per_block: 0,
            warp_size: 0,
            max_threads_per_block: 0,
            max_threads_dim: [0; 3],
            max_grid_size: [0; 3],
            clock_rate: 0,
            multi_processor_count: 0,
            compute_units: 0,
        }
    }
}

impl HipDeviceProperties {
    /// Extract the device name as a UTF-8 string.
    pub fn device_name(&self) -> String {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(self.name.len());
        String::from_utf8_lossy(&self.name[..end]).to_string()
    }

    /// Total global memory in MiB.
    pub fn total_memory_mib(&self) -> usize {
        self.total_global_mem / (1024 * 1024)
    }
}

/// Memory copy kind for hipMemcpy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}
