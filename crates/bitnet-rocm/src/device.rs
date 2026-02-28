//! HIP device enumeration via dynamic loading.

use crate::error::{Result, RocmError};
use crate::ffi::HipDeviceProperties;
use tracing::{info, warn};

/// Information about a discovered AMD GPU.
#[derive(Debug, Clone)]
pub struct RocmDeviceInfo {
    pub index: usize,
    pub name: String,
    pub total_memory_mib: usize,
    pub compute_units: i32,
    pub warp_size: i32,
}

/// Attempt to detect the HIP runtime and enumerate AMD GPUs.
///
/// Returns an empty list if the runtime is absent (no error).
pub fn enumerate_devices() -> Result<Vec<RocmDeviceInfo>> {
    // Try to detect HIP runtime presence.
    if !hip_runtime_available() {
        info!("HIP runtime not detected — ROCm backend unavailable");
        return Ok(vec![]);
    }

    let count = hip_device_count()?;
    if count == 0 {
        warn!("HIP runtime present but no AMD GPU devices found");
        return Ok(vec![]);
    }

    let mut devices = Vec::with_capacity(count);
    for i in 0..count {
        let props = hip_get_device_properties(i)?;
        devices.push(RocmDeviceInfo {
            index: i,
            name: props.device_name(),
            total_memory_mib: props.total_memory_mib(),
            compute_units: props.compute_units,
            warp_size: props.warp_size,
        });
        info!(
            index = i,
            name = %devices.last().unwrap().name,
            memory_mib = devices.last().unwrap().total_memory_mib,
            "discovered AMD GPU"
        );
    }

    Ok(devices)
}

/// Check if the HIP shared library can be located.
pub fn hip_runtime_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        unsafe { libloading::Library::new("libamdhip64.so").is_ok() }
    }
    #[cfg(target_os = "windows")]
    {
        unsafe { libloading::Library::new("amdhip64.dll").is_ok() }
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// Stub: query HIP device count via dynamic loading.
fn hip_device_count() -> Result<usize> {
    // In a real implementation this would dlsym hipGetDeviceCount.
    // For the microcrate scaffold we return 0 when the runtime is not linked.
    Ok(0)
}

/// Stub: query device properties for device `index`.
fn hip_get_device_properties(index: usize) -> Result<HipDeviceProperties> {
    let _ = index;
    Err(RocmError::NoDevice)
}
