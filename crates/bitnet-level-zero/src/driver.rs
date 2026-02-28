//! Driver and device enumeration for Level-Zero.
//!
//! Wraps `zeDriverGet` and `zeDeviceGet` via dynamic loading.

use crate::error::{LevelZeroError, Result};
use crate::ffi::ZeDeviceProperties;
use tracing::{debug, info, warn};

/// Information about a discovered Level-Zero driver.
#[derive(Debug, Clone)]
pub struct DriverInfo {
    /// Driver index (0-based).
    pub index: usize,
    /// Number of devices accessible through this driver.
    pub device_count: usize,
    /// API version reported by the driver.
    pub api_version: (u32, u32),
}

/// Enumerates available Level-Zero drivers.
///
/// Returns an empty vec if Level-Zero is not installed.
pub fn enumerate_drivers() -> Result<Vec<DriverInfo>> {
    info!("Level-Zero driver enumeration requested");

    if !is_runtime_available() {
        debug!("Level-Zero runtime not detected");
        return Ok(Vec::new());
    }

    warn!("Level-Zero runtime loading not yet implemented -- returning empty driver list");
    Ok(Vec::new())
}

/// Checks whether the Level-Zero loader library can be found on this system.
pub fn is_runtime_available() -> bool {
    let lib_name = loader_library_name();
    // SAFETY: We only probe for the library existence; no symbols are called.
    let result = unsafe { libloading::Library::new(lib_name) };
    result.is_ok()
}

/// Return the platform-specific loader library name.
pub fn loader_library_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "ze_loader.dll"
    }
    #[cfg(target_os = "linux")]
    {
        "libze_loader.so.1"
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        "libze_loader.so"
    }
}

/// A discovered Level-Zero device with its parent driver.
#[derive(Debug, Clone)]
pub struct DeviceEntry {
    /// Index of the parent driver.
    pub driver_index: usize,
    /// Index of this device within the driver.
    pub device_index: usize,
    /// Device properties (name, type, etc.).
    pub properties: ZeDeviceProperties,
}

/// Enumerate all GPU devices across all drivers.
pub fn enumerate_gpu_devices() -> Result<Vec<DeviceEntry>> {
    let drivers = enumerate_drivers()?;
    if drivers.is_empty() {
        return Ok(Vec::new());
    }

    let gpus = Vec::new();
    for driver in &drivers {
        debug!(
            "Driver {} reports {} device(s)",
            driver.index, driver.device_count
        );
    }
    Ok(gpus)
}

/// Select the best available GPU device (highest EU count).
pub fn select_best_gpu() -> Result<DeviceEntry> {
    let devices = enumerate_gpu_devices()?;
    devices
        .into_iter()
        .max_by_key(|d| {
            d.properties.num_slices as u64
                * d.properties.num_subslices_per_slice as u64
                * d.properties.num_eu_per_subslice as u64
        })
        .ok_or(LevelZeroError::NoDeviceFound)
}
