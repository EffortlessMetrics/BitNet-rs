//! Level Zero runtime visibility probing through installed command-line tools.

#[cfg(all(windows, feature = "level-zero-loader"))]
use std::ffi::c_void;

use serde::{Deserialize, Serialize};

use super::command_output;

/// Level Zero runtime visibility result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LevelZeroProbe {
    /// Whether Level Zero tooling was visible.
    pub runtime_available: bool,
    /// Best-effort device names parsed from `ze_info` or `sycl-ls`.
    pub devices: Vec<String>,
    /// Best-effort PCI/device IDs parsed from `ze_info`.
    pub device_ids: Vec<String>,
    /// Non-fatal probe error when the runtime tooling was absent or unusable.
    pub error: Option<String>,
}

impl LevelZeroProbe {
    /// Build an unavailable Level Zero probe result.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            runtime_available: false,
            devices: Vec::new(),
            device_ids: Vec::new(),
            error: Some(reason.into()),
        }
    }
}

/// Probe Level Zero visibility without compiling or dispatching kernels.
pub fn probe_level_zero() -> LevelZeroProbe {
    match command_output("ze_info", std::iter::empty::<&str>()) {
        Ok(stdout) => {
            let devices = parse_ze_info_devices(&stdout);
            let device_ids = parse_ze_info_device_ids(&stdout);
            LevelZeroProbe { runtime_available: true, devices, device_ids, error: None }
        }
        Err(ze_error) => match command_output("sycl-ls", std::iter::empty::<&str>()) {
            Ok(stdout) => {
                let devices = parse_sycl_ls_level_zero_devices(&stdout);
                LevelZeroProbe {
                    runtime_available: !devices.is_empty(),
                    devices,
                    device_ids: Vec::new(),
                    error: None,
                }
            }
            Err(sycl_error) => match probe_level_zero_loader() {
                Ok(probe) => probe,
                Err(loader_error) => {
                    LevelZeroProbe::unavailable(format!("{ze_error}; {sycl_error}; {loader_error}"))
                }
            },
        },
    }
}

pub(crate) fn parse_ze_info_devices(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("Device Name")
                .and_then(|rest| rest.split_once(':').map(|(_, value)| value.trim().to_owned()))
                .filter(|value| !value.is_empty())
        })
        .collect()
}

pub(crate) fn parse_ze_info_device_ids(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let lower = trimmed.to_ascii_lowercase();
            if !(lower.starts_with("device id") || lower.starts_with("deviceid")) {
                return None;
            }
            let value = trimmed
                .split_once(':')
                .map(|(_, value)| value.trim())
                .or_else(|| trimmed.split_once('=').map(|(_, value)| value.trim()))?;
            normalize_device_id(value)
        })
        .collect()
}

pub(crate) fn parse_sycl_ls_level_zero_devices(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("level_zero"))
        .map(|line| line.trim().to_owned())
        .filter(|line| !line.is_empty())
        .collect()
}

fn normalize_device_id(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_matches(['"', '\'']);
    if trimmed.is_empty() {
        return None;
    }
    let hex = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")).unwrap_or(trimmed);
    if hex.is_empty() || !hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("0x{}", hex.to_ascii_uppercase()))
}

#[cfg(any(test, all(windows, feature = "level-zero-loader")))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct LevelZeroLoaderDevice {
    name: String,
    device_id: u32,
}

#[cfg(any(test, all(windows, feature = "level-zero-loader")))]
fn level_zero_probe_from_loader_devices(
    devices: Vec<LevelZeroLoaderDevice>,
) -> Result<LevelZeroProbe, String> {
    let mut names = Vec::new();
    let mut device_ids = Vec::new();
    for device in devices {
        let name = device.name.trim();
        if !name.is_empty() {
            names.push(name.to_owned());
        }
        device_ids.push(format!("0x{:04X}", device.device_id));
    }

    if names.is_empty() && device_ids.is_empty() {
        Err("ze_loader.dll reported no Level Zero devices".to_owned())
    } else {
        Ok(LevelZeroProbe { runtime_available: true, devices: names, device_ids, error: None })
    }
}

#[cfg(all(windows, feature = "level-zero-loader"))]
fn probe_level_zero_loader() -> Result<LevelZeroProbe, String> {
    use std::ffi::CStr;

    use libloading::Library;

    type ZeInit = unsafe extern "system" fn(u32) -> i32;
    type ZeDriverGet = unsafe extern "system" fn(*mut u32, *mut *mut c_void) -> i32;
    type ZeDeviceGet = unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32;
    type ZeDeviceGetProperties =
        unsafe extern "system" fn(*mut c_void, *mut ZeDeviceProperties) -> i32;

    const ZE_RESULT_SUCCESS: i32 = 0;

    let library = unsafe { Library::new("ze_loader.dll") }
        .map_err(|err| format!("ze_loader.dll unavailable: {err}"))?;

    let ze_init = unsafe { library.get::<ZeInit>(b"zeInit") }
        .map_err(|err| format!("zeInit unavailable: {err}"))?;
    let ze_driver_get = unsafe { library.get::<ZeDriverGet>(b"zeDriverGet") }
        .map_err(|err| format!("zeDriverGet unavailable: {err}"))?;
    let ze_device_get = unsafe { library.get::<ZeDeviceGet>(b"zeDeviceGet") }
        .map_err(|err| format!("zeDeviceGet unavailable: {err}"))?;
    let ze_device_get_properties =
        unsafe { library.get::<ZeDeviceGetProperties>(b"zeDeviceGetProperties") }
            .map_err(|err| format!("zeDeviceGetProperties unavailable: {err}"))?;

    let init_result = unsafe { ze_init(0) };
    if init_result != ZE_RESULT_SUCCESS {
        return Err(format!("zeInit failed with {init_result}"));
    }

    let mut driver_count = 0_u32;
    let driver_count_result = unsafe { ze_driver_get(&mut driver_count, std::ptr::null_mut()) };
    if driver_count_result != ZE_RESULT_SUCCESS {
        return Err(format!("zeDriverGet count failed with {driver_count_result}"));
    }
    if driver_count == 0 {
        return Err("ze_loader.dll reported no Level Zero drivers".to_owned());
    }

    let mut drivers = vec![std::ptr::null_mut::<c_void>(); driver_count as usize];
    let driver_result = unsafe { ze_driver_get(&mut driver_count, drivers.as_mut_ptr()) };
    if driver_result != ZE_RESULT_SUCCESS {
        return Err(format!("zeDriverGet drivers failed with {driver_result}"));
    }

    let mut devices = Vec::new();
    for driver in drivers {
        let mut device_count = 0_u32;
        let device_count_result =
            unsafe { ze_device_get(driver, &mut device_count, std::ptr::null_mut()) };
        if device_count_result != ZE_RESULT_SUCCESS || device_count == 0 {
            continue;
        }

        let mut handles = vec![std::ptr::null_mut::<c_void>(); device_count as usize];
        let device_result =
            unsafe { ze_device_get(driver, &mut device_count, handles.as_mut_ptr()) };
        if device_result != ZE_RESULT_SUCCESS {
            continue;
        }

        for handle in handles {
            let mut properties = ZeDeviceProperties::default();
            let properties_result = unsafe { ze_device_get_properties(handle, &mut properties) };
            if properties_result != ZE_RESULT_SUCCESS {
                continue;
            }

            let name = unsafe { CStr::from_ptr(properties.name.as_ptr()) }
                .to_string_lossy()
                .trim()
                .to_owned();
            devices.push(LevelZeroLoaderDevice { name, device_id: properties.device_id });
        }
    }

    level_zero_probe_from_loader_devices(devices)
}

#[cfg(not(all(windows, feature = "level-zero-loader")))]
fn probe_level_zero_loader() -> Result<LevelZeroProbe, String> {
    Err("native Level Zero loader probe not compiled for this target".to_owned())
}

#[cfg(all(windows, feature = "level-zero-loader"))]
#[repr(C)]
struct ZeDeviceProperties {
    stype: u32,
    p_next: *mut c_void,
    device_type: u32,
    vendor_id: u32,
    device_id: u32,
    flags: u32,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_mem_alloc_size: u64,
    max_hardware_contexts: u32,
    max_command_queue_priority: u32,
    num_threads_per_eu: u32,
    physical_eu_simd_width: u32,
    num_eus_per_subslice: u32,
    num_subslices_per_slice: u32,
    num_slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [u8; 16],
    name: [std::ffi::c_char; ZE_MAX_DEVICE_NAME],
}

#[cfg(all(windows, feature = "level-zero-loader"))]
impl Default for ZeDeviceProperties {
    fn default() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
            p_next: std::ptr::null_mut(),
            device_type: 0,
            vendor_id: 0,
            device_id: 0,
            flags: 0,
            subdevice_id: 0,
            core_clock_rate: 0,
            max_mem_alloc_size: 0,
            max_hardware_contexts: 0,
            max_command_queue_priority: 0,
            num_threads_per_eu: 0,
            physical_eu_simd_width: 0,
            num_eus_per_subslice: 0,
            num_subslices_per_slice: 0,
            num_slices: 0,
            timer_resolution: 0,
            timestamp_valid_bits: 0,
            kernel_timestamp_valid_bits: 0,
            uuid: [0; 16],
            name: [0; ZE_MAX_DEVICE_NAME],
        }
    }
}

#[cfg(all(windows, feature = "level-zero-loader"))]
const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 1;
#[cfg(all(windows, feature = "level-zero-loader"))]
const ZE_MAX_DEVICE_NAME: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_level_zero_device_ids_from_ze_info() {
        let output = r"
            Device Name       : Intel(R) Arc(TM) 140V Graphics
            Device ID         : 0x64a0
            DeviceId          : 64A0
        ";

        assert_eq!(parse_ze_info_device_ids(output), vec!["0x64A0", "0x64A0"]);
    }

    #[test]
    fn loader_devices_become_runtime_visible_probe() {
        let probe = level_zero_probe_from_loader_devices(vec![LevelZeroLoaderDevice {
            name: "Intel(R) Arc(TM) 140V GPU (16GB)".to_owned(),
            device_id: 0x64A0,
        }])
        .unwrap();

        assert!(probe.runtime_available);
        assert_eq!(probe.devices, vec!["Intel(R) Arc(TM) 140V GPU (16GB)"]);
        assert_eq!(probe.device_ids, vec!["0x64A0"]);
        assert_eq!(probe.error, None);
    }

    #[test]
    fn loader_without_devices_stays_unavailable() {
        let err = level_zero_probe_from_loader_devices(Vec::new()).unwrap_err();

        assert!(err.contains("no Level Zero devices"));
    }
}
