//! OpenCL runtime discovery via the OpenCL ICD loader.

#[cfg(feature = "oneapi")]
use opencl3::{
    device::{CL_DEVICE_TYPE_ALL, Device},
    platform::get_platforms,
};

/// Discover OpenCL devices available through the installed ICD loader.
///
/// Discovery is best-effort: missing drivers, loader failures, or device query
/// errors produce an empty list instead of failing backend initialization.
#[cfg(feature = "oneapi")]
pub fn discover_devices() -> Vec<String> {
    let mut devices = Vec::new();

    let Ok(platforms) = get_platforms() else {
        return devices;
    };

    for platform in platforms {
        let platform_name = platform.name().unwrap_or_else(|_| "Unknown OpenCL Platform".into());
        let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_ALL) else {
            continue;
        };

        for device_id in device_ids {
            let device = Device::new(device_id);
            if let Ok(device_name) = device.name() {
                devices.push(format!("{platform_name} :: {device_name}"));
            }
        }
    }

    devices.sort();
    devices.dedup();
    devices
}

/// Device discovery is unavailable when the OpenCL runtime feature is disabled.
#[cfg(not(feature = "oneapi"))]
pub fn discover_devices() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::discover_devices;

    #[test]
    #[cfg(not(feature = "oneapi"))]
    fn discover_devices_is_empty_without_oneapi_feature() {
        assert!(discover_devices().is_empty());
    }

    #[test]
    #[cfg(feature = "oneapi")]
    fn discover_devices_does_not_panic_with_oneapi_feature() {
        let _ = discover_devices();
    }
}
