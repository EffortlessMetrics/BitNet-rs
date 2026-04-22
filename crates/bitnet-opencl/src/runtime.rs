//! OpenCL runtime: platform/device discovery via the OpenCL ICD loader.

#[cfg(feature = "oneapi")]
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device};
#[cfg(feature = "oneapi")]
use opencl3::platform::get_platforms;

/// Discover OpenCL devices as `"platform :: device"` display strings.
///
/// When the `oneapi` feature is disabled or the OpenCL ICD loader is not
/// available at runtime, this returns an empty list.
pub fn discover_devices() -> Vec<String> {
    #[cfg(feature = "oneapi")]
    {
        let mut discovered = Vec::new();
        let Ok(platforms) = get_platforms() else {
            return discovered;
        };

        for platform in platforms {
            let platform_name =
                platform.name().unwrap_or_else(|_| String::from("unknown-platform"));
            let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_ALL) else {
                continue;
            };

            for id in device_ids {
                let device = Device::new(id);
                let device_name = device.name().unwrap_or_else(|_| String::from("unknown-device"));
                discovered.push(format!("{platform_name} :: {device_name}"));
            }
        }

        discovered.sort_unstable();
        discovered.dedup();
        discovered
    }
    #[cfg(not(feature = "oneapi"))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::discover_devices;

    #[cfg(not(feature = "oneapi"))]
    #[test]
    fn discover_devices_is_empty_without_oneapi_feature() {
        assert!(discover_devices().is_empty());
    }

    #[cfg(feature = "oneapi")]
    #[test]
    fn discover_devices_never_panics_with_oneapi_feature() {
        let _ = discover_devices();
    }
}
