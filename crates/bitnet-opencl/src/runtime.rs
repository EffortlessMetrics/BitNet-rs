//! OpenCL runtime: platform/device discovery via the OpenCL ICD loader.

/// Placeholder for OpenCL runtime discovery.
///
/// Requires the `oneapi` feature and a working OpenCL ICD loader.
pub fn discover_devices() -> Vec<String> {
    discover_devices_impl()
}

#[cfg(feature = "oneapi")]
fn discover_devices_impl() -> Vec<String> {
    use opencl3::device::Device;
    use opencl3::platform::get_platforms;

    let mut discovered = Vec::new();

    let Ok(platforms) = get_platforms() else {
        return discovered;
    };

    for platform in platforms {
        let Ok(devices) = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_ALL) else {
            continue;
        };

        for device_id in devices {
            let device = Device::new(device_id);

            let vendor = device.vendor().unwrap_or_else(|_| String::from("Unknown vendor"));
            let name = device.name().unwrap_or_else(|_| String::from("Unknown device"));

            discovered.push(format!("{vendor} - {name}"));
        }
    }

    discovered
}

#[cfg(not(feature = "oneapi"))]
fn discover_devices_impl() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::discover_devices;

    #[test]
    fn discover_devices_never_panics() {
        let _ = discover_devices();
    }

    #[test]
    fn discovered_device_names_are_non_empty() {
        for device in discover_devices() {
            assert!(!device.trim().is_empty());
        }
    }
}
