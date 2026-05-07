//! OpenCL runtime: platform/device discovery via the OpenCL ICD loader.
//!
//! This is **discovery only**. The OpenCL backend is currently scaffolded;
//! none of the BitNet inference paths run on OpenCL today. Callers should
//! treat the returned device names as informational (probes, CLI listings,
//! tracking receipts), not as a guarantee that inference will succeed on
//! the listed devices.

/// Enumerate available OpenCL devices as `"<vendor> - <name>"` strings.
///
/// Returns an empty vector when:
/// - the `oneapi` feature is not enabled (no OpenCL ICD loader linked),
/// - no OpenCL platform is installed at runtime,
/// - the loader is present but no platforms expose any devices.
///
/// This function never panics, so it is safe to call from probes and
/// receipt-emitting code.
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
