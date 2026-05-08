//! Vulkan runtime: device discovery via the Vulkan loader.

/// Discover Vulkan physical devices available through the Vulkan loader.
///
/// Discovery is best-effort: missing loaders, instance creation failures, or
/// device query errors produce an empty list instead of failing backend setup.
#[must_use]
pub fn discover_devices() -> Vec<String> {
    discover_devices_impl()
}

#[cfg(not(feature = "vulkan-runtime"))]
fn discover_devices_impl() -> Vec<String> {
    Vec::new()
}

#[cfg(feature = "vulkan-runtime")]
fn discover_devices_impl() -> Vec<String> {
    use ash::{Entry, vk};
    use std::ffi::CStr;

    fn device_name_to_string(device_name: &[i8]) -> Option<String> {
        // SAFETY: Vulkan device name fields are fixed-size, null-terminated strings.
        let cstr = unsafe { CStr::from_ptr(device_name.as_ptr()) };
        let name = cstr.to_string_lossy().trim().to_owned();
        (!name.is_empty()).then_some(name)
    }

    let entry = match unsafe { Entry::load() } {
        Ok(entry) => entry,
        Err(error) => {
            log::debug!("failed to load Vulkan loader during device discovery: {error:?}");
            return Vec::new();
        }
    };

    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(instance) => instance,
        Err(error) => {
            log::debug!("failed to create Vulkan instance during device discovery: {error:?}");
            return Vec::new();
        }
    };

    let result = unsafe { instance.enumerate_physical_devices() };
    let mut devices = match result {
        Ok(physical_devices) => physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let properties =
                    unsafe { instance.get_physical_device_properties(physical_device) };
                device_name_to_string(&properties.device_name)
            })
            .collect::<Vec<_>>(),
        Err(error) => {
            log::debug!("failed to enumerate Vulkan physical devices: {error:?}");
            Vec::new()
        }
    };

    unsafe { instance.destroy_instance(None) };
    devices.sort();
    devices.dedup();
    devices
}
