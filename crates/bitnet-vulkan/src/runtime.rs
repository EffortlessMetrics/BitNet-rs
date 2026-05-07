//! Vulkan runtime: device discovery via the Vulkan loader.
//!
//! This is **discovery only**. The Vulkan backend is currently scaffolded;
//! none of the BitNet inference paths run on Vulkan today. Callers should
//! treat the returned device names as informational, not as a guarantee
//! that inference will succeed on the listed devices.

/// Discover Vulkan physical devices available through the Vulkan loader.
///
/// When the `vulkan-runtime` feature is disabled, this returns an empty list.
/// When enabled, failures to load the loader, create the instance, or
/// enumerate physical devices are handled gracefully and also return an
/// empty list — discovery never panics or returns a hard error.
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

    fn cstr_to_string_lossy(value: &[i8]) -> String {
        // SAFETY: Vulkan guarantees that fixed-size name fields are null-terminated.
        let cstr = unsafe { CStr::from_ptr(value.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    let entry = {
        // SAFETY: Loading Vulkan entry points is safe; errors are handled below.
        let result = unsafe { Entry::load() };
        match result {
            Ok(entry) => entry,
            Err(error) => {
                log::debug!("Failed to load Vulkan loader during device discovery: {error:?}");
                return Vec::new();
            }
        }
    };
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = {
        // SAFETY: `create_info` points to local data that lives until call returns.
        let result = unsafe { entry.create_instance(&create_info, None) };
        match result {
            Ok(instance) => instance,
            Err(error) => {
                log::debug!("Failed to create Vulkan instance during device discovery: {error:?}");
                return Vec::new();
            }
        }
    };

    let result = {
        // SAFETY: `instance` is valid while this call executes.
        unsafe { instance.enumerate_physical_devices() }
    };

    let devices = match result {
        Ok(physical_devices) => physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                // SAFETY: `physical_device` is returned by Vulkan for this instance.
                let properties =
                    unsafe { instance.get_physical_device_properties(physical_device) };
                let name = cstr_to_string_lossy(&properties.device_name);
                (!name.is_empty()).then_some(name)
            })
            .collect(),
        Err(error) => {
            log::debug!("Failed to enumerate Vulkan physical devices: {error:?}");
            Vec::new()
        }
    };

    // SAFETY: Instance was created in this function and no further Vulkan calls follow.
    unsafe { instance.destroy_instance(None) };
    devices
}
