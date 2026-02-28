//! Physical and logical device selection, preferring dedicated compute queues.

use crate::error::{Result, VulkanError};
use ash::vk;
use tracing::{debug, info};

/// Criteria for selecting a physical device.
#[derive(Debug, Clone)]
pub struct DeviceSelector {
    /// Prefer discrete GPUs over integrated.
    pub prefer_discrete: bool,
    /// Require a dedicated compute queue family (not shared with graphics).
    pub require_compute_queue: bool,
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self { prefer_discrete: true, require_compute_queue: false }
    }
}

/// Information about a selected physical device and its compute queue family.
#[derive(Debug, Clone)]
pub struct SelectedDevice {
    /// Vulkan physical device handle.
    pub physical_device: vk::PhysicalDevice,
    /// Index of the queue family used for compute.
    pub compute_queue_family: u32,
    /// Device name from the driver.
    pub device_name: String,
    /// Device type (discrete, integrated, etc.).
    pub device_type: vk::PhysicalDeviceType,
}

/// Score a physical device for compute suitability.
fn score_device(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    selector: &DeviceSelector,
) -> Option<(u32, u32)> {
    let props = unsafe { instance.get_physical_device_properties(device) };
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

    // Find a queue family that supports compute.
    let compute_family = queue_families.iter().enumerate().find_map(|(i, qf)| {
        if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) { Some(i as u32) } else { None }
    });

    let compute_family = compute_family?;

    // Check if there's a dedicated compute queue (no graphics bit).
    let has_dedicated_compute = queue_families.iter().any(|qf| {
        qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    });

    if selector.require_compute_queue && !has_dedicated_compute {
        return None;
    }

    let mut score: u32 = 0;
    if selector.prefer_discrete && props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        score += 1000;
    }
    if has_dedicated_compute {
        score += 500;
    }
    // Prefer dedicated compute queue family index.
    let best_compute_family = queue_families
        .iter()
        .enumerate()
        .find_map(|(i, qf)| {
            if qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                Some(i as u32)
            } else {
                None
            }
        })
        .unwrap_or(compute_family);

    Some((score, best_compute_family))
}

/// Select the best physical device for compute workloads.
pub fn select_physical_device(
    instance: &ash::Instance,
    selector: &DeviceSelector,
) -> Result<SelectedDevice> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| VulkanError::NoSuitableDevice(format!("{e}")))?
    };

    if devices.is_empty() {
        return Err(VulkanError::NoSuitableDevice("no Vulkan physical devices found".into()));
    }

    let best = devices
        .iter()
        .filter_map(|&dev| {
            let (score, family) = score_device(instance, dev, selector)?;
            Some((dev, score, family))
        })
        .max_by_key(|&(_, score, _)| score);

    match best {
        Some((device, _score, family)) => {
            let props = unsafe { instance.get_physical_device_properties(device) };
            let name = unsafe {
                std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned()
            };
            info!(
                "Selected Vulkan device: {} (type={:?}, compute_family={})",
                name, props.device_type, family
            );
            Ok(SelectedDevice {
                physical_device: device,
                compute_queue_family: family,
                device_name: name,
                device_type: props.device_type,
            })
        }
        None => Err(VulkanError::NoSuitableDevice("no device meets selection criteria".into())),
    }
}

/// Create a logical device with a single compute queue.
pub fn create_logical_device(
    instance: &ash::Instance,
    selected: &SelectedDevice,
) -> Result<(ash::Device, vk::Queue)> {
    let queue_priority = [1.0_f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected.compute_queue_family)
        .queue_priorities(&queue_priority);

    let queue_create_infos = [queue_create_info];
    let device_create_info =
        vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

    let device = unsafe {
        instance
            .create_device(selected.physical_device, &device_create_info, None)
            .map_err(|e| VulkanError::DeviceCreation(format!("{e}")))?
    };

    let queue = unsafe { device.get_device_queue(selected.compute_queue_family, 0) };
    debug!("Logical device created with compute queue");

    Ok((device, queue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_selector_defaults() {
        let sel = DeviceSelector::default();
        assert!(sel.prefer_discrete);
        assert!(!sel.require_compute_queue);
    }

    #[test]
    fn selected_device_debug_format() {
        let sel = SelectedDevice {
            physical_device: vk::PhysicalDevice::null(),
            compute_queue_family: 0,
            device_name: "Test GPU".into(),
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
        };
        let dbg = format!("{sel:?}");
        assert!(dbg.contains("Test GPU"));
    }
}
