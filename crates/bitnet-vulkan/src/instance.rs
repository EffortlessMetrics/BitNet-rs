//! Vulkan instance creation with validation layers for debug builds.

use crate::error::{Result, VulkanError};
use ash::vk;
use std::ffi::CStr;
use tracing::{debug, info};

/// Configuration for Vulkan instance creation.
#[derive(Debug, Clone)]
pub struct InstanceConfig {
    /// Application name reported to the Vulkan driver.
    pub app_name: String,
    /// Application version (Vulkan packed format).
    pub app_version: u32,
    /// Enable validation layers (automatically enabled in debug builds).
    pub enable_validation: bool,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        Self {
            app_name: "bitnet-vulkan".into(),
            app_version: vk::make_api_version(0, 0, 1, 0),
            enable_validation: cfg!(debug_assertions),
        }
    }
}

/// Wrapper around a Vulkan instance with its entry point.
pub struct VulkanInstance {
    #[allow(dead_code)]
    entry: ash::Entry,
    #[allow(dead_code)]
    instance: ash::Instance,
    validation_enabled: bool,
}

impl VulkanInstance {
    /// Create a new Vulkan instance with the given configuration.
    ///
    /// In debug builds (or when `config.enable_validation` is true),
    /// validation layers are requested if available.
    pub fn new(config: &InstanceConfig) -> Result<Self> {
        let entry = unsafe {
            ash::Entry::load().map_err(|e| {
                VulkanError::InstanceCreation(format!(
                    "failed to load Vulkan loader: {e}"
                ))
            })?
        };

        let app_name =
            std::ffi::CString::new(config.app_name.as_str()).unwrap_or_default();
        let engine_name = std::ffi::CString::new("bitnet-rs").unwrap_or_default();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(config.app_version)
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_2);

        let mut layer_names_raw = Vec::new();
        let validation_layer = c"VK_LAYER_KHRONOS_validation";

        let validation_enabled = if config.enable_validation {
            let available = unsafe {
                entry
                    .enumerate_instance_layer_properties()
                    .unwrap_or_default()
            };
            let found = available.iter().any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                name == validation_layer
            });
            if found {
                layer_names_raw.push(validation_layer.as_ptr());
                debug!("Vulkan validation layers enabled");
                true
            } else {
                debug!("Validation layers requested but not available");
                false
            }
        } else {
            false
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_raw);

        let instance = unsafe {
            entry.create_instance(&create_info, None).map_err(|e| {
                VulkanError::InstanceCreation(format!(
                    "vkCreateInstance failed: {e}"
                ))
            })?
        };

        info!(
            "Vulkan instance created (validation={})",
            validation_enabled
        );

        Ok(Self {
            entry,
            instance,
            validation_enabled,
        })
    }

    /// Whether validation layers are active on this instance.
    pub fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// Borrow the raw `ash::Instance`.
    pub fn raw(&self) -> &ash::Instance {
        &self.instance
    }

    /// Borrow the `ash::Entry`.
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_config_default_has_app_name() {
        let cfg = InstanceConfig::default();
        assert_eq!(cfg.app_name, "bitnet-vulkan");
        assert_eq!(cfg.enable_validation, cfg!(debug_assertions));
    }
}
