//! Vulkan compute backend for BitNet-rs inference.
//!
//! This crate provides a Vulkan 1.2 compute backend targeting GPU inference
//! via SPIR-V compute shaders. It uses the [`ash`] crate for raw Vulkan
//! bindings and is designed as a pluggable backend alongside CUDA and OpenCL.
//!
//! # Modules
//!
//! - [`instance`] - Vulkan instance creation with optional validation layers
//! - [`device`] - Physical/logical device selection preferring compute queues
//! - [`pipeline`] - Compute pipeline creation from SPIR-V
//! - [`buffer`] - GPU buffer management (staging + device-local)
//! - [`command`] - Command buffer recording and submission
//! - [`error`] - Error types

pub mod buffer;
pub mod command;
pub mod device;
pub mod error;
pub mod instance;
pub mod pipeline;

pub use buffer::{BufferDescriptor, BufferUsage, GpuBuffer, allocate_buffer};
pub use command::{CommandPool, record_and_submit_compute};
pub use device::{DeviceSelector, SelectedDevice, create_logical_device, select_physical_device};
pub use error::{Result, VulkanError};
pub use instance::{InstanceConfig, VulkanInstance};
pub use pipeline::{ComputePipeline, ComputePipelineBuilder};

/// Top-level Vulkan backend that owns instance, device, and command resources.
///
/// This is the primary entry point for consumers who want a fully initialised
/// Vulkan compute backend.
#[derive(Debug)]
pub struct VulkanBackend {
    config: InstanceConfig,
    device_selector: DeviceSelector,
    initialised: bool,
}

impl VulkanBackend {
    /// Create a new backend with default configuration.
    pub fn new() -> Self {
        Self {
            config: InstanceConfig::default(),
            device_selector: DeviceSelector::default(),
            initialised: false,
        }
    }

    /// Create a backend with custom instance and device configuration.
    pub fn with_config(config: InstanceConfig, selector: DeviceSelector) -> Self {
        Self { config, device_selector: selector, initialised: false }
    }

    /// Returns the instance configuration.
    pub fn config(&self) -> &InstanceConfig {
        &self.config
    }

    /// Returns the device selector.
    pub fn device_selector(&self) -> &DeviceSelector {
        &self.device_selector
    }

    /// Whether the backend has been successfully initialised.
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }

    /// Attempt to initialise the Vulkan backend.
    ///
    /// Creates a Vulkan instance, selects a physical device, and creates
    /// a logical device with a compute queue.
    pub fn initialise(&mut self) -> error::Result<()> {
        let instance = VulkanInstance::new(&self.config)?;
        let selected = select_physical_device(instance.raw(), &self.device_selector)?;
        let (_device, _queue) = create_logical_device(instance.raw(), &selected)?;

        self.initialised = true;
        Ok(())
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_new_not_initialised() {
        let backend = VulkanBackend::new();
        assert!(!backend.is_initialised());
    }

    #[test]
    fn backend_default_config() {
        let backend = VulkanBackend::default();
        assert_eq!(backend.config().app_name, "bitnet-vulkan");
        assert!(backend.device_selector().prefer_discrete);
    }

    #[test]
    fn backend_with_custom_config() {
        let config = InstanceConfig {
            app_name: "test-app".into(),
            app_version: 1,
            enable_validation: false,
        };
        let selector = DeviceSelector { prefer_discrete: false, require_compute_queue: true };
        let backend = VulkanBackend::with_config(config, selector);
        assert_eq!(backend.config().app_name, "test-app");
        assert!(backend.device_selector().require_compute_queue);
    }
}
/// Provides Vulkan compute shaders (GLSL 450) for GPU-accelerated operations.
/// Shaders are embedded at compile time; optional pre-compiled SPIR-V is
/// available with the `precompiled-spirv` feature when `glslc` is installed.
pub mod kernels;
