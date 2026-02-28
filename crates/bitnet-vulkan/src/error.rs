//! Error types for the Vulkan compute backend.

use thiserror::Error;

/// Errors that can occur in the Vulkan backend.
#[derive(Debug, Error)]
pub enum VulkanError {
    /// Vulkan instance creation failed.
    #[error("failed to create Vulkan instance: {0}")]
    InstanceCreation(String),

    /// No suitable physical device found.
    #[error("no suitable Vulkan physical device found: {0}")]
    NoSuitableDevice(String),

    /// Logical device creation failed.
    #[error("failed to create logical device: {0}")]
    DeviceCreation(String),

    /// Compute pipeline creation failed.
    #[error("failed to create compute pipeline: {0}")]
    PipelineCreation(String),

    /// Shader module compilation/loading failed.
    #[error("failed to load SPIR-V shader: {0}")]
    ShaderLoad(String),

    /// Buffer allocation failed.
    #[error("buffer allocation failed: {0}")]
    BufferAllocation(String),

    /// Memory type not found for the requested properties.
    #[error("no suitable memory type found: {0}")]
    NoSuitableMemoryType(String),

    /// Command buffer recording or submission error.
    #[error("command buffer error: {0}")]
    CommandBuffer(String),

    /// Queue submission or synchronization error.
    #[error("queue submission error: {0}")]
    QueueSubmit(String),

    /// Raw Vulkan API error code.
    #[error("Vulkan API error: {0}")]
    VkResult(ash::vk::Result),
}

/// Convenience result type for the Vulkan backend.
pub type Result<T> = std::result::Result<T, VulkanError>;

impl From<ash::vk::Result> for VulkanError {
    fn from(result: ash::vk::Result) -> Self {
        VulkanError::VkResult(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = VulkanError::InstanceCreation("missing layer".into());
        assert!(err.to_string().contains("missing layer"));

        let err = VulkanError::NoSuitableDevice("no discrete GPU".into());
        assert!(err.to_string().contains("no discrete GPU"));

        let err = VulkanError::VkResult(ash::vk::Result::ERROR_DEVICE_LOST);
        assert!(err.to_string().contains("Vulkan API error"));
    }

    #[test]
    fn error_from_vk_result() {
        let err: VulkanError = ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.into();
        match err {
            VulkanError::VkResult(r) => {
                assert_eq!(r, ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
            }
            _ => panic!("expected VkResult variant"),
        }
    }
}
