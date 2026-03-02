//! Vulkan support surface for `BitNet` GPU inference.
//!
//! `bitnet-vulkan` now focuses on Vulkan runtime integration points and
//! re-exports embedded shader sources from `bitnet-vulkan-shaders`.

pub use bitnet_vulkan_shaders::VulkanShaderSource;

/// Backward-compatible kernel module re-export.
pub mod kernels {
    pub use bitnet_vulkan_shaders::VulkanShaderSource;
}

pub mod runtime;
