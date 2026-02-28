//! Vulkan compute backend for BitNet-rs inference.
//!
//! Provides Vulkan compute shaders (GLSL 450) for GPU-accelerated operations.
//! Shaders are embedded at compile time; optional pre-compiled SPIR-V is
//! available with the `precompiled-spirv` feature when `glslc` is installed.

pub mod kernels;
