//! Vulkan GLSL compute shaders for `BitNet` GPU inference.
//!
//! This crate provides embedded GLSL compute shader sources for the
//! core neural-network operations needed by the `BitNet` inference engine:
//! matrix multiplication, softmax, RMS normalization, rotary position
//! embeddings, scaled dot-product attention, and element-wise activations.
//!
//! Shaders target Vulkan 1.0 (`#version 450`) and use tiled shared-memory
//! algorithms and subgroup operations where applicable.

pub mod kernels;
