//! WGSL compute shaders for I2_S 2-bit quantized BitNet inference.
//!
//! This crate provides GPU shader source strings for:
//! - **Dequantization**: Unpack 2-bit signed values to f32
//! - **Matrix-vector multiply**: Fused dequant + dot-product kernels
//! - **Quantization**: f32 → 2-bit signed with absmax scaling

pub mod dequantize;
pub mod matvec;
pub mod quantize;
