//! Optimized `I2_S` quantization and dequantization for CPU inference.
//!
//! This crate implements 2-bit signed (`I2_S`) quantization in two block formats:
//!
//! - **BitNet32-F16**: 32-element blocks with inline F16 scales (10 bytes/block)
//! - **QK256**: 256-element blocks without per-block scales (64 bytes/block)
//!
//! # Ternary code mapping
//!
//! The 2-bit codes map to signed weights:
//! - Code 0 maps to weight -1
//! - Code 1 maps to weight  0
//! - Code 2 maps to weight +1
//!
//! Code 3 is unused in standard `I2_S` ternary and saturates to +1.
//!
//! # SIMD acceleration
//!
//! Pack/unpack paths detect AVX2 (`x86_64`) and NEON (`aarch64`) at compile
//! time and dispatch to vectorised implementations where available, with a
//! portable scalar fallback.

mod error;
mod format;
mod pack;
mod quantize;
mod simd;

pub use error::I2SError;
pub use format::{BitNet32Block, BlockFormat, Qk256Block};
pub use pack::{pack_i2s, unpack_i2s};
pub use quantize::{
    QuantizeOpts, QuantizedTensor, dequantize_batch, dequantize_f32, quantize_batch, quantize_f32,
};
