//! ARM NEON optimized `LayerNorm`, `RMSNorm`, and `GroupNorm` operations.
//!
//! This crate provides high-performance normalization kernels that use
//! NEON SIMD intrinsics on `aarch64` targets and fall back to portable
//! scalar code everywhere else.
//!
//! ## Supported operations
//!
//! | Operation  | Description |
//! |------------|-------------|
//! | [`LayerNorm`] | Standard Layer Normalization (mean + variance) |
//! | [`RmsNorm`] | Root Mean Square Normalization (variance only) |
//! | [`GroupNorm`] | Group Normalization (channels split into groups) |
//!
//! Each operation supports:
//! - `f32` inputs via the primary API
//! - `f16` inputs via `*_f16` methods (using the [`half`] crate)
//! - Fused norm + scale + bias in a single pass
//!
//! ## Example
//!
//! ```
//! use bitnet_neon_layernorm::{LayerNorm, RmsNorm};
//!
//! let ln = LayerNorm::new(4, 1e-5);
//! let mut buf = [1.0f32, 2.0, 3.0, 4.0];
//! ln.forward(&mut buf);
//!
//! let rms = RmsNorm::new(4, 1e-5);
//! let mut buf2 = [1.0f32, 2.0, 3.0, 4.0];
//! rms.forward(&mut buf2);
//! ```

mod neon;
mod norm;
mod scalar;

pub use norm::{GroupNorm, LayerNorm, RmsNorm};

/// Compute `LayerNorm` in-place on `data` with the given `epsilon`.
///
/// This is a convenience wrapper around [`LayerNorm`].
pub fn layer_norm_inplace(data: &mut [f32], epsilon: f32) {
    let ln = LayerNorm::new(data.len(), epsilon);
    ln.forward(data);
}

/// Compute `RMSNorm` in-place on `data` with the given `epsilon`.
///
/// This is a convenience wrapper around [`RmsNorm`].
pub fn rms_norm_inplace(data: &mut [f32], epsilon: f32) {
    let rms = RmsNorm::new(data.len(), epsilon);
    rms.forward(data);
}
