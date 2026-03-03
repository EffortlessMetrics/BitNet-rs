//! CUDA tensor transpose operations for `BitNet` LLM inference.
//!
//! Provides efficient transpose primitives with shared-memory tiled
//! implementations for coalesced GPU memory access. All GPU-specific
//! code is gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//!
//! ## Operations
//!
//! - **2D transpose** – swap rows and columns of a matrix
//! - **Batched transpose** – transpose each matrix in a batch independently
//! - **Permute dimensions** – arbitrary axis reordering for N-D tensors
//! - **Contiguous conversion** – rewrite data into contiguous (row-major) layout
//! - **In-place transpose** – zero-copy transpose for square matrices
//!
//! ## CPU fallback
//!
//! When compiled without GPU features the crate provides identical
//! semantics via scalar CPU implementations so that tests and downstream
//! consumers can run on any host.
//!
//! ## Example
//!
//! ```
//! use bitnet_cuda_transpose::{TransposeDesc, transpose_2d};
//!
//! // 2×3 matrix stored row-major
//! let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let desc = TransposeDesc::new(2, 3);
//! let dst = transpose_2d(&src, &desc);
//! // Result is 3×2:
//! assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//! ```

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

mod permute;
mod tile;
mod transpose;

pub use permute::{PermuteDesc, contiguous_copy, permute_dims};
pub use tile::TileConfig;
pub use transpose::{
    BatchTransposeDesc, TransposeDesc, batched_transpose_2d, transpose_2d, transpose_2d_in_place,
};

// ---------------------------------------------------------------------------
// GPU kernel stubs (behind feature gate)
// ---------------------------------------------------------------------------

/// CUDA shared-memory tiled transpose kernel launcher.
///
/// When `gpu` or `cuda` is enabled this module exposes the device-side
/// launch helpers. On CPU-only builds the module is absent.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod cuda {
    use crate::{BatchTransposeDesc, TileConfig, TransposeDesc};

    /// Launch a 2-D transpose kernel on the current CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the CUDA runtime is unavailable or the launch
    /// parameters are invalid.
    pub fn launch_transpose_2d(
        _src: &[f32],
        _dst: &mut [f32],
        _desc: &TransposeDesc,
        _tile: &TileConfig,
    ) -> Result<(), String> {
        // Kernel launch will be wired once the CUDA driver crate lands.
        Err("CUDA runtime not linked – compile with nvcc support".into())
    }

    /// Launch a batched 2-D transpose kernel on the current CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the CUDA runtime is unavailable or the launch
    /// parameters are invalid.
    pub fn launch_batched_transpose_2d(
        _src: &[f32],
        _dst: &mut [f32],
        _desc: &BatchTransposeDesc,
        _tile: &TileConfig,
    ) -> Result<(), String> {
        Err("CUDA runtime not linked – compile with nvcc support".into())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
