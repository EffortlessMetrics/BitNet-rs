//! CUDA embedding lookup kernel stub.
//!
//! Token embedding lookup maps discrete token IDs to dense vectors.
//! The fused CUDA kernel combines table lookup, position encoding,
//! and write-back in a single pass.
//!
//! The CPU fallback and all types live in [`crate::embedding`]; this
//! module re-exports them for consistency with the `cuda::` namespace.

pub use crate::embedding::*;
