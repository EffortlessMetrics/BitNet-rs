//! Embedded OpenCL kernel sources for optimized GPU operations.
//!
//! Kernel sources are compiled at runtime by the OpenCL driver.
//! Each constant contains the `.cl` source as a `&str` for embedding
//! into the Rust binary without filesystem access at runtime.

/// Vectorized matrix multiplication with local memory tiling and fma.
pub const MATMUL_VECTORIZED_SRC: &str = include_str!("matmul_vectorized.cl");

/// Numerically stable softmax with work-group parallel reduction.
pub const SOFTMAX_OPTIMIZED_SRC: &str = include_str!("softmax_optimized.cl");

/// Fused scaled dot-product attention (full + causal variants).
pub const ATTENTION_OPTIMIZED_SRC: &str = include_str!("attention_optimized.cl");

/// Default tile size used by the tiled matmul kernel.
pub const DEFAULT_TILE_SIZE: usize = 16;

/// Vector width for the float4 matmul variant.
pub const VECTOR_WIDTH: usize = 4;
