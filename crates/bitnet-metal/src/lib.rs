//! Metal Shading Language (MSL) compute kernels for Apple Silicon GPU inference.
//!
//! This crate provides embedded MSL kernel sources for core neural network
//! operations. Kernels are compiled at runtime by the Metal framework on macOS.
//!
//! # Kernels
//!
//! - **matmul** — matrix multiplication (naive + tiled)
//! - **softmax** — numerically stable softmax with threadgroup reduction
//! - **rmsnorm** — RMS normalization with threadgroup reduction
//! - **rope** — rotary position embeddings + frequency table builder
//! - **attention** — scaled dot-product attention with causal mask
//! - **elementwise** — `SiLU`, GELU, add, mul, `silu_mul`, `scalar_mul`

pub mod kernels;

pub use kernels::{MetalKernelSource, kernel_function_names, kernel_source};
