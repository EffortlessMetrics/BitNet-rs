//! `bitnet-rocm` — HIP compute kernels for AMD GPU inference via ROCm.
//!
//! This crate embeds HIP kernel source files (`.hip`) that target AMD GPUs
//! through the ROCm platform.  The kernels cover the core operations needed
//! for BitNet transformer inference:
//!
//! | Kernel        | File               | Operations                        |
//! |---------------|--------------------|-----------------------------------|
//! | Matrix mul    | `matmul.hip`       | Tiled I2S matmul, simple matmul   |
//! | Softmax       | `softmax.hip`      | Numerically-stable row softmax    |
//! | RMSNorm       | `rmsnorm.hip`      | RMS normalisation                 |
//! | RoPE          | `rope.hip`         | Rotary position embeddings        |
//! | Attention     | `attention.hip`    | Scaled dot-product attention      |
//! | Element-wise  | `elementwise.hip`  | SiLU, GELU, add, mul             |
//!
//! # Usage
//!
//! ```rust
//! use bitnet_rocm::kernels::{HipKernelSource, kernel_source};
//!
//! let src = kernel_source(HipKernelSource::Matmul);
//! assert!(src.contains("__global__"));
//! ```

pub mod kernels;
