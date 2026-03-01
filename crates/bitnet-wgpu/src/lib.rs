//! `bitnet-wgpu` — wgpu-based GPU compute backend for BitNet inference.
//!
//! This crate provides the foundational infrastructure for running BitNet
//! inference on Vulkan, Metal, and DX12 via [wgpu](https://wgpu.rs/):
//!
//! - **Device management** — adapter discovery, device creation, capability queries
//! - **Buffer management** — typed GPU buffers with pool-based allocation
//! - **Pipeline infrastructure** — shader compilation and pipeline caching
//! - **Dispatch helpers** — workgroup sizing with NVIDIA-tuned heuristics
//!
//! # Status
//!
//! Foundation crate only — no compute shaders yet.  This establishes the seam
//! for future kernel implementations (matmul, quantization, attention).

pub mod buffer;
pub mod device;
pub mod dispatch;
pub mod error;
pub mod pipeline;

pub use buffer::{BufferPool, GpuBuffer, PoolStats};
pub use device::{DeviceInfo, WgpuDevice, WgpuDeviceConfig};
pub use dispatch::{
    DispatchConfig, DispatchEntry, DispatchRecorder, compute_dispatch_size,
    optimal_workgroup_size_nvidia,
};
pub use error::WgpuError;
pub use pipeline::{CacheStats, ComputePipeline, PipelineCache};
