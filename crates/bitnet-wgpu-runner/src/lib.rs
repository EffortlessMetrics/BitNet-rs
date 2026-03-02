//! Kernel execution runner for wgpu compute pipelines.
//!
//! Provides GPU shader compilation, buffer management, and dispatch
//! orchestration for BitNet compute kernels using the wgpu API.

pub mod error;
pub mod matmul;
pub mod runner;

pub use bitnet_matmul_ref_core::cpu_matmul;
pub use error::RunnerError;
pub use matmul::MatmulRunner;
pub use runner::{CompiledKernel, KernelRunner};
