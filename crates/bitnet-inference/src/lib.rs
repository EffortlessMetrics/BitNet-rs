//! Inference engines for BitNet models
//!
//! This crate provides high-performance inference engines for BitNet models with support
//! for CPU and GPU backends, streaming generation, and comprehensive configuration.

pub mod engine;
pub mod cpu;
pub mod gpu;
pub mod streaming;
pub mod batch;
pub mod sampling;
pub mod cache;
pub mod backend;
pub mod validation;

pub use engine::*;
pub use cpu::{CpuBackend, CpuInferenceEngine, CpuPerformanceConfig, CpuInferenceConfig};
pub use gpu::{GpuBackend, GpuInferenceEngine, GpuPerformanceConfig, GpuInferenceConfig, GpuMemoryStats};
pub use streaming::*;
pub use batch::*;
pub use sampling::*;
pub use cache::{KVCache, LayerCache, CacheMemoryStats};
pub use backend::*;
pub use validation::*;