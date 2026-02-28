//! OpenCL backend for BitNet GPU inference.
//!
//! Provides GPU warmup and JIT compilation at startup for Intel and
//! other OpenCL-capable GPUs.

pub mod warmup;
