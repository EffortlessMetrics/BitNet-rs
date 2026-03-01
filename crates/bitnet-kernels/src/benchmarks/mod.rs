//! Benchmark definitions for GPU kernel performance testing.
//!
//! Includes scenarios for Intel Arc A770, CUDA, and CPU baselines.

pub mod intel_arc;

pub use intel_arc::*;
