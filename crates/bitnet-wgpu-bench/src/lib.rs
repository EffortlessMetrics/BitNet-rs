//! Benchmarking and profiling harness for wgpu compute kernels.
//!
//! Provides receipt-based regression tracking, workgroup tuning grids,
//! and regression detection — all without requiring a live GPU.

pub mod receipt;
pub mod regression;
pub mod workgroup_tuner;
