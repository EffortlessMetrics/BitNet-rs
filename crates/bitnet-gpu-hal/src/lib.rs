//! GPU hardware abstraction layer with CPU reference implementations.
//!
//! Provides CPU-based reference kernels (softmax, RMS norm, matmul, `RoPE`,
//! sampling) that serve as correctness baselines for GPU kernel development.
