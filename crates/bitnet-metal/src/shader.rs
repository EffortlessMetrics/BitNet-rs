//! Inline MSL kernel sources.

/// Matrix multiplication MSL kernel.
pub const MATMUL_MSL: &str = include_str!("kernels/matmul.metal");

/// Row-wise softmax MSL kernel.
pub const SOFTMAX_MSL: &str = include_str!("kernels/softmax.metal");

/// RMS normalization MSL kernel.
pub const RMSNORM_MSL: &str = include_str!("kernels/rmsnorm.metal");
