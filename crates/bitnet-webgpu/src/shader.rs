//! WGSL shader source constants.

/// Matrix multiplication compute shader.
pub const MATMUL_WGSL: &str = include_str!("shaders/matmul.wgsl");

/// Softmax compute shader.
pub const SOFTMAX_WGSL: &str = include_str!("shaders/softmax.wgsl");

/// Element-wise operations compute shader (add, mul, ReLU, SiLU).
pub const ELEMENTWISE_WGSL: &str = include_str!("shaders/elementwise.wgsl");
