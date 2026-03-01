//! Kernel registry mapping names to WGSL shader source strings.
//!
//! Provides compile-time access to all WGSL compute shaders and a runtime
//! registry for looking them up by name.

mod registry;

pub use registry::{KernelEntry, KernelRegistry, REGISTRY};

// WGSL shader source constants — embedded at compile time.
pub const MATMUL_WGSL: &str = include_str!("matmul.wgsl");
pub const SOFTMAX_WGSL: &str = include_str!("softmax.wgsl");
pub const ATTENTION_WGSL: &str = include_str!("attention.wgsl");
pub const RMSNORM_WGSL: &str = include_str!("rmsnorm.wgsl");
