//! Inline MSL kernel sources.

/// Matrix multiplication MSL kernel.
pub const MATMUL_MSL: &str = include_str!("kernels/matmul.metal");

/// Row-wise softmax MSL kernel.
pub const SOFTMAX_MSL: &str = include_str!("kernels/softmax.metal");

/// RMS normalization MSL kernel.
pub const RMSNORM_MSL: &str = include_str!("kernels/rmsnorm.metal");

/// Rotary position embedding MSL kernel.
pub const ROPE_MSL: &str = include_str!("kernels/rope.metal");

/// Scaled dot-product attention MSL kernel.
pub const ATTENTION_MSL: &str = include_str!("kernels/attention.metal");

/// All available Metal kernel sources.
pub const ALL_KERNELS: &[(&str, &str)] = &[
    ("matmul", MATMUL_MSL),
    ("softmax", SOFTMAX_MSL),
    ("rmsnorm", RMSNORM_MSL),
    ("rope", ROPE_MSL),
    ("attention", ATTENTION_MSL),
];

/// Look up a kernel source by name.
pub fn get_kernel_source(name: &str) -> Option<&'static str> {
    ALL_KERNELS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, src)| *src)
}
