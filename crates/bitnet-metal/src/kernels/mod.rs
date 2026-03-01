//! Embedded Metal Shading Language (MSL) kernel sources for Apple Silicon.
//!
//! Each kernel is embedded at compile time via `include_str!` for portable
//! distribution without runtime file dependencies.

/// Identifies a Metal compute kernel source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalKernelSource {
    /// Matrix multiplication (naive + tiled variants).
    Matmul,
    /// Numerically stable softmax with threadgroup reduction.
    Softmax,
    /// RMS normalization with threadgroup reduction.
    RmsNorm,
    /// Rotary position embeddings (`RoPE`) + table builder.
    Rope,
    /// Scaled dot-product attention with optional causal mask.
    Attention,
    /// Element-wise ops: add, mul, `SiLU`, GELU, `silu_mul`, `scalar_mul`.
    Elementwise,
}

impl MetalKernelSource {
    /// Returns all kernel source variants.
    pub const ALL: &[Self] = &[
        Self::Matmul,
        Self::Softmax,
        Self::RmsNorm,
        Self::Rope,
        Self::Attention,
        Self::Elementwise,
    ];
}

/// Returns the embedded MSL source code for the given kernel.
pub const fn kernel_source(kernel: MetalKernelSource) -> &'static str {
    match kernel {
        MetalKernelSource::Matmul => {
            include_str!("matmul.metal")
        }
        MetalKernelSource::Softmax => {
            include_str!("softmax.metal")
        }
        MetalKernelSource::RmsNorm => {
            include_str!("rmsnorm.metal")
        }
        MetalKernelSource::Rope => {
            include_str!("rope.metal")
        }
        MetalKernelSource::Attention => {
            include_str!("attention.metal")
        }
        MetalKernelSource::Elementwise => {
            include_str!("elementwise.metal")
        }
    }
}

/// Returns the primary kernel function name(s) for each source.
pub const fn kernel_function_names(kernel: MetalKernelSource) -> &'static [&'static str] {
    match kernel {
        MetalKernelSource::Matmul => &["matmul", "matmul_tiled"],
        MetalKernelSource::Softmax => &["softmax"],
        MetalKernelSource::RmsNorm => &["rmsnorm"],
        MetalKernelSource::Rope => &["rope", "rope_build_tables"],
        MetalKernelSource::Attention => &["attention_scores", "attention_weighted_sum"],
        MetalKernelSource::Elementwise => &["add", "mul", "silu", "gelu", "silu_mul", "scalar_mul"],
    }
}
