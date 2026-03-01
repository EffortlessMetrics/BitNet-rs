//! Embedded HIP kernel sources for AMD ROCm compute.
//!
//! Each variant of [`HipKernelSource`] maps to a `.hip` file compiled
//! at runtime by the ROCm HIP-RTC compiler on the target machine.

/// Static kernel source strings, embedded at compile time.
pub const MATMUL_SRC: &str = include_str!("matmul.hip");
pub const SOFTMAX_SRC: &str = include_str!("softmax.hip");
pub const RMSNORM_SRC: &str = include_str!("rmsnorm.hip");
pub const ROPE_SRC: &str = include_str!("rope.hip");
pub const ATTENTION_SRC: &str = include_str!("attention.hip");
pub const ELEMENTWISE_SRC: &str = include_str!("elementwise.hip");

/// Enumeration of available HIP kernel sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HipKernelSource {
    /// Tiled I2S matrix multiplication.
    Matmul,
    /// Numerically-stable row-wise softmax.
    Softmax,
    /// RMS normalisation with learnable scale.
    RmsNorm,
    /// Rotary position embeddings (forward + table build).
    Rope,
    /// Scaled dot-product attention.
    Attention,
    /// Element-wise ops: SiLU, GELU, add, mul.
    Elementwise,
}

impl HipKernelSource {
    /// Returns the embedded HIP source code for the kernel.
    #[must_use]
    pub fn source(self) -> &'static str {
        match self {
            Self::Matmul => MATMUL_SRC,
            Self::Softmax => SOFTMAX_SRC,
            Self::RmsNorm => RMSNORM_SRC,
            Self::Rope => ROPE_SRC,
            Self::Attention => ATTENTION_SRC,
            Self::Elementwise => ELEMENTWISE_SRC,
        }
    }

    /// All kernel source variants.
    pub const ALL: &[HipKernelSource] = &[
        Self::Matmul,
        Self::Softmax,
        Self::RmsNorm,
        Self::Rope,
        Self::Attention,
        Self::Elementwise,
    ];
}

/// Convenience function returning the source for a given kernel.
#[must_use]
pub fn kernel_source(kernel: HipKernelSource) -> &'static str {
    kernel.source()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_return_non_empty_source() {
        for &k in HipKernelSource::ALL {
            assert!(!k.source().is_empty(), "{k:?} returned empty source");
        }
    }
}
