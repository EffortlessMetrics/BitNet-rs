//! Vulkan GLSL compute shader sources for GPU inference kernels.
//!
//! Each variant of [`VulkanShaderSource`] maps to a `.comp` GLSL file
//! embedded at compile time via [`include_str!`].

/// Enumerates the available Vulkan compute shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VulkanShaderSource {
    /// Tiled matrix multiplication with shared memory.
    Matmul,
    /// Numerically stable softmax with subgroup reduction.
    Softmax,
    /// RMS normalization with subgroup ops.
    RmsNorm,
    /// Rotary position embeddings.
    Rope,
    /// Scaled dot-product attention.
    Attention,
    /// Element-wise ops: `SiLU`, GELU, add, mul.
    Elementwise,
}

impl VulkanShaderSource {
    /// All shader variants in declaration order.
    pub const ALL: &[Self] = &[
        Self::Matmul,
        Self::Softmax,
        Self::RmsNorm,
        Self::Rope,
        Self::Attention,
        Self::Elementwise,
    ];

    /// Returns the raw GLSL source code for this shader.
    pub const fn glsl_source(&self) -> &'static str {
        match self {
            Self::Matmul => include_str!("matmul.comp"),
            Self::Softmax => include_str!("softmax.comp"),
            Self::RmsNorm => include_str!("rmsnorm.comp"),
            Self::Rope => include_str!("rope.comp"),
            Self::Attention => include_str!("attention.comp"),
            Self::Elementwise => include_str!("elementwise.comp"),
        }
    }

    /// The SPIR-V entry point name (always `"main"` per Vulkan convention).
    pub const fn entry_point(&self) -> &'static str {
        "main"
    }

    /// A human-readable name for logging / diagnostics.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Softmax => "softmax",
            Self::RmsNorm => "rmsnorm",
            Self::Rope => "rope",
            Self::Attention => "attention",
            Self::Elementwise => "elementwise",
        }
    }
}
