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
//! WGSL compute shader sources for WebGPU inference.
//!
//! All `.wgsl` files are embedded via [`include_str!`] for portable
//! distribution. Use [`WgslKernelSource`] to select a kernel and
//! [`kernel_source`] to retrieve the WGSL text.

/// Embedded WGSL source for the tiled matrix multiplication kernel.
pub const MATMUL_WGSL: &str = include_str!("matmul.wgsl");

/// Embedded WGSL source for the numerically-stable softmax kernel.
pub const SOFTMAX_WGSL: &str = include_str!("softmax.wgsl");

/// Embedded WGSL source for the RMS normalization kernel.
pub const RMSNORM_WGSL: &str = include_str!("rmsnorm.wgsl");

/// Embedded WGSL source for the rotary position embeddings kernel.
pub const ROPE_WGSL: &str = include_str!("rope.wgsl");

/// Embedded WGSL source for the scaled dot-product attention kernel.
pub const ATTENTION_WGSL: &str = include_str!("attention.wgsl");

/// Embedded WGSL source for elementwise operations (add, mul, `SiLU`, GELU).
pub const ELEMENTWISE_WGSL: &str = include_str!("elementwise.wgsl");

/// All available WGSL compute kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgslKernelSource {
    /// Tiled matrix multiplication (C = A × B).
    Matmul,
    /// Numerically-stable row-wise softmax.
    Softmax,
    /// RMS layer normalization.
    RmsNorm,
    /// Rotary position embeddings applied to Q/K tensors.
    Rope,
    /// Fused scaled dot-product attention with causal mask.
    Attention,
    /// Elementwise ops: add, mul, `SiLU`, GELU.
    Elementwise,
}

impl WgslKernelSource {
    /// Return the full list of kernel variants.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Matmul,
            Self::Softmax,
            Self::RmsNorm,
            Self::Rope,
            Self::Attention,
            Self::Elementwise,
        ]
    }

    /// Name of this kernel (lowercase, suitable for logging).
    #[must_use]
    pub const fn name(self) -> &'static str {
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

/// Return the embedded WGSL source for the given kernel.
#[must_use]
pub const fn kernel_source(kernel: WgslKernelSource) -> &'static str {
    match kernel {
        WgslKernelSource::Matmul => MATMUL_WGSL,
        WgslKernelSource::Softmax => SOFTMAX_WGSL,
        WgslKernelSource::RmsNorm => RMSNORM_WGSL,
        WgslKernelSource::Rope => ROPE_WGSL,
        WgslKernelSource::Attention => ATTENTION_WGSL,
        WgslKernelSource::Elementwise => ELEMENTWISE_WGSL,
    }
}

/// Basic structural validation of a WGSL source string.
///
/// Returns a list of human-readable issues found. An empty vec means the
/// source passes all checks.
#[must_use]
pub fn validate_wgsl_structure(source: &str) -> Vec<String> {
    let mut issues = Vec::new();

    if source.trim().is_empty() {
        issues.push("source is empty".to_string());
        return issues;
    }

    if !source.contains("@compute") {
        issues.push("missing @compute entry point".to_string());
    }

    if !source.contains("@workgroup_size") {
        issues.push("missing @workgroup_size declaration".to_string());
    }

    // Check for balanced braces
    let open = source.chars().filter(|&c| c == '{').count();
    let close = source.chars().filter(|&c| c == '}').count();
    if open != close {
        issues.push(format!("unbalanced braces: {open} opening vs {close} closing"));
    }

    // Check for unresolved preprocessor-style includes
    if source.contains("#include") || source.contains("#import") {
        issues.push("contains unresolved #include/#import".to_string());
    }

    issues
}
