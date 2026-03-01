//! OpenCL kernel registry with device-aware kernel selection.
//!
//! Maps operation types to the best available kernel implementation based on
//! detected hardware capabilities (A770 vs other Intel GPUs vs CPU fallback).

use std::collections::BTreeSet;
use std::fmt;

// ---------------------------------------------------------------------------
// KernelOp — the operations the registry knows about
// ---------------------------------------------------------------------------

/// Operation types supported by the kernel registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KernelOp {
    MatMul,
    MatVec,
    Softmax,
    RmsNorm,
    LayerNorm,
    RoPE,
    Attention,
    SiLU,
    GELU,
    ReLU,
    ElementwiseAdd,
    ElementwiseMul,
    Scale,
    Embedding,
    Dequantize,
    KvCacheAppend,
}

impl KernelOp {
    /// All defined operation variants (useful for iteration).
    pub const ALL: &'static [KernelOp] = &[
        Self::MatMul,
        Self::MatVec,
        Self::Softmax,
        Self::RmsNorm,
        Self::LayerNorm,
        Self::RoPE,
        Self::Attention,
        Self::SiLU,
        Self::GELU,
        Self::ReLU,
        Self::ElementwiseAdd,
        Self::ElementwiseMul,
        Self::Scale,
        Self::Embedding,
        Self::Dequantize,
        Self::KvCacheAppend,
    ];
}

impl fmt::Display for KernelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "MatMul"),
            Self::MatVec => write!(f, "MatVec"),
            Self::Softmax => write!(f, "Softmax"),
            Self::RmsNorm => write!(f, "RmsNorm"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::RoPE => write!(f, "RoPE"),
            Self::Attention => write!(f, "Attention"),
            Self::SiLU => write!(f, "SiLU"),
            Self::GELU => write!(f, "GELU"),
            Self::ReLU => write!(f, "ReLU"),
            Self::ElementwiseAdd => write!(f, "ElementwiseAdd"),
            Self::ElementwiseMul => write!(f, "ElementwiseMul"),
            Self::Scale => write!(f, "Scale"),
            Self::Embedding => write!(f, "Embedding"),
            Self::Dequantize => write!(f, "Dequantize"),
            Self::KvCacheAppend => write!(f, "KvCacheAppend"),
        }
    }
}

// ---------------------------------------------------------------------------
// KernelVariant — implementation strategy
// ---------------------------------------------------------------------------

/// Implementation strategy for a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelVariant {
    /// Basic OpenCL kernel (any GPU).
    OpenClScalar,
    /// Tiled/optimised OpenCL kernel (needs SLM, large workgroups).
    OpenClTiled,
    /// float4/float8 vectorised OpenCL kernel.
    OpenClVectorized,
    /// CPU SIMD fallback (AVX2/NEON).
    CpuSimd,
    /// Basic CPU scalar fallback.
    CpuScalar,
}

impl KernelVariant {
    /// Returns `true` when the variant targets a GPU backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::OpenClScalar | Self::OpenClTiled | Self::OpenClVectorized)
    }

    /// Priority score — higher is preferred.
    pub fn priority(&self) -> u32 {
        match self {
            Self::OpenClTiled => 50,
            Self::OpenClVectorized => 40,
            Self::OpenClScalar => 30,
            Self::CpuSimd => 20,
            Self::CpuScalar => 10,
        }
    }
}

impl fmt::Display for KernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenClScalar => write!(f, "OpenCL-scalar"),
            Self::OpenClTiled => write!(f, "OpenCL-tiled"),
            Self::OpenClVectorized => write!(f, "OpenCL-vec"),
            Self::CpuSimd => write!(f, "CPU-SIMD"),
            Self::CpuScalar => write!(f, "CPU-scalar"),
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceConstraints
// ---------------------------------------------------------------------------

/// Hardware constraints that govern kernel eligibility.
#[derive(Debug, Clone)]
pub struct DeviceConstraints {
    /// Maximum work-group size supported by the device.
    pub max_workgroup_size: usize,
    /// Shared local memory (SLM) in bytes.
    pub max_local_memory: usize,
    /// Supported subgroup sizes (e.g., [8, 16, 32]).
    pub subgroup_sizes: Vec<usize>,
    /// Whether FP16 is natively supported.
    pub has_fp16: bool,
    /// Whether int8 dot-product instructions are supported.
    pub has_int8_dot: bool,
    /// Number of compute units (Xe-cores on Arc).
    pub compute_units: usize,
}

impl DeviceConstraints {
    /// Returns `true` when a tiled kernel with the given `tile_size` can fit in SLM.
    ///
    /// A tiled kernel requires at least `2 * tile_size * tile_size * 4` bytes
    /// of SLM (two tiles of f32 values).
    pub fn supports_tiled(&self, tile_size: usize) -> bool {
        let required = 2 * tile_size * tile_size * size_of::<f32>();
        self.max_local_memory >= required
    }

    /// Default constraints for the Intel Arc A770 GPU.
    pub fn a770_defaults() -> Self {
        Self {
            max_workgroup_size: 1024,
            max_local_memory: 65536, // 64 KiB
            subgroup_sizes: vec![8, 16, 32],
            has_fp16: true,
            has_int8_dot: true,
            compute_units: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// KernelRegistration
// ---------------------------------------------------------------------------

/// A single kernel registration in the registry.
#[derive(Debug, Clone)]
pub struct KernelRegistration {
    /// The operation this kernel implements.
    pub op: KernelOp,
    /// Implementation strategy.
    pub variant: KernelVariant,
    /// Minimum work-group size required by this kernel.
    pub min_workgroup: usize,
    /// Local (SLM) memory required in bytes.
    pub local_memory_bytes: usize,
    /// Key referencing the OpenCL source constant name.
    pub source_key: &'static str,
}

// ---------------------------------------------------------------------------
// KernelRegistry
// ---------------------------------------------------------------------------

/// Registry that maps operations to the best available kernel implementation
/// for the current device constraints.
#[derive(Debug, Clone)]
pub struct KernelRegistry {
    registrations: Vec<KernelRegistration>,
    constraints: DeviceConstraints,
}

impl KernelRegistry {
    /// Create an empty registry with the given device constraints.
    pub fn new(constraints: DeviceConstraints) -> Self {
        Self { registrations: Vec::new(), constraints }
    }

    /// Register a kernel implementation. Returns `&mut Self` for chaining.
    pub fn register(&mut self, registration: KernelRegistration) -> &mut Self {
        self.registrations.push(registration);
        self
    }

    /// Build a registry pre-populated with all standard A770 kernels.
    pub fn with_default_a770_kernels() -> Self {
        let constraints = DeviceConstraints::a770_defaults();
        let mut registry = Self::new(constraints);

        // Helper: register a full set of variants for a single op.
        macro_rules! reg {
            ($op:expr, $key:expr) => {
                registry.register(KernelRegistration {
                    op: $op,
                    variant: KernelVariant::OpenClTiled,
                    min_workgroup: 256,
                    local_memory_bytes: 8192,
                    source_key: concat!($key, "_tiled"),
                });
                registry.register(KernelRegistration {
                    op: $op,
                    variant: KernelVariant::OpenClVectorized,
                    min_workgroup: 64,
                    local_memory_bytes: 0,
                    source_key: concat!($key, "_vec"),
                });
                registry.register(KernelRegistration {
                    op: $op,
                    variant: KernelVariant::OpenClScalar,
                    min_workgroup: 1,
                    local_memory_bytes: 0,
                    source_key: concat!($key, "_scalar"),
                });
                registry.register(KernelRegistration {
                    op: $op,
                    variant: KernelVariant::CpuSimd,
                    min_workgroup: 1,
                    local_memory_bytes: 0,
                    source_key: concat!($key, "_cpu_simd"),
                });
                registry.register(KernelRegistration {
                    op: $op,
                    variant: KernelVariant::CpuScalar,
                    min_workgroup: 1,
                    local_memory_bytes: 0,
                    source_key: concat!($key, "_cpu_scalar"),
                });
            };
        }

        reg!(KernelOp::MatMul, "matmul");
        reg!(KernelOp::MatVec, "matvec");
        reg!(KernelOp::Softmax, "softmax");
        reg!(KernelOp::RmsNorm, "rmsnorm");
        reg!(KernelOp::LayerNorm, "layernorm");
        reg!(KernelOp::RoPE, "rope");
        reg!(KernelOp::Attention, "attention");
        reg!(KernelOp::SiLU, "silu");
        reg!(KernelOp::GELU, "gelu");
        reg!(KernelOp::ReLU, "relu");
        reg!(KernelOp::ElementwiseAdd, "eltadd");
        reg!(KernelOp::ElementwiseMul, "eltmul");
        reg!(KernelOp::Scale, "scale");
        reg!(KernelOp::Embedding, "embedding");
        reg!(KernelOp::Dequantize, "dequant");
        reg!(KernelOp::KvCacheAppend, "kvcache");

        registry
    }

    /// Select the best eligible kernel for `op`, or `None` if nothing is
    /// registered.
    pub fn select(&self, op: KernelOp) -> Option<&KernelRegistration> {
        self.registrations
            .iter()
            .filter(|r| r.op == op)
            .filter(|r| self.is_eligible(r))
            .max_by_key(|r| r.variant.priority())
    }

    /// Select the best eligible kernel for `op`, always returning *something*.
    ///
    /// If no registered kernel is eligible the method synthesises a `CpuScalar`
    /// fallback on the fly.  The returned reference is either into
    /// `self.registrations` or into a leaked `Box` so it has `'static`
    /// lifetime.
    pub fn select_with_fallback(&self, op: KernelOp) -> &KernelRegistration {
        if let Some(reg) = self.select(op) {
            return reg;
        }

        // Synthesise a CPU-scalar fallback.  We leak a Box so we can hand
        // back a `&KernelRegistration` with an arbitrary lifetime.
        let fallback = Box::new(KernelRegistration {
            op,
            variant: KernelVariant::CpuScalar,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "cpu_scalar_fallback",
        });
        Box::leak(fallback)
    }

    /// All distinct ops that have at least one eligible registration.
    pub fn available_ops(&self) -> Vec<KernelOp> {
        let set: BTreeSet<KernelOp> =
            self.registrations.iter().filter(|r| self.is_eligible(r)).map(|r| r.op).collect();
        set.into_iter().collect()
    }

    /// Fraction of `KernelOp::ALL` that have at least one eligible GPU
    /// implementation (0.0–1.0).
    pub fn gpu_coverage(&self) -> f64 {
        let gpu_ops: BTreeSet<KernelOp> = self
            .registrations
            .iter()
            .filter(|r| r.variant.is_gpu() && self.is_eligible(r))
            .map(|r| r.op)
            .collect();
        gpu_ops.len() as f64 / KernelOp::ALL.len() as f64
    }

    /// Human-readable summary of the registry state.
    pub fn summary(&self) -> String {
        let total_ops = KernelOp::ALL.len();
        let available = self.available_ops().len();
        let gpu_pct = self.gpu_coverage() * 100.0;

        let mut lines = vec![format!(
            "KernelRegistry: {available}/{total_ops} ops available, {gpu_pct:.0}% GPU coverage"
        )];

        for &op in KernelOp::ALL {
            let label = if let Some(reg) = self.select(op) {
                format!("{}", reg.variant)
            } else {
                "none".to_string()
            };
            lines.push(format!("  {op}: {label}"));
        }

        lines.join("\n")
    }

    // -- private helpers ----------------------------------------------------

    fn is_eligible(&self, reg: &KernelRegistration) -> bool {
        if reg.min_workgroup > self.constraints.max_workgroup_size {
            return false;
        }
        if reg.local_memory_bytes > self.constraints.max_local_memory {
            return false;
        }
        true
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- KernelOp Display ---------------------------------------------------

    #[test]
    fn display_matmul() {
        assert_eq!(KernelOp::MatMul.to_string(), "MatMul");
    }

    #[test]
    fn display_matvec() {
        assert_eq!(KernelOp::MatVec.to_string(), "MatVec");
    }

    #[test]
    fn display_softmax() {
        assert_eq!(KernelOp::Softmax.to_string(), "Softmax");
    }

    #[test]
    fn display_rmsnorm() {
        assert_eq!(KernelOp::RmsNorm.to_string(), "RmsNorm");
    }

    #[test]
    fn display_layernorm() {
        assert_eq!(KernelOp::LayerNorm.to_string(), "LayerNorm");
    }

    #[test]
    fn display_rope() {
        assert_eq!(KernelOp::RoPE.to_string(), "RoPE");
    }

    #[test]
    fn display_attention() {
        assert_eq!(KernelOp::Attention.to_string(), "Attention");
    }

    #[test]
    fn display_all_ops_non_empty() {
        for op in KernelOp::ALL {
            let s = op.to_string();
            assert!(!s.is_empty(), "Display for {op:?} must not be empty");
        }
    }

    // -- KernelVariant ------------------------------------------------------

    #[test]
    fn variant_is_gpu() {
        assert!(KernelVariant::OpenClScalar.is_gpu());
        assert!(KernelVariant::OpenClTiled.is_gpu());
        assert!(KernelVariant::OpenClVectorized.is_gpu());
        assert!(!KernelVariant::CpuSimd.is_gpu());
        assert!(!KernelVariant::CpuScalar.is_gpu());
    }

    #[test]
    fn variant_priority_ordering() {
        assert!(KernelVariant::OpenClTiled.priority() > KernelVariant::OpenClVectorized.priority());
        assert!(
            KernelVariant::OpenClVectorized.priority() > KernelVariant::OpenClScalar.priority()
        );
        assert!(KernelVariant::OpenClScalar.priority() > KernelVariant::CpuSimd.priority());
        assert!(KernelVariant::CpuSimd.priority() > KernelVariant::CpuScalar.priority());
    }

    #[test]
    fn variant_display() {
        assert_eq!(KernelVariant::OpenClTiled.to_string(), "OpenCL-tiled");
        assert_eq!(KernelVariant::CpuScalar.to_string(), "CPU-scalar");
    }

    // -- DeviceConstraints --------------------------------------------------

    #[test]
    fn a770_defaults_workgroup() {
        let c = DeviceConstraints::a770_defaults();
        assert_eq!(c.max_workgroup_size, 1024);
    }

    #[test]
    fn a770_defaults_slm() {
        let c = DeviceConstraints::a770_defaults();
        assert_eq!(c.max_local_memory, 65536);
    }

    #[test]
    fn a770_defaults_subgroups() {
        let c = DeviceConstraints::a770_defaults();
        assert_eq!(c.subgroup_sizes, vec![8, 16, 32]);
    }

    #[test]
    fn a770_defaults_fp16_and_int8() {
        let c = DeviceConstraints::a770_defaults();
        assert!(c.has_fp16);
        assert!(c.has_int8_dot);
    }

    #[test]
    fn a770_defaults_compute_units() {
        let c = DeviceConstraints::a770_defaults();
        assert_eq!(c.compute_units, 32);
    }

    #[test]
    fn supports_tiled_true_for_small_tile() {
        let c = DeviceConstraints::a770_defaults();
        // tile_size=16 → 2*16*16*4 = 2048 bytes, well within 64 KiB
        assert!(c.supports_tiled(16));
    }

    #[test]
    fn supports_tiled_false_for_huge_tile() {
        let c = DeviceConstraints::a770_defaults();
        // tile_size=256 → 2*256*256*4 = 524288, exceeds 64 KiB
        assert!(!c.supports_tiled(256));
    }

    #[test]
    fn supports_tiled_exact_boundary() {
        // 2 * t * t * 4 == max_local_memory ⇒ t = sqrt(max/8)
        // For 64 KiB: t = sqrt(8192) ≈ 90.5, so t=90 fits, t=91 doesn't.
        let c = DeviceConstraints::a770_defaults();
        assert!(c.supports_tiled(90));
        assert!(!c.supports_tiled(91));
    }

    // -- KernelRegistry: default A770 registry ------------------------------

    #[test]
    fn default_a770_registry_has_all_ops() {
        let reg = KernelRegistry::with_default_a770_kernels();
        let available = reg.available_ops();
        for op in KernelOp::ALL {
            assert!(available.contains(op), "missing op: {op}");
        }
    }

    #[test]
    fn default_a770_registry_prefers_gpu() {
        let reg = KernelRegistry::with_default_a770_kernels();
        for &op in KernelOp::ALL {
            let best = reg.select(op).expect("should have registration");
            assert!(best.variant.is_gpu(), "{op} should select a GPU variant");
        }
    }

    #[test]
    fn default_a770_selects_tiled_for_matmul() {
        let reg = KernelRegistry::with_default_a770_kernels();
        let best = reg.select(KernelOp::MatMul).unwrap();
        assert_eq!(best.variant, KernelVariant::OpenClTiled);
    }

    #[test]
    fn gpu_coverage_full_for_a770() {
        let reg = KernelRegistry::with_default_a770_kernels();
        assert!((reg.gpu_coverage() - 1.0).abs() < f64::EPSILON);
    }

    // -- Selection with constrained devices ---------------------------------

    #[test]
    fn tiled_rejected_when_slm_too_small() {
        let constraints = DeviceConstraints {
            max_workgroup_size: 1024,
            max_local_memory: 128, // tiny SLM
            subgroup_sizes: vec![16],
            has_fp16: false,
            has_int8_dot: false,
            compute_units: 8,
        };
        let mut reg = KernelRegistry::new(constraints);
        reg.register(KernelRegistration {
            op: KernelOp::MatMul,
            variant: KernelVariant::OpenClTiled,
            min_workgroup: 256,
            local_memory_bytes: 8192, // exceeds 128 SLM
            source_key: "matmul_tiled",
        });
        reg.register(KernelRegistration {
            op: KernelOp::MatMul,
            variant: KernelVariant::OpenClScalar,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "matmul_scalar",
        });

        let best = reg.select(KernelOp::MatMul).unwrap();
        assert_eq!(best.variant, KernelVariant::OpenClScalar);
    }

    #[test]
    fn tiled_rejected_when_workgroup_too_small() {
        let constraints = DeviceConstraints {
            max_workgroup_size: 64, // too small for 256
            max_local_memory: 65536,
            subgroup_sizes: vec![16],
            has_fp16: false,
            has_int8_dot: false,
            compute_units: 8,
        };
        let mut reg = KernelRegistry::new(constraints);
        reg.register(KernelRegistration {
            op: KernelOp::Softmax,
            variant: KernelVariant::OpenClTiled,
            min_workgroup: 256,
            local_memory_bytes: 4096,
            source_key: "softmax_tiled",
        });
        reg.register(KernelRegistration {
            op: KernelOp::Softmax,
            variant: KernelVariant::CpuSimd,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "softmax_cpu_simd",
        });

        let best = reg.select(KernelOp::Softmax).unwrap();
        assert_eq!(best.variant, KernelVariant::CpuSimd);
    }

    // -- Fallback -----------------------------------------------------------

    #[test]
    fn fallback_always_returns_result() {
        let reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        // Empty registry — no registrations at all.
        let fb = reg.select_with_fallback(KernelOp::MatMul);
        assert_eq!(fb.variant, KernelVariant::CpuScalar);
        assert_eq!(fb.op, KernelOp::MatMul);
    }

    #[test]
    fn fallback_returns_best_when_available() {
        let reg = KernelRegistry::with_default_a770_kernels();
        let fb = reg.select_with_fallback(KernelOp::Softmax);
        assert!(fb.variant.is_gpu());
    }

    // -- Empty / edge cases -------------------------------------------------

    #[test]
    fn empty_registry_select_returns_none() {
        let reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        assert!(reg.select(KernelOp::Attention).is_none());
    }

    #[test]
    fn empty_registry_available_ops_empty() {
        let reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        assert!(reg.available_ops().is_empty());
    }

    #[test]
    fn empty_registry_gpu_coverage_zero() {
        let reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        assert!((reg.gpu_coverage()).abs() < f64::EPSILON);
    }

    // -- Custom registration ------------------------------------------------

    #[test]
    fn custom_registration_overrides_default() {
        let mut reg = KernelRegistry::with_default_a770_kernels();
        // Register a higher-priority custom tiled kernel for MatMul.
        // Same variant (OpenClTiled) so it will be selected over the default
        // since both have the same priority; last-registered wins with
        // max_by_key because it preserves last-of-equals.
        // Instead, add a tiled kernel with lower min_workgroup to ensure
        // the new one is also eligible and actually selected.
        reg.register(KernelRegistration {
            op: KernelOp::MatMul,
            variant: KernelVariant::OpenClTiled,
            min_workgroup: 128,
            local_memory_bytes: 4096,
            source_key: "matmul_tiled_v2",
        });
        let best = reg.select(KernelOp::MatMul).unwrap();
        // Both have the same priority; max_by_key returns the last match.
        assert_eq!(best.source_key, "matmul_tiled_v2");
    }

    #[test]
    fn duplicate_op_takes_highest_priority() {
        let mut reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        reg.register(KernelRegistration {
            op: KernelOp::RoPE,
            variant: KernelVariant::CpuScalar,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "rope_cpu",
        });
        reg.register(KernelRegistration {
            op: KernelOp::RoPE,
            variant: KernelVariant::OpenClVectorized,
            min_workgroup: 64,
            local_memory_bytes: 0,
            source_key: "rope_vec",
        });
        let best = reg.select(KernelOp::RoPE).unwrap();
        assert_eq!(best.variant, KernelVariant::OpenClVectorized);
    }

    // -- Summary formatting -------------------------------------------------

    #[test]
    fn summary_contains_coverage() {
        let reg = KernelRegistry::with_default_a770_kernels();
        let s = reg.summary();
        assert!(s.contains("100% GPU coverage"), "summary: {s}");
    }

    #[test]
    fn summary_lists_ops() {
        let reg = KernelRegistry::with_default_a770_kernels();
        let s = reg.summary();
        assert!(s.contains("MatMul"));
        assert!(s.contains("Attention"));
        assert!(s.contains("KvCacheAppend"));
    }

    #[test]
    fn summary_empty_registry() {
        let reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        let s = reg.summary();
        assert!(s.contains("0/16 ops available"));
    }

    // -- gpu_coverage partial -----------------------------------------------

    #[test]
    fn gpu_coverage_partial() {
        let mut reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        // Register GPU only for MatMul.
        reg.register(KernelRegistration {
            op: KernelOp::MatMul,
            variant: KernelVariant::OpenClScalar,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "matmul_scalar",
        });
        let expected = 1.0 / KernelOp::ALL.len() as f64;
        assert!((reg.gpu_coverage() - expected).abs() < f64::EPSILON);
    }

    // -- Register chaining --------------------------------------------------

    #[test]
    fn register_chaining() {
        let mut reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        reg.register(KernelRegistration {
            op: KernelOp::SiLU,
            variant: KernelVariant::CpuScalar,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "silu_cpu",
        })
        .register(KernelRegistration {
            op: KernelOp::GELU,
            variant: KernelVariant::CpuScalar,
            min_workgroup: 1,
            local_memory_bytes: 0,
            source_key: "gelu_cpu",
        });
        assert_eq!(reg.available_ops().len(), 2);
    }

    // -- KernelOp::ALL completeness -----------------------------------------

    #[test]
    fn kernel_op_all_count() {
        assert_eq!(KernelOp::ALL.len(), 16);
    }

    // -- CPU-only coverage is zero ------------------------------------------

    #[test]
    fn cpu_only_registry_gpu_coverage_zero() {
        let mut reg = KernelRegistry::new(DeviceConstraints::a770_defaults());
        for &op in KernelOp::ALL {
            reg.register(KernelRegistration {
                op,
                variant: KernelVariant::CpuScalar,
                min_workgroup: 1,
                local_memory_bytes: 0,
                source_key: "cpu_scalar_fallback",
            });
        }
        assert!((reg.gpu_coverage()).abs() < f64::EPSILON);
    }
}
