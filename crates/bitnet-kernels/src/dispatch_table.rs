//! Kernel dispatch table for operation routing.
//!
//! Maps operations to their best available kernel implementations.

use std::collections::HashMap;

/// Supported kernel operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    MatMul,
    MatMulI2S,
    MatMulQK256,
    LayerNorm,
    RmsNorm,
    Softmax,
    SiLU,
    ReLU,
    RoPE,
    Embedding,
    Attention,
    Quantize,
    Dequantize,
}

impl KernelOp {
    pub fn all() -> &'static [KernelOp] {
        &[
            Self::MatMul,
            Self::MatMulI2S,
            Self::MatMulQK256,
            Self::LayerNorm,
            Self::RmsNorm,
            Self::Softmax,
            Self::SiLU,
            Self::ReLU,
            Self::RoPE,
            Self::Embedding,
            Self::Attention,
            Self::Quantize,
            Self::Dequantize,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::MatMul => "matmul",
            Self::MatMulI2S => "matmul_i2s",
            Self::MatMulQK256 => "matmul_qk256",
            Self::LayerNorm => "layer_norm",
            Self::RmsNorm => "rms_norm",
            Self::Softmax => "softmax",
            Self::SiLU => "silu",
            Self::ReLU => "relu",
            Self::RoPE => "rope",
            Self::Embedding => "embedding",
            Self::Attention => "attention",
            Self::Quantize => "quantize",
            Self::Dequantize => "dequantize",
        }
    }
}

/// Backend for a kernel implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Cuda,
    Metal,
    OpenCL,
}

impl DispatchBackend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::OpenCL => "opencl",
        }
    }

    pub fn is_simd(&self) -> bool {
        matches!(self, Self::Avx2 | Self::Avx512 | Self::Neon)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::Metal | Self::OpenCL)
    }
}

/// A kernel implementation entry.
#[derive(Debug, Clone)]
pub struct DispatchEntry {
    pub op: KernelOp,
    pub backend: DispatchBackend,
    pub priority: u32,
    pub available: bool,
}

/// Kernel dispatch table.
#[derive(Debug, Clone)]
pub struct DispatchTable {
    entries: Vec<DispatchEntry>,
    overrides: HashMap<KernelOp, DispatchBackend>,
}

impl Default for DispatchTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchTable {
    pub fn new() -> Self {
        Self { entries: Vec::new(), overrides: HashMap::new() }
    }

    /// Register a kernel implementation.
    pub fn register(
        &mut self,
        op: KernelOp,
        backend: DispatchBackend,
        priority: u32,
        available: bool,
    ) {
        self.entries.push(DispatchEntry { op, backend, priority, available });
    }

    /// Force a specific backend for an operation.
    pub fn override_backend(&mut self, op: KernelOp, backend: DispatchBackend) {
        self.overrides.insert(op, backend);
    }

    /// Resolve the best backend for an operation.
    pub fn resolve(&self, op: KernelOp) -> Option<DispatchBackend> {
        // Check overrides first
        if let Some(&backend) = self.overrides.get(&op) {
            if self.entries.iter().any(|e| e.op == op && e.backend == backend && e.available) {
                return Some(backend);
            }
        }

        // Find highest-priority available entry
        self.entries
            .iter()
            .filter(|e| e.op == op && e.available)
            .max_by_key(|e| e.priority)
            .map(|e| e.backend)
    }

    /// List all available backends for an operation.
    pub fn available_backends(&self, op: KernelOp) -> Vec<DispatchBackend> {
        self.entries.iter().filter(|e| e.op == op && e.available).map(|e| e.backend).collect()
    }

    /// Total registered entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Operations with no available backend.
    pub fn unsupported_ops(&self) -> Vec<KernelOp> {
        KernelOp::all().iter().filter(|&&op| self.resolve(op).is_none()).copied().collect()
    }

    /// Build default CPU dispatch table.
    pub fn cpu_defaults() -> Self {
        let mut t = Self::new();
        for &op in KernelOp::all() {
            t.register(op, DispatchBackend::Scalar, 1, true);
        }
        // AVX2 upgrades (higher priority)
        for &op in &[
            KernelOp::MatMul,
            KernelOp::MatMulI2S,
            KernelOp::Softmax,
            KernelOp::LayerNorm,
            KernelOp::RmsNorm,
        ] {
            t.register(op, DispatchBackend::Avx2, 10, cfg!(target_arch = "x86_64"));
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_all() {
        let ops = KernelOp::all();
        assert_eq!(ops.len(), 13);
    }

    #[test]
    fn test_op_names() {
        assert_eq!(KernelOp::MatMul.name(), "matmul");
        assert_eq!(KernelOp::SiLU.name(), "silu");
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(DispatchBackend::Avx2.name(), "avx2");
        assert_eq!(DispatchBackend::Cuda.name(), "cuda");
    }

    #[test]
    fn test_backend_is_simd() {
        assert!(DispatchBackend::Avx2.is_simd());
        assert!(!DispatchBackend::Scalar.is_simd());
        assert!(!DispatchBackend::Cuda.is_simd());
    }

    #[test]
    fn test_backend_is_gpu() {
        assert!(DispatchBackend::Cuda.is_gpu());
        assert!(DispatchBackend::Metal.is_gpu());
        assert!(!DispatchBackend::Scalar.is_gpu());
    }

    #[test]
    fn test_register_and_resolve() {
        let mut t = DispatchTable::new();
        t.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
        assert_eq!(t.resolve(KernelOp::MatMul), Some(DispatchBackend::Scalar));
    }

    #[test]
    fn test_priority_resolution() {
        let mut t = DispatchTable::new();
        t.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
        t.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
        assert_eq!(t.resolve(KernelOp::MatMul), Some(DispatchBackend::Avx2));
    }

    #[test]
    fn test_unavailable_skipped() {
        let mut t = DispatchTable::new();
        t.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
        t.register(KernelOp::MatMul, DispatchBackend::Cuda, 100, false);
        assert_eq!(t.resolve(KernelOp::MatMul), Some(DispatchBackend::Scalar));
    }

    #[test]
    fn test_override() {
        let mut t = DispatchTable::new();
        t.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
        t.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
        t.override_backend(KernelOp::MatMul, DispatchBackend::Scalar);
        assert_eq!(t.resolve(KernelOp::MatMul), Some(DispatchBackend::Scalar));
    }

    #[test]
    fn test_override_unavailable() {
        let mut t = DispatchTable::new();
        t.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
        t.override_backend(KernelOp::MatMul, DispatchBackend::Cuda);
        // Cuda not registered as available, falls back to priority
        assert_eq!(t.resolve(KernelOp::MatMul), Some(DispatchBackend::Scalar));
    }

    #[test]
    fn test_available_backends() {
        let mut t = DispatchTable::new();
        t.register(KernelOp::MatMul, DispatchBackend::Scalar, 1, true);
        t.register(KernelOp::MatMul, DispatchBackend::Avx2, 10, true);
        t.register(KernelOp::MatMul, DispatchBackend::Cuda, 100, false);
        let backends = t.available_backends(KernelOp::MatMul);
        assert_eq!(backends.len(), 2);
    }

    #[test]
    fn test_unsupported_ops() {
        let t = DispatchTable::new();
        let unsupported = t.unsupported_ops();
        assert_eq!(unsupported.len(), 13); // all unsupported
    }

    #[test]
    fn test_cpu_defaults() {
        let t = DispatchTable::cpu_defaults();
        assert!(t.entry_count() > 0);
        // All ops have at least scalar
        for &op in KernelOp::all() {
            assert!(t.resolve(op).is_some(), "{:?} has no backend", op);
        }
    }

    #[test]
    fn test_entry_count() {
        let mut t = DispatchTable::new();
        assert_eq!(t.entry_count(), 0);
        t.register(KernelOp::SiLU, DispatchBackend::Scalar, 1, true);
        assert_eq!(t.entry_count(), 1);
    }

    #[test]
    fn test_no_resolve_empty() {
        let t = DispatchTable::new();
        assert_eq!(t.resolve(KernelOp::MatMul), None);
    }

    #[test]
    fn test_default_trait() {
        let t = DispatchTable::default();
        assert_eq!(t.entry_count(), 0);
    }
}
