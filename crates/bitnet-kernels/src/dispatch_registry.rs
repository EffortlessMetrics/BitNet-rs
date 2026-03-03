//! Kernel dispatch registry.
//!
//! Central registry of available kernel implementations, mapping
//! operation + backend → kernel function with capability checks.

use std::collections::HashMap;

/// Backend target for kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Cuda,
    OpenCl,
    Metal,
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Avx2 => "avx2",
            Backend::Avx512 => "avx512",
            Backend::Neon => "neon",
            Backend::Cuda => "cuda",
            Backend::OpenCl => "opencl",
            Backend::Metal => "metal",
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Backend::Scalar | Backend::Avx2 | Backend::Avx512 | Backend::Neon)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Backend::Cuda | Backend::OpenCl | Backend::Metal)
    }
}

/// Operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    Matmul,
    Attention,
    LayerNorm,
    RmsNorm,
    SiLU,
    ReLU,
    Softmax,
    RoPE,
    Embedding,
    Quantize,
    Dequantize,
}

impl OpType {
    pub fn name(&self) -> &'static str {
        match self {
            OpType::Matmul => "matmul",
            OpType::Attention => "attention",
            OpType::LayerNorm => "layer_norm",
            OpType::RmsNorm => "rms_norm",
            OpType::SiLU => "silu",
            OpType::ReLU => "relu",
            OpType::Softmax => "softmax",
            OpType::RoPE => "rope",
            OpType::Embedding => "embedding",
            OpType::Quantize => "quantize",
            OpType::Dequantize => "dequantize",
        }
    }
}

/// A registered kernel entry.
#[derive(Debug, Clone)]
pub struct KernelEntry {
    pub op: OpType,
    pub backend: Backend,
    pub priority: u32,
    pub description: String,
}

/// Key for registry lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RegistryKey {
    op: OpType,
    backend: Backend,
}

/// Central kernel dispatch registry.
#[derive(Debug)]
pub struct DispatchRegistry {
    entries: HashMap<RegistryKey, KernelEntry>,
    available_backends: Vec<Backend>,
}

impl DispatchRegistry {
    pub fn new() -> Self {
        Self { entries: HashMap::new(), available_backends: Vec::new() }
    }

    /// Mark a backend as available on this system.
    pub fn add_backend(&mut self, backend: Backend) {
        if !self.available_backends.contains(&backend) {
            self.available_backends.push(backend);
        }
    }

    /// Register a kernel implementation.
    pub fn register(
        &mut self,
        op: OpType,
        backend: Backend,
        priority: u32,
        desc: impl Into<String>,
    ) {
        let key = RegistryKey { op, backend };
        self.entries.insert(key, KernelEntry { op, backend, priority, description: desc.into() });
    }

    /// Look up the best available kernel for an operation.
    pub fn resolve(&self, op: OpType) -> Option<&KernelEntry> {
        self.available_backends
            .iter()
            .filter_map(|&b| self.entries.get(&RegistryKey { op, backend: b }))
            .max_by_key(|e| e.priority)
    }

    /// Look up a specific backend's kernel.
    pub fn get(&self, op: OpType, backend: Backend) -> Option<&KernelEntry> {
        self.entries.get(&RegistryKey { op, backend })
    }

    /// List all registered kernels for an operation.
    pub fn list_for_op(&self, op: OpType) -> Vec<&KernelEntry> {
        self.entries.values().filter(|e| e.op == op).collect()
    }

    /// List all available backends.
    pub fn backends(&self) -> &[Backend] {
        &self.available_backends
    }

    /// Total number of registered kernels.
    pub fn kernel_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if an operation has any implementation.
    pub fn has_op(&self, op: OpType) -> bool {
        self.entries.keys().any(|k| k.op == op)
    }

    /// Get all registered ops.
    pub fn registered_ops(&self) -> Vec<OpType> {
        let mut ops: Vec<_> = self.entries.keys().map(|k| k.op).collect();
        ops.sort_by_key(|o| o.name());
        ops.dedup();
        ops
    }
}

impl Default for DispatchRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a registry with standard CPU kernels registered.
pub fn default_cpu_registry() -> DispatchRegistry {
    let mut reg = DispatchRegistry::new();
    reg.add_backend(Backend::Scalar);

    let ops = [
        OpType::Matmul,
        OpType::Attention,
        OpType::LayerNorm,
        OpType::RmsNorm,
        OpType::SiLU,
        OpType::ReLU,
        OpType::Softmax,
        OpType::RoPE,
        OpType::Embedding,
        OpType::Quantize,
        OpType::Dequantize,
    ];

    for op in ops {
        reg.register(op, Backend::Scalar, 1, format!("scalar {}", op.name()));
    }

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_resolve() {
        let mut reg = DispatchRegistry::new();
        reg.add_backend(Backend::Scalar);
        reg.register(OpType::Matmul, Backend::Scalar, 1, "scalar matmul");
        let entry = reg.resolve(OpType::Matmul).unwrap();
        assert_eq!(entry.backend, Backend::Scalar);
    }

    #[test]
    fn test_priority_resolution() {
        let mut reg = DispatchRegistry::new();
        reg.add_backend(Backend::Scalar);
        reg.add_backend(Backend::Avx2);
        reg.register(OpType::Matmul, Backend::Scalar, 1, "scalar");
        reg.register(OpType::Matmul, Backend::Avx2, 10, "avx2");
        let entry = reg.resolve(OpType::Matmul).unwrap();
        assert_eq!(entry.backend, Backend::Avx2);
    }

    #[test]
    fn test_unavailable_backend() {
        let mut reg = DispatchRegistry::new();
        reg.add_backend(Backend::Scalar);
        reg.register(OpType::Matmul, Backend::Cuda, 100, "cuda");
        reg.register(OpType::Matmul, Backend::Scalar, 1, "scalar");
        let entry = reg.resolve(OpType::Matmul).unwrap();
        assert_eq!(entry.backend, Backend::Scalar); // CUDA not available
    }

    #[test]
    fn test_no_kernel() {
        let reg = DispatchRegistry::new();
        assert!(reg.resolve(OpType::Matmul).is_none());
    }

    #[test]
    fn test_get_specific() {
        let mut reg = DispatchRegistry::new();
        reg.register(OpType::SiLU, Backend::Neon, 5, "neon silu");
        let entry = reg.get(OpType::SiLU, Backend::Neon).unwrap();
        assert_eq!(entry.priority, 5);
    }

    #[test]
    fn test_list_for_op() {
        let mut reg = DispatchRegistry::new();
        reg.register(OpType::Softmax, Backend::Scalar, 1, "s");
        reg.register(OpType::Softmax, Backend::Avx2, 5, "a");
        let list = reg.list_for_op(OpType::Softmax);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_backend_properties() {
        assert!(Backend::Scalar.is_cpu());
        assert!(Backend::Avx2.is_cpu());
        assert!(Backend::Cuda.is_gpu());
        assert!(Backend::Metal.is_gpu());
        assert!(!Backend::Scalar.is_gpu());
    }

    #[test]
    fn test_default_cpu_registry() {
        let reg = default_cpu_registry();
        assert!(reg.kernel_count() >= 11);
        assert!(reg.has_op(OpType::Matmul));
        assert!(reg.has_op(OpType::SiLU));
    }

    #[test]
    fn test_registered_ops() {
        let reg = default_cpu_registry();
        let ops = reg.registered_ops();
        assert!(ops.len() >= 11);
    }

    #[test]
    fn test_kernel_count() {
        let mut reg = DispatchRegistry::new();
        reg.register(OpType::Matmul, Backend::Scalar, 1, "s");
        reg.register(OpType::SiLU, Backend::Scalar, 1, "s");
        assert_eq!(reg.kernel_count(), 2);
    }

    #[test]
    fn test_add_backend_dedup() {
        let mut reg = DispatchRegistry::new();
        reg.add_backend(Backend::Scalar);
        reg.add_backend(Backend::Scalar);
        assert_eq!(reg.backends().len(), 1);
    }

    #[test]
    fn test_op_name() {
        assert_eq!(OpType::Matmul.name(), "matmul");
        assert_eq!(OpType::RmsNorm.name(), "rms_norm");
        assert_eq!(Backend::Avx512.name(), "avx512");
    }
}
