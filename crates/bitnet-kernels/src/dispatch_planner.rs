//! Kernel dispatch planner.
//!
//! Plan kernel dispatch across operations, selecting optimal backends.

/// Backend type for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchBackend {
    CpuScalar,
    CpuSimd,
    Gpu,
}

impl DispatchBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuScalar => "cpu_scalar",
            Self::CpuSimd => "cpu_simd",
            Self::Gpu => "gpu",
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CpuScalar | Self::CpuSimd)
    }
}

/// Operation to dispatch.
#[derive(Debug, Clone)]
pub struct DispatchOp {
    pub name: String,
    pub op_type: OpType,
    pub size: usize,
    pub is_quantized: bool,
}

/// Type of operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    MatMul,
    Softmax,
    LayerNorm,
    RmsNorm,
    Activation,
    Embedding,
    Attention,
    RoPE,
}

impl OpType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MatMul => "matmul",
            Self::Softmax => "softmax",
            Self::LayerNorm => "layernorm",
            Self::RmsNorm => "rmsnorm",
            Self::Activation => "activation",
            Self::Embedding => "embedding",
            Self::Attention => "attention",
            Self::RoPE => "rope",
        }
    }

    pub fn is_compute_bound(&self) -> bool {
        matches!(self, Self::MatMul | Self::Attention)
    }

    pub fn is_memory_bound(&self) -> bool {
        matches!(self, Self::Embedding | Self::LayerNorm | Self::RmsNorm)
    }
}

/// Dispatch decision for a single op.
#[derive(Debug, Clone)]
pub struct DispatchDecision {
    pub op: DispatchOp,
    pub backend: DispatchBackend,
    pub reason: String,
}

/// Dispatch plan for a model's operations.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    pub decisions: Vec<DispatchDecision>,
}

impl DispatchPlan {
    pub fn new() -> Self {
        Self { decisions: Vec::new() }
    }

    pub fn add(&mut self, decision: DispatchDecision) {
        self.decisions.push(decision);
    }

    pub fn op_count(&self) -> usize {
        self.decisions.len()
    }

    pub fn by_backend(&self, backend: DispatchBackend) -> Vec<&DispatchDecision> {
        self.decisions.iter().filter(|d| d.backend == backend).collect()
    }

    pub fn gpu_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.backend == DispatchBackend::Gpu).count()
    }

    pub fn cpu_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.backend.is_cpu()).count()
    }

    pub fn summary(&self) -> PlanSummary {
        PlanSummary {
            total_ops: self.decisions.len(),
            gpu_ops: self.gpu_ops(),
            cpu_simd_ops: self.by_backend(DispatchBackend::CpuSimd).len(),
            cpu_scalar_ops: self.by_backend(DispatchBackend::CpuScalar).len(),
        }
    }
}

impl Default for DispatchPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PlanSummary {
    pub total_ops: usize,
    pub gpu_ops: usize,
    pub cpu_simd_ops: usize,
    pub cpu_scalar_ops: usize,
}

/// Plan dispatch for a set of operations.
pub fn plan_dispatch(
    ops: &[DispatchOp],
    has_gpu: bool,
    has_simd: bool,
    gpu_threshold: usize,
) -> DispatchPlan {
    let mut plan = DispatchPlan::new();
    for op in ops {
        let (backend, reason) = select_backend(op, has_gpu, has_simd, gpu_threshold);
        plan.add(DispatchDecision { op: op.clone(), backend, reason });
    }
    plan
}

fn select_backend(
    op: &DispatchOp,
    has_gpu: bool,
    has_simd: bool,
    gpu_threshold: usize,
) -> (DispatchBackend, String) {
    // Large compute-bound ops prefer GPU
    if has_gpu && op.op_type.is_compute_bound() && op.size >= gpu_threshold {
        return (DispatchBackend::Gpu, format!("Compute-bound, size {} >= threshold", op.size));
    }
    // SIMD for everything else if available
    if has_simd {
        return (DispatchBackend::CpuSimd, "SIMD available".into());
    }
    (DispatchBackend::CpuScalar, "Scalar fallback".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_op(size: usize) -> DispatchOp {
        DispatchOp { name: "matmul".into(), op_type: OpType::MatMul, size, is_quantized: false }
    }

    fn norm_op() -> DispatchOp {
        DispatchOp {
            name: "norm".into(),
            op_type: OpType::LayerNorm,
            size: 512,
            is_quantized: false,
        }
    }

    #[test]
    fn test_scalar_fallback() {
        let plan = plan_dispatch(&[matmul_op(100)], false, false, 1000);
        assert_eq!(plan.decisions[0].backend, DispatchBackend::CpuScalar);
    }

    #[test]
    fn test_simd_preferred() {
        let plan = plan_dispatch(&[matmul_op(100)], false, true, 1000);
        assert_eq!(plan.decisions[0].backend, DispatchBackend::CpuSimd);
    }

    #[test]
    fn test_gpu_for_large() {
        let plan = plan_dispatch(&[matmul_op(10000)], true, true, 1000);
        assert_eq!(plan.decisions[0].backend, DispatchBackend::Gpu);
    }

    #[test]
    fn test_gpu_skip_small() {
        let plan = plan_dispatch(&[matmul_op(100)], true, true, 1000);
        assert_eq!(plan.decisions[0].backend, DispatchBackend::CpuSimd);
    }

    #[test]
    fn test_memory_bound_cpu() {
        let plan = plan_dispatch(&[norm_op()], true, true, 100);
        assert_eq!(plan.decisions[0].backend, DispatchBackend::CpuSimd);
    }

    #[test]
    fn test_plan_summary() {
        let ops = vec![matmul_op(10000), norm_op(), matmul_op(50)];
        let plan = plan_dispatch(&ops, true, true, 1000);
        let s = plan.summary();
        assert_eq!(s.total_ops, 3);
        assert_eq!(s.gpu_ops, 1);
    }

    #[test]
    fn test_op_type_bound() {
        assert!(OpType::MatMul.is_compute_bound());
        assert!(OpType::LayerNorm.is_memory_bound());
        assert!(!OpType::MatMul.is_memory_bound());
    }

    #[test]
    fn test_backend_str() {
        assert_eq!(DispatchBackend::Gpu.as_str(), "gpu");
        assert!(DispatchBackend::CpuScalar.is_cpu());
    }

    #[test]
    fn test_by_backend() {
        let ops = vec![matmul_op(10000), norm_op()];
        let plan = plan_dispatch(&ops, true, true, 1000);
        let gpu = plan.by_backend(DispatchBackend::Gpu);
        assert_eq!(gpu.len(), 1);
    }

    #[test]
    fn test_empty_plan() {
        let plan = plan_dispatch(&[], false, false, 0);
        assert_eq!(plan.op_count(), 0);
    }

    #[test]
    fn test_op_type_str() {
        assert_eq!(OpType::Softmax.as_str(), "softmax");
        assert_eq!(OpType::RoPE.as_str(), "rope");
    }

    #[test]
    fn test_attention_compute_bound() {
        let op = DispatchOp {
            name: "attn".into(),
            op_type: OpType::Attention,
            size: 10000,
            is_quantized: false,
        };
        let plan = plan_dispatch(&[op], true, true, 1000);
        assert_eq!(plan.decisions[0].backend, DispatchBackend::Gpu);
    }
}
