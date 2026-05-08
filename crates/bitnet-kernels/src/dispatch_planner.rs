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

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu)
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
        self.decisions.iter().filter(|d| d.backend.is_gpu()).count()
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

/// Model family used by the model-aware CUDA planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    BitNet,
    DenseRegularLlm,
    Unknown,
}

impl ModelFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet",
            Self::DenseRegularLlm => "dense_regular_llm",
            Self::Unknown => "unknown",
        }
    }
}

/// Quantization or tensor layout family used by the model-aware CUDA planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationKind {
    BitnetI2S,
    Qk256,
    DenseFp16,
    DenseBf16,
    Unknown,
}

impl QuantizationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BitnetI2S => "bitnet_i2s",
            Self::Qk256 => "qk256",
            Self::DenseFp16 => "dense_fp16",
            Self::DenseBf16 => "dense_bf16",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_bitnet_qk256(&self) -> bool {
        matches!(self, Self::BitnetI2S | Self::Qk256)
    }

    pub fn is_dense_cuda(&self) -> bool {
        matches!(self, Self::DenseFp16 | Self::DenseBf16)
    }
}

/// Fallback policy for model-aware CUDA routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPolicy {
    StrictCuda,
    AllowCpuFallback,
}

/// CUDA capabilities visible to the model-aware planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPlannerCapabilities {
    pub cuda_available: bool,
    pub bitnet_qk256_cuda: bool,
    pub dense_regular_llm_cuda: bool,
}

impl CudaPlannerCapabilities {
    pub fn none() -> Self {
        Self { cuda_available: false, bitnet_qk256_cuda: false, dense_regular_llm_cuda: false }
    }

    pub fn bitnet_qk256() -> Self {
        Self { cuda_available: true, bitnet_qk256_cuda: true, dense_regular_llm_cuda: false }
    }

    pub fn dense_regular_llm() -> Self {
        Self { cuda_available: true, bitnet_qk256_cuda: false, dense_regular_llm_cuda: true }
    }
}

/// Model-aware dispatch inputs for CUDA routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDispatchSpec {
    pub model_family: ModelFamily,
    pub quantization: QuantizationKind,
    pub backend_policy: BackendPolicy,
    pub has_simd: bool,
    pub cuda: CudaPlannerCapabilities,
}

/// Backend route selected by the model-aware planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelDispatchBackend {
    CpuScalar,
    CpuSimd,
    CudaBitnetQk256,
    CudaDenseRegularLlm,
    Unsupported,
}

impl ModelDispatchBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuScalar => "cpu_scalar",
            Self::CpuSimd => "cpu_simd",
            Self::CudaBitnetQk256 => "cuda_bitnet_qk256",
            Self::CudaDenseRegularLlm => "cuda_dense_regular_llm",
            Self::Unsupported => "unsupported",
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CpuScalar | Self::CpuSimd)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::CudaBitnetQk256 | Self::CudaDenseRegularLlm)
    }

    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported)
    }
}

/// Model-aware dispatch decision for a single op.
#[derive(Debug, Clone)]
pub struct ModelDispatchDecision {
    pub op: DispatchOp,
    pub backend: ModelDispatchBackend,
    pub fallback_used: bool,
    pub reason: String,
}

/// Model-aware dispatch plan that keeps BitNet packed CUDA and dense CUDA separate.
#[derive(Debug, Clone)]
pub struct ModelDispatchPlan {
    pub decisions: Vec<ModelDispatchDecision>,
}

impl ModelDispatchPlan {
    pub fn new() -> Self {
        Self { decisions: Vec::new() }
    }

    pub fn add(&mut self, decision: ModelDispatchDecision) {
        self.decisions.push(decision);
    }

    pub fn op_count(&self) -> usize {
        self.decisions.len()
    }

    pub fn by_backend(&self, backend: ModelDispatchBackend) -> Vec<&ModelDispatchDecision> {
        self.decisions.iter().filter(|d| d.backend == backend).collect()
    }

    pub fn cuda_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.backend.is_cuda()).count()
    }

    pub fn cpu_fallback_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.fallback_used && d.backend.is_cpu()).count()
    }

    pub fn unsupported_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.backend.is_unsupported()).count()
    }
}

impl Default for ModelDispatchPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Plan model-aware dispatch without conflating BitNet QK256 and dense CUDA routes.
pub fn plan_model_dispatch(ops: &[DispatchOp], spec: ModelDispatchSpec) -> ModelDispatchPlan {
    let mut plan = ModelDispatchPlan::new();
    for op in ops {
        let (backend, fallback_used, reason) = select_model_backend(op, spec);
        plan.add(ModelDispatchDecision { op: op.clone(), backend, fallback_used, reason });
    }
    plan
}

fn select_model_backend(
    op: &DispatchOp,
    spec: ModelDispatchSpec,
) -> (ModelDispatchBackend, bool, String) {
    if let Some(backend) = select_model_cuda_backend(op, spec) {
        return (backend, false, format!("{} route selected for {}", backend.as_str(), op.name));
    }

    match spec.backend_policy {
        BackendPolicy::StrictCuda => (
            ModelDispatchBackend::Unsupported,
            false,
            format!(
                "strict CUDA rejects CPU fallback for {} {} {} op {}",
                spec.model_family.as_str(),
                spec.quantization.as_str(),
                op.op_type.as_str(),
                op.name
            ),
        ),
        BackendPolicy::AllowCpuFallback => {
            let backend = if spec.has_simd {
                ModelDispatchBackend::CpuSimd
            } else {
                ModelDispatchBackend::CpuScalar
            };
            (
                backend,
                true,
                format!(
                    "explicit CPU fallback for {} {} op {}",
                    spec.model_family.as_str(),
                    op.op_type.as_str(),
                    op.name
                ),
            )
        }
    }
}

fn select_model_cuda_backend(
    op: &DispatchOp,
    spec: ModelDispatchSpec,
) -> Option<ModelDispatchBackend> {
    if !spec.cuda.cuda_available || op.op_type != OpType::MatMul {
        return None;
    }

    match (spec.model_family, spec.quantization) {
        (ModelFamily::BitNet, quantization)
            if quantization.is_bitnet_qk256() && op.is_quantized && spec.cuda.bitnet_qk256_cuda =>
        {
            Some(ModelDispatchBackend::CudaBitnetQk256)
        }
        (ModelFamily::DenseRegularLlm, quantization)
            if quantization.is_dense_cuda()
                && !op.is_quantized
                && spec.cuda.dense_regular_llm_cuda =>
        {
            Some(ModelDispatchBackend::CudaDenseRegularLlm)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_op(size: usize) -> DispatchOp {
        DispatchOp { name: "matmul".into(), op_type: OpType::MatMul, size, is_quantized: false }
    }

    fn qk256_matmul_op() -> DispatchOp {
        DispatchOp {
            name: "blk.0.attn_q".into(),
            op_type: OpType::MatMul,
            size: 4096,
            is_quantized: true,
        }
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
        assert!(DispatchBackend::Gpu.is_gpu());
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

    #[test]
    fn model_aware_bitnet_qk256_routes_to_qk256_cuda() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::BitNet,
            quantization: QuantizationKind::BitnetI2S,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::bitnet_qk256(),
        };

        let plan = plan_model_dispatch(&[qk256_matmul_op()], spec);

        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CudaBitnetQk256);
        assert!(!plan.decisions[0].fallback_used);
        assert_eq!(plan.cuda_ops(), 1);
        assert_eq!(plan.unsupported_ops(), 0);
    }

    #[test]
    fn model_aware_dense_fp16_routes_to_dense_cuda() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::DenseRegularLlm,
            quantization: QuantizationKind::DenseFp16,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::dense_regular_llm(),
        };

        let plan = plan_model_dispatch(&[matmul_op(4096)], spec);

        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CudaDenseRegularLlm);
        assert!(!plan.decisions[0].fallback_used);
        assert_eq!(plan.cuda_ops(), 1);
    }

    #[test]
    fn model_aware_dense_cuda_does_not_satisfy_bitnet_qk256() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::BitNet,
            quantization: QuantizationKind::BitnetI2S,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::dense_regular_llm(),
        };

        let plan = plan_model_dispatch(&[qk256_matmul_op()], spec);

        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::Unsupported);
        assert!(plan.decisions[0].reason.contains("strict CUDA rejects CPU fallback"));
        assert_eq!(plan.cuda_ops(), 0);
        assert_eq!(plan.unsupported_ops(), 1);
    }

    #[test]
    fn model_aware_bitnet_qk256_does_not_satisfy_dense_cuda() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::DenseRegularLlm,
            quantization: QuantizationKind::DenseFp16,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::bitnet_qk256(),
        };

        let plan = plan_model_dispatch(&[matmul_op(4096)], spec);

        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::Unsupported);
        assert_eq!(plan.cuda_ops(), 0);
        assert_eq!(plan.unsupported_ops(), 1);
    }

    #[test]
    fn model_aware_strict_cuda_rejects_unsupported_cpu_fallback() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::BitNet,
            quantization: QuantizationKind::BitnetI2S,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::bitnet_qk256(),
        };

        let plan = plan_model_dispatch(&[norm_op()], spec);

        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::Unsupported);
        assert!(!plan.decisions[0].fallback_used);
        assert_eq!(plan.unsupported_ops(), 1);
    }

    #[test]
    fn model_aware_non_strict_uses_explicit_cpu_fallback() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::BitNet,
            quantization: QuantizationKind::BitnetI2S,
            backend_policy: BackendPolicy::AllowCpuFallback,
            has_simd: true,
            cuda: CudaPlannerCapabilities::none(),
        };

        let plan = plan_model_dispatch(&[qk256_matmul_op()], spec);

        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CpuSimd);
        assert!(plan.decisions[0].fallback_used);
        assert_eq!(plan.cpu_fallback_ops(), 1);
        assert_eq!(plan.unsupported_ops(), 0);
    }

    #[test]
    fn model_aware_backend_strs_are_claim_separated() {
        assert_eq!(ModelDispatchBackend::CudaBitnetQk256.as_str(), "cuda_bitnet_qk256");
        assert_eq!(ModelDispatchBackend::CudaDenseRegularLlm.as_str(), "cuda_dense_regular_llm");
        assert!(ModelDispatchBackend::CudaBitnetQk256.is_cuda());
        assert!(!ModelDispatchBackend::Unsupported.is_cuda());
    }
}
