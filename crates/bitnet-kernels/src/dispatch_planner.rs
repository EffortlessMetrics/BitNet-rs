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

    pub fn from_metadata_label(label: &str) -> Self {
        let label = normalized_metadata_label(label);
        if label.contains("bitnet") || label.contains("b1_58") || label.contains("w1_58") {
            return Self::BitNet;
        }

        let dense_families = [
            "qwen", "llama", "mistral", "mixtral", "phi", "gemma", "deepseek", "falcon", "yi",
            "internlm", "baichuan",
        ];
        if dense_families.iter().any(|family| label.contains(family)) {
            return Self::DenseRegularLlm;
        }

        Self::Unknown
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

    pub fn from_metadata_label(label: &str) -> Self {
        let label = normalized_metadata_label(label);

        if label.contains("qk256") {
            return Self::Qk256;
        }
        if label.contains("i2_s")
            || label.contains("i2s")
            || label.contains("w1_58")
            || label.contains("1_58")
            || label.contains("ternary")
        {
            return Self::BitnetI2S;
        }
        if label.contains("bf16") || label.contains("bfloat16") {
            return Self::DenseBf16;
        }
        if label.contains("fp16") || label.contains("f16") || label.contains("float16") {
            return Self::DenseFp16;
        }

        Self::Unknown
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

impl ModelDispatchSpec {
    pub fn from_metadata_labels(
        model_family: &str,
        quantization: &str,
        backend_policy: BackendPolicy,
        has_simd: bool,
        cuda: CudaPlannerCapabilities,
    ) -> Self {
        Self {
            model_family: ModelFamily::from_metadata_label(model_family),
            quantization: QuantizationKind::from_metadata_label(quantization),
            backend_policy,
            has_simd,
            cuda,
        }
    }
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

    pub fn receipt_route_label(&self) -> &'static str {
        match self {
            Self::CpuScalar => "cpu_scalar",
            Self::CpuSimd => "cpu_simd",
            Self::CudaBitnetQk256 => "bitnet_qk256_cuda",
            Self::CudaDenseRegularLlm => "dense_regular_llm_cuda",
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

    pub fn cuda_bitnet_qk256_ops(&self) -> usize {
        self.by_backend(ModelDispatchBackend::CudaBitnetQk256).len()
    }

    pub fn cuda_dense_regular_llm_ops(&self) -> usize {
        self.by_backend(ModelDispatchBackend::CudaDenseRegularLlm).len()
    }

    pub fn cpu_fallback_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.fallback_used && d.backend.is_cpu()).count()
    }

    pub fn unsupported_ops(&self) -> usize {
        self.decisions.iter().filter(|d| d.backend.is_unsupported()).count()
    }

    pub fn summary(&self) -> ModelDispatchSummary {
        let total_ops = self.op_count();
        let cuda_bitnet_qk256_ops = self.cuda_bitnet_qk256_ops();
        let cuda_dense_regular_llm_ops = self.cuda_dense_regular_llm_ops();
        let cpu_fallback_ops = self.cpu_fallback_ops();
        let unsupported_ops = self.unsupported_ops();
        let selected_route =
            select_unambiguous_cuda_route(cuda_bitnet_qk256_ops, cuda_dense_regular_llm_ops);
        let cuda_ops = cuda_bitnet_qk256_ops + cuda_dense_regular_llm_ops;

        ModelDispatchSummary {
            total_ops,
            cuda_bitnet_qk256_ops,
            cuda_dense_regular_llm_ops,
            cpu_fallback_ops,
            unsupported_ops,
            fallback_used: cpu_fallback_ops > 0,
            selected_route,
            strict_cuda_ready: total_ops > 0
                && selected_route.is_some()
                && cuda_ops == total_ops
                && cpu_fallback_ops == 0
                && unsupported_ops == 0,
        }
    }
}

impl Default for ModelDispatchPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Receipt-ready summary for model-aware dispatch plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDispatchSummary {
    pub total_ops: usize,
    pub cuda_bitnet_qk256_ops: usize,
    pub cuda_dense_regular_llm_ops: usize,
    pub cpu_fallback_ops: usize,
    pub unsupported_ops: usize,
    pub fallback_used: bool,
    pub selected_route: Option<ModelDispatchBackend>,
    pub strict_cuda_ready: bool,
}

impl ModelDispatchSummary {
    pub fn cuda_ops(&self) -> usize {
        self.cuda_bitnet_qk256_ops + self.cuda_dense_regular_llm_ops
    }

    pub fn has_any_cuda(&self) -> bool {
        self.cuda_ops() > 0
    }

    pub fn has_mixed_cuda_routes(&self) -> bool {
        self.cuda_bitnet_qk256_ops > 0 && self.cuda_dense_regular_llm_ops > 0
    }

    pub fn selected_route_label(&self) -> Option<&'static str> {
        self.selected_route.map(|route| route.receipt_route_label())
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

fn normalized_metadata_label(label: &str) -> String {
    label
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn select_unambiguous_cuda_route(
    cuda_bitnet_qk256_ops: usize,
    cuda_dense_regular_llm_ops: usize,
) -> Option<ModelDispatchBackend> {
    match (cuda_bitnet_qk256_ops > 0, cuda_dense_regular_llm_ops > 0) {
        (true, false) => Some(ModelDispatchBackend::CudaBitnetQk256),
        (false, true) => Some(ModelDispatchBackend::CudaDenseRegularLlm),
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
    fn model_aware_summary_counts_bitnet_qk256_cuda_route() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::BitNet,
            quantization: QuantizationKind::BitnetI2S,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::bitnet_qk256(),
        };

        let plan = plan_model_dispatch(&[qk256_matmul_op(), qk256_matmul_op()], spec);
        let summary = plan.summary();

        assert_eq!(
            summary,
            ModelDispatchSummary {
                total_ops: 2,
                cuda_bitnet_qk256_ops: 2,
                cuda_dense_regular_llm_ops: 0,
                cpu_fallback_ops: 0,
                unsupported_ops: 0,
                fallback_used: false,
                selected_route: Some(ModelDispatchBackend::CudaBitnetQk256),
                strict_cuda_ready: true,
            }
        );
        assert_eq!(summary.cuda_ops(), 2);
        assert!(summary.has_any_cuda());
        assert!(!summary.has_mixed_cuda_routes());
        assert_eq!(summary.selected_route_label(), Some("bitnet_qk256_cuda"));
    }

    #[test]
    fn model_aware_summary_counts_dense_regular_llm_cuda_route() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::DenseRegularLlm,
            quantization: QuantizationKind::DenseBf16,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::dense_regular_llm(),
        };

        let plan = plan_model_dispatch(&[matmul_op(4096), matmul_op(8192)], spec);
        let summary = plan.summary();

        assert_eq!(summary.total_ops, 2);
        assert_eq!(summary.cuda_bitnet_qk256_ops, 0);
        assert_eq!(summary.cuda_dense_regular_llm_ops, 2);
        assert_eq!(summary.cpu_fallback_ops, 0);
        assert_eq!(summary.unsupported_ops, 0);
        assert!(!summary.fallback_used);
        assert_eq!(summary.selected_route, Some(ModelDispatchBackend::CudaDenseRegularLlm));
        assert_eq!(summary.selected_route_label(), Some("dense_regular_llm_cuda"));
        assert!(summary.strict_cuda_ready);
    }

    #[test]
    fn model_aware_summary_rejects_mixed_cuda_route_claim() {
        let mut plan = ModelDispatchPlan::new();
        plan.add(ModelDispatchDecision {
            op: qk256_matmul_op(),
            backend: ModelDispatchBackend::CudaBitnetQk256,
            fallback_used: false,
            reason: "bitnet route".into(),
        });
        plan.add(ModelDispatchDecision {
            op: matmul_op(4096),
            backend: ModelDispatchBackend::CudaDenseRegularLlm,
            fallback_used: false,
            reason: "dense route".into(),
        });

        let summary = plan.summary();

        assert_eq!(summary.total_ops, 2);
        assert_eq!(summary.cuda_bitnet_qk256_ops, 1);
        assert_eq!(summary.cuda_dense_regular_llm_ops, 1);
        assert_eq!(summary.cuda_ops(), 2);
        assert!(summary.has_any_cuda());
        assert!(summary.has_mixed_cuda_routes());
        assert_eq!(summary.selected_route, None);
        assert_eq!(summary.selected_route_label(), None);
        assert!(!summary.strict_cuda_ready);
    }

    #[test]
    fn model_aware_summary_keeps_unsupported_strict_route_not_ready() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::BitNet,
            quantization: QuantizationKind::BitnetI2S,
            backend_policy: BackendPolicy::StrictCuda,
            has_simd: true,
            cuda: CudaPlannerCapabilities::bitnet_qk256(),
        };

        let plan = plan_model_dispatch(&[qk256_matmul_op(), norm_op()], spec);
        let summary = plan.summary();

        assert_eq!(summary.total_ops, 2);
        assert_eq!(summary.cuda_bitnet_qk256_ops, 1);
        assert_eq!(summary.cuda_dense_regular_llm_ops, 0);
        assert_eq!(summary.cpu_fallback_ops, 0);
        assert_eq!(summary.unsupported_ops, 1);
        assert!(!summary.fallback_used);
        assert_eq!(summary.selected_route, Some(ModelDispatchBackend::CudaBitnetQk256));
        assert!(!summary.strict_cuda_ready);
    }

    #[test]
    fn model_aware_summary_records_explicit_cpu_fallback() {
        let spec = ModelDispatchSpec {
            model_family: ModelFamily::DenseRegularLlm,
            quantization: QuantizationKind::DenseFp16,
            backend_policy: BackendPolicy::AllowCpuFallback,
            has_simd: true,
            cuda: CudaPlannerCapabilities::none(),
        };

        let plan = plan_model_dispatch(&[matmul_op(4096), norm_op()], spec);
        let summary = plan.summary();

        assert_eq!(summary.total_ops, 2);
        assert_eq!(summary.cuda_ops(), 0);
        assert_eq!(summary.cpu_fallback_ops, 2);
        assert_eq!(summary.unsupported_ops, 0);
        assert!(summary.fallback_used);
        assert_eq!(summary.selected_route, None);
        assert_eq!(summary.selected_route_label(), None);
        assert!(!summary.strict_cuda_ready);
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

    #[test]
    fn metadata_labels_map_official_bitnet_i2s_to_bitnet_cuda_spec() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "microsoft/bitnet-b1.58-2B-4T-gguf",
            "gguf_packed_i2_s",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities::bitnet_qk256(),
        );

        assert_eq!(spec.model_family, ModelFamily::BitNet);
        assert_eq!(spec.quantization, QuantizationKind::BitnetI2S);

        let plan = plan_model_dispatch(&[qk256_matmul_op()], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CudaBitnetQk256);
    }

    #[test]
    fn metadata_labels_map_qk256_to_bitnet_cuda_spec() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "BitNet b1_58",
            "QK256",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities::bitnet_qk256(),
        );

        assert_eq!(spec.model_family, ModelFamily::BitNet);
        assert_eq!(spec.quantization, QuantizationKind::Qk256);

        let plan = plan_model_dispatch(&[qk256_matmul_op()], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CudaBitnetQk256);
    }

    #[test]
    fn metadata_labels_map_qwen_fp16_to_dense_cuda_spec() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "Qwen3-0.6B",
            "fp16",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities::dense_regular_llm(),
        );

        assert_eq!(spec.model_family, ModelFamily::DenseRegularLlm);
        assert_eq!(spec.quantization, QuantizationKind::DenseFp16);

        let plan = plan_model_dispatch(&[matmul_op(4096)], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CudaDenseRegularLlm);
    }

    #[test]
    fn metadata_labels_map_llama_bf16_to_dense_cuda_spec() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "Llama-3",
            "bfloat16",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities::dense_regular_llm(),
        );

        assert_eq!(spec.model_family, ModelFamily::DenseRegularLlm);
        assert_eq!(spec.quantization, QuantizationKind::DenseBf16);

        let plan = plan_model_dispatch(&[matmul_op(4096)], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::CudaDenseRegularLlm);
    }

    #[test]
    fn unknown_metadata_stays_unsupported_under_strict_cuda() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "unclassified-model",
            "q4_k_m",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities {
                cuda_available: true,
                bitnet_qk256_cuda: true,
                dense_regular_llm_cuda: true,
            },
        );

        assert_eq!(spec.model_family, ModelFamily::Unknown);
        assert_eq!(spec.quantization, QuantizationKind::Unknown);

        let plan = plan_model_dispatch(&[matmul_op(4096)], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::Unsupported);
        assert!(!plan.decisions[0].fallback_used);
    }

    #[test]
    fn dense_family_with_qk256_quantization_does_not_route_to_bitnet_cuda() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "qwen3",
            "qk256",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities {
                cuda_available: true,
                bitnet_qk256_cuda: true,
                dense_regular_llm_cuda: true,
            },
        );

        assert_eq!(spec.model_family, ModelFamily::DenseRegularLlm);
        assert_eq!(spec.quantization, QuantizationKind::Qk256);

        let plan = plan_model_dispatch(&[qk256_matmul_op()], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::Unsupported);
    }

    #[test]
    fn bitnet_family_with_dense_quantization_does_not_route_to_dense_cuda() {
        let spec = ModelDispatchSpec::from_metadata_labels(
            "bitnet-b1.58",
            "fp16",
            BackendPolicy::StrictCuda,
            true,
            CudaPlannerCapabilities {
                cuda_available: true,
                bitnet_qk256_cuda: true,
                dense_regular_llm_cuda: true,
            },
        );

        assert_eq!(spec.model_family, ModelFamily::BitNet);
        assert_eq!(spec.quantization, QuantizationKind::DenseFp16);

        let plan = plan_model_dispatch(&[matmul_op(4096)], spec);
        assert_eq!(plan.decisions[0].backend, ModelDispatchBackend::Unsupported);
    }
}
