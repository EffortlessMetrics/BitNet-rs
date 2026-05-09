use bitnet_kernels::dispatch_planner::{ModelDispatchBackend, ModelDispatchSummary};
use serde_json::{Value, json};

pub const CUDA_PLANNER_RECEIPT_VERSION: &str = "cuda-planner-004";
pub const BITNET_B158_MODEL_FAMILY: &str = "bitnet_b1_58";
pub const BITNET_I2S_QK256_QUANTIZATION: &str = "i2_s_qk256";

#[derive(Debug, Clone, Copy)]
pub struct ExecutionPlanReceiptInput<'a> {
    pub model_family: &'a str,
    pub quantization: &'a str,
    pub requested_backend: &'a str,
    pub selected_backend: &'a str,
    pub runtime_api: &'a str,
    pub strict_fallback_policy: &'a str,
    pub summary: ModelDispatchSummary,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
}

pub fn execution_plan_receipt(input: ExecutionPlanReceiptInput<'_>) -> Value {
    let selected_route = input.summary.selected_route_label();
    let strict_cuda_ready = input.summary.strict_cuda_ready
        && input.runtime_api == "cuda"
        && input.strict_fallback_policy == "reject"
        && !input.summary.fallback_used
        && input.summary.unsupported_ops == 0;

    json!({
        "planner_version": CUDA_PLANNER_RECEIPT_VERSION,
        "model_family": input.model_family,
        "quantization": input.quantization,
        "selected_route": selected_route,
        "requested_backend": input.requested_backend,
        "selected_backend": input.selected_backend,
        "runtime_api": input.runtime_api,
        "strict_fallback_policy": input.strict_fallback_policy,
        "dense_regular_llm_cuda": input.summary.cuda_dense_regular_llm_ops > 0,
        "bitnet_packed_qk256_cuda": input.summary.cuda_bitnet_qk256_ops > 0,
        "cuda_bitnet_qk256_ops": input.summary.cuda_bitnet_qk256_ops,
        "cuda_dense_regular_llm_ops": input.summary.cuda_dense_regular_llm_ops,
        "cpu_fallback_ops": input.summary.cpu_fallback_ops,
        "unsupported_ops": input.summary.unsupported_ops,
        "total_ops": input.summary.total_ops,
        "cuda_ops": input.summary.cuda_ops(),
        "mixed_cuda_routes": input.summary.has_mixed_cuda_routes(),
        "fallback_used": input.summary.fallback_used,
        "strict_cuda_ready": strict_cuda_ready,
        "speedup_claim": input.speedup_claim,
        "full_cuda_residency_claimed": input.full_cuda_residency_claimed,
    })
}

pub fn bitnet_qk256_execution_plan_receipt(
    coverage: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    requested_backend: &str,
    selected_backend: &str,
    runtime_api: &str,
    strict_fallback_policy: &str,
) -> Value {
    let total_ops = usize::try_from(coverage.bitnet_linear_layers_total).unwrap_or(usize::MAX);
    let cuda_bitnet_qk256_ops =
        usize::try_from(coverage.bitnet_linear_layers_on_cuda).unwrap_or(usize::MAX);
    let cpu_fallback_ops =
        usize::try_from(coverage.bitnet_linear_layers_cpu_fallback).unwrap_or(usize::MAX);
    let unsupported_ops = coverage.unsupported_ops.len();
    let summary = ModelDispatchSummary {
        total_ops,
        cuda_bitnet_qk256_ops,
        cuda_dense_regular_llm_ops: 0,
        cpu_fallback_ops,
        unsupported_ops,
        fallback_used: cpu_fallback_ops > 0,
        selected_route: (cuda_bitnet_qk256_ops > 0)
            .then_some(ModelDispatchBackend::CudaBitnetQk256),
        strict_cuda_ready: total_ops > 0
            && cuda_bitnet_qk256_ops == total_ops
            && cpu_fallback_ops == 0
            && unsupported_ops == 0,
    };

    execution_plan_receipt(ExecutionPlanReceiptInput {
        model_family: BITNET_B158_MODEL_FAMILY,
        quantization: BITNET_I2S_QK256_QUANTIZATION,
        requested_backend,
        selected_backend,
        runtime_api,
        strict_fallback_policy,
        summary,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    })
}

pub fn strict_bitnet_qk256_execution_plan_failed_rules(plan: &Value) -> Vec<&'static str> {
    let mut failed = Vec::new();
    if plan.get("planner_version").and_then(Value::as_str) != Some(CUDA_PLANNER_RECEIPT_VERSION) {
        failed.push("execution_plan_planner_version");
    }
    if plan.get("model_family").and_then(Value::as_str) != Some(BITNET_B158_MODEL_FAMILY) {
        failed.push("execution_plan_model_family");
    }
    if plan.get("quantization").and_then(Value::as_str) != Some(BITNET_I2S_QK256_QUANTIZATION) {
        failed.push("execution_plan_quantization");
    }
    if plan.get("selected_route").and_then(Value::as_str) != Some("bitnet_qk256_cuda") {
        failed.push("execution_plan_selected_route_bitnet_qk256_cuda");
    }
    if plan.get("bitnet_packed_qk256_cuda").and_then(Value::as_bool) != Some(true) {
        failed.push("execution_plan_bitnet_packed_qk256_cuda");
    }
    if plan.get("dense_regular_llm_cuda").and_then(Value::as_bool) != Some(false) {
        failed.push("execution_plan_dense_regular_llm_cuda_false");
    }
    if plan.get("runtime_api").and_then(Value::as_str) != Some("cuda") {
        failed.push("execution_plan_runtime_api_cuda");
    }
    if plan.get("strict_fallback_policy").and_then(Value::as_str) != Some("reject") {
        failed.push("execution_plan_strict_fallback_policy_reject");
    }
    if plan.get("fallback_used").and_then(Value::as_bool) != Some(false) {
        failed.push("execution_plan_fallback_false");
    }
    if plan.get("cpu_fallback_ops").and_then(Value::as_u64).unwrap_or(1) != 0 {
        failed.push("execution_plan_cpu_fallback_ops_zero");
    }
    if plan.get("unsupported_ops").and_then(Value::as_u64).unwrap_or(1) != 0 {
        failed.push("execution_plan_unsupported_ops_zero");
    }
    if plan.get("cuda_bitnet_qk256_ops").and_then(Value::as_u64).unwrap_or(0) == 0 {
        failed.push("execution_plan_cuda_bitnet_qk256_ops_recorded");
    }
    if plan.get("cuda_dense_regular_llm_ops").and_then(Value::as_u64).unwrap_or(1) != 0 {
        failed.push("execution_plan_cuda_dense_regular_llm_ops_zero");
    }
    if plan.get("strict_cuda_ready").and_then(Value::as_bool) != Some(true) {
        failed.push("execution_plan_strict_cuda_ready");
    }
    if plan.get("speedup_claim").and_then(Value::as_bool) != Some(false) {
        failed.push("execution_plan_speedup_claim_false");
    }
    if plan.get("full_cuda_residency_claimed").and_then(Value::as_bool) != Some(false) {
        failed.push("execution_plan_full_cuda_residency_claimed_false");
    }
    failed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitnet_qk256_summary_stays_separate_from_dense_cuda() {
        let plan = execution_plan_receipt(ExecutionPlanReceiptInput {
            model_family: BITNET_B158_MODEL_FAMILY,
            quantization: BITNET_I2S_QK256_QUANTIZATION,
            requested_backend: "nvidia-rtx-5070-ti-cuda",
            selected_backend: "nvidia-rtx-5070-ti-cuda",
            runtime_api: "cuda",
            strict_fallback_policy: "reject",
            summary: ModelDispatchSummary {
                total_ops: 210,
                cuda_bitnet_qk256_ops: 210,
                cuda_dense_regular_llm_ops: 0,
                cpu_fallback_ops: 0,
                unsupported_ops: 0,
                fallback_used: false,
                selected_route: Some(ModelDispatchBackend::CudaBitnetQk256),
                strict_cuda_ready: true,
            },
            speedup_claim: false,
            full_cuda_residency_claimed: false,
        });

        assert!(strict_bitnet_qk256_execution_plan_failed_rules(&plan).is_empty());
        assert_eq!(plan["selected_route"], "bitnet_qk256_cuda");
        assert_eq!(plan["dense_regular_llm_cuda"], false);
    }

    #[test]
    fn dense_cuda_summary_cannot_satisfy_bitnet_qk256_proof() {
        let plan = execution_plan_receipt(ExecutionPlanReceiptInput {
            model_family: "qwen",
            quantization: "bf16",
            requested_backend: "nvidia-rtx-5070-ti-cuda",
            selected_backend: "nvidia-rtx-5070-ti-cuda",
            runtime_api: "cuda",
            strict_fallback_policy: "reject",
            summary: ModelDispatchSummary {
                total_ops: 12,
                cuda_bitnet_qk256_ops: 0,
                cuda_dense_regular_llm_ops: 12,
                cpu_fallback_ops: 0,
                unsupported_ops: 0,
                fallback_used: false,
                selected_route: Some(ModelDispatchBackend::CudaDenseRegularLlm),
                strict_cuda_ready: true,
            },
            speedup_claim: false,
            full_cuda_residency_claimed: false,
        });

        let failed = strict_bitnet_qk256_execution_plan_failed_rules(&plan);
        assert!(failed.contains(&"execution_plan_selected_route_bitnet_qk256_cuda"));
        assert!(failed.contains(&"execution_plan_dense_regular_llm_cuda_false"));
        assert!(failed.contains(&"execution_plan_cuda_bitnet_qk256_ops_recorded"));
    }
}
