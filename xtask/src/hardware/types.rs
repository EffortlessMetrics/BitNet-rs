use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(super) const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

#[derive(Debug, Deserialize)]
pub(super) struct CapabilityMatrix {
    pub(super) schema_version: u32,
    pub(super) matrix_id: String,
    pub(super) device_slug: String,
    pub(super) backend_family: String,
    pub(super) selected_backend: String,
    pub(super) quality_gated_benchmarks_required: bool,
    pub(super) claim_policy: ClaimPolicy,
    pub(super) kernels: Vec<KernelCapability>,
    pub(super) not_claims: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct ClaimPolicy {
    pub(super) no_model_family_claim_without_required_kernels: bool,
    pub(super) no_benchmark_claim_without_quality_passed: bool,
    pub(super) no_fallback_for_claimed_kernels: bool,
    pub(super) device_route_must_be_concrete: bool,
}

#[derive(Debug, Deserialize)]
pub(super) struct KernelCapability {
    pub(super) kernel: String,
    pub(super) model_families: Vec<String>,
    pub(super) status: String,
    pub(super) fallback_allowed_when_claimed: bool,
    pub(super) proof_receipts: Vec<String>,
    #[serde(default)]
    pub(super) reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct KernelSummary {
    pub(super) kernel: String,
    pub(super) model_families: Vec<String>,
    pub(super) status: String,
    pub(super) claimable: bool,
    pub(super) fallback_allowed_when_claimed: bool,
    pub(super) proof_receipt_count: usize,
    pub(super) reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct CapabilityCheckReport {
    pub(super) diagnostic: &'static str,
    pub(super) producer: &'static str,
    pub(super) matrix_path: String,
    pub(super) schema_version: u32,
    pub(super) matrix_id: String,
    pub(super) device_slug: String,
    pub(super) backend_family: String,
    pub(super) selected_backend: String,
    pub(super) quality_gated_benchmarks_required: bool,
    pub(super) claim_policy: ClaimPolicy,
    pub(super) passed: bool,
    pub(super) kernel_count: usize,
    pub(super) claimable_kernel_count: usize,
    pub(super) status_counts: BTreeMap<String, usize>,
    pub(super) kernels: Vec<KernelSummary>,
    pub(super) missing: Vec<String>,
    pub(super) not_claims: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct RouteTable {
    #[serde(default)]
    pub(super) route: Vec<RouteEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct RouteEntry {
    pub(super) route_id: String,
    pub(super) op: String,
    pub(super) model_family: String,
    pub(super) quantization: String,
    pub(super) backend_family: String,
    pub(super) selected_backend: String,
    pub(super) device_slug: String,
    pub(super) device_family: String,
    pub(super) device_models: Vec<String>,
    pub(super) kernel_variant: String,
    pub(super) claim_level: String,
    pub(super) fallback_allowed: bool,
    pub(super) proof_receipts: Vec<String>,
    #[serde(default)]
    pub(super) reason: Option<String>,
    #[serde(default)]
    pub(super) not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct RouteQuery {
    pub(super) device_slug: String,
    pub(super) selected_backend: String,
    pub(super) backend_family: String,
    pub(super) model_family: String,
    pub(super) quantization: String,
    pub(super) op: String,
}

#[derive(Debug, Serialize)]
pub(super) struct RouteResolveReport {
    pub(super) diagnostic: &'static str,
    pub(super) producer: &'static str,
    pub(super) routing_table: String,
    pub(super) query: RouteQuery,
    pub(super) passed: bool,
    pub(super) route_found: bool,
    pub(super) route_verified: bool,
    pub(super) claimable: bool,
    pub(super) classification: String,
    pub(super) route: Option<RouteEntry>,
    pub(super) failures: Vec<String>,
    pub(super) not_claims: Vec<String>,
}
