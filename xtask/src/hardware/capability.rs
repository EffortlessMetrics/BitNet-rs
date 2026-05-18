use super::status::{is_claimable_status, is_known_status, status_requires_receipts};
use super::types::{
    CRITICAL_NOT_CLAIMS, CapabilityCheckReport, CapabilityMatrix, ClaimPolicy, KernelSummary,
};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub(super) fn build_capability_check_report(matrix_path: &Path) -> Result<CapabilityCheckReport> {
    let raw = fs::read_to_string(matrix_path)
        .with_context(|| format!("reading {}", matrix_path.display()))?;
    let matrix: CapabilityMatrix =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", matrix_path.display()))?;

    let mut missing = Vec::new();
    if matrix.kernels.is_empty() {
        missing.push("matrix has no kernels".to_string());
    }
    if !matrix.quality_gated_benchmarks_required {
        missing.push("quality_gated_benchmarks_required must be true".to_string());
    }
    validate_claim_policy(&matrix.claim_policy, &mut missing);
    for not_claim in CRITICAL_NOT_CLAIMS {
        if !matrix.not_claims.iter().any(|value| value == not_claim) {
            missing.push(format!("missing critical not-claim {not_claim}"));
        }
    }

    let mut status_counts = BTreeMap::new();
    let mut kernels = Vec::new();
    let mut claimable_kernel_count = 0;
    for kernel in matrix.kernels {
        if kernel.kernel.trim().is_empty() {
            missing.push("kernel entry has empty kernel name".to_string());
        }
        if kernel.model_families.is_empty() {
            missing.push(format!("{} has no model_families", kernel.kernel));
        }
        if !is_known_status(&kernel.status) {
            missing.push(format!("{} has unknown status {}", kernel.kernel, kernel.status));
        }
        *status_counts.entry(kernel.status.clone()).or_insert(0) += 1;
        let claimable = is_claimable_status(&kernel.status);
        if claimable {
            claimable_kernel_count += 1;
            missing.push(format!(
                "{} is claimable; A770 capability rail requires claimable_kernel_count=0",
                kernel.kernel
            ));
        }
        if status_requires_receipts(&kernel.status) && kernel.proof_receipts.is_empty() {
            missing.push(format!(
                "{} status {} requires proof_receipts",
                kernel.kernel, kernel.status
            ));
        }
        if claimable && kernel.fallback_allowed_when_claimed {
            missing.push(format!(
                "{} is claimable but fallback_allowed_when_claimed=true",
                kernel.kernel
            ));
        }
        kernels.push(KernelSummary {
            kernel: kernel.kernel,
            model_families: kernel.model_families,
            status: kernel.status,
            claimable,
            fallback_allowed_when_claimed: kernel.fallback_allowed_when_claimed,
            proof_receipt_count: kernel.proof_receipts.len(),
            reason: kernel.reason,
        });
    }

    Ok(CapabilityCheckReport {
        diagnostic: "a770_kernel_capability_check",
        producer: "cargo xtask hardware a770 kernel-capability-check",
        matrix_path: matrix_path.display().to_string(),
        schema_version: matrix.schema_version,
        matrix_id: matrix.matrix_id,
        device_slug: matrix.device_slug,
        backend_family: matrix.backend_family,
        selected_backend: matrix.selected_backend,
        quality_gated_benchmarks_required: matrix.quality_gated_benchmarks_required,
        claim_policy: matrix.claim_policy,
        passed: missing.is_empty(),
        kernel_count: kernels.len(),
        claimable_kernel_count,
        status_counts,
        kernels,
        missing,
        not_claims: matrix.not_claims,
    })
}

fn validate_claim_policy(policy: &ClaimPolicy, missing: &mut Vec<String>) {
    if !policy.no_model_family_claim_without_required_kernels {
        missing.push(
            "claim_policy.no_model_family_claim_without_required_kernels must be true".to_string(),
        );
    }
    if !policy.no_benchmark_claim_without_quality_passed {
        missing.push(
            "claim_policy.no_benchmark_claim_without_quality_passed must be true".to_string(),
        );
    }
    if !policy.no_fallback_for_claimed_kernels {
        missing.push("claim_policy.no_fallback_for_claimed_kernels must be true".to_string());
    }
    if !policy.device_route_must_be_concrete {
        missing.push("claim_policy.device_route_must_be_concrete must be true".to_string());
    }
}
