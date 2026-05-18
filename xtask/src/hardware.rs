mod capability;
mod emit;
mod routing;
mod status;
mod types;

use anyhow::{Result, bail};
use capability::build_capability_check_report;
use emit::{emit_capability_report, emit_route_report};
use routing::build_route_resolve_report;
use std::path::Path;
use types::RouteQuery;

pub fn kernel_capability_check(matrix_path: &Path, format: &str) -> Result<()> {
    let report = build_capability_check_report(matrix_path)?;
    emit_capability_report(&report, format)?;
    if !report.passed {
        bail!("kernel capability check failed: {}", report.missing.join(", "));
    }
    Ok(())
}

pub fn route_resolve(
    routing_table: &Path,
    device_slug: &str,
    selected_backend: &str,
    backend_family: &str,
    model_family: &str,
    quantization: &str,
    op: &str,
    format: &str,
) -> Result<()> {
    let report = build_route_resolve_report(
        routing_table,
        RouteQuery {
            device_slug: device_slug.to_string(),
            selected_backend: selected_backend.to_string(),
            backend_family: backend_family.to_string(),
            model_family: model_family.to_string(),
            quantization: quantization.to_string(),
            op: op.to_string(),
        },
    )?;
    emit_route_report(&report, format)?;
    if !report.passed {
        bail!("kernel route resolve failed: {}", report.failures.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn capability_rejects_claimable_kernel_without_receipts() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let matrix = dir.path().join("matrix.json");
        fs::write(
            &matrix,
            r#"
{
  "schema_version": 1,
  "matrix_id": "test",
  "device_slug": "amd-5700x-intel-a770",
  "backend_family": "intel-opencl",
  "selected_backend": "intel-arc-a770-opencl",
  "quality_gated_benchmarks_required": true,
  "claim_policy": {
    "no_model_family_claim_without_required_kernels": true,
    "no_benchmark_claim_without_quality_passed": true,
    "no_fallback_for_claimed_kernels": true,
    "device_route_must_be_concrete": true
  },
  "kernels": [
    {
      "kernel": "qk256_i2s_gemv",
      "model_families": ["bitnet"],
      "status": "performance_proven",
      "fallback_allowed_when_claimed": false,
      "proof_receipts": []
    }
  ],
  "not_claims": [
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion"
  ]
}
"#,
        )?;

        let report = build_capability_check_report(&matrix)?;
        anyhow::ensure!(!report.passed, "claimable kernel unexpectedly passed");
        anyhow::ensure!(
            report.missing.iter().any(|failure| failure.contains("requires proof_receipts")),
            "missing report did not mention proof receipts: {:?}",
            report.missing
        );
        anyhow::ensure!(
            report.missing.iter().any(|failure| failure.contains("claimable_kernel_count=0")),
            "missing report did not enforce zero claimable kernels: {:?}",
            report.missing
        );
        Ok(())
    }

    #[test]
    fn route_rejects_wildcard_device_inheritance() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let table = dir.path().join("routes.toml");
        fs::write(
            &table,
            r#"
[[route]]
route_id = "bad.wildcard"
op = "qk256_i2s_gemv"
model_family = "bitnet"
quantization = "i2_s"
backend_family = "intel-opencl"
selected_backend = "intel-arc-a770-opencl"
device_slug = "*"
device_family = "arc_alchemist"
device_models = ["*"]
kernel_variant = "some_kernel"
claim_level = "diagnostic"
fallback_allowed = false
proof_receipts = []
"#,
        )?;

        let report = build_route_resolve_report(
            &table,
            RouteQuery {
                device_slug: "*".to_string(),
                selected_backend: "intel-arc-a770-opencl".to_string(),
                backend_family: "intel-opencl".to_string(),
                model_family: "bitnet".to_string(),
                quantization: "i2_s".to_string(),
                op: "qk256_i2s_gemv".to_string(),
            },
        )?;
        anyhow::ensure!(!report.passed, "wildcard route unexpectedly passed");
        anyhow::ensure!(!report.route_verified, "wildcard route was marked verified");
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("wildcard")),
            "failures did not mention wildcard: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn route_resolve_rejects_invalid_unmatched_routes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let table = dir.path().join("routes.toml");
        fs::write(
            &table,
            r#"
[[route]]
route_id = "a770.bitnet.i2s.qk256"
op = "qk256_i2s_gemv"
model_family = "bitnet"
quantization = "i2_s"
backend_family = "intel-opencl"
selected_backend = "intel-arc-a770-opencl"
device_slug = "amd-5700x-intel-a770"
device_family = "arc_alchemist"
device_models = ["arc-a770-16gb"]
kernel_variant = "a770_opencl_qk256_i2s_route_pending_claim_receipts"
claim_level = "diagnostic"
fallback_allowed = false
proof_receipts = []

[[route]]
route_id = "bad.unmatched"
op = "embedding_lookup"
model_family = "bitnet"
quantization = "i2_s"
backend_family = "intel-opencl"
selected_backend = "intel-arc-a770-opencl"
device_slug = "*"
device_family = "arc_alchemist"
device_models = ["*"]
kernel_variant = "bad"
claim_level = "diagnostic"
fallback_allowed = false
proof_receipts = []
"#,
        )?;

        let report = build_route_resolve_report(
            &table,
            RouteQuery {
                device_slug: "amd-5700x-intel-a770".to_string(),
                selected_backend: "intel-arc-a770-opencl".to_string(),
                backend_family: "intel-opencl".to_string(),
                model_family: "bitnet".to_string(),
                quantization: "i2_s".to_string(),
                op: "qk256_i2s_gemv".to_string(),
            },
        )?;
        anyhow::ensure!(!report.passed, "table with invalid unmatched route unexpectedly passed");
        anyhow::ensure!(
            !report.route_verified,
            "matching route was verified despite table failure"
        );
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("bad.unmatched")),
            "failures did not mention invalid unmatched route: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn diagnostic_a770_route_resolves_without_claim() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let table = dir.path().join("routes.toml");
        fs::write(
            &table,
            r#"
[[route]]
route_id = "a770.bitnet.i2s.qk256"
op = "qk256_i2s_gemv"
model_family = "bitnet"
quantization = "i2_s"
backend_family = "intel-opencl"
selected_backend = "intel-arc-a770-opencl"
device_slug = "amd-5700x-intel-a770"
device_family = "arc_alchemist"
device_models = ["arc-a770-16gb"]
kernel_variant = "a770_opencl_qk256_i2s_route_pending_claim_receipts"
claim_level = "diagnostic"
fallback_allowed = false
proof_receipts = []
"#,
        )?;

        let report = build_route_resolve_report(
            &table,
            RouteQuery {
                device_slug: "amd-5700x-intel-a770".to_string(),
                selected_backend: "intel-arc-a770-opencl".to_string(),
                backend_family: "intel-opencl".to_string(),
                model_family: "bitnet".to_string(),
                quantization: "i2_s".to_string(),
                op: "qk256_i2s_gemv".to_string(),
            },
        )?;
        anyhow::ensure!(report.passed, "diagnostic route failed: {:?}", report.failures);
        anyhow::ensure!(report.route_verified, "diagnostic route was not verified");
        anyhow::ensure!(!report.claimable, "diagnostic route was marked claimable");
        anyhow::ensure!(
            report.classification == "diagnostic_route",
            "unexpected classification {}",
            report.classification
        );
        Ok(())
    }
}
