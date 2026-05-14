use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const CRITICAL_NOT_CLAIMS: &[&str] = &[
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
struct CapabilityMatrix {
    schema_version: u32,
    matrix_id: String,
    device_slug: String,
    backend_family: String,
    selected_backend: String,
    quality_gated_benchmarks_required: bool,
    kernels: Vec<KernelCapability>,
    not_claims: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct KernelCapability {
    kernel: String,
    model_families: Vec<String>,
    status: String,
    fallback_allowed_when_claimed: bool,
    proof_receipts: Vec<String>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct KernelSummary {
    kernel: String,
    model_families: Vec<String>,
    status: String,
    claimable: bool,
    fallback_allowed_when_claimed: bool,
    proof_receipt_count: usize,
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct CapabilityCheckReport {
    diagnostic: &'static str,
    producer: &'static str,
    matrix_path: String,
    schema_version: u32,
    matrix_id: String,
    device_slug: String,
    backend_family: String,
    selected_backend: String,
    quality_gated_benchmarks_required: bool,
    passed: bool,
    kernel_count: usize,
    claimable_kernel_count: usize,
    status_counts: BTreeMap<String, usize>,
    kernels: Vec<KernelSummary>,
    missing: Vec<String>,
    not_claims: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RouteTable {
    #[serde(default)]
    route: Vec<RouteEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RouteEntry {
    route_id: String,
    op: String,
    model_family: String,
    quantization: String,
    backend_family: String,
    selected_backend: String,
    device_slug: String,
    device_family: String,
    device_models: Vec<String>,
    kernel_variant: String,
    claim_level: String,
    fallback_allowed: bool,
    proof_receipts: Vec<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RouteQuery {
    device_slug: String,
    selected_backend: String,
    backend_family: String,
    model_family: String,
    quantization: String,
    op: String,
}

#[derive(Debug, Serialize)]
struct RouteResolveReport {
    diagnostic: &'static str,
    producer: &'static str,
    routing_table: String,
    query: RouteQuery,
    passed: bool,
    route_found: bool,
    route_verified: bool,
    claimable: bool,
    classification: String,
    route: Option<RouteEntry>,
    failures: Vec<String>,
    not_claims: Vec<String>,
}

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

fn build_capability_check_report(matrix_path: &Path) -> Result<CapabilityCheckReport> {
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
        passed: missing.is_empty(),
        kernel_count: kernels.len(),
        claimable_kernel_count,
        status_counts,
        kernels,
        missing,
        not_claims: matrix.not_claims,
    })
}

fn build_route_resolve_report(
    routing_table: &Path,
    query: RouteQuery,
) -> Result<RouteResolveReport> {
    let raw = fs::read_to_string(routing_table)
        .with_context(|| format!("reading {}", routing_table.display()))?;
    let table: RouteTable =
        toml::from_str(&raw).with_context(|| format!("parsing {}", routing_table.display()))?;

    let route = table.route.into_iter().find(|route| {
        route.device_slug == query.device_slug
            && route.selected_backend == query.selected_backend
            && route.backend_family == query.backend_family
            && route.model_family == query.model_family
            && route.quantization == query.quantization
            && route.op == query.op
    });

    let mut failures = Vec::new();
    let mut not_claims = CRITICAL_NOT_CLAIMS.iter().map(|value| (*value).to_string()).collect();
    let mut route_verified = false;
    let mut claimable = false;
    let classification;

    if let Some(route) = &route {
        claimable = is_claimable_status(&route.claim_level);
        route_verified = true;
        if route.device_slug == "*" {
            failures.push(format!("{} uses wildcard device_slug", route.route_id));
            route_verified = false;
        }
        if route.device_models.iter().any(|model| model == "*") {
            failures.push(format!("{} uses wildcard device_models", route.route_id));
            route_verified = false;
        }
        if route.kernel_variant.trim().is_empty() {
            failures.push(format!("{} has empty kernel_variant", route.route_id));
            route_verified = false;
        }
        if route.kernel_variant == "missing" && route.claim_level != "unsupported" {
            failures.push(format!(
                "{} uses missing kernel_variant but claim_level={}",
                route.route_id, route.claim_level
            ));
            route_verified = false;
        }
        if !is_known_status(&route.claim_level) {
            failures
                .push(format!("{} has unknown claim_level {}", route.route_id, route.claim_level));
            route_verified = false;
        }
        if claimable && route.fallback_allowed {
            failures.push(format!("{} is claimable but fallback_allowed=true", route.route_id));
            route_verified = false;
        }
        if status_requires_receipts(&route.claim_level) && route.proof_receipts.is_empty() {
            failures.push(format!(
                "{} claim_level {} requires proof_receipts",
                route.route_id, route.claim_level
            ));
            route_verified = false;
        }
        if !route.not_claims.is_empty() {
            not_claims = route.not_claims.clone();
        }
        classification = if claimable {
            "claimable_route"
        } else if route.claim_level == "unsupported" {
            "unsupported_route"
        } else {
            "diagnostic_route"
        }
        .to_string();
    } else {
        failures.push("no matching route".to_string());
        classification = "route_missing".to_string();
    }

    Ok(RouteResolveReport {
        diagnostic: "kernel_route_resolve",
        producer: "cargo xtask hardware route resolve",
        routing_table: routing_table.display().to_string(),
        query,
        passed: failures.is_empty(),
        route_found: route.is_some(),
        route_verified,
        claimable,
        classification,
        route,
        failures,
        not_claims,
    })
}

fn is_known_status(status: &str) -> bool {
    matches!(
        status,
        "unsupported"
            | "missing"
            | "diagnostic"
            | "load_proven"
            | "parity_proven"
            | "quality_proven"
            | "performance_proven"
            | "resident_proven"
            | "complete"
    )
}

fn is_claimable_status(status: &str) -> bool {
    matches!(status, "quality_proven" | "performance_proven" | "resident_proven" | "complete")
}

fn status_requires_receipts(status: &str) -> bool {
    !matches!(status, "unsupported" | "missing" | "diagnostic")
}

fn emit_capability_report(report: &CapabilityCheckReport, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            println!("a770 kernel capability check: passed={}", report.passed);
            println!("matrix: {}", report.matrix_path);
            println!("device: {}", report.device_slug);
            println!("backend: {}", report.selected_backend);
            println!("kernels: {}", report.kernel_count);
            println!("claimable kernels: {}", report.claimable_kernel_count);
            if !report.missing.is_empty() {
                println!("missing: {}", report.missing.join(", "));
            }
            println!("not_claims: {}", report.not_claims.join(", "));
        }
        other => bail!("unsupported hardware output format: {other}"),
    }
    Ok(())
}

fn emit_route_report(report: &RouteResolveReport, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            println!("kernel route resolve: passed={}", report.passed);
            println!("classification: {}", report.classification);
            println!("route_found: {}", report.route_found);
            println!("route_verified: {}", report.route_verified);
            println!("claimable: {}", report.claimable);
            if let Some(route) = &report.route {
                println!("route_id: {}", route.route_id);
                println!("kernel_variant: {}", route.kernel_variant);
                println!("claim_level: {}", route.claim_level);
            }
            if !report.failures.is_empty() {
                println!("failures: {}", report.failures.join(", "));
            }
            println!("not_claims: {}", report.not_claims.join(", "));
        }
        other => bail!("unsupported hardware output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_rejects_claimable_kernel_without_receipts() {
        let dir = tempfile::tempdir().unwrap();
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
        )
        .unwrap();

        let report = build_capability_check_report(&matrix).unwrap();
        assert!(!report.passed);
        assert!(report.missing.iter().any(|failure| failure.contains("requires proof_receipts")));
    }

    #[test]
    fn route_rejects_wildcard_device_inheritance() {
        let dir = tempfile::tempdir().unwrap();
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
        )
        .unwrap();

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
        )
        .unwrap();
        assert!(!report.passed);
        assert!(!report.route_verified);
        assert!(report.failures.iter().any(|failure| failure.contains("wildcard")));
    }

    #[test]
    fn diagnostic_a770_route_resolves_without_claim() {
        let dir = tempfile::tempdir().unwrap();
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
        )
        .unwrap();

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
        )
        .unwrap();
        assert!(report.passed);
        assert!(report.route_verified);
        assert!(!report.claimable);
        assert_eq!(report.classification, "diagnostic_route");
    }
}
