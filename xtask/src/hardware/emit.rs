use super::types::{CapabilityCheckReport, RouteResolveReport};
use anyhow::{Result, bail};

pub(super) fn emit_capability_report(report: &CapabilityCheckReport, format: &str) -> Result<()> {
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

pub(super) fn emit_route_report(report: &RouteResolveReport, format: &str) -> Result<()> {
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
