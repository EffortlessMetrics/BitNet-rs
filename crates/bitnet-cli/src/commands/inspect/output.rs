//! Render inspect-command scan results as JSON or human-readable text.
//!
//! Single responsibility: take the model hash, ruleset name, scan results,
//! and strict-mode flag, and write a representation to stdout. The
//! formatters do not decide whether the run "failed" overall — they only
//! describe what was observed; the exit-code decision is the
//! orchestrator's job.

use anyhow::Result;
use serde_json::json;

use super::tensor_scanner::{ScanResults, TensorKind};

/// Shared context passed to either output formatter.
pub(crate) struct OutputContext<'a> {
    pub(crate) model_sha256: &'a str,
    pub(crate) ruleset_name: &'a str,
    pub(crate) strict_mode: bool,
}

/// Pretty-print scan results as JSON.
pub(crate) fn write_json(ctx: &OutputContext, scan: &ScanResults) -> Result<()> {
    let tensors: Vec<_> = scan
        .stats
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "kind": kind_label_json(s.kind),
                "rms": format!("{:.4}", s.rms),
                "status": if s.is_ok { "ok" } else { "suspicious" }
            })
        })
        .collect();

    let total_bad = scan.total_bad();

    let output = json!({
        "model_sha256": ctx.model_sha256,
        "ruleset": ctx.ruleset_name,
        "layernorm": {
            "total": scan.ln_total_count,
            "suspicious": scan.ln_bad_count,
        },
        "projection": {
            "total": scan.proj_total_count,
            "suspicious": scan.proj_bad_count,
        },
        "strict_mode": ctx.strict_mode,
        "tensors": tensors,
        "status": if total_bad > 0 {
            if ctx.strict_mode { "failed" } else { "warning" }
        } else {
            "ok"
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Print scan results as human-readable text with status icons.
pub(crate) fn write_text(ctx: &OutputContext, scan: &ScanResults) -> Result<()> {
    println!("model_sha256: {}", ctx.model_sha256);
    println!("ruleset: {}", ctx.ruleset_name);
    println!();

    for stat in &scan.stats {
        let status_icon = if stat.is_ok { "✅" } else { "❌" };
        let kind_str = kind_label_text(stat.kind);
        println!(
            "{:<64} {:<8} rms={:<8} {}",
            stat.name,
            kind_str,
            format!("{:.4}", stat.rms),
            status_icon
        );
    }

    println!();

    write_kind_summary_text(
        "LN",
        "LayerNorm gamma",
        "layers",
        scan.ln_bad_count,
        scan.ln_total_count,
        ctx.ruleset_name,
        ctx.strict_mode,
    );

    write_kind_summary_text(
        "Projection",
        "projection weights",
        "tensors",
        scan.proj_bad_count,
        scan.proj_total_count,
        ctx.ruleset_name,
        ctx.strict_mode,
    );

    if scan.total_bad() > 0 && ctx.strict_mode {
        println!();
        println!("❌ STRICT MODE: Validation failed");
    }

    Ok(())
}

fn write_kind_summary_text(
    gate_label: &str,
    descriptor: &str,
    unit: &str,
    bad_count: usize,
    total_count: usize,
    ruleset_name: &str,
    strict_mode: bool,
) {
    if bad_count > 0 {
        if strict_mode {
            println!(
                "❌ {} RMS gate failed: {}/{} out of envelope ({})",
                gate_label, bad_count, total_count, ruleset_name
            );
        } else {
            println!(
                "⚠️  WARNING: suspicious {} detected ({}/{} {})",
                descriptor, bad_count, total_count, unit
            );
        }
    } else if total_count > 0 {
        println!("✅ {} RMS gate passed ({})", gate_label, ruleset_name);
    }
}

fn kind_label_json(kind: TensorKind) -> &'static str {
    match kind {
        TensorKind::LayerNorm => "layernorm",
        TensorKind::Projection => "projection",
    }
}

fn kind_label_text(kind: TensorKind) -> &'static str {
    match kind {
        TensorKind::LayerNorm => "[LN]",
        TensorKind::Projection => "[PROJ]",
    }
}
