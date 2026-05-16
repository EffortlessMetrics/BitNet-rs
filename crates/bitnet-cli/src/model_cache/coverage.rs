//! Model coverage matrix and device status reporting for `bitnet model`.

use anyhow::{Context, Result, anyhow};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

const MODEL_COVERAGE_MATRIX_RELATIVE: &[&str] =
    &["ci", "model-artifacts", "model-coverage-matrix.toml"];

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ModelStatusFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct ModelCoverageMatrix {
    pub(super) schema: u32,
    pub(super) artifact_kind: String,
    pub(super) updated: String,
    pub(super) work_item: String,
    pub(super) claim_boundary: String,
    #[serde(default)]
    pub(super) tier: Vec<ModelCoverageTier>,
    #[serde(default)]
    pub(super) entry: Vec<ModelCoverageEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct ModelCoverageTier {
    pub(super) id: String,
    pub(super) rank: u32,
    #[serde(default)]
    pub(super) requires: Vec<String>,
    pub(super) meaning: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct ModelCoverageEntry {
    pub(super) id: String,
    pub(super) model_class: String,
    pub(super) family: String,
    pub(super) artifact_kind: String,
    #[serde(default)]
    pub(super) contract_id: Option<String>,
    #[serde(default)]
    pub(super) capability_id: Option<String>,
    pub(super) status: String,
    pub(super) current_tier: String,
    pub(super) verifier_surface: String,
    pub(super) tokenizer_authority: String,
    pub(super) prompt_authority: String,
    pub(super) cpu_reference: String,
    #[serde(default)]
    pub(super) accelerator_routes: Vec<String>,
    #[serde(default)]
    pub(super) required_receipts: Vec<String>,
    #[serde(default)]
    pub(super) forbidden_claims: Vec<String>,
    pub(super) next_proof: String,
    pub(super) claim_boundary: String,
    pub(super) claims: ModelCoverageClaims,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct ModelCoverageClaims {
    pub(super) registered: bool,
    pub(super) structurally_valid: bool,
    pub(super) reference_good: bool,
    pub(super) cpu_answer_ready: bool,
    pub(super) accelerator_answer_ready: bool,
    pub(super) benchmark_qualified: bool,
    pub(super) product_cli_ready: bool,
    pub(super) server_ready: bool,
    pub(super) speedup_claim: bool,
    pub(super) full_residency_claim: bool,
    pub(super) bitnet_packed_i2s_qk256_proof: bool,
    pub(super) dense_regular_llm_cuda_proof: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelCoverageMatrixOutput<'a> {
    matrix_path: &'a Path,
    matrix: &'a ModelCoverageMatrix,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelCoverageEntryOutput<'a> {
    matrix_path: &'a Path,
    entry: &'a ModelCoverageEntry,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelStatusDashboard {
    pub(super) schema_version: u32,
    pub(super) device: String,
    pub(super) source: PathBuf,
    pub(super) note: &'static str,
    pub(super) models: Vec<ModelStatusRow>,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelStatusRow {
    pub(super) id: String,
    pub(super) display_name: String,
    pub(super) model_class: String,
    pub(super) route: Option<String>,
    pub(super) tier: String,
    pub(super) status: String,
    pub(super) category: String,
    pub(super) cpu_answer_ready: bool,
    pub(super) accelerator_answer_ready: bool,
    pub(super) benchmark_qualified: bool,
    pub(super) speedup_claim: bool,
    pub(super) server_ready: bool,
    pub(super) full_residency_claim: bool,
    pub(super) bitnet_packed_i2s_qk256_proof: bool,
    pub(super) dense_regular_llm_cuda_proof: bool,
    pub(super) ask: String,
    pub(super) one_token: String,
    pub(super) short_decode: String,
    pub(super) warm_session: String,
    pub(super) benchmark: String,
    pub(super) server: String,
    pub(super) claim_boundary: String,
    pub(super) next_proof: String,
}

pub(super) fn list_model_coverage(
    id: Option<&str>,
    matrix: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let matrix_path = resolve_model_coverage_matrix_path(matrix)?;
    let matrix = read_model_coverage_matrix(&matrix_path)?;

    if let Some(id) = id {
        let entry = find_model_coverage_entry(&matrix, id).ok_or_else(|| {
            let known = matrix.entry.iter().map(|entry| entry.id.as_str()).collect::<Vec<_>>();
            anyhow!("unknown model coverage id `{id}`. Known coverage ids: {}", known.join(", "))
        })?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&ModelCoverageEntryOutput {
                    matrix_path: &matrix_path,
                    entry,
                })?
            );
        } else {
            print_model_coverage_entry(&matrix_path, entry);
        }
        return Ok(());
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&ModelCoverageMatrixOutput {
                matrix_path: &matrix_path,
                matrix: &matrix,
            })?
        );
        return Ok(());
    }

    print_model_coverage_overview(&matrix_path, &matrix);
    Ok(())
}

fn resolve_model_coverage_matrix_path(matrix: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = matrix {
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("BITNET_MODEL_COVERAGE_MATRIX") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(current_dir) = std::env::current_dir()
        && let Some(path) = find_model_coverage_matrix_from(&current_dir)
    {
        return Ok(path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
        && let Some(path) = find_model_coverage_matrix_from(parent)
    {
        return Ok(path);
    }

    anyhow::bail!(
        "could not locate {}; run from the BitNet-rs repo, pass --matrix <PATH>, or set BITNET_MODEL_COVERAGE_MATRIX",
        MODEL_COVERAGE_MATRIX_RELATIVE.join("/")
    )
}

fn find_model_coverage_matrix_from(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let mut candidate = ancestor.to_path_buf();
        for segment in MODEL_COVERAGE_MATRIX_RELATIVE {
            candidate.push(segment);
        }
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

pub(super) fn read_model_coverage_matrix(path: &Path) -> Result<ModelCoverageMatrix> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let matrix: ModelCoverageMatrix = toml::from_str(&text)
        .with_context(|| format!("failed to parse model coverage matrix {}", path.display()))?;
    if matrix.schema != 1 {
        anyhow::bail!("unsupported model coverage schema {}", matrix.schema);
    }
    if matrix.artifact_kind != "model_coverage_matrix" {
        anyhow::bail!("expected artifact_kind=model_coverage_matrix, got {}", matrix.artifact_kind);
    }
    if matrix.tier.is_empty() {
        anyhow::bail!("model coverage matrix has no tiers");
    }
    if matrix.entry.is_empty() {
        anyhow::bail!("model coverage matrix has no entries");
    }
    Ok(matrix)
}

pub(super) fn find_model_coverage_entry<'a>(
    matrix: &'a ModelCoverageMatrix,
    id: &str,
) -> Option<&'a ModelCoverageEntry> {
    matrix.entry.iter().find(|entry| entry.id.eq_ignore_ascii_case(id))
}

fn print_model_coverage_overview(path: &Path, matrix: &ModelCoverageMatrix) {
    println!("Model coverage matrix: {} entries, {} tiers", matrix.entry.len(), matrix.tier.len());
    println!("source: {}", path.display());
    println!("updated: {} ({})", matrix.updated, matrix.work_item);
    println!("claim boundary: {}", matrix.claim_boundary);
    println!();
    println!("{:<42} {:<20} {:<20} {:<28} Routes", "ID", "Class", "Tier", "Status");
    println!("{}", "-".repeat(128));
    for entry in &matrix.entry {
        let routes = if entry.accelerator_routes.is_empty() {
            "-".to_string()
        } else {
            entry.accelerator_routes.join(", ")
        };
        println!(
            "{:<42} {:<20} {:<20} {:<28} {}",
            entry.id, entry.model_class, entry.current_tier, entry.status, routes
        );
    }
    println!();
    println!("Use `bitnet model coverage <id>` for one row, or add --json for receipts tooling.");
}

fn print_model_coverage_entry(path: &Path, entry: &ModelCoverageEntry) {
    println!("coverage: {}", entry.id);
    println!("source: {}", path.display());
    println!("class: {} / {}", entry.model_class, entry.family);
    println!("artifact: {}", entry.artifact_kind);
    if let Some(contract_id) = &entry.contract_id {
        println!("contract: {contract_id}");
    }
    if let Some(capability_id) = &entry.capability_id {
        println!("capability: {capability_id}");
    }
    println!("status: {}", entry.status);
    println!("tier: {}", entry.current_tier);
    println!("verifier: {}", entry.verifier_surface);
    println!("tokenizer authority: {}", entry.tokenizer_authority);
    println!("prompt authority: {}", entry.prompt_authority);
    println!("cpu reference: {}", entry.cpu_reference);
    if entry.accelerator_routes.is_empty() {
        println!("routes: -");
    } else {
        println!("routes: {}", entry.accelerator_routes.join(", "));
    }
    println!("required receipts: {}", entry.required_receipts.join(", "));
    println!("forbidden claims: {}", entry.forbidden_claims.join(", "));
    println!("next proof: {}", entry.next_proof);
    println!("claim boundary: {}", entry.claim_boundary);
    println!("claims:");
    println!("  registered: {}", entry.claims.registered);
    println!("  structurally_valid: {}", entry.claims.structurally_valid);
    println!("  reference_good: {}", entry.claims.reference_good);
    println!("  cpu_answer_ready: {}", entry.claims.cpu_answer_ready);
    println!("  accelerator_answer_ready: {}", entry.claims.accelerator_answer_ready);
    println!("  benchmark_qualified: {}", entry.claims.benchmark_qualified);
    println!("  product_cli_ready: {}", entry.claims.product_cli_ready);
    println!("  server_ready: {}", entry.claims.server_ready);
    println!("  speedup_claim: {}", entry.claims.speedup_claim);
    println!("  full_residency_claim: {}", entry.claims.full_residency_claim);
    println!("  bitnet_packed_i2s_qk256_proof: {}", entry.claims.bitnet_packed_i2s_qk256_proof);
    println!("  dense_regular_llm_cuda_proof: {}", entry.claims.dense_regular_llm_cuda_proof);
}

pub(super) fn print_model_status(
    device: &str,
    matrix: Option<PathBuf>,
    format: ModelStatusFormat,
) -> Result<()> {
    let matrix_path = resolve_model_coverage_matrix_path(matrix)?;
    let matrix = read_model_coverage_matrix(&matrix_path)?;
    let dashboard = model_status_dashboard(device, &matrix_path, &matrix);

    match format {
        ModelStatusFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&dashboard)?);
        }
        ModelStatusFormat::Text => print_model_status_text(&dashboard),
    }

    Ok(())
}

pub(super) fn model_status_dashboard(
    device: &str,
    matrix_path: &Path,
    matrix: &ModelCoverageMatrix,
) -> ModelStatusDashboard {
    let models = matrix
        .entry
        .iter()
        .filter(|entry| model_status_includes_entry(device, entry))
        .map(model_status_row)
        .collect();

    ModelStatusDashboard {
        schema_version: 1,
        device: device.to_string(),
        source: matrix_path.to_path_buf(),
        note: "Read-only model coverage view; it does not probe hardware or create new proof.",
        models,
    }
}

fn model_status_includes_entry(device: &str, entry: &ModelCoverageEntry) -> bool {
    if device != "nvidia-rtx-5070-ti-cuda" {
        return false;
    }
    if entry.claims.product_cli_ready
        && (entry.accelerator_routes.iter().any(|route| route == "bitnet_qk256_cuda")
            || entry.accelerator_routes.iter().any(|route| route == "dense_regular_llm_cuda"))
    {
        return true;
    }

    !entry.claims.product_cli_ready
        && entry.status.contains("candidate")
        && matches!(entry.model_class.as_str(), "dense_slm" | "small_llm")
}

fn model_status_row(entry: &ModelCoverageEntry) -> ModelStatusRow {
    let route = entry.accelerator_routes.first().cloned();
    let category =
        if entry.claims.product_cli_ready { "supported" } else { "candidate" }.to_string();
    let benchmark = benchmark_status(entry);
    let warm_session = warm_session_status(entry);
    let ask = ask_status(entry);
    let one_token = dense_receipt_status(entry, "one_token");
    let short_decode = dense_receipt_status(entry, "short_decode");

    ModelStatusRow {
        id: entry.id.clone(),
        display_name: model_status_display_name(entry),
        model_class: entry.model_class.clone(),
        route,
        tier: entry.current_tier.clone(),
        status: entry.status.clone(),
        category,
        cpu_answer_ready: entry.claims.cpu_answer_ready,
        accelerator_answer_ready: entry.claims.accelerator_answer_ready,
        benchmark_qualified: entry.claims.benchmark_qualified,
        speedup_claim: entry.claims.speedup_claim,
        server_ready: entry.claims.server_ready,
        full_residency_claim: entry.claims.full_residency_claim,
        bitnet_packed_i2s_qk256_proof: entry.claims.bitnet_packed_i2s_qk256_proof,
        dense_regular_llm_cuda_proof: entry.claims.dense_regular_llm_cuda_proof,
        ask,
        one_token,
        short_decode,
        warm_session,
        benchmark,
        server: if entry.claims.server_ready { "ready" } else { "not ready" }.to_string(),
        claim_boundary: entry.claim_boundary.clone(),
        next_proof: entry.next_proof.clone(),
    }
}

pub(super) fn print_model_status_text(dashboard: &ModelStatusDashboard) {
    println!("CUDA model status for {}", dashboard.device);
    println!("source: {}", dashboard.source.display());
    println!("{}", dashboard.note);
    println!();

    print_model_status_group(dashboard, "Supported", "supported");
    println!();
    print_model_status_group(dashboard, "Candidates", "candidate");
}

pub(super) fn print_model_status_group(
    dashboard: &ModelStatusDashboard,
    title: &str,
    category: &str,
) {
    println!("{title}:");
    let mut printed = false;
    for row in dashboard.models.iter().filter(|row| row.category == category) {
        printed = true;
        println!("  {}", row.display_name);
        println!("    id: {}", row.id);
        println!("    class: {}", model_status_class_label(&row.model_class));
        println!("    route: {}", row.route.as_deref().unwrap_or("not ready"));
        println!("    tier: {}", row.tier);
        println!("    cpu answer: {}", ready_label(row.cpu_answer_ready));
        println!("    cuda answer: {}", ready_label(row.accelerator_answer_ready));
        if row.model_class == "dense_slm" && row.category == "supported" {
            println!("    one-token: {}", row.one_token);
            println!("    short-decode: {}", row.short_decode);
        } else {
            println!("    ask: {}", row.ask);
        }
        println!("    warm-session: {}", row.warm_session);
        println!("    benchmark: {}", row.benchmark);
        println!("    speedup: {}", if row.speedup_claim { "qualified" } else { "not qualified" });
        println!("    server: {}", row.server);
        println!(
            "    full residency: {}",
            if row.full_residency_claim { "claimed" } else { "not claimed" }
        );
        println!("    claim boundary: {}", row.claim_boundary);
        if row.category == "candidate" {
            println!("    next proof: {}", row.next_proof);
        }
        println!();
    }

    if !printed {
        println!("  none");
    }
}

fn model_status_display_name(entry: &ModelCoverageEntry) -> String {
    if let Some(id) = &entry.capability_id {
        return id.clone();
    }
    if let Some(id) = entry.verifier_surface.split_whitespace().last()
        && !id.is_empty()
        && id != "only"
        && id != "matrix"
    {
        return id.to_string();
    }
    entry.contract_id.clone().unwrap_or_else(|| entry.id.clone())
}

fn model_status_class_label(model_class: &str) -> &'static str {
    match model_class {
        "bitnet" => "BitNet",
        "dense_slm" => "dense SLM",
        "small_llm" => "small dense LLM",
        "modern_llm_docs_only" => "docs-only modern LLM",
        _ => "model",
    }
}

fn ready_label(ready: bool) -> &'static str {
    if ready { "ready" } else { "not ready" }
}

fn ask_status(entry: &ModelCoverageEntry) -> String {
    if entry.claims.product_cli_ready && entry.claims.accelerator_answer_ready {
        "ready".to_string()
    } else {
        "not ready".to_string()
    }
}

fn dense_receipt_status(entry: &ModelCoverageEntry, receipt_fragment: &str) -> String {
    if entry.required_receipts.iter().any(|receipt| receipt.contains(receipt_fragment)) {
        "ready".to_string()
    } else {
        "not ready".to_string()
    }
}

fn warm_session_status(entry: &ModelCoverageEntry) -> String {
    if entry.required_receipts.iter().any(|receipt| receipt.contains("warm_session"))
        && entry.claims.accelerator_answer_ready
    {
        "ready".to_string()
    } else {
        "not ready".to_string()
    }
}

fn benchmark_status(entry: &ModelCoverageEntry) -> String {
    if entry.claims.benchmark_qualified && entry.claims.speedup_claim {
        return "qualified".to_string();
    }
    if entry.claims.product_cli_ready
        && entry.required_receipts.iter().any(|receipt| receipt.contains("benchmark"))
    {
        return "reviewed, speedup not accepted".to_string();
    }
    "not ready".to_string()
}
