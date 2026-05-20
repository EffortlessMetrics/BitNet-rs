//! User-facing model cache management.

use anyhow::{Context, Result, anyhow};
use bitnet_models::{
    capability_check::detect_capabilities,
    model_contracts::{BitnetModelContract, bitnet_model_contracts, find_bitnet_model_contract},
};
use clap::{Args, Subcommand, ValueEnum};
use futures::StreamExt;
use humansize::{DECIMAL, format_size};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::process::Command;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use tokio::io::AsyncWriteExt;

const DEFAULT_CACHE_RELATIVE: &[&str] = &["bitnet-rs", "models"];
const MODEL_COVERAGE_MATRIX_RELATIVE: &[&str] =
    &["ci", "model-artifacts", "model-coverage-matrix.toml"];
const LOW_DISK_HEADROOM_BYTES: u64 = 1_073_741_824;
#[cfg(feature = "full-cli")]
pub(crate) const M4_SLM_RUNTIME_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q8_0";

/// Manage supported local model artifacts.
#[derive(Debug, Args)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub action: ModelAction,
}

#[derive(Debug, Subcommand)]
pub enum ModelAction {
    /// Fetch a supported model artifact into the local cache.
    Fetch {
        /// Supported model id, for example qwen2.5-0.5b-instruct-q8_0.
        id: String,

        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Do not use the network; pass if an already verified artifact is cached.
        #[arg(long, default_value_t = false)]
        offline: bool,

        /// Re-download even when a verified artifact already exists.
        #[arg(long, default_value_t = false)]
        force: bool,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Verify a cached or explicit supported model artifact.
    Verify {
        /// Supported model id.
        id: String,

        /// Verify this file instead of the cached path for the model id.
        #[arg(long, value_name = "PATH")]
        path: Option<PathBuf>,

        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Inspect BitNet-family model contracts and claim boundaries.
    Contracts {
        /// Optional contract id or alias. Omit to list every BitNet-family contract.
        id: Option<String>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Inspect cross-family model coverage tiers and claim boundaries.
    Coverage {
        /// Optional coverage entry id. Omit to list every model coverage row.
        id: Option<String>,

        /// Override the coverage matrix path. Defaults to ci/model-artifacts/model-coverage-matrix.toml when run from this repo.
        #[arg(long, value_name = "PATH", env = "BITNET_MODEL_COVERAGE_MATRIX")]
        matrix: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Show user-facing model support for a device from the coverage matrix.
    Status {
        /// Device label to summarize, for example cuda or nvidia-rtx-5070-ti-cuda.
        #[arg(long, value_name = "DEVICE")]
        device: String,

        /// Override the coverage matrix path. Defaults to ci/model-artifacts/model-coverage-matrix.toml when run from this repo.
        #[arg(long, value_name = "PATH", env = "BITNET_MODEL_COVERAGE_MATRIX")]
        matrix: Option<PathBuf>,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ModelStatusFormat::Text)]
        format: ModelStatusFormat,
    },

    /// List supported model artifacts and cache status.
    List {
        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Remove cached model artifacts.
    Prune {
        /// Supported model id to remove. Use --all to remove every supported artifact.
        id: Option<String>,

        /// Remove every supported cached artifact.
        #[arg(long, default_value_t = false)]
        all: bool,

        /// Show what would be removed without deleting files.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Override cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ModelStatusFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct SupportedModel {
    id: &'static str,
    aliases: &'static [&'static str],
    display_name: &'static str,
    repo: &'static str,
    revision: &'static str,
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
    architecture: &'static str,
    quantization: &'static str,
    tokenizer_model: &'static str,
    tokenizer_pre: &'static str,
    tokenizer_sha256: Option<&'static str>,
    tokenizer_sha256_status: &'static str,
    tokenizer_path: Option<&'static str>,
    chat_template: bool,
    license_spdx: Option<&'static str>,
    redistribution_boundary: &'static str,
    prompt_template: &'static str,
    prompt_template_source: &'static str,
    provenance_manifests: &'static [&'static str],
    model_contract: Option<&'static str>,
    apple_m4_cpu_neon_supported: bool,
    support_note: &'static str,
}

#[derive(Debug, Serialize)]
struct CacheStatus {
    model: SupportedModel,
    cache_path: PathBuf,
    metadata_path: PathBuf,
    symlink_target: Option<PathBuf>,
    symlink_status: String,
    present: bool,
    cached: bool,
    size_matches: bool,
    metadata_present: bool,
    verified: Option<bool>,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Serialize)]
struct AppleM4ModelCatalog {
    cache_root: PathBuf,
    default_model_id: &'static str,
    disk: AppleM4DiskSummary,
    claim_boundary: &'static str,
    rows: Vec<AppleM4ModelRow>,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Serialize)]
struct AppleM4DiskSummary {
    probe_path: PathBuf,
    available_bytes: Option<u64>,
    available: Option<String>,
    default_model_headroom_bytes: u64,
    default_model_headroom: String,
    smallest_supported_headroom_bytes: u64,
    smallest_supported_headroom: String,
    largest_supported_headroom_bytes: u64,
    largest_supported_headroom: String,
    low_disk: Option<bool>,
    recommended_first_model_id: Option<String>,
    recommendation: String,
    guidance: String,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Serialize)]
struct AppleM4ModelRow {
    id: String,
    display_name: String,
    state: String,
    cache_state: Option<String>,
    cache_path: Option<PathBuf>,
    size_bytes: Option<u64>,
    size: String,
    quantization: Option<String>,
    tokenizer_authority: Option<String>,
    prompt_authority: Option<String>,
    route: String,
    selection: String,
    reason: String,
    mac_ask_enabled: bool,
    mac_bitnet_warm_enabled: bool,
    mac_chat_enabled: bool,
    mac_ask_chat_enabled: bool,
    mac_serve_enabled: bool,
    proof_status: Option<String>,
    proof_command: Option<String>,
    proof_receipt_path: Option<String>,
    warm_command: Option<String>,
    warm_receipt_path: Option<String>,
    recommended_fetch_headroom_bytes: Option<u64>,
    recommended_fetch_headroom: Option<String>,
    fits_current_disk: Option<bool>,
    disk_state: Option<String>,
    fetch_command: Option<String>,
    verify_command: Option<String>,
    repair_command: Option<String>,
    provenance_manifest: Option<ModelProvenanceManifest>,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Clone, Copy)]
struct AppleM4PolicyModel {
    id: &'static str,
    state: &'static str,
    quantization: &'static str,
    display_name: &'static str,
    selection: &'static str,
    reason: &'static str,
}

#[cfg(feature = "full-cli")]
const APPLE_M4_POLICY_MODELS: &[AppleM4PolicyModel] = &[
    AppleM4PolicyModel {
        id: "qwen3-0.6b-q8_0",
        state: "diagnostic-only",
        quantization: "Q8_0",
        display_name: "upstream dense Qwen diagnostic artifact",
        selection: "not selectable",
        reason: "loader/tokenizer/architecture debugging only; not accepted for M4 local answers.",
    },
    AppleM4PolicyModel {
        id: "qwen-small-instruct-1b-ish",
        state: "candidate",
        quantization: "-",
        display_name: "exact official GGUF required",
        selection: "not selectable",
        reason: "candidate family only until exact source, revision, SHA256, tokenizer, reference output, Rust M4 quality, cache, and receipt gates pass.",
    },
    AppleM4PolicyModel {
        id: "small-gemma-phi-smollm-instruct-gguf",
        state: "candidate",
        quantization: "-",
        display_name: "exact trusted GGUF required",
        selection: "not selectable",
        reason: "cross-family candidate only; likely needs family-specific tokenizer and prompt-template support before M4 promotion.",
    },
    AppleM4PolicyModel {
        id: "qwen3.5-hybrid-vision-moe-state-space",
        state: "rejected",
        quantization: "-",
        display_name: "outside this lane",
        selection: "not selectable",
        reason: "hybrid, vision, MoE, and state-space variants require separate architecture work before reconsideration.",
    },
    AppleM4PolicyModel {
        id: "random-or-unpinned-community-gguf",
        state: "rejected",
        quantization: "-",
        display_name: "untrusted or unpinned",
        selection: "not selectable",
        reason: "artifact hygiene failure: source, revision, size, SHA256, tokenizer authority, and prompt authority are missing or ambiguous.",
    },
];

#[cfg(feature = "full-cli")]
const APPLE_M4_BITNET_ROUTE_BOUNDARY: &str = "BitNet answer-ready artifact authority and local Apple M4 CPU/NEON answer-corpus proof receipts exist via MODEL-ARTIFACT-007/M4-QA-001 evidence. The BitNet Mac route is limited to explicit one-shot `bitnet mac ask` and fixed-prompt `bitnet mac bitnet-warm` with a verified GGUF and external tokenizer authority; `bitnet mac chat` and `bitnet mac serve` remain disabled for BitNet, and dense SLM success must not be counted as BitNet Mac UX proof.";

#[cfg(feature = "full-cli")]
const APPLE_M4_BITNET_PROOF_MODEL_PATH: &str = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
#[cfg(feature = "full-cli")]
const APPLE_M4_BITNET_PROOF_RECEIPT_PATH: &str = "ci/hardware/apple-m4-mac-mini/YYYY-MM-DD/bitnet-local-answer/bitnet-answer-corpus-full-release.json";
#[cfg(feature = "full-cli")]
const APPLE_M4_BITNET_DEFAULT_TOKENIZER_PATH: &str =
    "models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json";
#[cfg(feature = "full-cli")]
const APPLE_M4_BITNET_WARM_RECEIPT_PATH: &str = "ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-warm/bitnet-mac-bitnet-warm-runtime-receipt.json";

#[derive(Debug, Serialize)]
struct VerifyResult {
    id: String,
    path: PathBuf,
    expected_sha256: String,
    actual_sha256: Option<String>,
    expected_bytes: u64,
    actual_bytes: Option<u64>,
    passed: bool,
    model: SupportedModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_contract: Option<VerifyContractSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_capability: Option<VerifyModelCapabilitySummary>,
    artifact_provenance: ModelProvenanceManifest,
}

#[derive(Debug, Clone, Serialize)]
struct VerifyContractSummary {
    id: String,
    model_family: String,
    artifact_format: String,
    artifact_id: Option<String>,
    kernel_family: String,
    status: String,
    architecture_support: Vec<VerifyArchitectureSupportSummary>,
    tokenizer_authority: String,
    prompt_authority: String,
    cpu_oracle: String,
    accelerator_routes: Vec<VerifyRouteSummary>,
    permitted_claims: Vec<String>,
    required_receipts: Vec<String>,
    claim_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct VerifyArchitectureSupportSummary {
    arch: String,
    kernel: String,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct VerifyRouteSummary {
    backend: String,
    route: String,
    status: String,
}

#[derive(Debug, Serialize)]
struct ContractOnlyVerifyResult {
    id: String,
    path: Option<PathBuf>,
    passed: bool,
    supported_artifact: bool,
    reason: String,
    model_contract: VerifyContractSummary,
}

#[derive(Debug, Clone, Serialize)]
struct VerifyModelCapabilitySummary {
    id: String,
    model_family: String,
    model_class: String,
    artifact_format: String,
    quantization: String,
    tokenizer_authority: String,
    prompt_authority: String,
    cpu_oracle: String,
    accelerator_routes: Vec<VerifyRouteSummary>,
    capabilities: Vec<String>,
    permitted_claims: Vec<String>,
    required_receipts: Vec<String>,
    claim_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct ModelProvenanceManifest {
    schema_version: &'static str,
    artifact_kind: &'static str,
    id: String,
    display_name: String,
    source: ProvenanceSource,
    license: ProvenanceLicense,
    artifact: ProvenanceArtifact,
    tokenizer: ProvenanceTokenizer,
    prompt_template: ProvenancePromptTemplate,
    local_cache: ProvenanceLocalCache,
    repair: ProvenanceRepair,
    claim_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenanceSource {
    repo: String,
    revision: String,
    url: String,
    manifests: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenanceLicense {
    spdx: Option<String>,
    redistribution_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenanceArtifact {
    format: &'static str,
    filename: String,
    size_bytes: u64,
    sha256: String,
    architecture: String,
    quantization: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenanceTokenizer {
    authority: String,
    model: String,
    pre_tokenizer: String,
    sha256: Option<String>,
    sha256_status: String,
    external_path: Option<String>,
    source: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenancePromptTemplate {
    identity: String,
    identity_sha256: String,
    source: String,
    chat_template_present: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenanceLocalCache {
    cache_root: PathBuf,
    cache_path: PathBuf,
    metadata_path: PathBuf,
    verify_path: PathBuf,
    path_role: String,
    symlink_target: Option<PathBuf>,
    symlink_status: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProvenanceRepair {
    command: String,
    fetch_command: String,
    verify_command: String,
    prune_command: String,
    cache_state: String,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Clone)]
pub(crate) struct VerifiedCachedModel {
    pub id: String,
    pub display_name: String,
    pub path: PathBuf,
    pub cache_root: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub architecture: String,
    pub quantization: String,
    pub tokenizer_model: String,
    pub tokenizer_pre: String,
    pub chat_template: bool,
    pub support_note: String,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    not(feature = "full-cli"),
    expect(
        dead_code,
        reason = "Apple M4 SLM receipt metadata is consumed by the full-cli answer-corpus path"
    )
)]
pub(crate) struct AppleM4SlmModelReceiptMetadata {
    pub id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub file: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub family: &'static str,
    pub architecture: &'static str,
    pub quantization: &'static str,
    pub tokenizer_authority: &'static str,
    pub prompt_template: &'static str,
    pub prompt_template_source: &'static str,
    pub prompt_template_sha256: String,
}

#[derive(Debug, Serialize)]
struct PruneResult {
    id: String,
    path: PathBuf,
    existed: bool,
    removed: bool,
    dry_run: bool,
    action: String,
    expected_bytes: u64,
    estimated_reclaim_bytes: u64,
    repair_guidance: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelCoverageMatrix {
    schema: u32,
    artifact_kind: String,
    updated: String,
    work_item: String,
    claim_boundary: String,
    #[serde(default)]
    tier: Vec<ModelCoverageTier>,
    #[serde(default)]
    entry: Vec<ModelCoverageEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelCoverageTier {
    id: String,
    rank: u32,
    #[serde(default)]
    requires: Vec<String>,
    meaning: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelCoverageEntry {
    id: String,
    model_class: String,
    family: String,
    artifact_kind: String,
    #[serde(default)]
    contract_id: Option<String>,
    #[serde(default)]
    capability_id: Option<String>,
    status: String,
    current_tier: String,
    verifier_surface: String,
    tokenizer_authority: String,
    prompt_authority: String,
    cpu_reference: String,
    #[serde(default)]
    accelerator_routes: Vec<String>,
    #[serde(default)]
    required_receipts: Vec<String>,
    #[serde(default)]
    forbidden_claims: Vec<String>,
    next_proof: String,
    claim_boundary: String,
    claims: ModelCoverageClaims,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelCoverageClaims {
    registered: bool,
    structurally_valid: bool,
    reference_good: bool,
    cpu_answer_ready: bool,
    accelerator_answer_ready: bool,
    benchmark_qualified: bool,
    product_cli_ready: bool,
    server_ready: bool,
    speedup_claim: bool,
    full_residency_claim: bool,
    bitnet_packed_i2s_qk256_proof: bool,
    dense_regular_llm_cuda_proof: bool,
}

#[derive(Debug, Serialize)]
struct ModelCoverageMatrixOutput<'a> {
    matrix_path: &'a Path,
    matrix: &'a ModelCoverageMatrix,
}

#[derive(Debug, Serialize)]
struct ModelCoverageEntryOutput<'a> {
    matrix_path: &'a Path,
    entry: &'a ModelCoverageEntry,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelStatusDashboard {
    schema_version: u32,
    device: String,
    requested_backend: String,
    selected_backend: Option<String>,
    source: PathBuf,
    note: &'static str,
    models: Vec<ModelStatusRow>,
}

impl ModelStatusDashboard {
    #[cfg(feature = "full-cli")]
    pub(crate) fn next_proof_for_row(&self, row: &str) -> Option<&str> {
        self.models
            .iter()
            .find(|model| model.model_coverage_row == row)
            .map(|model| model.next_proof.as_str())
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelStatusRow {
    id: String,
    model_coverage_row: String,
    display_name: String,
    model_class: String,
    route: Option<String>,
    selected_route: Option<String>,
    requested_backend: String,
    selected_backend: String,
    tier: String,
    current_tier: String,
    status: String,
    category: String,
    fallback_used: Option<bool>,
    cpu_answer_ready: bool,
    accelerator_answer_ready: bool,
    benchmark_qualified: bool,
    product_cli_ready: bool,
    speedup_claim: bool,
    server_ready: bool,
    full_residency_claim: bool,
    bitnet_packed_i2s_qk256_proof: bool,
    dense_regular_llm_cuda_proof: bool,
    ask: String,
    one_token: String,
    short_decode: String,
    warm_session: String,
    benchmark: String,
    server: String,
    server_ready_scope: Option<String>,
    server_scope: Option<String>,
    server_endpoint: Option<String>,
    server_streaming: Option<bool>,
    server_smoke: bool,
    server_reason: Option<String>,
    claim_boundary: String,
    next_proof: String,
}

impl ModelCommand {
    pub async fn execute(self) -> Result<()> {
        match self.action {
            ModelAction::Fetch { id, cache_dir, offline, force, json } => {
                fetch_model(&id, cache_dir, offline, force, json).await
            }
            ModelAction::Verify { id, path, cache_dir, json } => {
                verify_model_command(&id, path, cache_dir, json)
            }
            ModelAction::Contracts { id, json } => list_model_contracts(id.as_deref(), json),
            ModelAction::Coverage { id, matrix, json } => {
                list_model_coverage(id.as_deref(), matrix, json)
            }
            ModelAction::Status { device, matrix, format } => {
                print_model_status(&device, matrix, format)
            }
            ModelAction::List { cache_dir, json } => list_models(cache_dir, json),
            ModelAction::Prune { id, all, dry_run, cache_dir, json } => {
                prune_models(id, all, dry_run, cache_dir, json)
            }
        }
    }
}

#[cfg(feature = "full-cli")]
pub(crate) fn verified_apple_m4_slm_model(
    id: &str,
    cache_dir: Option<PathBuf>,
) -> Result<VerifiedCachedModel> {
    let model = apple_m4_slm_supported_model(id)?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let path = model_path(&cache_root, model);
    let status = cache_status(&cache_root, *model, true)?;
    if !cache_ready(&status) {
        anyhow::bail!(
            "cached Apple M4 SLM model `{}` is not ready at {}.\nState: {}\n{}",
            model.id,
            path.display(),
            cache_state_label(&status),
            apple_m4_cache_repair_guidance(&cache_root, &status)
        );
    }
    let result = verify_model(model, &path, Some(&cache_root))?;
    write_cache_metadata(&cache_root, model, &path, &result)?;
    Ok(VerifiedCachedModel {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        path,
        cache_root,
        sha256: model.sha256.to_string(),
        bytes: model.bytes,
        architecture: model.architecture.to_string(),
        quantization: model.quantization.to_string(),
        tokenizer_model: model.tokenizer_model.to_string(),
        tokenizer_pre: model.tokenizer_pre.to_string(),
        chat_template: model.chat_template,
        support_note: model.support_note.to_string(),
    })
}

#[cfg(feature = "full-cli")]
pub(crate) fn is_apple_m4_bitnet_artifact_id(id: &str) -> bool {
    find_supported_model(id).is_some_and(|model| model.id == "microsoft-bitnet-b1.58-2B-4T-i2s")
}

#[cfg(feature = "full-cli")]
pub(crate) fn verified_apple_m4_bitnet_model(
    id: &str,
    cache_dir: Option<PathBuf>,
    explicit_model_path: Option<PathBuf>,
) -> Result<VerifiedCachedModel> {
    let model = find_supported_model(id)
        .filter(|model| model.id == "microsoft-bitnet-b1.58-2B-4T-i2s")
        .ok_or_else(|| {
            anyhow!(
                "model `{id}` is not the accepted Apple M4 BitNet artifact; use microsoft-bitnet-b1.58-2B-4T-i2s"
            )
        })?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let cache_path = model_path(&cache_root, model);
    let path = explicit_model_path.unwrap_or_else(|| cache_path.clone());
    let result = verify_model(model, &path, Some(&cache_root))?;
    if !result.passed {
        let source = if path == cache_path {
            format!(
                "run `bitnet model fetch {}` or pass --model-path <accepted-bitnet-gguf>",
                model.id
            )
        } else {
            "replace --model-path with the accepted Microsoft I2_S GGUF".to_string()
        };
        anyhow::bail!(
            "Apple M4 BitNet ask requires the accepted GGUF for `{}` at {}; expected bytes={}, sha256={}; got bytes={:?}, sha256={:?}. {source}",
            model.id,
            path.display(),
            model.bytes,
            model.sha256,
            result.actual_bytes,
            result.actual_sha256
        );
    }
    if path == cache_path {
        write_cache_metadata(&cache_root, model, &path, &result)?;
    }
    Ok(VerifiedCachedModel {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        path,
        cache_root,
        sha256: model.sha256.to_string(),
        bytes: model.bytes,
        architecture: model.architecture.to_string(),
        quantization: model.quantization.to_string(),
        tokenizer_model: model.tokenizer_model.to_string(),
        tokenizer_pre: model.tokenizer_pre.to_string(),
        chat_template: model.chat_template,
        support_note: model.support_note.to_string(),
    })
}

#[cfg(feature = "full-cli")]
pub(crate) fn verified_dense_qwen_cuda_model_arg(
    model_arg: &Path,
    cache_dir: Option<PathBuf>,
) -> Result<Option<VerifiedCachedModel>> {
    let model_label = model_arg.as_os_str().to_string_lossy();
    let Some(model) = SUPPORTED_MODELS.iter().find(|model| {
        model.id.eq_ignore_ascii_case(model_label.as_ref())
            || model.aliases.iter().any(|alias| alias.eq_ignore_ascii_case(model_label.as_ref()))
    }) else {
        return Ok(None);
    };

    if model.id != M4_SLM_RUNTIME_MODEL_ID || model.quantization != "Q8_0" {
        anyhow::bail!(
            "model `{}` is not supported for dense RTX 5070 Ti CUDA ask/chat yet; CUDA-UX-003/004 are scoped to qwen2.5-0.5b-instruct-q8_0",
            model.id
        );
    }

    let cache_root = resolve_cache_root(cache_dir)?;
    let path = model_path(&cache_root, model);
    let status = cache_status(&cache_root, *model, true)?;
    if !cache_ready(&status) {
        anyhow::bail!(
            "cached dense Qwen CUDA model `{}` is not ready at {}.\nState: {}\n{}",
            model.id,
            path.display(),
            cache_state_label(&status),
            cache_repair_guidance(&cache_root, &status)
        );
    }
    let result = verify_model(model, &path, Some(&cache_root))?;
    write_cache_metadata(&cache_root, model, &path, &result)?;
    Ok(Some(VerifiedCachedModel {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        path,
        cache_root,
        sha256: model.sha256.to_string(),
        bytes: model.bytes,
        architecture: model.architecture.to_string(),
        quantization: model.quantization.to_string(),
        tokenizer_model: model.tokenizer_model.to_string(),
        tokenizer_pre: model.tokenizer_pre.to_string(),
        chat_template: model.chat_template,
        support_note: model.support_note.to_string(),
    }))
}

#[cfg(feature = "full-cli")]
pub(crate) fn apple_m4_slm_cache_status_json(
    id: &str,
    cache_dir: Option<PathBuf>,
    verify: bool,
) -> Result<serde_json::Value> {
    let model = apple_m4_slm_supported_model(id)?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let status = cache_status(&cache_root, *model, verify)?;
    let ready = cache_ready(&status);
    let cache_state = cache_state_label(&status);
    let artifact_provenance = model_provenance_manifest(
        model,
        &cache_root,
        &status.cache_path,
        cache_state,
        None,
        None,
        status.verified,
    );
    Ok(serde_json::json!({
        "artifact_kind": "apple_m4_slm_model_cache_check",
        "id": model.id,
        "display_name": model.display_name,
        "cache_root": cache_root,
        "cache_path": status.cache_path,
        "metadata_path": status.metadata_path,
        "symlink_target": status.symlink_target.clone(),
        "symlink_status": status.symlink_status.clone(),
        "stale_symlink": status.symlink_status == "stale_symlink",
        "state": cache_state,
        "ready": ready,
        "present": status.present,
        "size_matches": status.size_matches,
        "metadata_present": status.metadata_present,
        "verified": status.verified,
        "expected": {
            "repo": model.repo,
            "revision": model.revision,
            "filename": model.filename,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
        },
        "runtime_support": {
            "apple_m4_cpu_neon": model.apple_m4_cpu_neon_supported,
            "note": model.support_note,
        },
        "artifact_provenance": artifact_provenance,
        "next_step": if ready {
            serde_json::Value::Null
        } else {
            serde_json::json!(apple_m4_cache_repair_guidance(&cache_root, &status))
        },
    }))
}

const SUPPORTED_MODELS: &[SupportedModel] = &[
    SupportedModel {
        id: "microsoft-bitnet-b1.58-2B-4T-i2s",
        aliases: &[
            "microsoft-bitnet-b1.58-2b-4t-i2s",
            "microsoft_bitnet_b158_2b_4t_gguf_i2s_current",
            "microsoft/bitnet-b1.58-2B-4T-gguf",
            "ggml-model-i2_s.gguf",
        ],
        display_name: "Microsoft BitNet-b1.58 2B 4T I2_S",
        repo: "microsoft/bitnet-b1.58-2B-4T-gguf",
        revision: "a1f2f1c765812aa8af3f6eda4a313707064bba15",
        filename: "ggml-model-i2_s.gguf",
        url: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/a1f2f1c765812aa8af3f6eda4a313707064bba15/ggml-model-i2_s.gguf",
        sha256: "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
        bytes: 1_187_801_280,
        architecture: "bitnet_b1_58",
        quantization: "I2_S/QK256",
        tokenizer_model: "gpt2",
        tokenizer_pre: "llama-bpe-external",
        tokenizer_sha256: Some("e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"),
        tokenizer_sha256_status: "external_tokenizer_json_sha256_recorded",
        tokenizer_path: Some("models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json"),
        chat_template: true,
        license_spdx: None,
        redistribution_boundary: "Redistribution boundary recorded instead of a repo-local license assertion: BitNet-rs pins the upstream Microsoft/Hugging Face GGUF URL, revision, size, and SHA256, but does not vendor or redistribute the artifact.",
        prompt_template: "bitnetcpp-answer",
        prompt_template_source: "MODEL-ARTIFACT-007 Microsoft BitNet.cpp answer prompt envelope",
        provenance_manifests: &[
            "ci/model-artifacts/artifact-manifest.toml",
            "ci/model-artifacts/tokenizer-authority.toml",
            "ci/quality/apple-m4-local-answer-model-artifacts.toml",
            "docs/reports/MODEL_ARTIFACT_007_MICROSOFT_BITNETCPP_EXTERNAL_PRETOKENIZER.md",
        ],
        model_contract: Some("microsoft_bitnet_b158_2b_4t_i2s"),
        apple_m4_cpu_neon_supported: false,
        support_note: "Answer-ready BitNet artifact for backend gates when paired with external llama-bpe tokenizer authority and bitnetcpp-answer prompt authority. RTX 5070 Ti CUDA and x86 CPU routes require their own strict receipts; speedup_claim remains false unless profile-qualified.",
    },
    SupportedModel {
        id: "qwen2.5-0.5b-instruct-q8_0",
        aliases: &[],
        display_name: "Qwen2.5 0.5B Instruct Q8_0",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision: "9217f5db79a29953eb74d5343926648285ec7e67",
        filename: "qwen2.5-0.5b-instruct-q8_0.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q8_0.gguf",
        sha256: "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        bytes: 675_710_816,
        architecture: "qwen2",
        quantization: "Q8_0",
        tokenizer_model: "gpt2",
        tokenizer_pre: "qwen2",
        tokenizer_sha256: None,
        tokenizer_sha256_status: "embedded_gguf_metadata_bound_to_model_sha256",
        tokenizer_path: None,
        chat_template: true,
        license_spdx: Some("apache-2.0"),
        redistribution_boundary: "Upstream Hugging Face GGUF artifact is fetched by URL and pinned by revision, size, and SHA256; BitNet-rs does not vendor or redistribute the model file.",
        prompt_template: "qwen2.5",
        prompt_template_source: "GGUF tokenizer.chat_template / Qwen2.5 ChatML identity",
        provenance_manifests: &[
            "ci/quality/apple-m4-slm-answer-first-token-parity.toml",
            "docs/slm/apple-m4-dense-slm-model-support-matrix.md",
        ],
        model_contract: None,
        apple_m4_cpu_neon_supported: true,
        support_note: "Rust-native Apple M4 CPU/NEON SLM baseline artifact; RTX 5070 Ti dense CUDA ask is bounded to the CUDA-UX-003 receipt gate.",
    },
    SupportedModel {
        id: "qwen2.5-0.5b-instruct-q4_k_m",
        aliases: &[],
        display_name: "Qwen2.5 0.5B Instruct Q4_K_M",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision: "9217f5db79a29953eb74d5343926648285ec7e67",
        filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        sha256: "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        bytes: 491_400_032,
        architecture: "qwen2",
        quantization: "Q4_K_M",
        tokenizer_model: "gpt2",
        tokenizer_pre: "qwen2",
        tokenizer_sha256: None,
        tokenizer_sha256_status: "embedded_gguf_metadata_bound_to_model_sha256",
        tokenizer_path: None,
        chat_template: true,
        license_spdx: Some("apache-2.0"),
        redistribution_boundary: "Upstream Hugging Face GGUF artifact is fetched by URL and pinned by revision, size, and SHA256; BitNet-rs does not vendor or redistribute the model file.",
        prompt_template: "qwen2.5",
        prompt_template_source: "GGUF tokenizer.chat_template / Qwen2.5 ChatML identity",
        provenance_manifests: &[
            "ci/quality/apple-m4-slm-answer-model-artifacts.toml",
            "docs/reports/M4_SLM_EX_006_Q4KM_SECOND_MODEL.md",
            "docs/slm/apple-m4-dense-slm-model-support-matrix.md",
        ],
        model_contract: None,
        apple_m4_cpu_neon_supported: true,
        support_note: "Rust-native Apple M4 CPU/NEON storage-conscious SLM artifact.",
    },
    SupportedModel {
        id: "qwen2.5-1.5b-instruct-q4_k_m",
        aliases: &[],
        display_name: "Qwen2.5 1.5B Instruct Q4_K_M",
        repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        revision: "91cad51170dc346986eccefdc2dd33a9da36ead9",
        filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        sha256: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        bytes: 1_117_320_736,
        architecture: "qwen2",
        quantization: "Q4_K_M",
        tokenizer_model: "gpt2",
        tokenizer_pre: "qwen2",
        tokenizer_sha256: None,
        tokenizer_sha256_status: "embedded_gguf_metadata_bound_to_model_sha256",
        tokenizer_path: None,
        chat_template: true,
        license_spdx: Some("apache-2.0"),
        redistribution_boundary: "Upstream Hugging Face GGUF artifact is fetched by URL and pinned by revision, size, and SHA256; BitNet-rs does not vendor or redistribute the model file.",
        prompt_template: "qwen2.5",
        prompt_template_source: "GGUF tokenizer.chat_template / Qwen2.5 ChatML identity",
        provenance_manifests: &[
            "ci/quality/apple-m4-slm-model-breadth-qwen15-reference-sanity.toml",
            "ci/quality/apple-m4-slm-model-breadth-qwen15-rust-m4-quality.toml",
            "docs/slm/apple-m4-dense-slm-model-support-matrix.md",
        ],
        model_contract: None,
        apple_m4_cpu_neon_supported: true,
        support_note: "Rust-native Apple M4 CPU/NEON larger Qwen-class SLM artifact; non-default.",
    },
];

async fn fetch_model(
    id: &str,
    cache_dir: Option<PathBuf>,
    offline: bool,
    force: bool,
    json: bool,
) -> Result<()> {
    let model = supported_model(id)?;
    let cache_root = resolve_cache_root(cache_dir)?;
    let path = model_path(&cache_root, model);

    if path.exists() && !force {
        let result = verify_model(model, &path, Some(&cache_root))?;
        if result.passed {
            write_cache_metadata(&cache_root, model, &path, &result)?;
            return print_fetch_result("cached", &result, json);
        }
    }

    if bitnet_download::offline_enabled(offline) {
        let status = cache_status(&cache_root, *model, true)?;
        anyhow::bail!(
            "model `{}` is not verified in cache and offline mode is enabled.\n{}",
            model.id,
            offline_repair_guidance(&cache_root, &status)
        );
    }

    warn_if_low_disk(&cache_root, model.bytes);
    fs::create_dir_all(model_dir(&cache_root, model))
        .with_context(|| format!("failed to create cache dir {}", cache_root.display()))?;

    let tmp_path = path.with_extension("gguf.part");
    let client = reqwest::Client::new();
    let response = client
        .get(model.url)
        .send()
        .await
        .with_context(|| format!("failed to request {}", model.url))?
        .error_for_status()
        .with_context(|| format!("download request failed for {}", model.url))?;

    let expected_len = response.content_length();
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    let mut downloaded = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("download failed for {}", model.url))?;
        downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    }
    file.flush().await.with_context(|| format!("failed to flush {}", tmp_path.display()))?;
    drop(file);

    if let Err(err) = bitnet_download::validate_downloaded_len(downloaded, expected_len) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err)
            .with_context(|| format!("download length mismatch for {}", model.filename));
    }
    if let Err(err) = bitnet_download::validate_downloaded_len(downloaded, Some(model.bytes)) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err).with_context(|| {
            format!("downloaded size for {} did not match manifest", model.filename)
        });
    }

    let result = verify_model(model, &tmp_path, Some(&cache_root))?;
    if !result.passed {
        let _ = fs::remove_file(&tmp_path);
        anyhow::bail!(
            "downloaded `{}` failed verification: expected sha256 {}, got {:?}",
            model.id,
            model.sha256,
            result.actual_sha256
        );
    }
    replace_cached_file(&tmp_path, &path)
        .with_context(|| format!("failed to move {} to {}", tmp_path.display(), path.display()))?;
    let result = verify_model(model, &path, Some(&cache_root))?;
    write_cache_metadata(&cache_root, model, &path, &result)?;
    print_fetch_result("downloaded", &result, json)
}

fn verify_model_command(
    id: &str,
    path: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let Some(model) = find_supported_model(id) else {
        return verify_contract_without_supported_artifact(id, path, json);
    };
    let cache_root = resolve_cache_root(cache_dir)?;
    let cache_path = model_path(&cache_root, model);
    let path = path.unwrap_or_else(|| cache_path.clone());
    let result = verify_model(model, &path, Some(&cache_root))?;
    if result.passed && path == cache_path {
        write_cache_metadata(&cache_root, model, &path, &result)?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else if result.passed {
        println!("verified {} at {}", model.id, path.display());
        if let Some(contract) = &result.model_contract {
            println!("contract: {} ({}, {})", contract.id, contract.kernel_family, contract.status);
            println!("claim boundary: {}", contract.claim_boundary);
        }
        if let Some(capability) = &result.model_capability {
            println!(
                "capability: {} ({}, {})",
                capability.id, capability.model_family, capability.model_class
            );
            println!("claim boundary: {}", capability.claim_boundary);
        }
    } else {
        println!("verification failed for {} at {}", model.id, path.display());
        eprintln!("{}", verify_failure_guidance(&cache_root, model, &path, &result));
    }
    if result.passed { Ok(()) } else { anyhow::bail!("model `{}` failed verification", model.id) }
}

fn verify_contract_without_supported_artifact(
    id: &str,
    path: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let Some(contract) = find_bitnet_model_contract(id) else {
        return supported_model(id).map(|_| ());
    };
    let result = contract_only_verify_result(id, path, contract);

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("verification rejected for {}", result.id);
        if let Some(path) = &result.path {
            println!("path: {}", path.display());
        }
        println!(
            "contract: {} ({}, {})",
            result.model_contract.id,
            result.model_contract.kernel_family,
            result.model_contract.status
        );
        println!("reason: {}", result.reason);
        println!("claim boundary: {}", result.model_contract.claim_boundary);
    }

    anyhow::bail!(
        "model contract `{}` is not a supported artifact for byte verification",
        result.model_contract.id
    )
}

fn contract_only_verify_result(
    id: &str,
    path: Option<PathBuf>,
    contract: &BitnetModelContract,
) -> ContractOnlyVerifyResult {
    ContractOnlyVerifyResult {
        id: id.to_string(),
        path,
        passed: false,
        supported_artifact: false,
        reason: "known BitNet-family contract has no supported artifact identity and SHA256 registered for `bitnet model verify`; diagnostic or unsupported-path receipts remain allowed according to the contract".to_string(),
        model_contract: contract_summary(contract),
    }
}

fn list_model_contracts(id: Option<&str>, json: bool) -> Result<()> {
    if let Some(id) = id {
        let contract = find_bitnet_model_contract(id).ok_or_else(|| {
            let known = bitnet_model_contracts()
                .iter()
                .map(|contract| contract.id)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("unknown BitNet model contract `{id}`. Known contracts: {known}")
        })?;
        let summary = contract_summary(contract);
        if json {
            println!("{}", serde_json::to_string_pretty(&summary)?);
        } else {
            print_contract_summary(&summary);
        }
        return Ok(());
    }

    let summaries: Vec<_> = bitnet_model_contracts().iter().map(contract_summary).collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
        return Ok(());
    }

    println!("{:<45} {:<16} {:<24} Artifact", "Contract", "Kernel", "Status");
    println!("{}", "-".repeat(112));
    for summary in &summaries {
        println!(
            "{:<45} {:<16} {:<24} {}",
            summary.id,
            summary.kernel_family,
            summary.status,
            summary.artifact_id.as_deref().unwrap_or("-")
        );
    }
    Ok(())
}

fn print_contract_summary(summary: &VerifyContractSummary) {
    println!("contract: {}", summary.id);
    println!("family: {}", summary.model_family);
    println!("format: {}", summary.artifact_format);
    println!("kernel: {}", summary.kernel_family);
    println!("status: {}", summary.status);
    if let Some(artifact_id) = &summary.artifact_id {
        println!("artifact: {artifact_id}");
    }
    if !summary.architecture_support.is_empty() {
        println!("architecture support:");
        for support in &summary.architecture_support {
            println!("  {} / {} ({})", support.arch, support.kernel, support.status);
        }
    }
    println!("tokenizer authority: {}", summary.tokenizer_authority);
    println!("prompt authority: {}", summary.prompt_authority);
    println!("cpu oracle: {}", summary.cpu_oracle);
    if !summary.accelerator_routes.is_empty() {
        println!("routes:");
        for route in &summary.accelerator_routes {
            println!("  {} -> {} ({})", route.backend, route.route, route.status);
        }
    }
    println!("permitted claims: {}", summary.permitted_claims.join(", "));
    println!("required receipts: {}", summary.required_receipts.join(", "));
    println!("claim boundary: {}", summary.claim_boundary);
}

fn list_model_coverage(id: Option<&str>, matrix: Option<PathBuf>, json: bool) -> Result<()> {
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

fn read_model_coverage_matrix(path: &Path) -> Result<ModelCoverageMatrix> {
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

fn find_model_coverage_entry<'a>(
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

fn print_model_status(
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

#[cfg(feature = "full-cli")]
pub(crate) fn model_status_dashboard_for_device(
    device: &str,
    matrix: Option<PathBuf>,
) -> Result<ModelStatusDashboard> {
    let matrix_path = resolve_model_coverage_matrix_path(matrix)?;
    let matrix = read_model_coverage_matrix(&matrix_path)?;
    Ok(model_status_dashboard(device, &matrix_path, &matrix))
}

fn model_status_dashboard(
    device: &str,
    matrix_path: &Path,
    matrix: &ModelCoverageMatrix,
) -> ModelStatusDashboard {
    let selected_backend = model_status_selected_backend(device);
    let models = selected_backend
        .as_deref()
        .map(|backend| {
            matrix
                .entry
                .iter()
                .filter(|entry| model_status_includes_entry(backend, entry))
                .map(|entry| model_status_row(device, backend, entry))
                .collect()
        })
        .unwrap_or_default();

    ModelStatusDashboard {
        schema_version: 1,
        device: device.to_string(),
        requested_backend: device.to_string(),
        selected_backend,
        source: matrix_path.to_path_buf(),
        note: "Read-only model coverage view; it does not probe hardware or create new proof.",
        models,
    }
}

fn model_status_selected_backend(device: &str) -> Option<String> {
    match device {
        "cuda" | "nvidia-rtx-5070-ti-cuda" => Some("nvidia-rtx-5070-ti-cuda".to_string()),
        _ => None,
    }
}

fn model_status_includes_entry(selected_backend: &str, entry: &ModelCoverageEntry) -> bool {
    if selected_backend != "nvidia-rtx-5070-ti-cuda" {
        return false;
    }

    if entry.claims.product_cli_ready
        && (entry.accelerator_routes.iter().any(|route| route == "bitnet_qk256_cuda")
            || entry.accelerator_routes.iter().any(|route| route == "dense_regular_llm_cuda"))
    {
        return true;
    }

    let actionable_candidate = entry.status.contains("candidate")
        || entry.status.contains("blocked")
        || entry.id.ends_with("_candidate");
    !entry.claims.product_cli_ready
        && actionable_candidate
        && matches!(entry.model_class.as_str(), "dense_slm" | "small_llm")
}

fn model_status_row(
    requested_backend: &str,
    selected_backend: &str,
    entry: &ModelCoverageEntry,
) -> ModelStatusRow {
    let route = entry.accelerator_routes.first().cloned();
    let category =
        if entry.claims.product_cli_ready { "supported" } else { "candidate" }.to_string();
    let benchmark = benchmark_status(entry);
    let warm_session = warm_session_status(entry);
    let ask = ask_status(entry);
    let one_token = dense_receipt_status(entry, "one_token");
    let short_decode = dense_receipt_status(entry, "short_decode");
    let server = server_status(entry);

    ModelStatusRow {
        id: entry.id.clone(),
        model_coverage_row: entry.id.clone(),
        display_name: model_status_display_name(entry),
        model_class: entry.model_class.clone(),
        route: route.clone(),
        selected_route: route.clone(),
        requested_backend: requested_backend.to_string(),
        selected_backend: selected_backend.to_string(),
        tier: entry.current_tier.clone(),
        current_tier: entry.current_tier.clone(),
        status: entry.status.clone(),
        category,
        fallback_used: route.is_some().then_some(false),
        cpu_answer_ready: entry.claims.cpu_answer_ready,
        accelerator_answer_ready: entry.claims.accelerator_answer_ready,
        benchmark_qualified: entry.claims.benchmark_qualified,
        product_cli_ready: entry.claims.product_cli_ready,
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
        server: server.label,
        server_ready_scope: server.scope.clone(),
        server_scope: server.scope,
        server_endpoint: server.endpoint,
        server_streaming: server.streaming,
        server_smoke: server.smoke,
        server_reason: server.reason,
        claim_boundary: entry.claim_boundary.clone(),
        next_proof: entry.next_proof.clone(),
    }
}

struct ServerStatus {
    label: String,
    scope: Option<String>,
    endpoint: Option<String>,
    streaming: Option<bool>,
    smoke: bool,
    reason: Option<String>,
}

fn server_status(entry: &ModelCoverageEntry) -> ServerStatus {
    let has_shared_engine_receipt = entry
        .required_receipts
        .iter()
        .any(|receipt| receipt == "server_shared_engine_chat_completion");
    let exact_profile_ready = entry.claims.server_ready
        && has_shared_engine_receipt
        && entry.claim_boundary.contains("exact-profile");
    let server_smoke_only = !entry.claims.server_ready
        && has_shared_engine_receipt
        && entry.claim_boundary.contains("server-smoke evidence");

    if exact_profile_ready {
        return ServerStatus {
            label: "exact-profile ready (/v1/chat/completions, streaming=false)".to_string(),
            scope: Some("exact_profile".to_string()),
            endpoint: Some("/v1/chat/completions".to_string()),
            streaming: Some(false),
            smoke: true,
            reason: None,
        };
    }

    if server_smoke_only {
        return ServerStatus {
            label: "smoke only, not broad-ready".to_string(),
            scope: None,
            endpoint: Some("/v1/chat/completions".to_string()),
            streaming: Some(false),
            smoke: true,
            reason: Some("broad production readiness not qualified".to_string()),
        };
    }

    ServerStatus {
        label: if entry.claims.server_ready { "ready" } else { "not ready" }.to_string(),
        scope: entry.claims.server_ready.then(|| "unspecified".to_string()),
        endpoint: None,
        streaming: None,
        smoke: false,
        reason: None,
    }
}

fn print_model_status_text(dashboard: &ModelStatusDashboard) {
    println!("CUDA model status for {}", dashboard.device);
    println!("requested backend: {}", dashboard.requested_backend);
    if let Some(selected_backend) = &dashboard.selected_backend {
        println!("selected backend: {selected_backend}");
    } else {
        println!("selected backend: none");
    }
    println!("source: {}", dashboard.source.display());
    println!("{}", dashboard.note);
    println!();

    print_model_status_group(dashboard, "Supported", "supported");
    println!();
    print_model_status_group(dashboard, "Candidates", "candidate");
}

fn print_model_status_group(dashboard: &ModelStatusDashboard, title: &str, category: &str) {
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
        println!("    ask: {}", row.ask);
        if matches!(row.route.as_deref(), Some("dense_regular_llm_cuda" | "bitnet_qk256_cuda")) {
            println!("    one-token: {}", row.one_token);
            println!("    short-decode: {}", row.short_decode);
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
    if (entry.claims.product_cli_ready || entry.claims.accelerator_answer_ready)
        && entry.required_receipts.iter().any(|receipt| receipt.contains("benchmark"))
    {
        return "reviewed, speedup not accepted".to_string();
    }
    "not ready".to_string()
}

fn list_models(cache_dir: Option<PathBuf>, json: bool) -> Result<()> {
    let cache_root = resolve_cache_root(cache_dir)?;
    let statuses: Vec<_> = SUPPORTED_MODELS
        .iter()
        .map(|model| cache_status(&cache_root, *model, false))
        .collect::<Result<_>>()?;

    if json {
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }

    println!("Cache: {}", cache_root.display());
    println!("{:<40} {:<13} {:<12} {:<11} Contract", "ID", "Cache", "Quant", "M4 CPU");
    println!("{}", "-".repeat(104));
    for status in statuses {
        let m4_cpu = if status.model.apple_m4_cpu_neon_supported { "supported" } else { "no" };
        let cache_state = cache_state_label(&status);
        let contract = status.model.model_contract.unwrap_or("-");
        println!(
            "{:<40} {:<13} {:<12} {:<11} {}",
            status.model.id, cache_state, status.model.quantization, m4_cpu, contract,
        );
    }
    Ok(())
}

#[cfg(feature = "full-cli")]
pub(crate) fn list_apple_m4_models(cache_dir: Option<PathBuf>, json: bool) -> Result<()> {
    let catalog = apple_m4_model_catalog(cache_dir)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&catalog)?);
        return Ok(());
    }

    println!("Cache: {}", catalog.cache_root.display());
    println!("Default model: {}", catalog.default_model_id);
    let available = catalog.disk.available.as_deref().unwrap_or("unknown");
    let low_disk = catalog
        .disk
        .low_disk
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!(
        "Disk: available={}, default_fetch_headroom={}, low_disk={}",
        available, catalog.disk.default_model_headroom, low_disk
    );
    println!("Recommendation: {}", catalog.disk.recommendation);
    println!("{}", apple_m4_model_next_step(&catalog));
    println!(
        "{:<42} {:<15} {:<13} {:<10} {:<10} {:<16} Selection",
        "ID", "State", "Cache", "Quant", "Size", "Disk"
    );
    println!("{}", "-".repeat(131));
    for row in &catalog.rows {
        println!(
            "{:<42} {:<15} {:<13} {:<10} {:<10} {:<16} {}",
            row.id,
            row.state,
            row.cache_state.as_deref().unwrap_or("-"),
            row.quantization.as_deref().unwrap_or("-"),
            row.size,
            row.disk_state.as_deref().unwrap_or("-"),
            row.selection,
        );
    }
    println!("Claim boundary: {}", catalog.claim_boundary);
    for row in &catalog.rows {
        if let Some(command) = &row.proof_command {
            println!("Proof bridge: {} -> {}", row.id, command);
        }
        if let Some(command) = &row.warm_command {
            println!("Warm bridge: {} -> {}", row.id, command);
        }
    }
    Ok(())
}

#[cfg(feature = "full-cli")]
fn apple_m4_model_next_step(catalog: &AppleM4ModelCatalog) -> String {
    let Some(model_id) = catalog.disk.recommended_first_model_id.as_deref() else {
        return "Next fetch: no supported model has enough disk headroom; prune models or move the cache before fetching."
            .to_string();
    };
    let Some(row) = catalog.rows.iter().find(|row| row.id == model_id) else {
        return format!(
            "Next fetch: recommendation refers to `{model_id}`, but it is not in the selectable catalog rows."
        );
    };
    match (&row.fetch_command, &row.verify_command) {
        (Some(fetch), Some(verify)) => {
            format!("Next fetch: {fetch}\nNext verify: {verify}")
        }
        (Some(fetch), None) => format!("Next fetch: {fetch}"),
        _ => format!(
            "Next fetch: `{model_id}` is not selectable; inspect the recommendation before fetching."
        ),
    }
}

#[cfg(feature = "full-cli")]
pub(crate) fn apple_m4_models_catalog_json(
    cache_dir: Option<PathBuf>,
) -> Result<serde_json::Value> {
    Ok(serde_json::to_value(apple_m4_model_catalog(cache_dir)?)?)
}

#[cfg(feature = "full-cli")]
fn apple_m4_model_catalog(cache_dir: Option<PathBuf>) -> Result<AppleM4ModelCatalog> {
    let cache_root = resolve_cache_root(cache_dir)?;
    let disk = apple_m4_disk_summary(&cache_root)?;
    let mut rows = Vec::new();

    for model in SUPPORTED_MODELS {
        let status = cache_status(&cache_root, *model, false)?;
        rows.push(apple_m4_registered_model_row(&cache_root, &status, disk.available_bytes));
    }
    rows.sort_by_key(|row| apple_m4_state_order(&row.state));
    rows.extend(apple_m4_policy_rows());

    Ok(AppleM4ModelCatalog {
        cache_root,
        default_model_id: M4_SLM_RUNTIME_MODEL_ID,
        disk,
        claim_boundary: "default/supported rows are dense Qwen Apple M4 CPU/NEON answer paths; supported-ask BitNet rows are limited to one-shot ask plus fixed-prompt warm-session proof with explicit GGUF/tokenizer authority; diagnostic-only, candidate, and rejected rows are not selectable M4 answer claims",
        rows,
    })
}

#[cfg(feature = "full-cli")]
fn apple_m4_registered_model_row(
    cache_root: &Path,
    status: &CacheStatus,
    available_bytes: Option<u64>,
) -> AppleM4ModelRow {
    let model = &status.model;
    let (state, selection, route, reason) = if model.id == M4_SLM_RUNTIME_MODEL_ID {
        (
            "default",
            "implicit default or explicit --model-id",
            "apple-m4-cpu-neon",
            model.support_note,
        )
    } else if model.apple_m4_cpu_neon_supported {
        ("supported", "explicit --model-id only", "apple-m4-cpu-neon", model.support_note)
    } else if model.id == "microsoft-bitnet-b1.58-2B-4T-i2s" {
        (
            "supported-ask",
            "explicit --model-id with --model-path/--tokenizer for one-shot ask or fixed warm route only",
            "apple-m4-cpu-neon bitnet one-shot ask + fixed-prompt warm",
            APPLE_M4_BITNET_ROUTE_BOUNDARY,
        )
    } else {
        (
            "rejected",
            "not selectable",
            "none",
            "Registered artifact is not supported by the Apple M4 dense SLM answer lane.",
        )
    };
    let selectable_m4_answer = matches!(state, "default" | "supported");
    let bitnet_ask_only = state == "supported-ask" && model.model_contract.is_some();
    let bitnet_contract_only = bitnet_ask_only;
    let recommended_fetch_headroom_bytes = (selectable_m4_answer || bitnet_ask_only)
        .then(|| recommended_fetch_headroom_bytes(model.bytes));
    let recommended_fetch_headroom =
        recommended_fetch_headroom_bytes.map(|bytes| format_size(bytes, DECIMAL));
    let fits_current_disk = recommended_fetch_headroom_bytes
        .and_then(|required| available_bytes.map(|available| available >= required));
    let disk_state = match fits_current_disk {
        Some(true) => Some("enough-headroom".to_string()),
        Some(false) => Some("low-headroom".to_string()),
        None if selectable_m4_answer || bitnet_ask_only => Some("unknown".to_string()),
        None => None,
    };
    let cache_state = cache_state_label(status);
    let provenance_manifest = (selectable_m4_answer || bitnet_ask_only).then(|| {
        model_provenance_manifest(
            model,
            cache_root,
            &status.cache_path,
            cache_state,
            None,
            None,
            status.verified,
        )
    });
    let repair_command =
        provenance_manifest.as_ref().map(|manifest| manifest.repair.command.clone());

    AppleM4ModelRow {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        state: state.to_string(),
        cache_state: Some(cache_state.to_string()),
        cache_path: Some(status.cache_path.clone()),
        size_bytes: Some(model.bytes),
        size: format_size(model.bytes, DECIMAL),
        quantization: Some(model.quantization.to_string()),
        tokenizer_authority: Some(format!(
            "tokenizer.ggml.model={}, tokenizer.ggml.pre={}",
            model.tokenizer_model, model.tokenizer_pre
        )),
        prompt_authority: Some(if model.chat_template {
            QWEN_OR_CONTRACT_PROMPT_AUTHORITY
                .iter()
                .find_map(|(id, prompt)| (*id == model.id).then_some(*prompt))
                .unwrap_or("chat_template")
                .to_string()
        } else {
            "none".to_string()
        }),
        route: route.to_string(),
        selection: selection.to_string(),
        reason: reason.to_string(),
        mac_ask_enabled: selectable_m4_answer || bitnet_ask_only,
        mac_bitnet_warm_enabled: bitnet_ask_only,
        mac_chat_enabled: selectable_m4_answer,
        mac_ask_chat_enabled: selectable_m4_answer,
        mac_serve_enabled: selectable_m4_answer,
        proof_status: bitnet_contract_only
            .then(|| "answer-corpus-and-warm-session-proof-passed-explicit-artifact".to_string()),
        proof_command: bitnet_contract_only.then(apple_m4_bitnet_proof_command),
        proof_receipt_path: bitnet_contract_only
            .then(|| APPLE_M4_BITNET_PROOF_RECEIPT_PATH.to_string()),
        warm_command: bitnet_contract_only.then(apple_m4_bitnet_warm_command),
        warm_receipt_path: bitnet_contract_only
            .then(|| APPLE_M4_BITNET_WARM_RECEIPT_PATH.to_string()),
        recommended_fetch_headroom_bytes,
        recommended_fetch_headroom,
        fits_current_disk,
        disk_state,
        fetch_command: (selectable_m4_answer || bitnet_ask_only)
            .then(|| model_command("fetch", model, Some(cache_root))),
        verify_command: (selectable_m4_answer || bitnet_ask_only)
            .then(|| model_command("verify", model, Some(cache_root))),
        repair_command,
        provenance_manifest,
    }
}

#[cfg(feature = "full-cli")]
fn apple_m4_bitnet_proof_command() -> String {
    format!(
        "bitnet --device apple-m4-cpu-neon mac bitnet-proof --model {APPLE_M4_BITNET_PROOF_MODEL_PATH} --proof-receipt {APPLE_M4_BITNET_PROOF_RECEIPT_PATH} --strict"
    )
}

#[cfg(feature = "full-cli")]
fn apple_m4_bitnet_warm_command() -> String {
    format!(
        "bitnet --device apple-m4-cpu-neon mac bitnet-warm --model-id microsoft-bitnet-b1.58-2B-4T-i2s --model-path {APPLE_M4_BITNET_PROOF_MODEL_PATH} --tokenizer {APPLE_M4_BITNET_DEFAULT_TOKENIZER_PATH} --max-new-tokens 8 --json-out {APPLE_M4_BITNET_WARM_RECEIPT_PATH}"
    )
}

#[cfg(feature = "full-cli")]
const QWEN_OR_CONTRACT_PROMPT_AUTHORITY: &[(&str, &str)] = &[
    ("qwen2.5-0.5b-instruct-q8_0", "qwen2.5"),
    ("qwen2.5-0.5b-instruct-q4_k_m", "qwen2.5"),
    ("qwen2.5-1.5b-instruct-q4_k_m", "qwen2.5"),
    ("microsoft-bitnet-b1.58-2B-4T-i2s", "bitnetcpp-answer"),
];

#[cfg(feature = "full-cli")]
fn apple_m4_policy_rows() -> Vec<AppleM4ModelRow> {
    APPLE_M4_POLICY_MODELS
        .iter()
        .map(|model| AppleM4ModelRow {
            id: model.id.to_string(),
            display_name: model.display_name.to_string(),
            state: model.state.to_string(),
            cache_state: None,
            cache_path: None,
            size_bytes: None,
            size: "-".to_string(),
            quantization: (model.quantization != "-").then(|| model.quantization.to_string()),
            tokenizer_authority: None,
            prompt_authority: None,
            route: "none".to_string(),
            selection: model.selection.to_string(),
            reason: model.reason.to_string(),
            mac_ask_enabled: false,
            mac_bitnet_warm_enabled: false,
            mac_chat_enabled: false,
            mac_ask_chat_enabled: false,
            mac_serve_enabled: false,
            proof_status: None,
            proof_command: None,
            proof_receipt_path: None,
            warm_command: None,
            warm_receipt_path: None,
            recommended_fetch_headroom_bytes: None,
            recommended_fetch_headroom: None,
            fits_current_disk: None,
            disk_state: None,
            fetch_command: None,
            verify_command: None,
            repair_command: None,
            provenance_manifest: None,
        })
        .collect()
}

#[cfg(feature = "full-cli")]
fn apple_m4_disk_summary(cache_root: &Path) -> Result<AppleM4DiskSummary> {
    let probe_path =
        cache_root.ancestors().find(|path| path.exists()).unwrap_or_else(|| Path::new("."));
    let available = available_bytes(probe_path);
    let default_model = supported_model(M4_SLM_RUNTIME_MODEL_ID)?;
    let supported_models: Vec<_> =
        SUPPORTED_MODELS.iter().filter(|model| model.apple_m4_cpu_neon_supported).collect();
    let smallest_model = supported_models
        .iter()
        .min_by_key(|model| model.bytes)
        .copied()
        .ok_or_else(|| anyhow!("Apple M4 model catalog has no supported dense models"))?;
    let largest_model = supported_models
        .iter()
        .max_by_key(|model| model.bytes)
        .copied()
        .ok_or_else(|| anyhow!("Apple M4 model catalog has no supported dense models"))?;
    let default_model_headroom = recommended_fetch_headroom_bytes(default_model.bytes);
    let smallest_supported_headroom = recommended_fetch_headroom_bytes(smallest_model.bytes);
    let largest_supported_headroom = recommended_fetch_headroom_bytes(largest_model.bytes);
    let (recommended_first_model_id, recommendation) =
        apple_m4_disk_recommendation(available, default_model, smallest_model, largest_model);

    Ok(AppleM4DiskSummary {
        probe_path: probe_path.to_path_buf(),
        available_bytes: available,
        available: available.map(|bytes| format_size(bytes, DECIMAL)),
        default_model_headroom_bytes: default_model_headroom,
        default_model_headroom: format_size(default_model_headroom, DECIMAL),
        smallest_supported_headroom_bytes: smallest_supported_headroom,
        smallest_supported_headroom: format_size(smallest_supported_headroom, DECIMAL),
        largest_supported_headroom_bytes: largest_supported_headroom,
        largest_supported_headroom: format_size(largest_supported_headroom, DECIMAL),
        low_disk: available.map(|bytes| bytes < default_model_headroom),
        recommended_first_model_id,
        recommendation,
        guidance: "When low_disk=true, run `bitnet model prune --all` or set BITNET_MODEL_CACHE_DIR / --cache-dir to a larger volume before fetching."
            .to_string(),
    })
}

#[cfg(feature = "full-cli")]
fn apple_m4_disk_recommendation(
    available: Option<u64>,
    default_model: &SupportedModel,
    smallest_model: &SupportedModel,
    largest_model: &SupportedModel,
) -> (Option<String>, String) {
    let default_required = recommended_fetch_headroom_bytes(default_model.bytes);
    let smallest_required = recommended_fetch_headroom_bytes(smallest_model.bytes);
    let largest_required = recommended_fetch_headroom_bytes(largest_model.bytes);

    let Some(available) = available else {
        return (
            Some(default_model.id.to_string()),
            format!(
                "Disk availability is unknown; start with the default `{}` and use `bitnet model verify` after fetch.",
                default_model.id
            ),
        );
    };

    if available >= largest_required {
        return (
            Some(default_model.id.to_string()),
            format!(
                "Start with the default `{}`. Current headroom can also fetch explicit-only supported models such as `{}`.",
                default_model.id, largest_model.id
            ),
        );
    }

    if available >= default_required {
        return (
            Some(default_model.id.to_string()),
            format!(
                "Start with the default `{}`. Larger explicit-only models may need pruning or a larger cache volume first.",
                default_model.id
            ),
        );
    }

    if available >= smallest_required {
        return (
            Some(smallest_model.id.to_string()),
            format!(
                "Disk is tight for the default; fetch storage-conscious `{}` first or move the cache to a larger volume.",
                smallest_model.id
            ),
        );
    }

    (
        None,
        "Disk headroom is below the smallest supported M4 model fetch envelope; prune cached models or set BITNET_MODEL_CACHE_DIR / --cache-dir before fetching."
            .to_string(),
    )
}

fn recommended_fetch_headroom_bytes(expected_bytes: u64) -> u64 {
    expected_bytes.saturating_mul(2).saturating_add(LOW_DISK_HEADROOM_BYTES)
}

#[cfg(feature = "full-cli")]
fn apple_m4_state_order(state: &str) -> u8 {
    match state {
        "default" => 0,
        "supported" => 1,
        "supported-ask" => 2,
        "blocked" => 3,
        "diagnostic-only" => 4,
        "candidate" => 5,
        "rejected" => 6,
        _ => 7,
    }
}

#[cfg(feature = "full-cli")]
fn apple_m4_slm_supported_model(id: &str) -> Result<&'static SupportedModel> {
    if let Some(model) = find_supported_model(id) {
        if model.apple_m4_cpu_neon_supported {
            return Ok(model);
        }
        let state = if model.id == "microsoft-bitnet-b1.58-2B-4T-i2s" {
            "supported-ask"
        } else if model.model_contract.is_some() {
            "blocked"
        } else {
            "not selectable"
        };
        anyhow::bail!(
            "model `{}` is {state} for Apple M4 CPU/NEON local answers and cannot be used by this dense SLM command. {}\nSelectable dense Apple M4 models: {}.\nUse `bitnet mac models` for cache/disk guidance. {}",
            model.id,
            model.support_note,
            selectable_apple_m4_model_ids(),
            APPLE_M4_BITNET_ROUTE_BOUNDARY
        );
    }

    if let Some(policy) = find_apple_m4_policy_model(id) {
        anyhow::bail!(
            "model `{}` is {} for Apple M4 CPU/NEON local answers and is not selectable. {}\nSelectable Apple M4 models: {}.\nUse `bitnet mac models` for cache/disk guidance before fetching.",
            policy.id,
            policy.state,
            policy.reason,
            selectable_apple_m4_model_ids()
        );
    }

    anyhow::bail!(
        "unsupported Apple M4 model `{id}`. Selectable dense Apple M4 models: {}. Use `bitnet mac models` to inspect default/supported/supported-ask/candidate/rejected states.",
        selectable_apple_m4_model_ids()
    )
}

#[cfg(feature = "full-cli")]
fn selectable_apple_m4_model_ids() -> String {
    SUPPORTED_MODELS
        .iter()
        .filter(|model| model.apple_m4_cpu_neon_supported)
        .map(|model| model.id)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(feature = "full-cli")]
fn find_apple_m4_policy_model(id: &str) -> Option<&'static AppleM4PolicyModel> {
    APPLE_M4_POLICY_MODELS.iter().find(|model| model.id.eq_ignore_ascii_case(id))
}

fn prune_models(
    id: Option<String>,
    all: bool,
    dry_run: bool,
    cache_dir: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    if all && id.is_some() {
        anyhow::bail!("pass either a model id or --all, not both");
    }
    if !all && id.is_none() && !dry_run {
        anyhow::bail!("pass a model id or --all");
    }

    let cache_root = resolve_cache_root(cache_dir)?;
    let scope =
        if all || (dry_run && id.is_none()) { "all_supported_models" } else { "single_model" };
    let models: Vec<_> = if all || (dry_run && id.is_none()) {
        SUPPORTED_MODELS.iter().collect()
    } else {
        vec![supported_model(id.as_deref().ok_or_else(|| anyhow!("pass a model id or --all"))?)?]
    };
    let mut results = Vec::new();

    for model in models {
        let path = model_dir(&cache_root, model);
        let existed = path.exists();
        let estimated_reclaim_bytes =
            if existed { directory_size_bytes(&path).unwrap_or(model.bytes) } else { 0 };
        let removed = if existed && !dry_run {
            fs::remove_dir_all(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            true
        } else {
            false
        };
        let action = if dry_run {
            if existed { "would_remove" } else { "not_cached" }
        } else if removed {
            "removed"
        } else if existed {
            "kept"
        } else {
            "not_cached"
        };
        let repair_guidance = if existed && dry_run {
            format!(
                "dry run only; rerun `{}` without --dry-run to remove this cached artifact",
                model_command("prune", model, Some(&cache_root))
            )
        } else if existed {
            "cache entry was removed; fetch again before running inference".to_string()
        } else {
            format!(
                "cache entry is absent; fetch with `{}`",
                model_command("fetch", model, Some(&cache_root))
            )
        };
        results.push(PruneResult {
            id: model.id.to_string(),
            path,
            existed,
            removed,
            dry_run,
            action: action.to_string(),
            expected_bytes: model.bytes,
            estimated_reclaim_bytes,
            repair_guidance,
        });
    }

    if json {
        let estimated_reclaim_bytes =
            results.iter().map(|result| result.estimated_reclaim_bytes).sum::<u64>();
        let would_remove_count =
            results.iter().filter(|result| result.action == "would_remove").count();
        let removed_count = results.iter().filter(|result| result.removed).count();
        let artifact_kind =
            if dry_run { "bitnet_model_prune_dry_run" } else { "bitnet_model_prune" };
        let payload = serde_json::json!({
            "schema_version": "1.0.0",
            "artifact_kind": artifact_kind,
            "cache_root": cache_root,
            "scope": scope,
            "dry_run": dry_run,
            "deletes_user_data": false,
            "removed_count": removed_count,
            "would_remove_count": would_remove_count,
            "estimated_reclaim_bytes": estimated_reclaim_bytes,
            "estimated_reclaim": format_size(estimated_reclaim_bytes, DECIMAL),
            "guidance": if dry_run {
                "dry run only; no files were deleted. Review results before rerunning prune without --dry-run."
            } else {
                "Only supported BitNet-rs model cache entries under cache_root were considered."
            },
            "claim_boundary": "Model cache prune receipt only; does not run inference, delete user data outside the BitNet-rs model cache, or prove model quality/performance.",
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        for result in &results {
            let action = if result.dry_run {
                "would remove"
            } else if result.removed {
                "removed"
            } else if result.existed {
                "kept"
            } else {
                "not cached"
            };
            println!("{action}: {} ({})", result.id, result.path.display());
            if !result.existed
                && let Ok(model) = supported_model(&result.id)
            {
                println!(
                    "next: fetch it with `{}`",
                    model_command("fetch", model, Some(&cache_root))
                );
            }
        }
    }
    Ok(())
}

fn supported_model(id: &str) -> Result<&'static SupportedModel> {
    find_supported_model(id).ok_or_else(|| {
        let known = SUPPORTED_MODELS.iter().map(|model| model.id).collect::<Vec<_>>().join(", ");
        anyhow!("unsupported model `{id}`. Supported models: {known}")
    })
}

#[cfg_attr(
    not(feature = "full-cli"),
    expect(
        dead_code,
        reason = "Apple M4 SLM receipt metadata is consumed by the full-cli answer-corpus path"
    )
)]
pub(crate) fn apple_m4_slm_model_receipt_metadata(
    id: &str,
) -> Result<AppleM4SlmModelReceiptMetadata> {
    let Some(model) = find_supported_model(id) else {
        anyhow::bail!(
            "unsupported Apple M4 dense SLM model `{id}`. Selectable dense Apple M4 models: {}",
            selectable_apple_m4_slm_model_ids()
        );
    };
    if !model.apple_m4_cpu_neon_supported {
        anyhow::bail!(
            "model `{}` is not selectable for dense Apple M4 SLM answer-corpus receipts. Selectable dense Apple M4 models: {}",
            model.id,
            selectable_apple_m4_slm_model_ids()
        );
    }

    Ok(AppleM4SlmModelReceiptMetadata {
        id: model.id,
        repo: model.repo,
        revision: model.revision,
        file: model.filename,
        sha256: model.sha256,
        bytes: model.bytes,
        family: "qwen",
        architecture: model.architecture,
        quantization: model.quantization,
        tokenizer_authority: model.tokenizer_pre,
        prompt_template: model.prompt_template,
        prompt_template_source: model.prompt_template_source,
        prompt_template_sha256: sha256_hex(model.prompt_template.as_bytes()),
    })
}

fn selectable_apple_m4_slm_model_ids() -> String {
    SUPPORTED_MODELS
        .iter()
        .filter(|model| model.apple_m4_cpu_neon_supported)
        .map(|model| model.id)
        .collect::<Vec<_>>()
        .join(", ")
}

fn find_supported_model(id: &str) -> Option<&'static SupportedModel> {
    let needle = id.to_ascii_lowercase();
    SUPPORTED_MODELS.iter().find(|model| {
        model.id.eq_ignore_ascii_case(id)
            || model.aliases.iter().any(|alias| alias.to_ascii_lowercase() == needle)
    })
}

fn resolve_cache_root(cache_dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = cache_dir {
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("BITNET_MODEL_CACHE_DIR") {
        return Ok(PathBuf::from(path));
    }
    let mut root = dirs::cache_dir().ok_or_else(|| {
        anyhow!(
            "could not resolve user cache directory; pass --cache-dir or set BITNET_MODEL_CACHE_DIR"
        )
    })?;
    for segment in DEFAULT_CACHE_RELATIVE {
        root.push(segment);
    }
    Ok(root)
}

fn model_dir(cache_root: &Path, model: &SupportedModel) -> PathBuf {
    cache_root.join(model.id)
}

fn model_path(cache_root: &Path, model: &SupportedModel) -> PathBuf {
    model_dir(cache_root, model).join(model.filename)
}

fn metadata_path(cache_root: &Path, model: &SupportedModel) -> PathBuf {
    model_dir(cache_root, model).join("bitnet-model-cache.json")
}

fn cache_status(cache_root: &Path, model: SupportedModel, verify: bool) -> Result<CacheStatus> {
    let path = model_path(cache_root, &model);
    let metadata = metadata_path(cache_root, &model);
    let symlink_target = symlink_target(&path);
    let symlink_status = symlink_status(&path, symlink_target.as_ref());
    let present = path.exists();
    let size_matches = present
        && fs::metadata(&path).map(|metadata| metadata.len() == model.bytes).unwrap_or(false);
    let metadata_present = metadata.exists();
    let cached = present && size_matches && metadata_present;
    let verified = if present && verify {
        Some(verify_model(&model, &path, Some(cache_root))?.passed)
    } else {
        None
    };
    Ok(CacheStatus {
        model,
        cache_path: path,
        metadata_path: metadata.clone(),
        symlink_target,
        symlink_status,
        present,
        cached,
        size_matches,
        metadata_present,
        verified,
    })
}

#[cfg(any(test, feature = "full-cli"))]
fn cache_ready(status: &CacheStatus) -> bool {
    status.present
        && status.size_matches
        && status.metadata_present
        && status.verified.unwrap_or(true)
}

fn cache_state_label(status: &CacheStatus) -> &'static str {
    if status.symlink_status == "stale_symlink" {
        "stale-symlink"
    } else if !status.present {
        "missing"
    } else if !status.size_matches {
        "invalid-size"
    } else if status.verified == Some(false) {
        "invalid-sha"
    } else if !status.metadata_present {
        "unverified"
    } else {
        "ready"
    }
}

#[cfg(any(test, feature = "full-cli"))]
fn cache_repair_guidance(cache_root: &Path, status: &CacheStatus) -> String {
    let model = &status.model;
    let fetch = model_command("fetch", model, Some(cache_root));
    let verify = model_command("verify", model, Some(cache_root));
    let prune = model_command("prune", model, Some(cache_root));
    match cache_state_label(status) {
        "missing" => format!(
            "First run: the supported model is missing from {}. Run `{fetch}`.",
            status.cache_path.display()
        ),
        "invalid-size" => format!(
            "Cache repair: {} has the wrong size or is incomplete. Run `{prune}`, then `{fetch}`.",
            status.cache_path.display()
        ),
        "invalid-sha" => format!(
            "Cache repair: {} failed SHA256 verification. Run `{prune}`, then `{fetch}`.",
            status.cache_path.display()
        ),
        "unverified" => format!(
            "Cache repair: {} is present but missing BitNet-rs model-cache metadata. Run `{verify}`; if verification fails, run `{prune}` then `{fetch}`.",
            status.cache_path.display()
        ),
        "stale-symlink" => format!(
            "Cache repair: {} is a stale symlink. Run `{prune}`, then `{fetch}`, or replace the symlink target with the verified artifact.",
            status.cache_path.display()
        ),
        _ => format!("Cache is ready. Optional check: `{verify}`."),
    }
}

#[cfg(feature = "full-cli")]
fn apple_m4_cache_repair_guidance(cache_root: &Path, status: &CacheStatus) -> String {
    let base = cache_repair_guidance(cache_root, status);
    let models_command = format!("bitnet mac models --cache-dir {}", shellish_path(cache_root));
    let disk_guidance = apple_m4_disk_summary(cache_root)
        .map(|summary| summary.recommendation)
        .unwrap_or_else(|_| {
            "Disk availability could not be checked; inspect cache state before fetching."
                .to_string()
        });
    if cache_state_label(status) == "missing" {
        format!(
            "{base} First-run model selection: run `{models_command}` to inspect default/supported/supported-ask model states, BitNet warm-route readiness, and disk headroom. Disk guidance: {disk_guidance}"
        )
    } else {
        format!(
            "{base} Inspect current Mac model/cache/disk state with `{models_command}`. Disk guidance: {disk_guidance}"
        )
    }
}

fn offline_repair_guidance(cache_root: &Path, status: &CacheStatus) -> String {
    let model = &status.model;
    let fetch = model_command("fetch", model, Some(cache_root));
    let verify = model_command("verify", model, Some(cache_root));
    let prune = model_command("prune", model, Some(cache_root));
    format!(
        "Offline mode cannot repair the {} cache state. Disable offline mode and run `{fetch}`, or pre-seed the GGUF at {} and run `{verify}`. If a partial artifact is present, run `{prune}` before fetching.",
        cache_state_label(status),
        status.cache_path.display()
    )
}

fn verify_failure_guidance(
    cache_root: &Path,
    model: &SupportedModel,
    path: &Path,
    result: &VerifyResult,
) -> String {
    let actual_bytes =
        result.actual_bytes.map(|bytes| bytes.to_string()).unwrap_or_else(|| "missing".to_string());
    let actual_sha = result.actual_sha256.as_deref().unwrap_or("missing");
    let fetch = model_command("fetch", model, Some(cache_root));
    let prune = model_command("prune", model, Some(cache_root));
    format!(
        "repair: {} is not the supported artifact (expected bytes={}, sha256={}; got bytes={}, sha256={}). If this is the cached file, run `{prune}` then `{fetch}`. If this is an explicit --path, replace it with a verified artifact.",
        path.display(),
        model.bytes,
        model.sha256,
        actual_bytes,
        actual_sha
    )
}

fn verify_model(
    model: &SupportedModel,
    path: &Path,
    cache_root: Option<&Path>,
) -> Result<VerifyResult> {
    let default_cache_root;
    let cache_root = if let Some(cache_root) = cache_root {
        cache_root
    } else {
        default_cache_root = resolve_cache_root(None)?;
        &default_cache_root
    };
    let metadata = fs::metadata(path).ok();
    let actual_bytes = metadata.as_ref().map(fs::Metadata::len);
    let actual_sha256 = if metadata.is_some() { Some(compute_sha256(path)?) } else { None };
    let passed =
        actual_bytes == Some(model.bytes) && actual_sha256.as_deref() == Some(model.sha256);
    let cache_state = verification_cache_state(
        model,
        path,
        cache_root,
        actual_bytes,
        actual_sha256.as_deref(),
        passed,
    );
    let artifact_provenance = model_provenance_manifest(
        model,
        cache_root,
        path,
        cache_state,
        actual_bytes,
        actual_sha256.as_deref(),
        Some(passed),
    );
    Ok(VerifyResult {
        id: model.id.to_string(),
        path: path.to_path_buf(),
        expected_sha256: model.sha256.to_string(),
        actual_sha256,
        expected_bytes: model.bytes,
        actual_bytes,
        passed,
        model: *model,
        model_contract: model_contract_summary(model)?,
        model_capability: model_capability_summary(model),
        artifact_provenance,
    })
}

fn verification_cache_state(
    model: &SupportedModel,
    path: &Path,
    cache_root: &Path,
    actual_bytes: Option<u64>,
    actual_sha256: Option<&str>,
    passed: bool,
) -> &'static str {
    if passed {
        return "ready";
    }
    if actual_bytes.is_none() {
        return "missing";
    }
    if actual_bytes != Some(model.bytes) {
        return "invalid-size";
    }
    if actual_sha256 != Some(model.sha256) {
        return "invalid-sha";
    }
    if path != model_path(cache_root, model).as_path() {
        return "explicit-path-invalid";
    }
    "invalid-artifact"
}

fn model_provenance_manifest(
    model: &SupportedModel,
    cache_root: &Path,
    verify_path: &Path,
    cache_state: &str,
    _actual_bytes: Option<u64>,
    _actual_sha256: Option<&str>,
    passed: Option<bool>,
) -> ModelProvenanceManifest {
    let cache_path = model_path(cache_root, model);
    let metadata_path = metadata_path(cache_root, model);
    let symlink_target = symlink_target(verify_path);
    let path_role =
        if verify_path == cache_path.as_path() { "cache_path" } else { "explicit_path" };
    let fetch_command = model_command("fetch", model, Some(cache_root));
    let verify_command = if verify_path == cache_path.as_path() {
        model_command("verify", model, Some(cache_root))
    } else {
        format!(
            "{} --path {}",
            model_command("verify", model, Some(cache_root)),
            shellish_path(verify_path)
        )
    };
    let prune_command = model_command("prune", model, Some(cache_root));
    let command = provenance_repair_command(
        model,
        cache_root,
        verify_path,
        cache_state,
        passed,
        &fetch_command,
        &verify_command,
        &prune_command,
    );
    let prompt_identity = model.prompt_template.to_string();

    ModelProvenanceManifest {
        schema_version: "1.0.0",
        artifact_kind: "m4_supported_model_provenance",
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        source: ProvenanceSource {
            repo: model.repo.to_string(),
            revision: model.revision.to_string(),
            url: model.url.to_string(),
            manifests: model.provenance_manifests.iter().map(|path| (*path).to_string()).collect(),
        },
        license: ProvenanceLicense {
            spdx: model.license_spdx.map(str::to_string),
            redistribution_boundary: model.redistribution_boundary.to_string(),
        },
        artifact: ProvenanceArtifact {
            format: "gguf",
            filename: model.filename.to_string(),
            size_bytes: model.bytes,
            sha256: model.sha256.to_string(),
            architecture: model.architecture.to_string(),
            quantization: model.quantization.to_string(),
        },
        tokenizer: ProvenanceTokenizer {
            authority: model.tokenizer_pre.to_string(),
            model: model.tokenizer_model.to_string(),
            pre_tokenizer: model.tokenizer_pre.to_string(),
            sha256: model.tokenizer_sha256.map(str::to_string),
            sha256_status: model.tokenizer_sha256_status.to_string(),
            external_path: model.tokenizer_path.map(str::to_string),
            source: if model.tokenizer_path.is_some() {
                "external_tokenizer_file"
            } else {
                "gguf_metadata"
            }
            .to_string(),
        },
        prompt_template: ProvenancePromptTemplate {
            identity_sha256: stable_sha256_hex(prompt_identity.as_bytes()),
            identity: prompt_identity,
            source: model.prompt_template_source.to_string(),
            chat_template_present: model.chat_template,
        },
        local_cache: ProvenanceLocalCache {
            cache_root: cache_root.to_path_buf(),
            cache_path,
            metadata_path,
            verify_path: verify_path.to_path_buf(),
            path_role: path_role.to_string(),
            symlink_target: symlink_target.clone(),
            symlink_status: if symlink_target.is_some() { "symlink" } else { "not_symlink" }
                .to_string(),
        },
        repair: ProvenanceRepair {
            command,
            fetch_command,
            verify_command,
            prune_command,
            cache_state: cache_state.to_string(),
        },
        claim_boundary: artifact_provenance_claim_boundary(model).to_string(),
    }
}

fn provenance_repair_command(
    model: &SupportedModel,
    cache_root: &Path,
    verify_path: &Path,
    cache_state: &str,
    passed: Option<bool>,
    fetch_command: &str,
    verify_command: &str,
    prune_command: &str,
) -> String {
    if passed == Some(true) {
        return verify_command.to_string();
    }
    if verify_path != model_path(cache_root, model).as_path() && passed == Some(false) {
        return format!(
            "replace explicit --path with the supported artifact, then run `{verify_command}`"
        );
    }
    match cache_state {
        "missing" => fetch_command.to_string(),
        "invalid-size" | "invalid-sha" | "invalid-artifact" | "stale-symlink" => {
            format!("{prune_command} && {fetch_command}")
        }
        "unverified" => verify_command.to_string(),
        _ => verify_command.to_string(),
    }
}

fn artifact_provenance_claim_boundary(model: &SupportedModel) -> &'static str {
    if model.id == "microsoft-bitnet-b1.58-2B-4T-i2s" {
        "Artifact provenance only for the accepted BitNet GGUF/tokenizer identity; does not prove BitNet chat, BitNet serve, Metal, QK256 acceleration, speedup, broad quality, or broad performance."
    } else if model.apple_m4_cpu_neon_supported {
        "Artifact provenance only for supported dense Qwen Apple M4 CPU/NEON model identity; runtime quality and performance require separate eval, benchmark, and regression receipts."
    } else {
        "Artifact provenance only; not a supported M4 runtime claim."
    }
}

fn symlink_target(path: &Path) -> Option<PathBuf> {
    fs::symlink_metadata(path)
        .ok()
        .filter(|metadata| metadata.file_type().is_symlink())
        .and_then(|_| fs::read_link(path).ok())
}

fn symlink_status(path: &Path, target: Option<&PathBuf>) -> String {
    let Some(target) = target else {
        return "not_symlink".to_string();
    };
    let resolved = if target.is_absolute() {
        target.clone()
    } else {
        path.parent().unwrap_or_else(|| Path::new(".")).join(target)
    };
    if resolved.exists() { "symlink".to_string() } else { "stale_symlink".to_string() }
}

fn directory_size_bytes(path: &Path) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if metadata.file_type().is_symlink() {
        return Ok(0);
    }
    let mut total = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        total += directory_size_bytes(&entry.path()).unwrap_or(0);
    }
    Ok(total)
}

fn stable_sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn model_contract_summary(model: &SupportedModel) -> Result<Option<VerifyContractSummary>> {
    let Some(label) = model.model_contract else {
        return Ok(None);
    };
    let contract = find_bitnet_model_contract(label)
        .ok_or_else(|| anyhow!("model `{}` references unknown contract `{label}`", model.id))?;
    Ok(Some(contract_summary(contract)))
}

fn model_capability_summary(model: &SupportedModel) -> Option<VerifyModelCapabilitySummary> {
    if model.model_contract.is_some() || !is_qwen_dense_model(model) {
        return None;
    }

    let capability_report = detect_capabilities(model.architecture);
    let mut capabilities: Vec<_> = capability_report
        .capabilities
        .iter()
        .map(|capability| capability.name().to_string())
        .collect();
    capabilities.sort();

    let supported_cpu_neon = model.apple_m4_cpu_neon_supported;

    let (id, cpu_oracle, accelerator_routes, permitted_claims, required_receipts, claim_boundary) =
        if supported_cpu_neon {
            let capability_id = if model.id.contains("1.5b") {
                "qwen_dense_slm_1_5b_q4_k_m"
            } else if model.quantization.eq_ignore_ascii_case("Q8_0") {
                "qwen_dense_slm_q8_0"
            } else if model.quantization.eq_ignore_ascii_case("Q4_K_M") {
                "qwen_dense_slm_q4_k_m"
            } else {
                "qwen_dense_slm_supported"
            };
            let mut accelerator_routes = vec![VerifyRouteSummary {
                backend: "apple-m4-cpu-neon".to_string(),
                route: "dense_qwen_cpu_neon_slm".to_string(),
                status: "answer_lane".to_string(),
            }];
            let mut permitted_claims = vec![
                "artifact_inspection".to_string(),
                "model_verify".to_string(),
                "apple_m4_cpu_neon_slm_answer".to_string(),
            ];
            let mut required_receipts = vec![
                "model_verify".to_string(),
                "slm_answer_receipt".to_string(),
                "fallback_free_backend_receipt".to_string(),
            ];
            let claim_boundary = if model.quantization.eq_ignore_ascii_case("Q8_0") {
                accelerator_routes.push(VerifyRouteSummary {
                    backend: "nvidia-rtx-5070-ti-cuda".to_string(),
                    route: "dense_regular_llm_cuda".to_string(),
                    status: "bounded_ask_chat_receipt_gate".to_string(),
                });
                permitted_claims.push("nvidia_rtx_5070_ti_dense_cuda_ask".to_string());
                permitted_claims.push("nvidia_rtx_5070_ti_dense_cuda_chat".to_string());
                required_receipts.push("dense_gguf_qwen_ask_strict_cuda_proof".to_string());
                required_receipts.push("dense_gguf_qwen_chat_strict_cuda_proof".to_string());
                "Dense Qwen SLM artifact for the Apple M4 CPU/NEON answer lane and bounded RTX 5070 Ti dense CUDA ask/chat gates; does not prove broad dense GGUF inference, BitNet packed QK256, server, speedup, or full-residency claims."
            } else {
                "Dense Qwen SLM artifact for the Apple M4 CPU/NEON answer lane; does not prove dense CUDA, BitNet packed QK256, server, speedup, or full-residency claims."
            };
            (
                capability_id,
                "apple_m4_cpu_neon_slm_answer_lane",
                accelerator_routes,
                permitted_claims,
                required_receipts,
                claim_boundary,
            )
        } else {
            (
                "qwen_dense_slm_q4_k_m_storage_reference",
                "none_strict_rust_execution_unsupported",
                Vec::new(),
                vec!["artifact_inspection".to_string(), "storage_reference".to_string()],
                vec!["model_verify".to_string(), "unsupported_execution_receipt".to_string()],
                "Storage/reference dense Qwen artifact; strict Rust execution remains unsupported and does not prove CPU, CUDA, server, speedup, or full-residency claims.",
            )
        };

    Some(VerifyModelCapabilitySummary {
        id: id.to_string(),
        model_family: "qwen".to_string(),
        model_class: "dense_slm_gguf".to_string(),
        artifact_format: "gguf".to_string(),
        quantization: model.quantization.to_string(),
        tokenizer_authority: model.tokenizer_pre.to_string(),
        prompt_authority: "qwen2.5".to_string(),
        cpu_oracle: cpu_oracle.to_string(),
        accelerator_routes,
        capabilities,
        permitted_claims,
        required_receipts,
        claim_boundary: claim_boundary.to_string(),
    })
}

fn is_qwen_dense_model(model: &SupportedModel) -> bool {
    model.architecture.to_ascii_lowercase().contains("qwen")
        || model.tokenizer_pre.eq_ignore_ascii_case("qwen2")
}

fn contract_summary(contract: &BitnetModelContract) -> VerifyContractSummary {
    VerifyContractSummary {
        id: contract.id.to_string(),
        model_family: contract.model_family.to_string(),
        artifact_format: contract.artifact_format.as_str().to_string(),
        artifact_id: contract.artifact_id.map(str::to_string),
        kernel_family: contract.kernel_family.as_str().to_string(),
        status: contract.status.as_str().to_string(),
        architecture_support: contract
            .architecture_support
            .iter()
            .map(|support| VerifyArchitectureSupportSummary {
                arch: support.arch.to_string(),
                kernel: support.kernel.to_string(),
                status: support.status.to_string(),
            })
            .collect(),
        tokenizer_authority: contract.tokenizer_authority.to_string(),
        prompt_authority: contract.prompt_authority.to_string(),
        cpu_oracle: contract.cpu_oracle.to_string(),
        accelerator_routes: contract
            .accelerator_routes
            .iter()
            .map(|route| VerifyRouteSummary {
                backend: route.backend.to_string(),
                route: route.route.to_string(),
                status: route.status.to_string(),
            })
            .collect(),
        permitted_claims: contract
            .permitted_claims
            .iter()
            .map(|claim| claim.as_str().to_string())
            .collect(),
        required_receipts: contract
            .required_receipts
            .iter()
            .map(|receipt| (*receipt).to_string())
            .collect(),
        claim_boundary: contract.claim_boundary.to_string(),
    }
}

fn compute_sha256(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n =
            file.read(&mut buffer).with_context(|| format!("failed to read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn write_cache_metadata(
    cache_root: &Path,
    model: &SupportedModel,
    path: &Path,
    verify: &VerifyResult,
) -> Result<()> {
    let payload = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_model_cache_entry",
        "id": model.id,
        "display_name": model.display_name,
        "repo": model.repo,
        "revision": model.revision,
        "filename": model.filename,
        "source_url": model.url,
        "path": path,
        "sha256": model.sha256,
        "bytes": model.bytes,
        "architecture": model.architecture,
        "quantization": model.quantization,
        "tokenizer": {
            "model": model.tokenizer_model,
            "pre_tokenizer": model.tokenizer_pre,
            "chat_template_present": model.chat_template,
        },
        "model_contract": &verify.model_contract,
        "model_capability": &verify.model_capability,
        "artifact_provenance": &verify.artifact_provenance,
        "runtime_support": {
            "apple_m4_cpu_neon": model.apple_m4_cpu_neon_supported,
            "note": model.support_note,
        },
        "verification": verify,
        "verified_at": chrono::Utc::now().to_rfc3339(),
    });
    let metadata = metadata_path(cache_root, model);
    fs::create_dir_all(metadata.parent().unwrap_or(cache_root))?;
    let bytes = serde_json::to_vec_pretty(&payload)?;
    bitnet_download::atomic_write(&metadata, &bytes)
        .with_context(|| format!("failed to write {}", metadata.display()))?;
    Ok(())
}

fn replace_cached_file(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_file(dst)
            .with_context(|| format!("failed to remove old cache file {}", dst.display()))?;
    }
    fs::rename(src, dst)
        .with_context(|| format!("failed to rename {} to {}", src.display(), dst.display()))?;
    Ok(())
}

fn print_fetch_result(status: &str, verify: &VerifyResult, json: bool) -> Result<()> {
    let payload = serde_json::json!({
        "status": status,
        "id": verify.id,
        "path": verify.path,
        "sha256": verify.actual_sha256,
        "bytes": verify.actual_bytes,
        "verified": verify.passed,
        "apple_m4_cpu_neon_supported": verify.model.apple_m4_cpu_neon_supported,
        "support_note": verify.model.support_note,
        "model_contract": &verify.model_contract,
        "model_capability": &verify.model_capability,
        "artifact_provenance": &verify.artifact_provenance,
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!(
            "{status}: {} at {} ({}, verified={})",
            verify.id,
            verify.path.display(),
            verify.actual_bytes.map(|bytes| format_size(bytes, DECIMAL)).unwrap_or_default(),
            verify.passed,
        );
        if !verify.model.apple_m4_cpu_neon_supported {
            println!("note: {}", verify.model.support_note);
        }
    }
    Ok(())
}

fn model_command(action: &str, model: &SupportedModel, cache_root: Option<&Path>) -> String {
    let cache_arg =
        cache_root.map(|path| format!(" --cache-dir {}", shellish_path(path))).unwrap_or_default();
    format!("bitnet model {action} {}{cache_arg}", model.id)
}

fn shellish_path(path: &Path) -> String {
    let value = path.display().to_string();
    if value.chars().any(char::is_whitespace) { format!("{value:?}") } else { value }
}

fn warn_if_low_disk(cache_root: &Path, expected_bytes: u64) {
    let parent =
        cache_root.ancestors().find(|path| path.exists()).unwrap_or_else(|| Path::new("."));
    let Some(available) = available_bytes(parent) else {
        return;
    };
    let recommended = recommended_fetch_headroom_bytes(expected_bytes);
    if available < recommended {
        eprintln!(
            "warning: low disk headroom for model fetch: available={}, recommended>={}",
            format_size(available, DECIMAL),
            format_size(recommended, DECIMAL)
        );
        eprintln!(
            "warning: free space before fetching with `bitnet model prune --all` or choose a larger cache with `--cache-dir <PATH>` / BITNET_MODEL_CACHE_DIR."
        );
    }
}

fn available_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        let output = Command::new("df").arg("-k").arg(path).output().ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().nth(1)?;
        let available_kib = line.split_whitespace().nth(3)?.parse::<u64>().ok()?;
        Some(available_kib.saturating_mul(1024))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_model_coverage_matrix_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("ci")
            .join("model-artifacts")
            .join("model-coverage-matrix.toml")
    }

    fn model_status_row_for<'a>(
        dashboard: &'a ModelStatusDashboard,
        id: &str,
    ) -> Result<&'a ModelStatusRow> {
        dashboard
            .models
            .iter()
            .find(|row| row.id == id)
            .with_context(|| format!("missing model status row {id}"))
    }

    fn model_status_json_row_for<'a>(
        value: &'a serde_json::Value,
        id: &str,
    ) -> Result<&'a serde_json::Value> {
        value["models"]
            .as_array()
            .and_then(|models| models.iter().find(|model| model["id"] == id))
            .with_context(|| format!("missing model status JSON row {id}"))
    }

    #[test]
    fn supported_manifest_contains_m4_runtime_artifact() {
        let model = supported_model("qwen2.5-0.5b-instruct-q8_0").unwrap();
        assert!(model.apple_m4_cpu_neon_supported);
        assert_eq!(model.sha256.len(), 64);
        assert_eq!(model.bytes, 675_710_816);
        assert_eq!(model.tokenizer_pre, "qwen2");
    }

    #[test]
    fn supported_manifest_contains_official_bitnet_i2s_contract_artifact() {
        let model = supported_model("microsoft-bitnet-b1.58-2B-4T-i2s").unwrap();
        assert_eq!(model.repo, "microsoft/bitnet-b1.58-2B-4T-gguf");
        assert_eq!(model.filename, "ggml-model-i2_s.gguf");
        assert_eq!(
            model.sha256,
            "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
        );
        assert_eq!(model.bytes, 1_187_801_280);
        assert_eq!(model.model_contract, Some("microsoft_bitnet_b158_2b_4t_i2s"));
        assert_eq!(model.tokenizer_pre, "llama-bpe-external");
    }

    #[test]
    fn supported_model_lookup_accepts_official_bitnet_aliases() {
        let by_alias = supported_model("microsoft_bitnet_b158_2b_4t_gguf_i2s_current").unwrap();
        let by_case = supported_model("MICROSOFT-BITNET-B1.58-2B-4T-I2S").unwrap();

        assert_eq!(by_alias.id, "microsoft-bitnet-b1.58-2B-4T-i2s");
        assert_eq!(by_case.id, "microsoft-bitnet-b1.58-2B-4T-i2s");
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_slm_supported_model_rejects_bitnet_for_dense_commands() {
        let error = apple_m4_slm_supported_model("microsoft-bitnet-b1.58-2B-4T-i2s")
            .expect_err("BitNet is not a dense SLM artifact");
        let message = error.to_string();

        assert!(message.contains("supported-ask for Apple M4 CPU/NEON local answers"));
        assert!(message.contains("MODEL-ARTIFACT-007"));
        assert!(message.contains("M4-QA-001"));
        assert!(message.contains("one-shot `bitnet mac ask`"));
        assert!(message.contains("fixed-prompt `bitnet mac bitnet-warm`"));
        assert!(message.contains("bitnet mac models"));
        assert!(message.contains("qwen2.5-0.5b-instruct-q8_0"));
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_slm_supported_model_rejects_policy_only_rows() {
        let error =
            apple_m4_slm_supported_model("qwen3-0.6b-q8_0").expect_err("diagnostic-only model");
        let message = error.to_string();

        assert!(message.contains("diagnostic-only"));
        assert!(message.contains("not selectable"));
        assert!(message.contains("bitnet mac models"));
    }

    #[test]
    fn verify_model_includes_bitnet_contract_summary() -> Result<(), Box<dyn std::error::Error>> {
        let model = supported_model("microsoft-bitnet-b1.58-2B-4T-i2s")?;
        let result = verify_model(model, Path::new("/tmp/missing-bitnet-model.gguf"), None)?;
        let contract = result.model_contract.expect("model contract summary");

        assert!(!result.passed);
        assert_eq!(contract.id, "microsoft_bitnet_b158_2b_4t_i2s");
        assert_eq!(contract.kernel_family, "i2_s_qk256");
        assert_eq!(contract.status, "reference_ready");
        assert!(contract.architecture_support.iter().any(|support| {
            support.arch == "x86"
                && support.kernel == "i2_s"
                && support.status == "supported_reference"
        }));
        assert_eq!(contract.tokenizer_authority, "external_llama_bpe");
        assert_eq!(contract.prompt_authority, "bitnetcpp-answer");
        assert!(contract.accelerator_routes.iter().any(|route| route.route == "bitnet_qk256_cuda"));
        assert!(contract.permitted_claims.contains(&"answer_ready".to_string()));
        assert!(contract.required_receipts.contains(&"execution_plan".to_string()));
        Ok(())
    }

    #[test]
    fn contract_summary_exposes_unsupported_three_b_i2s_boundary() {
        let contract = find_bitnet_model_contract("1bitLLM/bitnet_b1_58-3B:i2_s:x86")
            .expect("3B x86 I2_S contract");
        let summary = contract_summary(contract);

        assert_eq!(summary.id, "onebitllm_bitnet_b158_3b_i2s_x86");
        assert_eq!(summary.kernel_family, "unsupported_i2_s");
        assert_eq!(summary.status, "upstream_unsupported");
        assert_eq!(summary.architecture_support.len(), 1);
        assert_eq!(summary.architecture_support[0].arch, "x86");
        assert_eq!(summary.architecture_support[0].kernel, "i2_s");
        assert_eq!(summary.architecture_support[0].status, "unsupported_upstream");
        assert!(summary.permitted_claims.contains(&"unsupported_path_receipt".to_string()));
        assert!(!summary.permitted_claims.contains(&"answer_ready".to_string()));
        assert!(summary.required_receipts.contains(&"unsupported_path_receipt".to_string()));
    }

    #[test]
    fn known_contract_without_supported_artifact_fails_closed_with_contract_summary() {
        let contract = find_bitnet_model_contract("microsoft_bitnet_b158_2b_4t_tl2")
            .expect("official TL2 contract");
        let result = contract_only_verify_result(
            "microsoft_bitnet_b158_2b_4t_tl2",
            Some(PathBuf::from("/tmp/tl2.gguf")),
            contract,
        );

        assert!(!result.passed);
        assert!(!result.supported_artifact);
        assert_eq!(result.path.as_deref(), Some(Path::new("/tmp/tl2.gguf")));
        assert_eq!(result.model_contract.id, "microsoft_bitnet_b158_2b_4t_tl2");
        assert_eq!(result.model_contract.status, "planned_proof_required");
        assert!(result.reason.contains("no supported artifact identity"));
    }

    #[test]
    fn model_contracts_surface_covers_every_registry_contract() {
        let summaries: Vec<_> = bitnet_model_contracts().iter().map(contract_summary).collect();

        assert_eq!(summaries.len(), bitnet_model_contracts().len());
        assert!(summaries.iter().any(|summary| summary.id == "microsoft_bitnet_b158_2b_4t_i2s"));
        assert!(summaries.iter().any(|summary| summary.id == "microsoft_bitnet_b158_2b_4t_tl1"));
        assert!(summaries.iter().any(|summary| summary.id == "microsoft_bitnet_b158_2b_4t_tl2"));
        assert!(summaries.iter().any(|summary| summary.id == "onebitllm_bitnet_b158_3b_i2s_x86"));
        assert!(summaries.iter().all(|summary| !summary.claim_boundary.trim().is_empty()));
        assert!(summaries.iter().all(|summary| {
            summary.id == "tdh111_bitnet_b158_2b_4t_iq2_bn_r4"
                || !summary.architecture_support.is_empty()
        }));
    }

    #[test]
    fn supported_manifest_keeps_q4_reference_boundary() {
        let model = supported_model("qwen2.5-0.5b-instruct-q4_k_m").unwrap();
        assert!(model.apple_m4_cpu_neon_supported);
        assert!(model.support_note.contains("storage-conscious"));
    }

    #[test]
    fn supported_manifest_contains_larger_qwen_candidate() {
        let model = supported_model("qwen2.5-1.5b-instruct-q4_k_m").unwrap();

        assert!(model.apple_m4_cpu_neon_supported);
        assert_eq!(model.repo, "Qwen/Qwen2.5-1.5B-Instruct-GGUF");
        assert_eq!(model.revision, "91cad51170dc346986eccefdc2dd33a9da36ead9");
        assert_eq!(model.bytes, 1_117_320_736);
        assert_eq!(
            model.sha256,
            "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
        );
        assert_eq!(model.tokenizer_pre, "qwen2");
        assert_eq!(model.quantization, "Q4_K_M");
        assert!(model.support_note.contains("non-default"));
    }

    #[test]
    fn m4_supported_models_emit_complete_artifact_provenance_manifests()
    -> Result<(), Box<dyn std::error::Error>> {
        let cache_root = Path::new("/tmp/bitnet-m4-provenance-cache");
        for id in [
            "qwen2.5-0.5b-instruct-q8_0",
            "qwen2.5-0.5b-instruct-q4_k_m",
            "qwen2.5-1.5b-instruct-q4_k_m",
            "microsoft-bitnet-b1.58-2B-4T-i2s",
        ] {
            let model = supported_model(id)?;
            let cache_path = model_path(cache_root, model);
            let manifest = model_provenance_manifest(
                model,
                cache_root,
                &cache_path,
                "missing",
                None,
                None,
                None,
            );

            assert_eq!(manifest.artifact_kind, "m4_supported_model_provenance");
            assert_eq!(manifest.source.repo, model.repo);
            assert_eq!(manifest.artifact.size_bytes, model.bytes);
            assert_eq!(manifest.artifact.sha256, model.sha256);
            assert!(!manifest.license.redistribution_boundary.trim().is_empty());
            assert_eq!(manifest.tokenizer.authority, model.tokenizer_pre);
            assert!(!manifest.tokenizer.sha256_status.trim().is_empty());
            assert_eq!(manifest.prompt_template.identity, model.prompt_template);
            assert_eq!(manifest.local_cache.cache_path, cache_path);
            assert_eq!(manifest.local_cache.symlink_status, "not_symlink");
            assert!(manifest.repair.command.contains("bitnet model fetch"));
            assert!(manifest.repair.verify_command.contains(model.id));
            assert!(manifest.claim_boundary.contains("Artifact provenance"));
        }
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_disk_recommendation_prefers_default_when_headroom_is_good()
    -> Result<(), Box<dyn std::error::Error>> {
        let default = supported_model("qwen2.5-0.5b-instruct-q8_0")?;
        let smallest = supported_model("qwen2.5-0.5b-instruct-q4_k_m")?;
        let largest = supported_model("qwen2.5-1.5b-instruct-q4_k_m")?;
        let available = recommended_fetch_headroom_bytes(largest.bytes);

        let (recommended, guidance) =
            apple_m4_disk_recommendation(Some(available), default, smallest, largest);

        assert_eq!(recommended.as_deref(), Some(default.id));
        assert!(guidance.contains(largest.id));
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_disk_recommendation_suggests_q4_when_default_headroom_is_tight()
    -> Result<(), Box<dyn std::error::Error>> {
        let default = supported_model("qwen2.5-0.5b-instruct-q8_0")?;
        let smallest = supported_model("qwen2.5-0.5b-instruct-q4_k_m")?;
        let largest = supported_model("qwen2.5-1.5b-instruct-q4_k_m")?;
        let available = recommended_fetch_headroom_bytes(smallest.bytes);

        let (recommended, guidance) =
            apple_m4_disk_recommendation(Some(available), default, smallest, largest);

        assert_eq!(recommended.as_deref(), Some(smallest.id));
        assert!(guidance.contains("storage-conscious"));
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_disk_recommendation_blocks_fetch_when_headroom_is_too_low()
    -> Result<(), Box<dyn std::error::Error>> {
        let default = supported_model("qwen2.5-0.5b-instruct-q8_0")?;
        let smallest = supported_model("qwen2.5-0.5b-instruct-q4_k_m")?;
        let largest = supported_model("qwen2.5-1.5b-instruct-q4_k_m")?;
        let available = recommended_fetch_headroom_bytes(smallest.bytes).saturating_sub(1);

        let (recommended, guidance) =
            apple_m4_disk_recommendation(Some(available), default, smallest, largest);

        assert!(recommended.is_none());
        assert!(guidance.contains("below the smallest supported M4 model"));
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_model_next_step_prints_recommended_fetch_and_verify()
    -> Result<(), Box<dyn std::error::Error>> {
        let cache_root = PathBuf::from("/tmp/bitnet-cache");
        let default = supported_model("qwen2.5-0.5b-instruct-q8_0")?;
        let disk = AppleM4DiskSummary {
            probe_path: cache_root.clone(),
            available_bytes: Some(recommended_fetch_headroom_bytes(default.bytes)),
            available: Some("2.43 GB".to_string()),
            default_model_headroom_bytes: recommended_fetch_headroom_bytes(default.bytes),
            default_model_headroom: "2.43 GB".to_string(),
            smallest_supported_headroom_bytes: recommended_fetch_headroom_bytes(default.bytes),
            smallest_supported_headroom: "2.43 GB".to_string(),
            largest_supported_headroom_bytes: recommended_fetch_headroom_bytes(default.bytes),
            largest_supported_headroom: "2.43 GB".to_string(),
            low_disk: Some(false),
            recommended_first_model_id: Some(default.id.to_string()),
            recommendation: "start with default".to_string(),
            guidance: "test guidance".to_string(),
        };
        let status = CacheStatus {
            model: *default,
            cache_path: model_path(&cache_root, default),
            metadata_path: metadata_path(&cache_root, default),
            symlink_target: None,
            symlink_status: "not_symlink".to_string(),
            present: false,
            cached: false,
            size_matches: false,
            metadata_present: false,
            verified: None,
        };
        let row = apple_m4_registered_model_row(&cache_root, &status, disk.available_bytes);
        let catalog = AppleM4ModelCatalog {
            cache_root,
            default_model_id: M4_SLM_RUNTIME_MODEL_ID,
            disk,
            claim_boundary: "test",
            rows: vec![row],
        };

        let next = apple_m4_model_next_step(&catalog);

        assert!(
            next.contains("Next fetch: bitnet model fetch qwen2.5-0.5b-instruct-q8_0 --cache-dir")
        );
        assert!(
            next.contains(
                "Next verify: bitnet model verify qwen2.5-0.5b-instruct-q8_0 --cache-dir"
            )
        );
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_model_next_step_blocks_fetch_when_disk_is_too_low() {
        let cache_root = PathBuf::from("/tmp/bitnet-cache");
        let disk = AppleM4DiskSummary {
            probe_path: cache_root.clone(),
            available_bytes: Some(1),
            available: Some("1 B".to_string()),
            default_model_headroom_bytes: 2,
            default_model_headroom: "2 B".to_string(),
            smallest_supported_headroom_bytes: 2,
            smallest_supported_headroom: "2 B".to_string(),
            largest_supported_headroom_bytes: 2,
            largest_supported_headroom: "2 B".to_string(),
            low_disk: Some(true),
            recommended_first_model_id: None,
            recommendation: "too low".to_string(),
            guidance: "test guidance".to_string(),
        };
        let catalog = AppleM4ModelCatalog {
            cache_root,
            default_model_id: M4_SLM_RUNTIME_MODEL_ID,
            disk,
            claim_boundary: "test",
            rows: Vec::new(),
        };

        let next = apple_m4_model_next_step(&catalog);

        assert!(next.contains("no supported model has enough disk headroom"));
        assert!(!next.contains("bitnet model fetch"));
    }

    #[test]
    fn verify_model_includes_qwen_dense_capability_summary()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = supported_model("qwen2.5-0.5b-instruct-q8_0")?;
        let result = verify_model(model, Path::new("/tmp/missing-qwen-q8.gguf"), None)?;
        let capability = result.model_capability.expect("model capability summary");

        assert!(!result.passed);
        assert!(result.model_contract.is_none());
        assert_eq!(capability.id, "qwen_dense_slm_q8_0");
        assert_eq!(capability.model_family, "qwen");
        assert_eq!(capability.model_class, "dense_slm_gguf");
        assert_eq!(capability.quantization, "Q8_0");
        assert_eq!(capability.tokenizer_authority, "qwen2");
        assert_eq!(capability.prompt_authority, "qwen2.5");
        assert!(capability.capabilities.contains(&"chat_completion".to_string()));
        assert!(capability.capabilities.contains(&"text_generation".to_string()));
        assert!(capability.accelerator_routes.iter().any(|route| {
            route.backend == "apple-m4-cpu-neon"
                && route.route == "dense_qwen_cpu_neon_slm"
                && route.status == "answer_lane"
        }));
        assert!(capability.accelerator_routes.iter().any(|route| {
            route.backend == "nvidia-rtx-5070-ti-cuda"
                && route.route == "dense_regular_llm_cuda"
                && route.status == "bounded_ask_chat_receipt_gate"
        }));
        assert!(capability.permitted_claims.contains(&"apple_m4_cpu_neon_slm_answer".to_string()));
        assert!(
            capability.permitted_claims.contains(&"nvidia_rtx_5070_ti_dense_cuda_ask".to_string())
        );
        assert!(
            capability.permitted_claims.contains(&"nvidia_rtx_5070_ti_dense_cuda_chat".to_string())
        );
        assert!(
            capability
                .required_receipts
                .contains(&"dense_gguf_qwen_ask_strict_cuda_proof".to_string())
        );
        assert!(
            capability
                .required_receipts
                .contains(&"dense_gguf_qwen_chat_strict_cuda_proof".to_string())
        );
        assert!(capability.claim_boundary.contains("bounded RTX 5070 Ti dense CUDA ask/chat"));
        assert!(capability.claim_boundary.contains("does not prove broad dense GGUF inference"));
        Ok(())
    }

    #[test]
    fn qwen_q4_capability_is_storage_conscious_answer_lane()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = supported_model("qwen2.5-0.5b-instruct-q4_k_m")?;
        let result = verify_model(model, Path::new("/tmp/missing-qwen-q4.gguf"), None)?;
        let capability = result.model_capability.expect("model capability summary");

        assert!(!result.passed);
        assert_eq!(capability.id, "qwen_dense_slm_q4_k_m");
        assert_eq!(capability.quantization, "Q4_K_M");
        assert_eq!(capability.cpu_oracle, "apple_m4_cpu_neon_slm_answer_lane");
        assert!(capability.accelerator_routes.iter().any(|route| {
            route.backend == "apple-m4-cpu-neon"
                && route.route == "dense_qwen_cpu_neon_slm"
                && route.status == "answer_lane"
        }));
        assert!(capability.permitted_claims.contains(&"apple_m4_cpu_neon_slm_answer".to_string()));
        assert!(capability.required_receipts.contains(&"slm_answer_receipt".to_string()));
        assert!(capability.claim_boundary.contains("does not prove dense CUDA"));
        Ok(())
    }

    #[test]
    fn larger_qwen_capability_is_non_default_answer_lane() -> Result<(), Box<dyn std::error::Error>>
    {
        let model = supported_model("qwen2.5-1.5b-instruct-q4_k_m")?;
        let result = verify_model(model, Path::new("/tmp/missing-qwen-15-q4.gguf"), None)?;
        let capability = result.model_capability.expect("model capability summary");

        assert!(!result.passed);
        assert_eq!(capability.id, "qwen_dense_slm_1_5b_q4_k_m");
        assert_eq!(capability.model_family, "qwen");
        assert_eq!(capability.quantization, "Q4_K_M");
        assert_eq!(capability.cpu_oracle, "apple_m4_cpu_neon_slm_answer_lane");
        assert!(capability.accelerator_routes.iter().any(|route| {
            route.backend == "apple-m4-cpu-neon"
                && route.route == "dense_qwen_cpu_neon_slm"
                && route.status == "answer_lane"
        }));
        assert!(capability.permitted_claims.contains(&"apple_m4_cpu_neon_slm_answer".to_string()));
        assert!(
            capability.required_receipts.contains(&"fallback_free_backend_receipt".to_string())
        );
        assert!(capability.claim_boundary.contains("does not prove dense CUDA"));
        Ok(())
    }

    #[test]
    fn bitnet_contract_artifact_does_not_emit_dense_capability_summary()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = supported_model("microsoft-bitnet-b1.58-2B-4T-i2s")?;
        let result = verify_model(model, Path::new("/tmp/missing-bitnet-model.gguf"), None)?;

        assert!(result.model_contract.is_some());
        assert!(result.model_capability.is_none());
        Ok(())
    }

    #[test]
    fn model_coverage_matrix_parse_exposes_bitnet_and_dense_boundaries() {
        let matrix = read_model_coverage_matrix(&workspace_model_coverage_matrix_path()).unwrap();

        assert_eq!(matrix.schema, 1);
        assert_eq!(matrix.artifact_kind, "model_coverage_matrix");
        assert!(matrix.tier.iter().any(|tier| tier.id == "product_cli_ready"));

        let bitnet = find_model_coverage_entry(&matrix, "bitnet_official_2b_i2s_qk256")
            .expect("official BitNet coverage row");
        assert_eq!(bitnet.current_tier, "product_cli_ready");
        assert_eq!(bitnet.accelerator_routes, ["bitnet_qk256_cuda"]);
        assert!(bitnet.claims.bitnet_packed_i2s_qk256_proof);
        assert!(!bitnet.claims.dense_regular_llm_cuda_proof);
        assert!(!bitnet.claims.speedup_claim);

        let dense = find_model_coverage_entry(&matrix, "dense_qwen25_05b_q8_cuda")
            .expect("dense Qwen coverage row");
        assert_eq!(dense.current_tier, "product_cli_ready");
        assert_eq!(dense.accelerator_routes, ["dense_regular_llm_cuda"]);
        assert!(dense.claims.dense_regular_llm_cuda_proof);
        assert!(!dense.claims.bitnet_packed_i2s_qk256_proof);
        assert!(!dense.claims.speedup_claim);
    }

    #[test]
    fn model_coverage_matrix_preserves_unsupported_and_docs_only_boundaries() {
        let matrix = read_model_coverage_matrix(&workspace_model_coverage_matrix_path()).unwrap();

        let unsupported = find_model_coverage_entry(&matrix, "bitnet_3b_x86_i2s_unsupported")
            .expect("3B unsupported coverage row");
        assert_eq!(unsupported.status, "unsupported_upstream");
        assert_eq!(unsupported.accelerator_routes, Vec::<String>::new());
        assert!(unsupported.forbidden_claims.iter().any(|claim| claim == "answer_ready"));
        assert!(!unsupported.claims.cpu_answer_ready);
        assert!(!unsupported.claims.accelerator_answer_ready);

        let docs_only = find_model_coverage_entry(&matrix, "modern_llm_dense_frontier_placeholder")
            .expect("docs-only modern LLM coverage row");
        assert_eq!(docs_only.model_class, "modern_llm_docs_only");
        assert_eq!(docs_only.accelerator_routes, Vec::<String>::new());
        assert!(
            docs_only
                .required_receipts
                .iter()
                .any(|receipt| { receipt == "unsupported_on_current_hardware_receipt" })
        );
        assert!(docs_only.claims.registered);
        assert!(!docs_only.claims.structurally_valid);
        assert!(!docs_only.claims.dense_regular_llm_cuda_proof);
        assert!(!docs_only.claims.server_ready);
    }

    #[test]
    fn model_coverage_lookup_is_case_insensitive_and_fails_closed() {
        let matrix = read_model_coverage_matrix(&workspace_model_coverage_matrix_path()).unwrap();

        let entry = find_model_coverage_entry(&matrix, "DENSE_QWEN25_05B_Q8_CUDA")
            .expect("case-insensitive coverage lookup");
        assert_eq!(entry.id, "dense_qwen25_05b_q8_cuda");
        assert!(find_model_coverage_entry(&matrix, "missing_model_coverage_row").is_none());
    }

    #[test]
    fn every_supported_model_has_model_coverage_matrix_row() {
        let matrix = read_model_coverage_matrix(&workspace_model_coverage_matrix_path()).unwrap();

        for model in SUPPORTED_MODELS {
            let has_coverage = matrix.entry.iter().any(|entry| {
                entry.capability_id.as_deref() == Some(model.id)
                    || model
                        .model_contract
                        .is_some_and(|contract| entry.contract_id.as_deref() == Some(contract))
                    || entry.verifier_surface.split_whitespace().any(|part| part == model.id)
            });

            assert!(
                has_coverage,
                "supported model `{}` must have a model coverage matrix row through capability_id, contract_id, or verifier_surface",
                model.id
            );
        }
    }

    #[test]
    fn model_status_dashboard_shows_cuda_supported_rows_without_speed_or_server_claims()
    -> Result<()> {
        let matrix_path = workspace_model_coverage_matrix_path();
        let matrix = read_model_coverage_matrix(&matrix_path)?;
        let dashboard = model_status_dashboard("nvidia-rtx-5070-ti-cuda", &matrix_path, &matrix);

        assert_eq!(dashboard.schema_version, 1);
        assert_eq!(dashboard.device, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(dashboard.requested_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(dashboard.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert!(dashboard.note.contains("does not probe hardware"));

        let bitnet = model_status_row_for(&dashboard, "bitnet_official_2b_i2s_qk256")?;
        assert_eq!(bitnet.display_name, "microsoft-bitnet-b1.58-2B-4T-i2s");
        assert_eq!(bitnet.model_coverage_row, "bitnet_official_2b_i2s_qk256");
        assert_eq!(bitnet.current_tier, "product_cli_ready");
        assert_eq!(bitnet.requested_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(bitnet.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(bitnet.selected_route.as_deref(), Some("bitnet_qk256_cuda"));
        assert_eq!(bitnet.fallback_used, Some(false));
        assert_eq!(bitnet.category, "supported");
        assert_eq!(bitnet.model_class, "bitnet");
        assert_eq!(bitnet.route.as_deref(), Some("bitnet_qk256_cuda"));
        assert!(bitnet.cpu_answer_ready);
        assert!(bitnet.accelerator_answer_ready);
        assert!(bitnet.product_cli_ready);
        assert!(bitnet.bitnet_packed_i2s_qk256_proof);
        assert!(!bitnet.dense_regular_llm_cuda_proof);
        assert!(!bitnet.benchmark_qualified);
        assert!(!bitnet.speedup_claim);
        assert!(!bitnet.server_ready);
        assert_eq!(bitnet.server, "smoke only, not broad-ready");
        assert_eq!(bitnet.server_ready_scope, None);
        assert_eq!(bitnet.server_scope, None);
        assert_eq!(bitnet.server_endpoint.as_deref(), Some("/v1/chat/completions"));
        assert_eq!(bitnet.server_streaming, Some(false));
        assert!(bitnet.server_smoke);
        assert_eq!(
            bitnet.server_reason.as_deref(),
            Some("broad production readiness not qualified")
        );
        assert!(!bitnet.full_residency_claim);
        assert_eq!(bitnet.ask, "ready");
        assert_eq!(bitnet.one_token, "ready");
        assert_eq!(bitnet.short_decode, "ready");
        assert_eq!(bitnet.warm_session, "ready");
        assert_eq!(bitnet.benchmark, "reviewed, speedup not accepted");
        assert!(bitnet.claim_boundary.contains("does not prove dense regular-LLM CUDA"));

        let dense = model_status_row_for(&dashboard, "dense_qwen25_05b_q8_cuda")?;
        assert_eq!(dense.display_name, "qwen2.5-0.5b-instruct-q8_0");
        assert_eq!(dense.model_coverage_row, "dense_qwen25_05b_q8_cuda");
        assert_eq!(dense.current_tier, "product_cli_ready");
        assert_eq!(dense.requested_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(dense.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(dense.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(dense.fallback_used, Some(false));
        assert_eq!(dense.category, "supported");
        assert_eq!(dense.model_class, "dense_slm");
        assert_eq!(dense.route.as_deref(), Some("dense_regular_llm_cuda"));
        assert!(dense.cpu_answer_ready);
        assert!(dense.accelerator_answer_ready);
        assert!(dense.product_cli_ready);
        assert!(dense.dense_regular_llm_cuda_proof);
        assert!(!dense.bitnet_packed_i2s_qk256_proof);
        assert!(!dense.benchmark_qualified);
        assert!(!dense.speedup_claim);
        assert!(dense.server_ready);
        assert_eq!(dense.server, "exact-profile ready (/v1/chat/completions, streaming=false)");
        assert_eq!(dense.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(dense.server_scope.as_deref(), Some("exact_profile"));
        assert_eq!(dense.server_endpoint.as_deref(), Some("/v1/chat/completions"));
        assert_eq!(dense.server_streaming, Some(false));
        assert!(dense.server_smoke);
        assert_eq!(dense.server_reason, None);
        assert!(!dense.full_residency_claim);
        assert_eq!(dense.one_token, "ready");
        assert_eq!(dense.short_decode, "ready");
        assert_eq!(dense.warm_session, "ready");
        assert_eq!(dense.benchmark, "reviewed, speedup not accepted");
        assert!(dense.claim_boundary.contains("do not satisfy BitNet packed I2_S/QK256"));
        Ok(())
    }

    #[test]
    fn model_status_dashboard_lists_qwen3_as_product_cli_ready() -> Result<()> {
        let matrix_path = workspace_model_coverage_matrix_path();
        let matrix = read_model_coverage_matrix(&matrix_path)?;
        let dashboard = model_status_dashboard("nvidia-rtx-5070-ti-cuda", &matrix_path, &matrix);

        let qwen3 = model_status_row_for(&dashboard, "dense_qwen3_06b_q8_candidate")?;
        assert_eq!(qwen3.display_name, "qwen3-0.6b-instruct-q8_0");
        assert_eq!(qwen3.category, "supported");
        assert_eq!(qwen3.route.as_deref(), Some("dense_regular_llm_cuda"));
        assert!(qwen3.cpu_answer_ready);
        assert!(qwen3.accelerator_answer_ready);
        assert!(qwen3.product_cli_ready);
        assert!(qwen3.dense_regular_llm_cuda_proof);
        assert!(!qwen3.bitnet_packed_i2s_qk256_proof);
        assert!(!qwen3.speedup_claim);
        assert!(qwen3.server_ready);
        assert_eq!(qwen3.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(qwen3.server_endpoint.as_deref(), Some("/v1/chat/completions"));
        assert_eq!(qwen3.server_streaming, Some(false));
        assert!(qwen3.server_smoke);
        assert_eq!(qwen3.server_reason, None);
        assert!(!qwen3.full_residency_claim);
        assert_eq!(qwen3.ask, "ready");
        assert_eq!(qwen3.one_token, "ready");
        assert_eq!(qwen3.short_decode, "ready");
        assert_eq!(qwen3.warm_session, "ready");
        assert_eq!(qwen3.benchmark, "reviewed, speedup not accepted");
        assert_eq!(qwen3.tier, "product_cli_ready");
        assert!(qwen3.next_proof.contains("Qwen3 exact-profile server readiness"));
        assert!(qwen3.claim_boundary.contains("dense_regular_llm_cuda RTX 5070 Ti route"));
        assert!(qwen3.claim_boundary.contains("does not inherit Qwen2.5 CUDA receipts"));
        Ok(())
    }

    #[test]
    fn model_status_dashboard_lists_smollm2_structural_blocker() -> Result<()> {
        let matrix_path = workspace_model_coverage_matrix_path();
        let matrix = read_model_coverage_matrix(&matrix_path)?;
        let dashboard = model_status_dashboard("nvidia-rtx-5070-ti-cuda", &matrix_path, &matrix);

        let smollm2 = model_status_row_for(&dashboard, "dense_smollm2_360m_candidate")?;
        assert_eq!(smollm2.display_name, "smollm2-360m-instruct");
        assert_eq!(smollm2.current_tier, "structurally_valid");
        assert_eq!(smollm2.category, "candidate");
        assert_eq!(smollm2.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(smollm2.selected_route, None);
        assert_eq!(smollm2.fallback_used, None);
        assert!(!smollm2.product_cli_ready);
        assert!(!smollm2.cpu_answer_ready);
        assert!(!smollm2.accelerator_answer_ready);
        assert!(!smollm2.dense_regular_llm_cuda_proof);
        assert!(!smollm2.bitnet_packed_i2s_qk256_proof);
        assert!(!smollm2.speedup_claim);
        assert!(!smollm2.server_ready);
        assert_eq!(smollm2.server_ready_scope, None);
        assert!(!smollm2.full_residency_claim);
        assert!(smollm2.next_proof.contains("same-prompt SmolLM2"));
        assert!(smollm2.claim_boundary.contains("no CPU answer readiness"));
        Ok(())
    }

    #[test]
    fn model_status_dashboard_resolves_generic_cuda_to_strict_backend() -> Result<()> {
        let matrix_path = workspace_model_coverage_matrix_path();
        let matrix = read_model_coverage_matrix(&matrix_path)?;
        let dashboard = model_status_dashboard("cuda", &matrix_path, &matrix);

        assert_eq!(dashboard.device, "cuda");
        assert_eq!(dashboard.requested_backend, "cuda");
        assert_eq!(dashboard.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));

        let bitnet = model_status_row_for(&dashboard, "bitnet_official_2b_i2s_qk256")?;
        assert_eq!(bitnet.requested_backend, "cuda");
        assert_eq!(bitnet.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(bitnet.selected_route.as_deref(), Some("bitnet_qk256_cuda"));
        assert_eq!(bitnet.fallback_used, Some(false));
        assert!(!bitnet.speedup_claim);
        assert!(!bitnet.full_residency_claim);
        assert!(bitnet.bitnet_packed_i2s_qk256_proof);
        assert!(!bitnet.dense_regular_llm_cuda_proof);
        Ok(())
    }

    #[test]
    fn model_status_dashboard_stays_empty_for_unknown_device() -> Result<()> {
        let matrix_path = workspace_model_coverage_matrix_path();
        let matrix = read_model_coverage_matrix(&matrix_path)?;
        let dashboard = model_status_dashboard("cuda-experimental", &matrix_path, &matrix);

        assert_eq!(dashboard.device, "cuda-experimental");
        assert_eq!(dashboard.requested_backend, "cuda-experimental");
        assert_eq!(dashboard.selected_backend, None);
        assert!(dashboard.models.is_empty());
        Ok(())
    }

    #[test]
    fn model_status_dashboard_json_shape_is_stable() -> Result<()> {
        let matrix_path = workspace_model_coverage_matrix_path();
        let matrix = read_model_coverage_matrix(&matrix_path)?;
        let dashboard = model_status_dashboard("nvidia-rtx-5070-ti-cuda", &matrix_path, &matrix);
        let value = serde_json::to_value(&dashboard)?;

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["device"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["requested_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["selected_backend"], "nvidia-rtx-5070-ti-cuda");

        let bitnet = model_status_json_row_for(&value, "bitnet_official_2b_i2s_qk256")?;
        assert_eq!(bitnet["model_coverage_row"], "bitnet_official_2b_i2s_qk256");
        assert_eq!(bitnet["current_tier"], "product_cli_ready");
        assert_eq!(bitnet["requested_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(bitnet["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(bitnet["selected_route"], "bitnet_qk256_cuda");
        assert_eq!(bitnet["fallback_used"], false);
        assert_eq!(bitnet["product_cli_ready"], true);
        assert_eq!(bitnet["route"], "bitnet_qk256_cuda");
        assert_eq!(bitnet["speedup_claim"], false);
        assert_eq!(bitnet["server_ready"], false);
        assert!(bitnet["server_ready_scope"].is_null());
        assert!(bitnet["server_scope"].is_null());
        assert_eq!(bitnet["full_residency_claim"], false);
        assert_eq!(bitnet["server_endpoint"], "/v1/chat/completions");
        assert_eq!(bitnet["server_streaming"], false);
        assert_eq!(bitnet["server_smoke"], true);
        assert_eq!(bitnet["server_reason"], "broad production readiness not qualified");
        assert_eq!(bitnet["bitnet_packed_i2s_qk256_proof"], true);
        assert_eq!(bitnet["dense_regular_llm_cuda_proof"], false);

        let qwen25 = model_status_json_row_for(&value, "dense_qwen25_05b_q8_cuda")?;
        assert_eq!(qwen25["model_coverage_row"], "dense_qwen25_05b_q8_cuda");
        assert_eq!(qwen25["current_tier"], "product_cli_ready");
        assert_eq!(qwen25["requested_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(qwen25["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(qwen25["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(qwen25["fallback_used"], false);
        assert_eq!(qwen25["product_cli_ready"], true);
        assert_eq!(qwen25["route"], "dense_regular_llm_cuda");
        assert_eq!(qwen25["speedup_claim"], false);
        assert_eq!(qwen25["server_ready"], true);
        assert_eq!(qwen25["server_ready_scope"], "exact_profile");
        assert_eq!(qwen25["server_scope"], "exact_profile");
        assert_eq!(qwen25["full_residency_claim"], false);
        assert_eq!(qwen25["server_endpoint"], "/v1/chat/completions");
        assert_eq!(qwen25["server_streaming"], false);
        assert_eq!(qwen25["server_smoke"], true);
        assert!(qwen25["server_reason"].is_null());
        assert_eq!(qwen25["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(qwen25["dense_regular_llm_cuda_proof"], true);

        let qwen3 = model_status_json_row_for(&value, "dense_qwen3_06b_q8_candidate")?;
        assert_eq!(qwen3["model_coverage_row"], "dense_qwen3_06b_q8_candidate");
        assert_eq!(
            qwen3["next_proof"],
            "Qwen3 exact-profile server readiness is promoted only for the current-source non-streaming /v1/chat/completions RTX 5070 Ti shared-engine receipt at ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json. Separate exact-profile performance comparator evidence is still required before any speed, benchmark-qualified, or full-residency promotion."
        );
        assert_eq!(qwen3["category"], "supported");
        assert_eq!(qwen3["current_tier"], "product_cli_ready");
        assert_eq!(qwen3["requested_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(qwen3["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(qwen3["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(qwen3["fallback_used"], false);
        assert_eq!(qwen3["tier"], "product_cli_ready");
        assert_eq!(qwen3["product_cli_ready"], true);
        assert_eq!(qwen3["route"], "dense_regular_llm_cuda");
        assert_eq!(qwen3["accelerator_answer_ready"], true);
        assert_eq!(qwen3["speedup_claim"], false);
        assert_eq!(qwen3["server_ready"], true);
        assert_eq!(qwen3["server_ready_scope"], "exact_profile");
        assert_eq!(qwen3["server_scope"], "exact_profile");
        assert_eq!(qwen3["full_residency_claim"], false);
        assert_eq!(qwen3["server_endpoint"], "/v1/chat/completions");
        assert_eq!(qwen3["server_streaming"], false);
        assert_eq!(qwen3["server_smoke"], true);
        assert!(qwen3["server_reason"].is_null());
        assert_eq!(qwen3["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(qwen3["dense_regular_llm_cuda_proof"], true);

        let smollm2 = model_status_json_row_for(&value, "dense_smollm2_360m_candidate")?;
        assert_eq!(smollm2["model_coverage_row"], "dense_smollm2_360m_candidate");
        assert_eq!(smollm2["category"], "candidate");
        assert_eq!(smollm2["current_tier"], "structurally_valid");
        assert_eq!(smollm2["requested_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(smollm2["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert!(smollm2["selected_route"].is_null());
        assert!(smollm2["fallback_used"].is_null());
        assert_eq!(smollm2["product_cli_ready"], false);
        assert_eq!(smollm2["cpu_answer_ready"], false);
        assert_eq!(smollm2["accelerator_answer_ready"], false);
        assert_eq!(smollm2["speedup_claim"], false);
        assert_eq!(smollm2["server_ready"], false);
        assert!(smollm2["server_ready_scope"].is_null());
        assert_eq!(smollm2["full_residency_claim"], false);
        assert_eq!(smollm2["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(smollm2["dense_regular_llm_cuda_proof"], false);
        assert!(
            smollm2["next_proof"]
                .as_str()
                .is_some_and(|next| { next.contains("same-prompt SmolLM2") })
        );
        Ok(())
    }

    #[test]
    fn cache_paths_are_under_model_id() {
        let root = PathBuf::from("/tmp/bitnet-cache");
        let model = supported_model("qwen2.5-0.5b-instruct-q8_0").unwrap();
        let path = model_path(&root, model);
        assert!(path.ends_with("qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf"));
    }

    #[test]
    fn cache_state_distinguishes_bad_hash_from_ready() -> Result<(), Box<dyn std::error::Error>> {
        let root = PathBuf::from("/tmp/bitnet-cache");
        let model = *supported_model("qwen2.5-0.5b-instruct-q8_0")?;
        let status = CacheStatus {
            model,
            cache_path: model_path(&root, &model),
            metadata_path: metadata_path(&root, &model),
            symlink_target: None,
            symlink_status: "not_symlink".to_string(),
            present: true,
            cached: true,
            size_matches: true,
            metadata_present: true,
            verified: Some(false),
        };

        assert!(!cache_ready(&status));
        assert_eq!(cache_state_label(&status), "invalid-sha");
        let guidance = cache_repair_guidance(&root, &status);
        assert!(guidance.contains("bitnet model prune qwen2.5-0.5b-instruct-q8_0"));
        assert!(guidance.contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
        Ok(())
    }

    #[test]
    fn cache_repair_guidance_handles_metadata_missing() {
        let root = PathBuf::from("/tmp/bitnet-cache");
        let model = *supported_model("qwen2.5-0.5b-instruct-q8_0").unwrap();
        let status = CacheStatus {
            model,
            cache_path: model_path(&root, &model),
            metadata_path: metadata_path(&root, &model),
            symlink_target: None,
            symlink_status: "not_symlink".to_string(),
            present: true,
            cached: false,
            size_matches: true,
            metadata_present: false,
            verified: Some(true),
        };

        assert!(!cache_ready(&status));
        assert_eq!(cache_state_label(&status), "unverified");
        assert!(
            cache_repair_guidance(&root, &status)
                .contains("bitnet model verify qwen2.5-0.5b-instruct-q8_0")
        );
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn apple_m4_cache_repair_guidance_adds_first_run_model_selection() {
        let root = PathBuf::from("/tmp/bitnet-cache");
        let model = *supported_model("qwen2.5-0.5b-instruct-q8_0").unwrap();
        let status = CacheStatus {
            model,
            cache_path: model_path(&root, &model),
            metadata_path: metadata_path(&root, &model),
            symlink_target: None,
            symlink_status: "not_symlink".to_string(),
            present: false,
            cached: false,
            size_matches: false,
            metadata_present: false,
            verified: None,
        };

        let guidance = apple_m4_cache_repair_guidance(&root, &status);

        assert!(guidance.contains("First run"));
        assert!(guidance.contains("First-run model selection"));
        assert!(guidance.contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0"));
        assert!(guidance.contains("bitnet mac models --cache-dir /tmp/bitnet-cache"));
        assert!(guidance.contains("Disk guidance:"));
    }
}
