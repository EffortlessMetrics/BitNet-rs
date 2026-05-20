use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

#[derive(Debug, Deserialize)]
struct CoverageMatrix {
    schema: u32,
    artifact_kind: String,
    work_item: String,
    claim_boundary: String,
    tier: Vec<Tier>,
    entry: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct Tier {
    id: String,
    rank: u32,
    requires: Vec<String>,
    meaning: String,
}

#[derive(Debug, Deserialize)]
struct Entry {
    id: String,
    model_class: String,
    family: String,
    artifact_kind: String,
    #[allow(dead_code)]
    contract_id: Option<String>,
    #[allow(dead_code)]
    capability_id: Option<String>,
    status: String,
    current_tier: String,
    verifier_surface: String,
    tokenizer_authority: String,
    prompt_authority: String,
    cpu_reference: String,
    accelerator_routes: Vec<String>,
    required_receipts: Vec<String>,
    forbidden_claims: Vec<String>,
    next_proof: String,
    claim_boundary: String,
    claims: Claims,
}

#[derive(Debug, Deserialize)]
struct Claims {
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

pub fn run(matrix_path: PathBuf) -> Result<()> {
    let matrix = load_matrix(&matrix_path)?;
    validate_matrix(&matrix)?;
    println!(
        "model coverage matrix passed: {} entries, {} tiers ({})",
        matrix.entry.len(),
        matrix.tier.len(),
        matrix_path.display()
    );
    Ok(())
}

fn load_matrix(matrix_path: &PathBuf) -> Result<CoverageMatrix> {
    let raw = fs::read_to_string(matrix_path)
        .with_context(|| format!("reading {}", matrix_path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", matrix_path.display()))
}

fn validate_matrix(matrix: &CoverageMatrix) -> Result<()> {
    require_eq(matrix.schema, 1, "schema")?;
    require_eq(matrix.artifact_kind.as_str(), "model_coverage_matrix", "artifact_kind")?;
    require_eq(matrix.work_item.as_str(), "MODEL-COVERAGE-005", "work_item")?;
    require_nonempty(&matrix.claim_boundary, "claim_boundary")?;
    validate_tiers(&matrix.tier)?;
    validate_entries(matrix)?;
    Ok(())
}

fn validate_tiers(tiers: &[Tier]) -> Result<()> {
    if tiers.is_empty() {
        bail!("matrix has no coverage tiers");
    }
    let mut seen_ids = HashSet::new();
    let mut seen_ranks = HashSet::new();
    let ranks_by_id: HashMap<_, _> =
        tiers.iter().map(|tier| (tier.id.as_str(), tier.rank)).collect();

    for tier in tiers {
        require_nonempty(&tier.id, "tier.id")?;
        require_nonempty(&tier.meaning, "tier.meaning")?;
        if !seen_ids.insert(tier.id.as_str()) {
            bail!("duplicate tier id `{}`", tier.id);
        }
        if !seen_ranks.insert(tier.rank) {
            bail!("duplicate tier rank `{}`", tier.rank);
        }
        for required in &tier.requires {
            let Some(required_rank) = ranks_by_id.get(required.as_str()) else {
                bail!("tier `{}` requires unknown tier `{}`", tier.id, required);
            };
            if *required_rank >= tier.rank {
                bail!(
                    "tier `{}` requires `{}` but required rank {} is not lower than {}",
                    tier.id,
                    required,
                    required_rank,
                    tier.rank
                );
            }
        }
    }

    Ok(())
}

fn validate_entries(matrix: &CoverageMatrix) -> Result<()> {
    if matrix.entry.is_empty() {
        bail!("matrix has no entries");
    }

    let tiers: HashSet<_> = matrix.tier.iter().map(|tier| tier.id.as_str()).collect();
    let mut seen_ids = HashSet::new();
    for entry in &matrix.entry {
        validate_entry(entry, &tiers)?;
        if !seen_ids.insert(entry.id.as_str()) {
            bail!("duplicate entry id `{}`", entry.id);
        }
    }

    for required in [
        "bitnet_official_2b_i2s_qk256",
        "bitnet_official_2b_tl1_arm_candidate",
        "bitnet_official_2b_tl2_x86_candidate",
        "bitnet_official_2b_bf16_gpu_int2_candidate",
        "dense_qwen25_05b_q8_cuda",
        "dense_qwen3_06b_q8_candidate",
        "dense_smollm2_360m_candidate",
        "dense_smollm2_17b_candidate",
        "dense_llama32_1b_candidate",
        "dense_llama32_3b_candidate",
        "dense_gemma_small_candidate",
        "dense_phi_small_candidate",
        "bitnet_3b_x86_i2s_unsupported",
        "bitnet_3b_x86_tl2_candidate",
        "bitnet_onebit_large_diagnostic",
        "bitnet_llama3_8b_158_diagnostic",
        "bitnet_falcon3_falcon_e_158_diagnostic",
        "bitnet_mcu_tiny_fixture",
        "small_llm_qwen25_15b_q4km_candidate",
        "small_llm_qwen3_17b_q8_candidate",
        "small_llm_llama32_3b_candidate",
        "small_llm_gemma_2b_candidate",
        "modern_llm_dense_frontier_placeholder",
        "modern_llm_moe_frontier_placeholder",
        "modern_llm_multimodal_placeholder",
        "modern_llm_placeholder_contract",
    ] {
        if !seen_ids.contains(required) {
            bail!("matrix missing required coverage entry `{required}`");
        }
    }

    Ok(())
}

fn validate_entry(entry: &Entry, tiers: &HashSet<&str>) -> Result<()> {
    for (field, value) in [
        ("id", &entry.id),
        ("model_class", &entry.model_class),
        ("family", &entry.family),
        ("artifact_kind", &entry.artifact_kind),
        ("status", &entry.status),
        ("current_tier", &entry.current_tier),
        ("verifier_surface", &entry.verifier_surface),
        ("tokenizer_authority", &entry.tokenizer_authority),
        ("prompt_authority", &entry.prompt_authority),
        ("cpu_reference", &entry.cpu_reference),
        ("next_proof", &entry.next_proof),
        ("claim_boundary", &entry.claim_boundary),
    ] {
        require_nonempty(value, field).with_context(|| format!("entry `{}`", entry.id))?;
    }

    if !tiers.contains(entry.current_tier.as_str()) {
        bail!("entry `{}` uses unknown tier `{}`", entry.id, entry.current_tier);
    }
    if entry.required_receipts.is_empty() {
        bail!("entry `{}` has no required_receipts", entry.id);
    }
    if entry.forbidden_claims.is_empty() {
        bail!("entry `{}` has no forbidden_claims", entry.id);
    }
    if !entry.claims.registered {
        bail!("entry `{}` must at least be registered", entry.id);
    }

    validate_claim_progression(entry)?;
    validate_claim_boundaries(entry)?;
    Ok(())
}

fn validate_claim_progression(entry: &Entry) -> Result<()> {
    let c = &entry.claims;
    if c.structurally_valid && !c.registered {
        bail!("entry `{}` is structurally_valid without registered", entry.id);
    }
    if c.reference_good && !c.structurally_valid {
        bail!("entry `{}` is reference_good without structural validity", entry.id);
    }
    if c.cpu_answer_ready && !c.reference_good {
        bail!("entry `{}` is cpu_answer_ready without reference_good", entry.id);
    }
    if c.accelerator_answer_ready && !c.cpu_answer_ready {
        bail!("entry `{}` is accelerator_answer_ready without cpu_answer_ready", entry.id);
    }
    if c.benchmark_qualified && !c.accelerator_answer_ready {
        bail!("entry `{}` is benchmark_qualified without accelerator_answer_ready", entry.id);
    }
    if c.product_cli_ready && !c.accelerator_answer_ready {
        bail!("entry `{}` is product_cli_ready without accelerator_answer_ready", entry.id);
    }
    if c.server_ready && !c.product_cli_ready {
        bail!("entry `{}` is server_ready without product_cli_ready", entry.id);
    }
    if c.speedup_claim && !c.benchmark_qualified {
        bail!("entry `{}` claims speedup without benchmark qualification", entry.id);
    }
    if c.full_residency_claim && !c.accelerator_answer_ready {
        bail!("entry `{}` claims full residency without accelerator answer readiness", entry.id);
    }
    Ok(())
}

fn validate_claim_boundaries(entry: &Entry) -> Result<()> {
    let c = &entry.claims;
    if entry.status == "unsupported_upstream"
        && (c.structurally_valid
            || c.reference_good
            || c.cpu_answer_ready
            || c.accelerator_answer_ready
            || c.benchmark_qualified
            || c.product_cli_ready
            || c.server_ready
            || c.speedup_claim)
    {
        bail!("unsupported entry `{}` has a proof or product claim", entry.id);
    }
    if entry.model_class != "bitnet" && c.bitnet_packed_i2s_qk256_proof {
        bail!("non-BitNet entry `{}` claims BitNet packed I2_S/QK256 proof", entry.id);
    }
    if entry.model_class == "bitnet"
        && entry.artifact_kind != "gguf_i2_s"
        && c.bitnet_packed_i2s_qk256_proof
    {
        bail!(
            "BitNet entry `{}` claims packed I2_S/QK256 proof for non-I2_S artifact `{}`",
            entry.id,
            entry.artifact_kind
        );
    }
    if entry.model_class == "bitnet" && c.dense_regular_llm_cuda_proof {
        bail!("BitNet entry `{}` claims dense regular-LLM CUDA proof", entry.id);
    }
    if c.dense_regular_llm_cuda_proof
        && !entry.accelerator_routes.iter().any(|route| route == "dense_regular_llm_cuda")
    {
        bail!(
            "entry `{}` claims dense regular-LLM CUDA proof without a dense_regular_llm_cuda route",
            entry.id
        );
    }
    if c.accelerator_answer_ready && entry.accelerator_routes.is_empty() {
        bail!("entry `{}` is accelerator-ready but has no accelerator route", entry.id);
    }
    if entry.model_class == "small_llm"
        && !entry.required_receipts.iter().any(|receipt| receipt == "memory_envelope")
    {
        bail!("small LLM entry `{}` must require a memory_envelope receipt", entry.id);
    }
    if entry.model_class == "modern_llm_docs_only" {
        if entry.status != "docs_only_placeholder" {
            bail!("modern LLM docs-only entry `{}` must stay docs_only_placeholder", entry.id);
        }
        if !entry.accelerator_routes.is_empty() {
            bail!("modern LLM docs-only entry `{}` cannot define accelerator routes", entry.id);
        }
        if !entry
            .required_receipts
            .iter()
            .any(|receipt| receipt == "unsupported_on_current_hardware_receipt")
        {
            bail!(
                "modern LLM docs-only entry `{}` must require an unsupported_on_current_hardware_receipt",
                entry.id
            );
        }
        if c.structurally_valid
            || c.reference_good
            || c.cpu_answer_ready
            || c.accelerator_answer_ready
            || c.benchmark_qualified
            || c.product_cli_ready
            || c.server_ready
            || c.speedup_claim
            || c.full_residency_claim
            || c.bitnet_packed_i2s_qk256_proof
            || c.dense_regular_llm_cuda_proof
        {
            bail!("modern LLM docs-only entry `{}` has a runtime or proof claim", entry.id);
        }
    }
    if c.server_ready {
        validate_server_ready_claim(entry)?;
    }
    Ok(())
}

fn validate_server_ready_claim(entry: &Entry) -> Result<()> {
    const ACCEPTED_SERVER_READY_ENTRIES: &[&str] =
        &["dense_qwen25_05b_q8_cuda", "dense_qwen3_06b_q8_candidate"];
    if !ACCEPTED_SERVER_READY_ENTRIES.contains(&entry.id.as_str()) {
        bail!(
            "entry `{}` claims server readiness without an accepted exact-profile server receipt",
            entry.id,
        );
    }
    if !entry
        .required_receipts
        .iter()
        .any(|receipt| receipt == "server_shared_engine_chat_completion")
    {
        bail!(
            "entry `{}` claims server readiness without a server_shared_engine_chat_completion receipt",
            entry.id
        );
    }

    let c = &entry.claims;
    if c.speedup_claim || c.full_residency_claim || c.bitnet_packed_i2s_qk256_proof {
        bail!(
            "entry `{}` claims server readiness with an incompatible speed, residency, or cross-family proof claim",
            entry.id
        );
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} is empty");
    }
    Ok(())
}

fn require_eq<T>(actual: T, expected: T, field: &str) -> Result<()>
where
    T: std::fmt::Debug + PartialEq,
{
    if actual != expected {
        bail!("{field} mismatch: expected {expected:?}, got {actual:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_matrix_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map_or_else(|| PathBuf::from("."), PathBuf::from)
            .join("ci/model-artifacts/model-coverage-matrix.toml")
    }

    #[test]
    fn current_matrix_validates() -> Result<()> {
        let matrix = load_matrix(&workspace_matrix_path())?;
        validate_matrix(&matrix)
    }

    #[test]
    fn dense_entries_cannot_claim_bitnet_packed_proof() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "dense_qwen25_05b_q8_cuda")
        else {
            bail!("missing dense qwen entry");
        };
        entry.claims.bitnet_packed_i2s_qk256_proof = true;
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("dense BitNet proof leak must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("claims BitNet packed I2_S/QK256 proof"), "{err}");
        Ok(())
    }

    #[test]
    fn unsupported_entries_cannot_claim_answer_readiness() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "bitnet_3b_x86_i2s_unsupported")
        else {
            bail!("missing unsupported 3B entry");
        };
        entry.claims.cpu_answer_ready = true;
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("unsupported answer claim must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("without reference_good"), "{err}");
        Ok(())
    }

    #[test]
    fn tl2_entries_cannot_claim_i2s_qk256_proof() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "bitnet_3b_x86_tl2_candidate")
        else {
            bail!("missing 3B TL2 entry");
        };
        entry.claims.bitnet_packed_i2s_qk256_proof = true;
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("TL2 I2_S/QK256 proof leak must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("non-I2_S artifact"), "{err}");
        Ok(())
    }

    #[test]
    fn slm_candidates_cannot_claim_dense_cuda_without_route() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "dense_smollm2_360m_candidate")
        else {
            bail!("missing SmolLM2 360M entry");
        };
        entry.claims.dense_regular_llm_cuda_proof = true;
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("SLM candidate dense CUDA proof leak must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("without a dense_regular_llm_cuda route"), "{err}");
        Ok(())
    }

    #[test]
    fn small_llm_entries_require_memory_envelope_receipts() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "small_llm_qwen25_15b_q4km_candidate")
        else {
            bail!("missing small Qwen entry");
        };
        entry.required_receipts.retain(|receipt| receipt != "memory_envelope");
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("small LLM without memory envelope must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("must require a memory_envelope receipt"), "{err}");
        Ok(())
    }

    #[test]
    fn non_promoted_entries_cannot_claim_server_readiness() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "bitnet_official_2b_i2s_qk256")
        else {
            bail!("missing official BitNet entry");
        };
        entry.claims.server_ready = true;
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("unaccepted server-ready row must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("accepted exact-profile server receipt"), "{err}");
        Ok(())
    }

    #[test]
    fn accepted_server_ready_row_requires_shared_engine_receipt() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) =
            matrix.entry.iter_mut().find(|entry| entry.id == "dense_qwen25_05b_q8_cuda")
        else {
            bail!("missing dense Qwen entry");
        };
        entry.claims.server_ready = true;
        entry.required_receipts.retain(|receipt| receipt != "server_shared_engine_chat_completion");
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("server-ready row without shared-engine receipt must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("server_shared_engine_chat_completion"), "{err}");
        Ok(())
    }

    #[test]
    fn modern_llm_docs_only_entries_cannot_define_routes() -> Result<()> {
        let mut matrix = load_matrix(&workspace_matrix_path())?;
        let Some(entry) = matrix
            .entry
            .iter_mut()
            .find(|entry| entry.id == "modern_llm_dense_frontier_placeholder")
        else {
            bail!("missing modern dense placeholder entry");
        };
        entry.accelerator_routes.push("dense_regular_llm_cuda".to_string());
        let err = match validate_matrix(&matrix) {
            Ok(()) => bail!("modern docs-only route leak must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("cannot define accelerator routes"), "{err}");
        Ok(())
    }
}
