use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bitnet_prompt_templates::TemplateType;

const REQUIRED_NOT_CLAIMS: &[&str] = &[
    "prompt_suite_quality_claim",
    "benchmark_speed_claim",
    "selected_attention_residency",
    "resident_kv_decode",
    "full_device_residency",
    "completion",
];

#[derive(Debug, Deserialize)]
struct PromptSuite {
    schema_version: u32,
    suite_id: String,
    #[serde(default)]
    description: Option<String>,
    template: String,
    #[serde(default = "default_true")]
    add_bos: bool,
    #[serde(default = "default_true")]
    parse_special: bool,
    #[serde(default)]
    manual_review_allowed_for_claims: bool,
    required_categories: Vec<String>,
    anti_fakery: AntiFakeryPolicy,
    #[serde(default)]
    case: Vec<PromptCase>,
}

#[derive(Debug, Deserialize)]
struct AntiFakeryPolicy {
    require_seed_binding: bool,
    require_answer_binding: bool,
    require_name_slots: bool,
    minimum_slot_values: usize,
    min_generation_policy_coverage: f64,
    min_answer_bound_surface_coverage: f64,
    min_required_surface_category_coverage: f64,
}

#[derive(Debug, Deserialize)]
struct PromptCase {
    id: String,
    seed: u64,
    category: String,
    #[serde(default)]
    difficulty: Option<String>,
    oracle: String,
    max_new_tokens: u32,
    temperature: f64,
    #[serde(default)]
    requires_pair: bool,
    generation_policy: String,
    answer_bound_surface: String,
    required_surface_category: String,
    expected_behavior: String,
    prompt: String,
    #[serde(default)]
    pair_prompt: Option<String>,
    #[serde(default)]
    stop_expectation: Option<String>,
    #[serde(default)]
    name_slots: Vec<NameSlot>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NameSlot {
    slot: String,
    values: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PromptSuiteVerifyReport {
    diagnostic: &'static str,
    producer: &'static str,
    suite_path: String,
    schema_version: u32,
    suite_id: String,
    description: Option<String>,
    template: String,
    passed: bool,
    case_count: usize,
    required_category_count: usize,
    covered_category_count: usize,
    generation_policy_coverage: f64,
    answer_bound_surface_coverage: f64,
    required_surface_category_coverage: f64,
    seeded_name_slot_case_count: usize,
    manual_review_case_count: usize,
    categories_missing: Vec<String>,
    failures: Vec<String>,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PromptSuiteRenderReport {
    diagnostic: &'static str,
    producer: &'static str,
    suite_path: String,
    suite_id: String,
    template: String,
    model_contract: Option<String>,
    tokenizer_authority: bool,
    tokenizer_path: Option<String>,
    tokenizer_missing: Option<String>,
    add_bos: bool,
    parse_special: bool,
    case_count: usize,
    cases: Vec<RenderedCase>,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RenderedCase {
    id: String,
    category: String,
    difficulty: Option<String>,
    oracle: String,
    max_new_tokens: u32,
    temperature: f64,
    slot_bindings: BTreeMap<String, String>,
    raw_prompt_sha256: String,
    rendered_prompt_sha256: String,
    prompt_token_ids_sha256: Option<String>,
    prompt_token_count: Option<usize>,
    pair_raw_prompt_sha256: Option<String>,
    pair_rendered_prompt_sha256: Option<String>,
    pair_token_ids_sha256: Option<String>,
    pair_prompt_token_count: Option<usize>,
}

struct TokenizerAuthority {
    path: String,
    tokenizer: Option<Arc<dyn bitnet_tokenizers::Tokenizer + Send + Sync>>,
    missing: Option<String>,
}

pub fn verify_suite(suite_path: &Path, format: &str) -> Result<()> {
    let suite = read_suite(suite_path)?;
    let report = build_verify_report(suite_path, &suite);
    emit_verify_report(&report, format)?;
    if !report.passed {
        bail!("prompt-suite verify failed: {}", report.failures.join(", "));
    }
    Ok(())
}

pub fn render_suite(suite_path: &Path, model_contract: Option<&Path>, format: &str) -> Result<()> {
    let suite = read_suite(suite_path)?;
    let verify_report = build_verify_report(suite_path, &suite);
    if !verify_report.passed {
        bail!(
            "prompt-suite render requires a verified suite: {}",
            verify_report.failures.join(", ")
        );
    }
    let tokenizer = if let Some(contract) = model_contract {
        Some(load_contract_tokenizer(contract)?)
    } else {
        None
    };
    let report = build_render_report(suite_path, &suite, model_contract, tokenizer)?;
    emit_render_report(&report, format)?;
    Ok(())
}

fn read_suite(path: &Path) -> Result<PromptSuite> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn build_verify_report(suite_path: &Path, suite: &PromptSuite) -> PromptSuiteVerifyReport {
    let mut failures = Vec::new();
    if suite.schema_version != 1 {
        failures.push(format!("schema_version must be 1, got {}", suite.schema_version));
    }
    if suite.suite_id.trim().is_empty() {
        failures.push("suite_id must not be empty".to_string());
    }
    match suite.template.parse::<TemplateType>() {
        Ok(template_type) => {
            if suite.add_bos != template_type.should_add_bos() {
                failures.push(format!(
                    "suite add_bos={} does not match template {} policy {}",
                    suite.add_bos,
                    suite.template,
                    template_type.should_add_bos()
                ));
            }
            if suite.parse_special != template_type.parse_special() {
                failures.push(format!(
                    "suite parse_special={} does not match template {} policy {}",
                    suite.parse_special,
                    suite.template,
                    template_type.parse_special()
                ));
            }
        }
        Err(_) => failures.push(format!("invalid prompt template {}", suite.template)),
    }
    if suite.case.is_empty() {
        failures.push("suite has no cases".to_string());
    }
    if suite.required_categories.is_empty() {
        failures.push("suite has no required_categories".to_string());
    }
    if suite.manual_review_allowed_for_claims {
        failures.push("manual_review_allowed_for_claims must be false".to_string());
    }
    if !suite.anti_fakery.require_seed_binding {
        failures.push("anti_fakery.require_seed_binding must be true".to_string());
    }
    if !suite.anti_fakery.require_answer_binding {
        failures.push("anti_fakery.require_answer_binding must be true".to_string());
    }
    if !suite.anti_fakery.require_name_slots {
        failures.push("anti_fakery.require_name_slots must be true".to_string());
    }
    if suite.anti_fakery.minimum_slot_values < 3 {
        failures.push("anti_fakery.minimum_slot_values must be at least 3".to_string());
    }

    let mut ids = BTreeSet::new();
    let mut categories = BTreeSet::new();
    let mut surface_categories = BTreeSet::new();
    let mut generation_policy_count = 0;
    let mut answer_bound_surface_count = 0;
    let mut seeded_name_slot_case_count = 0;
    let mut manual_review_case_count = 0;

    for case in &suite.case {
        if !ids.insert(case.id.clone()) {
            failures.push(format!("duplicate case id {}", case.id));
        }
        if case.seed == 0 && suite.anti_fakery.require_seed_binding {
            failures.push(format!("{} has zero seed", case.id));
        }
        if case.prompt.trim().is_empty() {
            failures.push(format!("{} has empty prompt", case.id));
        }
        if case.expected_behavior.trim().is_empty() && suite.anti_fakery.require_answer_binding {
            failures.push(format!("{} missing expected_behavior", case.id));
        }
        if case.max_new_tokens == 0 {
            failures.push(format!("{} has max_new_tokens=0", case.id));
        }
        if case.temperature != 0.0 {
            failures.push(format!(
                "{} has non-deterministic temperature {}",
                case.id, case.temperature
            ));
        }
        if case.oracle == "manual_review_needed" {
            manual_review_case_count += 1;
            failures.push(format!("{} uses manual_review_needed", case.id));
        }
        if !is_known_oracle(&case.oracle) {
            failures.push(format!("{} has unknown oracle {}", case.id, case.oracle));
        }
        if (case.requires_pair || case.oracle == "semantic_pair_difference")
            && case.pair_prompt.as_deref().is_none_or(str::is_empty)
        {
            failures.push(format!("{} requires pair_prompt", case.id));
        }
        if (case.oracle == "repetition_guard"
            || case.oracle == "stop_condition"
            || case.category == "stop_repetition_stress")
            && case.stop_expectation.as_deref().is_none_or(str::is_empty)
        {
            failures.push(format!("{} missing stop_expectation", case.id));
        }
        if !case.generation_policy.trim().is_empty() {
            generation_policy_count += 1;
        }
        if !case.answer_bound_surface.trim().is_empty() {
            answer_bound_surface_count += 1;
        }
        if !case.required_surface_category.trim().is_empty() {
            surface_categories.insert(case.required_surface_category.clone());
        }
        categories.insert(case.category.clone());

        if case.name_slots.is_empty() && suite.anti_fakery.require_name_slots {
            failures.push(format!("{} missing seeded name_slots", case.id));
        }
        if !case.name_slots.is_empty() {
            seeded_name_slot_case_count += 1;
        }
        for slot in &case.name_slots {
            if slot.values.len() < suite.anti_fakery.minimum_slot_values {
                failures.push(format!(
                    "{} slot {} has fewer than {} values",
                    case.id, slot.slot, suite.anti_fakery.minimum_slot_values
                ));
            }
            let needle = format!("{{{}}}", slot.slot);
            let prompt_has_slot = case.prompt.contains(&needle)
                || case.pair_prompt.as_deref().is_some_and(|pair| pair.contains(&needle));
            if !prompt_has_slot {
                failures.push(format!("{} slot {} is not used in prompt text", case.id, slot.slot));
            }
        }
    }

    let categories_missing = suite
        .required_categories
        .iter()
        .filter(|category| !categories.contains(*category))
        .cloned()
        .collect::<Vec<_>>();
    for category in &categories_missing {
        failures.push(format!("missing required category {category}"));
    }

    let case_count = suite.case.len();
    let denominator = case_count.max(1) as f64;
    let generation_policy_coverage = generation_policy_count as f64 / denominator;
    let answer_bound_surface_coverage = answer_bound_surface_count as f64 / denominator;
    let required_surface_category_coverage =
        surface_categories.len() as f64 / suite.required_categories.len().max(1) as f64;
    if generation_policy_coverage < suite.anti_fakery.min_generation_policy_coverage {
        failures.push("generation policy coverage below minimum".to_string());
    }
    if answer_bound_surface_coverage < suite.anti_fakery.min_answer_bound_surface_coverage {
        failures.push("answer-bound surface coverage below minimum".to_string());
    }
    if required_surface_category_coverage < suite.anti_fakery.min_required_surface_category_coverage
    {
        failures.push("required surface category coverage below minimum".to_string());
    }

    PromptSuiteVerifyReport {
        diagnostic: "prompt_suite_verify",
        producer: "cargo xtask prompt-suite verify",
        suite_path: suite_path.display().to_string(),
        schema_version: suite.schema_version,
        suite_id: suite.suite_id.clone(),
        description: suite.description.clone(),
        template: suite.template.clone(),
        passed: failures.is_empty(),
        case_count,
        required_category_count: suite.required_categories.len(),
        covered_category_count: categories.len(),
        generation_policy_coverage,
        answer_bound_surface_coverage,
        required_surface_category_coverage,
        seeded_name_slot_case_count,
        manual_review_case_count,
        categories_missing,
        failures,
        not_claims: REQUIRED_NOT_CLAIMS.iter().map(|value| (*value).to_string()).collect(),
    }
}

fn build_render_report(
    suite_path: &Path,
    suite: &PromptSuite,
    model_contract: Option<&Path>,
    tokenizer: Option<TokenizerAuthority>,
) -> Result<PromptSuiteRenderReport> {
    let template_type: TemplateType = suite
        .template
        .parse()
        .with_context(|| format!("parsing prompt-suite template {}", suite.template))?;
    let tokenizer_missing = tokenizer.as_ref().and_then(|authority| authority.missing.clone());
    let tokenizer_ref = tokenizer
        .as_ref()
        .and_then(|authority| authority.tokenizer.as_ref())
        .map(|tokenizer| tokenizer.as_ref());
    let tokenizer_path = tokenizer.as_ref().map(|authority| authority.path.clone());
    let mut cases = Vec::new();
    for case in &suite.case {
        cases.push(render_case(
            case,
            template_type,
            tokenizer_ref,
            suite.add_bos,
            suite.parse_special,
        )?);
    }

    Ok(PromptSuiteRenderReport {
        diagnostic: "prompt_suite_render",
        producer: "cargo xtask prompt-suite render",
        suite_path: suite_path.display().to_string(),
        suite_id: suite.suite_id.clone(),
        template: suite.template.clone(),
        model_contract: model_contract.map(|path| path.display().to_string()),
        tokenizer_authority: tokenizer_ref.is_some(),
        tokenizer_path,
        tokenizer_missing,
        add_bos: suite.add_bos,
        parse_special: suite.parse_special,
        case_count: cases.len(),
        cases,
        not_claims: REQUIRED_NOT_CLAIMS.iter().map(|value| (*value).to_string()).collect(),
    })
}

fn render_case(
    case: &PromptCase,
    template_type: TemplateType,
    tokenizer: Option<&(dyn bitnet_tokenizers::Tokenizer + Send + Sync)>,
    add_bos: bool,
    parse_special: bool,
) -> Result<RenderedCase> {
    let slot_bindings = bind_slots(case.seed, &case.name_slots);
    let raw_prompt = apply_slots(&case.prompt, &slot_bindings);
    let rendered = template_type.apply(&raw_prompt, None);
    let (prompt_token_ids_sha256, prompt_token_count) =
        tokenize_hash(tokenizer, &rendered, add_bos, parse_special)?;
    let pair_raw = case.pair_prompt.as_ref().map(|prompt| apply_slots(prompt, &slot_bindings));
    let pair_rendered = pair_raw.as_ref().map(|prompt| template_type.apply(prompt, None));
    let (pair_token_ids_sha256, pair_prompt_token_count) = if let Some(pair) = &pair_rendered {
        tokenize_hash(tokenizer, pair, add_bos, parse_special)?
    } else {
        (None, None)
    };

    Ok(RenderedCase {
        id: case.id.clone(),
        category: case.category.clone(),
        difficulty: case.difficulty.clone(),
        oracle: case.oracle.clone(),
        max_new_tokens: case.max_new_tokens,
        temperature: case.temperature,
        slot_bindings,
        raw_prompt_sha256: sha256_text(&raw_prompt),
        rendered_prompt_sha256: sha256_text(&rendered),
        prompt_token_ids_sha256,
        prompt_token_count,
        pair_raw_prompt_sha256: pair_raw.as_deref().map(sha256_text),
        pair_rendered_prompt_sha256: pair_rendered.as_deref().map(sha256_text),
        pair_token_ids_sha256,
        pair_prompt_token_count,
    })
}

fn bind_slots(seed: u64, slots: &[NameSlot]) -> BTreeMap<String, String> {
    let mut bindings = BTreeMap::new();
    for slot in slots {
        if slot.values.is_empty() {
            continue;
        }
        let mut hasher = Sha256::new();
        hasher.update(seed.to_le_bytes());
        hasher.update(slot.slot.as_bytes());
        let digest = hasher.finalize();
        let index_seed = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]);
        let index = index_seed as usize % slot.values.len();
        bindings.insert(slot.slot.clone(), slot.values[index].clone());
    }
    bindings
}

fn is_known_oracle(oracle: &str) -> bool {
    matches!(
        oracle,
        "regex"
            | "semantic_nonempty"
            | "json_schema"
            | "semantic_pair_difference"
            | "structural"
            | "repetition_guard"
            | "stop_condition"
    )
}

fn apply_slots(prompt: &str, bindings: &BTreeMap<String, String>) -> String {
    let mut rendered = prompt.to_string();
    for (slot, value) in bindings {
        rendered = rendered.replace(&format!("{{{slot}}}"), value);
    }
    rendered
}

fn tokenize_hash(
    tokenizer: Option<&(dyn bitnet_tokenizers::Tokenizer + Send + Sync)>,
    rendered: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<(Option<String>, Option<usize>)> {
    let Some(tokenizer) = tokenizer else {
        return Ok((None, None));
    };
    let tokens = tokenizer
        .encode(rendered, add_bos, parse_special)
        .with_context(|| "tokenizing rendered prompt")?;
    let encoded = serde_json::to_vec(&tokens)?;
    Ok((Some(sha256_bytes(&encoded)), Some(tokens.len())))
}

fn load_contract_tokenizer(contract_path: &Path) -> Result<TokenizerAuthority> {
    let raw = fs::read_to_string(contract_path)
        .with_context(|| format!("reading {}", contract_path.display()))?;
    let value: Value = serde_yaml::from_str(&raw)
        .with_context(|| format!("parsing {}", contract_path.display()))?;
    let tokenizer_path = value
        .pointer("/tokenizer/path")
        .and_then(Value::as_str)
        .context("model contract missing /tokenizer/path")?;
    let path = PathBuf::from(tokenizer_path);
    if !path.exists() {
        return Ok(TokenizerAuthority {
            path: tokenizer_path.to_string(),
            tokenizer: None,
            missing: Some(format!("tokenizer path does not exist: {}", path.display())),
        });
    }
    let tokenizer = bitnet_tokenizers::load_tokenizer(&path)
        .with_context(|| format!("loading tokenizer {}", path.display()))?;
    Ok(TokenizerAuthority {
        path: tokenizer_path.to_string(),
        tokenizer: Some(tokenizer),
        missing: None,
    })
}

fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
}

fn sha256_bytes(value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}

fn emit_verify_report(report: &PromptSuiteVerifyReport, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            println!("prompt-suite verify: passed={}", report.passed);
            println!("suite: {}", report.suite_id);
            println!("cases: {}", report.case_count);
            println!(
                "coverage: generation_policy={:.2} answer_bound_surface={:.2} surface_category={:.2}",
                report.generation_policy_coverage,
                report.answer_bound_surface_coverage,
                report.required_surface_category_coverage
            );
            if !report.failures.is_empty() {
                println!("failures: {}", report.failures.join(", "));
            }
            println!("not_claims: {}", report.not_claims.join(", "));
        }
        other => bail!("unsupported prompt-suite output format: {other}"),
    }
    Ok(())
}

fn emit_render_report(report: &PromptSuiteRenderReport, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            println!("prompt-suite render: suite={}", report.suite_id);
            println!("cases: {}", report.case_count);
            println!("tokenizer_authority: {}", report.tokenizer_authority);
            if let Some(missing) = &report.tokenizer_missing {
                println!("tokenizer_missing: {missing}");
            }
            println!("not_claims: {}", report.not_claims.join(", "));
        }
        other => bail!("unsupported prompt-suite output format: {other}"),
    }
    Ok(())
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_suite(extra_case: &str) -> String {
        format!(
            r#"
schema_version = 1
suite_id = "test-suite"
template = "llama3-chat"
add_bos = false
parse_special = true
required_categories = ["short_explanation"]
manual_review_allowed_for_claims = false

[anti_fakery]
require_seed_binding = true
require_answer_binding = true
require_name_slots = true
minimum_slot_values = 3
min_generation_policy_coverage = 1.0
min_answer_bound_surface_coverage = 1.0
min_required_surface_category_coverage = 1.0

{extra_case}
"#
        )
    }

    fn valid_case(id: &str) -> String {
        format!(
            r#"
[[case]]
id = "{id}"
seed = 7
category = "short_explanation"
oracle = "semantic_nonempty"
max_new_tokens = 16
temperature = 0.0
generation_policy = "deterministic_greedy"
answer_bound_surface = "must mention reuse"
required_surface_category = "short_explanation"
expected_behavior = "explain briefly"
prompt = "{{name}} asks why caching helps."
name_slots = [
  {{ slot = "name", values = ["Ava", "Bo", "Cy"] }},
]
"#
        )
    }

    #[test]
    fn verify_rejects_duplicate_ids() -> Result<()> {
        let raw = minimal_suite(&format!("{}\n{}", valid_case("same"), valid_case("same")));
        let suite: PromptSuite = toml::from_str(&raw)?;
        let report = build_verify_report(Path::new("suite.toml"), &suite);
        anyhow::ensure!(!report.passed, "duplicate IDs unexpectedly passed");
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("duplicate case id")),
            "failures did not mention duplicate ID: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn verify_rejects_manual_review_oracle() -> Result<()> {
        let mut case = valid_case("manual");
        case = case.replace("semantic_nonempty", "manual_review_needed");
        let raw = minimal_suite(&case);
        let suite: PromptSuite = toml::from_str(&raw)?;
        let report = build_verify_report(Path::new("suite.toml"), &suite);
        anyhow::ensure!(!report.passed, "manual review oracle unexpectedly passed");
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("manual_review_needed")),
            "failures did not mention manual_review_needed: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn verify_rejects_disabled_anti_fakery_policy() -> Result<()> {
        let raw = minimal_suite(&valid_case("policy"))
            .replace("require_seed_binding = true", "require_seed_binding = false");
        let suite: PromptSuite = toml::from_str(&raw)?;
        let report = build_verify_report(Path::new("suite.toml"), &suite);
        anyhow::ensure!(!report.passed, "disabled anti-fakery policy unexpectedly passed");
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("require_seed_binding")),
            "failures did not mention require_seed_binding: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn verify_rejects_unknown_oracle() -> Result<()> {
        let raw = minimal_suite(&valid_case("oracle")).replace("semantic_nonempty", "unknown");
        let suite: PromptSuite = toml::from_str(&raw)?;
        let report = build_verify_report(Path::new("suite.toml"), &suite);
        anyhow::ensure!(!report.passed, "unknown oracle unexpectedly passed");
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("unknown oracle")),
            "failures did not mention unknown oracle: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn verify_rejects_template_policy_mismatch() -> Result<()> {
        let raw = minimal_suite(&valid_case("policy")).replace("add_bos = false", "add_bos = true");
        let suite: PromptSuite = toml::from_str(&raw)?;
        let report = build_verify_report(Path::new("suite.toml"), &suite);
        anyhow::ensure!(!report.passed, "template policy mismatch unexpectedly passed");
        anyhow::ensure!(
            report.failures.iter().any(|failure| failure.contains("add_bos=true")),
            "failures did not mention add_bos mismatch: {:?}",
            report.failures
        );
        Ok(())
    }

    #[test]
    fn render_binds_seeded_slots_and_hashes_prompt() -> Result<()> {
        let raw = minimal_suite(&valid_case("render"));
        let suite: PromptSuite = toml::from_str(&raw)?;
        let report = build_render_report(Path::new("suite.toml"), &suite, None, None)?;
        anyhow::ensure!(report.case_count == 1, "unexpected case count {}", report.case_count);
        let case = &report.cases[0];
        anyhow::ensure!(case.slot_bindings.contains_key("name"), "missing name slot binding");
        let raw_prompt = apply_slots(&suite.case[0].prompt, &case.slot_bindings);
        let rendered_prompt = TemplateType::Llama3Chat.apply(&raw_prompt, None);
        anyhow::ensure!(
            case.raw_prompt_sha256 == sha256_text(&raw_prompt),
            "raw prompt hash mismatch"
        );
        anyhow::ensure!(
            case.rendered_prompt_sha256 == sha256_text(&rendered_prompt),
            "rendered prompt hash mismatch"
        );
        anyhow::ensure!(
            rendered_prompt.starts_with("<|begin_of_text|><|start_header_id|>user"),
            "rendered prompt did not use the Llama 3 chat template: {rendered_prompt}"
        );
        anyhow::ensure!(
            case.rendered_prompt_sha256.len() == 64,
            "unexpected prompt hash length {}",
            case.rendered_prompt_sha256.len()
        );
        anyhow::ensure!(
            case.prompt_token_ids_sha256.is_none(),
            "token hash should be absent without tokenizer"
        );
        Ok(())
    }
}
