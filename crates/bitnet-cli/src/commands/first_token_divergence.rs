//! First-token divergence classifier for Lunar Lake BitNet CPU bring-up.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Classify the first available divergence between external reference and 258V CPU receipts.
#[derive(Args, Debug)]
pub struct FirstTokenDivergenceCommand {
    /// External first-token reference capture artifact.
    #[arg(long, value_name = "PATH")]
    pub external_reference: PathBuf,

    /// Prompt-authority audit artifact for the same model/prompt policy.
    #[arg(long, value_name = "PATH")]
    pub prompt_audit: PathBuf,

    /// Scalar CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub scalar_answer_corpus: PathBuf,

    /// AVX2 CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub avx2_answer_corpus: PathBuf,

    /// Scalar-vs-AVX2 answer-parity receipt.
    #[arg(long, value_name = "PATH")]
    pub answer_parity: PathBuf,

    /// Output divergence-classification receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/first-token-divergence-classification.json"
    )]
    pub json_out: PathBuf,
}

impl FirstTokenDivergenceCommand {
    /// Execute offline divergence classification.
    pub async fn execute(&self) -> Result<()> {
        let external_reference = read_json(&self.external_reference)?;
        let prompt_audit = read_json(&self.prompt_audit)?;
        let scalar = read_json(&self.scalar_answer_corpus)?;
        let avx2 = read_json(&self.avx2_answer_corpus)?;
        let answer_parity = read_json(&self.answer_parity)?;

        let receipt = build_first_token_divergence_receipt(
            &FirstTokenInputs {
                external_reference: &self.external_reference,
                prompt_audit: &self.prompt_audit,
                scalar_answer_corpus: &self.scalar_answer_corpus,
                avx2_answer_corpus: &self.avx2_answer_corpus,
                answer_parity: &self.answer_parity,
            },
            &external_reference,
            &prompt_audit,
            &scalar,
            &avx2,
            &answer_parity,
        );

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("first-token divergence classification written to {}", self.json_out.display());

        if receipt["validation"]["passed"].as_bool() != Some(true) {
            anyhow::bail!(
                "first-token divergence inputs failed validation; receipt written to {}",
                self.json_out.display()
            );
        }
        Ok(())
    }
}

struct FirstTokenInputs<'a> {
    external_reference: &'a Path,
    prompt_audit: &'a Path,
    scalar_answer_corpus: &'a Path,
    avx2_answer_corpus: &'a Path,
    answer_parity: &'a Path,
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn build_first_token_divergence_receipt(
    inputs: &FirstTokenInputs<'_>,
    external_reference: &Value,
    prompt_audit: &Value,
    scalar: &Value,
    avx2: &Value,
    answer_parity: &Value,
) -> Value {
    let validation_failures =
        validate_inputs(external_reference, prompt_audit, scalar, avx2, answer_parity);
    let bos_id = external_reference["tokenizer"]["bos_token_id"].as_u64().unwrap_or(128000);
    let cases = external_reference["cases"]
        .as_array()
        .map(|items| {
            items.iter().map(|case| classify_case(case, scalar, avx2, bos_id)).collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let reference_evidence = summarize_reference_evidence(external_reference);
    let case_summaries = cases.iter().map(|case| &case["classification"]).collect::<Vec<_>>();
    let scalar_avx2_parity_passed = answer_parity["summary"]["failed"].as_u64() == Some(0)
        && answer_parity["summary"]["first_divergence"].is_null();
    let first_divergence = cases
        .iter()
        .find_map(|case| {
            let stage = case["classification"]["first_divergence_stage"].as_str()?;
            (stage != "inconclusive" && stage != "none").then(|| case["classification"].clone())
        })
        .or_else(|| {
            cases.iter().find_map(|case| {
                (case["classification"]["first_divergence_stage"] == "inconclusive")
                    .then(|| case["classification"].clone())
            })
        })
        .unwrap_or_else(|| no_divergence_summary(&cases));
    let next_required_evidence =
        next_required_evidence(external_reference, &reference_evidence, &first_divergence);
    let classification = first_divergence["classification"].as_str().unwrap_or("unknown");

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_first_token_divergence_classification",
        "machine_id": external_reference["machine_id"].clone(),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "first_token_divergence_classified",
        "claim": "cpu258v_external_reference_divergence_classifier",
        "inputs": {
            "external_reference": inputs.external_reference.display().to_string(),
            "prompt_audit": inputs.prompt_audit.display().to_string(),
            "scalar_answer_corpus": inputs.scalar_answer_corpus.display().to_string(),
            "avx2_answer_corpus": inputs.avx2_answer_corpus.display().to_string(),
            "answer_parity": inputs.answer_parity.display().to_string(),
        },
        "validation": {
            "passed": validation_failures.is_empty(),
            "failed_rules": validation_failures,
        },
        "model": external_reference["model"].clone(),
        "tokenizer": external_reference["tokenizer"].clone(),
        "external_reference": {
            "runner": external_reference["reference"]["runner"].clone(),
            "command_shape": external_reference["reference"]["command_shape"].clone(),
            "generated_token_ids_available": external_reference["reference"]["generated_token_ids_available"].clone(),
            "logits_available": external_reference["reference"]["logits_available"].clone(),
            "missing_logits_status": external_reference["reference"]["missing_logits_status"].clone(),
        },
        "prompt_authority_audit": {
            "first_divergence_stage": prompt_audit["classification"]["first_divergence_stage"].clone(),
            "first_mismatch_index": prompt_audit["classification"]["first_mismatch_index"].clone(),
            "notes": prompt_audit["classification"]["notes"].clone(),
        },
        "scalar_avx2_parity": {
            "passed": scalar_avx2_parity_passed,
            "summary": answer_parity["summary"].clone(),
        },
        "summary": {
            "cases_total": cases.len(),
            "cases_inconclusive": count_classification(&case_summaries, "inconclusive"),
            "prompt_token_exact_matches": count_bool(&cases, &["comparisons", "reference_vs_scalar_prompt_exact_match"]),
            "prompt_token_local_bos_prefix_matches": count_bool(&cases, &["comparisons", "reference_vs_scalar_prompt_local_bos_prefix_match"]),
            "generated_text_trimmed_scalar_matches": count_bool(&cases, &["comparisons", "reference_scalar_generated_text_trimmed_match"]),
            "generated_text_trimmed_avx2_matches": count_bool(&cases, &["comparisons", "reference_avx2_generated_text_trimmed_match"]),
            "generated_text_trimmed_scalar_avx2_matches": count_scalar_avx2_text_matches(&cases),
            "scalar_avx2_parity_passed": scalar_avx2_parity_passed,
            "cases_with_reference_generated_token_ids": reference_evidence.cases_with_generated_token_ids,
            "cases_with_reference_first_token_topk_logits": reference_evidence.cases_with_first_token_topk_logits,
            "reference_generated_token_ids_available": reference_evidence.generated_token_ids_available,
            "reference_logits_available": reference_evidence.first_token_logits_available,
            "first_divergence": first_divergence,
            "classification": classification,
            "next_required_evidence": next_required_evidence,
        },
        "cases": cases,
        "fallback_used": false,
        "claim_boundary": {
            "may_claim": [
                "The receipt classifies the first available evidence boundary between external BitNet reference text and 258V scalar/AVX2 CPU receipts.",
                "When the external reference supplies direct generated-token IDs, the receipt classifies first-generated-token matches or mismatches against the local CPU receipts.",
                "Scalar-vs-AVX2 agreement can be kept separate from missing external generated-token/logit evidence.",
                "Prompt-token comparisons distinguish exact matches from local BOS-prefix policy deltas."
            ],
            "must_not_claim": [
                "first-token logits parity",
                "full generated-token sequence parity beyond the recorded first-token boundary",
                "general answer quality",
                "CPU speed or sustained throughput",
                "Arc 140V execution or acceleration",
                "Intel NPU execution or acceleration",
                "QK256 decode correctness"
            ]
        }
    })
}

fn next_required_evidence(
    external_reference: &Value,
    reference_evidence: &ReferenceEvidenceSummary,
    first_divergence: &Value,
) -> Value {
    if reference_evidence.generated_token_ids_available {
        return match first_divergence["first_divergence_stage"].as_str() {
            Some("none") => json!(
                "none_for_first_generated_token_boundary; use deeper sequence, logits, or layer evidence only if needed"
            ),
            Some("generated_token") if reference_evidence.first_token_logits_available => json!(
                "compare direct reference top-k/logits against local top-k, then classify model math or sampler boundary"
            ),
            Some("generated_token") => json!(
                "capture direct reference first-token top-k/logits to separate sampler from logits or model math"
            ),
            _ => json!(
                "fix the classified earlier divergence before deeper token or logits evidence"
            ),
        };
    }
    external_reference["summary"]["next_required_evidence"].clone()
}

#[derive(Default)]
struct ReferenceEvidenceSummary {
    cases_with_generated_token_ids: usize,
    cases_with_first_token_topk_logits: usize,
    generated_token_ids_available: bool,
    first_token_logits_available: bool,
}

fn summarize_reference_evidence(external_reference: &Value) -> ReferenceEvidenceSummary {
    let cases = external_reference["cases"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mut summary = ReferenceEvidenceSummary::default();
    for case in cases {
        if ids(case.get("generated_token_ids").unwrap_or(&Value::Null))
            .is_some_and(|ids| !ids.is_empty())
        {
            summary.cases_with_generated_token_ids += 1;
        }
        if reference_first_token_topk(case)
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        {
            summary.cases_with_first_token_topk_logits += 1;
        }
    }
    summary.generated_token_ids_available =
        !cases.is_empty() && summary.cases_with_generated_token_ids == cases.len();
    summary.first_token_logits_available =
        !cases.is_empty() && summary.cases_with_first_token_topk_logits == cases.len();
    summary
}

fn no_divergence_summary(cases: &[Value]) -> Value {
    if cases.is_empty() {
        return json!({
            "first_divergence_stage": "unknown",
            "classification": "no_cases_available",
            "evidence_boundary": "no_external_reference_cases_available"
        });
    }
    if cases
        .iter()
        .all(|case| case["classification"]["first_divergence_stage"].as_str() == Some("none"))
    {
        return json!({
            "first_divergence_stage": "none",
            "classification": "no_divergence_at_first_generated_token",
            "evidence_boundary": "all_available_reference_first_generated_token_ids_match_local_cpu"
        });
    }
    json!({
        "first_divergence_stage": "unknown",
        "classification": "no_nonmatching_or_inconclusive_cases_selected",
        "evidence_boundary": "case_classifications_did_not_select_a_divergence_boundary"
    })
}

fn validate_inputs(
    external_reference: &Value,
    prompt_audit: &Value,
    scalar: &Value,
    avx2: &Value,
    answer_parity: &Value,
) -> Vec<&'static str> {
    let mut failures = Vec::new();
    if !is_supported_external_reference_kind(external_reference["artifact_kind"].as_str()) {
        failures.push("external_reference_artifact_kind");
    }
    if prompt_audit["artifact_kind"].as_str() != Some("bitnet_prompt_token_authority_audit") {
        failures.push("prompt_audit_artifact_kind");
    }
    if scalar["artifact_kind"].as_str() != Some("bitnet_cpu_answer_corpus") {
        failures.push("scalar_answer_corpus_artifact_kind");
    }
    if avx2["artifact_kind"].as_str() != Some("bitnet_cpu_answer_corpus") {
        failures.push("avx2_answer_corpus_artifact_kind");
    }
    if answer_parity["artifact_kind"].as_str() != Some("bitnet_cpu_answer_parity") {
        failures.push("answer_parity_artifact_kind");
    }
    if external_reference["cases"].as_array().is_none_or(|cases| cases.is_empty()) {
        failures.push("external_reference_cases");
    }
    if scalar["cases"].as_array().is_none_or(|cases| cases.is_empty()) {
        failures.push("scalar_cases");
    }
    if avx2["cases"].as_array().is_none_or(|cases| cases.is_empty()) {
        failures.push("avx2_cases");
    }
    if answer_parity["summary"]["failed"].as_u64().is_none() {
        failures.push("answer_parity_summary_failed");
    }
    failures
}

fn is_supported_external_reference_kind(kind: Option<&str>) -> bool {
    matches!(
        kind,
        Some(
            "bitnet_external_first_token_reference_capture"
                | "bitnet_external_reference_direct_token_boundary"
        )
    )
}

fn classify_case(reference_case: &Value, scalar: &Value, avx2: &Value, bos_id: u64) -> Value {
    let case_id = reference_case["case_id"].as_str().unwrap_or_default();
    let scalar_case = find_case(scalar, case_id);
    let avx2_case = find_case(avx2, case_id);
    let reference_prompt_ids = ids(&reference_case["prompt_token_ids"]).unwrap_or_default();
    let scalar_prompt_ids = scalar_case.and_then(local_prompt_ids).unwrap_or_default();
    let avx2_prompt_ids = avx2_case.and_then(local_prompt_ids).unwrap_or_default();
    let scalar_generated_ids = scalar_case.and_then(local_generated_ids).unwrap_or_default();
    let avx2_generated_ids = avx2_case.and_then(local_generated_ids).unwrap_or_default();

    let scalar_prompt_exact = reference_prompt_ids == scalar_prompt_ids;
    let avx2_prompt_exact = reference_prompt_ids == avx2_prompt_ids;
    let scalar_prompt_bos = has_local_bos_prefix(&reference_prompt_ids, &scalar_prompt_ids, bos_id);
    let avx2_prompt_bos = has_local_bos_prefix(&reference_prompt_ids, &avx2_prompt_ids, bos_id);
    let scalar_answer = scalar_case.and_then(|case| case["answer"].as_str()).unwrap_or_default();
    let avx2_answer = avx2_case.and_then(|case| case["answer"].as_str()).unwrap_or_default();
    let reference_text = reference_case["reference_generated_text"].as_str().unwrap_or_default();
    let reference_text_matches_scalar = trimmed_eq(reference_text, scalar_answer);
    let reference_text_matches_avx2 = trimmed_eq(reference_text, avx2_answer);
    let reference_generated_ids =
        ids(reference_case.get("generated_token_ids").unwrap_or(&Value::Null)).unwrap_or_default();
    let reference_first_token = reference_case["first_generated_token_id"]
        .as_u64()
        .or_else(|| reference_generated_ids.first().copied());
    let reference_has_generated_ids = !reference_generated_ids.is_empty();
    let reference_topk_logits = reference_first_token_topk(reference_case);
    let reference_has_logits =
        reference_topk_logits.and_then(Value::as_array).is_some_and(|items| !items.is_empty());

    let classification = classify_case_boundary(
        case_id,
        scalar_case.is_some(),
        avx2_case.is_some(),
        scalar_prompt_exact || scalar_prompt_bos,
        avx2_prompt_exact || avx2_prompt_bos,
        &scalar_generated_ids,
        &avx2_generated_ids,
        reference_first_token,
        reference_has_generated_ids,
        reference_has_logits,
        reference_text_matches_scalar && reference_text_matches_avx2,
    );

    json!({
        "case_id": case_id,
        "question": reference_case["question"].clone(),
        "reference": {
            "prompt": reference_case["reference_prompt"].clone(),
            "prompt_token_ids": reference_prompt_ids,
            "generated_text": reference_case["reference_generated_text"].clone(),
            "first_generated_token_id": reference_case["first_generated_token_id"].clone(),
            "derived_first_generated_token_id": reference_first_token,
            "decoded_first_token": reference_case["decoded_first_token"].clone(),
            "generated_token_ids": reference_generated_ids,
            "generated_token_ids_available": reference_has_generated_ids,
            "logits_available": reference_has_logits,
            "first_token_top_k_logits": reference_topk_logits.cloned().unwrap_or(Value::Null),
            "missing_reference_fields": reference_case["missing_reference_fields"].clone(),
        },
        "scalar": local_case_summary(scalar_case),
        "avx2": local_case_summary(avx2_case),
        "comparisons": {
            "reference_vs_scalar_prompt_exact_match": scalar_prompt_exact,
            "reference_vs_scalar_prompt_local_bos_prefix_match": scalar_prompt_bos,
            "reference_vs_avx2_prompt_exact_match": avx2_prompt_exact,
            "reference_vs_avx2_prompt_local_bos_prefix_match": avx2_prompt_bos,
            "scalar_avx2_prompt_match": scalar_prompt_ids == avx2_prompt_ids,
            "scalar_avx2_generated_match": scalar_generated_ids == avx2_generated_ids,
            "reference_scalar_generated_text_trimmed_match": reference_text_matches_scalar,
            "reference_avx2_generated_text_trimmed_match": reference_text_matches_avx2,
        },
        "classification": classification,
    })
}

#[allow(clippy::too_many_arguments)]
fn classify_case_boundary(
    case_id: &str,
    scalar_present: bool,
    avx2_present: bool,
    scalar_prompt_match_or_bos_prefix: bool,
    avx2_prompt_match_or_bos_prefix: bool,
    scalar_generated_ids: &[u64],
    avx2_generated_ids: &[u64],
    reference_first_token: Option<u64>,
    reference_has_generated_ids: bool,
    reference_has_logits: bool,
    generated_text_matches: bool,
) -> Value {
    if !scalar_present || !avx2_present {
        return classification(
            case_id,
            "receipt_contract",
            "missing_local_cpu_case",
            "scalar_or_avx2_answer_corpus_missing_case",
        );
    }
    if scalar_generated_ids != avx2_generated_ids {
        return classification(
            case_id,
            "local_scalar_avx2",
            "local_cpu_kernel_divergence",
            "scalar_and_avx2_generated_token_ids_differ_before_external_reference_classification",
        );
    }
    if !scalar_prompt_match_or_bos_prefix || !avx2_prompt_match_or_bos_prefix {
        return classification(
            case_id,
            "prompt_token_ids",
            "prompt_token_policy_mismatch",
            "external_reference_prompt_ids_do_not_match_local_ids_or_single_local_bos_prefix",
        );
    }
    if let Some(reference_first_token) = reference_first_token {
        let scalar_first = scalar_generated_ids.first().copied();
        if scalar_first != Some(reference_first_token) {
            return classification(
                case_id,
                "generated_token",
                if reference_has_logits {
                    "logits_or_shared_transformer_math"
                } else {
                    "generated_token_mismatch_reference_logits_unavailable"
                },
                "reference_first_generated_token_id_differs_from_local_cpu_first_token",
            );
        }
        return classification(
            case_id,
            "none",
            "first_generated_token_matches",
            "reference_and_local_first_generated_token_ids_match",
        );
    }
    if generated_text_matches && !reference_has_generated_ids && !reference_has_logits {
        return classification(
            case_id,
            "inconclusive",
            "reference_generated_token_ids_and_logits_unavailable",
            "external_reference_text_matches_after_trimming_but_generated_token_ids_and_logits_are_unavailable",
        );
    }
    if !generated_text_matches && !reference_has_generated_ids {
        return classification(
            case_id,
            "inconclusive",
            "reference_text_mismatch_generated_token_ids_unavailable",
            "external_reference_text_differs_but_reference_generated_token_ids_are_unavailable",
        );
    }
    classification(
        case_id,
        "unknown",
        "insufficient_evidence",
        "case_did_not_match_a_specific_divergence_rule",
    )
}

fn classification(
    case_id: &str,
    first_divergence_stage: &'static str,
    classification: &'static str,
    evidence_boundary: &'static str,
) -> Value {
    json!({
        "case_id": case_id,
        "first_divergence_stage": first_divergence_stage,
        "classification": classification,
        "evidence_boundary": evidence_boundary,
    })
}

fn find_case<'a>(artifact: &'a Value, case_id: &str) -> Option<&'a Value> {
    artifact["cases"].as_array()?.iter().find(|case| {
        case["id"].as_str() == Some(case_id) || case["case_id"].as_str() == Some(case_id)
    })
}

fn local_case_summary(case: Option<&Value>) -> Value {
    let Some(case) = case else {
        return Value::Null;
    };
    json!({
        "answer": case["answer"].clone(),
        "prompt_token_ids": local_prompt_ids(case).unwrap_or_default(),
        "generated_token_ids": local_generated_ids(case).unwrap_or_default(),
        "first_generated_token_id": local_generated_ids(case).and_then(|ids| ids.first().copied()),
        "selected_kernel": case["kernel"]["selected_kernel"].clone(),
        "fallback_used": case["backend"]["fallback_used"].clone(),
        "topk_step0": topk_step0(case).cloned().unwrap_or(Value::Null),
    })
}

fn local_prompt_ids(case: &Value) -> Option<Vec<u64>> {
    ids(case.pointer("/token_ids/prompt").unwrap_or(&Value::Null))
        .or_else(|| ids(case.pointer("/tokens/prompt_ids").unwrap_or(&Value::Null)))
}

fn local_generated_ids(case: &Value) -> Option<Vec<u64>> {
    ids(case.pointer("/token_ids/generated").unwrap_or(&Value::Null))
        .or_else(|| ids(case.pointer("/tokens/generated_ids").unwrap_or(&Value::Null)))
}

fn topk_step0(case: &Value) -> Option<&Value> {
    case["logits_dump"].as_array()?.first().and_then(|step| step.get("top_logits"))
}

fn ids(value: &Value) -> Option<Vec<u64>> {
    value.as_array()?.iter().map(Value::as_u64).collect()
}

fn reference_first_token_topk(case: &Value) -> Option<&Value> {
    case.get("first_token_top_k_logits").or_else(|| case.get("first_token_topk_logits"))
}

fn has_local_bos_prefix(reference: &[u64], local: &[u64], bos_id: u64) -> bool {
    local.len() == reference.len() + 1
        && local.first().copied() == Some(bos_id)
        && local.get(1..) == Some(reference)
}

fn trimmed_eq(left: &str, right: &str) -> bool {
    left.trim() == right.trim()
}

fn count_bool(cases: &[Value], path: &[&str]) -> usize {
    cases.iter().filter(|case| bool_path(case, path) == Some(true)).count()
}

fn count_scalar_avx2_text_matches(cases: &[Value]) -> usize {
    cases
        .iter()
        .filter(|case| {
            let comparisons = &case["comparisons"];
            comparisons["reference_scalar_generated_text_trimmed_match"].as_bool() == Some(true)
                && comparisons["reference_avx2_generated_text_trimmed_match"].as_bool()
                    == Some(true)
        })
        .count()
}

fn count_classification(classifications: &[&Value], stage: &str) -> usize {
    classifications
        .iter()
        .filter(|classification| classification["first_divergence_stage"].as_str() == Some(stage))
        .count()
}

fn bool_path(value: &Value, path: &[&str]) -> Option<bool> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn external_case(case_id: &str, prompt_ids: &[u64], text: &str) -> Value {
        json!({
            "case_id": case_id,
            "question": "What is 2+2?",
            "reference_prompt": "User: What is 2+2?<|eot_id|>Assistant:",
            "prompt_token_ids": prompt_ids,
            "reference_generated_text": text,
            "first_generated_token_id": null,
            "decoded_first_token": null,
            "generated_token_ids_available": false,
            "logits_available": false,
            "missing_reference_fields": [
                "first_generated_token_id",
                "generated_token_ids",
                "first_token_top_k_logits"
            ]
        })
    }

    fn corpus(
        case_id: &str,
        prompt_ids: &[u64],
        generated_ids: &[u64],
        text: &str,
        kernel: &str,
    ) -> Value {
        json!({
            "artifact_kind": "bitnet_cpu_answer_corpus",
            "cases": [{
                "id": case_id,
                "answer": text,
                "token_ids": {
                    "prompt": prompt_ids,
                    "generated": generated_ids
                },
                "backend": {
                    "fallback_used": false
                },
                "kernel": {
                    "selected_kernel": kernel
                },
                "logits_dump": [{
                    "step": 0,
                    "chosen_id": generated_ids.first().copied().unwrap_or_default(),
                    "top_logits": [
                        {"token_id": generated_ids.first().copied().unwrap_or_default(), "logit": 10.0}
                    ]
                }]
            }]
        })
    }

    fn external(prompt_ids: &[u64], text: &str) -> Value {
        json!({
            "schema_version": "1.0.0",
            "artifact_kind": "bitnet_external_first_token_reference_capture",
            "machine_id": "intel-258v",
            "reference": {
                "runner": "Microsoft BitNet.cpp / llama-cli",
                "command_shape": "llama-cli -p ...",
                "generated_token_ids_available": false,
                "logits_available": false,
                "missing_logits_status": "reference report does not expose logits or top-k"
            },
            "model": {
                "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
            },
            "tokenizer": {
                "bos_token_id": 128000
            },
            "summary": {
                "reference_generated_token_ids_available": false,
                "reference_logits_available": false,
                "next_required_evidence": "instrument reference runner"
            },
            "cases": [
                external_case("math", prompt_ids, text)
            ]
        })
    }

    fn prompt_audit() -> Value {
        json!({
            "artifact_kind": "bitnet_prompt_token_authority_audit",
            "classification": {
                "first_divergence_stage": "unknown",
                "first_mismatch_index": null,
                "notes": ["current_default_and_metadata_authority_prompt_tokens_match"]
            }
        })
    }

    fn parity() -> Value {
        json!({
            "artifact_kind": "bitnet_cpu_answer_parity",
            "summary": {
                "failed": 0,
                "first_divergence": null
            }
        })
    }

    fn inputs() -> FirstTokenInputs<'static> {
        FirstTokenInputs {
            external_reference: Path::new("external.json"),
            prompt_audit: Path::new("audit.json"),
            scalar_answer_corpus: Path::new("scalar.json"),
            avx2_answer_corpus: Path::new("avx2.json"),
            answer_parity: Path::new("parity.json"),
        }
    }

    #[test]
    fn classifies_text_match_without_reference_ids_as_inconclusive() {
        let reference = external(&[1502, 25], "4");
        let scalar = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-scalar-reference");
        let avx2 = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-avx2-reference");

        let receipt = build_first_token_divergence_receipt(
            &inputs(),
            &reference,
            &prompt_audit(),
            &scalar,
            &avx2,
            &parity(),
        );

        assert_eq!(receipt["validation"]["passed"], true);
        assert_eq!(
            receipt["summary"]["first_divergence"]["first_divergence_stage"],
            "inconclusive"
        );
        assert_eq!(
            receipt["summary"]["first_divergence"]["classification"],
            "reference_generated_token_ids_and_logits_unavailable"
        );
        assert_eq!(
            receipt["cases"][0]["comparisons"]["reference_vs_scalar_prompt_local_bos_prefix_match"],
            true
        );
        assert_eq!(receipt["summary"]["generated_text_trimmed_scalar_matches"], 1);
        assert_eq!(receipt["summary"]["generated_text_trimmed_avx2_matches"], 1);
        assert_eq!(receipt["summary"]["generated_text_trimmed_scalar_avx2_matches"], 1);
    }

    #[test]
    fn classifies_unexplained_prompt_id_mismatch() {
        let reference = external(&[1502, 25], "4");
        let scalar = corpus("math", &[128000, 777, 25], &[220, 19], " 4", "i2_s-scalar-reference");
        let avx2 = corpus("math", &[128000, 777, 25], &[220, 19], " 4", "i2_s-avx2-reference");

        let receipt = build_first_token_divergence_receipt(
            &inputs(),
            &reference,
            &prompt_audit(),
            &scalar,
            &avx2,
            &parity(),
        );

        assert_eq!(
            receipt["summary"]["first_divergence"]["first_divergence_stage"],
            "prompt_token_ids"
        );
        assert_eq!(
            receipt["summary"]["first_divergence"]["classification"],
            "prompt_token_policy_mismatch"
        );
    }

    #[test]
    fn classifies_scalar_avx2_generated_divergence_before_external_boundary() {
        let reference = external(&[1502, 25], "4");
        let scalar = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-scalar-reference");
        let avx2 = corpus("math", &[128000, 1502, 25], &[19], "4", "i2_s-avx2-reference");

        let receipt = build_first_token_divergence_receipt(
            &inputs(),
            &reference,
            &prompt_audit(),
            &scalar,
            &avx2,
            &parity(),
        );

        assert_eq!(
            receipt["summary"]["first_divergence"]["first_divergence_stage"],
            "local_scalar_avx2"
        );
        assert_eq!(
            receipt["summary"]["first_divergence"]["classification"],
            "local_cpu_kernel_divergence"
        );
        assert_eq!(receipt["summary"]["generated_text_trimmed_scalar_matches"], 1);
        assert_eq!(receipt["summary"]["generated_text_trimmed_avx2_matches"], 1);
        assert_eq!(receipt["summary"]["generated_text_trimmed_scalar_avx2_matches"], 1);
    }

    #[test]
    fn derives_reference_first_token_from_direct_generated_ids() {
        let mut reference = external(&[1502, 25], "4");
        reference["cases"][0]["generated_token_ids"] = json!([220, 19]);
        reference["cases"][0]["generated_token_ids_available"] = json!(false);
        reference["summary"]["reference_generated_token_ids_available"] = json!(false);
        let scalar = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-scalar-reference");
        let avx2 = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-avx2-reference");

        let receipt = build_first_token_divergence_receipt(
            &inputs(),
            &reference,
            &prompt_audit(),
            &scalar,
            &avx2,
            &parity(),
        );

        assert_eq!(receipt["cases"][0]["reference"]["generated_token_ids_available"], true);
        assert_eq!(receipt["cases"][0]["reference"]["derived_first_generated_token_id"], 220);
        assert_eq!(receipt["summary"]["reference_generated_token_ids_available"], true);
        assert_eq!(receipt["summary"]["first_divergence"]["first_divergence_stage"], "none");
        assert_eq!(
            receipt["summary"]["first_divergence"]["classification"],
            "no_divergence_at_first_generated_token"
        );
    }

    #[test]
    fn direct_reference_generated_ids_classify_generated_token_mismatch() {
        let mut reference = external(&[1502, 25], "9");
        reference["cases"][0]["generated_token_ids"] = json!([24]);
        let scalar = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-scalar-reference");
        let avx2 = corpus("math", &[128000, 1502, 25], &[220, 19], " 4", "i2_s-avx2-reference");

        let receipt = build_first_token_divergence_receipt(
            &inputs(),
            &reference,
            &prompt_audit(),
            &scalar,
            &avx2,
            &parity(),
        );

        assert_eq!(
            receipt["summary"]["first_divergence"]["first_divergence_stage"],
            "generated_token"
        );
        assert_eq!(
            receipt["summary"]["first_divergence"]["classification"],
            "generated_token_mismatch_reference_logits_unavailable"
        );
        assert_eq!(receipt["summary"]["reference_generated_token_ids_available"], true);
    }

    #[test]
    fn accepts_direct_reference_artifact_kind() {
        let mut reference = external(&[128000, 1502, 25], "4");
        reference["artifact_kind"] = json!("bitnet_external_reference_direct_token_boundary");
        reference["reference"]["runner"] = json!("Microsoft BitNet.cpp / llama-server");
        reference["reference"]["generated_token_ids_available"] = json!(true);
        reference["reference"]["logits_available"] = json!(true);
        reference["cases"][0]["prompt_token_ids"] = json!([128000, 1502, 25]);
        reference["cases"][0]["generated_token_ids"] = json!([19, 128009]);
        reference["cases"][0]["first_generated_token_id"] = json!(19);
        reference["cases"][0]["first_token_top_k_logits"] = json!([
            {"token_id": 19, "token_text": "4", "logit": 20.0, "probability": 0.99}
        ]);
        let scalar =
            corpus("math", &[128000, 1502, 25], &[19, 128009], "4", "i2_s-scalar-reference");
        let avx2 = corpus("math", &[128000, 1502, 25], &[19, 128009], "4", "i2_s-avx2-reference");

        let receipt = build_first_token_divergence_receipt(
            &inputs(),
            &reference,
            &prompt_audit(),
            &scalar,
            &avx2,
            &parity(),
        );

        assert_eq!(receipt["validation"]["passed"], true);
        assert_eq!(receipt["summary"]["reference_generated_token_ids_available"], true);
        assert_eq!(receipt["summary"]["reference_logits_available"], true);
        assert_eq!(
            receipt["summary"]["first_divergence"]["classification"],
            "no_divergence_at_first_generated_token"
        );
    }
}
