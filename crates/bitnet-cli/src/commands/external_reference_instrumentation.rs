//! External reference instrumentation boundary classifier for Lunar Lake CPU proof.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Classify whether the external BitNet reference can expose token/logit evidence.
#[derive(Args, Debug)]
pub struct ExternalReferenceInstrumentationCommand {
    /// External first-token reference capture artifact.
    #[arg(long, value_name = "PATH")]
    pub external_reference: PathBuf,

    /// Optional captured `llama-cli --help` output for capability evidence.
    #[arg(long, value_name = "PATH")]
    pub runner_help: Option<PathBuf>,

    /// Output instrumentation-boundary receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/external-reference-instrumentation.json"
    )]
    pub json_out: PathBuf,
}

impl ExternalReferenceInstrumentationCommand {
    /// Execute offline instrumentation-boundary classification.
    pub async fn execute(&self) -> Result<()> {
        let external_reference = read_json(&self.external_reference)?;
        let runner_help = self.runner_help.as_deref().map(read_text).transpose()?;
        let receipt = build_external_reference_instrumentation_receipt(
            &self.external_reference,
            self.runner_help.as_deref(),
            runner_help.as_deref(),
            &external_reference,
        );

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!(
            "external reference instrumentation receipt written to {}",
            self.json_out.display()
        );

        if receipt["validation"]["passed"].as_bool() != Some(true) {
            anyhow::bail!(
                "external reference instrumentation validation failed; receipt written to {}",
                self.json_out.display()
            );
        }
        Ok(())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn read_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

fn build_external_reference_instrumentation_receipt(
    external_reference_path: &Path,
    runner_help_path: Option<&Path>,
    runner_help: Option<&str>,
    external_reference: &Value,
) -> Value {
    let validation_failures = validate_external_reference(external_reference);
    let cases = external_reference["cases"]
        .as_array()
        .map(|items| items.iter().map(case_boundary).collect::<Vec<_>>())
        .unwrap_or_default();
    let summary = summarize_cases(&cases);
    let help_caps = runner_help.map(analyze_runner_help);
    let classification = classify_boundary(&summary, help_caps.as_ref());

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_external_reference_instrumentation_probe",
        "machine_id": external_reference["machine_id"].clone(),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "external_reference_instrumentation_classified",
        "claim": "cpu258v_external_reference_evidence_boundary",
        "inputs": {
            "external_reference": external_reference_path.display().to_string(),
            "runner_help": runner_help_path.map(|path| path.display().to_string()),
        },
        "validation": {
            "passed": validation_failures.is_empty(),
            "failed_rules": validation_failures,
        },
        "reference": {
            "runner": external_reference["reference"]["runner"].clone(),
            "command_shape": external_reference["reference"]["command_shape"].clone(),
            "bitnet_cpp_commit": external_reference["reference"]["bitnet_cpp_commit"].clone(),
            "llama_cpp_submodule_commit": external_reference["reference"]["llama_cpp_submodule_commit"].clone(),
            "generated_token_ids_available": external_reference["reference"]["generated_token_ids_available"].clone(),
            "logits_available": external_reference["reference"]["logits_available"].clone(),
            "missing_logits_status": external_reference["reference"]["missing_logits_status"].clone(),
        },
        "model": external_reference["model"].clone(),
        "tokenizer": external_reference["tokenizer"].clone(),
        "runner_capabilities": help_caps.unwrap_or_else(|| json!({
            "runner_help_provided": false,
            "advertised_prompt_token_dump": false,
            "advertised_generated_token_dump": false,
            "advertised_logits_dump": false,
            "notes": [
                "No runner help text was supplied, so direct runner capability support is unverified."
            ]
        })),
        "summary": {
            "cases_total": summary.cases_total,
            "cases_with_generated_text": summary.cases_with_generated_text,
            "cases_with_first_generated_token_id": summary.cases_with_first_generated_token_id,
            "cases_with_generated_token_ids": summary.cases_with_generated_token_ids,
            "cases_with_decoded_first_token": summary.cases_with_decoded_first_token,
            "cases_with_first_token_topk_logits": summary.cases_with_first_token_topk_logits,
            "generated_token_ids_available": summary.generated_token_ids_available,
            "first_token_logits_available": summary.first_token_logits_available,
            "classification": classification["classification"].clone(),
            "next_required_evidence": classification["next_required_evidence"].clone(),
        },
        "classification": classification,
        "cases": cases,
        "fallback_used": false,
        "claim_boundary": {
            "may_claim": [
                "The external reference evidence boundary was classified for generated-token ID and first-token logits/top-k availability.",
                "Missing reference token/logit fields are explicit blockers rather than inferred parity.",
                "Runner help text can be recorded as capability evidence when supplied."
            ],
            "must_not_claim": [
                "generated-token-ID parity when direct reference generated IDs are unavailable",
                "first-token logits parity when reference logits are unavailable",
                "BitNet answer quality is newly proven",
                "CPU speed or sustained throughput",
                "Arc 140V execution or acceleration",
                "Intel NPU execution or acceleration",
                "QK256 decode correctness"
            ]
        }
    })
}

fn validate_external_reference(external_reference: &Value) -> Vec<&'static str> {
    let mut failures = Vec::new();
    if external_reference["artifact_kind"].as_str()
        != Some("bitnet_external_first_token_reference_capture")
    {
        failures.push("external_reference_artifact_kind");
    }
    if external_reference["cases"].as_array().is_none_or(|cases| cases.is_empty()) {
        failures.push("external_reference_cases");
    }
    if external_reference["reference"]["runner"].as_str().unwrap_or_default().is_empty() {
        failures.push("reference_runner");
    }
    if external_reference["reference"]["command_shape"].as_str().unwrap_or_default().is_empty() {
        failures.push("reference_command_shape");
    }
    failures
}

#[derive(Default)]
struct CaseSummary {
    cases_total: usize,
    cases_with_generated_text: usize,
    cases_with_first_generated_token_id: usize,
    cases_with_generated_token_ids: usize,
    cases_with_decoded_first_token: usize,
    cases_with_first_token_topk_logits: usize,
    generated_token_ids_available: bool,
    first_token_logits_available: bool,
}

fn summarize_cases(cases: &[Value]) -> CaseSummary {
    let mut summary = CaseSummary { cases_total: cases.len(), ..CaseSummary::default() };
    for case in cases {
        if case["reference_generated_text_available"].as_bool() == Some(true) {
            summary.cases_with_generated_text += 1;
        }
        if case["first_generated_token_id_available"].as_bool() == Some(true) {
            summary.cases_with_first_generated_token_id += 1;
        }
        if case["generated_token_ids_available"].as_bool() == Some(true) {
            summary.cases_with_generated_token_ids += 1;
        }
        if case["decoded_first_token_available"].as_bool() == Some(true) {
            summary.cases_with_decoded_first_token += 1;
        }
        if case["first_token_topk_logits_available"].as_bool() == Some(true) {
            summary.cases_with_first_token_topk_logits += 1;
        }
    }
    summary.generated_token_ids_available =
        summary.cases_total > 0 && summary.cases_with_generated_token_ids == summary.cases_total;
    summary.first_token_logits_available = summary.cases_total > 0
        && summary.cases_with_first_token_topk_logits == summary.cases_total;
    summary
}

fn case_boundary(case: &Value) -> Value {
    let generated_token_ids_available =
        ids(case.get("generated_token_ids")).is_some_and(|ids| !ids.is_empty());
    let first_token_topk_logits_available = case
        .get("first_token_top_k_logits")
        .or_else(|| case.get("first_token_topk_logits"))
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());
    let missing_fields =
        missing_fields(case, generated_token_ids_available, first_token_topk_logits_available);

    json!({
        "case_id": case["case_id"].clone(),
        "reference_generated_text_available": case["reference_generated_text"].as_str().is_some(),
        "first_generated_token_id_available": case["first_generated_token_id"].as_u64().is_some(),
        "decoded_first_token_available": case["decoded_first_token"].as_str().is_some(),
        "generated_token_ids_available": generated_token_ids_available,
        "first_token_topk_logits_available": first_token_topk_logits_available,
        "missing_reference_fields": missing_fields,
    })
}

fn missing_fields(
    case: &Value,
    generated_token_ids_available: bool,
    first_token_topk_logits_available: bool,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if case["first_generated_token_id"].as_u64().is_none() {
        missing.push("first_generated_token_id");
    }
    if case["decoded_first_token"].as_str().is_none() {
        missing.push("decoded_first_token");
    }
    if !generated_token_ids_available {
        missing.push("generated_token_ids");
    }
    if !first_token_topk_logits_available {
        missing.push("first_token_top_k_logits");
    }
    missing
}

fn ids(value: Option<&Value>) -> Option<Vec<u64>> {
    value?.as_array()?.iter().map(Value::as_u64).collect()
}

fn analyze_runner_help(help: &str) -> Value {
    let lower = help.to_ascii_lowercase();
    let advertised_prompt_token_dump =
        lower.contains("verbose-prompt") || lower.contains("dump-prompt");
    let advertised_generated_token_dump = lower.contains("dump-generated")
        || lower.contains("generated-ids")
        || lower.contains("dump-ids");
    let advertised_logits_dump = lower.contains("logits")
        && (lower.contains("dump") || lower.contains("top-k") || lower.contains("topk"));

    json!({
        "runner_help_provided": true,
        "advertised_prompt_token_dump": advertised_prompt_token_dump,
        "advertised_generated_token_dump": advertised_generated_token_dump,
        "advertised_logits_dump": advertised_logits_dump,
        "notes": capability_notes(
            advertised_prompt_token_dump,
            advertised_generated_token_dump,
            advertised_logits_dump,
        ),
    })
}

fn capability_notes(
    prompt_token_dump: bool,
    generated_token_dump: bool,
    logits_dump: bool,
) -> Vec<&'static str> {
    let mut notes = Vec::new();
    if prompt_token_dump {
        notes.push("Runner help advertises a prompt-token dump surface.");
    }
    if generated_token_dump {
        notes.push("Runner help advertises a generated-token dump surface.");
    }
    if logits_dump {
        notes.push("Runner help advertises a logits or top-k dump surface.");
    }
    if notes.is_empty() {
        notes.push("Runner help does not advertise direct generated-token or logits dump support.");
    }
    notes
}

fn classify_boundary(summary: &CaseSummary, help_caps: Option<&Value>) -> Value {
    if summary.generated_token_ids_available && summary.first_token_logits_available {
        return json!({
            "first_divergence_stage": "none",
            "classification": "reference_generated_token_ids_and_logits_available",
            "evidence_boundary": "reference_artifact_contains_direct_generated_token_ids_and_first_token_logits",
            "next_required_evidence": "rerun first-token-divergence with direct generated-token and logits evidence",
        });
    }
    if summary.generated_token_ids_available {
        return json!({
            "first_divergence_stage": "logits",
            "classification": "reference_generated_token_ids_available_logits_unavailable",
            "evidence_boundary": "reference_artifact_contains_generated_token_ids_but_not_first_token_logits",
            "next_required_evidence": "instrument reference runner for first-token logits/top-k before claiming logits parity",
        });
    }

    let help_generated = help_caps
        .and_then(|caps| caps["advertised_generated_token_dump"].as_bool())
        .unwrap_or(false);
    let help_logits =
        help_caps.and_then(|caps| caps["advertised_logits_dump"].as_bool()).unwrap_or(false);
    let classification = if help_generated || help_logits {
        "reference_runner_capability_needs_capture"
    } else {
        "reference_runner_requires_instrumentation"
    };
    json!({
        "first_divergence_stage": "reference_instrumentation",
        "classification": classification,
        "evidence_boundary": "external_reference_generated_text_exists_but_direct_generated_token_ids_or_first_token_logits_are_missing",
        "next_required_evidence": "capture direct reference generated-token IDs and first-token logits/top-k, or patch/script the reference runner to expose them without text re-tokenization",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_case(case_id: &str) -> Value {
        json!({
            "case_id": case_id,
            "reference_generated_text": "4",
            "first_generated_token_id": null,
            "decoded_first_token": null,
            "generated_token_ids_available": false,
            "logits_available": false
        })
    }

    fn external(cases: Vec<Value>) -> Value {
        json!({
            "artifact_kind": "bitnet_external_first_token_reference_capture",
            "machine_id": "intel-258v",
            "reference": {
                "runner": "Microsoft BitNet.cpp / llama-cli",
                "command_shape": "llama-cli -p ...",
                "generated_token_ids_available": false,
                "logits_available": false,
                "missing_logits_status": "not exposed"
            },
            "model": {
                "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
            },
            "tokenizer": {
                "eos_token_id": 128009
            },
            "cases": cases
        })
    }

    #[test]
    fn classifies_current_reference_as_needing_instrumentation() {
        let artifact = external(vec![reference_case("math")]);
        let receipt = build_external_reference_instrumentation_receipt(
            Path::new("external.json"),
            None,
            None,
            &artifact,
        );

        assert_eq!(receipt["validation"]["passed"], true);
        assert_eq!(
            receipt["summary"]["classification"],
            "reference_runner_requires_instrumentation"
        );
        assert_eq!(receipt["summary"]["cases_with_generated_text"], 1);
        assert_eq!(receipt["summary"]["cases_with_generated_token_ids"], 0);
        assert_eq!(receipt["summary"]["cases_with_first_token_topk_logits"], 0);
    }

    #[test]
    fn records_help_capability_without_inferring_evidence() {
        let artifact = external(vec![reference_case("math")]);
        let receipt = build_external_reference_instrumentation_receipt(
            Path::new("external.json"),
            Some(Path::new("help.txt")),
            Some("--dump-generated-ids --dump-logits-topk"),
            &artifact,
        );

        assert_eq!(
            receipt["summary"]["classification"],
            "reference_runner_capability_needs_capture"
        );
        assert_eq!(receipt["runner_capabilities"]["advertised_generated_token_dump"], true);
        assert_eq!(receipt["runner_capabilities"]["advertised_logits_dump"], true);
        assert_eq!(receipt["summary"]["generated_token_ids_available"], false);
    }

    #[test]
    fn ignores_generated_token_ids_flag_without_direct_ids() {
        let mut case = reference_case("math");
        case["generated_token_ids_available"] = json!(true);
        let artifact = external(vec![case]);
        let receipt = build_external_reference_instrumentation_receipt(
            Path::new("external.json"),
            None,
            None,
            &artifact,
        );

        assert_eq!(receipt["summary"]["cases_with_generated_token_ids"], 0);
        assert_eq!(receipt["summary"]["generated_token_ids_available"], false);
        assert_eq!(
            receipt["classification"]["evidence_boundary"],
            "external_reference_generated_text_exists_but_direct_generated_token_ids_or_first_token_logits_are_missing"
        );
        assert_eq!(
            receipt["cases"][0]["missing_reference_fields"],
            json!([
                "first_generated_token_id",
                "decoded_first_token",
                "generated_token_ids",
                "first_token_top_k_logits"
            ])
        );
    }

    #[test]
    fn recognizes_fully_instrumented_reference_artifact() {
        let mut case = reference_case("math");
        case["first_generated_token_id"] = json!(220);
        case["decoded_first_token"] = json!(" ");
        case["generated_token_ids"] = json!([220, 19]);
        case["first_token_top_k_logits"] = json!([
            {"token_id": 220, "logit": 12.0},
            {"token_id": 19, "logit": 9.0}
        ]);
        let artifact = external(vec![case]);
        let receipt = build_external_reference_instrumentation_receipt(
            Path::new("external.json"),
            None,
            None,
            &artifact,
        );

        assert_eq!(
            receipt["summary"]["classification"],
            "reference_generated_token_ids_and_logits_available"
        );
        assert_eq!(receipt["summary"]["generated_token_ids_available"], true);
        assert_eq!(receipt["summary"]["first_token_logits_available"], true);
    }
}
