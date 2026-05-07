//! Scalar-vs-AVX2 answer-corpus parity comparator.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

/// Compare scalar and AVX2 strict CPU answer-corpus receipts.
#[derive(Args, Debug)]
pub struct AnswerParityCommand {
    /// Scalar strict CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub scalar: PathBuf,

    /// AVX2 strict CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub avx2: PathBuf,

    /// Output scalar-vs-AVX2 parity receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/cpu-answer-parity.json"
    )]
    pub json_out: PathBuf,
}

impl AnswerParityCommand {
    /// Execute the offline scalar-vs-AVX2 answer parity comparison.
    pub async fn execute(&self) -> Result<()> {
        let scalar = read_json(&self.scalar)?;
        let avx2 = read_json(&self.avx2)?;
        let receipt = build_answer_parity_receipt(&self.scalar, &scalar, &self.avx2, &avx2);
        let failed = receipt["summary"]["failed"].as_u64().unwrap_or(1);

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("answer parity receipt written to {}", self.json_out.display());

        if failed > 0 {
            anyhow::bail!(
                "scalar-vs-AVX2 answer parity failed for {failed} case(s); receipt written to {}",
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

fn build_answer_parity_receipt(
    scalar_path: &Path,
    scalar: &Value,
    avx2_path: &Path,
    avx2: &Value,
) -> Value {
    let mut shared_failures = Vec::new();
    compare_top_level_contract(scalar, avx2, &mut shared_failures);

    let scalar_cases = cases_by_id(scalar);
    let avx2_cases = cases_by_id(avx2);
    let case_ids = scalar_cases.keys().chain(avx2_cases.keys()).cloned().collect::<BTreeSet<_>>();

    let mut first_divergence = None;
    let cases = case_ids
        .iter()
        .map(|id| {
            compare_case(
                id,
                scalar_cases.get(id).copied(),
                avx2_cases.get(id).copied(),
                &mut first_divergence,
            )
        })
        .collect::<Vec<_>>();

    let passed = cases.iter().filter(|case| case["passed"] == true).count();
    let failed = cases.len().saturating_sub(passed) + usize::from(!shared_failures.is_empty());

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_cpu_answer_parity",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "full_decode_parity_compared",
        "claim": "scalar_avx2_full_decode_answer_parity",
        "speedup_claim": false,
        "inputs": {
            "scalar_receipt_path": scalar_path.display().to_string(),
            "avx2_receipt_path": avx2_path.display().to_string(),
        },
        "shared_contract": {
            "same_real_gguf": shared_failures.iter().all(|failure| *failure != "model_contract"),
            "same_tokenizer": shared_failures.iter().all(|failure| *failure != "tokenizer_contract"),
            "same_prompt_template": shared_failures.iter().all(|failure| *failure != "prompt_template"),
            "same_greedy_settings": shared_failures.iter().all(|failure| *failure != "generation_contract"),
            "failed_rules": shared_failures,
        },
        "summary": {
            "total": cases.len(),
            "passed": passed,
            "failed": failed,
            "first_divergence": first_divergence,
        },
        "cases": cases,
        "may_claim": [
            "Scalar versus AVX2 full-decode answer parity can be audited for the compared receipts.",
            "First divergence evidence can separate AVX2 kernel issues from shared prompt, tokenizer, logits, sampler, or text decoding issues."
        ],
        "must_not_claim": [
            "General chat quality is proven.",
            "Sustained CPU throughput is proven.",
            "Server inference is complete.",
            "GPU or NPU execution is involved."
        ],
    })
}

fn compare_top_level_contract(scalar: &Value, avx2: &Value, failures: &mut Vec<&'static str>) {
    if scalar["artifact_kind"] != "bitnet_cpu_answer_corpus"
        || avx2["artifact_kind"] != "bitnet_cpu_answer_corpus"
        || scalar["model"]["loader_mode"] != "real_gguf"
        || avx2["model"]["loader_mode"] != "real_gguf"
        || scalar["model"]["repo"] != avx2["model"]["repo"]
        || scalar["model"]["file"] != avx2["model"]["file"]
        || scalar["model"]["path"] != avx2["model"]["path"]
    {
        failures.push("model_contract");
    }

    if scalar["model"]["tokenizer_path"] != avx2["model"]["tokenizer_path"] {
        failures.push("tokenizer_contract");
    }

    if scalar["prompt_template"] != avx2["prompt_template"] {
        failures.push("prompt_template");
    }

    for field in [
        "/generation/mode",
        "/generation/temperature",
        "/generation/deterministic",
        "/generation/strict_loader",
        "/generation/default_max_new_tokens",
        "/generation/logits_dump_steps",
        "/generation/logits_topk",
    ] {
        if scalar.pointer(field) != avx2.pointer(field) {
            failures.push("generation_contract");
            break;
        }
    }

    if !strict_cpu_backend(scalar) || !strict_cpu_backend(avx2) {
        failures.push("strict_cpu_backend");
    }
}

fn strict_cpu_backend(receipt: &Value) -> bool {
    receipt["backend"]["requested_backend"] == "cpu"
        && receipt["backend"]["selected_backend"] == "cpu"
        && receipt["backend"]["runtime_api"] == "cpu"
        && receipt["backend"]["fallback_used"] == false
}

fn cases_by_id(receipt: &Value) -> BTreeMap<String, &Value> {
    receipt["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|case| case["id"].as_str().map(|id| (id.to_string(), case)))
        .collect()
}

fn compare_case(
    id: &str,
    scalar: Option<&Value>,
    avx2: Option<&Value>,
    first_divergence: &mut Option<Value>,
) -> Value {
    let Some(scalar) = scalar else {
        set_first(first_divergence, id, "case_missing_in_scalar", None, Value::Null, Value::Null);
        return failed_case(id, &["case_missing_in_scalar"]);
    };
    let Some(avx2) = avx2 else {
        set_first(first_divergence, id, "case_missing_in_avx2", None, Value::Null, Value::Null);
        return failed_case(id, &["case_missing_in_avx2"]);
    };

    let mut failures = Vec::new();
    check_case_contract(id, "scalar", scalar, &mut failures, first_divergence);
    check_case_contract(id, "avx2", avx2, &mut failures, first_divergence);
    check_equal(
        id,
        "question",
        None,
        &scalar["question"],
        &avx2["question"],
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "prompt_template",
        None,
        &scalar["prompt_template"],
        &avx2["prompt_template"],
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "tokenizer_source",
        None,
        &scalar["tokenizer"]["source"],
        &avx2["tokenizer"]["source"],
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "tokenizer_strict",
        None,
        &scalar["tokenizer"]["strict"],
        &avx2["tokenizer"]["strict"],
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "prompt_token_ids",
        None,
        &scalar["token_ids"]["prompt"],
        &avx2["token_ids"]["prompt"],
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "generated_token_ids",
        None,
        &scalar["token_ids"]["generated"],
        &avx2["token_ids"]["generated"],
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "decoded_text",
        None,
        &scalar["answer"],
        &avx2["answer"],
        &mut failures,
        first_divergence,
    );
    check_kernel_lane(id, "scalar", scalar, &mut failures, first_divergence);
    check_kernel_lane(id, "avx2", avx2, &mut failures, first_divergence);
    compare_logits_dump(
        id,
        &scalar["logits_dump"],
        &avx2["logits_dump"],
        &mut failures,
        first_divergence,
    );

    json!({
        "id": id,
        "passed": failures.is_empty(),
        "failed_rules": failures,
        "scalar": case_summary(scalar),
        "avx2": case_summary(avx2),
    })
}

fn failed_case(id: &str, failures: &[&str]) -> Value {
    json!({
        "id": id,
        "passed": false,
        "failed_rules": failures,
    })
}

fn check_case_contract(
    id: &str,
    lane: &'static str,
    case: &Value,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let status_rule = if lane == "scalar" { "scalar_case_passed" } else { "avx2_case_passed" };
    if case["status"] != "passed" || case["quality"]["passed"] != true {
        failures.push(status_rule);
        set_first(first_divergence, id, status_rule, None, case["status"].clone(), json!("passed"));
    }

    let backend_rule =
        if lane == "scalar" { "scalar_strict_cpu_backend" } else { "avx2_strict_cpu_backend" };
    if case["backend"]["requested_backend"] != "cpu"
        || !matches!(case["backend"]["selected_backend"].as_str(), Some("cpu" | "cpu-rust"))
        || case["backend"]["runtime_api"] != "cpu"
        || case["backend"]["fallback_used"] != false
    {
        failures.push(backend_rule);
        set_first(first_divergence, id, backend_rule, None, case["backend"].clone(), json!("cpu"));
    }
}

fn check_equal(
    id: &str,
    rule: &'static str,
    step: Option<usize>,
    left: &Value,
    right: &Value,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    if left != right {
        failures.push(rule);
        set_first(first_divergence, id, rule, step, left.clone(), right.clone());
    }
}

fn check_kernel_lane(
    id: &str,
    lane: &'static str,
    case: &Value,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let selected = case["kernel"]["selected_kernel"].as_str().unwrap_or_default();
    if !selected.to_ascii_lowercase().contains(lane) {
        let rule = if lane == "scalar" { "scalar_kernel_identity" } else { "avx2_kernel_identity" };
        failures.push(rule);
        set_first(first_divergence, id, rule, None, json!(selected), json!(lane));
    }
}

fn compare_logits_dump(
    id: &str,
    scalar: &Value,
    avx2: &Value,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let Some(scalar_steps) = scalar.as_array() else {
        failures.push("scalar_logits_dump_recorded");
        set_first(
            first_divergence,
            id,
            "scalar_logits_dump_recorded",
            None,
            scalar.clone(),
            Value::Null,
        );
        return;
    };
    let Some(avx2_steps) = avx2.as_array() else {
        failures.push("avx2_logits_dump_recorded");
        set_first(
            first_divergence,
            id,
            "avx2_logits_dump_recorded",
            None,
            avx2.clone(),
            Value::Null,
        );
        return;
    };
    if scalar_steps.is_empty() {
        failures.push("scalar_logits_dump_recorded");
        set_first(
            first_divergence,
            id,
            "scalar_logits_dump_recorded",
            None,
            scalar.clone(),
            Value::Null,
        );
        return;
    }
    if avx2_steps.is_empty() {
        failures.push("avx2_logits_dump_recorded");
        set_first(
            first_divergence,
            id,
            "avx2_logits_dump_recorded",
            None,
            avx2.clone(),
            Value::Null,
        );
        return;
    }
    check_equal(
        id,
        "logits_step_count",
        None,
        &json!(scalar_steps.len()),
        &json!(avx2_steps.len()),
        failures,
        first_divergence,
    );
    for (step, (scalar_step, avx2_step)) in scalar_steps.iter().zip(avx2_steps).enumerate() {
        check_equal(
            id,
            "logits_topk",
            Some(step),
            scalar_step,
            avx2_step,
            failures,
            first_divergence,
        );
    }
}

fn case_summary(case: &Value) -> Value {
    json!({
        "status": case["status"],
        "selected_kernel": case["kernel"]["selected_kernel"],
        "prompt_token_ids": case["token_ids"]["prompt"],
        "generated_token_ids": case["token_ids"]["generated"],
        "answer": case["answer"],
        "logits_steps": case["logits_dump"].as_array().map(Vec::len),
    })
}

fn set_first(
    first_divergence: &mut Option<Value>,
    id: &str,
    kind: &'static str,
    step: Option<usize>,
    scalar: Value,
    avx2: Value,
) {
    if first_divergence.is_none() {
        *first_divergence = Some(json!({
            "case_id": id,
            "kind": kind,
            "step": step,
            "scalar": scalar,
            "avx2": avx2,
            "scope": divergence_scope(kind),
        }));
    }
}

fn divergence_scope(kind: &str) -> &'static str {
    match kind {
        "prompt_token_ids" | "question" | "prompt_template" => "prompt_or_tokenizer",
        "generated_token_ids" => "decode_or_sampler",
        "decoded_text" => "text_decode",
        kind if kind.contains("logits") => "logits_or_kernel",
        kind if kind.contains("kernel_identity") => "kernel_selection",
        _ => "receipt_contract",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(kernel: &str, generated: &[u64], answer: &str, logits: Value) -> Value {
        json!({
            "artifact_kind": "bitnet_cpu_answer_corpus",
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf",
                "path": "models/ggml-model-i2_s.gguf",
                "loader_mode": "real_gguf",
                "tokenizer_path": "models/tokenizer.json"
            },
            "backend": {
                "requested_backend": "cpu",
                "selected_backend": "cpu",
                "runtime_api": "cpu",
                "fallback_used": false
            },
            "prompt_template": { "family": "llama3-chat" },
            "generation": {
                "mode": "greedy",
                "temperature": 0.0,
                "deterministic": true,
                "strict_loader": true,
                "default_max_new_tokens": 1
            },
            "cases": [{
                "id": "math",
                "question": "Answer with a single digit: 2+2=",
                "status": "passed",
                "answer": answer,
                "quality": { "passed": true },
                "backend": {
                    "requested_backend": "cpu",
                    "selected_backend": "cpu-rust",
                    "runtime_api": "cpu",
                    "fallback_used": false
                },
                "tokenizer": {
                    "source": "explicit",
                    "strict": true
                },
                "token_ids": {
                    "prompt": [1, 2, 3],
                    "generated": generated
                },
                "logits_dump": logits,
                "prompt_template": "llama3-chat",
                "kernel": {
                    "selected_kernel": kernel,
                    "family": "i2_s"
                }
            }]
        })
    }

    fn logits() -> Value {
        json!([{
            "step": 0,
            "chosen_id": 4,
            "top_logits": [
                { "token_id": 4, "logit": 10.0 },
                { "token_id": 5, "logit": 1.0 }
            ]
        }])
    }

    #[test]
    fn parity_receipt_passes_matching_scalar_and_avx2_runs() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let avx2 = receipt("i2_s-avx2-reference", &[4], "4", logits());

        let report = build_answer_parity_receipt(
            Path::new("scalar.json"),
            &scalar,
            Path::new("avx2.json"),
            &avx2,
        );

        assert_eq!(report["artifact_kind"], "bitnet_cpu_answer_parity");
        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["cases"][0]["passed"], true);
        assert!(report["summary"]["first_divergence"].is_null());
    }

    #[test]
    fn parity_receipt_records_first_generated_token_divergence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let avx2 = receipt("i2_s-avx2-reference", &[5], "5", logits());

        let report = build_answer_parity_receipt(
            Path::new("scalar.json"),
            &scalar,
            Path::new("avx2.json"),
            &avx2,
        );

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["cases"][0]["passed"], false);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(report["summary"]["first_divergence"]["scope"], "decode_or_sampler");
    }

    #[test]
    fn parity_receipt_requires_logit_evidence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", Value::Null);
        let avx2 = receipt("i2_s-avx2-reference", &[4], "4", logits());

        let report = build_answer_parity_receipt(
            Path::new("scalar.json"),
            &scalar,
            Path::new("avx2.json"),
            &avx2,
        );

        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "scalar_logits_dump_recorded"));
    }
}
