//! Reference divergence artifact validator for answer-readiness bring-up.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Validate and normalize an external-reference comparison artifact.
#[derive(Args, Debug)]
pub struct ReferenceCompareCommand {
    /// Reference comparison artifact to validate.
    #[arg(long, value_name = "PATH")]
    pub artifact: PathBuf,

    /// Output normalized validation receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/slm-reference-divergence.json"
    )]
    pub json_out: PathBuf,

    /// Fail if bitnet-rs diverges from the reference.
    #[arg(long, default_value_t = false)]
    pub require_match: bool,
}

impl ReferenceCompareCommand {
    /// Execute offline validation.
    pub async fn execute(&self) -> Result<()> {
        let artifact = read_json(&self.artifact)?;
        let receipt = build_reference_divergence_receipt(&self.artifact, &artifact);
        let valid = receipt["validation"]["passed"].as_bool().unwrap_or(false);
        let matched = receipt["comparison"]["passed"].as_bool().unwrap_or(false);

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("reference divergence receipt written to {}", self.json_out.display());

        if !valid {
            anyhow::bail!(
                "reference artifact validation failed; receipt written to {}",
                self.json_out.display()
            );
        }
        if self.require_match && !matched {
            anyhow::bail!(
                "reference artifact diverged; receipt written to {}",
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

fn build_reference_divergence_receipt(path: &Path, artifact: &Value) -> Value {
    let validation_failures = validate_artifact(artifact);
    let first_divergence =
        if validation_failures.is_empty() { first_divergence(artifact) } else { None };
    let passed = validation_failures.is_empty() && first_divergence.is_none();
    let bitnet_artifact = is_bitnet_artifact(artifact);
    let artifact_kind = if bitnet_artifact {
        "bitnet_cpu_reference_divergence_validation"
    } else {
        "slm_reference_divergence_validation"
    };
    let claim = if bitnet_artifact {
        "bitnet_cpu_reference_divergence_diagnostic"
    } else {
        "slm_reference_divergence_diagnostic"
    };

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": artifact_kind,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "external_reference_compared",
        "claim": claim,
        "speedup_claim": false,
        "inputs": {
            "artifact_path": path.display().to_string(),
        },
        "model": {
            "sha256": artifact["model_sha256"].clone(),
            "family": artifact["model_family"].clone(),
        },
        "prompt": {
            "text": artifact["prompt_text"].clone(),
            "bytes": artifact["prompt_bytes"].clone(),
            "template": artifact["prompt_template"].clone(),
            "bos": artifact.get("bos").or_else(|| artifact.get("add_bos")).cloned().unwrap_or(Value::Null),
        },
        "validation": {
            "passed": validation_failures.is_empty(),
            "failed_rules": validation_failures,
        },
        "comparison": {
            "passed": passed,
            "first_divergence": first_divergence,
            "reference": side_summary(&artifact["reference"]),
            "bitnet_rs": side_summary(bitnet_side(artifact)),
        },
        "may_claim": [
            "The artifact is machine-checkable against an external reference run.",
            "First divergence evidence can separate tokenizer, prompt-template, decode, logits, and text-decoding issues.",
            "For BitNet CPU artifacts, strict loader, tokenizer, backend, kernel, and fallback evidence was checked."
        ],
        "must_not_claim": [
            "BitNet-rs can run the external reference engine.",
            "General chat quality is proven.",
            "Sustained CPU throughput is proven.",
            "Server, GPU, OpenVINO, UHD 620, or NPU execution is involved."
        ],
    })
}

fn validate_artifact(artifact: &Value) -> Vec<&'static str> {
    let mut failures = Vec::new();
    let bitnet_artifact = is_bitnet_artifact(artifact);
    if !matches!(
        artifact["artifact_kind"].as_str(),
        Some(
            "backend_reference_compare"
                | "slm_reference_divergence"
                | "bitnet_cpu_reference_compare"
        )
    ) {
        failures.push("artifact_kind");
    }
    if artifact["schema_version"].as_str().is_none() {
        failures.push("schema_version");
    }
    let sha = artifact["model_sha256"].as_str().unwrap_or_default();
    if sha.len() != 64 || !sha.chars().all(|ch| ch.is_ascii_hexdigit()) {
        failures.push("model_sha256");
    }
    if artifact["model_family"].as_str().unwrap_or_default().is_empty() {
        failures.push("model_family");
    }
    if bitnet_artifact && artifact["model_family"].as_str() != Some("bitnet") {
        failures.push("bitnet_model_family");
    }
    if artifact["prompt_text"].as_str().unwrap_or_default().is_empty() {
        failures.push("prompt_text");
    }
    if bitnet_artifact && artifact["prompt_bytes"].as_array().is_none_or(|bytes| bytes.is_empty()) {
        failures.push("prompt_bytes");
    }
    if !artifact["prompt_template"].is_string() && !artifact["prompt_template"].is_object() {
        failures.push("prompt_template");
    }
    if !artifact["bos"].is_boolean() && !artifact["add_bos"].is_boolean() {
        failures.push("bos_policy");
    }
    validate_side("reference", &artifact["reference"], &mut failures);
    validate_side("bitnet_rs", bitnet_side(artifact), &mut failures);
    if bitnet_artifact {
        validate_bitnet_contract(artifact, &mut failures);
    }
    failures
}

fn validate_side(label: &'static str, side: &Value, failures: &mut Vec<&'static str>) {
    if !side.is_object() {
        failures.push(if label == "reference" { "reference_object" } else { "bitnet_rs_object" });
        return;
    }
    if side["backend"].as_str().unwrap_or_default().is_empty() {
        failures.push(if label == "reference" { "reference_backend" } else { "bitnet_rs_backend" });
    }
    if side["kernel"].as_str().unwrap_or_default().is_empty() {
        failures.push(if label == "reference" { "reference_kernel" } else { "bitnet_rs_kernel" });
    }
    if ids(side.get("prompt_ids")).is_none_or(|ids| ids.is_empty()) {
        failures.push(if label == "reference" {
            "reference_prompt_ids"
        } else {
            "bitnet_rs_prompt_ids"
        });
    }
    if ids(side.get("generated_ids")).is_none() {
        failures.push(if label == "reference" {
            "reference_generated_ids"
        } else {
            "bitnet_rs_generated_ids"
        });
    }
    if side["text"].as_str().is_none() {
        failures.push(if label == "reference" { "reference_text" } else { "bitnet_rs_text" });
    }
}

fn validate_bitnet_contract(artifact: &Value, failures: &mut Vec<&'static str>) {
    let reference = &artifact["reference"];
    let bitnet = bitnet_side(artifact);

    if string_at(bitnet, &[&["loader_mode"], &["loader", "mode"]]) != Some("real_gguf") {
        failures.push("bitnet_loader_mode");
    }
    if string_at(bitnet, &[&["tokenizer_source"], &["tokenizer", "source"]])
        .unwrap_or_default()
        .is_empty()
    {
        failures.push("bitnet_tokenizer_source");
    }
    if bool_at(bitnet, &[&["tokenizer_strict"], &["tokenizer", "strict"]]) != Some(true) {
        failures.push("bitnet_tokenizer_strict");
    }
    if bool_at(bitnet, &[&["fallback_used"], &["backend", "fallback_used"]]) != Some(false) {
        failures.push("bitnet_fallback_used");
    }
    if string_at(bitnet, &[&["runtime_api"], &["backend", "runtime_api"]]) != Some("cpu") {
        failures.push("bitnet_runtime_api");
    }
    let kernel = bitnet["kernel"]
        .as_str()
        .or_else(|| string_at(bitnet, &[&["selected_kernel"], &["kernel", "selected_kernel"]]));
    if kernel.map(|value| value.contains("mock") || value.contains("diagnostic")).unwrap_or(true) {
        failures.push("bitnet_selected_kernel");
    }
    if topk(reference).is_none() {
        failures.push("reference_topk_step0");
    }
    if topk(bitnet).is_none() {
        failures.push("bitnet_rs_topk_step0");
    }
    if chosen_id(reference).is_none() {
        failures.push("reference_chosen_id");
    }
    if chosen_id(bitnet).is_none() {
        failures.push("bitnet_rs_chosen_id");
    }
}

fn bitnet_side(artifact: &Value) -> &Value {
    artifact.get("bitnet_rs").or_else(|| artifact.get("candidate")).unwrap_or(&Value::Null)
}

fn first_divergence(artifact: &Value) -> Option<Value> {
    let reference = &artifact["reference"];
    let bitnet = bitnet_side(artifact);
    if let Some(index) =
        first_id_divergence(ids(reference.get("prompt_ids"))?, ids(bitnet.get("prompt_ids"))?)
    {
        return Some(divergence(
            "prompt",
            "prompt_tokenizer_template",
            index,
            &reference["prompt_ids"],
            &bitnet["prompt_ids"],
        ));
    }
    let generated_divergence = first_id_divergence(
        ids(reference.get("generated_ids"))?,
        ids(bitnet.get("generated_ids"))?,
    );
    if generated_divergence.is_some()
        && let (Some(reference_topk), Some(bitnet_topk)) =
            (topk_pairs(reference), topk_pairs(bitnet))
        && reference_topk != bitnet_topk
    {
        return Some(divergence(
            "logits",
            "logits_or_shared_transformer_math",
            0,
            topk(reference).unwrap_or(&Value::Null),
            topk(bitnet).unwrap_or(&Value::Null),
        ));
    }
    if let Some(index) = generated_divergence {
        if let (Some(reference_topk), Some(bitnet_topk)) =
            (topk_pairs(reference), topk_pairs(bitnet))
            && reference_topk == bitnet_topk
            && chosen_id(reference)
                .zip(chosen_id(bitnet))
                .is_some_and(|(left, right)| left != right)
        {
            return Some(divergence(
                "sampler",
                "sampler",
                index,
                &reference["generated_ids"],
                &bitnet["generated_ids"],
            ));
        }
        return Some(divergence(
            "decode",
            "output_head_vocab_indexing_or_shared_transformer_math",
            index,
            &reference["generated_ids"],
            &bitnet["generated_ids"],
        ));
    }
    if let (Some(reference_topk), Some(bitnet_topk)) = (topk_pairs(reference), topk_pairs(bitnet))
        && reference_topk != bitnet_topk
    {
        return Some(divergence(
            "logits",
            "logits_or_shared_transformer_math",
            0,
            topk(reference).unwrap_or(&Value::Null),
            topk(bitnet).unwrap_or(&Value::Null),
        ));
    }
    if reference["text"] != bitnet["text"] {
        return Some(divergence(
            "text",
            "tokenizer_decode",
            0,
            &reference["text"],
            &bitnet["text"],
        ));
    }
    None
}

fn first_id_divergence(left: Vec<u64>, right: Vec<u64>) -> Option<usize> {
    let shared = left.len().min(right.len());
    for index in 0..shared {
        if left[index] != right[index] {
            return Some(index);
        }
    }
    (left.len() != right.len()).then_some(shared)
}

fn ids(value: Option<&Value>) -> Option<Vec<u64>> {
    value?.as_array()?.iter().map(Value::as_u64).collect()
}

fn topk(side: &Value) -> Option<&Value> {
    side.get("topk").or_else(|| side.get("topk_step0")).or_else(|| {
        side.get("logits_dump")
            .and_then(Value::as_array)
            .and_then(|steps| steps.first())
            .and_then(|step| step.get("top_logits"))
    })
}

fn topk_pairs(side: &Value) -> Option<Vec<(u64, f64)>> {
    topk(side)?.as_array()?.iter().map(topk_entry_pair).collect()
}

fn topk_entry_pair(entry: &Value) -> Option<(u64, f64)> {
    if let Some(values) = entry.as_array() {
        return Some((values.first()?.as_u64()?, values.get(1)?.as_f64()?));
    }
    Some((entry.get("token_id")?.as_u64()?, entry.get("logit")?.as_f64()?))
}

fn chosen_id(side: &Value) -> Option<u64> {
    side.get("chosen_id").and_then(Value::as_u64).or_else(|| {
        side.get("logits_dump")
            .and_then(Value::as_array)
            .and_then(|steps| steps.first())
            .and_then(|step| step.get("chosen_id"))
            .and_then(Value::as_u64)
    })
}

fn divergence(
    phase: &'static str,
    classification: &'static str,
    index: usize,
    reference: &Value,
    bitnet: &Value,
) -> Value {
    json!({
        "phase": phase,
        "classification": classification,
        "index": index,
        "reference": reference,
        "bitnet_rs": bitnet,
    })
}

fn side_summary(side: &Value) -> Value {
    json!({
        "backend": side["backend"],
        "kernel": side["kernel"],
        "loader_mode": string_at(side, &[&["loader_mode"], &["loader", "mode"]]),
        "tokenizer_source": string_at(side, &[&["tokenizer_source"], &["tokenizer", "source"]]),
        "tokenizer_strict": bool_at(side, &[&["tokenizer_strict"], &["tokenizer", "strict"]]),
        "fallback_used": bool_at(side, &[&["fallback_used"], &["backend", "fallback_used"]]),
        "runtime_api": string_at(side, &[&["runtime_api"], &["backend", "runtime_api"]]),
        "prompt_ids": side["prompt_ids"],
        "generated_ids": side["generated_ids"],
        "text": side["text"],
        "chosen_id": chosen_id(side),
        "topk_step0": topk(side).cloned().unwrap_or(Value::Null),
    })
}

fn is_bitnet_artifact(artifact: &Value) -> bool {
    artifact["model_family"].as_str() == Some("bitnet")
        || artifact["artifact_kind"].as_str() == Some("bitnet_cpu_reference_compare")
}

fn string_at<'a>(value: &'a Value, paths: &[&[&str]]) -> Option<&'a str> {
    'paths: for path in paths {
        let mut current = value;
        for key in *path {
            let Some(next) = current.get(*key) else {
                continue 'paths;
            };
            current = next;
        }
        if let Some(text) = current.as_str() {
            return Some(text);
        }
    }
    None
}

fn bool_at(value: &Value, paths: &[&[&str]]) -> Option<bool> {
    'paths: for path in paths {
        let mut current = value;
        for key in *path {
            let Some(next) = current.get(*key) else {
                continue 'paths;
            };
            current = next;
        }
        if let Some(flag) = current.as_bool() {
            return Some(flag);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(generated: &[u64], text: &str) -> Value {
        json!({
            "schema_version": "1.0.0",
            "artifact_kind": "backend_reference_compare",
            "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
            "model_family": "qwen3",
            "prompt_text": "What is 2+2?",
            "prompt_template": "qwen",
            "bos": false,
            "reference": {
                "backend": "known-good-reference",
                "kernel": "reference",
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4],
                "text": "4",
                "topk_step0": [[4, 10.0], [5, 1.0]],
                "chosen_id": 4
            },
            "bitnet_rs": {
                "backend": "cpu-rust",
                "kernel": "dense-q8_0-reference",
                "prompt_ids": [1, 2, 3],
                "generated_ids": generated,
                "text": text,
                "topk_step0": [[4, 10.0], [5, 1.0]],
                "chosen_id": generated.first().copied().unwrap_or(0)
            }
        })
    }

    fn bitnet_artifact() -> Value {
        json!({
            "schema_version": "1.0.0",
            "artifact_kind": "bitnet_cpu_reference_compare",
            "model_sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "model_family": "bitnet",
            "prompt_text": "What is 2+2?",
            "prompt_bytes": [87, 104, 97, 116, 32, 105, 115, 32, 50, 43, 50, 63],
            "prompt_template": "llama3-chat",
            "bos": true,
            "reference": {
                "backend": "known-good-bitnet-reference",
                "kernel": "reference",
                "prompt_ids": [128000, 3923, 374, 220, 17, 10, 17, 30],
                "generated_ids": [19],
                "text": "4",
                "topk_step0": [[19, 12.0], [20, 8.0]],
                "chosen_id": 19
            },
            "bitnet_rs": {
                "backend": "cpu-rust",
                "runtime_api": "cpu",
                "kernel": "i2_s-avx2-reference",
                "loader": {
                    "mode": "real_gguf"
                },
                "tokenizer": {
                    "source": "gguf_metadata",
                    "strict": true
                },
                "fallback_used": false,
                "prompt_ids": [128000, 3923, 374, 220, 17, 10, 17, 30],
                "generated_ids": [19],
                "text": "4",
                "logits_dump": [{
                    "step": 0,
                    "chosen_id": 19,
                    "top_logits": [
                        {"token_id": 19, "logit": 12.0},
                        {"token_id": 21, "logit": 8.0}
                    ]
                }]
            }
        })
    }

    #[test]
    fn reference_divergence_passes_matching_artifact() {
        let report =
            build_reference_divergence_receipt(Path::new("compare.json"), &artifact(&[4], "4"));

        assert_eq!(report["artifact_kind"], "slm_reference_divergence_validation");
        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], true);
        assert!(report["comparison"]["first_divergence"].is_null());
    }

    #[test]
    fn reference_divergence_records_sampler_mismatch_when_topk_matches() {
        let report =
            build_reference_divergence_receipt(Path::new("compare.json"), &artifact(&[5], "5"));

        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], false);
        assert_eq!(report["comparison"]["first_divergence"]["phase"], "sampler");
        assert_eq!(report["comparison"]["first_divergence"]["classification"], "sampler");
        assert_eq!(report["comparison"]["first_divergence"]["index"], 0);
    }

    #[test]
    fn reference_divergence_records_logit_mismatch_before_token_mismatch() {
        let mut input = artifact(&[5], "5");
        input["bitnet_rs"]["topk_step0"] = json!([[5, 10.0], [4, 1.0]]);

        let report = build_reference_divergence_receipt(Path::new("compare.json"), &input);

        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], false);
        assert_eq!(report["comparison"]["first_divergence"]["phase"], "logits");
        assert_eq!(
            report["comparison"]["first_divergence"]["classification"],
            "logits_or_shared_transformer_math"
        );
    }

    #[test]
    fn reference_divergence_keeps_decode_classification_without_topk() {
        let mut input = artifact(&[5], "5");
        input["reference"].as_object_mut().unwrap().remove("topk_step0");
        input["bitnet_rs"].as_object_mut().unwrap().remove("topk_step0");

        let report = build_reference_divergence_receipt(Path::new("compare.json"), &input);

        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], false);
        assert_eq!(report["comparison"]["first_divergence"]["phase"], "decode");
        assert_eq!(
            report["comparison"]["first_divergence"]["classification"],
            "output_head_vocab_indexing_or_shared_transformer_math"
        );
    }

    #[test]
    fn reference_divergence_normalizes_logit_dump_topk_objects() {
        let mut input = artifact(&[4], "4");
        input["bitnet_rs"].as_object_mut().unwrap().remove("topk_step0");
        input["bitnet_rs"]["logits_dump"] = json!([{
            "step": 0,
            "chosen_id": 4,
            "top_logits": [
                {"token_id": 4, "logit": 10.0},
                {"token_id": 5, "logit": 1.0}
            ]
        }]);

        let report = build_reference_divergence_receipt(Path::new("compare.json"), &input);

        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], true);
        assert!(report["comparison"]["first_divergence"].is_null());
    }

    #[test]
    fn reference_divergence_rejects_missing_bos_policy() {
        let mut input = artifact(&[4], "4");
        input.as_object_mut().unwrap().remove("bos");

        let report = build_reference_divergence_receipt(Path::new("compare.json"), &input);

        assert_eq!(report["validation"]["passed"], false);
        let failed = report["validation"]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "bos_policy"));
    }

    #[test]
    fn bitnet_reference_divergence_validates_strict_cpu_provenance() {
        let report =
            build_reference_divergence_receipt(Path::new("compare.json"), &bitnet_artifact());

        assert_eq!(report["artifact_kind"], "bitnet_cpu_reference_divergence_validation");
        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], false);
        assert_eq!(report["comparison"]["first_divergence"]["phase"], "logits");
        assert_eq!(report["comparison"]["bitnet_rs"]["loader_mode"], "real_gguf");
        assert_eq!(report["comparison"]["bitnet_rs"]["tokenizer_source"], "gguf_metadata");
    }

    #[test]
    fn bitnet_reference_divergence_rejects_hidden_fallback() {
        let mut input = bitnet_artifact();
        input["bitnet_rs"]["fallback_used"] = Value::Bool(true);

        let report = build_reference_divergence_receipt(Path::new("compare.json"), &input);

        assert_eq!(report["validation"]["passed"], false);
        let failed = report["validation"]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "bitnet_fallback_used"));
    }

    #[test]
    fn bitnet_reference_divergence_requires_prompt_bytes() {
        let mut input = bitnet_artifact();
        input.as_object_mut().unwrap().remove("prompt_bytes");

        let report = build_reference_divergence_receipt(Path::new("compare.json"), &input);

        assert_eq!(report["validation"]["passed"], false);
        let failed = report["validation"]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "prompt_bytes"));
    }
}
