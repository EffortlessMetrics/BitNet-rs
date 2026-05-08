//! SLM reference divergence artifact validator.

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

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": "slm_reference_divergence_validation",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "external_reference_compared",
        "claim": "slm_reference_divergence_diagnostic",
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
            "First divergence evidence can separate tokenizer, prompt-template, decode, logits, and text-decoding issues."
        ],
        "must_not_claim": [
            "BitNet-rs can run the external reference engine.",
            "General chat quality is proven.",
            "Sustained 8250U throughput is proven.",
            "Server, GPU, OpenVINO, UHD 620, or NPU execution is involved."
        ],
    })
}

fn validate_artifact(artifact: &Value) -> Vec<&'static str> {
    let mut failures = Vec::new();
    if !matches!(
        artifact["artifact_kind"].as_str(),
        Some("backend_reference_compare" | "slm_reference_divergence")
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
    if artifact["prompt_text"].as_str().unwrap_or_default().is_empty() {
        failures.push("prompt_text");
    }
    if !artifact["prompt_template"].is_string() && !artifact["prompt_template"].is_object() {
        failures.push("prompt_template");
    }
    if !artifact["bos"].is_boolean() && !artifact["add_bos"].is_boolean() {
        failures.push("bos_policy");
    }
    validate_side("reference", &artifact["reference"], &mut failures);
    validate_side("bitnet_rs", bitnet_side(artifact), &mut failures);
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

fn bitnet_side(artifact: &Value) -> &Value {
    artifact.get("bitnet_rs").or_else(|| artifact.get("candidate")).unwrap_or(&Value::Null)
}

fn first_divergence(artifact: &Value) -> Option<Value> {
    let reference = &artifact["reference"];
    let bitnet = bitnet_side(artifact);
    if let Some(index) =
        first_id_divergence(ids(reference.get("prompt_ids"))?, ids(bitnet.get("prompt_ids"))?)
    {
        return Some(divergence("prompt", index, &reference["prompt_ids"], &bitnet["prompt_ids"]));
    }
    if let Some(index) =
        first_id_divergence(ids(reference.get("generated_ids"))?, ids(bitnet.get("generated_ids"))?)
    {
        return Some(divergence(
            "decode",
            index,
            &reference["generated_ids"],
            &bitnet["generated_ids"],
        ));
    }
    if reference["text"] != bitnet["text"] {
        return Some(divergence("text", 0, &reference["text"], &bitnet["text"]));
    }
    if let (Some(reference_topk), Some(bitnet_topk)) = (topk(reference), topk(bitnet))
        && reference_topk != bitnet_topk
    {
        return Some(divergence("logits", 0, reference_topk, bitnet_topk));
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
    side.get("topk").or_else(|| side.get("topk_step0"))
}

fn divergence(phase: &'static str, index: usize, reference: &Value, bitnet: &Value) -> Value {
    json!({
        "phase": phase,
        "index": index,
        "reference": reference,
        "bitnet_rs": bitnet,
    })
}

fn side_summary(side: &Value) -> Value {
    json!({
        "backend": side["backend"],
        "kernel": side["kernel"],
        "prompt_ids": side["prompt_ids"],
        "generated_ids": side["generated_ids"],
        "text": side["text"],
        "chosen_id": side["chosen_id"],
        "topk_step0": topk(side).cloned().unwrap_or(Value::Null),
    })
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
    fn reference_divergence_records_decode_token_mismatch() {
        let report =
            build_reference_divergence_receipt(Path::new("compare.json"), &artifact(&[5], "5"));

        assert_eq!(report["validation"]["passed"], true);
        assert_eq!(report["comparison"]["passed"], false);
        assert_eq!(report["comparison"]["first_divergence"]["phase"], "decode");
        assert_eq!(report["comparison"]["first_divergence"]["index"], 0);
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
}
