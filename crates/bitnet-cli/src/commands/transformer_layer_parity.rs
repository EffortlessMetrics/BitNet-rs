//! Transformer-layer parity ladder for Lunar Lake BitNet CPU proof.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const CHECKSUM_TOLERANCE: f64 = 1e-5;

#[derive(Clone, Copy)]
struct BoundarySpec {
    id: &'static str,
    label: &'static str,
    stages: &'static [&'static str],
    require_all: bool,
}

const BOUNDARIES: &[BoundarySpec] = &[
    BoundarySpec {
        id: "embedding",
        label: "embedding input",
        stages: &["decode.input_embedding", "full_prompt.input_embedding"],
        require_all: false,
    },
    BoundarySpec {
        id: "attention_norm",
        label: "attention RMSNorm/subln input boundary",
        stages: &["block.attention_norm"],
        require_all: true,
    },
    BoundarySpec {
        id: "qkv_projection",
        label: "Q/K/V projection boundary",
        stages: &["attention.q_proj", "attention.k_proj", "attention.v_proj"],
        require_all: true,
    },
    BoundarySpec {
        id: "rope",
        label: "RoPE boundary",
        stages: &["attention.rope_metadata", "attention.q_rope", "attention.k_rope"],
        require_all: false,
    },
    BoundarySpec {
        id: "attention_scores",
        label: "attention score boundary",
        stages: &["attention.scores_post_mask"],
        require_all: true,
    },
    BoundarySpec {
        id: "attention_softmax",
        label: "attention softmax boundary",
        stages: &["attention.weights"],
        require_all: true,
    },
    BoundarySpec {
        id: "attention_output",
        label: "attention output projection boundary",
        stages: &["attention.output_heads", "attention.o_proj"],
        require_all: true,
    },
    BoundarySpec {
        id: "ffn_norm",
        label: "FFN RMSNorm/subln input boundary",
        stages: &["block.ffn_norm"],
        require_all: true,
    },
    BoundarySpec {
        id: "ffn_relu2",
        label: "FFN ReLU2/gated activation boundary",
        stages: &["mlp.gate_proj", "mlp.gate_activation", "mlp.up_proj", "mlp.gated_product"],
        require_all: true,
    },
    BoundarySpec {
        id: "ffn_down_projection",
        label: "FFN down projection boundary",
        stages: &["mlp.down_proj"],
        require_all: true,
    },
    BoundarySpec {
        id: "residual",
        label: "residual/block output boundary",
        stages: &["block.post_attention_residual", "block.output"],
        require_all: true,
    },
    BoundarySpec {
        id: "final_norm",
        label: "final norm boundary",
        stages: &["model.final_norm", "decode.forward_output"],
        require_all: false,
    },
    BoundarySpec {
        id: "lm_head",
        label: "lm_head logits boundary",
        stages: &["lm_head.input_hidden", "lm_head.logits", "lm_head.top_logits"],
        require_all: true,
    },
];

/// Classify the 258V CPU transformer-layer trace ladder after prior boundary checks.
#[derive(Args, Debug)]
pub struct TransformerLayerParityCommand {
    /// Local BitNet-rs Qwen/transformer trace JSONL from `run --qwen-trace-jsonl`.
    #[arg(long, value_name = "PATH")]
    pub trace_jsonl: PathBuf,

    /// Optional comparison/reference transformer trace JSONL for checksum comparison.
    #[arg(
        long = "comparison-trace-jsonl",
        visible_alias = "reference-trace-jsonl",
        value_name = "PATH"
    )]
    pub comparison_trace_jsonl: Option<PathBuf>,

    /// Optional prompt-authority audit receipt for the same model/prompt policy.
    #[arg(long, value_name = "PATH")]
    pub prompt_audit: Option<PathBuf>,

    /// Optional first-token divergence classifier receipt.
    #[arg(long, value_name = "PATH")]
    pub first_token_divergence: Option<PathBuf>,

    /// Optional CPU QK256/I2_S/I8_S semantic audit receipt.
    #[arg(long, value_name = "PATH")]
    pub qk256_semantic_audit: Option<PathBuf>,

    /// Optional output-head/logits-index audit receipt.
    #[arg(long, value_name = "PATH")]
    pub output_head_logits_audit: Option<PathBuf>,

    /// Output transformer-layer parity ladder receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/transformer-layer-parity.json"
    )]
    pub json_out: PathBuf,
}

impl TransformerLayerParityCommand {
    /// Execute offline transformer-layer ladder classification.
    pub async fn execute(&self) -> Result<()> {
        let local_trace = read_jsonl(&self.trace_jsonl)?;
        let comparison_trace =
            self.comparison_trace_jsonl.as_deref().map(read_jsonl).transpose()?.unwrap_or_default();
        let prompt_audit = self.prompt_audit.as_deref().map(read_json).transpose()?;
        let first_token_divergence =
            self.first_token_divergence.as_deref().map(read_json).transpose()?;
        let qk256_semantic_audit =
            self.qk256_semantic_audit.as_deref().map(read_json).transpose()?;
        let output_head_logits_audit =
            self.output_head_logits_audit.as_deref().map(read_json).transpose()?;

        let receipt = build_transformer_layer_parity_receipt(
            &TransformerLayerInputs {
                trace_jsonl: &self.trace_jsonl,
                comparison_trace_jsonl: self.comparison_trace_jsonl.as_deref(),
                prompt_audit: self.prompt_audit.as_deref(),
                first_token_divergence: self.first_token_divergence.as_deref(),
                qk256_semantic_audit: self.qk256_semantic_audit.as_deref(),
                output_head_logits_audit: self.output_head_logits_audit.as_deref(),
            },
            &local_trace,
            &comparison_trace,
            prompt_audit.as_ref(),
            first_token_divergence.as_ref(),
            qk256_semantic_audit.as_ref(),
            output_head_logits_audit.as_ref(),
        );

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("transformer-layer parity ladder written to {}", self.json_out.display());

        if receipt["validation"]["passed"].as_bool() != Some(true) {
            anyhow::bail!(
                "transformer-layer parity ladder validation failed; receipt written to {}",
                self.json_out.display()
            );
        }
        Ok(())
    }
}

struct TransformerLayerInputs<'a> {
    trace_jsonl: &'a Path,
    comparison_trace_jsonl: Option<&'a Path>,
    prompt_audit: Option<&'a Path>,
    first_token_divergence: Option<&'a Path>,
    qk256_semantic_audit: Option<&'a Path>,
    output_head_logits_audit: Option<&'a Path>,
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut values = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        values.push(
            serde_json::from_str::<Value>(line)
                .with_context(|| format!("failed to parse {} line {}", path.display(), idx + 1))?,
        );
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn build_transformer_layer_parity_receipt(
    inputs: &TransformerLayerInputs<'_>,
    local_trace: &[Value],
    comparison_trace: &[Value],
    prompt_audit: Option<&Value>,
    first_token_divergence: Option<&Value>,
    qk256_semantic_audit: Option<&Value>,
    output_head_logits_audit: Option<&Value>,
) -> Value {
    let validation_failures = validate_inputs(local_trace);
    let stage_index = stage_counts(local_trace);
    let local_by_stage = group_by_stage(local_trace);
    let comparison_by_stage = group_by_stage(comparison_trace);
    let comparison_supplied = !comparison_trace.is_empty();

    let boundaries = BOUNDARIES
        .iter()
        .map(|boundary| {
            boundary_receipt(boundary, &local_by_stage, &comparison_by_stage, comparison_supplied)
        })
        .collect::<Vec<_>>();
    let classification = classify_ladder(&boundaries, comparison_supplied, &validation_failures);
    let summary = summarize_boundaries(&boundaries, &classification);

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_transformer_layer_parity_ladder",
        "machine_id": "intel-258v",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "transformer_layer_ladder_recorded",
        "claim": "cpu258v_transformer_layer_parity_ladder",
        "inputs": {
            "trace_jsonl": inputs.trace_jsonl.display().to_string(),
            "comparison_trace_jsonl": inputs.comparison_trace_jsonl.map(|path| path.display().to_string()),
            "prompt_audit": inputs.prompt_audit.map(|path| path.display().to_string()),
            "first_token_divergence": inputs.first_token_divergence.map(|path| path.display().to_string()),
            "qk256_semantic_audit": inputs.qk256_semantic_audit.map(|path| path.display().to_string()),
            "output_head_logits_audit": inputs.output_head_logits_audit.map(|path| path.display().to_string()),
        },
        "validation": {
            "passed": validation_failures.is_empty(),
            "failed_rules": validation_failures,
        },
        "prior_boundaries": {
            "prompt_authority": audit_summary(prompt_audit),
            "first_token_divergence": audit_summary(first_token_divergence),
            "qk256_semantics": audit_summary(qk256_semantic_audit),
            "output_head_logits_index": audit_summary(output_head_logits_audit),
        },
        "trace": {
            "local_events": local_trace.len(),
            "comparison_events": comparison_trace.len(),
            "comparison_trace_supplied": comparison_supplied,
            "local_stages": stage_index,
        },
        "summary": summary,
        "classification": classification,
        "boundaries": boundaries,
        "fallback_used": false,
        "claim_boundary": {
            "may_claim": [
                "The 258V CPU transformer-layer trace ladder records which internal boundaries have local BitNet-rs evidence.",
                "When a comparison or reference trace is supplied, matching stages can be compared by dimensions, finite status, and checksum tolerance.",
                "Missing comparison/reference layer traces are recorded as blockers instead of inferred parity."
            ],
            "must_not_claim": [
                "BitNet answer quality is newly proven",
                "First-token logits parity with the external reference is proven unless a reference trace/logit source is supplied and matched",
                "CPU speed or sustained throughput",
                "Arc 140V execution or acceleration",
                "Intel NPU execution or acceleration",
                "Full model correctness"
            ]
        }
    })
}

fn validate_inputs(local_trace: &[Value]) -> Vec<&'static str> {
    let mut failures = Vec::new();
    if local_trace.is_empty() {
        failures.push("local_trace_events");
    }
    if !local_trace.iter().any(|event| event["stage"].as_str() == Some("trace_start")) {
        failures.push("trace_start_event");
    }
    if !local_trace.iter().any(|event| event["kind"].as_str() == Some("qwen_trace_tensor")) {
        failures.push("local_tensor_trace_events");
    }
    failures
}

fn stage_counts(events: &[Value]) -> Value {
    let mut counts = BTreeMap::<String, usize>::new();
    for event in events {
        if let Some(stage) = event["stage"].as_str() {
            *counts.entry(stage.to_string()).or_default() += 1;
        }
    }
    json!(counts)
}

fn group_by_stage(events: &[Value]) -> BTreeMap<String, Vec<&Value>> {
    let mut by_stage = BTreeMap::<String, Vec<&Value>>::new();
    for event in events {
        if let Some(stage) = event["stage"].as_str() {
            by_stage.entry(stage.to_string()).or_default().push(event);
        }
    }
    by_stage
}

fn boundary_receipt(
    spec: &BoundarySpec,
    local_by_stage: &BTreeMap<String, Vec<&Value>>,
    comparison_by_stage: &BTreeMap<String, Vec<&Value>>,
    comparison_supplied: bool,
) -> Value {
    let local_stage_set = matching_stages(spec.stages, local_by_stage);
    let comparison_stage_set = matching_stages(spec.stages, comparison_by_stage);
    let local_events = local_stage_set
        .iter()
        .flat_map(|stage| local_by_stage.get(stage).into_iter().flat_map(|events| events.iter()))
        .map(|event| event_summary(event))
        .collect::<Vec<_>>();
    let comparison_events = comparison_stage_set
        .iter()
        .flat_map(|stage| {
            comparison_by_stage.get(stage).into_iter().flat_map(|events| events.iter())
        })
        .map(|event| event_summary(event))
        .collect::<Vec<_>>();
    let missing_local_stages = missing_stages(spec, &local_stage_set);
    let status = boundary_status(spec, &local_stage_set, &local_events);
    let comparison = if comparison_supplied {
        compare_boundary(spec, local_by_stage, comparison_by_stage)
    } else {
        json!({
            "status": "comparison_trace_missing",
            "first_mismatch_stage": null,
            "notes": ["No comparison/reference transformer-layer trace was supplied for this boundary."]
        })
    };

    json!({
        "id": spec.id,
        "label": spec.label,
        "required_stages": spec.stages,
        "require_all_stages": spec.require_all,
        "status": status,
        "local_stages_recorded": local_stage_set,
        "missing_local_stages": missing_local_stages,
        "local_events": local_events,
        "comparison_stages_recorded": comparison_stage_set,
        "comparison_events": comparison_events,
        "comparison": comparison,
    })
}

fn matching_stages(stages: &[&str], by_stage: &BTreeMap<String, Vec<&Value>>) -> BTreeSet<String> {
    stages
        .iter()
        .filter(|stage| by_stage.contains_key(**stage))
        .map(|stage| (*stage).to_string())
        .collect()
}

fn missing_stages(spec: &BoundarySpec, found: &BTreeSet<String>) -> Vec<&'static str> {
    if !spec.require_all && !found.is_empty() {
        return Vec::new();
    }
    spec.stages.iter().copied().filter(|stage| !found.contains(*stage)).collect()
}

fn boundary_status(
    spec: &BoundarySpec,
    found: &BTreeSet<String>,
    local_events: &[Value],
) -> &'static str {
    if found.is_empty() {
        return "missing";
    }
    if local_events.iter().any(|event| event["nonfinite"].as_u64().unwrap_or(0) > 0) {
        return "nonfinite";
    }
    if spec.require_all && found.len() < spec.stages.len() {
        return "partial";
    }
    "recorded"
}

fn event_summary(event: &Value) -> Value {
    json!({
        "kind": event["kind"].clone(),
        "stage": event["stage"].clone(),
        "step": event["step"].clone(),
        "layer": event["layer"].clone(),
        "dims": event["dims"].clone(),
        "len": event["len"].clone(),
        "finite": event["finite"].clone(),
        "nonfinite": event["nonfinite"].clone(),
        "checksum": event["checksum"].clone(),
        "rms": event["rms"].clone(),
        "top_logits": event["top_logits"].clone(),
        "chosen_id": event["chosen_id"].clone(),
    })
}

fn compare_boundary(
    spec: &BoundarySpec,
    local_by_stage: &BTreeMap<String, Vec<&Value>>,
    comparison_by_stage: &BTreeMap<String, Vec<&Value>>,
) -> Value {
    let mut compared = Vec::new();
    let mut missing_comparison = Vec::new();
    let mut first_mismatch: Option<Value> = None;

    for stage in spec.stages {
        let Some(local) = local_by_stage.get(*stage).and_then(|events| events.first()).copied()
        else {
            continue;
        };
        let Some(comparison) =
            comparison_by_stage.get(*stage).and_then(|events| events.first()).copied()
        else {
            missing_comparison.push(*stage);
            continue;
        };
        let stage_comparison = compare_event(stage, local, comparison);
        if first_mismatch.is_none() && stage_comparison["matched"].as_bool() == Some(false) {
            first_mismatch = Some(stage_comparison.clone());
        }
        compared.push(stage_comparison);
    }

    let status = if first_mismatch.is_some() {
        "mismatch"
    } else if compared.is_empty() {
        "no_overlapping_comparison_stages"
    } else if !missing_comparison.is_empty() {
        "partial_comparison"
    } else {
        "matched"
    };

    json!({
        "status": status,
        "first_mismatch_stage": first_mismatch.as_ref().and_then(|value| value["stage"].as_str()).map(str::to_string),
        "first_mismatch": first_mismatch,
        "compared_stages": compared,
        "missing_comparison_stages": missing_comparison,
    })
}

fn compare_event(stage: &str, local: &Value, comparison: &Value) -> Value {
    let local_len = local["len"].as_u64();
    let comparison_len = comparison["len"].as_u64();
    let local_dims = &local["dims"];
    let comparison_dims = &comparison["dims"];
    let local_nonfinite = local["nonfinite"].as_u64().unwrap_or(0);
    let comparison_nonfinite = comparison["nonfinite"].as_u64().unwrap_or(0);
    let checksum_delta = match (local["checksum"].as_f64(), comparison["checksum"].as_f64()) {
        (Some(left), Some(right)) => Some((left - right).abs()),
        _ => None,
    };
    let evidence_match = checksum_delta
        .map(|delta| delta <= CHECKSUM_TOLERANCE)
        .unwrap_or_else(|| event_payload_match(local, comparison));
    let matched = local_len == comparison_len
        && local_dims == comparison_dims
        && local_nonfinite == 0
        && comparison_nonfinite == 0
        && evidence_match;

    json!({
        "stage": stage,
        "matched": matched,
        "local_len": local_len,
        "comparison_len": comparison_len,
        "dims_match": local_dims == comparison_dims,
        "local_nonfinite": local_nonfinite,
        "comparison_nonfinite": comparison_nonfinite,
        "checksum_delta": checksum_delta,
        "checksum_tolerance": CHECKSUM_TOLERANCE,
    })
}

fn event_payload_match(local: &Value, comparison: &Value) -> bool {
    if local.get("top_logits").is_some() || comparison.get("top_logits").is_some() {
        return local["chosen_id"] == comparison["chosen_id"]
            && local["top_logits"] == comparison["top_logits"];
    }

    let ignored = BTreeSet::from(["kind", "stage"]);
    let Some(local_object) = local.as_object() else {
        return local == comparison;
    };
    let Some(comparison_object) = comparison.as_object() else {
        return false;
    };
    let local_filtered = local_object
        .iter()
        .filter(|(key, _)| !ignored.contains(key.as_str()))
        .collect::<BTreeMap<_, _>>();
    let comparison_filtered = comparison_object
        .iter()
        .filter(|(key, _)| !ignored.contains(key.as_str()))
        .collect::<BTreeMap<_, _>>();
    local_filtered == comparison_filtered
}

fn classify_ladder(
    boundaries: &[Value],
    comparison_supplied: bool,
    validation_failures: &[&'static str],
) -> Value {
    if !validation_failures.is_empty() {
        return json!({
            "classification": "transformer_layer_trace_invalid",
            "first_divergence_stage": "receipt_contract",
            "first_internal_boundary": null,
            "first_mismatch": null,
            "next_required_evidence": "record a local run --qwen-trace-jsonl trace with tensor events",
        });
    }

    if let Some(boundary) = boundaries
        .iter()
        .find(|boundary| boundary["comparison"]["status"].as_str() == Some("mismatch"))
    {
        return json!({
            "classification": "comparison_layer_trace_mismatch",
            "first_divergence_stage": "transformer_layer",
            "first_internal_boundary": boundary["id"].clone(),
            "first_mismatch": boundary["comparison"]["first_mismatch"].clone(),
            "next_required_evidence": "inspect the first mismatched transformer boundary against the comparison/reference trace",
        });
    }

    if let Some(boundary) = boundaries.iter().find(|boundary| {
        matches!(boundary["status"].as_str(), Some("missing" | "partial" | "nonfinite"))
    }) {
        return json!({
            "classification": "local_transformer_layer_ladder_has_gaps",
            "first_divergence_stage": "transformer_layer_instrumentation",
            "first_internal_boundary": boundary["id"].clone(),
            "first_mismatch": null,
            "next_required_evidence": "record the missing or nonfinite local transformer trace boundary",
        });
    }

    if comparison_supplied {
        json!({
            "classification": "primary_and_comparison_transformer_layer_ladder_matched",
            "first_divergence_stage": "none",
            "first_internal_boundary": null,
            "first_mismatch": null,
            "next_required_evidence": "rerun answer corpus against the matched transformer-layer boundary evidence",
        })
    } else {
        json!({
            "classification": "local_transformer_layer_ladder_recorded_comparison_trace_missing",
            "first_divergence_stage": "comparison_layer_trace_missing",
            "first_internal_boundary": null,
            "first_mismatch": null,
            "next_required_evidence": "capture a comparison/reference transformer-layer trace or layer dump to compare against the local ladder",
        })
    }
}

fn summarize_boundaries(boundaries: &[Value], classification: &Value) -> Value {
    let recorded = boundaries
        .iter()
        .filter(|boundary| boundary["status"].as_str() == Some("recorded"))
        .count();
    let partial =
        boundaries.iter().filter(|boundary| boundary["status"].as_str() == Some("partial")).count();
    let missing =
        boundaries.iter().filter(|boundary| boundary["status"].as_str() == Some("missing")).count();
    let nonfinite = boundaries
        .iter()
        .filter(|boundary| boundary["status"].as_str() == Some("nonfinite"))
        .count();
    let matched = boundaries
        .iter()
        .filter(|boundary| boundary["comparison"]["status"].as_str() == Some("matched"))
        .count();
    let mismatched = boundaries
        .iter()
        .filter(|boundary| boundary["comparison"]["status"].as_str() == Some("mismatch"))
        .count();

    json!({
        "boundaries_total": boundaries.len(),
        "boundaries_recorded": recorded,
        "boundaries_partial": partial,
        "boundaries_missing": missing,
        "boundaries_nonfinite": nonfinite,
        "comparison_boundaries_matched": matched,
        "comparison_boundaries_mismatched": mismatched,
        "first_divergence": classification,
        "classification": classification["classification"].clone(),
    })
}

fn audit_summary(audit: Option<&Value>) -> Value {
    let Some(audit) = audit else {
        return json!({
            "provided": false,
            "artifact_kind": null,
            "classification": null,
        });
    };
    json!({
        "provided": true,
        "artifact_kind": audit["artifact_kind"].clone(),
        "proof_stage": audit["proof_stage"].clone(),
        "validation": audit["validation"].clone(),
        "summary": audit["summary"].clone(),
        "classification": audit["classification"].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(stage: &str, checksum: f64) -> Value {
        json!({
            "kind": "qwen_trace_tensor",
            "stage": stage,
            "step": 0,
            "layer": 0,
            "dims": [1, 1, 4],
            "len": 4,
            "finite": 4,
            "nonfinite": 0,
            "mean": 0.5,
            "rms": 1.0,
            "checksum": checksum,
            "sample": [0.1, 0.2, 0.3, 0.4]
        })
    }

    fn logits(stage: &str) -> Value {
        json!({
            "kind": "qwen_trace_logits",
            "stage": stage,
            "step": 0,
            "chosen_id": 19,
            "top_logits": [
                {"token_id": 19, "logit": 42.0}
            ]
        })
    }

    fn full_local_trace() -> Vec<Value> {
        vec![
            json!({"kind": "qwen_trace_event", "stage": "trace_start"}),
            tensor("decode.input_embedding", 1.0),
            tensor("block.attention_norm", 2.0),
            tensor("attention.q_proj", 3.0),
            tensor("attention.k_proj", 4.0),
            tensor("attention.v_proj", 5.0),
            json!({"kind": "qwen_trace_event", "stage": "attention.rope_metadata", "step": 0, "layer": 0}),
            tensor("attention.scores_post_mask", 6.0),
            tensor("attention.weights", 7.0),
            tensor("attention.output_heads", 8.0),
            tensor("attention.o_proj", 9.0),
            tensor("block.ffn_norm", 10.0),
            tensor("mlp.gate_proj", 11.0),
            tensor("mlp.gate_activation", 12.0),
            tensor("mlp.up_proj", 13.0),
            tensor("mlp.gated_product", 14.0),
            tensor("mlp.down_proj", 15.0),
            tensor("block.post_attention_residual", 16.0),
            tensor("block.output", 17.0),
            tensor("model.final_norm", 18.0),
            tensor("lm_head.input_hidden", 19.0),
            tensor("lm_head.logits", 20.0),
            logits("lm_head.top_logits"),
        ]
    }

    fn inputs() -> TransformerLayerInputs<'static> {
        TransformerLayerInputs {
            trace_jsonl: Path::new("trace.jsonl"),
            comparison_trace_jsonl: None,
            prompt_audit: None,
            first_token_divergence: None,
            qk256_semantic_audit: None,
            output_head_logits_audit: None,
        }
    }

    #[test]
    fn records_full_local_ladder_without_comparison_trace() {
        let receipt = build_transformer_layer_parity_receipt(
            &inputs(),
            &full_local_trace(),
            &[],
            None,
            None,
            None,
            None,
        );

        assert_eq!(receipt["validation"]["passed"], true);
        assert_eq!(receipt["summary"]["boundaries_missing"], 0);
        assert_eq!(
            receipt["summary"]["classification"],
            "local_transformer_layer_ladder_recorded_comparison_trace_missing"
        );
        assert_eq!(
            receipt["classification"]["first_divergence_stage"],
            "comparison_layer_trace_missing"
        );
    }

    #[test]
    fn classifies_first_comparison_checksum_mismatch() {
        let local = full_local_trace();
        let mut reference = full_local_trace();
        for event in &mut reference {
            if event["stage"].as_str() == Some("attention.q_proj") {
                event["checksum"] = json!(999.0);
            }
        }

        let receipt = build_transformer_layer_parity_receipt(
            &inputs(),
            &local,
            &reference,
            None,
            None,
            None,
            None,
        );

        assert_eq!(receipt["summary"]["classification"], "comparison_layer_trace_mismatch");
        assert_eq!(receipt["classification"]["first_internal_boundary"], "qkv_projection");
        assert_eq!(receipt["classification"]["first_mismatch"]["stage"], "attention.q_proj");
    }

    #[test]
    fn records_missing_boundary_as_ladder_gap() {
        let mut local = full_local_trace();
        local.retain(|event| event["stage"].as_str() != Some("attention.weights"));

        let receipt =
            build_transformer_layer_parity_receipt(&inputs(), &local, &[], None, None, None, None);

        assert_eq!(receipt["summary"]["classification"], "local_transformer_layer_ladder_has_gaps");
        assert_eq!(receipt["classification"]["first_internal_boundary"], "attention_softmax");
    }
}
