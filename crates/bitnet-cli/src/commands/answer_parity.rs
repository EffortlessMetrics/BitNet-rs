//! Answer-corpus parity comparator.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

/// Compare answer-corpus receipts.
#[derive(Args, Debug)]
pub struct AnswerParityCommand {
    /// Scalar strict CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub scalar: Option<PathBuf>,

    /// AVX2 strict CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub avx2: Option<PathBuf>,

    /// Left/baseline answer-corpus receipt for generic backend comparison.
    #[arg(long, value_name = "PATH")]
    pub left: Option<PathBuf>,

    /// Right/variant answer-corpus receipt for generic backend comparison.
    #[arg(long, value_name = "PATH")]
    pub right: Option<PathBuf>,

    /// Label to use for the left/baseline receipt in generic comparison output.
    #[arg(long, value_name = "LABEL")]
    pub left_label: Option<String>,

    /// Label to use for the right/variant receipt in generic comparison output.
    #[arg(long, value_name = "LABEL")]
    pub right_label: Option<String>,

    /// Output parity receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/cpu-answer-parity.json"
    )]
    pub json_out: PathBuf,

    /// Machine identifier for hardware-scoped parity artifacts.
    #[arg(long, value_name = "ID")]
    pub machine: Option<String>,

    /// Same-machine platform probe artifact to cross-link for CPU topology and power context.
    #[arg(long, value_name = "PATH")]
    pub platform_artifact: Option<PathBuf>,
}

impl AnswerParityCommand {
    /// Execute the offline answer parity comparison.
    pub async fn execute(&self) -> Result<()> {
        let inputs = self.resolve_inputs()?;
        let left = read_json(inputs.left_path)?;
        let right = read_json(inputs.right_path)?;
        let left_label = inputs
            .left_label
            .map(str::to_string)
            .unwrap_or_else(|| infer_lane_label(&left, inputs.left_path, "left"));
        let right_label = inputs
            .right_label
            .map(str::to_string)
            .unwrap_or_else(|| infer_lane_label(&right, inputs.right_path, "right"));
        let platform = match &self.platform_artifact {
            Some(path) => Some(read_json(path)?),
            None => None,
        };
        let receipt = build_answer_parity_receipt(
            inputs.left_path,
            &left,
            inputs.right_path,
            &right,
            &left_label,
            &right_label,
            inputs.legacy_scalar_avx2,
            self.machine.as_deref(),
            self.platform_artifact.as_deref(),
            platform.as_ref(),
        );
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
                "{}-vs-{} answer parity failed for {failed} case(s); receipt written to {}",
                left_label,
                right_label,
                self.json_out.display()
            );
        }
        Ok(())
    }

    fn resolve_inputs(&self) -> Result<ParityInputs<'_>> {
        let generic_requested = self.left.is_some()
            || self.right.is_some()
            || self.left_label.is_some()
            || self.right_label.is_some();
        if generic_requested {
            if self.scalar.is_some() || self.avx2.is_some() {
                anyhow::bail!(
                    "use either --left/--right for generic parity or --scalar/--avx2 for legacy CPU parity"
                );
            }
            let left_path =
                self.left.as_deref().context("--left is required for generic answer parity")?;
            let right_path =
                self.right.as_deref().context("--right is required for generic answer parity")?;
            return Ok(ParityInputs {
                left_path,
                right_path,
                left_label: self.left_label.as_deref(),
                right_label: self.right_label.as_deref(),
                legacy_scalar_avx2: false,
            });
        }

        let left_path = self
            .scalar
            .as_deref()
            .context("--scalar and --avx2 are required for legacy CPU answer parity")?;
        let right_path = self
            .avx2
            .as_deref()
            .context("--scalar and --avx2 are required for legacy CPU answer parity")?;
        Ok(ParityInputs {
            left_path,
            right_path,
            left_label: Some("scalar"),
            right_label: Some("avx2"),
            legacy_scalar_avx2: true,
        })
    }
}

struct ParityInputs<'a> {
    left_path: &'a Path,
    right_path: &'a Path,
    left_label: Option<&'a str>,
    right_label: Option<&'a str>,
    legacy_scalar_avx2: bool,
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

#[allow(clippy::too_many_arguments)]
fn build_answer_parity_receipt(
    left_path: &Path,
    left: &Value,
    right_path: &Path,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    machine: Option<&str>,
    platform_artifact: Option<&Path>,
    platform: Option<&Value>,
) -> Value {
    let mut shared_failures = Vec::new();
    compare_top_level_contract(left, right, legacy_scalar_avx2, &mut shared_failures);

    let left_cases = cases_by_id(left);
    let right_cases = cases_by_id(right);
    let case_ids = left_cases.keys().chain(right_cases.keys()).cloned().collect::<BTreeSet<_>>();

    let mut first_divergence = None;
    let cases = case_ids
        .iter()
        .map(|id| {
            compare_case(
                id,
                left_cases.get(id).copied(),
                right_cases.get(id).copied(),
                left_label,
                right_label,
                legacy_scalar_avx2,
                &mut first_divergence,
            )
        })
        .collect::<Vec<_>>();
    let logits_topk_frontier =
        build_logits_topk_frontier(&case_ids, &left_cases, &right_cases, left_label, right_label);
    let generated_output_frontier = build_generated_output_frontier(
        &case_ids,
        &left_cases,
        &right_cases,
        left_label,
        right_label,
    );

    let passed = cases.iter().filter(|case| case["passed"] == true).count();
    let failed = cases.len().saturating_sub(passed) + usize::from(!shared_failures.is_empty());
    let backend = parity_backend_summary(left, right, left_label, right_label, legacy_scalar_avx2);
    let kernel = parity_kernel_summary(left, right, left_label, right_label, legacy_scalar_avx2);

    let inputs = if legacy_scalar_avx2 {
        json!({
            "scalar_receipt_path": left_path.display().to_string(),
            "avx2_receipt_path": right_path.display().to_string(),
        })
    } else {
        json!({
            "left_receipt_path": left_path.display().to_string(),
            "right_receipt_path": right_path.display().to_string(),
            "left_label": left_label,
            "right_label": right_label,
        })
    };
    let may_claim = if legacy_scalar_avx2 {
        json!([
            "Scalar versus AVX2 full-decode answer parity can be audited for the compared receipts.",
            "First divergence evidence can separate AVX2 kernel issues from shared prompt, tokenizer, logits, sampler, or text decoding issues."
        ])
    } else {
        json!([
            "Full-decode answer parity can be audited for the compared receipts.",
            "First divergence evidence can separate backend issues from shared prompt, tokenizer, logits, sampler, or text decoding issues."
        ])
    };
    let must_not_claim = if legacy_scalar_avx2 {
        json!([
            "General chat quality is proven.",
            "Sustained CPU throughput is proven.",
            "Server inference is complete.",
            "GPU or NPU execution is involved."
        ])
    } else {
        json!([
            "General chat quality is proven.",
            "Sustained throughput is proven.",
            "Server inference is complete.",
            "A backend is answer-ready when the compared model artifact is not answer-ready."
        ])
    };

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": if legacy_scalar_avx2 {
            "bitnet_cpu_answer_parity"
        } else {
            "bitnet_answer_corpus_parity"
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "full_decode_parity_compared",
        "requested_backend": backend["requested_backend"],
        "selected_backend": backend["selected_backend"],
        "runtime_api": backend["runtime_api"],
        "fallback_used": backend["fallback_used"],
        "backend": backend,
        "kernel": kernel,
        "claim": if legacy_scalar_avx2 {
            "scalar_avx2_full_decode_answer_parity"
        } else {
            "full_decode_answer_parity_diagnostic"
        },
        "speedup_claim": false,
        "inputs": inputs,
        "machine": {
            "machine_id": machine,
            "platform_artifact": platform_artifact.map(|path| path.display().to_string()),
            "cpu": platform.and_then(|value| value.get("cpu")).cloned().unwrap_or(Value::Null),
            "memory": platform.and_then(|value| value.get("memory")).cloned().unwrap_or(Value::Null),
            "power": platform.and_then(|value| value.get("power")).cloned().unwrap_or(Value::Null),
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
        "logits_topk_frontier": logits_topk_frontier,
        "generated_output_frontier": generated_output_frontier,
        "cases": cases,
        "may_claim": may_claim,
        "must_not_claim": must_not_claim,
    })
}

fn parity_backend_summary(
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) -> Value {
    let left_backend = receipt_backend_identity(left);
    let right_backend = receipt_backend_identity(right);
    let fallback_used = left_backend["fallback_used"].as_bool().unwrap_or(true)
        || right_backend["fallback_used"].as_bool().unwrap_or(true);
    let lane_labels = if legacy_scalar_avx2 {
        json!({
            "scalar": left_backend,
            "avx2": right_backend,
        })
    } else {
        json!({
            "left": {
                "label": left_label,
                "backend": left_backend,
            },
            "right": {
                "label": right_label,
                "backend": right_backend,
            },
        })
    };

    json!({
        "requested_backend": common_or_mixed(
            left_backend["requested_backend"].as_str(),
            right_backend["requested_backend"].as_str(),
        ),
        "selected_backend": common_or_mixed(
            left_backend["selected_backend"].as_str(),
            right_backend["selected_backend"].as_str(),
        ),
        "runtime_api": common_or_mixed(
            left_backend["runtime_api"].as_str(),
            right_backend["runtime_api"].as_str(),
        ),
        "fallback_used": fallback_used,
        "lanes": lane_labels,
    })
}

fn receipt_backend_identity(receipt: &Value) -> Value {
    json!({
        "requested_backend": backend_str(receipt, "requested_backend").unwrap_or("unknown"),
        "selected_backend": backend_str(receipt, "selected_backend").unwrap_or("unknown"),
        "runtime_api": backend_str(receipt, "runtime_api").unwrap_or("unknown"),
        "fallback_used": receipt_fallback_used(receipt).unwrap_or(true),
    })
}

fn backend_str<'a>(receipt: &'a Value, field: &str) -> Option<&'a str> {
    first_case_backend_field(receipt, field)
        .and_then(Value::as_str)
        .or_else(|| receipt["backend"][field].as_str())
        .or_else(|| receipt[field].as_str())
}

fn first_case_backend_field<'a>(receipt: &'a Value, field: &str) -> Option<&'a Value> {
    receipt["cases"].as_array()?.iter().find_map(|case| case["backend"].get(field))
}

fn receipt_fallback_used(receipt: &Value) -> Option<bool> {
    let mut saw_evidence = false;
    let mut fallback_used = false;
    for candidate in [
        receipt.get("fallback_used"),
        receipt.get("backend").and_then(|backend| backend.get("fallback_used")),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(value) = candidate.as_bool() {
            saw_evidence = true;
            fallback_used |= value;
        }
    }
    for case in receipt["cases"].as_array().into_iter().flatten() {
        if let Some(value) = case["backend"]["fallback_used"].as_bool() {
            saw_evidence = true;
            fallback_used |= value;
        }
    }
    saw_evidence.then_some(fallback_used)
}

fn common_or_mixed(left: Option<&str>, right: Option<&str>) -> Value {
    match (left, right) {
        (Some(left), Some(right)) if left == right && !left.is_empty() => json!(left),
        (None, None) => json!("unknown"),
        _ => json!("mixed"),
    }
}

fn parity_kernel_summary(
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) -> Value {
    let left_kernels = selected_kernels(left);
    let right_kernels = selected_kernels(right);
    if legacy_scalar_avx2 {
        json!({
            "scalar_selected_kernels": left_kernels,
            "avx2_selected_kernels": right_kernels,
        })
    } else {
        json!({
            "left_label": left_label,
            "right_label": right_label,
            "left_selected_kernels": left_kernels,
            "right_selected_kernels": right_kernels,
        })
    }
}

fn selected_kernels(receipt: &Value) -> Vec<String> {
    receipt["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|case| case["kernel"]["selected_kernel"].as_str())
        .filter(|kernel| !kernel.is_empty())
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn compare_top_level_contract(
    left: &Value,
    right: &Value,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
) {
    if !answer_corpus_artifact_kind_allowed(left["artifact_kind"].as_str())
        || !answer_corpus_artifact_kind_allowed(right["artifact_kind"].as_str())
        || (legacy_scalar_avx2
            && (left["artifact_kind"] != "bitnet_cpu_answer_corpus"
                || right["artifact_kind"] != "bitnet_cpu_answer_corpus"))
    {
        failures.push("artifact_kind_contract");
    }

    let path_mismatch = match (left["model"]["path"].as_str(), right["model"]["path"].as_str()) {
        (Some(left_path), Some(right_path)) if !left_path.is_empty() && !right_path.is_empty() => {
            left_path != right_path
        }
        _ => false,
    };
    if left["model"]["loader_mode"] != "real_gguf"
        || right["model"]["loader_mode"] != "real_gguf"
        || left["model"]["repo"] != right["model"]["repo"]
        || left["model"]["file"] != right["model"]["file"]
        || path_mismatch
    {
        failures.push("model_contract");
    }

    if left["model"]["tokenizer_path"] != right["model"]["tokenizer_path"] {
        failures.push("tokenizer_contract");
    }

    if left["prompt_template"] != right["prompt_template"] {
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
        if left.pointer(field) != right.pointer(field) {
            failures.push("generation_contract");
            break;
        }
    }

    if legacy_scalar_avx2 && (!strict_cpu_backend(left) || !strict_cpu_backend(right)) {
        failures.push("strict_cpu_backend");
    }

    if !legacy_scalar_avx2
        && (!generic_backend_contract(left["backend"].as_object())
            || !generic_backend_contract(right["backend"].as_object()))
    {
        failures.push("backend_contract");
    }
}

fn answer_corpus_artifact_kind_allowed(kind: Option<&str>) -> bool {
    matches!(
        kind,
        Some(
            "bitnet_cpu_answer_corpus"
                | "bitnet_a770_opencl_answer_diagnostic_corpus"
                | "bitnet_cuda_answer_corpus"
                | "bitnet_cuda_answer_diagnostic_corpus"
                | "bitnet_apple_m4_local_answer_corpus"
        )
    )
}

fn strict_cpu_backend(receipt: &Value) -> bool {
    receipt["backend"]["requested_backend"] == "cpu"
        && receipt["backend"]["selected_backend"] == "cpu"
        && receipt["backend"]["runtime_api"] == "cpu"
        && receipt["backend"]["fallback_used"] == false
}

fn generic_backend_contract(backend: Option<&serde_json::Map<String, Value>>) -> bool {
    let Some(backend) = backend else {
        return false;
    };
    !backend.get("requested_backend").and_then(Value::as_str).unwrap_or_default().is_empty()
        && !backend.get("selected_backend").and_then(Value::as_str).unwrap_or_default().is_empty()
        && !backend.get("runtime_api").and_then(Value::as_str).unwrap_or_default().is_empty()
        && backend.get("fallback_used").and_then(Value::as_bool) == Some(false)
}

fn infer_lane_label(receipt: &Value, path: &Path, fallback: &str) -> String {
    receipt["generation"]["requested_cpu_kernel"]
        .as_str()
        .or_else(|| receipt["backend"]["requested_backend"].as_str())
        .or_else(|| receipt["backend"]["selected_backend"].as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .filter(|stem| !stem.is_empty())
                .unwrap_or(fallback)
                .to_string()
        })
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
    left: Option<&Value>,
    right: Option<&Value>,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    first_divergence: &mut Option<Value>,
) -> Value {
    let missing_left_rule =
        if legacy_scalar_avx2 { "case_missing_in_scalar" } else { "case_missing_in_left" };
    let missing_right_rule =
        if legacy_scalar_avx2 { "case_missing_in_avx2" } else { "case_missing_in_right" };
    let Some(left) = left else {
        set_first(
            first_divergence,
            id,
            missing_left_rule,
            None,
            Value::Null,
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return failed_case(id, &[missing_left_rule]);
    };
    let Some(right) = right else {
        set_first(
            first_divergence,
            id,
            missing_right_rule,
            None,
            Value::Null,
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return failed_case(id, &[missing_right_rule]);
    };

    let mut failures = Vec::new();
    if !case_has_execution_evidence(left) || !case_has_execution_evidence(right) {
        set_first(
            first_divergence,
            id,
            "execution_evidence_recorded",
            None,
            case_status_summary(left),
            case_status_summary(right),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        check_equal(
            id,
            "question",
            None,
            &left["question"],
            &right["question"],
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_equal(
            id,
            "status",
            None,
            &left["status"],
            &right["status"],
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_equal(
            id,
            "quality_failed_rules",
            None,
            &left["quality"]["failed_rules"],
            &right["quality"]["failed_rules"],
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        failures.push("execution_evidence_recorded");
        return case_comparison_row(
            id,
            failures,
            left,
            right,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }

    check_case_contract(
        id,
        left_label,
        left,
        legacy_scalar_avx2,
        true,
        left_label,
        right_label,
        &mut failures,
        first_divergence,
    );
    check_case_contract(
        id,
        right_label,
        right,
        legacy_scalar_avx2,
        false,
        left_label,
        right_label,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "question",
        None,
        &left["question"],
        &right["question"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "prompt_template",
        None,
        &left["prompt_template"],
        &right["prompt_template"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "tokenizer_source",
        None,
        &left["tokenizer"]["source"],
        &right["tokenizer"]["source"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "tokenizer_strict",
        None,
        &left["tokenizer"]["strict"],
        &right["tokenizer"]["strict"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "prompt_token_ids",
        None,
        &left["token_ids"]["prompt"],
        &right["token_ids"]["prompt"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "generated_token_ids",
        None,
        &left["token_ids"]["generated"],
        &right["token_ids"]["generated"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "decoded_text",
        None,
        &left["answer"],
        &right["answer"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    if legacy_scalar_avx2 {
        check_kernel_lane(
            id,
            "scalar",
            left,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_kernel_lane(
            id,
            "avx2",
            right,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
    } else {
        check_generic_kernel_recorded(
            id,
            "left",
            left,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_generic_kernel_recorded(
            id,
            "right",
            right,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
    }
    compare_logits_dump(
        id,
        &left["logits_dump"],
        &right["logits_dump"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );

    case_comparison_row(id, failures, left, right, left_label, right_label, legacy_scalar_avx2)
}

fn failed_case(id: &str, failures: &[&str]) -> Value {
    json!({
        "id": id,
        "passed": false,
        "failed_rules": failures,
    })
}

fn case_has_execution_evidence(case: &Value) -> bool {
    case["backend"].is_object()
        && case["kernel"]["selected_kernel"].as_str().is_some_and(|kernel| !kernel.is_empty())
        && case["token_ids"]["prompt"].is_array()
        && case["token_ids"]["generated"].is_array()
}

fn case_comparison_row(
    id: &str,
    failures: Vec<&'static str>,
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) -> Value {
    if legacy_scalar_avx2 {
        json!({
            "id": id,
            "passed": failures.is_empty(),
            "failed_rules": failures,
            "scalar": case_summary(left),
            "avx2": case_summary(right),
        })
    } else {
        json!({
            "id": id,
            "passed": failures.is_empty(),
            "failed_rules": failures,
            "left": labeled_case_summary(left, left_label),
            "right": labeled_case_summary(right, right_label),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn check_case_contract(
    id: &str,
    lane: &str,
    case: &Value,
    legacy_scalar_avx2: bool,
    is_left: bool,
    left_label: &str,
    right_label: &str,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let backend_rule = match (legacy_scalar_avx2, is_left) {
        (true, true) => "scalar_strict_cpu_backend",
        (true, false) => "avx2_strict_cpu_backend",
        (false, true) => "left_backend_contract",
        (false, false) => "right_backend_contract",
    };
    let backend_ok = if legacy_scalar_avx2 {
        case["backend"]["requested_backend"] == "cpu"
            && matches!(case["backend"]["selected_backend"].as_str(), Some("cpu" | "cpu-rust"))
            && case["backend"]["runtime_api"] == "cpu"
            && case["backend"]["fallback_used"] == false
    } else {
        generic_backend_contract(case["backend"].as_object())
    };
    if !backend_ok {
        failures.push(backend_rule);
        set_first(
            first_divergence,
            id,
            backend_rule,
            None,
            case["backend"].clone(),
            json!(lane),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_equal(
    id: &str,
    rule: &'static str,
    step: Option<usize>,
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    if left != right {
        failures.push(rule);
        set_first(
            first_divergence,
            id,
            rule,
            step,
            left.clone(),
            right.clone(),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

fn check_kernel_lane(
    id: &str,
    lane: &'static str,
    case: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let selected = case["kernel"]["selected_kernel"].as_str().unwrap_or_default();
    if !selected.to_ascii_lowercase().contains(lane) {
        let rule = if lane == "scalar" { "scalar_kernel_identity" } else { "avx2_kernel_identity" };
        failures.push(rule);
        set_first(
            first_divergence,
            id,
            rule,
            None,
            json!(selected),
            json!(lane),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

fn check_generic_kernel_recorded(
    id: &str,
    lane: &'static str,
    case: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let selected = case["kernel"]["selected_kernel"].as_str().unwrap_or_default();
    if selected.is_empty() || selected.contains("mock") || selected.contains("diagnostic") {
        let rule = if lane == "left" { "left_kernel_recorded" } else { "right_kernel_recorded" };
        failures.push(rule);
        set_first(
            first_divergence,
            id,
            rule,
            None,
            json!(selected),
            json!(lane),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

fn compare_logits_dump(
    id: &str,
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let left_logits_rule = if legacy_scalar_avx2 {
        "scalar_logits_dump_recorded"
    } else {
        "left_logits_dump_recorded"
    };
    let right_logits_rule =
        if legacy_scalar_avx2 { "avx2_logits_dump_recorded" } else { "right_logits_dump_recorded" };
    let Some(left_steps) = left.as_array() else {
        failures.push(left_logits_rule);
        set_first(
            first_divergence,
            id,
            left_logits_rule,
            None,
            left.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    };
    let Some(right_steps) = right.as_array() else {
        failures.push(right_logits_rule);
        set_first(
            first_divergence,
            id,
            right_logits_rule,
            None,
            right.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    };
    if left_steps.is_empty() {
        failures.push(left_logits_rule);
        set_first(
            first_divergence,
            id,
            left_logits_rule,
            None,
            left.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    }
    if right_steps.is_empty() {
        failures.push(right_logits_rule);
        set_first(
            first_divergence,
            id,
            right_logits_rule,
            None,
            right.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    }
    check_equal(
        id,
        "logits_step_count",
        None,
        &json!(left_steps.len()),
        &json!(right_steps.len()),
        left_label,
        right_label,
        legacy_scalar_avx2,
        failures,
        first_divergence,
    );
    for (step, (left_step, right_step)) in left_steps.iter().zip(right_steps).enumerate() {
        check_equal(
            id,
            "logits_topk",
            Some(step),
            left_step,
            right_step,
            left_label,
            right_label,
            legacy_scalar_avx2,
            failures,
            first_divergence,
        );
    }
}

fn build_logits_topk_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut compared_step_count = 0usize;
    let mut mismatch_count = 0usize;
    let mut same_generated_output_count = 0usize;
    let mut generated_output_divergence_count = 0usize;
    let mut same_chosen_token_count = 0usize;
    let mut different_chosen_token_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut max_common_token_abs_delta = 0.0f64;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                json!({
                    "case_id": id,
                    "classification": "logits_topk_missing_context",
                    "reason": "left_case_missing",
                }),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                json!({
                    "case_id": id,
                    "classification": "logits_topk_missing_context",
                    "reason": "right_case_missing",
                }),
            );
            continue;
        };
        let Some(left_steps) = left_case["logits_dump"].as_array() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                logits_missing_context_row(id, None, "left_logits_dump_missing"),
            );
            continue;
        };
        let Some(right_steps) = right_case["logits_dump"].as_array() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                logits_missing_context_row(id, None, "right_logits_dump_missing"),
            );
            continue;
        };
        if left_steps.is_empty() || right_steps.is_empty() {
            missing_context_count += 1;
            let reason = if left_steps.is_empty() {
                "left_logits_dump_empty"
            } else {
                "right_logits_dump_empty"
            };
            push_limited_row(&mut rows, ROW_LIMIT, logits_missing_context_row(id, None, reason));
            continue;
        }

        for (step, (left_step, right_step)) in left_steps.iter().zip(right_steps).enumerate() {
            compared_step_count += 1;
            if left_step == right_step {
                continue;
            }
            mismatch_count += 1;
            let row =
                logits_topk_frontier_row(id, step, left_case, right_case, left_step, right_step);
            if row["generated_token_ids_match"].as_bool().unwrap_or(false)
                && row["decoded_text_match"].as_bool().unwrap_or(false)
            {
                same_generated_output_count += 1;
            } else {
                generated_output_divergence_count += 1;
            }
            if row["same_chosen_id"].as_bool().unwrap_or(false) {
                same_chosen_token_count += 1;
            } else {
                different_chosen_token_count += 1;
            }
            max_common_token_abs_delta = max_common_token_abs_delta
                .max(row["max_common_token_abs_delta"].as_f64().unwrap_or(0.0));
            push_limited_row(&mut rows, ROW_LIMIT, row);
        }

        if left_steps.len() != right_steps.len() {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                json!({
                    "case_id": id,
                    "classification": "logits_topk_missing_context",
                    "reason": "logits_step_count_mismatch",
                    "left_step_count": left_steps.len(),
                    "right_step_count": right_steps.len(),
                }),
            );
        }
    }

    let classification = if missing_context_count > 0 {
        "logits_topk_frontier_missing_context"
    } else if generated_output_divergence_count > 0 {
        "logits_topk_frontier_generated_output_divergence"
    } else if different_chosen_token_count > 0 {
        "logits_topk_frontier_different_chosen_same_output"
    } else if mismatch_count > 0 {
        "logits_topk_frontier_same_chosen_same_output"
    } else {
        "logits_topk_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "compared_step_count": compared_step_count,
        "logits_topk_mismatch_count": mismatch_count,
        "same_generated_output_count": same_generated_output_count,
        "generated_output_divergence_count": generated_output_divergence_count,
        "same_chosen_token_count": same_chosen_token_count,
        "different_chosen_token_count": different_chosen_token_count,
        "missing_context_count": missing_context_count,
        "max_common_token_abs_delta": max_common_token_abs_delta,
        "rows_truncated": mismatch_count.saturating_add(missing_context_count) > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn push_limited_row(rows: &mut Vec<Value>, row_limit: usize, row: Value) {
    if rows.len() < row_limit {
        rows.push(row);
    }
}

fn logits_missing_context_row(id: &str, step: Option<usize>, reason: &str) -> Value {
    json!({
        "case_id": id,
        "step": step,
        "classification": "logits_topk_missing_context",
        "reason": reason,
    })
}

fn logits_topk_frontier_row(
    id: &str,
    step: usize,
    left_case: &Value,
    right_case: &Value,
    left_step: &Value,
    right_step: &Value,
) -> Value {
    let left_chosen_id = left_step["chosen_id"].as_u64();
    let right_chosen_id = right_step["chosen_id"].as_u64();
    let generated_token_ids_match =
        left_case["token_ids"]["generated"] == right_case["token_ids"]["generated"];
    let decoded_text_match = left_case["answer"] == right_case["answer"];
    let same_chosen_id = left_chosen_id == right_chosen_id;
    let (common_top_token_count, max_common_delta, left_only, right_only) =
        top_logits_common_delta(&left_step["top_logits"], &right_step["top_logits"]);

    let classification = if !generated_token_ids_match || !decoded_text_match {
        "logits_topk_generated_output_divergence"
    } else if same_chosen_id {
        "logits_topk_same_chosen_same_output"
    } else {
        "logits_topk_different_chosen_same_output"
    };

    json!({
        "case_id": id,
        "step": step,
        "classification": classification,
        "left_chosen_id": left_chosen_id,
        "right_chosen_id": right_chosen_id,
        "same_chosen_id": same_chosen_id,
        "generated_token_ids_match": generated_token_ids_match,
        "decoded_text_match": decoded_text_match,
        "first_different_rank": first_different_topk_rank(&left_step["top_logits"], &right_step["top_logits"]),
        "left_topk_count": left_step["top_logits"].as_array().map(Vec::len),
        "right_topk_count": right_step["top_logits"].as_array().map(Vec::len),
        "common_top_token_count": common_top_token_count,
        "max_common_token_abs_delta": max_common_delta,
        "left_only_top_token_ids": left_only,
        "right_only_top_token_ids": right_only,
    })
}

fn top_logits_common_delta(left: &Value, right: &Value) -> (usize, f64, Vec<u64>, Vec<u64>) {
    let left_logits = top_logits_by_token(left);
    let right_logits = top_logits_by_token(right);
    let mut common_count = 0usize;
    let mut max_delta = 0.0f64;
    for (token, left_logit) in &left_logits {
        if let Some(right_logit) = right_logits.get(token) {
            common_count += 1;
            max_delta = max_delta.max((left_logit - right_logit).abs());
        }
    }
    let left_only = left_logits
        .keys()
        .filter(|token| !right_logits.contains_key(token))
        .take(3)
        .copied()
        .collect();
    let right_only = right_logits
        .keys()
        .filter(|token| !left_logits.contains_key(token))
        .take(3)
        .copied()
        .collect();
    (common_count, max_delta, left_only, right_only)
}

fn top_logits_by_token(top_logits: &Value) -> BTreeMap<u64, f64> {
    top_logits
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| Some((entry["token_id"].as_u64()?, entry["logit"].as_f64()?)))
        .collect()
}

fn first_different_topk_rank(left: &Value, right: &Value) -> Value {
    let left = left.as_array().map(Vec::as_slice).unwrap_or(&[]);
    let right = right.as_array().map(Vec::as_slice).unwrap_or(&[]);
    for index in 0..left.len().max(right.len()) {
        let left_token = left.get(index).and_then(|entry| entry["token_id"].as_u64());
        let right_token = right.get(index).and_then(|entry| entry["token_id"].as_u64());
        if left_token != right_token {
            return json!(index);
        }
    }
    Value::Null
}

fn build_generated_output_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut generated_output_mismatch_count = 0usize;
    let mut decoded_text_mismatch_count = 0usize;
    let mut mismatch_with_logit_context_count = 0usize;
    let mut missing_logit_context_count = 0usize;
    let mut missing_context_count = 0usize;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "left_case_missing"),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "right_case_missing"),
            );
            continue;
        };

        let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "left_generated_token_ids_missing"),
            );
            continue;
        };
        let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "right_generated_token_ids_missing"),
            );
            continue;
        };

        let decoded_text_match = left_case["answer"] == right_case["answer"];
        if !decoded_text_match {
            decoded_text_mismatch_count += 1;
        }

        let Some(first_mismatch_index) =
            first_different_token_index(&left_generated, &right_generated)
        else {
            continue;
        };

        generated_output_mismatch_count += 1;
        let row = generated_output_frontier_row(id, left_case, right_case, first_mismatch_index);
        if row["has_logit_context_at_first_mismatch"].as_bool().unwrap_or(false) {
            mismatch_with_logit_context_count += 1;
        } else {
            missing_logit_context_count += 1;
        }
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_frontier_missing_context"
    } else if missing_logit_context_count > 0 {
        "generated_output_frontier_first_mismatch_missing_logit_context"
    } else if generated_output_mismatch_count > 0 {
        "generated_output_frontier_first_mismatch_has_logit_context"
    } else {
        "generated_output_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "generated_output_mismatch_count": generated_output_mismatch_count,
        "decoded_text_mismatch_count": decoded_text_mismatch_count,
        "mismatch_with_logit_context_count": mismatch_with_logit_context_count,
        "missing_logit_context_count": missing_logit_context_count,
        "missing_context_count": missing_context_count,
        "rows_truncated": generated_output_mismatch_count.saturating_add(missing_context_count) > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_missing_context",
        "reason": reason,
    })
}

fn generated_output_frontier_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
    first_mismatch_index: usize,
) -> Value {
    let left_generated = token_id_vec(&left_case["token_ids"]["generated"]).unwrap_or_default();
    let right_generated = token_id_vec(&right_case["token_ids"]["generated"]).unwrap_or_default();
    let left_steps = left_case["logits_dump"].as_array();
    let right_steps = right_case["logits_dump"].as_array();
    let left_step = left_steps.and_then(|steps| steps.get(first_mismatch_index));
    let right_step = right_steps.and_then(|steps| steps.get(first_mismatch_index));
    let has_logit_context_at_first_mismatch = left_step.is_some() && right_step.is_some();
    let left_chosen_id = left_step.and_then(|step| step["chosen_id"].as_u64());
    let right_chosen_id = right_step.and_then(|step| step["chosen_id"].as_u64());
    let same_chosen_id_at_first_mismatch = match (left_chosen_id, right_chosen_id) {
        (Some(left), Some(right)) => json!(left == right),
        _ => Value::Null,
    };
    let classification = if has_logit_context_at_first_mismatch {
        "generated_output_first_mismatch_has_logit_context"
    } else {
        "generated_output_first_mismatch_missing_logit_context"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_generated_len": left_generated.len(),
        "right_generated_len": right_generated.len(),
        "decoded_text_match": left_case["answer"] == right_case["answer"],
        "left_logits_step_count": left_steps.map(Vec::len),
        "right_logits_step_count": right_steps.map(Vec::len),
        "has_logit_context_at_first_mismatch": has_logit_context_at_first_mismatch,
        "left_chosen_id_at_first_mismatch": left_chosen_id,
        "right_chosen_id_at_first_mismatch": right_chosen_id,
        "same_chosen_id_at_first_mismatch": same_chosen_id_at_first_mismatch,
        "first_different_rank_at_first_mismatch": match (left_step, right_step) {
            (Some(left), Some(right)) => {
                first_different_topk_rank(&left["top_logits"], &right["top_logits"])
            }
            _ => Value::Null,
        },
    })
}

fn token_id_vec(value: &Value) -> Option<Vec<u64>> {
    value.as_array()?.iter().map(Value::as_u64).collect()
}

fn first_different_token_index(left: &[u64], right: &[u64]) -> Option<usize> {
    for index in 0..left.len().max(right.len()) {
        if left.get(index) != right.get(index) {
            return Some(index);
        }
    }
    None
}

fn case_summary(case: &Value) -> Value {
    json!({
        "status": case["status"],
        "quality_passed": case["quality"]["passed"],
        "quality_failed_rules": case["quality"]["failed_rules"],
        "selected_kernel": case["kernel"]["selected_kernel"],
        "prompt_token_ids": case["token_ids"]["prompt"],
        "generated_token_ids": case["token_ids"]["generated"],
        "answer": case["answer"],
        "logits_steps": case["logits_dump"].as_array().map(Vec::len),
    })
}

fn labeled_case_summary(case: &Value, label: &str) -> Value {
    let mut summary = case_summary(case);
    summary["label"] = json!(label);
    summary["backend"] = case["backend"].clone();
    summary
}

fn case_status_summary(case: &Value) -> Value {
    json!({
        "status": case["status"],
        "quality_failed_rules": case["quality"]["failed_rules"],
        "run_receipt_path": case["run_receipt_path"],
        "exit_code": case["exit_code"],
        "child_process": case["child_process"],
        "child_invocation": {
            "expected_receipt_path": case["child_invocation"]["expected_receipt_path"],
            "timeout_seconds": case["child_invocation"]["timeout_seconds"],
        },
        "reason": case["reason"],
    })
}

#[allow(clippy::too_many_arguments)]
fn set_first(
    first_divergence: &mut Option<Value>,
    id: &str,
    kind: &'static str,
    step: Option<usize>,
    left: Value,
    right: Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) {
    if first_divergence.is_none() {
        let divergence = if legacy_scalar_avx2 {
            json!({
                "case_id": id,
                "kind": kind,
                "step": step,
                "scalar": left,
                "avx2": right,
                "scope": divergence_scope(kind),
            })
        } else {
            json!({
                "case_id": id,
                "kind": kind,
                "step": step,
                "left_label": left_label,
                "right_label": right_label,
                "left": left,
                "right": right,
                "scope": divergence_scope(kind),
            })
        };
        *first_divergence = Some(divergence);
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

    fn logits_same_chosen_drift() -> Value {
        json!([{
            "step": 0,
            "chosen_id": 4,
            "top_logits": [
                { "token_id": 4, "logit": 9.75 },
                { "token_id": 5, "logit": 1.5 },
                { "token_id": 6, "logit": 0.25 }
            ]
        }])
    }

    fn logits_different_chosen() -> Value {
        json!([{
            "step": 0,
            "chosen_id": 5,
            "top_logits": [
                { "token_id": 5, "logit": 10.0 },
                { "token_id": 4, "logit": 9.0 }
            ]
        }])
    }

    fn logits_for_chosen(chosen: &[u64]) -> Value {
        Value::Array(
            chosen
                .iter()
                .enumerate()
                .map(|(step, token)| {
                    json!({
                        "step": step,
                        "chosen_id": token,
                        "top_logits": [
                            { "token_id": token, "logit": 10.0 },
                            { "token_id": 999, "logit": 1.0 }
                        ]
                    })
                })
                .collect(),
        )
    }

    fn cuda_receipt(generated: &[u64], answer: &str, logits: Value) -> Value {
        let mut receipt = receipt("qk256_gemv_cuda", generated, answer, logits);
        receipt["artifact_kind"] = json!("bitnet_cuda_answer_diagnostic_corpus");
        receipt["backend"] = json!({
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false
        });
        receipt["cases"][0]["backend"] = receipt["backend"].clone();
        receipt
    }

    fn a770_receipt(generated: &[u64], answer: &str, logits: Value) -> Value {
        let mut receipt = receipt(
            "a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate",
            generated,
            answer,
            logits,
        );
        receipt["artifact_kind"] = json!("bitnet_a770_opencl_answer_diagnostic_corpus");
        receipt["backend"] = json!({
            "requested_backend": "intel-a770-opencl",
            "selected_backend": "intel-a770-opencl",
            "runtime_api": "opencl",
            "fallback_used": false
        });
        receipt["cases"][0]["backend"] = receipt["backend"].clone();
        receipt
    }

    fn build_legacy_report(scalar: &Value, avx2: &Value) -> Value {
        build_answer_parity_receipt(
            Path::new("scalar.json"),
            scalar,
            Path::new("avx2.json"),
            avx2,
            "scalar",
            "avx2",
            true,
            None,
            None,
            None,
        )
    }

    fn build_generic_report(left: &Value, right: &Value) -> Value {
        build_answer_parity_receipt(
            Path::new("left.json"),
            left,
            Path::new("right.json"),
            right,
            "scalar",
            "cuda",
            false,
            None,
            None,
            None,
        )
    }

    #[test]
    fn parity_receipt_passes_matching_scalar_and_avx2_runs() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let avx2 = receipt("i2_s-avx2-reference", &[4], "4", logits());

        let report = build_legacy_report(&scalar, &avx2);

        assert_eq!(report["artifact_kind"], "bitnet_cpu_answer_parity");
        assert_eq!(report["claim"], "scalar_avx2_full_decode_answer_parity");
        assert_eq!(report["inputs"]["scalar_receipt_path"], "scalar.json");
        assert!(report["inputs"]["left_receipt_path"].is_null());
        assert_eq!(report["requested_backend"], "cpu");
        assert_eq!(report["selected_backend"], "cpu-rust");
        assert_eq!(report["runtime_api"], "cpu");
        assert_eq!(report["fallback_used"], false);
        assert_eq!(report["backend"]["fallback_used"], false);
        assert_eq!(report["backend"]["lanes"]["scalar"]["selected_backend"], "cpu-rust");
        assert_eq!(report["backend"]["lanes"]["avx2"]["selected_backend"], "cpu-rust");
        assert_eq!(report["kernel"]["scalar_selected_kernels"], json!(["i2_s-scalar-reference"]));
        assert_eq!(report["kernel"]["avx2_selected_kernels"], json!(["i2_s-avx2-reference"]));
        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["cases"][0]["passed"], true);
        assert!(report["cases"][0]["left"].is_null());
        assert!(report["summary"]["first_divergence"].is_null());
    }

    #[test]
    fn parity_receipt_passes_matching_shared_quality_failures() {
        let mut scalar = receipt("i2_s-scalar-reference", &[999], "bad", logits());
        let mut avx2 = receipt("i2_s-avx2-reference", &[999], "bad", logits());
        scalar["cases"][0]["status"] = json!("quality_failed");
        scalar["cases"][0]["quality"] = json!({
            "passed": false,
            "failed_rules": ["exact_trimmed"]
        });
        avx2["cases"][0]["status"] = json!("quality_failed");
        avx2["cases"][0]["quality"] = json!({
            "passed": false,
            "failed_rules": ["exact_trimmed"]
        });

        let report = build_answer_parity_receipt(
            Path::new("scalar.json"),
            &scalar,
            Path::new("avx2.json"),
            &avx2,
            "scalar",
            "avx2",
            true,
            Some("intel-258v"),
            Some(Path::new("platform-probe.json")),
            None,
        );

        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["machine"]["machine_id"], "intel-258v");
        assert_eq!(report["cases"][0]["passed"], true);
        assert_eq!(report["cases"][0]["scalar"]["quality_passed"], false);
        assert!(report["summary"]["first_divergence"].is_null());
    }

    #[test]
    fn parity_receipt_records_first_generated_token_divergence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let avx2 = receipt("i2_s-avx2-reference", &[5], "5", logits());

        let report = build_legacy_report(&scalar, &avx2);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["cases"][0]["passed"], false);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(report["summary"]["first_divergence"]["scope"], "decode_or_sampler");
    }

    #[test]
    fn parity_receipt_requires_logit_evidence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", Value::Null);
        let avx2 = receipt("i2_s-avx2-reference", &[4], "4", logits());

        let report = build_legacy_report(&scalar, &avx2);

        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "scalar_logits_dump_recorded"));
    }

    #[test]
    fn generic_parity_accepts_matching_cpu_and_cuda_answer_corpus_receipts() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let cuda = cuda_receipt(&[4], "4", logits());

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["artifact_kind"], "bitnet_answer_corpus_parity");
        assert_eq!(report["claim"], "full_decode_answer_parity_diagnostic");
        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["inputs"]["left_label"], "scalar");
        assert_eq!(report["inputs"]["right_label"], "cuda");
        assert_eq!(report["requested_backend"], "mixed");
        assert_eq!(report["selected_backend"], "mixed");
        assert_eq!(report["runtime_api"], "mixed");
        assert_eq!(report["fallback_used"], false);
        assert_eq!(report["backend"]["lanes"]["left"]["label"], "scalar");
        assert_eq!(report["backend"]["lanes"]["right"]["label"], "cuda");
        assert_eq!(report["kernel"]["left_selected_kernels"], json!(["i2_s-scalar-reference"]));
        assert_eq!(report["kernel"]["right_selected_kernels"], json!(["qk256_gemv_cuda"]));
        assert_eq!(report["cases"][0]["passed"], true);
        assert_eq!(report["cases"][0]["right"]["selected_kernel"], "qk256_gemv_cuda");
        assert!(report["cases"][0]["scalar"].is_null());
    }

    #[test]
    fn generic_parity_accepts_matching_cpu_and_a770_answer_corpus_receipts() {
        let scalar = receipt("i2_s-avx2-reference", &[4], "4", logits());
        let a770 = a770_receipt(&[4], "4", logits());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(report["artifact_kind"], "bitnet_answer_corpus_parity");
        assert_eq!(report["summary"]["failed"], 0);
        assert!(
            report["shared_contract"]["failed_rules"]
                .as_array()
                .is_some_and(|shared| !shared.iter().any(|rule| rule == "artifact_kind_contract"))
        );
        assert_eq!(
            report["backend"]["lanes"]["right"]["backend"]["selected_backend"],
            "intel-a770-opencl"
        );
        assert_eq!(
            report["kernel"]["right_selected_kernels"],
            json!(["a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate"])
        );
    }

    #[test]
    fn generic_parity_summarizes_same_output_logits_topk_frontier() {
        let scalar = receipt("i2_s-avx2-reference", &[4], "4", logits());
        let a770 = a770_receipt(&[4], "4", logits_same_chosen_drift());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "logits_topk");
        assert_eq!(
            report["logits_topk_frontier"]["classification"],
            "logits_topk_frontier_same_chosen_same_output"
        );
        assert_eq!(report["logits_topk_frontier"]["logits_topk_mismatch_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["same_generated_output_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["generated_output_divergence_count"], 0);
        assert_eq!(report["logits_topk_frontier"]["same_chosen_token_count"], 1);
        assert_eq!(
            report["logits_topk_frontier"]["rows"][0]["classification"],
            "logits_topk_same_chosen_same_output"
        );
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["same_chosen_id"], true);
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["generated_token_ids_match"], true);
    }

    #[test]
    fn generic_parity_summarizes_generated_output_logits_topk_frontier() {
        let scalar = receipt("i2_s-avx2-reference", &[4], "4", logits());
        let a770 = a770_receipt(&[5], "5", logits_different_chosen());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(
            report["logits_topk_frontier"]["classification"],
            "logits_topk_frontier_generated_output_divergence"
        );
        assert_eq!(report["logits_topk_frontier"]["logits_topk_mismatch_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["generated_output_divergence_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["different_chosen_token_count"], 1);
        assert_eq!(
            report["logits_topk_frontier"]["rows"][0]["classification"],
            "logits_topk_generated_output_divergence"
        );
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["same_chosen_id"], false);
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["generated_token_ids_match"], false);
    }

    #[test]
    fn generic_parity_summarizes_generated_output_frontier_with_logit_context() {
        let scalar =
            receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", logits_for_chosen(&[4, 5, 6]));
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_for_chosen(&[4, 5, 7]));

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(
            report["generated_output_frontier"]["classification"],
            "generated_output_frontier_first_mismatch_has_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["generated_output_mismatch_count"], 1);
        assert_eq!(report["generated_output_frontier"]["mismatch_with_logit_context_count"], 1);
        assert_eq!(report["generated_output_frontier"]["missing_logit_context_count"], 0);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["classification"],
            "generated_output_first_mismatch_has_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["rows"][0]["first_mismatch_index"], 2);
        assert_eq!(report["generated_output_frontier"]["rows"][0]["left_token_id"], 6);
        assert_eq!(report["generated_output_frontier"]["rows"][0]["right_token_id"], 7);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["has_logit_context_at_first_mismatch"],
            true
        );
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["same_chosen_id_at_first_mismatch"],
            false
        );
    }

    #[test]
    fn generic_parity_summarizes_generated_output_frontier_missing_logit_context() {
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", logits());
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(
            report["generated_output_frontier"]["classification"],
            "generated_output_frontier_first_mismatch_missing_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["generated_output_mismatch_count"], 1);
        assert_eq!(report["generated_output_frontier"]["mismatch_with_logit_context_count"], 0);
        assert_eq!(report["generated_output_frontier"]["missing_logit_context_count"], 1);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["classification"],
            "generated_output_first_mismatch_missing_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["rows"][0]["first_mismatch_index"], 2);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["has_logit_context_at_first_mismatch"],
            false
        );
        assert!(
            report["generated_output_frontier"]["rows"][0]["same_chosen_id_at_first_mismatch"]
                .is_null()
        );
    }

    #[test]
    fn generic_parity_records_left_right_divergence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let cuda = cuda_receipt(&[5], "5", logits());

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(report["summary"]["first_divergence"]["left_label"], "scalar");
        assert_eq!(report["summary"]["first_divergence"]["right_label"], "cuda");
        assert!(report["summary"]["first_divergence"]["left"].is_array());
        assert!(report["summary"]["first_divergence"]["scalar"].is_null());
    }

    #[test]
    fn legacy_parity_still_rejects_cuda_receipts() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let cuda = cuda_receipt(&[4], "4", logits());

        let report = build_legacy_report(&scalar, &cuda);

        assert_ne!(report["summary"]["failed"], 0);
        let shared = report["shared_contract"]["failed_rules"].as_array().unwrap();
        assert!(shared.iter().any(|rule| rule == "artifact_kind_contract"));
        assert!(shared.iter().any(|rule| rule == "strict_cpu_backend"));
    }

    #[test]
    fn generic_parity_classifies_matching_command_failures_as_missing_execution_evidence() {
        let mut scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let mut cuda = cuda_receipt(&[4], "4", logits());
        let failed_case = json!({
            "id": "math",
            "question": "Answer with a single digit: 2+2=",
            "status": "command_failed",
            "exit_code": 7,
            "run_receipt_path": "target/bitnet/receipts/math.json",
            "quality": {
                "passed": false,
                "failed_rules": ["command_failed"]
            }
        });
        scalar["cases"][0] = failed_case.clone();
        cuda["cases"][0] = failed_case;

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "execution_evidence_recorded");
        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "execution_evidence_recorded"));
        assert!(!failed.iter().any(|rule| rule == "left_backend_contract"));
        assert!(!failed.iter().any(|rule| rule == "right_backend_contract"));
    }

    #[test]
    fn generic_parity_prioritizes_missing_child_receipt_over_status_difference() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let mut cuda = cuda_receipt(&[4], "4", logits());
        cuda["cases"][0] = json!({
            "id": "math",
            "question": "Answer with a single digit: 2+2=",
            "status": "command_failed",
            "exit_code": -1_073_740_791,
            "run_receipt_path": "target/bitnet/receipts/math.json",
            "quality": {
                "passed": false,
                "failed_rules": ["command_failed"]
            },
            "child_process": {
                "success": false,
                "timed_out": false,
                "exit_code": -1_073_740_791,
                "exit_code_hex": "0xC0000409",
                "crash_class": "windows_stack_buffer_overrun_or_fast_fail",
                "receipt_observed": false
            },
            "child_invocation": {
                "expected_receipt_path": "target/bitnet/receipts/math.json",
                "timeout_seconds": 120
            }
        });

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "execution_evidence_recorded");
        assert_eq!(
            report["summary"]["first_divergence"]["right"]["child_process"]["crash_class"],
            "windows_stack_buffer_overrun_or_fast_fail"
        );
        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "status"));
        assert!(failed.iter().any(|rule| rule == "execution_evidence_recorded"));
    }
}
