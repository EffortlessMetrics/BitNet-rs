use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::fs;
use std::path::Path;

const CRITICAL_A770_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

const VALID_STATUSES: &[&str] = &[
    "unsupported",
    "diagnostic",
    "load_proven",
    "quality_proven",
    "performance_proven",
    "resident_proven",
    "complete",
];

const EVIDENCE_REQUIRED_STATUSES: &[&str] =
    &["load_proven", "quality_proven", "performance_proven", "resident_proven", "complete"];

const A770_PROMOTED_STATUSES: &[&str] = &["performance_proven", "resident_proven", "complete"];

pub fn verify(
    ledger_path: &Path,
    a770_capability_matrix: &Path,
    llm_experience_receipt: Option<&Path>,
    format: &str,
) -> Result<()> {
    let ledger = read_json(ledger_path)?;
    let matrix = read_json(a770_capability_matrix)?;
    let experience = llm_experience_receipt.map(read_json).transpose()?;
    let report = build_verify_report(
        ledger_path,
        &ledger,
        a770_capability_matrix,
        &matrix,
        llm_experience_receipt,
        experience.as_ref(),
    );
    emit_value(&report, format)?;
    if !report["passed"].as_bool().unwrap_or(false) {
        bail!("claims verify failed: {}", report["failures"].clone());
    }
    Ok(())
}

pub fn docs(ledger_path: &Path, output: &Path, check: bool, format: &str) -> Result<()> {
    let ledger = read_json(ledger_path)?;
    let markdown = render_docs(&ledger);
    if check {
        let existing =
            fs::read_to_string(output).with_context(|| format!("reading {}", output.display()))?;
        if normalize_newlines(&existing) != normalize_newlines(&markdown) {
            let report = json!({
                "diagnostic": "claims_docs",
                "producer": "cargo xtask claims docs",
                "passed": false,
                "check": true,
                "output": output.display().to_string(),
                "failures": ["claim docs are stale"],
            });
            emit_value(&report, format)?;
            bail!("claim docs are stale: {}", output.display());
        }
    } else {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, markdown).with_context(|| format!("writing {}", output.display()))?;
    }
    let report = json!({
        "diagnostic": "claims_docs",
        "producer": "cargo xtask claims docs",
        "passed": true,
        "check": check,
        "ledger": ledger_path.display().to_string(),
        "output": output.display().to_string(),
        "claim_count": ledger.pointer("/claims").and_then(Value::as_array).map_or(0, Vec::len),
    });
    emit_value(&report, format)?;
    Ok(())
}

fn build_verify_report(
    ledger_path: &Path,
    ledger: &Value,
    matrix_path: &Path,
    matrix: &Value,
    llm_experience_receipt_path: Option<&Path>,
    llm_experience_receipt: Option<&Value>,
) -> Value {
    let mut failures = Vec::new();
    let claims = ledger.pointer("/claims").and_then(Value::as_array);
    let Some(claims) = claims else {
        return json!({
            "diagnostic": "claims_verify",
            "producer": "cargo xtask claims verify",
            "ledger": ledger_path.display().to_string(),
            "a770_capability_matrix": matrix_path.display().to_string(),
            "passed": false,
            "claim_count": 0,
            "failures": ["ledger missing /claims array"],
        });
    };

    let a770_claimable_kernel_count = claimable_kernel_count(matrix);
    let experience_status = experience_status(llm_experience_receipt_path, llm_experience_receipt);
    let mut promoted_a770_claims = Vec::new();
    let mut promoted_a770_experience_claims = Vec::new();
    for claim in claims {
        let claim_id = str_at(claim, "/claim_id").unwrap_or("");
        let status = str_at(claim, "/status").unwrap_or("");
        let scope = str_at(claim, "/scope").unwrap_or("");
        if claim_id.is_empty() {
            failures.push("claim missing claim_id".to_string());
        }
        if !VALID_STATUSES.contains(&status) {
            failures.push(format!("{claim_id}: invalid status {status}"));
        }
        let evidence = claim.pointer("/evidence").and_then(Value::as_array);
        if EVIDENCE_REQUIRED_STATUSES.contains(&status)
            && evidence.is_none_or(|items| items.is_empty())
        {
            failures.push(format!("{claim_id}: status {status} requires evidence"));
        }

        let not_claims = array_strings(claim, "/not_claims");
        if claim_id.starts_with("a770.") || scope.to_ascii_lowercase().contains("a770") {
            for not_claim in CRITICAL_A770_NOT_CLAIMS {
                if !not_claims.iter().any(|item| item == not_claim) {
                    failures.push(format!("{claim_id}: missing critical not-claim {not_claim}"));
                }
            }
        }

        if claim_id.contains("selected_attention")
            && !matches!(status, "unsupported" | "diagnostic")
        {
            failures.push(format!("{claim_id}: selected attention cannot be promoted by ledger"));
        }
        if claim_id.contains("full_residency") && !matches!(status, "unsupported" | "diagnostic") {
            failures.push(format!("{claim_id}: full residency cannot be promoted by ledger"));
        }
        if claim_id.starts_with("a770.") && A770_PROMOTED_STATUSES.contains(&status) {
            promoted_a770_claims.push(claim_id.to_string());
        }
        if claim_id == "a770.bitnet.trusted_partial_experience"
            && A770_PROMOTED_STATUSES.contains(&status)
        {
            promoted_a770_experience_claims.push(claim_id.to_string());
        }
    }

    if a770_claimable_kernel_count == 0 && !promoted_a770_claims.is_empty() {
        failures.push(format!(
            "A770 promoted claims require claimable A770 kernels; promoted={}",
            promoted_a770_claims.join(", ")
        ));
    }
    if !promoted_a770_experience_claims.is_empty()
        && !bool_at(&experience_status, "/claim_allowed").unwrap_or(false)
    {
        failures.push(format!(
            "A770 trusted experience promotion requires claimable llm_experience_run receipt; promoted={}",
            promoted_a770_experience_claims.join(", ")
        ));
    }

    json!({
        "diagnostic": "claims_verify",
        "producer": "cargo xtask claims verify",
        "ledger": ledger_path.display().to_string(),
        "a770_capability_matrix": matrix_path.display().to_string(),
        "llm_experience_receipt": experience_status,
        "passed": failures.is_empty(),
        "claim_count": claims.len(),
        "a770_claimable_kernel_count": a770_claimable_kernel_count,
        "promoted_a770_claims": promoted_a770_claims,
        "failures": failures,
    })
}

fn render_docs(ledger: &Value) -> String {
    let mut output = String::new();
    output.push_str("# Claim Ledger\n\n");
    output.push_str("Generated from `ci/claims/claim-ledger.json`.\n\n");
    output.push_str("| Claim | Status | Scope | Blocker |\n");
    output.push_str("| --- | --- | --- | --- |\n");
    if let Some(claims) = ledger.pointer("/claims").and_then(Value::as_array) {
        for claim in claims {
            output.push_str(&format!(
                "| `{}` | `{}` | {} | {} |\n",
                str_at(claim, "/claim_id").unwrap_or("unknown"),
                str_at(claim, "/status").unwrap_or("unknown"),
                escape_markdown_cell(str_at(claim, "/scope").unwrap_or("")),
                escape_markdown_cell(str_at(claim, "/blocker").unwrap_or(""))
            ));
        }
    }
    output.push_str("\n## A770 Not Claims\n\n");
    for not_claim in CRITICAL_A770_NOT_CLAIMS {
        output.push_str(&format!("- `{not_claim}`\n"));
    }
    output
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn emit_value(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(value)?),
        "human" => {
            println!("diagnostic: {}", str_at(value, "/diagnostic").unwrap_or("claims"));
            if let Some(passed) = value.pointer("/passed").and_then(Value::as_bool) {
                println!("passed: {passed}");
            }
            if let Some(claim_count) = value.pointer("/claim_count").and_then(Value::as_u64) {
                println!("claim_count: {claim_count}");
            }
            if let Some(failures) = value.pointer("/failures") {
                println!("failures: {failures}");
            }
        }
        other => bail!("unsupported claims output format: {other}"),
    }
    Ok(())
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn array_strings(value: &Value, pointer: &str) -> Vec<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect())
        .unwrap_or_default()
}

fn claimable_kernel_count(matrix: &Value) -> u64 {
    if let Some(count) = matrix.pointer("/claimable_kernel_count").and_then(Value::as_u64) {
        return count;
    }

    matrix
        .pointer("/kernels")
        .and_then(Value::as_array)
        .map(|kernels| {
            kernels
                .iter()
                .filter(|kernel| {
                    str_at(kernel, "/status").is_some_and(|status| {
                        matches!(
                            status,
                            "quality_proven"
                                | "performance_proven"
                                | "resident_proven"
                                | "complete"
                        )
                    })
                })
                .count() as u64
        })
        .unwrap_or(0)
}

fn experience_status(path: Option<&Path>, receipt: Option<&Value>) -> Value {
    let Some(receipt) = receipt else {
        return json!({
            "present": false,
            "path": path.map(|path| path.display().to_string()),
            "claim_allowed": false,
            "classification": "missing",
            "receipt_type": Value::Null,
            "critical_not_claims_present": false,
        });
    };
    let not_claims = array_strings(receipt, "/not_claims");
    let critical_not_claims_present = CRITICAL_A770_NOT_CLAIMS
        .iter()
        .all(|not_claim| not_claims.iter().any(|item| item == not_claim));
    let receipt_type = str_at(receipt, "/receipt_type").unwrap_or("");
    let claim_allowed = receipt_type == "llm_experience_run"
        && bool_at(receipt, "/claim_gate/claim_allowed").unwrap_or(false)
        && critical_not_claims_present;
    json!({
        "present": true,
        "path": path.map(|path| path.display().to_string()),
        "receipt_type": receipt_type,
        "claim_allowed": claim_allowed,
        "raw_claim_allowed": bool_at(receipt, "/claim_gate/claim_allowed").unwrap_or(false),
        "classification": str_at(receipt, "/claim_gate/classification").unwrap_or("diagnostic_only"),
        "quality_passed": bool_at(receipt, "/quality/passed").unwrap_or(false),
        "fallback_used": bool_at(receipt, "/backend/fallback_used").unwrap_or(true),
        "route_verified": bool_at(receipt, "/kernel_route/route_verified").unwrap_or(false),
        "model_contract": str_at(receipt, "/model/contract").unwrap_or(""),
        "selected_backend": str_at(receipt, "/backend/selected_backend").unwrap_or(""),
        "critical_not_claims_present": critical_not_claims_present,
    })
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn escape_markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn normalize_newlines(value: &str) -> String {
    value.replace("\r\n", "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger(status: &str) -> Value {
        json!({
            "schema_version": 1,
            "claims": [
                {
                    "claim_id": "a770.bitnet.trusted_partial_experience",
                    "scope": "BitNet b1.58 i2_s on AMD 5700X + Intel Arc A770",
                    "status": status,
                    "evidence": [],
                    "blocker": "clean claim-grade parent benchmark and history are not present",
                    "not_claims": CRITICAL_A770_NOT_CLAIMS
                }
            ]
        })
    }

    fn matrix(claimable_kernel_count: u64) -> Value {
        json!({
            "claimable_kernel_count": claimable_kernel_count
        })
    }

    #[test]
    fn diagnostic_a770_claim_passes_without_claimable_kernels() {
        let report = build_verify_report(
            Path::new("claims.json"),
            &ledger("diagnostic"),
            Path::new("matrix.json"),
            &matrix(0),
            None,
            None,
        );
        assert_eq!(report["passed"], true);
    }

    #[test]
    fn performance_a770_claim_requires_claimable_kernels() {
        let report = build_verify_report(
            Path::new("claims.json"),
            &ledger("performance_proven"),
            Path::new("matrix.json"),
            &matrix(0),
            None,
            None,
        );
        assert_eq!(report["passed"], false);
        assert!(
            report["failures"].as_array().unwrap().iter().any(|failure| failure
                .as_str()
                .unwrap()
                .contains("require claimable A770 kernels"))
        );
    }

    #[test]
    fn promoted_a770_experience_requires_claimable_experience_receipt() {
        let report = build_verify_report(
            Path::new("claims.json"),
            &ledger("performance_proven"),
            Path::new("matrix.json"),
            &matrix(3),
            None,
            None,
        );
        assert_eq!(report["passed"], false);
        assert!(report["failures"].as_array().unwrap().iter().any(|failure| {
            failure.as_str().unwrap().contains("requires claimable llm_experience_run receipt")
        }));
    }

    #[test]
    fn promoted_a770_experience_accepts_claimable_experience_receipt() {
        let report = build_verify_report(
            Path::new("claims.json"),
            &promoted_ledger_with_evidence(),
            Path::new("matrix.json"),
            &matrix(3),
            Some(Path::new("experience.json")),
            Some(&experience_receipt(true)),
        );
        assert_eq!(report["passed"], true);
        assert_eq!(report["llm_experience_receipt"]["claim_allowed"], true);
    }

    fn promoted_ledger_with_evidence() -> Value {
        let mut ledger = ledger("performance_proven");
        ledger["claims"][0]["evidence"] = json!(["experience.json"]);
        ledger
    }

    fn experience_receipt(claim_allowed: bool) -> Value {
        json!({
            "receipt_type": "llm_experience_run",
            "model": {
                "contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml"
            },
            "backend": {
                "selected_backend": "intel-arc-a770-opencl",
                "fallback_used": false
            },
            "kernel_route": {
                "route_verified": claim_allowed
            },
            "quality": {
                "passed": claim_allowed
            },
            "claim_gate": {
                "claim_allowed": claim_allowed,
                "classification": if claim_allowed { "performance_proven" } else { "diagnostic_only" }
            },
            "not_claims": CRITICAL_A770_NOT_CLAIMS
        })
    }
}
