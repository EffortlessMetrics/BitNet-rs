use bitnet_receipts::{
    M4_RUN_IDENTITY_CONTRACT_VERSION, m4_run_identity_sha256,
    validate_m4_run_identity_contract_json,
};
use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn valid_receipt() -> Value {
    let run_identity = json!({
        "contract_version": M4_RUN_IDENTITY_CONTRACT_VERSION,
        "machine_id": "apple-m4-mac-mini",
        "soc": "apple-m4",
        "artifact_kind": "apple_m4_regression_dashboard",
        "evidence_family": "operator",
        "os": {
            "name": "macos",
            "version": "15.5",
            "version_source": "sw_vers"
        },
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "commit_source": "git_rev_parse"
        },
        "binary": {
            "crate_version": "0.1.0",
            "build_profile": "release"
        },
        "command": {
            "class": "mac regression-dashboard",
            "live_model_run": false
        },
        "model": {
            "id": "not_applicable",
            "sha256": "not_applicable",
            "identity_scope": "model_free"
        },
        "tokenizer": {
            "authority": "not_applicable",
            "sha256": "not_applicable",
            "identity_scope": "model_free"
        },
        "prompt_template": {
            "id": "not_applicable",
            "sha256": "243ffa2eeced1cbfa18357fe8edf03833381b9a83359bf0930ae5e8e862ab30e",
            "identity_scope": "model_free"
        },
        "backend": {
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false
        },
        "evidence_identity": {
            "scope": "regression_dashboard",
            "seed": "not_applicable",
            "corpus_id": "not_applicable",
            "profile_id": "not_applicable"
        },
        "timing": {
            "source": "wall_clock_utc"
        }
    });
    let run_identity_sha256 = m4_run_identity_sha256(&run_identity).unwrap();
    json!({
        "artifact_kind": "apple_m4_regression_dashboard",
        "requested_backend": "apple-m4-cpu-neon",
        "selected_backend": "apple-m4-cpu-neon",
        "runtime_api": "cpu",
        "fallback_used": false,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256
    })
}

fn refresh_run_identity_digest(receipt: &mut Value) {
    receipt["run_identity_sha256"] =
        json!(m4_run_identity_sha256(&receipt["run_identity"]).unwrap());
}

fn receipt_for_family(artifact_kind: &str, evidence_family: &str, command_class: &str) -> Value {
    let mut receipt = valid_receipt();
    receipt["artifact_kind"] = json!(artifact_kind);
    receipt["run_identity"]["artifact_kind"] = json!(artifact_kind);
    receipt["run_identity"]["evidence_family"] = json!(evidence_family);
    receipt["run_identity"]["command"]["class"] = json!(command_class);
    refresh_run_identity_digest(&mut receipt);
    receipt
}

fn remove_run_identity_field(receipt: &mut Value, section: &str, field: &str) -> TestResult {
    receipt["run_identity"][section]
        .as_object_mut()
        .ok_or("run_identity section must be an object")?
        .remove(field);
    Ok(())
}

#[test]
fn accepts_complete_m4_run_identity_contract() {
    let receipt = valid_receipt();
    validate_m4_run_identity_contract_json(&receipt).unwrap();
}

#[test]
fn accepts_m4_excellence_receipt_family_contracts() -> TestResult {
    for (artifact_kind, evidence_family, command_class) in [
        ("apple_m4_slm_eval_summary", "dense_slm_eval_v2", "mac eval"),
        ("apple_m4_slm_benchmark_v2", "dense_slm_benchmark_v2", "mac benchmark"),
        ("bitnet_apple_m4_warm_session", "bitnet_warm_session", "mac bitnet-warm"),
        ("bitnet_apple_m4_chat_gate", "bitnet_chat_gate", "mac bitnet-chat-gate"),
        ("bitnet_apple_m4_serve_gate", "bitnet_serve_gate", "mac bitnet-serve-gate"),
        ("apple_m4_regression_dashboard", "operator", "mac regression-dashboard"),
        ("bitnet_apple_m4_mac_ask_failure", "bitnet_failure", "mac ask"),
    ] {
        let receipt = receipt_for_family(artifact_kind, evidence_family, command_class);
        validate_m4_run_identity_contract_json(&receipt)
            .map_err(|err| format!("{artifact_kind} should validate: {err}"))?;
    }
    Ok(())
}

#[test]
fn rejects_missing_machine_id() {
    let mut receipt = valid_receipt();
    receipt["run_identity"].as_object_mut().unwrap().remove("machine_id");

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("machine_id"), "got: {err}");
}

#[test]
fn rejects_missing_binary_hash_and_build_profile() {
    let mut receipt = valid_receipt();
    let binary = receipt["run_identity"]["binary"].as_object_mut().unwrap();
    binary.remove("build_profile");
    binary.remove("binary_sha256");
    receipt["run_identity_sha256"] =
        json!(m4_run_identity_sha256(&receipt["run_identity"]).unwrap());

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("build_profile or binary_sha256"), "got: {err}");
}

#[test]
fn rejects_invalid_model_sha() {
    let mut receipt = valid_receipt();
    receipt["run_identity"]["model"]["id"] = json!("qwen2.5-0.5b-instruct-q8_0");
    receipt["run_identity"]["model"]["sha256"] = json!("not-a-sha");
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("sha256"), "got: {err}");
}

#[test]
fn rejects_missing_tokenizer_authority() -> TestResult {
    let mut receipt = valid_receipt();
    remove_run_identity_field(&mut receipt, "tokenizer", "authority")?;
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("authority"), "got: {err}");
    Ok(())
}

#[test]
fn rejects_missing_tokenizer_sha() -> TestResult {
    let mut receipt = valid_receipt();
    remove_run_identity_field(&mut receipt, "tokenizer", "sha256")?;
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("sha256"), "got: {err}");
    Ok(())
}

#[test]
fn rejects_backend_mismatch() {
    let mut receipt = valid_receipt();
    receipt["run_identity"]["backend"]["selected_backend"] = json!("apple-m4-metal");
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("run_identity backend selection"), "got: {err}");
}

#[test]
fn rejects_missing_identity_fallback_state() -> TestResult {
    let mut receipt = valid_receipt();
    remove_run_identity_field(&mut receipt, "backend", "fallback_used")?;
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "got: {err}");
    Ok(())
}

#[test]
fn rejects_fallback_mismatch() {
    let mut receipt = valid_receipt();
    receipt["fallback_used"] = json!(true);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "got: {err}");
}

#[test]
fn rejects_malformed_timing_source() {
    let mut receipt = valid_receipt();
    receipt["run_identity"]["timing"]["source"] = json!("");
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("source"), "got: {err}");
}

#[test]
fn rejects_artifact_kind_mismatch() {
    let mut receipt = valid_receipt();
    receipt["run_identity"]["artifact_kind"] = json!("bitnet_apple_m4_serve_gate");
    refresh_run_identity_digest(&mut receipt);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("artifact_kind"), "got: {err}");
}

#[test]
fn rejects_digest_mismatch() {
    let mut receipt = valid_receipt();
    receipt["run_identity_sha256"] =
        json!("0000000000000000000000000000000000000000000000000000000000000000");

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("does not match run_identity"), "got: {err}");
}
