use bitnet_receipts::{
    M4_RUN_IDENTITY_CONTRACT_VERSION, m4_run_identity_sha256,
    validate_m4_run_identity_contract_json,
};
use serde_json::{Value, json};

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

#[test]
fn accepts_complete_m4_run_identity_contract() {
    let receipt = valid_receipt();
    validate_m4_run_identity_contract_json(&receipt).unwrap();
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
    receipt["run_identity_sha256"] =
        json!(m4_run_identity_sha256(&receipt["run_identity"]).unwrap());

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("sha256"), "got: {err}");
}

#[test]
fn rejects_backend_mismatch() {
    let mut receipt = valid_receipt();
    receipt["run_identity"]["backend"]["selected_backend"] = json!("apple-m4-metal");
    receipt["run_identity_sha256"] =
        json!(m4_run_identity_sha256(&receipt["run_identity"]).unwrap());

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("run_identity backend selection"), "got: {err}");
}

#[test]
fn rejects_fallback_mismatch() {
    let mut receipt = valid_receipt();
    receipt["fallback_used"] = json!(true);

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "got: {err}");
}

#[test]
fn rejects_digest_mismatch() {
    let mut receipt = valid_receipt();
    receipt["run_identity_sha256"] =
        json!("0000000000000000000000000000000000000000000000000000000000000000");

    let err = validate_m4_run_identity_contract_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("does not match run_identity"), "got: {err}");
}
