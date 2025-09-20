use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use std::path::PathBuf;

fn get_test_model_path() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // Go up from xtask to BitNet-rs root
    path.push("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    path.to_string_lossy().to_string()
}

#[test]
fn verify_strict_exits_15() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify","--model","/nope/bad.gguf","--strict"]);
    cmd.assert().failure().code(15);
}

#[test]
fn verify_json_clean() {
    let model_path = get_test_model_path();
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify","--model",&model_path,"--format","json"])
       .env_remove("RUST_LOG");
    let out = cmd.assert().success().get_output().stdout.clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert!(v.get("model_path").is_some());
}

#[test]
fn infer_mock_json_deterministic() {
    let model_path = get_test_model_path();
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "infer","--model",&model_path,
        "--prompt","hi","--max-new-tokens","4","--allow-mock","--deterministic","--format","json"
    ]);
    let out = cmd.assert().success().get_output().stdout.clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["config"]["temperature"], 0.0);
    assert_eq!(v["config"]["seed"], 42);
}

#[test]
fn benchmark_json_file_exists() {
    let model_path = get_test_model_path();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "benchmark","--model",&model_path,
        "--allow-mock","--tokens","4","--warmup-tokens","2","--json",
        tmp.path().to_str().unwrap()
    ]);
    cmd.assert().success();
    let data = std::fs::read_to_string(tmp.path()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert!(v["performance"]["tokens_per_sec"].is_number());
}

#[test]
fn infer_exits_16_on_inference_failure() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["infer","--model","/nope/bad.gguf","--tokenizer","/dev/null","--prompt","test"]);
    cmd.assert().failure().code(16);
}

#[test]
fn benchmark_exits_17_on_failure() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["benchmark","--model","/nope/bad.gguf","--tokens","1"]);
    cmd.assert().failure().code(17);
}

#[test]
fn verify_shows_heads_info_on_valid_model() {
    let model_path = get_test_model_path();
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify","--model",&model_path]);
    cmd.assert().success()
       .stdout(predicate::str::contains("heads:"));
}

#[test]
fn benchmark_zero_tokens_short_circuit() {
    let model_path = get_test_model_path();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "benchmark","--model",&model_path,
        "--allow-mock","--tokens","0","--json",tmp.path().to_str().unwrap()
    ]);
    cmd.assert().success();

    let data = std::fs::read_to_string(tmp.path()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(v["success"], true);
    assert_eq!(v["timing"]["total_ms"], 0);
}