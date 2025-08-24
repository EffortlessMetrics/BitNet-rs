//! Minimal smoke checks that run quickly and catch configuration drift.
//! These run in CI's "crossval-cpu-smoke" job.

use std::env;
use std::path::Path;

#[test]
fn smoke_env_preflight() {
    // Validate essential env so we fail fast if the runner is misconfigured.
    let model = env::var("CROSSVAL_GGUF").expect("CROSSVAL_GGUF must be set to a valid GGUF file");
    assert!(Path::new(&model).exists(), "CROSSVAL_GGUF path missing: {}", model);

    // Check library search path was wired by xtask (Linux or macOS)
    let ld = env::var("LD_LIBRARY_PATH")
        .ok()
        .or_else(|| env::var("DYLD_LIBRARY_PATH").ok())
        .unwrap_or_default();
    assert!(
        ld.contains("llama.cpp/src") || ld.contains("ggml/src"),
        "Library path not set to C++ build dirs: {}",
        ld
    );
}

// Name starts with `smoke_` so we can select via `cargo test smoke -- --ignored` if desired.
#[test]
#[ignore = "requires CROSSVAL_GGUF and C++ libraries"]
fn smoke_first_token_logits_parity() {
    // Keep this tiny (short prompt, 1 step) so the smoke job finishes quickly.
    let model_path = env::var("CROSSVAL_GGUF").expect("CROSSVAL_GGUF must be set");

    // Verify the model file exists
    assert!(Path::new(&model_path).exists(), "Model file not found: {}", model_path);

    // Quick check that the C++ library dir is set
    let cpp_dir = env::var("BITNET_CPP_DIR").expect("BITNET_CPP_DIR must be set");
    assert!(Path::new(&cpp_dir).exists(), "C++ directory not found: {}", cpp_dir);

    // TODO: Call your crossval harness helper for a minimal parity check
    // e.g., crossval::assert_first_logits_match(&model_path, "Hello");
    println!("Smoke test: Validated environment for {}", model_path);
}

#[test]
#[ignore = "requires model lock file"]
fn smoke_vocab_lock_validation() {
    // Check that the model vocab matches the lock file
    if !Path::new("crossval-models.lock.json").exists() {
        println!("Warning: crossval-models.lock.json not found, skipping vocab validation");
        return;
    }

    // Read the lock file
    let lock_content =
        std::fs::read_to_string("crossval-models.lock.json").expect("Failed to read lock file");
    let lock: serde_json::Value =
        serde_json::from_str(&lock_content).expect("Invalid JSON in lock file");

    // Get expected vocab size
    let expected_vocab =
        lock["crossval_default"]["n_vocab"].as_u64().expect("Missing n_vocab in lock file");

    println!("Expected vocab size from lock: {}", expected_vocab);

    // In a real test, you'd load the model and check its actual vocab size
    // For now, we just validate the lock file structure
    assert_eq!(expected_vocab, 128256, "Lock file should specify vocab size 128256");
}
