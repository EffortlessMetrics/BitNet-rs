//! Unit tests for benchmark receipt validation.

use super::*;
use crate::validation::expected_cpu_profile_phase;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

fn expect_validation_error(result: Result<(), ReceiptError>) -> Result<String, String> {
    match result {
        Ok(()) => Err("expected validation failure".to_string()),
        Err(err) => Ok(err.to_string()),
    }
}

fn json_array_mut<'a>(value: &'a mut Value, pointer: &str) -> Result<&'a mut Vec<Value>, String> {
    value
        .pointer_mut(pointer)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| format!("{pointer} must point to an array"))
}

fn json_object_mut<'a>(
    value: &'a mut Value,
    pointer: &str,
) -> Result<&'a mut serde_json::Map<String, Value>, String> {
    value
        .pointer_mut(pointer)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| format!("{pointer} must point to an object"))
}

fn value_object_mut<'a>(
    value: &'a mut Value,
    name: &str,
) -> Result<&'a mut serde_json::Map<String, Value>, String> {
    value.as_object_mut().ok_or_else(|| format!("{name} must be an object"))
}

fn sample_receipt(name: &str, elapsed_us: u64) -> BenchReceipt {
    BenchReceipt::new(
        name,
        [256, 1, 1],
        [1024, 1, 1],
        elapsed_us,
        42.0,
        1_700_000_000,
        "Test GPU",
        "vulkan",
    )
}

#[test]
fn test_new_sets_all_fields() {
    let r = sample_receipt("matmul", 500);
    assert_eq!(r.kernel_name, "matmul");
    assert_eq!(r.workgroup_size, [256, 1, 1]);
    assert_eq!(r.dispatch_size, [1024, 1, 1]);
    assert_eq!(r.elapsed_us, 500);
    assert_eq!(r.device_name, "Test GPU");
    assert_eq!(r.backend, "vulkan");
}

#[test]
fn test_to_json_produces_valid_json() {
    let r = sample_receipt("softmax", 100);
    let json = r.to_json();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["kernel_name"], "softmax");
}

#[test]
fn test_from_json_roundtrip() {
    let r = sample_receipt("rms_norm", 250);
    let json = r.to_json();
    let r2 = BenchReceipt::from_json(&json).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn test_from_json_invalid_returns_error() {
    let result = BenchReceipt::from_json("not json");
    assert!(result.is_err());
}

#[test]
fn test_from_json_missing_field() {
    let result = BenchReceipt::from_json(r#"{"kernel_name":"x"}"#);
    assert!(result.is_err());
}

#[test]
fn test_serialization_preserves_workgroup_array() {
    let r = sample_receipt("conv", 300);
    let json = r.to_json();
    assert!(json.contains("[256,1,1]"));
}

#[test]
fn test_store_append_and_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("receipts.jsonl");

    let r1 = sample_receipt("k1", 100);
    let r2 = sample_receipt("k2", 200);
    ReceiptStore::append(&path, &r1).unwrap();
    ReceiptStore::append(&path, &r2).unwrap();

    let loaded = ReceiptStore::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0], r1);
    assert_eq!(loaded[1], r2);
}

#[test]
fn test_store_load_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.jsonl");
    std::fs::File::create(&path).unwrap();

    let loaded = ReceiptStore::load(&path).unwrap();
    assert!(loaded.is_empty());
}

#[test]
fn test_store_load_nonexistent_file() {
    let result = ReceiptStore::load(Path::new("/nonexistent/path.jsonl"));
    assert!(result.is_err());
}

#[test]
fn test_store_skips_blank_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blanks.jsonl");
    let r = sample_receipt("k1", 100);
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "{}", r.to_json()).unwrap();
    writeln!(f).unwrap();
    writeln!(f, "{}", r.to_json()).unwrap();
    drop(f);

    let loaded = ReceiptStore::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
}

#[test]
fn test_throughput_precision() {
    let r = BenchReceipt::new("k", [1, 1, 1], [1, 1, 1], 1, std::f64::consts::PI, 0, "", "");
    let r2 = BenchReceipt::from_json(&r.to_json()).unwrap();
    assert!((r2.throughput_gflops - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn test_store_append_creates_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("new.jsonl");
    assert!(!path.exists());

    ReceiptStore::append(&path, &sample_receipt("k", 1)).unwrap();
    assert!(path.exists());
}

#[test]
fn rtx5070ti_cuda_benchmark_receipt_validates() {
    let receipt = sample_cuda_benchmark_receipt();
    validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).unwrap();
}

#[test]
fn rtx5070ti_cuda_benchmark_rejects_generic_cuda_backend() {
    let mut receipt = sample_cuda_benchmark_receipt();
    receipt["selected_backend"] = json!("cuda");
    assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
}

#[test]
fn rtx5070ti_cuda_benchmark_rejects_fallback() {
    let mut receipt = sample_cuda_benchmark_receipt();
    receipt["fallback_used"] = json!(true);
    assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
}

#[test]
fn rtx5070ti_cuda_benchmark_rejects_speedup_claim() {
    let mut receipt = sample_cuda_benchmark_receipt();
    receipt["speedup_claim"] = json!(true);
    assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
}

#[test]
fn rtx5070ti_cuda_benchmark_rejects_missing_required_profile() {
    let mut receipt = sample_cuda_benchmark_receipt();
    receipt["profiles"] = json!([
        { "profile": "cuda_tiny_smoke", "status": "measured" }
    ]);
    assert!(validate_rtx5070ti_cuda_benchmark_receipt_json(&receipt).is_err());
}

#[test]
fn strict_cpu_benchmark_receipt_validates() {
    let receipt = sample_cpu_benchmark_receipt();
    validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap();
}

#[test]
fn committed_lunar_lake_i2s_tiling_matrix_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ci/hardware/intel-258v/2026-05-08/cpu-bitnet-perf-002-i2s-tiling-matrix.json");
    let result = validate_strict_cpu_benchmark_receipt_file(&path);
    assert!(
        result.is_ok(),
        "committed Lunar Lake tiling matrix receipt should validate: {result:?}"
    );
}

#[test]
fn strict_cpu_benchmark_rejects_missing_decode_profile() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["profiles"] = json!([
        measured_cpu_profile("micro"),
        measured_cpu_profile("layer"),
        measured_cpu_profile("prefill"),
        measured_cpu_profile("first_token")
    ]);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("decode"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_hidden_fallback() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["fallback_used"] = json!(true);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_avx2_without_fma() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["cpu"]["features"] = json!(["avx2"]);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("avx2 and fma"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_speedup_claim() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["speedup_claim"] = json!(true);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_profile_phase_mismatch() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["profiles"][4]["execution_phase"] = json!("prefill");

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("execution_phase"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_profile_without_shape() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["profiles"][0]["shape"] = serde_json::Value::Null;

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("shape"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_incomplete_i2s_microbench() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["i2s_microbench"]["profiles"] = json!([measured_i2s_microbench_profile("gemv")]);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("missing gemm"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_i2s_tiling_matrix_speedup_claim() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["i2s_tiling_thread_matrix"]["speedup_claim"] = json!(true);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_incomplete_i2s_tiling_matrix() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["i2s_tiling_thread_matrix"]["measured_runs"] =
        json!([measured_i2s_tiling_matrix_run("gemv")]);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("missing gemm"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_applied_thread_matrix_speedup_claim() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["i2s_applied_thread_matrix"]["speedup_claim"] = json!(true);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_unapplied_applied_thread_matrix() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["i2s_applied_thread_matrix"]["measured_runs"][0]["candidate"]["thread_count_applied"] =
        json!(false);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("thread_count_applied"), "unexpected error: {err}");
}

#[test]
fn strict_cpu_benchmark_rejects_embedding_quantization_speedup_claim() {
    let mut receipt = sample_cpu_benchmark_receipt();
    receipt["embedding_quantization_evidence"]["speedup_claim"] = json!(true);

    let err = validate_strict_cpu_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn committed_lunar_lake_embedding_quantization_evidence_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../ci/hardware/intel-258v/2026-05-08/cpu-bitnet-embd-001-q6k-embedding-evidence.json",
    );
    let result = validate_strict_cpu_benchmark_receipt_file(&path);
    assert!(
        result.is_ok(),
        "committed Lunar Lake embedding quantization evidence receipt should validate: {result:?}"
    );
}

#[test]
fn committed_lunar_lake_applied_thread_matrix_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/intel-258v/2026-05-08/cpu-bitnet-perf-003-i2s-applied-thread-matrix.json",
        );
    let result = validate_strict_cpu_benchmark_receipt_file(&path);
    assert!(
        result.is_ok(),
        "committed Lunar Lake applied-thread matrix receipt should validate: {result:?}"
    );
}

#[test]
fn committed_rtx5070ti_cuda_benchmark_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json");
    validate_rtx5070ti_cuda_benchmark_receipt_file(&path).unwrap();
}

#[test]
fn strict_bitnet_cuda_benchmark_receipt_validates() {
    let receipt = sample_strict_bitnet_cuda_benchmark_receipt();
    validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap();
}

#[test]
fn strict_bitnet_cuda_benchmark_rejects_generic_cuda_backend() {
    let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
    receipt["selected_backend"] = json!("cuda");

    let err = validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("selected_backend"), "unexpected error: {err}");
}

#[test]
fn strict_bitnet_cuda_benchmark_rejects_fallback() {
    let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
    receipt["fallback_used"] = json!(true);

    let err = validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn strict_bitnet_cuda_benchmark_rejects_speedup_claim() {
    let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
    receipt["speedup_claim"] = json!(true);

    let err = validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn strict_bitnet_cuda_benchmark_rejects_missing_cpu_profile() {
    let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
    receipt["profiles"] = json!([
        not_run_bitnet_profile("amd-9950x3d-cpu-scalar"),
        measured_bitnet_profile("amd-9950x3d-cpu-avx512", "cpu"),
        measured_bitnet_profile("nvidia-rtx-5070-ti-cuda", "cuda")
    ]);

    let err = validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("amd-9950x3d-cpu-avx2"), "unexpected error: {err}");
}

#[test]
fn strict_bitnet_cuda_benchmark_requires_measured_cuda_profile() {
    let mut receipt = sample_strict_bitnet_cuda_benchmark_receipt();
    receipt["profiles"][3] = not_run_bitnet_profile("nvidia-rtx-5070-ti-cuda");

    let err = validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("must be measured"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_answer_path_benchmark_receipt_validates() {
    let receipt = sample_strict_cuda_answer_path_benchmark_receipt();
    validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt).unwrap();
}

#[test]
fn strict_cuda_answer_path_benchmark_rejects_speedup_claim() {
    let mut receipt = sample_strict_cuda_answer_path_benchmark_receipt();
    receipt["speedup_claim"] = json!(true);

    let err =
        validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_answer_path_benchmark_rejects_hidden_fallback() {
    let mut receipt = sample_strict_cuda_answer_path_benchmark_receipt();
    receipt["profiles"][1]["fallback_used"] = json!(true);

    let err =
        validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_answer_path_benchmark_rejects_dense_execution_plan() {
    let mut receipt = sample_strict_cuda_answer_path_benchmark_receipt();
    receipt["execution_plan"]["selected_route"] = json!("dense_regular_llm_cuda");
    receipt["execution_plan"]["bitnet_packed_qk256_cuda"] = json!(false);
    receipt["execution_plan"]["dense_regular_llm_cuda"] = json!(true);

    let err =
        validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("selected_route"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_answer_path_benchmark_rejects_missing_long_profile_disposition() {
    let mut receipt = sample_strict_cuda_answer_path_benchmark_receipt();
    receipt["profiles"] = json!([
        measured_answer_path_profile("strict_ask_math_8", "amd-9950x3d-cpu-avx512", "cpu"),
        measured_answer_path_profile("strict_ask_math_8", "nvidia-rtx-5070-ti-cuda", "cuda"),
        existing_answer_path_profile("answer_corpus_5", "amd-9950x3d-cpu-avx512"),
        existing_answer_path_profile("answer_corpus_5", "nvidia-rtx-5070-ti-cuda")
    ]);

    let err =
        validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("prefill_512_decode_128"), "unexpected error: {err}");
}

#[test]
fn committed_strict_bitnet_cuda_benchmark_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json",
    );
    validate_strict_bitnet_cuda_benchmark_receipt_file(&path).unwrap();
}

#[test]
fn committed_strict_cuda_answer_path_benchmark_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-prod-004-answer-path-benchmark.json",
        );
    validate_strict_cuda_answer_path_benchmark_receipt_file(&path).unwrap();
}

#[test]
fn strict_cuda_repeated_ask_benchmark_receipt_validates() {
    let receipt = sample_strict_cuda_repeated_ask_benchmark_receipt();
    validate_strict_cuda_repeated_ask_benchmark_receipt_json(&receipt).unwrap();
}

#[test]
fn strict_cuda_repeated_ask_benchmark_rejects_single_run() {
    let mut receipt = sample_strict_cuda_repeated_ask_benchmark_receipt();
    receipt["repeat_policy"]["runs_per_backend"] = json!(1);

    let err =
        validate_strict_cuda_repeated_ask_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("runs_per_backend"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_repeated_ask_benchmark_rejects_unmeasured_transfer_bytes() {
    let mut receipt = sample_strict_cuda_repeated_ask_benchmark_receipt();
    receipt["kernel_stats"][0]["host_to_device_bytes"] = json!(0);

    let err =
        validate_strict_cuda_repeated_ask_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("host_to_device_bytes"), "unexpected error: {err}");
}

#[test]
fn committed_strict_cuda_repeated_ask_benchmark_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json",
        );
    validate_strict_cuda_repeated_ask_benchmark_receipt_file(&path).unwrap();
}

#[test]
fn strict_cuda_warm_session_benchmark_receipt_validates() {
    let receipt = sample_strict_cuda_warm_session_benchmark_receipt();
    validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt).unwrap();
}

#[test]
fn strict_cuda_warm_session_benchmark_rejects_single_run() {
    let mut receipt = sample_strict_cuda_warm_session_benchmark_receipt();
    receipt["session_contract"]["runs_per_backend"] = json!(1);

    let err =
        validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("runs_per_backend"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_warm_session_benchmark_rejects_hidden_fallback() {
    let mut receipt = sample_strict_cuda_warm_session_benchmark_receipt();
    receipt["runs"][0]["fallback_used"] = json!(true);

    let err =
        validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_warm_session_benchmark_rejects_speedup_claim() {
    let mut receipt = sample_strict_cuda_warm_session_benchmark_receipt();
    receipt["speedup_claim"] = json!(true);

    let err =
        validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_warm_session_benchmark_rejects_missing_transfer_bytes() {
    let mut receipt = sample_strict_cuda_warm_session_benchmark_receipt();
    receipt["kernel_stats"][0]["host_to_device_bytes"] = json!(0);

    let err =
        validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("host_to_device_bytes"), "unexpected error: {err}");
}

#[test]
fn committed_strict_cuda_warm_session_benchmark_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json",
        );
    validate_strict_cuda_warm_session_benchmark_receipt_file(&path).unwrap();
}

#[test]
fn strict_cuda_benchmark_qualification_receipt_validates() -> Result<(), String> {
    let receipt = sample_strict_cuda_benchmark_qualification_receipt();
    validate_strict_cuda_benchmark_qualification_receipt_json(&receipt)
        .map_err(|err| err.to_string())
}

#[test]
fn strict_cuda_benchmark_qualification_rejects_speedup_claim() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_benchmark_qualification_receipt();
    receipt["speedup_claim"] = json!(true);

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_rejects_accepted_profile() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_benchmark_qualification_receipt();
    receipt["profile_reviews"][0]["decision"] = json!("accepted");

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("not_accepted"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_requires_blocked_requirement() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_benchmark_qualification_receipt();
    for requirement in json_array_mut(&mut receipt, "/qualification_requirements")? {
        requirement["status"] = json!("passed");
        value_object_mut(requirement, "qualification requirement")?.remove("blocker");
    }

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("blocked requirement"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_validates_product_profiles() -> Result<(), String> {
    let receipt = sample_strict_cuda_product_benchmark_qualification_receipt();
    validate_strict_cuda_benchmark_qualification_receipt_json(&receipt)
        .map_err(|err| err.to_string())
}

#[test]
fn strict_cuda_benchmark_qualification_rejects_missing_target_profile() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_product_benchmark_qualification_receipt();
    json_array_mut(&mut receipt, "/target_profiles")?.pop();

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("target_profiles"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_rejects_extra_blocked_profile() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_product_benchmark_qualification_receipt();
    json_array_mut(&mut receipt, "/qualification_decision/blocked_profiles")?
        .push(json!("dense_qwen_short_decode"));

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("blocked_profiles"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_rejects_extra_profile_review() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_product_benchmark_qualification_receipt();
    json_array_mut(&mut receipt, "/profile_reviews")?.push(json!({
        "profile": "dense_qwen_short_decode",
        "decision": "not_accepted",
        "evidence_status": "missing",
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": false,
        "quality_passed": false,
        "dense_cuda_evidence_used": false,
        "reason": "Dense evidence is out of scope for BitNet QK256 qualification.",
        "blockers": ["wrong proof family"]
    }));

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("profile_reviews"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_requires_strict_ask_evidence() -> Result<(), String> {
    let mut receipt = sample_strict_cuda_product_benchmark_qualification_receipt();
    json_object_mut(&mut receipt, "/evidence_summary")?.remove("strict_ask_math_8");

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("strict_ask_math_8"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn strict_cuda_benchmark_qualification_requires_warm_session_receipt_boundary() -> Result<(), String>
{
    let mut receipt = sample_strict_cuda_product_benchmark_qualification_receipt();
    *receipt
        .pointer_mut(
            "/evidence_summary/strict_cuda_warm_session_2_turns/qk256_weights_uploaded_once",
        )
        .ok_or_else(|| "warm-session qk256_weights_uploaded_once field must exist".to_string())? =
        json!(false);

    let err = expect_validation_error(validate_strict_cuda_benchmark_qualification_receipt_json(
        &receipt,
    ))?;
    assert!(err.contains("qk256_weights_uploaded_once"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn committed_strict_cuda_benchmark_qualification_receipt_validates() -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-004-benchmark-qualification.json",
        );
    validate_strict_cuda_benchmark_qualification_receipt_file(&path).map_err(|err| err.to_string())
}

#[test]
fn committed_strict_cuda_product_benchmark_qualification_receipt_validates() -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/cuda-prod-010-benchmark-qualification.json",
        );
    validate_strict_cuda_benchmark_qualification_receipt_file(&path).map_err(|err| err.to_string())
}

#[test]
fn dense_gguf_qwen_cuda_benchmark_baseline_receipt_validates() {
    let receipt = sample_dense_gguf_qwen_cuda_benchmark_baseline_receipt();
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_gguf_qwen_cuda_benchmark_baseline_rejects_speedup_claim() {
    let mut receipt = sample_dense_gguf_qwen_cuda_benchmark_baseline_receipt();
    receipt["speedup_claim"] = json!(true);

    let err = validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_cuda_benchmark_baseline_rejects_bitnet_proof_claim() {
    let mut receipt = sample_dense_gguf_qwen_cuda_benchmark_baseline_receipt();
    receipt["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_cuda_benchmark_baseline_rejects_missing_warm_profile() {
    let mut receipt = sample_dense_gguf_qwen_cuda_benchmark_baseline_receipt();
    receipt["profiles"] = json!([
        dense_qwen_benchmark_profile("one_token", 1, 8),
        dense_qwen_benchmark_profile("short_decode_8", 8, 8)
    ]);

    let err = validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("warm_session_3_turns"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_cuda_benchmark_baseline_rejects_hidden_fallback() {
    let mut receipt = sample_dense_gguf_qwen_cuda_benchmark_baseline_receipt();
    receipt["profiles"][0]["fallback_used"] = json!(true);

    let err = validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn committed_dense_gguf_qwen_cuda_benchmark_baseline_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json",
        );
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_file(&path).unwrap();
}

#[test]
fn committed_dense_gguf_qwen_repeated_comparator_receipt_validates() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json",
        );
    validate_dense_gguf_qwen_repeated_comparator_receipt_file(&path).unwrap();
}

#[test]
fn dense_gguf_qwen_repeated_comparator_receipt_validates() {
    let receipt = sample_dense_gguf_qwen_repeated_comparator_receipt();
    validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_gguf_qwen_repeated_comparator_rejects_speedup_claim() {
    let mut receipt = sample_dense_gguf_qwen_repeated_comparator_receipt();
    receipt["speedup_claim"] = json!(true);

    let err = validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_repeated_comparator_rejects_single_run_profile() {
    let mut receipt = sample_dense_gguf_qwen_repeated_comparator_receipt();
    receipt["profiles"][0]["run_count"] = json!(1);
    receipt["profiles"][0]["cpu_runs"] = json!(1);
    receipt["profiles"][0]["cuda_runs"] = json!(1);
    receipt["profiles"][0]["runs"] = json!([dense_qwen_comparator_run("one_token", 1)]);

    let err = validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("run"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_repeated_comparator_rejects_hidden_fallback() {
    let mut receipt = sample_dense_gguf_qwen_repeated_comparator_receipt();
    receipt["profiles"][1]["runs"][0]["fallback_used"] = json!(true);

    let err = validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_repeated_comparator_rejects_missing_transfer_timing_status() {
    let mut receipt = sample_dense_gguf_qwen_repeated_comparator_receipt();
    receipt["profiles"][2]["transfer_timing_status"] = json!("");

    let err = validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("transfer_timing_status"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_repeated_comparator_rejects_duplicate_source_path() {
    let mut receipt = sample_dense_gguf_qwen_repeated_comparator_receipt();
    receipt["profiles"][0]["runs"][1]["source_receipt_path"] =
        receipt["profiles"][0]["runs"][0]["source_receipt_path"].clone();

    let err = validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unique"), "unexpected error: {err}");
}

#[test]
fn qwen3_cuda_repeated_comparator_receipt_validates() -> Result<(), ReceiptError> {
    let receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn qwen3_cuda_repeated_comparator_rejects_short_decode_32_token_drift() {
    let mut receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    receipt["profiles"][2]["runs"][0]["generated_tokens"] = json!(8);

    let err =
        validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("generated_tokens"), "unexpected error: {err}");
}

#[test]
fn qwen3_cuda_repeated_comparator_rejects_qwen25_inheritance() {
    let mut receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    receipt["qwen25_proof_inherited"] = json!(true);

    let err =
        validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("qwen25_proof_inherited"), "unexpected error: {err}");
}

#[test]
fn qwen3_cuda_repeated_comparator_rejects_bitnet_packed_proof() {
    let mut receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    receipt["profiles"][0]["runs"][0]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err =
        validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn qwen3_cuda_repeated_comparator_rejects_missing_decode_128_profile() -> Result<(), String> {
    let mut receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    let duplicate_profile = receipt["profiles"][0].clone();
    json_array_mut(&mut receipt, "/profiles")?.pop();
    json_array_mut(&mut receipt, "/profiles")?.push(duplicate_profile);

    let err =
        expect_validation_error(validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt))?;
    assert!(err.contains("decode_128_from_warm_context"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn qwen3_cuda_repeated_comparator_rejects_extra_profile() -> Result<(), String> {
    let mut receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    let extra_profile = receipt["profiles"][0].clone();
    json_array_mut(&mut receipt, "/profiles")?.push(extra_profile);

    let err =
        expect_validation_error(validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt))?;
    assert!(err.contains("exactly 5"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn qwen3_cuda_repeated_comparator_rejects_profile_count_drift() {
    let mut receipt = sample_qwen3_cuda_repeated_comparator_receipt();
    receipt["comparator_summary"]["profiles_recorded"] = json!(6);

    let err =
        validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("profiles_recorded"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_receipt_validates() {
    let receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt();
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_accepts_h2d_envelope() {
    let receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt_with_h2d_envelope();
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_rejects_speedup_claim() {
    let mut receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt();
    receipt["speedup_claim"] = json!(true);

    let err = validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_rejects_accepted_profile() {
    let mut receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt();
    receipt["profile_reviews"][0]["decision"] = json!("accepted");

    let err = validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not_accepted"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_requires_blocked_requirement() {
    let mut receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt();
    for requirement in receipt["qualification_requirements"].as_array_mut().unwrap() {
        requirement["status"] = json!("passed");
        requirement.as_object_mut().unwrap().remove("blocker");
    }

    let err = validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("blocked requirement"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_requires_h2d_timing_source() {
    let mut receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt();
    receipt["profile_reviews"][0]["host_to_device_ms_source"] = json!("");

    let err = validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("host_to_device_ms_source"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_benchmark_qualification_rejects_missing_h2d_envelope_scope() {
    let mut receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt_with_h2d_envelope();
    receipt["profile_reviews"][0].as_object_mut().unwrap().remove("host_to_device_ms_scope");

    let err = validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("host_to_device_ms_scope"), "unexpected error: {err}");
}

#[test]
fn committed_dense_gguf_qwen_benchmark_qualification_receipt_validates() -> Result<(), ReceiptError>
{
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-benchmark-qualification.json",
        );
    validate_dense_gguf_qwen_benchmark_qualification_receipt_file(&path)
}

#[test]
fn committed_qwen3_benchmark_qualification_receipt_validates() -> Result<(), ReceiptError> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-benchmark-qualification.json",
        );
    validate_dense_gguf_qwen_benchmark_qualification_receipt_file(&path)
}

fn sample_dense_gguf_qwen_cuda_benchmark_baseline_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_qwen_cuda_benchmark_baseline",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-10T05:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "claim": "dense_gguf_qwen_cuda_benchmark_baseline",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "dense_gguf_inference_claimed": false,
        "server_ready_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "claim_boundary": {
            "dense_gguf_qwen_cuda_benchmark_baseline_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "qwen_chat_cuda_claimed": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false
        },
        "model": {
            "id": "qwen2.5-0.5b-instruct-q8_0",
            "model_family": "qwen",
            "artifact_kind": "dense_gguf",
            "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
            "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
        },
        "tokenizer_prompt_authority": {
            "tokenizer_authority": "contract_authoritative",
            "prompt_authority": "contract_authoritative",
            "prompt_template": "qwen-chat-raw-deterministic",
            "prompt_token_count": 8,
            "prompt_token_ids_sha256": "c94c330c606eb24adef37d7fd276fc45a01c9fb21b6a8cea482102513f767144",
            "rendered_prompt_sha256": "52cb6b5e4a038af1756708f98afb718a08c75b87b2f03dbee4dd9c8139c15c5e"
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_gguf_q8_0_f16_qwen_warm_session_contract",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "cuda_dense_regular_llm_ops": 8112,
            "cuda_bitnet_qk256_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0
        },
        "proof_inputs": {
            "one_token": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "artifact_kind": "dense_gguf_qwen_one_token_strict_cuda_proof"
            },
            "short_decode": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json",
                "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "artifact_kind": "dense_gguf_qwen_short_decode_strict_cuda_proof"
            },
            "warm_session": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json",
                "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof"
            }
        },
        "profiles": [
            dense_qwen_benchmark_profile("one_token", 1, 8),
            dense_qwen_benchmark_profile("short_decode_8", 8, 8),
            dense_qwen_benchmark_profile("warm_session_3_turns", 24, 22)
        ],
        "kernel_summary": {
            "total_kernel_invocations": 11154,
            "total_kernel_launches": 11154,
            "total_kernel_time_ms": 13182.4159,
            "total_host_to_device_bytes": 2027132448,
            "total_device_to_host_bytes": 20055552,
            "total_cpu_fallback_invocations": 0,
            "fallback_used": false
        },
        "benchmark_summary": {
            "status": "baseline_only",
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "profiles_recorded": 3,
            "accepted_speedup_profiles": [],
            "qualification_blockers": [
                "missing repeated CPU/CUDA comparator for dense profiles",
                "missing profile-specific speedup thresholds"
            ],
            "next_step": "CUDA-DENSE-PERF-002 repeated CPU/CUDA comparator"
        },
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64
        },
        "claim_boundaries": [
            "speedup_claim=false; dense Qwen CUDA timing is baseline evidence only.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof.",
            "full_cuda_residency_claimed=false; warm-session residency remains scoped."
        ]
    })
}

fn dense_qwen_benchmark_profile(
    profile: &str,
    generated_tokens: u64,
    prompt_tokens: u64,
) -> serde_json::Value {
    let mut value = json!({
        "profile": profile,
        "backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "status": "measured_existing_receipt",
        "source_receipt_path": format!("ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/{profile}.json"),
        "source_receipt_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "source_artifact_kind": "dense_gguf_qwen_short_decode_strict_cuda_proof",
        "fallback_used": false,
        "quality_passed": true,
        "parity_passed": true,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_ms": 1.0,
        "first_token_ms": 1.0,
        "decode_total_ms": 1.0,
        "kernel_time_ms": 1.0,
        "kernel_invocations": 1,
        "kernel_launches": 1,
        "host_to_device_bytes": 1,
        "device_to_host_bytes": 1,
        "cpu_reference_total_ms": 1.0
    });
    if profile == "warm_session_3_turns" {
        value["turns_count"] = json!(3);
    }
    value
}

fn sample_dense_gguf_qwen_repeated_comparator_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_qwen_repeated_comparator",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-10T15:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "claim": "dense_gguf_qwen_repeated_comparator",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "dense_gguf_inference_claimed": false,
        "server_ready_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "claim_boundary": {
            "dense_gguf_qwen_repeated_comparator_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "qwen_chat_cuda_claimed": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false
        },
        "model": {
            "id": "qwen2.5-0.5b-instruct-q8_0",
            "model_family": "qwen",
            "artifact_kind": "dense_gguf",
            "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
            "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
        },
        "tokenizer_prompt_authority": {
            "tokenizer_authority": "contract_authoritative",
            "prompt_authority": "contract_authoritative",
            "prompt_template": "qwen-chat-raw-deterministic",
            "deterministic_prompt": true,
            "prompt_token_count": 8,
            "prompt_token_ids_sha256": "c94c330c606eb24adef37d7fd276fc45a01c9fb21b6a8cea482102513f767144",
            "rendered_prompt_sha256": "52cb6b5e4a038af1756708f98afb718a08c75b87b2f03dbee4dd9c8139c15c5e"
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_gguf_q8_0_f16_qwen_warm_session_contract",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "cuda_dense_regular_llm_ops": 8112,
            "cuda_bitnet_qk256_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0
        },
        "baseline_input": {
            "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "artifact_kind": "dense_gguf_qwen_cuda_benchmark_baseline"
        },
        "profiles": [
            dense_qwen_comparator_profile("one_token"),
            dense_qwen_comparator_profile("short_decode_8"),
            dense_qwen_comparator_profile("warm_session_3_turns")
        ],
        "comparator_summary": {
            "status": "repeated_comparator_only",
            "profiles_recorded": 3,
            "min_runs_per_backend": 3,
            "total_cpu_runs": 9,
            "total_cuda_runs": 9,
            "fallback_free": true,
            "same_artifact_sha": true,
            "same_tokenizer_prompt_authority": true,
            "deterministic_generation_policy": true,
            "generated_tokens_compared": true,
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "accepted_speedup_profiles": [],
            "remaining_qualification_blockers": [
                "host/device transfer timing is not yet measured as CUDA event timing",
                "profile-specific speedup thresholds remain unreviewed"
            ],
            "next_step": "CUDA-DENSE-PERF-003 profile-specific speedup qualification review"
        },
        "transfer_timing": {
            "status": "not_measured_in_source_receipts",
            "source": "source strict CUDA proof receipts record H2D/D2H bytes but do not yet expose transfer event timing",
            "host_to_device_bytes_recorded": true,
            "device_to_host_bytes_recorded": true,
            "host_to_device_timing_recorded": false,
            "device_to_host_timing_recorded": false
        },
        "hardware_context": {
            "vram_bytes": 17094475776u64,
            "power_draw_watts_min": 32.0,
            "power_draw_watts_max": 50.0,
            "temperature_c_min": 44.0,
            "temperature_c_max": 47.0,
            "source": "NVML fields recorded in source strict CUDA proof receipts"
        },
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64
        },
        "claim_boundaries": [
            "speedup_claim=false; repeated CPU/CUDA comparator evidence is not a speedup qualification.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof."
        ]
    })
}

fn dense_qwen_comparator_profile(profile: &str) -> serde_json::Value {
    json!({
        "profile": profile,
        "status": "repeated_same_artifact_cpu_cuda_comparator",
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "run_count": 3,
        "cpu_runs": 3,
        "cuda_runs": 3,
        "min_runs_per_backend": 3,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_authority": true,
        "deterministic_generation_policy": true,
        "generated_token_ids_match": true,
        "first_divergence_report": "none",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "transfer_timing_status": "not_measured_in_source_receipts",
        "cpu_total_ms": dense_qwen_number_summary(),
        "cuda_total_ms": dense_qwen_number_summary(),
        "first_token_ms": dense_qwen_number_summary(),
        "decode_total_ms": dense_qwen_number_summary(),
        "kernel_time_ms": dense_qwen_number_summary(),
        "host_to_device_bytes": dense_qwen_u64_summary(),
        "device_to_host_bytes": dense_qwen_u64_summary(),
        "runs": [
            dense_qwen_comparator_run(profile, 1),
            dense_qwen_comparator_run(profile, 2),
            dense_qwen_comparator_run(profile, 3)
        ]
    })
}

fn dense_qwen_comparator_run(profile: &str, index: u64) -> serde_json::Value {
    let mut run = json!({
        "run_id": format!("run-{index:02}"),
        "profile": profile,
        "source_receipt_path": format!("ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-{index:02}/{profile}.json"),
        "source_receipt_sha256": format!("{index:064x}"),
        "source_artifact_kind": "dense_gguf_qwen_short_decode_strict_cuda_proof",
        "model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        "prompt_template": "qwen-chat-raw-deterministic",
        "prompt_token_count": 8,
        "generation_policy": "greedy",
        "deterministic_generation": true,
        "generated_tokens": 8,
        "generated_token_ids_sha256": "638e3358dd291728156e5d98d5fc9cb7e66afa992c8cd16833c85a552572aa4f",
        "generated_token_ids_match": true,
        "first_divergence_report": "none",
        "top_k_compared": true,
        "fallback_used": false,
        "quality_passed": true,
        "parity_passed": true,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "timing": {
            "cpu_total_ms": 3000.0,
            "cuda_total_ms": 1000.0,
            "first_token_ms": 100.0,
            "decode_total_ms": 100.0,
            "kernel_time_ms": 90.0,
            "kernel_invocations": 1,
            "kernel_launches": 1,
            "host_to_device_bytes": 1,
            "device_to_host_bytes": 1,
            "host_to_device_ms": null,
            "host_to_device_ms_source": "not_measured_in_source_receipt",
            "device_to_host_ms": null,
            "device_to_host_ms_source": "not_measured_in_source_receipt"
        }
    });
    if profile == "warm_session_3_turns" {
        run["turns_count"] = json!(3);
    }
    run
}

fn dense_qwen_number_summary() -> serde_json::Value {
    json!({
        "count": 3,
        "min": 1.0,
        "mean": 1.0,
        "max": 1.0
    })
}

fn dense_qwen_u64_summary() -> serde_json::Value {
    json!({
        "count": 3,
        "min": 1,
        "mean": 1.0,
        "max": 1
    })
}

fn sample_qwen3_cuda_repeated_comparator_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "qwen3_cuda_repeated_comparator",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-19T16:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "claim": "qwen3_cuda_repeated_comparator",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "dense_gguf_inference_claimed": false,
        "broad_dense_gguf_ready_claimed": false,
        "qwen25_proof_inherited": false,
        "server_ready_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "claim_boundary": {
            "qwen3_cuda_repeated_comparator_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "qwen_chat_cuda_claimed": true,
            "server_ready_claimed": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "broad_dense_gguf_ready_claimed": false,
            "qwen25_proof_inherited": false,
            "bitnet_packed_i2s_qk256_proof": false
        },
        "model": {
            "id": "qwen3-0.6b-instruct-q8_0",
            "architecture": "qwen3",
            "model_family": "qwen",
            "artifact_kind": "dense_gguf",
            "file": "Qwen3-0.6B-Q8_0.gguf",
            "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
        },
        "tokenizer_prompt_authority": {
            "tokenizer_authority": "contract_authoritative",
            "prompt_authority": "contract_authoritative",
            "prompt_template": "qwen-chat-raw-deterministic",
            "prompt_policy": "profile-local deterministic prompts; same tokenizer and prompt policy across all runs",
            "deterministic_prompt": true
        },
        "execution_plan": {
            "planner_version": "cuda-planner-qwen3-product",
            "model_family": "qwen",
            "quantization": "dense_gguf_q8_0_qwen3_product_contract",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "cuda_dense_regular_llm_ops": 8112,
            "cuda_bitnet_qk256_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0
        },
        "proof_inputs": {
            "one_token": qwen3_profile_proof_input("one_token"),
            "short_decode_8": qwen3_profile_proof_input("short_decode_8"),
            "short_decode_32": qwen3_profile_proof_input("short_decode_32"),
            "warm_session_3_turns": qwen3_profile_proof_input("warm_session_3_turns"),
            "decode_128_from_warm_context": qwen3_profile_proof_input("decode_128_from_warm_context")
        },
        "profiles": [
            qwen3_comparator_profile("one_token"),
            qwen3_comparator_profile("short_decode_8"),
            qwen3_comparator_profile("short_decode_32"),
            qwen3_comparator_profile("warm_session_3_turns"),
            qwen3_comparator_profile("decode_128_from_warm_context")
        ],
        "comparator_summary": {
            "status": "repeated_comparator_only",
            "profiles_recorded": 5,
            "min_runs_per_backend": 3,
            "total_cpu_runs": 15,
            "total_cuda_runs": 15,
            "fallback_free": true,
            "same_artifact_sha": true,
            "same_tokenizer_prompt_policy": true,
            "deterministic_generation_policy": true,
            "generated_tokens_compared": true,
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "accepted_speedup_profiles": [],
            "remaining_qualification_blockers": [
                "profile-specific speedup thresholds remain unreviewed",
                "pure host-to-device timing remains separated from the model-load envelope"
            ],
            "next_step": "CUDA-MODEL benchmark qualification review after repeated hardware receipts land"
        },
        "transfer_timing": {
            "status": "host_to_device_model_load_envelope_device_to_host_measured",
            "source": "Qwen3 source receipts record H2D model-load envelopes and D2H wall-clock timing",
            "host_to_device_bytes_recorded": true,
            "device_to_host_bytes_recorded": true,
            "host_to_device_timing_recorded": true,
            "device_to_host_timing_recorded": true,
            "pure_host_to_device_timing_recorded": false
        },
        "hardware_context": {
            "vram_bytes": 17094475776u64,
            "power_draw_watts_min": 32.0,
            "power_draw_watts_max": 50.0,
            "temperature_c_min": 44.0,
            "temperature_c_max": 47.0,
            "source": "NVML fields recorded in Qwen3 strict CUDA proof receipts"
        },
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64
        },
        "claim_boundaries": [
            "speedup_claim=false; repeated CPU/CUDA comparator evidence is not a speedup qualification.",
            "benchmark_qualified_speedup=false until a separate exact-profile review accepts a profile.",
            "Qwen3 repeated comparator evidence cannot inherit Qwen2.5 evidence.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof."
        ]
    })
}

fn qwen3_profile_proof_input(profile: &str) -> serde_json::Value {
    json!({
        "path": format!("ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/{profile}/"),
        "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "artifact_kind": "qwen3_profile_repeated_runs"
    })
}

fn qwen3_comparator_profile(profile: &str) -> serde_json::Value {
    json!({
        "profile": profile,
        "status": "repeated_same_artifact_cpu_cuda_comparator",
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "run_count": 3,
        "cpu_runs": 3,
        "cuda_runs": 3,
        "min_runs_per_backend": 3,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_policy": true,
        "deterministic_generation_policy": true,
        "generated_token_ids_match": true,
        "first_divergence_report": "none",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
        "model_load_ms": dense_qwen_number_summary(),
        "tokenizer_load_ms": dense_qwen_number_summary(),
        "prompt_render_ms": dense_qwen_number_summary(),
        "tokenize_ms": dense_qwen_number_summary(),
        "cuda_context_init_ms": dense_qwen_number_summary(),
        "weight_upload_ms": dense_qwen_number_summary(),
        "cpu_total_ms": dense_qwen_number_summary(),
        "cuda_total_ms": dense_qwen_number_summary(),
        "prefill_ms": dense_qwen_number_summary(),
        "first_token_ms": dense_qwen_number_summary(),
        "decode_total_ms": dense_qwen_number_summary(),
        "steady_tok_per_s": dense_qwen_number_summary(),
        "kernel_time_ms": dense_qwen_number_summary(),
        "launch_count": dense_qwen_u64_summary(),
        "host_to_device_bytes": dense_qwen_u64_summary(),
        "host_to_device_ms": dense_qwen_number_summary(),
        "device_to_host_bytes": dense_qwen_u64_summary(),
        "device_to_host_ms": dense_qwen_number_summary(),
        "vram_high_water_bytes": dense_qwen_u64_summary(),
        "runs": [
            qwen3_comparator_run(profile, 1),
            qwen3_comparator_run(profile, 2),
            qwen3_comparator_run(profile, 3)
        ]
    })
}

fn qwen3_comparator_run(profile: &str, index: u64) -> serde_json::Value {
    let mut run = json!({
        "run_id": format!("run-{index:02}"),
        "profile": profile,
        "source_receipt_path": format!("ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-{index:02}/{profile}.json"),
        "source_receipt_sha256": format!("{:064x}", 100 + index),
        "source_artifact_kind": qwen3_source_artifact_kind(profile),
        "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
        "prompt_template": "qwen-chat-raw-deterministic",
        "prompt_token_count": 8,
        "generation_policy": "greedy",
        "deterministic_generation": true,
        "generated_tokens": qwen3_generated_tokens(profile),
        "generated_token_ids_sha256": "638e3358dd291728156e5d98d5fc9cb7e66afa992c8cd16833c85a552572aa4f",
        "generated_token_ids_match": true,
        "first_divergence_report": "none",
        "top_k_compared": true,
        "fallback_used": false,
        "quality_passed": true,
        "parity_passed": true,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "timing": {
            "model_load_ms": 4000.0,
            "tokenizer_load_ms": 100.0,
            "prompt_render_ms": 1.0,
            "tokenize_ms": 1.0,
            "cuda_context_init_ms": 10.0,
            "weight_upload_ms": 3900.0,
            "cpu_total_ms": 3000.0,
            "cuda_total_ms": 5000.0,
            "prefill_ms": 100.0,
            "first_token_ms": 100.0,
            "decode_total_ms": 100.0,
            "steady_tok_per_s": 10.0,
            "kernel_time_ms": 90.0,
            "launch_count": 1,
            "kernel_invocations": 1,
            "host_to_device_bytes": 639446688u64,
            "host_to_device_ms": 3900.0,
            "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
            "device_to_host_bytes": 607744u64,
            "device_to_host_ms": 1.0,
            "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
            "vram_high_water_bytes": 17094475776u64,
            "power_temperature_context": "NVML power and temperature sampled during source receipt"
        }
    });
    if profile == "warm_session_3_turns" {
        run["turns_count"] = json!(3);
    }
    if profile == "decode_128_from_warm_context" {
        run["warm_context_reused"] = json!(true);
    }
    run
}

fn qwen3_source_artifact_kind(profile: &str) -> &'static str {
    match profile {
        "one_token" => "dense_gguf_qwen_one_token_strict_cuda_proof",
        "warm_session_3_turns" => "dense_gguf_qwen_warm_session_strict_cuda_proof",
        "decode_128_from_warm_context" => "dense_gguf_qwen_warm_decode_strict_cuda_proof",
        _ => "dense_gguf_qwen_short_decode_strict_cuda_proof",
    }
}

fn qwen3_generated_tokens(profile: &str) -> u64 {
    match profile {
        "one_token" => 1,
        "short_decode_8" => 8,
        "short_decode_32" => 32,
        "warm_session_3_turns" => 24,
        "decode_128_from_warm_context" => 128,
        _ => 0,
    }
}

fn sample_dense_gguf_qwen_benchmark_qualification_receipt() -> serde_json::Value {
    let comparator = sample_dense_gguf_qwen_repeated_comparator_receipt();
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_qwen_benchmark_qualification_review",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-10T18:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "claim": "dense_gguf_qwen_benchmark_qualification_review",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "dense_gguf_inference_claimed": false,
        "server_ready_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "claim_boundary": {
            "dense_gguf_qwen_benchmark_qualification_review_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "qwen_chat_cuda_claimed": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false
        },
        "model": comparator["model"].clone(),
        "tokenizer_prompt_authority": comparator["tokenizer_prompt_authority"].clone(),
        "execution_plan": comparator["execution_plan"].clone(),
        "proof_inputs": {
            "benchmark_baseline": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json",
                "sha256": "d84b094b29763d96820f69479394a3e7770ef9745ff4e14e7fad4cc36a6805c1",
                "artifact_kind": "dense_gguf_qwen_cuda_benchmark_baseline"
            },
            "repeated_comparator": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json",
                "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "artifact_kind": "dense_gguf_qwen_repeated_comparator"
            },
            "one_token_transfer_timing": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json",
                "sha256": "0b74e8d094a341a10710e3ec7c70a94062f67246bdff4f30fd3e9612c94c4ec4",
                "artifact_kind": "dense_gguf_qwen_one_token_strict_cuda_proof"
            },
            "short_decode_transfer_timing": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json",
                "sha256": "f72609cf52764f5e05aa0c981622c5333720794bcb87be59810722cf35beb502",
                "artifact_kind": "dense_gguf_qwen_short_decode_strict_cuda_proof"
            },
            "warm_session_transfer_timing": {
                "path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json",
                "sha256": "ecf07d473bfdad091b806887e0744cc075af331e306b7a2fb9ad6c55d2f5d6ff",
                "artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof"
            }
        },
        "qualification_decision": {
            "status": "not_accepted",
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "accepted_profiles": [],
            "blocked_profiles": ["one_token", "short_decode_8", "warm_session_3_turns"],
            "reason": "Reviewed dense Qwen profiles are fallback-free and repeated, but CUDA mean total time is slower than CPU mean total time and H2D timing is still unmeasured."
        },
        "qualification_requirements": [
            {
                "id": "same_artifact_tokenizer_prompt_policy",
                "description": "Same artifact SHA, tokenizer authority, prompt template, deterministic policy, and fallback-free CPU/CUDA route.",
                "status": "passed"
            },
            {
                "id": "repeated_cpu_cuda_comparator",
                "description": "At least three CPU and CUDA runs exist for every reviewed dense Qwen profile.",
                "status": "passed"
            },
            {
                "id": "device_to_host_transfer_timing",
                "description": "Device-to-host logits transfer timing is measured for one-token, short-decode, and warm-session runtime receipts.",
                "status": "passed"
            },
            {
                "id": "host_to_device_transfer_timing",
                "description": "Host-to-device timing is measured, not only host-to-device byte counts.",
                "status": "blocked",
                "blocker": "CUDA-DENSE-PERF-003 keeps host_to_device_ms explicitly null with not_measured_by_dense_qwen_runtime source fields."
            },
            {
                "id": "profile_outperforms_cpu_reference",
                "description": "Each profile's CUDA mean total time is faster than the same-artifact CPU reference mean total time.",
                "status": "blocked",
                "blocker": "The reviewed one-token, short-decode, and warm-session CUDA means are all slower than their CPU reference means."
            },
            {
                "id": "profile_specific_thresholds",
                "description": "Profile-specific speedup thresholds are accepted before benchmark_qualified_speedup may become true.",
                "status": "blocked",
                "blocker": "No dense Qwen profile-specific speedup threshold has been accepted."
            }
        ],
        "profile_reviews": [
            dense_qwen_qualification_profile("one_token", 2872.8428, 3978.5710, 0.8534),
            dense_qwen_qualification_profile("short_decode_8", 3528.0687, 4199.9896, 6.3089),
            dense_qwen_qualification_profile("warm_session_3_turns", 4596.1352, 5034.9288, 18.7415)
        ],
        "evidence_summary": {
            "one_token": dense_qwen_qualification_evidence("one_token", 2872.8428, 3978.5710, 0.8534),
            "short_decode_8": dense_qwen_qualification_evidence("short_decode_8", 3528.0687, 4199.9896, 6.3089),
            "warm_session_3_turns": dense_qwen_qualification_evidence("warm_session_3_turns", 4596.1352, 5034.9288, 18.7415)
        },
        "transfer_timing_review": {
            "status": "device_to_host_measured_host_to_device_unmeasured",
            "device_to_host_timing_recorded": true,
            "host_to_device_timing_recorded": false,
            "host_to_device_blocker": "The dense Qwen runtime still records H2D bytes but not H2D elapsed timing.",
            "device_to_host_source": "wall_clock_extract_logits_2d_local",
            "host_to_device_source": "not_measured_by_dense_qwen_runtime"
        },
        "hardware_context": comparator["hardware_context"].clone(),
        "cuda": comparator["cuda"].clone(),
        "claim_boundaries": [
            "speedup_claim=false; no dense Qwen profile is upgraded by this review.",
            "benchmark_qualified_speedup=false; current CUDA means are slower than same-artifact CPU means.",
            "H2D transfer timing remains explicitly unmeasured.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof."
        ]
    })
}

fn sample_dense_gguf_qwen_benchmark_qualification_receipt_with_h2d_envelope() -> serde_json::Value {
    let mut receipt = sample_dense_gguf_qwen_benchmark_qualification_receipt();
    for profile in receipt["profile_reviews"].as_array_mut().unwrap() {
        add_h2d_envelope_fields(profile, 100.0);
    }
    for profile in receipt["evidence_summary"].as_object_mut().unwrap().values_mut() {
        add_h2d_envelope_fields(profile, 100.0);
    }
    receipt["qualification_decision"]["reason"] = json!(
        "Reviewed dense Qwen profiles are fallback-free and repeated with H2D model-load envelope timing, but CUDA mean total time remains slower than CPU mean total time and pure H2D copy timing remains unmeasured."
    );
    receipt["qualification_requirements"][3] = json!({
        "id": "host_to_device_model_load_envelope",
        "description": "Host-to-device model-load wall-clock envelope timing is recorded with explicit non-transfer-overhead labeling.",
        "status": "passed"
    });
    receipt["qualification_requirements"]
            .as_array_mut()
            .unwrap()
            .insert(
                4,
                json!({
                    "id": "pure_host_to_device_transfer_timing",
                    "description": "Pure host-to-device copy timing is measured separately from model-load and upload overhead.",
                    "status": "blocked",
                    "blocker": "CUDA-DENSE-PERF-005 records a model-load wall-clock envelope, not pure CUDA event copy timing."
                }),
            );
    receipt["transfer_timing_review"] = json!({
        "status": "host_to_device_model_load_envelope_device_to_host_measured",
        "device_to_host_timing_recorded": true,
        "host_to_device_timing_recorded": true,
        "host_to_device_model_load_envelope_recorded": true,
        "host_to_device_pure_transfer_timing_recorded": false,
        "host_to_device_blocker": "The dense Qwen runtime records an H2D model-load wall-clock envelope, but not pure CUDA event copy timing.",
        "device_to_host_source": "wall_clock_extract_logits_2d_local",
        "host_to_device_source": "wall_clock_model_load_with_cuda_weight_upload",
        "host_to_device_scope": "model_load_wall_clock_envelope",
        "host_to_device_ms_includes_non_transfer_overhead": true
    });
    receipt["claim_boundaries"][2] = json!(
        "H2D model-load envelope timing is recorded, but pure CUDA event H2D copy timing remains unmeasured."
    );
    receipt
}

fn add_h2d_envelope_fields(profile: &mut serde_json::Value, h2d_ms: f64) {
    profile["host_to_device_ms"] = json!(h2d_ms);
    profile["host_to_device_ms_source"] = json!("wall_clock_model_load_with_cuda_weight_upload");
    profile["host_to_device_ms_scope"] = json!("model_load_wall_clock_envelope");
    profile["host_to_device_ms_includes_non_transfer_overhead"] = json!(true);
    profile["pure_host_to_device_ms"] = serde_json::Value::Null;
    profile["pure_host_to_device_ms_source"] = json!("not_measured_by_dense_qwen_runtime");
}

fn dense_qwen_qualification_profile(
    profile: &str,
    cpu_mean_ms: f64,
    cuda_mean_ms: f64,
    d2h_ms: f64,
) -> serde_json::Value {
    json!({
        "profile": profile,
        "decision": "not_accepted",
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": true,
        "quality_passed": true,
        "generated_token_ids_match": true,
        "dense_cuda_evidence_used": true,
        "runs_per_backend": 3,
        "cpu_total_ms_mean": cpu_mean_ms,
        "cuda_total_ms_mean": cuda_mean_ms,
        "observed_cpu_total_ms_div_cuda_total_ms": cpu_mean_ms / cuda_mean_ms,
        "cuda_mean_slower_than_cpu": true,
        "host_to_device_ms": null,
        "host_to_device_ms_source": "not_measured_by_dense_qwen_runtime",
        "device_to_host_ms": d2h_ms,
        "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
        "reason": "This profile remains baseline/comparator evidence because CUDA mean total time is not faster than the same-artifact CPU reference and H2D timing is incomplete.",
        "blockers": [
            "CUDA mean total time is slower than CPU mean total time",
            "host-to-device transfer timing is unmeasured",
            "no profile-specific speedup threshold has been accepted"
        ]
    })
}

fn dense_qwen_qualification_evidence(
    profile: &str,
    cpu_mean_ms: f64,
    cuda_mean_ms: f64,
    d2h_ms: f64,
) -> serde_json::Value {
    json!({
        "profile": profile,
        "runs_per_backend": 3,
        "fallback_free": true,
        "quality_passed": true,
        "generated_token_ids_match": true,
        "cpu_total_ms_mean": cpu_mean_ms,
        "cuda_total_ms_mean": cuda_mean_ms,
        "observed_cpu_total_ms_div_cuda_total_ms": cpu_mean_ms / cuda_mean_ms,
        "cuda_mean_slower_than_cpu": true,
        "device_to_host_ms": d2h_ms,
        "host_to_device_ms": null,
        "host_to_device_ms_source": "not_measured_by_dense_qwen_runtime",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false
    })
}

fn sample_strict_cuda_benchmark_qualification_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_benchmark_qualification_review",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-08T20:05:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "strict_cuda_benchmark_qualification_review",
        "fallback_used": false,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "loader_mode": "strict_real_gguf",
            "fallback_loader_used": false
        },
        "tokenizer": {
            "source": "explicit",
            "strict": true,
            "type": "llama3",
            "pretokenizer_authority": "llama-bpe"
        },
        "prompt_template": {
            "family": "bitnetcpp-answer",
            "rendered_sha256": "dee5b2fff5b96df948252b7a589ab7ea1a6b6a10ed1b2d9ed70a63ebbde554f3"
        },
        "proof_inputs": {
            "answer_path_baseline_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-prod-004-answer-path-benchmark.json",
            "repeated_strict_ask_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json",
            "warm_session_benchmark_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json",
            "cpu_cuda_answer_parity_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json"
        },
        "qualification_decision": {
            "status": "not_accepted",
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "accepted_profiles": [],
            "blocked_profiles": [
                "strict_ask_math_8",
                "strict_cuda_warm_session_2_turns"
            ],
            "reason": "Current evidence is strong baseline data, but speedup acceptance still lacks decode-profile repetitions, profile-specific acceptance thresholds, and complete transfer timing/power coverage."
        },
        "qualification_requirements": [
            {
                "id": "same_model_tokenizer_prompt_policy",
                "description": "Same official model, tokenizer authority, prompt template, deterministic policy, and fallback-free CPU/CUDA path.",
                "status": "passed"
            },
            {
                "id": "repeated_strict_ask_cpu_cuda",
                "description": "Repeated strict ask measurements exist for CPU AVX-512 and RTX 5070 Ti CUDA.",
                "status": "passed"
            },
            {
                "id": "warm_session_cuda_repeated",
                "description": "Repeated strict CUDA warm-session measurements exist with load/context/upload reuse.",
                "status": "passed"
            },
            {
                "id": "decode_profile_repeated_cpu_cuda",
                "description": "Repeated same-model decode profile measurements exist for CPU AVX-512 and CUDA.",
                "status": "blocked",
                "blocker": "No repeated prefill/decode profile receipt is committed for CPU AVX-512 and CUDA."
            },
            {
                "id": "transfer_timing",
                "description": "Host/device transfer timing is measured, not only byte counts.",
                "status": "blocked",
                "blocker": "Current receipts record QK256 transfer bytes but transfer_timing_claimed remains false."
            },
            {
                "id": "power_thermal_complete",
                "description": "Driver, runtime, VRAM, power, and thermal context are complete for reviewed profiles.",
                "status": "blocked",
                "blocker": "Warm-session has power/thermal samples; repeated strict ask has null power and temperature fields."
            }
        ],
        "profile_reviews": [
            {
                "profile": "strict_ask_math_8",
                "decision": "not_accepted",
                "speedup_claim_allowed": false,
                "benchmark_qualified_speedup": false,
                "fallback_free": true,
                "quality_passed": true,
                "dense_cuda_evidence_used": false,
                "reason": "The repeated strict ask ratio is baseline evidence only until acceptance thresholds and transfer/power evidence are complete.",
                "blockers": [
                    "transfer timing not measured",
                    "power and thermal samples incomplete",
                    "no explicit benchmark qualification threshold accepted"
                ]
            },
            {
                "profile": "strict_cuda_warm_session_2_turns",
                "decision": "not_accepted",
                "speedup_claim_allowed": false,
                "benchmark_qualified_speedup": false,
                "fallback_free": true,
                "quality_passed": true,
                "dense_cuda_evidence_used": false,
                "reason": "The warm-session receipt is CUDA-only baseline evidence and has no same-profile CPU AVX-512 comparator.",
                "blockers": [
                    "CPU AVX-512 warm-session comparator absent",
                    "speedup acceptance threshold not reviewed"
                ]
            }
        ],
        "evidence_summary": {
            "strict_ask_math_8": {
                "runs_per_backend": 3,
                "cpu_avx512_median_total_ms": 18797.0,
                "cuda_median_total_ms": 2136.0,
                "observed_median_cpu_total_ms_div_cuda_total_ms": 8.800093632958802,
                "cpu_cuda_answer_match": true,
                "fallback_free": true,
                "qk256_kernel_time_ms": 2977.406,
                "host_to_device_bytes": 168376320u64,
                "device_to_host_bytes": 172247040u64,
                "speedup_claim": false
            },
            "strict_cuda_warm_session_2_turns": {
                "cuda_runs": 3,
                "turns_per_run": 2,
                "cuda_median_total_session_ms": 8038.352,
                "cuda_median_kernel_time_ms": 2036.619,
                "cuda_median_generated_tokens_per_second": 1.368439700077827,
                "fallback_free": true,
                "model_tokenizer_context_loaded_once": true,
                "qk256_weights_uploaded_once": true,
                "speedup_claim": false
            }
        },
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64,
            "memory_hwm_bytes": 10078912512u64
        },
        "next_required_evidence": [
            "repeated CPU AVX-512 and CUDA decode-profile receipt",
            "host/device transfer timing fields, not only byte counters",
            "complete power and thermal samples for strict ask repeats",
            "profile-specific speedup acceptance thresholds"
        ],
        "claim_boundaries": [
            "speedup_claim=false; no reviewed profile is upgraded by this receipt.",
            "This review uses only BitNet packed QK256 evidence and explicitly excludes dense regular-LLM CUDA proof.",
            "This review does not claim broad chat quality, production server readiness, or full CUDA residency."
        ],
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-004-benchmark-qualification.json"
    })
}

fn sample_strict_cuda_product_benchmark_qualification_receipt() -> serde_json::Value {
    let mut receipt = sample_strict_cuda_benchmark_qualification_receipt();
    let strict_ask_evidence = receipt["evidence_summary"]["strict_ask_math_8"].clone();
    let warm_session_evidence =
        receipt["evidence_summary"]["strict_cuda_warm_session_2_turns"].clone();
    receipt["target_profiles"] = json!([
        "one_token",
        "short_decode_8",
        "short_decode_32",
        "warm_session_3_turns",
        "warm_session_10_turns"
    ]);
    receipt["benchmark_policy"] = json!({
        "profile_specific_decisions_only": true,
        "global_speedup_claim": false,
        "dense_cuda_evidence_used": false,
        "bitnet_packed_i2s_qk256_only": true
    });
    receipt["qualification_decision"]["blocked_profiles"] = json!([
        "one_token",
        "short_decode_8",
        "short_decode_32",
        "warm_session_3_turns",
        "warm_session_10_turns"
    ]);
    receipt["profile_reviews"] = json!([
        strict_cuda_product_profile_review("one_token", "missing"),
        strict_cuda_product_profile_review("short_decode_8", "single_run_baseline"),
        strict_cuda_product_profile_review("short_decode_32", "missing"),
        strict_cuda_product_profile_review("warm_session_3_turns", "missing"),
        strict_cuda_product_profile_review("warm_session_10_turns", "missing")
    ]);
    receipt["evidence_summary"] = json!({
        "strict_ask_math_8": strict_ask_evidence,
        "strict_cuda_warm_session_2_turns": warm_session_evidence,
        "one_token": strict_cuda_product_missing_evidence("one_token"),
        "short_decode_8": {
            "profile": "short_decode_8",
            "decision": "not_accepted",
            "evidence_status": "single_run_baseline",
            "fallback_free": true,
            "quality_passed": true,
            "cpu_cuda_output_match": true,
            "cpu_total_ms": 147593.0,
            "cuda_total_ms": 1866.0,
            "observed_cpu_total_ms_div_cuda_total_ms": 79.09592711682744,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "reason": "One short-decode CPU/CUDA baseline exists, but profile-specific speedup remains blocked until repeated governed profile evidence and transfer timing are reviewed.",
            "blockers": [
                "profile is not repeated",
                "transfer timing is incomplete",
                "no profile-specific speedup threshold has been accepted"
            ]
        },
        "short_decode_32": strict_cuda_product_missing_evidence("short_decode_32"),
        "warm_session_3_turns": strict_cuda_product_missing_evidence("warm_session_3_turns"),
        "warm_session_10_turns": strict_cuda_product_missing_evidence("warm_session_10_turns")
    });
    receipt
}

fn strict_cuda_product_profile_review(profile: &str, evidence_status: &str) -> serde_json::Value {
    let has_evidence = evidence_status != "missing";
    json!({
        "profile": profile,
        "decision": "not_accepted",
        "evidence_status": evidence_status,
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": has_evidence,
        "quality_passed": has_evidence,
        "dense_cuda_evidence_used": false,
        "reason": if has_evidence {
            "Profile evidence exists but remains baseline-only until repeated governed proof and transfer timing are complete."
        } else {
            "Required product benchmark profile evidence is not committed yet."
        },
        "blockers": if has_evidence {
            json!([
                "profile is not repeated",
                "transfer timing is incomplete",
                "no profile-specific speedup threshold has been accepted"
            ])
        } else {
            json!(["profile receipt missing"])
        }
    })
}

fn strict_cuda_product_missing_evidence(profile: &str) -> serde_json::Value {
    json!({
        "profile": profile,
        "decision": "not_accepted",
        "evidence_status": "missing",
        "fallback_free": false,
        "quality_passed": false,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "reason": "Required product benchmark profile evidence is not committed yet.",
        "blockers": ["profile receipt missing"]
    })
}

fn sample_cpu_benchmark_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "cpu_benchmark",
        "machine_id": "intel-i5-8250u-cpu-avx2",
        "hardware_lane": "intel-i5-8250u-cpu-avx2",
        "timestamp_utc": "2026-05-06T00:00:00Z",
        "requested_backend": "cpu",
        "selected_backend": "intel-i5-8250u-cpu-avx2",
        "runtime_api": "cpu",
        "claim": "cpu_benchmark_receipt",
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_reason": null,
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "abc123def456",
            "family": "bitnet",
            "quant_format": "QK256/I2_S"
        },
        "tokenizer": {
            "source": "gguf_metadata",
            "strict": true
        },
        "kernel": {
            "kernel_family": "i2_s_qk256",
            "requested_kernel": "qk256-avx2-gemv",
            "selected_kernel": "qk256-avx2-gemv",
            "oracle_kernel": "qk256-scalar-gemv",
            "gemm_oracle_kernel": "qk256-scalar-gemm",
            "fallback_used": false,
            "fallback_reason": null,
            "dequantizes_before_compute": false
        },
        "cpu": {
            "model": "Intel Core i5-8250U",
            "arch": "x86_64",
            "features": ["avx2", "fma"],
            "threads": 8,
            "avx512": false,
            "power_mode": "unknown",
            "temperature_c": null,
            "frequency_mhz": null
        },
        "workload": {
            "prompt_tokens": 512,
            "generated_tokens": 128,
            "batch_size": 1
        },
        "i2s_microbench": {
            "work_item": "CPU-BITNET-PERF-001",
            "artifact_kind": "cpu_bitnet_i2s_microbench",
            "claim": "i2_s_gemv_gemm_microbench_receipt",
            "kernel_family": "i2_s_qk256",
            "quantization": "QK256/I2_S",
            "speedup_claim": false,
            "fallback_used": false,
            "fallback_reason": null,
            "profiles": [
                measured_i2s_microbench_profile("gemv"),
                measured_i2s_microbench_profile("gemm")
            ],
            "claim_boundary": [
                "Records QK256/I2_S GEMV and GEMM microbench timing only.",
                "Does not claim answer quality, sustained decode throughput, Arc/NPU execution, acceleration, or QK256 semantic changes."
            ]
        },
        "i2s_tiling_thread_matrix": {
            "work_item": "CPU-BITNET-PERF-002",
            "artifact_kind": "cpu_bitnet_i2s_tiling_thread_matrix",
            "claim": "i2_s_tiling_thread_matrix_receipt",
            "kernel_family": "i2_s_qk256",
            "quantization": "QK256/I2_S",
            "speedup_claim": false,
            "fallback_used": false,
            "fallback_reason": null,
            "candidate_grid": {
                "parallelism_degrees": [2, 4, 8],
                "row_blocks": [2, 4, 8, 16],
                "col_blocks": [64, 128, 256, 512],
                "thread_counts": [1, 2, 4, 6, 8],
                "candidate_count": 240
            },
            "coverage": {
                "status": "sampled_baseline",
                "measured_candidate_count": 2,
                "full_matrix_candidate_count": 240,
                "thread_counts_recorded_not_applied": true,
                "reason": "Sample fixture records candidate coverage without upgrading any profile to a speedup claim."
            },
            "measured_runs": [
                measured_i2s_tiling_matrix_run("gemv"),
                measured_i2s_tiling_matrix_run("gemm")
            ],
            "claim_boundary": [
                "Records a Lunar Lake QK256/I2_S tiling/thread candidate matrix and sampled GEMV/GEMM timings.",
                "Does not claim answer quality, sustained decode throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness."
            ]
        },
        "embedding_quantization_evidence": sample_embedding_quantization_evidence(),
        "i2s_applied_thread_matrix": sample_i2s_applied_thread_matrix(),
        "profiles": [
            measured_cpu_profile("micro"),
            measured_cpu_profile("layer"),
            measured_cpu_profile("prefill"),
            measured_cpu_profile("first_token"),
            measured_cpu_profile("decode")
        ],
        "artifact_path": "ci/hardware/intel-i5-8250u-cpu-avx2/benchmark-receipt.json"
    })
}

fn sample_i2s_applied_thread_matrix() -> serde_json::Value {
    json!({
        "work_item": "CPU-BITNET-PERF-003",
        "artifact_kind": "cpu_bitnet_i2s_applied_thread_matrix",
        "claim": "i2_s_applied_thread_matrix_receipt",
        "kernel_family": "i2_s_qk256",
        "quantization": "QK256/I2_S",
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_reason": null,
        "candidate_grid": {
            "parallelism_degrees": [2, 4, 8],
            "row_blocks": [2, 4, 8, 16],
            "col_blocks": [64, 128, 256, 512],
            "thread_counts": [1, 2, 4, 6, 8],
            "candidate_count": 240
        },
        "coverage": {
            "status": "sampled_applied_thread_baseline",
            "measured_candidate_count": 2,
            "full_matrix_candidate_count": 240,
            "thread_counts_applied": true,
            "thread_count_policy": "applied_scoped_threads",
            "thread_partitions": ["rows", "tokens"],
            "reason": "Sample fixture applies thread counts without upgrading any profile to a speedup claim."
        },
        "measured_runs": [
            measured_i2s_applied_thread_matrix_run("gemv"),
            measured_i2s_applied_thread_matrix_run("gemm")
        ],
        "claim_boundary": [
            "Records sampled Lunar Lake QK256/I2_S GEMV/GEMM timings with scoped worker threads applied inside the synthetic microbench.",
            "Does not claim answer quality, sustained decode throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness.",
            "Does not claim the full BitNet runtime applies this worker-thread policy outside this benchmark receipt."
        ]
    })
}

fn sample_embedding_quantization_evidence() -> serde_json::Value {
    json!({
        "work_item": "CPU-BITNET-EMBD-001",
        "artifact_kind": "cpu_bitnet_embedding_quantization_evidence",
        "claim": "bitnet_embedding_quantization_evidence_receipt",
        "source_tensor_boundary_audit": "ci/hardware/intel-258v/2026-05-08/output-head-logits-index-audit.json",
        "target_quantization": "Q6_K",
        "fallback_used": false,
        "fallback_reason": null,
        "speedup_claim": false,
        "answer_quality_claim": false,
        "acceleration_claim": false,
        "qk256_semantic_change_claim": false,
        "current_embedding": {
            "name": "token_embd.weight",
            "tensor_type": "F16",
            "shape": [2560, 128256],
            "size_bytes": 656670720
        },
        "current_embedding_quantization": "F16",
        "current_artifact_contains_q6_k_embedding": false,
        "q6_k_embedding_proven": false,
        "evidence_status": "q6_k_embedding_not_present_in_current_canonical_artifact",
        "loader_scope": {
            "q6_k_tensor_type_known": true,
            "q6_k_dense_standard_dequantizer_present": true,
            "q6_k_embedding_operating_path": "not_applied_to_current_bitnet_artifact",
            "note": "Sample fixture records the Q6_K embedding evidence contract without claiming it is active."
        },
        "recommended_next_step": "Acquire or generate a canonical BitNet b1.58 Q6_K embedding variant before claiming embedding-quantization support.",
        "claim_boundary": [
            "Records BitNet embedding tensor quantization evidence from the committed 258V tensor boundary audit.",
            "Does not claim answer quality, speedup, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness.",
            "Does not claim Q6_K embedding quantization is active unless the current canonical BitNet artifact records a Q6_K embedding tensor."
        ]
    })
}

fn sample_bitnet_qk256_execution_plan() -> serde_json::Value {
    json!({
        "planner_version": "cuda-planner-004",
        "model_family": "bitnet_b1_58",
        "quantization": "i2_s_qk256",
        "selected_route": "bitnet_qk256_cuda",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "strict_fallback_policy": "reject",
        "dense_regular_llm_cuda": false,
        "bitnet_packed_qk256_cuda": true,
        "cuda_bitnet_qk256_ops": 4410,
        "cuda_dense_regular_llm_ops": 0,
        "cpu_fallback_ops": 0,
        "unsupported_ops": 0,
        "total_ops": 4410,
        "cuda_ops": 4410,
        "mixed_cuda_routes": false,
        "fallback_used": false,
        "strict_cuda_ready": true,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    })
}

fn sample_strict_cuda_warm_session_benchmark_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_warm_session_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-08T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "claim": "strict_cuda_warm_session_benchmark_baseline",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "execution_plan": sample_bitnet_qk256_execution_plan(),
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "loader_mode": "strict_real_gguf",
            "fallback_loader_used": false
        },
        "tokenizer": {
            "source": "explicit",
            "strict": true,
            "type": "llama3",
            "model_family": "llama3",
            "pretokenizer_authority": "llama-bpe"
        },
        "generation": {
            "prompt_template": "bitnetcpp-answer",
            "mode": "greedy",
            "deterministic": true,
            "temperature": 0.0,
            "max_new_tokens": 8
        },
        "session_contract": {
            "runs_per_backend": 2,
            "turn_count": 2,
            "same_model": true,
            "same_tokenizer": true,
            "same_prompts": true,
            "same_sampling_policy": true,
            "fallback_free": true,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "cuda_context_initialized_once": true,
            "qk256_weights_uploaded_once": true,
            "per_token_weight_upload": false,
            "kv_cache_reuse_policy": "recreated_per_turn_for_prompt_isolation",
            "kv_cache_reuse_claimed": false,
            "speedup_claim": false
        },
        "workload": {
            "profile": "strict_cuda_warm_session_2_turns",
            "turn_count": 2,
            "generated_tokens_total": 11,
            "prompt_tokens_total": 34,
            "quality_passed": true,
            "prompts": [
                {
                    "turn_index": 1,
                    "prompt": "What is 2+2? Answer with only the number.",
                    "expected_answer_scope": "exact_trimmed_4"
                },
                {
                    "turn_index": 2,
                    "prompt": "Answer yes or no: is water wet?",
                    "expected_answer_scope": "quality_gate"
                }
            ],
            "answers": [
                " 4",
                " No. Water is not wet; it"
            ]
        },
        "benchmark": {
            "profile": "strict_cuda_warm_session_2_turns",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "runs_per_backend": 2,
            "turns_per_run": 2,
            "cuda_median_total_session_ms": 8000.0,
            "cuda_median_kernel_time_ms": 2030.0,
            "cuda_median_generated_tokens_per_second": 1.37,
            "cuda_median_host_to_device_bytes": 114923520,
            "cuda_median_device_to_host_bytes": 117565440,
            "quality_passed": true,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false
        },
        "summary": {
            "backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "runs": 2,
            "quality_passed": true,
            "fallback_used": false,
            "total_session_ms": warm_session_metric_summary(),
            "model_load_ms": warm_session_metric_summary(),
            "tokenizer_load_ms": warm_session_metric_summary(),
            "cuda_probe_ms": warm_session_metric_summary(),
            "kernel_time_ms": warm_session_metric_summary(),
            "generated_tokens_per_second": warm_session_metric_summary(),
            "host_to_device_bytes": warm_session_u64_summary(),
            "device_to_host_bytes": warm_session_u64_summary(),
            "memory_hwm_bytes": warm_session_u64_summary()
        },
        "runs": [
            warm_session_run(1),
            warm_session_run(2)
        ],
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64,
            "memory_hwm_bytes": 10078912512u64,
            "cuda_kernel_invocations": 18060
        },
        "kernel_stats": [
            {
                "kernel_id": "qk256_gemv_cuda",
                "invocations": 18060,
                "fallback_invocations": 0,
                "kernel_launches": 18060,
                "kernel_time_ms": 4060.0,
                "host_to_device_bytes": 229847040u64,
                "device_to_host_bytes": 235130880u64
            }
        ],
        "cuda_execution_residency": {
            "schema_version": "1.0.0",
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "host_device_transfer_accounting": {
                "status": "qk256_measured",
                "host_to_device_bytes": 229847040u64,
                "device_to_host_bytes": 235130880u64,
                "kernel_time_ms": 4060.0
            }
        },
        "claim_boundaries": [
            "speedup_claim=false; repeated warm-session timing remains baseline evidence only.",
            "This receipt does not claim broad chat quality, production server readiness, or full CUDA residency."
        ],
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json"
    })
}

fn warm_session_metric_summary() -> serde_json::Value {
    json!({
        "samples": 2,
        "min": 1.0,
        "max": 2.0,
        "mean": 1.5,
        "median": 1.5
    })
}

fn warm_session_u64_summary() -> serde_json::Value {
    json!({
        "samples": 2,
        "min": 4096,
        "max": 8192,
        "mean": 6144.0,
        "median": 6144.0
    })
}

fn warm_session_run(index: u64) -> serde_json::Value {
    json!({
        "profile": "strict_cuda_warm_session_2_turns",
        "backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "status": "measured",
        "repeat_index": index,
        "source_receipt_path": format!("target/bitnet/receipts/cuda-bitnet-perf-003/cuda-warm-session-run-{index}.json"),
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "kernel_id": "qk256_gemv_cuda",
        "quality_passed": true,
        "fallback_used": false,
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "cuda_context_initialized_once": true,
        "qk256_weights_uploaded_once": true,
        "per_token_weight_upload": false,
        "turn_count": 2,
        "generated_tokens_total": 11,
        "prompt_tokens_total": 34,
        "total_session_ms": 8000.0 + index as f64,
        "model_load_ms": 2600.0,
        "tokenizer_load_ms": 330.0,
        "cuda_probe_ms": 250.0,
        "kernel_time_ms": 2030.0,
        "generated_tokens_per_second": 1.37,
        "kernel_invocations": 9030,
        "host_to_device_bytes": 114923520u64,
        "device_to_host_bytes": 117565440u64,
        "memory_hwm_bytes": 10078912512u64,
        "turns": [
            warm_session_turn(1, "4", 3, 19, 1000.0, 56125440, 57415680),
            warm_session_turn(2, "No. Water is not wet; it", 8, 15, 1030.0, 58798080, 60149760)
        ]
    })
}

fn warm_session_turn(
    index: u64,
    answer: &str,
    generated_tokens: u64,
    prompt_tokens: u64,
    kernel_time_ms: f64,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
) -> serde_json::Value {
    json!({
        "turn_index": index,
        "answer_trimmed": answer,
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "quality_passed": true,
        "fallback_used": false,
        "kernel_time_ms": kernel_time_ms,
        "host_to_device_bytes": host_to_device_bytes,
        "device_to_host_bytes": device_to_host_bytes
    })
}

fn sample_strict_cuda_repeated_ask_benchmark_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_repeated_ask_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-08T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "strict_cuda_repeated_ask_benchmark_baseline",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "execution_plan": sample_bitnet_qk256_execution_plan(),
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "loader_mode": "strict_real_gguf",
            "fallback_loader_used": false
        },
        "tokenizer": {
            "source": "explicit",
            "strict": true,
            "type": "llama3",
            "pretokenizer_authority": "llama-bpe"
        },
        "prompt_template": {
            "family": "bitnetcpp-answer",
            "rendered_sha256": "dee5b2fff5b96df948252b7a589ab7ea1a6b6a10ed1b2d9ed70a63ebbde554f3"
        },
        "workload": {
            "profile": "strict_ask_math_8",
            "question": "What is 2+2? Answer with only the number.",
            "answer": " 4",
            "prompt_tokens": 19,
            "generated_tokens": 3,
            "quality_passed": true,
            "cpu_cuda_answer_match": true,
            "cpu_cuda_generated_ids_match": true
        },
        "repeat_policy": {
            "runs_per_backend": 2,
            "cold_warm_split": "process-level repeated strict ask; each run reloads the model and reinitializes backend state",
            "same_model": true,
            "same_tokenizer": true,
            "same_prompt_template": true,
            "same_question": true,
            "same_sampling_policy": true,
            "fallback_free": true,
            "speedup_claim": false
        },
        "benchmark": {
            "profile": "strict_ask_math_8",
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "runs_per_backend": 2,
            "cpu_avx512_median_total_ms": 19450.0,
            "cuda_median_total_ms": 1830.0,
            "observed_median_cpu_total_ms_div_cuda_total_ms": 10.6284,
            "cpu_cuda_answer_match": true,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false
        },
        "summary": {
            "cpu_avx512": repeated_backend_summary("amd-9950x3d-cpu-avx512", "cpu", false),
            "cuda": repeated_backend_summary("nvidia-rtx-5070-ti-cuda", "cuda", true)
        },
        "runs": [
            repeated_run(1, "amd-9950x3d-cpu-avx512", "cpu", "i2_s-avx512-reference"),
            repeated_run(2, "amd-9950x3d-cpu-avx512", "cpu", "i2_s-avx512-reference"),
            repeated_run(1, "nvidia-rtx-5070-ti-cuda", "cuda", "qk256_gemv_cuda"),
            repeated_run(2, "nvidia-rtx-5070-ti-cuda", "cuda", "qk256_gemv_cuda")
        ],
        "pair_contracts": [
            repeated_pair_contract(1),
            repeated_pair_contract(2)
        ],
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64,
            "memory_hwm_bytes": 9201254400u64,
            "cuda_kernel_invocations": 8820
        },
        "kernel_stats": [
            {
                "kernel_id": "qk256_gemv_cuda",
                "invocations": 8820,
                "fallback_invocations": 0,
                "kernel_launches": 8820,
                "kernel_time_ms": 12.5,
                "host_to_device_bytes": 8192,
                "device_to_host_bytes": 4096
            }
        ],
        "cuda_execution_residency": {
            "schema_version": "1.0.0",
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "host_device_transfer_accounting": {
                "status": "qk256_measured",
                "host_to_device_bytes": 8192,
                "device_to_host_bytes": 4096,
                "kernel_time_ms": 12.5
            }
        },
        "claim_boundaries": [
            "speedup_claim=false; repeated strict ask timing remains baseline evidence only.",
            "This receipt does not claim broad chat quality, production server readiness, or full CUDA residency."
        ],
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json"
    })
}

fn repeated_backend_summary(backend: &str, runtime_api: &str, cuda: bool) -> serde_json::Value {
    let mut summary = json!({
        "backend": backend,
        "runtime_api": runtime_api,
        "runs": 2,
        "quality_passed": true,
        "fallback_used": false,
        "total_ms": repeated_metric_summary(),
        "first_token_ms": repeated_metric_summary(),
        "decode_total_ms": repeated_metric_summary(),
        "tokens_per_second": repeated_metric_summary()
    });
    if cuda {
        let object = summary.as_object_mut().expect("summary object");
        object.insert("kernel_time_ms".to_string(), repeated_metric_summary());
        object.insert("host_to_device_bytes".to_string(), repeated_u64_summary());
        object.insert("device_to_host_bytes".to_string(), repeated_u64_summary());
    }
    summary
}

fn repeated_metric_summary() -> serde_json::Value {
    json!({
        "samples": 2,
        "min": 1.0,
        "max": 2.0,
        "mean": 1.5,
        "median": 1.5
    })
}

fn repeated_u64_summary() -> serde_json::Value {
    json!({
        "samples": 2,
        "min": 4096,
        "max": 8192,
        "mean": 6144.0,
        "median": 6144.0
    })
}

fn repeated_run(
    index: u64,
    backend: &str,
    runtime_api: &str,
    kernel_id: &str,
) -> serde_json::Value {
    let mut run = json!({
        "profile": "strict_ask_math_8",
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured",
        "repeat_index": index,
        "source_receipt_path": format!("target/bitnet/receipts/cuda-bitnet-perf-002/{runtime_api}-{index}.json"),
        "selected_backend": if runtime_api == "cuda" { "nvidia-rtx-5070-ti-cuda" } else { "cpu-rust" },
        "kernel_id": kernel_id,
        "total_ms": 1.0 + index as f64,
        "first_token_ms": 1.0 + index as f64,
        "decode_total_ms": 1.0,
        "tokens_per_second": 1.0,
        "prompt_tokens": 19,
        "generated_tokens": 3,
        "answer_trimmed": "4",
        "generated_token_ids": [220, 19, 128009],
        "quality_passed": true,
        "fallback_used": false
    });
    if runtime_api == "cuda" {
        let object = run.as_object_mut().expect("run object");
        object.insert("kernel_invocations".to_string(), json!(4410));
        object.insert("kernel_time_ms".to_string(), json!(6.25));
        object.insert("host_to_device_bytes".to_string(), json!(4096));
        object.insert("device_to_host_bytes".to_string(), json!(2048));
    }
    run
}

fn repeated_pair_contract(index: u64) -> serde_json::Value {
    json!({
        "repeat_index": index,
        "same_model": true,
        "same_tokenizer": true,
        "same_prompt_template": true,
        "same_question": true,
        "same_sampling_policy": true,
        "same_generated_token_ids": true,
        "same_answer": true,
        "fallback_free": true
    })
}

fn measured_cpu_profile(profile: &str) -> serde_json::Value {
    json!({
        "profile": profile,
        "execution_phase": expected_cpu_profile_phase(profile),
        "status": "measured",
        "requested_kernel": "qk256-avx2-gemv",
        "selected_kernel": "qk256-avx2-gemv",
        "fallback_used": false,
        "fallback_reason": null,
        "shape": {
            "rows": 512,
            "cols": 1024,
            "iterations": 8
        },
        "wall_time_ms": 1.0,
        "median_ms": 1.0,
        "p95_ms": 1.0,
        "bandwidth_gbps": 0.0,
        "tokens_per_second": 0.0
    })
}

fn measured_i2s_microbench_profile(operation: &str) -> serde_json::Value {
    let selected_kernel = match operation {
        "gemm" => "qk256-scalar-gemm",
        _ => "qk256-avx2-gemv",
    };
    json!({
        "profile": format!("i2s_qk256_{operation}_microbench"),
        "operation": operation,
        "execution_phase": format!("{operation}_micro_kernel"),
        "status": "measured",
        "requested_kernel": selected_kernel,
        "selected_kernel": selected_kernel,
        "fallback_used": false,
        "fallback_reason": null,
        "shape": {
            "rows": 64,
            "cols": 1024,
            "tokens": if operation == "gemm" { 16 } else { 1 },
            "iterations": 16
        },
        "wall_time_ms": 1.0,
        "median_ms": 1.0,
        "p95_ms": 1.0,
        "bandwidth_gbps": 0.0,
        "tokens_per_second": 0.0
    })
}

fn measured_i2s_tiling_matrix_run(operation: &str) -> serde_json::Value {
    let selected_kernel = match operation {
        "gemm" => "qk256-scalar-gemm",
        _ => "qk256-avx2-gemv",
    };
    json!({
        "profile": format!("i2s_qk256_tiling_matrix_{operation}"),
        "operation": operation,
        "execution_phase": format!("{operation}_tiling_sample"),
        "status": "measured",
        "candidate": {
            "parallelism_degree": 4,
            "row_block": 4,
            "col_block": 128,
            "thread_count": 2,
            "thread_count_applied": false,
            "thread_count_policy": "recorded_not_applied",
            "thread_count_note": "Thread count is recorded as a search candidate in this fixture."
        },
        "requested_kernel": selected_kernel,
        "selected_kernel": selected_kernel,
        "fallback_used": false,
        "fallback_reason": null,
        "shape": {
            "rows": 16,
            "cols": 256,
            "tokens": if operation == "gemm" { 4 } else { 1 },
            "iterations": 4,
            "cols_rounded_to_qk256_block": true
        },
        "wall_time_ms": 1.0,
        "median_ms": 1.0,
        "p95_ms": 1.0,
        "bandwidth_gbps": 0.0,
        "tokens_per_second": 0.0,
        "speedup_claim": false
    })
}

fn measured_i2s_applied_thread_matrix_run(operation: &str) -> serde_json::Value {
    let selected_kernel = match operation {
        "gemm" => "qk256-scalar-gemm",
        _ => "qk256-avx2-gemv",
    };
    json!({
        "profile": format!("i2s_qk256_applied_thread_matrix_{operation}"),
        "operation": operation,
        "execution_phase": if operation == "gemm" {
            "prefill_gemm_applied_thread_sample"
        } else {
            "decode_gemv_applied_thread_sample"
        },
        "status": "measured",
        "candidate": {
            "parallelism_degree": 4,
            "row_block": 4,
            "col_block": 128,
            "thread_count": 2,
            "thread_count_applied": true,
            "thread_count_policy": "applied_scoped_threads",
            "applied_thread_count": 2,
            "thread_partition": if operation == "gemm" { "tokens" } else { "rows" },
            "thread_count_note": "Thread count is applied to scoped workers in this fixture."
        },
        "requested_kernel": selected_kernel,
        "selected_kernel": selected_kernel,
        "fallback_used": false,
        "fallback_reason": null,
        "shape": {
            "rows": 16,
            "cols": 256,
            "tokens": if operation == "gemm" { 4 } else { 1 },
            "iterations": 4,
            "cols_rounded_to_qk256_block": true
        },
        "wall_time_ms": 1.0,
        "median_ms": 1.0,
        "p95_ms": 1.0,
        "bandwidth_gbps": 0.0,
        "tokens_per_second": 0.0,
        "speedup_claim": false
    })
}

fn sample_cuda_benchmark_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "cuda_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-06T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "cuda_benchmark_baseline",
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "cuda": {
            "available": true,
            "device_count": 1,
            "selected_device_index": 0,
            "selected_device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "570.00",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "nvml_available": true,
            "vram_bytes": 17179869184u64,
            "power_limit_watts": 300.0,
            "power_draw_watts": 50.0,
            "temperature_c": 45.0
        },
        "machine": {
            "cpu": "AMD Ryzen 9 9950X3D",
            "gpu": "NVIDIA GeForce RTX 5070 Ti"
        },
        "benchmark": {
            "profile": "cuda_tiny_smoke",
            "kernel_id": "cuda_tiny_vector_add",
            "fixture_id": "cuda_tiny_vector_add_1024",
            "input_len": 1024,
            "iterations": 10,
            "cold_warm": {
                "compile_ms": 1.0,
                "first_iteration_total_ms": 1.0,
                "warm_iterations": 10
            },
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "cpu_reference_ms": 0.1,
            "cuda_total_ms": 0.2,
            "cuda_kernel_ms": 0.1,
            "host_to_device_ms": 0.01,
            "device_to_host_ms": 0.01,
            "allocation_ms": 0.01,
            "speedup_vs_cpu": 0.5,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "passed": true
        },
        "profiles": [
            { "profile": "cuda_tiny_smoke", "status": "measured" },
            { "profile": "cuda_transfer_h2d_d2h", "status": "measured" },
            { "profile": "cuda_fp32_matmul_small", "status": "not_run" },
            { "profile": "cuda_i2s_matmul_small", "status": "not_run" },
            { "profile": "cuda_i2s_matmul_medium", "status": "not_run" }
        ],
        "kernel_stats": [
            {
                "kernel_id": "cuda_tiny_vector_add",
                "invocations": 10,
                "fallback_invocations": 0,
                "host_to_device_bytes": 81920,
                "device_to_host_bytes": 40960,
                "kernel_launches": 10,
                "kernel_time_ms": 0.1,
                "selected_device_index": 0,
                "selected_device_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0"
            }
        ],
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json"
    })
}

fn sample_strict_bitnet_cuda_benchmark_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "strict_bitnet_cuda_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-07T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "strict_bitnet_cuda_benchmark_baseline",
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "loader_mode": "strict",
            "fallback_loader_used": false
        },
        "tokenizer": {
            "source": "explicit",
            "strict": true
        },
        "bitnet": {
            "quantization": "W1.58A8",
            "kernel_family": "qk256",
            "layout": "gguf_packed_i2_s",
            "weights_uploaded_once": false,
            "per_token_weight_upload": true
        },
        "workload": {
            "profile": "short_decode_8",
            "prompt": "fixture prompt",
            "prompt_tokens": 37,
            "generated_tokens": 8,
            "generated_text": "'E'E'E'E'E'E'E'E",
            "cpu_cuda_output_match": true
        },
        "comparison_contract": {
            "same_model": true,
            "same_tokenizer": true,
            "same_prompt": true,
            "same_generated_token_count": true,
            "same_strict_loader_mode": true,
            "same_sampling_policy": true,
            "fallback_free": true
        },
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64,
            "memory_hwm_bytes": 5949620224u64,
            "cuda_kernel_invocations": 1680
        },
        "benchmark": {
            "profile": "short_decode_8",
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "cpu_avx512_total_ms": 141559.0,
            "cuda_total_ms": 190129.0,
            "cpu_avx512_tokens_per_second": 0.0565,
            "cuda_tokens_per_second": 0.0421,
            "cpu_avx512_total_ms_div_cuda_total_ms": 0.7445,
            "cuda_kernel_invocations": 1680,
            "cpu_cuda_output_match": true,
            "speedup_claim": false
        },
        "profiles": [
            not_run_bitnet_profile("amd-9950x3d-cpu-scalar"),
            not_run_bitnet_profile("amd-9950x3d-cpu-avx2"),
            measured_bitnet_profile("amd-9950x3d-cpu-avx512", "cpu"),
            measured_bitnet_profile("nvidia-rtx-5070-ti-cuda", "cuda")
        ],
        "kernel_stats": [
            {
                "kernel_id": "qk256_gemv_cuda",
                "invocations": 1680,
                "fallback_invocations": 0,
                "kernel_launches": 1680,
                "kernel_time_ms": null
            }
        ],
        "claim_boundaries": [
            "speedup_claim=false; this receipt records a baseline only.",
            "CPU scalar and AVX2 strict end-to-end profiles are explicitly present but not_run because this CLI path does not expose selectors for those modes."
        ],
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json"
    })
}

fn sample_strict_cuda_answer_path_benchmark_receipt() -> serde_json::Value {
    json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_answer_path_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": "2026-05-08T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "strict_cuda_answer_path_benchmark_baseline",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "execution_plan": sample_bitnet_qk256_execution_plan(),
        "proof_inputs": {
            "cpu_avx512_ask_receipt": "target/bitnet/receipts/cpu.json",
            "cuda_ask_receipt": "target/bitnet/receipts/cuda.json",
            "cpu_avx512_answer_corpus_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json",
            "cuda_answer_corpus_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json",
            "cpu_cuda_answer_parity_receipt": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json"
        },
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "loader_mode": "strict_real_gguf",
            "fallback_loader_used": false
        },
        "tokenizer": {
            "source": "explicit",
            "strict": true,
            "type": "llama3",
            "pretokenizer_authority": "llama-bpe"
        },
        "prompt_template": {
            "family": "bitnetcpp-answer",
            "rendered_sha256": "dee5b2fff5b96df948252b7a589ab7ea1a6b6a10ed1b2d9ed70a63ebbde554f3"
        },
        "workload": {
            "profile": "strict_ask_math_8",
            "question": "What is 2+2? Answer with only the number.",
            "answer": " 4",
            "prompt_tokens": 19,
            "generated_tokens": 3,
            "quality_passed": true,
            "cpu_cuda_answer_match": true,
            "cpu_cuda_generated_ids_match": true
        },
        "comparison_contract": {
            "same_model": true,
            "same_tokenizer": true,
            "same_prompt_template": true,
            "same_question": true,
            "same_sampling_policy": true,
            "same_generated_token_ids": true,
            "same_answer": true,
            "fallback_free": true
        },
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "vram_bytes": 17094475776u64,
            "memory_hwm_bytes": 9201254400u64,
            "cuda_kernel_invocations": 4410,
            "power_draw_watts": null,
            "temperature_c": null
        },
        "benchmark": {
            "profile": "strict_ask_math_8",
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "cpu_avx512_total_ms": 19410.0,
            "cuda_total_ms": 1833.0,
            "cpu_avx512_tokens_per_second": 0.1545,
            "cuda_tokens_per_second": 1.6366,
            "observed_cpu_total_ms_div_cuda_total_ms": 10.59,
            "cuda_kernel_invocations": 4410,
            "cpu_cuda_answer_match": true,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false
        },
        "timing_split": {
            "cpu_avx512": {
                "model_load_ms": 2781.206,
                "tokenizer_load_ms": 340.326,
                "prompt_render_tokenize_ms": 0.258,
                "prefill_ms": 16497.948,
                "first_token_ms": 17460.0,
                "decode_total_ms": 2912.929,
                "steady_decode_tokens_per_second": 1.026
            },
            "cuda": {
                "model_load_ms": 2845.54,
                "tokenizer_load_ms": 307.915,
                "prompt_render_tokenize_ms": 0.257,
                "prefill_ms": 1547.927,
                "first_token_ms": 1645.0,
                "decode_total_ms": 285.554,
                "steady_decode_tokens_per_second": 10.668,
                "cuda_context_init_ms": null,
                "cuda_context_init_ms_source": "not_separately_measured",
                "weight_upload_ms": null,
                "weight_upload_ms_source": "not_separately_measured",
                "kernel_time_ms": null,
                "kernel_time_ms_source": "not_measured_by_current_receipt",
                "host_to_device_bytes": null,
                "host_to_device_bytes_source": "not_measured_by_current_receipt",
                "device_to_host_bytes": null,
                "device_to_host_bytes_source": "not_measured_by_current_receipt"
            }
        },
        "profiles": [
            measured_answer_path_profile("strict_ask_math_8", "amd-9950x3d-cpu-avx512", "cpu"),
            measured_answer_path_profile("strict_ask_math_8", "nvidia-rtx-5070-ti-cuda", "cuda"),
            existing_answer_path_profile("answer_corpus_5", "amd-9950x3d-cpu-avx512"),
            existing_answer_path_profile("answer_corpus_5", "nvidia-rtx-5070-ti-cuda"),
            {
                "profile": "prefill_512_decode_128",
                "backend": "amd-9950x3d-cpu-avx512",
                "runtime_api": "cpu",
                "status": "blocked_timeout",
                "timeout_seconds": 1800,
                "reason": "30-minute CPU AVX-512 phase benchmark timed out before producing profile receipts"
            }
        ],
        "kernel_stats": [
            {
                "kernel_id": "qk256_gemv_cuda",
                "invocations": 4410,
                "fallback_invocations": 0,
                "kernel_launches": 4410,
                "kernel_time_ms": null
            }
        ],
        "cuda_execution_residency": {
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundaries": [
            "speedup_claim=false",
            "strict ask timing is measured; long prefill/decode remains blocked",
            "kernel time and transfer byte timing are not separately measured"
        ],
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-prod-004-answer-path-benchmark.json"
    })
}

fn measured_answer_path_profile(
    profile: &str,
    backend: &str,
    runtime_api: &str,
) -> serde_json::Value {
    json!({
        "profile": profile,
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured",
        "total_ms": 1.0,
        "first_token_ms": 1.0,
        "tokens_per_second": 1.0,
        "prompt_tokens": 19,
        "generated_tokens": 3,
        "quality_passed": true,
        "fallback_used": false
    })
}

fn existing_answer_path_profile(profile: &str, backend: &str) -> serde_json::Value {
    json!({
        "profile": profile,
        "backend": backend,
        "runtime_api": if backend.contains("cuda") { "cuda" } else { "cpu" },
        "status": "measured_existing_receipt",
        "receipt_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json",
        "quality_passed": true,
        "fallback_used": false
    })
}

fn measured_bitnet_profile(backend: &str, runtime_api: &str) -> serde_json::Value {
    json!({
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured",
        "total_ms": 1.0,
        "first_token_ms": 1.0,
        "tokens_per_second": 1.0,
        "prompt_tokens": 37,
        "generated_tokens": 8,
        "fallback_used": false
    })
}

fn not_run_bitnet_profile(backend: &str) -> serde_json::Value {
    json!({
        "backend": backend,
        "runtime_api": "cpu",
        "status": "not_run",
        "reason": "current CLI does not expose a strict end-to-end selector for this CPU SIMD mode"
    })
}
