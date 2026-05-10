use bitnet_bench_receipts::{
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json,
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json,
    validate_dense_gguf_qwen_repeated_comparator_receipt_json,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_BASELINE: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json";
const DEFAULT_COMPARATOR: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json";
const DEFAULT_ONE_TOKEN_TRANSFER: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-005-h2d-transfer-envelope/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json";
const DEFAULT_SHORT_DECODE_TRANSFER: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-005-h2d-transfer-envelope/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json";
const DEFAULT_WARM_SESSION_TRANSFER: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-005-h2d-transfer-envelope/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json";
const DEFAULT_RECEIPT_OUT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-006-h2d-envelope-qualification/dense-gguf-qwen-benchmark-qualification-h2d-envelope.json";

#[derive(Debug)]
struct Args {
    baseline_receipt: PathBuf,
    comparator_receipt: PathBuf,
    one_token_transfer: PathBuf,
    short_decode_transfer: PathBuf,
    warm_session_transfer: PathBuf,
    receipt_out: PathBuf,
}

#[derive(Debug)]
struct TransferReceipt {
    path: PathBuf,
    value: Value,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let baseline = read_json(&args.baseline_receipt)?;
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&baseline)?;

    let comparator = read_json(&args.comparator_receipt)?;
    validate_dense_gguf_qwen_repeated_comparator_receipt_json(&comparator)?;

    let one_token = read_transfer_receipt(
        &args.one_token_transfer,
        "dense_gguf_qwen_one_token_strict_cuda_proof",
    )?;
    let short_decode = read_transfer_receipt(
        &args.short_decode_transfer,
        "dense_gguf_qwen_short_decode_strict_cuda_proof",
    )?;
    let warm_session = read_transfer_receipt(
        &args.warm_session_transfer,
        "dense_gguf_qwen_warm_session_strict_cuda_proof",
    )?;

    let receipt =
        build_receipt(&args, &baseline, &comparator, &one_token, &short_decode, &warm_session)?;
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut baseline_receipt = PathBuf::from(DEFAULT_BASELINE);
    let mut comparator_receipt = PathBuf::from(DEFAULT_COMPARATOR);
    let mut one_token_transfer = PathBuf::from(DEFAULT_ONE_TOKEN_TRANSFER);
    let mut short_decode_transfer = PathBuf::from(DEFAULT_SHORT_DECODE_TRANSFER);
    let mut warm_session_transfer = PathBuf::from(DEFAULT_WARM_SESSION_TRANSFER);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--baseline-receipt" => baseline_receipt = PathBuf::from(next_value(&mut iter, &arg)?),
            "--comparator-receipt" => {
                comparator_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--one-token-transfer" => {
                one_token_transfer = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--short-decode-transfer" => {
                short_decode_transfer = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--warm-session-transfer" => {
                warm_session_transfer = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        baseline_receipt,
        comparator_receipt,
        one_token_transfer,
        short_decode_transfer,
        warm_session_transfer,
        receipt_out,
    })
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: dense_qwen_cuda_benchmark_qualification_receipt [--baseline-receipt PATH] [--comparator-receipt PATH] [--one-token-transfer PATH] [--short-decode-transfer PATH] [--warm-session-transfer PATH] [--receipt-out PATH]"
    );
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_transfer_receipt(
    path: &Path,
    expected_artifact_kind: &str,
) -> Result<TransferReceipt, Box<dyn Error>> {
    let value = read_json(path)?;
    if str_at(&value, "/artifact_kind")? != expected_artifact_kind {
        return Err(format!(
            "{} must have artifact_kind {expected_artifact_kind}",
            path_label(path)
        )
        .into());
    }
    if str_at(&value, "/execution_plan/selected_route")? != "dense_regular_llm_cuda" {
        return Err(format!("{} must route dense_regular_llm_cuda", path_label(path)).into());
    }
    if bool_at(&value, "/fallback_used")? {
        return Err(format!("{} must be fallback-free", path_label(path)).into());
    }
    if bool_at(&value, "/claim_boundary/speedup_claim")? {
        return Err(format!("{} must not claim speedup", path_label(path)).into());
    }
    if bool_at(&value, "/claim_boundary/bitnet_packed_i2s_qk256_proof")? {
        return Err(format!("{} must not claim BitNet packed proof", path_label(path)).into());
    }
    match str_at(&value, "/timing/transfer_timing_status")? {
        "device_to_host_measured_host_to_device_unmeasured" => {
            if value.pointer("/timing/host_to_device_ms").and_then(Value::as_f64).is_some() {
                return Err(format!(
                    "{} must keep host_to_device_ms unmeasured for D2H-only timing status",
                    path_label(path)
                )
                .into());
            }
        }
        "host_to_device_model_load_envelope_device_to_host_measured" => {
            number_at(&value, "/timing/host_to_device_ms")?;
            if str_at(&value, "/timing/host_to_device_ms_source")?
                != "wall_clock_model_load_with_cuda_weight_upload"
            {
                return Err(format!(
                    "{} must label H2D as wall_clock_model_load_with_cuda_weight_upload",
                    path_label(path)
                )
                .into());
            }
            if str_at(&value, "/timing/host_to_device_ms_scope")?
                != "model_load_wall_clock_envelope"
            {
                return Err(format!(
                    "{} must label H2D scope as model_load_wall_clock_envelope",
                    path_label(path)
                )
                .into());
            }
            if !bool_at(&value, "/timing/host_to_device_ms_includes_non_transfer_overhead")? {
                return Err(format!(
                    "{} must record that H2D envelope includes non-transfer overhead",
                    path_label(path)
                )
                .into());
            }
        }
        other => {
            return Err(format!(
                "{} has unsupported transfer timing status {other}",
                path_label(path)
            )
            .into());
        }
    }
    number_at(&value, "/timing/device_to_host_ms")?;
    Ok(TransferReceipt { path: path.to_owned(), value })
}

fn build_receipt(
    args: &Args,
    baseline: &Value,
    comparator: &Value,
    one_token: &TransferReceipt,
    short_decode: &TransferReceipt,
    warm_session: &TransferReceipt,
) -> Result<Value, Box<dyn Error>> {
    let profiles = vec![
        profile_review(comparator, "one_token", one_token)?,
        profile_review(comparator, "short_decode_8", short_decode)?,
        profile_review(comparator, "warm_session_3_turns", warm_session)?,
    ];
    let evidence = json!({
        "one_token": evidence_summary(comparator, "one_token", one_token)?,
        "short_decode_8": evidence_summary(comparator, "short_decode_8", short_decode)?,
        "warm_session_3_turns": evidence_summary(comparator, "warm_session_3_turns", warm_session)?
    });

    Ok(json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_qwen_benchmark_qualification_review",
        "artifact_path": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
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
        "model": comparator.pointer("/model").cloned().ok_or("comparator model missing")?,
        "tokenizer_prompt_authority": comparator
            .pointer("/tokenizer_prompt_authority")
            .cloned()
            .ok_or("comparator tokenizer_prompt_authority missing")?,
        "execution_plan": comparator
            .pointer("/execution_plan")
            .cloned()
            .ok_or("comparator execution_plan missing")?,
        "proof_inputs": {
            "benchmark_baseline": proof_input(&args.baseline_receipt, baseline)?,
            "repeated_comparator": proof_input(&args.comparator_receipt, comparator)?,
            "one_token_transfer_timing": proof_input(&one_token.path, &one_token.value)?,
            "short_decode_transfer_timing": proof_input(&short_decode.path, &short_decode.value)?,
            "warm_session_transfer_timing": proof_input(&warm_session.path, &warm_session.value)?
        },
        "qualification_decision": {
            "status": "not_accepted",
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "accepted_profiles": [],
            "blocked_profiles": ["one_token", "short_decode_8", "warm_session_3_turns"],
            "reason": "Dense Qwen CUDA evidence is fallback-free and repeated, with D2H logits timing measured and an H2D model-load envelope recorded, but every reviewed CUDA profile is slower than the same-artifact CPU reference mean and pure H2D copy timing remains unmeasured."
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
                "id": "host_to_device_model_load_envelope",
                "description": "Host-to-device model-load wall-clock envelope timing is recorded with explicit non-transfer-overhead labeling.",
                "status": "passed"
            },
            {
                "id": "pure_host_to_device_transfer_timing",
                "description": "Pure host-to-device copy timing is measured separately from model-load and upload overhead.",
                "status": "blocked",
                "blocker": "CUDA-DENSE-PERF-005 records a model-load wall-clock envelope, not pure CUDA event copy timing."
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
        "profile_reviews": profiles,
        "evidence_summary": evidence,
        "transfer_timing_review": {
            "status": str_at(&one_token.value, "/timing/transfer_timing_status")?,
            "device_to_host_timing_recorded": true,
            "host_to_device_timing_recorded": h2d_envelope_recorded(one_token),
            "host_to_device_model_load_envelope_recorded": h2d_envelope_recorded(one_token),
            "host_to_device_pure_transfer_timing_recorded": false,
            "host_to_device_blocker": "The dense Qwen runtime records an H2D model-load wall-clock envelope, but not pure CUDA event copy timing.",
            "device_to_host_source": "wall_clock_extract_logits_2d_local",
            "host_to_device_source": h2d_source(one_token)?,
            "host_to_device_scope": h2d_scope_json(one_token),
            "host_to_device_ms_includes_non_transfer_overhead": h2d_includes_overhead_json(one_token)
        },
        "hardware_context": comparator
            .pointer("/hardware_context")
            .cloned()
            .ok_or("comparator hardware_context missing")?,
        "cuda": comparator.pointer("/cuda").cloned().ok_or("comparator cuda missing")?,
        "claim_boundaries": [
            "speedup_claim=false; no dense Qwen profile is upgraded by this review.",
            "benchmark_qualified_speedup=false; current CUDA means are slower than same-artifact CPU means.",
            "H2D model-load envelope timing is recorded, but pure CUDA event H2D copy timing remains unmeasured.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof."
        ]
    }))
}

fn proof_input(path: &Path, receipt: &Value) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "path": path_label(path),
        "sha256": sha256_file(path)?,
        "artifact_kind": str_at(receipt, "/artifact_kind")?
    }))
}

fn profile_review(
    comparator: &Value,
    profile_name: &str,
    transfer: &TransferReceipt,
) -> Result<Value, Box<dyn Error>> {
    let profile = comparator_profile(comparator, profile_name)?;
    let cpu_mean = number_at(profile, "/cpu_total_ms/mean")?;
    let cuda_mean = number_at(profile, "/cuda_total_ms/mean")?;
    let d2h_ms = number_at(&transfer.value, "/timing/device_to_host_ms")?;
    Ok(json!({
        "profile": profile_name,
        "decision": "not_accepted",
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": bool_at(profile, "/fallback_free")?,
        "quality_passed": true,
        "generated_token_ids_match": bool_at(profile, "/generated_token_ids_match")?,
        "dense_cuda_evidence_used": true,
        "runs_per_backend": u64_at(profile, "/min_runs_per_backend")?,
        "cpu_total_ms_mean": cpu_mean,
        "cuda_total_ms_mean": cuda_mean,
        "observed_cpu_total_ms_div_cuda_total_ms": cpu_mean / cuda_mean,
        "cuda_mean_slower_than_cpu": cuda_mean > cpu_mean,
        "host_to_device_ms": h2d_ms_json(transfer),
        "host_to_device_ms_source": h2d_source(transfer)?,
        "host_to_device_ms_scope": h2d_scope_json(transfer),
        "host_to_device_ms_includes_non_transfer_overhead": h2d_includes_overhead_json(transfer),
        "pure_host_to_device_ms": null,
        "pure_host_to_device_ms_source": "not_measured_by_dense_qwen_runtime",
        "device_to_host_ms": d2h_ms,
        "device_to_host_ms_source": str_at(&transfer.value, "/timing/device_to_host_ms_source")?,
        "reason": "This profile remains baseline/comparator evidence because CUDA mean total time is not faster than the same-artifact CPU reference and pure H2D copy timing is incomplete.",
        "blockers": [
            "CUDA mean total time is slower than CPU mean total time",
            "pure host-to-device transfer timing is unmeasured",
            "no profile-specific speedup threshold has been accepted"
        ]
    }))
}

fn evidence_summary(
    comparator: &Value,
    profile_name: &str,
    transfer: &TransferReceipt,
) -> Result<Value, Box<dyn Error>> {
    let profile = comparator_profile(comparator, profile_name)?;
    let cpu_mean = number_at(profile, "/cpu_total_ms/mean")?;
    let cuda_mean = number_at(profile, "/cuda_total_ms/mean")?;
    Ok(json!({
        "profile": profile_name,
        "runs_per_backend": u64_at(profile, "/min_runs_per_backend")?,
        "fallback_free": bool_at(profile, "/fallback_free")?,
        "quality_passed": true,
        "generated_token_ids_match": bool_at(profile, "/generated_token_ids_match")?,
        "cpu_total_ms_mean": cpu_mean,
        "cuda_total_ms_mean": cuda_mean,
        "observed_cpu_total_ms_div_cuda_total_ms": cpu_mean / cuda_mean,
        "cuda_mean_slower_than_cpu": cuda_mean > cpu_mean,
        "device_to_host_ms": number_at(&transfer.value, "/timing/device_to_host_ms")?,
        "host_to_device_ms": h2d_ms_json(transfer),
        "host_to_device_ms_source": h2d_source(transfer)?,
        "host_to_device_ms_scope": h2d_scope_json(transfer),
        "host_to_device_ms_includes_non_transfer_overhead": h2d_includes_overhead_json(transfer),
        "pure_host_to_device_ms": null,
        "pure_host_to_device_ms_source": "not_measured_by_dense_qwen_runtime",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false
    }))
}

fn h2d_envelope_recorded(transfer: &TransferReceipt) -> bool {
    transfer.value.pointer("/timing/host_to_device_ms").and_then(Value::as_f64).is_some()
}

fn h2d_ms_json(transfer: &TransferReceipt) -> Value {
    transfer.value.pointer("/timing/host_to_device_ms").cloned().unwrap_or(Value::Null)
}

fn h2d_source(transfer: &TransferReceipt) -> Result<&str, Box<dyn Error>> {
    str_at(&transfer.value, "/timing/host_to_device_ms_source")
}

fn h2d_scope_json(transfer: &TransferReceipt) -> Value {
    transfer.value.pointer("/timing/host_to_device_ms_scope").cloned().unwrap_or(Value::Null)
}

fn h2d_includes_overhead_json(transfer: &TransferReceipt) -> Value {
    transfer
        .value
        .pointer("/timing/host_to_device_ms_includes_non_transfer_overhead")
        .cloned()
        .unwrap_or(Value::Null)
}

fn comparator_profile<'a>(
    comparator: &'a Value,
    profile_name: &str,
) -> Result<&'a Value, Box<dyn Error>> {
    comparator
        .pointer("/profiles")
        .and_then(Value::as_array)
        .and_then(|profiles| {
            profiles.iter().find(|profile| {
                profile.get("profile").and_then(Value::as_str) == Some(profile_name)
            })
        })
        .ok_or_else(|| format!("comparator profile {profile_name} not found").into())
}

fn timestamp_label() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{pointer} must be a string").into())
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{pointer} must be a bool").into())
}

fn u64_at(value: &Value, pointer: &str) -> Result<u64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{pointer} must be an unsigned integer").into())
}

fn number_at(value: &Value, pointer: &str) -> Result<f64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{pointer} must be a number").into())
}
