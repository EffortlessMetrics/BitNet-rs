use bitnet_bench_receipts::validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_ONE_TOKEN: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json";
const DEFAULT_SHORT_DECODE: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json";
const DEFAULT_WARM_SESSION: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json";
const DEFAULT_RECEIPT_OUT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json";

#[derive(Debug)]
struct Args {
    one_token_receipt: PathBuf,
    short_decode_receipt: PathBuf,
    warm_session_receipt: PathBuf,
    receipt_out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let one_token = read_json(&args.one_token_receipt)?;
    let short_decode = read_json(&args.short_decode_receipt)?;
    let warm_session = read_json(&args.warm_session_receipt)?;

    assert_source_receipt(
        &one_token,
        "dense_gguf_qwen_one_token_strict_cuda_proof",
        "dense_gguf_qwen_one_token_strict_cuda_proof_recorded",
    )?;
    assert_source_receipt(
        &short_decode,
        "dense_gguf_qwen_short_decode_strict_cuda_proof",
        "dense_gguf_qwen_short_decode_strict_cuda_proof_recorded",
    )?;
    assert_source_receipt(
        &warm_session,
        "dense_gguf_qwen_warm_session_strict_cuda_proof",
        "dense_gguf_qwen_warm_session_strict_cuda_proof_recorded",
    )?;
    assert_same_model_and_authority(&one_token, &short_decode, &warm_session)?;

    let receipt = build_receipt(&args, &one_token, &short_decode, &warm_session)?;
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut one_token_receipt = PathBuf::from(DEFAULT_ONE_TOKEN);
    let mut short_decode_receipt = PathBuf::from(DEFAULT_SHORT_DECODE);
    let mut warm_session_receipt = PathBuf::from(DEFAULT_WARM_SESSION);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--one-token-receipt" => {
                one_token_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--short-decode-receipt" => {
                short_decode_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--warm-session-receipt" => {
                warm_session_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args { one_token_receipt, short_decode_receipt, warm_session_receipt, receipt_out })
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: dense_qwen_cuda_benchmark_baseline_receipt [--one-token-receipt PATH] [--short-decode-receipt PATH] [--warm-session-receipt PATH] [--receipt-out PATH]"
    );
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn build_receipt(
    args: &Args,
    one_token: &Value,
    short_decode: &Value,
    warm_session: &Value,
) -> Result<Value, Box<dyn Error>> {
    let profiles = vec![
        profile_from_receipt("one_token", &args.one_token_receipt, one_token)?,
        profile_from_receipt("short_decode_8", &args.short_decode_receipt, short_decode)?,
        profile_from_receipt("warm_session_3_turns", &args.warm_session_receipt, warm_session)?,
    ];
    let kernel_summary = aggregate_kernel_summary(&profiles);

    Ok(json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_qwen_cuda_benchmark_baseline",
        "artifact_path": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
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
        "model": model_from_source(warm_session)?,
        "tokenizer_prompt_authority": one_token
            .pointer("/tokenizer_prompt_authority")
            .cloned()
            .ok_or("tokenizer_prompt_authority missing")?,
        "execution_plan": warm_session
            .pointer("/execution_plan")
            .cloned()
            .ok_or("execution_plan missing")?,
        "proof_inputs": {
            "one_token": proof_input(&args.one_token_receipt, one_token)?,
            "short_decode": proof_input(&args.short_decode_receipt, short_decode)?,
            "warm_session": proof_input(&args.warm_session_receipt, warm_session)?
        },
        "profiles": profiles,
        "kernel_summary": kernel_summary,
        "benchmark_summary": {
            "status": "baseline_only",
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "profiles_recorded": 3,
            "accepted_speedup_profiles": [],
            "qualification_blockers": [
                "missing repeated same-artifact CPU/CUDA comparator for dense profiles",
                "missing cold/warm split across repeated dense Qwen benchmark runs",
                "missing profile-specific speedup thresholds and review"
            ],
            "next_step": "CUDA-DENSE-PERF-002 repeated CPU/CUDA comparator"
        },
        "cuda": warm_session
            .pointer("/cuda")
            .cloned()
            .ok_or("cuda block missing")?,
        "claim_boundaries": [
            "speedup_claim=false; dense Qwen CUDA timing is baseline evidence only.",
            "benchmark_qualified_speedup=false; no dense profile is accepted by this receipt.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof.",
            "full_cuda_residency_claimed=false; warm-session residency remains scoped."
        ]
    }))
}

fn profile_from_receipt(
    profile: &str,
    path: &Path,
    receipt: &Value,
) -> Result<Value, Box<dyn Error>> {
    let prompt_tokens = match profile {
        "warm_session_3_turns" => {
            u64_at(receipt, "/tokenizer_prompt_authority/prompt_token_count_total")?
        }
        _ => u64_at(receipt, "/tokenizer_prompt_authority/prompt_token_count")?,
    };
    let generated_tokens = match profile {
        "one_token" => 1,
        "short_decode_8" => u64_at(receipt, "/short_decode_proof/generated_tokens_count")?,
        "warm_session_3_turns" => u64_at(receipt, "/warm_session_proof/generated_tokens_total")?,
        _ => return Err(format!("unknown profile {profile}").into()),
    };
    let turns_count = if profile == "warm_session_3_turns" {
        Some(u64_at(receipt, "/warm_session_proof/turns_count")?)
    } else {
        None
    };
    let timing = receipt.pointer("/timing").ok_or("timing missing")?;
    let decode_total_ms = timing
        .get("decode_total_ms")
        .or_else(|| timing.get("decode_ms"))
        .and_then(Value::as_f64)
        .ok_or("timing decode field missing")?;
    let mut value = json!({
        "profile": profile,
        "backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "status": "measured_existing_receipt",
        "source_receipt_path": path_label(path),
        "source_receipt_sha256": sha256_file(path)?,
        "source_artifact_kind": str_at(receipt, "/artifact_kind")?,
        "fallback_used": bool_at(receipt, "/fallback_used")?,
        "quality_passed": bool_at(receipt, "/quality_gate/passed")?,
        "parity_passed": bool_at(receipt, "/parity/passed")?,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_ms": number_at(receipt, "/timing/total_ms")?,
        "first_token_ms": number_at(receipt, "/timing/first_token_ms")?,
        "decode_total_ms": decode_total_ms,
        "kernel_time_ms": number_at(receipt, "/timing/kernel_time_ms")?,
        "kernel_invocations": u64_at(receipt, "/timing/kernel_invocations")?,
        "kernel_launches": u64_at(receipt, "/timing/kernel_launches")?,
        "host_to_device_bytes": u64_at(receipt, "/timing/host_to_device_bytes")?,
        "device_to_host_bytes": u64_at(receipt, "/timing/device_to_host_bytes")?,
        "cpu_reference_total_ms": number_at(receipt, "/timing/cpu_reference_total_ms")?,
        "top_k_max_abs_error": number_at(receipt, "/parity/max_abs_error")?,
        "top_k_mean_abs_error": number_at(receipt, "/parity/mean_abs_error")?
    });
    if let Some(turns_count) = turns_count {
        value["turns_count"] = json!(turns_count);
    }
    Ok(value)
}

fn aggregate_kernel_summary(profiles: &[Value]) -> Value {
    let total_kernel_invocations =
        profiles.iter().map(|p| u64_value(p, "kernel_invocations")).sum::<u64>();
    let total_kernel_launches =
        profiles.iter().map(|p| u64_value(p, "kernel_launches")).sum::<u64>();
    let total_kernel_time_ms =
        profiles.iter().map(|p| number_value(p, "kernel_time_ms")).sum::<f64>();
    let total_host_to_device_bytes =
        profiles.iter().map(|p| u64_value(p, "host_to_device_bytes")).sum::<u64>();
    let total_device_to_host_bytes =
        profiles.iter().map(|p| u64_value(p, "device_to_host_bytes")).sum::<u64>();
    json!({
        "total_kernel_invocations": total_kernel_invocations,
        "total_kernel_launches": total_kernel_launches,
        "total_kernel_time_ms": total_kernel_time_ms,
        "total_host_to_device_bytes": total_host_to_device_bytes,
        "total_device_to_host_bytes": total_device_to_host_bytes,
        "total_cpu_fallback_invocations": 0,
        "fallback_used": false
    })
}

fn proof_input(path: &Path, receipt: &Value) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "path": path_label(path),
        "sha256": sha256_file(path)?,
        "artifact_kind": str_at(receipt, "/artifact_kind")?
    }))
}

fn model_from_source(receipt: &Value) -> Result<Value, Box<dyn Error>> {
    let model = receipt.pointer("/model").ok_or("model missing")?;
    Ok(json!({
        "id": str_at(model, "/id")?,
        "model_family": str_at(model, "/model_family")?,
        "artifact_kind": str_at(model, "/artifact_kind")?,
        "file": str_at(model, "/file")?,
        "sha256": str_at(model, "/sha256")?
    }))
}

fn assert_source_receipt(
    receipt: &Value,
    expected_kind: &str,
    expected_claim: &str,
) -> Result<(), Box<dyn Error>> {
    if str_at(receipt, "/artifact_kind")? != expected_kind {
        return Err(format!("artifact_kind must be {expected_kind}").into());
    }
    if str_at(receipt, "/claim")? != expected_claim {
        return Err(format!("claim must be {expected_claim}").into());
    }
    if str_at(receipt, "/execution_plan/selected_route")? != "dense_regular_llm_cuda" {
        return Err("source receipt must route dense_regular_llm_cuda".into());
    }
    if str_at(receipt, "/selected_backend")? != "nvidia-rtx-5070-ti-cuda" {
        return Err("source receipt must select nvidia-rtx-5070-ti-cuda".into());
    }
    if bool_at(receipt, "/fallback_used")? {
        return Err("source receipt must be fallback-free".into());
    }
    if bool_at(receipt, "/claim_boundary/speedup_claim")? {
        return Err("source receipt must not claim speedup".into());
    }
    if bool_at(receipt, "/claim_boundary/bitnet_packed_i2s_qk256_proof")? {
        return Err("source receipt must not claim BitNet packed proof".into());
    }
    Ok(())
}

fn assert_same_model_and_authority(
    one_token: &Value,
    short_decode: &Value,
    warm_session: &Value,
) -> Result<(), Box<dyn Error>> {
    let model_sha = str_at(one_token, "/model/sha256")?;
    let prompt_hash = str_at(one_token, "/tokenizer_prompt_authority/rendered_prompt_sha256")?;
    let token_hash = str_at(one_token, "/tokenizer_prompt_authority/prompt_token_ids_sha256")?;
    for receipt in [short_decode, warm_session] {
        if str_at(receipt, "/model/sha256")? != model_sha {
            return Err("source receipts must use the same model sha256".into());
        }
    }
    if str_at(short_decode, "/tokenizer_prompt_authority/rendered_prompt_sha256")? != prompt_hash {
        return Err(
            "one-token and short-decode receipts must use the same rendered prompt hash".into()
        );
    }
    if str_at(short_decode, "/tokenizer_prompt_authority/prompt_token_ids_sha256")? != token_hash {
        return Err(
            "one-token and short-decode receipts must use the same prompt token ids hash".into()
        );
    }
    if str_at(warm_session, "/tokenizer_prompt_authority/turns/0/rendered_prompt_sha256")?
        != prompt_hash
    {
        return Err("warm-session first turn must use the same rendered prompt hash".into());
    }
    if str_at(warm_session, "/tokenizer_prompt_authority/turns/0/prompt_token_ids_sha256")?
        != token_hash
    {
        return Err("warm-session first turn must use the same prompt token ids hash".into());
    }
    Ok(())
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

fn u64_value(value: &Value, field: &str) -> u64 {
    value.get(field).and_then(Value::as_u64).unwrap_or(0)
}

fn number_value(value: &Value, field: &str) -> f64 {
    value.get(field).and_then(Value::as_f64).unwrap_or(0.0)
}
