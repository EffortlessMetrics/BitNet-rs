use bitnet_bench_receipts::validate_dense_gguf_qwen_benchmark_qualification_receipt_json;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_ONE_TOKEN_PROOF: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-one-token-cuda.json";
const DEFAULT_SHORT_DECODE_PROOF: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-short-decode-cuda.json";
const DEFAULT_WARM_SESSION_PROOF: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-warm-session-cuda.json";
const DEFAULT_RECEIPT_OUT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-benchmark-qualification.json";

#[derive(Debug)]
struct Args {
    one_token_proof: PathBuf,
    short_decode_proof: PathBuf,
    warm_session_proof: PathBuf,
    receipt_out: PathBuf,
}

#[derive(Debug)]
struct ProofReceipt {
    path: PathBuf,
    value: Value,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let one_token =
        read_proof_receipt(&args.one_token_proof, "dense_gguf_qwen_one_token_strict_cuda_proof")?;
    let short_decode = read_proof_receipt(
        &args.short_decode_proof,
        "dense_gguf_qwen_short_decode_strict_cuda_proof",
    )?;
    let warm_session = read_proof_receipt(
        &args.warm_session_proof,
        "dense_gguf_qwen_warm_session_strict_cuda_proof",
    )?;

    ensure_same_model(&one_token.value, &short_decode.value)?;
    ensure_same_model(&one_token.value, &warm_session.value)?;

    let receipt = build_receipt(&args, &one_token, &short_decode, &warm_session)?;
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut one_token_proof = PathBuf::from(DEFAULT_ONE_TOKEN_PROOF);
    let mut short_decode_proof = PathBuf::from(DEFAULT_SHORT_DECODE_PROOF);
    let mut warm_session_proof = PathBuf::from(DEFAULT_WARM_SESSION_PROOF);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--one-token-proof" => one_token_proof = PathBuf::from(next_value(&mut iter, &arg)?),
            "--short-decode-proof" => {
                short_decode_proof = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--warm-session-proof" => {
                warm_session_proof = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args { one_token_proof, short_decode_proof, warm_session_proof, receipt_out })
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: qwen3_cuda_benchmark_qualification_receipt [--one-token-proof PATH] [--short-decode-proof PATH] [--warm-session-proof PATH] [--receipt-out PATH]"
    );
}

fn read_proof_receipt(
    path: &Path,
    expected_artifact_kind: &str,
) -> Result<ProofReceipt, Box<dyn Error>> {
    let value = read_json(path)?;
    if str_at(&value, "/artifact_kind")? != expected_artifact_kind {
        return Err(format!(
            "{} must have artifact_kind {expected_artifact_kind}",
            path_label(path)
        )
        .into());
    }
    if str_at(&value, "/model/id")? != "qwen3-0.6b-instruct-q8_0" {
        return Err(format!("{} must be the Qwen3 0.6B artifact", path_label(path)).into());
    }
    if str_at(&value, "/model/sha256")?
        != "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
    {
        return Err(format!("{} has unexpected Qwen3 SHA", path_label(path)).into());
    }
    if str_at(&value, "/execution_plan/selected_route")? != "dense_regular_llm_cuda" {
        return Err(format!("{} must route dense_regular_llm_cuda", path_label(path)).into());
    }
    if str_at(&value, "/selected_backend")? != "nvidia-rtx-5070-ti-cuda" {
        return Err(format!("{} must select RTX 5070 Ti CUDA", path_label(path)).into());
    }
    if bool_at(&value, "/fallback_used")? {
        return Err(format!("{} must be fallback-free", path_label(path)).into());
    }
    if bool_at(&value, "/speedup_claim")? {
        return Err(format!("{} must not claim speedup", path_label(path)).into());
    }
    if bool_at(&value, "/claim_boundary/bitnet_packed_i2s_qk256_proof")? {
        return Err(format!("{} must not claim BitNet QK256 proof", path_label(path)).into());
    }
    Ok(ProofReceipt { path: path.to_owned(), value })
}

fn build_receipt(
    args: &Args,
    one_token: &ProofReceipt,
    short_decode: &ProofReceipt,
    warm_session: &ProofReceipt,
) -> Result<Value, Box<dyn Error>> {
    let profiles = vec![
        profile_review("one_token", one_token)?,
        profile_review("short_decode_8", short_decode)?,
        profile_review("warm_session_3_turns", warm_session)?,
    ];
    let evidence = json!({
        "one_token": evidence_summary("one_token", one_token)?,
        "short_decode_8": evidence_summary("short_decode_8", short_decode)?,
        "warm_session_3_turns": evidence_summary("warm_session_3_turns", warm_session)?
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
        "model": model_for_review(warm_session)?,
        "tokenizer_prompt_authority": tokenizer_authority_for_review(one_token)?,
        "execution_plan": warm_session
            .value
            .pointer("/execution_plan")
            .cloned()
            .ok_or("execution_plan missing")?,
        "proof_inputs": {
            "benchmark_baseline": {
                "status": "missing",
                "reason": "CUDA-MODEL-007 reviews current Qwen3 proof receipts before repeated benchmark baseline evidence exists."
            },
            "repeated_comparator": {
                "status": "missing",
                "reason": "Qwen3 repeated same-artifact CPU/CUDA comparator receipts have not landed yet."
            },
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
            "reason": "Qwen3 strict CUDA proof receipts are fallback-free and quality-gated, but every reviewed profile is single-run evidence, CUDA total time is slower than the same-artifact CPU reference in the committed receipts, pure H2D copy timing is unmeasured, and no profile-specific speedup threshold has been accepted."
        },
        "qualification_requirements": [
            {
                "id": "same_artifact_tokenizer_prompt_policy",
                "description": "Same Qwen3 artifact SHA, tokenizer authority, prompt template, deterministic policy, and fallback-free CPU/CUDA route.",
                "status": "passed"
            },
            {
                "id": "current_proof_receipts",
                "description": "One-token, short-decode, and warm-session strict CUDA proof receipts exist for the exact Qwen3 artifact.",
                "status": "passed"
            },
            {
                "id": "repeated_cpu_cuda_comparator",
                "description": "At least three CPU and CUDA runs exist for every reviewed Qwen3 profile.",
                "status": "blocked",
                "blocker": "CUDA-MODEL-007 only has one committed proof receipt per profile; repeated benchmark comparator receipts are still missing."
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
                "blocker": "The current Qwen3 receipts record a model-load wall-clock envelope, not pure CUDA event copy timing."
            },
            {
                "id": "profile_outperforms_cpu_reference",
                "description": "Each profile's CUDA total time is faster than the same-artifact CPU reference total time.",
                "status": "blocked",
                "blocker": "The committed Qwen3 one-token, short-decode, and warm-session CUDA total times are all slower than their CPU reference totals."
            },
            {
                "id": "profile_specific_thresholds",
                "description": "Profile-specific speedup thresholds are accepted before benchmark_qualified_speedup may become true.",
                "status": "blocked",
                "blocker": "No Qwen3 profile-specific speedup threshold has been accepted."
            }
        ],
        "profile_reviews": profiles,
        "evidence_summary": evidence,
        "transfer_timing_review": {
            "status": "host_to_device_model_load_envelope_device_to_host_measured",
            "device_to_host_timing_recorded": true,
            "host_to_device_timing_recorded": true,
            "host_to_device_model_load_envelope_recorded": true,
            "host_to_device_pure_transfer_timing_recorded": false,
            "host_to_device_blocker": "The Qwen3 runtime records an H2D model-load wall-clock envelope, but not pure CUDA event copy timing.",
            "device_to_host_source": "wall_clock_extract_logits_2d_local",
            "host_to_device_source": "wall_clock_model_load_with_cuda_weight_upload",
            "host_to_device_scope": "model_load_wall_clock_envelope",
            "host_to_device_ms_includes_non_transfer_overhead": true
        },
        "hardware_context": hardware_context(&[one_token, short_decode, warm_session])?,
        "cuda": warm_session.value.pointer("/cuda").cloned().ok_or("cuda missing")?,
        "claim_boundaries": [
            "speedup_claim=false; no Qwen3 profile is upgraded by this review.",
            "benchmark_qualified_speedup=false; current Qwen3 CUDA totals are slower than same-artifact CPU totals.",
            "Only one proof receipt per profile exists; repeated benchmark comparator evidence is still required.",
            "H2D model-load envelope timing is recorded, but pure CUDA event H2D copy timing remains unmeasured.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof."
        ]
    }))
}

fn profile_review(profile: &str, proof: &ProofReceipt) -> Result<Value, Box<dyn Error>> {
    let cpu_total = number_at(&proof.value, "/timing/cpu_reference_total_ms")?;
    let cuda_total = number_at(&proof.value, "/timing/total_ms")?;
    Ok(json!({
        "profile": profile,
        "decision": "not_accepted",
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": true,
        "quality_passed": bool_at(&proof.value, "/quality_gate/passed")?,
        "generated_token_ids_match": generated_token_ids_match(profile, &proof.value)?,
        "dense_cuda_evidence_used": true,
        "runs_per_backend": 1,
        "repeated_evidence": false,
        "cpu_total_ms_mean": cpu_total,
        "cpu_total_ms_p50": cpu_total,
        "cpu_total_ms_p95": cpu_total,
        "cuda_total_ms_mean": cuda_total,
        "cuda_total_ms_p50": cuda_total,
        "cuda_total_ms_p95": cuda_total,
        "observed_cpu_total_ms_div_cuda_total_ms": cpu_total / cuda_total,
        "cuda_mean_slower_than_cpu": cuda_total > cpu_total,
        "first_token_ms": number_at(&proof.value, "/timing/first_token_ms")?,
        "decode_total_ms": decode_total_ms(&proof.value)?,
        "kernel_time_ms": number_at(&proof.value, "/timing/kernel_time_ms")?,
        "host_to_device_bytes": u64_at(&proof.value, "/timing/host_to_device_bytes")?,
        "device_to_host_bytes": u64_at(&proof.value, "/timing/device_to_host_bytes")?,
        "host_to_device_ms": number_at(&proof.value, "/timing/host_to_device_ms")?,
        "host_to_device_ms_source": str_at(&proof.value, "/timing/host_to_device_ms_source")?,
        "host_to_device_ms_scope": str_at(&proof.value, "/timing/host_to_device_ms_scope")?,
        "host_to_device_ms_includes_non_transfer_overhead": bool_at(&proof.value, "/timing/host_to_device_ms_includes_non_transfer_overhead")?,
        "pure_host_to_device_ms": null,
        "pure_host_to_device_ms_source": "not_measured_by_qwen3_runtime",
        "device_to_host_ms": number_at(&proof.value, "/timing/device_to_host_ms")?,
        "device_to_host_ms_source": str_at(&proof.value, "/timing/device_to_host_ms_source")?,
        "reason": "This Qwen3 profile remains proof evidence, not speed evidence, because it has only one CPU/CUDA receipt, CUDA total time is slower than CPU total time, pure H2D timing is incomplete, and no profile threshold is accepted.",
        "blockers": [
            "repeated CPU/CUDA comparator evidence is missing",
            "CUDA total time is slower than CPU total time in the committed proof receipt",
            "pure host-to-device transfer timing is unmeasured",
            "no profile-specific speedup threshold has been accepted"
        ]
    }))
}

fn evidence_summary(profile: &str, proof: &ProofReceipt) -> Result<Value, Box<dyn Error>> {
    let cpu_total = number_at(&proof.value, "/timing/cpu_reference_total_ms")?;
    let cuda_total = number_at(&proof.value, "/timing/total_ms")?;
    Ok(json!({
        "profile": profile,
        "runs_per_backend": 1,
        "repeated_evidence": false,
        "fallback_free": true,
        "quality_passed": bool_at(&proof.value, "/quality_gate/passed")?,
        "generated_token_ids_match": generated_token_ids_match(profile, &proof.value)?,
        "cpu_total_ms_mean": cpu_total,
        "cpu_total_ms_p50": cpu_total,
        "cpu_total_ms_p95": cpu_total,
        "cuda_total_ms_mean": cuda_total,
        "cuda_total_ms_p50": cuda_total,
        "cuda_total_ms_p95": cuda_total,
        "observed_cpu_total_ms_div_cuda_total_ms": cpu_total / cuda_total,
        "cuda_mean_slower_than_cpu": cuda_total > cpu_total,
        "device_to_host_ms": number_at(&proof.value, "/timing/device_to_host_ms")?,
        "host_to_device_ms": number_at(&proof.value, "/timing/host_to_device_ms")?,
        "host_to_device_ms_source": str_at(&proof.value, "/timing/host_to_device_ms_source")?,
        "host_to_device_ms_scope": str_at(&proof.value, "/timing/host_to_device_ms_scope")?,
        "host_to_device_ms_includes_non_transfer_overhead": bool_at(&proof.value, "/timing/host_to_device_ms_includes_non_transfer_overhead")?,
        "pure_host_to_device_ms": null,
        "pure_host_to_device_ms_source": "not_measured_by_qwen3_runtime",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false
    }))
}

fn proof_input(path: &Path, receipt: &Value) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "path": path_label(path),
        "sha256": sha256_file(path)?,
        "artifact_kind": str_at(receipt, "/artifact_kind")?
    }))
}

fn model_for_review(warm_session: &ProofReceipt) -> Result<Value, Box<dyn Error>> {
    let mut model = warm_session.value.pointer("/model").cloned().ok_or("model missing")?;
    if let Some(object) = model.as_object_mut() {
        object.remove("path");
    }
    Ok(model)
}

fn tokenizer_authority_for_review(one_token: &ProofReceipt) -> Result<Value, Box<dyn Error>> {
    let mut authority = one_token
        .value
        .pointer("/tokenizer_prompt_authority")
        .cloned()
        .ok_or("tokenizer_prompt_authority missing")?;
    if authority.get("prompt_token_count").is_none() {
        authority["prompt_token_count"] =
            json!(u64_at(&one_token.value, "/one_token_proof/prompt_token_count")?);
    }
    Ok(authority)
}

fn hardware_context(proofs: &[&ProofReceipt; 3]) -> Result<Value, Box<dyn Error>> {
    let mut power_values = Vec::new();
    let mut temp_values = Vec::new();
    let mut vram_values = Vec::new();
    for proof in proofs {
        power_values.push(number_at(&proof.value, "/cuda/power_draw_watts")?);
        temp_values.push(number_at(&proof.value, "/cuda/temperature_c")?);
        vram_values.push(u64_at(&proof.value, "/cuda/vram_bytes")?);
    }
    Ok(json!({
        "source": "qwen3_cuda_model_007_proof_receipts",
        "power_draw_watts_min": min_f64(&power_values),
        "power_draw_watts_max": max_f64(&power_values),
        "temperature_c_min": min_f64(&temp_values),
        "temperature_c_max": max_f64(&temp_values),
        "vram_bytes": *vram_values.iter().min().ok_or("vram values missing")?
    }))
}

fn generated_token_ids_match(profile: &str, proof: &Value) -> Result<bool, Box<dyn Error>> {
    match profile {
        "one_token" => bool_at(proof, "/one_token_proof/selected_token_match"),
        "short_decode_8" => bool_at(proof, "/short_decode_proof/generated_token_ids_match"),
        "warm_session_3_turns" => bool_at(proof, "/warm_session_proof/generated_token_ids_match"),
        other => Err(format!("unsupported profile {other}").into()),
    }
}

fn decode_total_ms(proof: &Value) -> Result<f64, Box<dyn Error>> {
    if let Ok(value) = number_at(proof, "/timing/decode_total_ms") {
        return Ok(value);
    }
    number_at(proof, "/timing/decode_ms")
}

fn ensure_same_model(left: &Value, right: &Value) -> Result<(), Box<dyn Error>> {
    if str_at(left, "/model/id")? != str_at(right, "/model/id")?
        || str_at(left, "/model/sha256")? != str_at(right, "/model/sha256")?
    {
        return Err("proof receipts must use the same Qwen3 model identity".into());
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
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
        .ok_or_else(|| format!("missing string at {pointer}").into())
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing bool at {pointer}").into())
}

fn number_at(value: &Value, pointer: &str) -> Result<f64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing number at {pointer}").into())
}

fn u64_at(value: &Value, pointer: &str) -> Result<u64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing integer at {pointer}").into())
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
