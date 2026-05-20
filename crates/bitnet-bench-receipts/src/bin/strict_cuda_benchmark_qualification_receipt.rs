#![recursion_limit = "256"]

use bitnet_bench_receipts::{
    validate_strict_bitnet_cuda_benchmark_receipt_json,
    validate_strict_cuda_benchmark_qualification_receipt_json,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_PREVIOUS_QUALIFICATION: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-004-benchmark-qualification.json";
const DEFAULT_SHORT_DECODE_BENCHMARK: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json";
const DEFAULT_RECEIPT_OUT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-20/cuda-bitnet-perf-005-profile-matrix-contract.json";
const PERF_005_PROFILES: &[&str] = &[
    "one_token",
    "short_decode_8",
    "short_decode_32",
    "prefill_128_decode_16",
    "prefill_512_decode_32",
    "warm_session_3_turns",
    "warm_session_10_turns",
    "decode_128_from_warm_context",
];

#[derive(Debug)]
struct Args {
    previous_qualification: PathBuf,
    short_decode_benchmark: PathBuf,
    receipt_out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let previous = read_json(&args.previous_qualification)?;
    validate_strict_cuda_benchmark_qualification_receipt_json(&previous)?;

    let short_decode = read_json(&args.short_decode_benchmark)?;
    validate_strict_bitnet_cuda_benchmark_receipt_json(&short_decode)?;

    let receipt = build_receipt(&args, &previous, &short_decode)?;
    validate_strict_cuda_benchmark_qualification_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut previous_qualification = PathBuf::from(DEFAULT_PREVIOUS_QUALIFICATION);
    let mut short_decode_benchmark = PathBuf::from(DEFAULT_SHORT_DECODE_BENCHMARK);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--previous-qualification" => {
                previous_qualification = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--short-decode-benchmark" => {
                short_decode_benchmark = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args { previous_qualification, short_decode_benchmark, receipt_out })
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: strict_cuda_benchmark_qualification_receipt [--previous-qualification PATH] [--short-decode-benchmark PATH] [--receipt-out PATH]"
    );
}

fn build_receipt(
    args: &Args,
    previous: &Value,
    short_decode: &Value,
) -> Result<Value, Box<dyn Error>> {
    let benchmark = object_at(short_decode, "/benchmark")?;
    let cpu_total_ms = number_at(benchmark, "/cpu_avx512_total_ms")?;
    let cuda_total_ms = number_at(benchmark, "/cuda_total_ms")?;
    let ratio = number_at(benchmark, "/cpu_avx512_total_ms_div_cuda_total_ms")?;

    Ok(json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_benchmark_qualification_review",
        "artifact_path": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "strict_cuda_benchmark_qualification_review",
        "fallback_used": false,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "profile_matrix_id": "cuda-bitnet-perf-005",
        "target_profiles": PERF_005_PROFILES,
        "benchmark_policy": {
            "profile_specific_decisions_only": true,
            "global_speedup_claim": false,
            "dense_cuda_evidence_used": false,
            "bitnet_packed_i2s_qk256_only": true
        },
        "model": previous.pointer("/model").cloned().ok_or("previous model missing")?,
        "tokenizer": previous.pointer("/tokenizer").cloned().ok_or("previous tokenizer missing")?,
        "prompt_template": previous
            .pointer("/prompt_template")
            .cloned()
            .ok_or("previous prompt_template missing")?,
        "proof_inputs": {
            "answer_path_baseline_receipt": str_at(previous, "/proof_inputs/answer_path_baseline_receipt")?,
            "repeated_strict_ask_receipt": str_at(previous, "/proof_inputs/repeated_strict_ask_receipt")?,
            "warm_session_benchmark_receipt": str_at(previous, "/proof_inputs/warm_session_benchmark_receipt")?,
            "cpu_cuda_answer_parity_receipt": str_at(previous, "/proof_inputs/cpu_cuda_answer_parity_receipt")?,
            "short_decode_8_benchmark_receipt": path_label(&args.short_decode_benchmark),
            "previous_qualification_review": path_label(&args.previous_qualification)
        },
        "proof_input_hashes": {
            "previous_qualification_review": proof_input(&args.previous_qualification, previous)?,
            "short_decode_8_benchmark_receipt": proof_input(&args.short_decode_benchmark, short_decode)?
        },
        "qualification_decision": {
            "status": "not_accepted",
            "speedup_claim_allowed": false,
            "benchmark_qualified_speedup": false,
            "accepted_profiles": [],
            "blocked_profiles": [
                "one_token",
                "short_decode_8",
                "short_decode_32",
                "prefill_128_decode_16",
                "prefill_512_decode_32",
                "warm_session_3_turns",
                "warm_session_10_turns",
                "decode_128_from_warm_context"
            ],
            "reason": "The official BitNet I2_S/QK256 CUDA path has strong strict-execution evidence, but the product benchmark profile set is not yet governed enough to accept any speedup claim."
        },
        "qualification_requirements": [
            {
                "id": "same_model_tokenizer_prompt_policy",
                "description": "Official Microsoft I2_S GGUF, explicit tokenizer authority, BitNet answer prompt template, deterministic policy, and fallback-free CPU/CUDA route.",
                "status": "passed"
            },
            {
                "id": "short_decode_8_cpu_cuda_baseline",
                "description": "A same-model CPU AVX-512 and RTX 5070 Ti CUDA short_decode_8 baseline exists.",
                "status": "passed"
            },
            {
                "id": "target_profile_repetitions",
                "description": "Each target profile has repeated same-artifact CPU/CUDA measurements.",
                "status": "blocked",
                "blocker": "Only short_decode_8 has a committed CPU/CUDA baseline, and it is not a repeated governed profile set."
            },
            {
                "id": "target_profile_coverage",
                "description": "All CUDA-BITNET-PERF-005 profiles are covered: one_token, short_decode_8, short_decode_32, prefill_128_decode_16, prefill_512_decode_32, warm_session_3_turns, warm_session_10_turns, and decode_128_from_warm_context.",
                "status": "blocked",
                "blocker": "one_token, short_decode_32, prefill_128_decode_16, prefill_512_decode_32, warm_session_3_turns, warm_session_10_turns, and decode_128_from_warm_context product benchmark receipts are not committed."
            },
            {
                "id": "transfer_timing",
                "description": "Host/device transfer timing is measured, not only byte counters.",
                "status": "blocked",
                "blocker": "Current BitNet QK256 benchmark receipts do not record complete H2D/D2H timing for every target profile."
            },
            {
                "id": "profile_specific_speedup_thresholds",
                "description": "Profile-specific speedup acceptance thresholds are reviewed before benchmark_qualified_speedup can become true.",
                "status": "blocked",
                "blocker": "No BitNet QK256 product profile threshold is accepted by this review."
            }
        ],
        "profile_reviews": [
            missing_profile_review("one_token"),
            short_decode_profile_review(cpu_total_ms, cuda_total_ms, ratio),
            missing_profile_review("short_decode_32"),
            missing_profile_review("prefill_128_decode_16"),
            missing_profile_review("prefill_512_decode_32"),
            missing_profile_review("warm_session_3_turns"),
            missing_profile_review("warm_session_10_turns"),
            missing_profile_review("decode_128_from_warm_context")
        ],
        "evidence_summary": {
            "strict_ask_math_8": previous
                .pointer("/evidence_summary/strict_ask_math_8")
                .cloned()
                .ok_or("previous strict_ask_math_8 evidence missing")?,
            "strict_cuda_warm_session_2_turns": previous
                .pointer("/evidence_summary/strict_cuda_warm_session_2_turns")
                .cloned()
                .ok_or("previous warm-session evidence missing")?,
            "one_token": missing_profile_evidence("one_token"),
            "short_decode_8": {
                "profile": "short_decode_8",
                "decision": "not_accepted",
                "evidence_status": "single_run_baseline",
                "fallback_free": true,
                "quality_passed": true,
                "cpu_cuda_output_match": bool_at(benchmark, "/cpu_cuda_output_match")?,
                "cpu_total_ms": cpu_total_ms,
                "cuda_total_ms": cuda_total_ms,
                "observed_cpu_total_ms_div_cuda_total_ms": ratio,
                "speedup_claim": false,
                "benchmark_qualified_speedup": false,
                "reason": "The short_decode_8 CPU/CUDA baseline is useful, but speedup remains blocked until repeated governed profile evidence and transfer timing are complete.",
                "blockers": [
                    "profile is not repeated",
                    "transfer timing is incomplete",
                    "no profile-specific speedup threshold has been accepted"
                ]
            },
            "short_decode_32": missing_profile_evidence("short_decode_32"),
            "prefill_128_decode_16": missing_profile_evidence("prefill_128_decode_16"),
            "prefill_512_decode_32": missing_profile_evidence("prefill_512_decode_32"),
            "warm_session_3_turns": missing_profile_evidence("warm_session_3_turns"),
            "warm_session_10_turns": missing_profile_evidence("warm_session_10_turns"),
            "decode_128_from_warm_context": missing_profile_evidence("decode_128_from_warm_context")
        },
        "cuda": previous.pointer("/cuda").cloned().ok_or("previous cuda missing")?,
        "claim_boundaries": [
            "speedup_claim=false; no BitNet QK256 product benchmark profile is accepted by this receipt.",
            "Profile decisions are explicit and profile-specific; this receipt makes no global CUDA speed claim.",
            "Dense regular-LLM CUDA evidence is excluded and cannot satisfy BitNet packed I2S/QK256 proof.",
            "Generic cuda, WGPU, Vulkan, CPU fallback, or hardware visibility cannot satisfy strict RTX 5070 Ti CUDA benchmark qualification."
        ]
    }))
}

fn missing_profile_review(profile: &str) -> Value {
    json!({
        "profile": profile,
        "decision": "not_accepted",
        "evidence_status": "missing",
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": false,
        "quality_passed": false,
        "dense_cuda_evidence_used": false,
        "reason": "Required product benchmark profile evidence is not committed yet.",
        "blockers": ["profile receipt missing"]
    })
}

fn short_decode_profile_review(cpu_total_ms: f64, cuda_total_ms: f64, ratio: f64) -> Value {
    json!({
        "profile": "short_decode_8",
        "decision": "not_accepted",
        "evidence_status": "single_run_baseline",
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "fallback_free": true,
        "quality_passed": true,
        "dense_cuda_evidence_used": false,
        "cpu_total_ms": cpu_total_ms,
        "cuda_total_ms": cuda_total_ms,
        "observed_cpu_total_ms_div_cuda_total_ms": ratio,
        "reason": "A single short_decode_8 CPU/CUDA baseline exists, but governed speedup requires repeated profile evidence, complete transfer timing, and an accepted profile-specific threshold.",
        "blockers": [
            "profile is not repeated",
            "transfer timing is incomplete",
            "no profile-specific speedup threshold has been accepted"
        ]
    })
}

fn missing_profile_evidence(profile: &str) -> Value {
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

fn proof_input(path: &Path, receipt: &Value) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "path": path_label(path),
        "sha256": sha256_file(path)?,
        "artifact_kind": str_at(receipt, "/artifact_kind")?
    }))
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

fn object_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value, Box<dyn Error>> {
    let object = value.pointer(pointer).ok_or_else(|| format!("{pointer} is missing"))?;
    if !object.is_object() {
        return Err(format!("{pointer} must be an object").into());
    }
    Ok(object)
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

fn number_at(value: &Value, pointer: &str) -> Result<f64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{pointer} must be a number").into())
}
