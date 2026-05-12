use bitnet_bench_receipts::{
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
const DEFAULT_RECEIPT_OUT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json";

const DEFAULT_ONE_TOKEN_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-01/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-02/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-03/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json",
];
const DEFAULT_SHORT_DECODE_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-01/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-02/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-03/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json",
];
const DEFAULT_WARM_SESSION_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-01/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-02/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-002/run-03/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json",
];

#[derive(Debug)]
struct Args {
    baseline_receipt: PathBuf,
    one_token_runs: Vec<PathBuf>,
    short_decode_runs: Vec<PathBuf>,
    warm_session_runs: Vec<PathBuf>,
    receipt_out: PathBuf,
}

#[derive(Debug)]
struct ProfileRuns {
    one_token: Vec<(PathBuf, Value)>,
    short_decode: Vec<(PathBuf, Value)>,
    warm_session: Vec<(PathBuf, Value)>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let baseline = read_json(&args.baseline_receipt)?;
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json(&baseline)?;

    let runs = ProfileRuns {
        one_token: read_runs(&args.one_token_runs)?,
        short_decode: read_runs(&args.short_decode_runs)?,
        warm_session: read_runs(&args.warm_session_runs)?,
    };
    assert_repeated_sources(&runs)?;

    let receipt = build_receipt(&args, &baseline, &runs)?;
    validate_dense_gguf_qwen_repeated_comparator_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut baseline_receipt = PathBuf::from(DEFAULT_BASELINE);
    let mut one_token_runs: Vec<PathBuf> =
        DEFAULT_ONE_TOKEN_RUNS.iter().map(PathBuf::from).collect();
    let mut short_decode_runs: Vec<PathBuf> =
        DEFAULT_SHORT_DECODE_RUNS.iter().map(PathBuf::from).collect();
    let mut warm_session_runs: Vec<PathBuf> =
        DEFAULT_WARM_SESSION_RUNS.iter().map(PathBuf::from).collect();
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    let mut one_token_overridden = false;
    let mut short_decode_overridden = false;
    let mut warm_session_overridden = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--baseline-receipt" => baseline_receipt = PathBuf::from(next_value(&mut iter, &arg)?),
            "--one-token-run" => {
                if !one_token_overridden {
                    one_token_runs.clear();
                    one_token_overridden = true;
                }
                one_token_runs.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--short-decode-run" => {
                if !short_decode_overridden {
                    short_decode_runs.clear();
                    short_decode_overridden = true;
                }
                short_decode_runs.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--warm-session-run" => {
                if !warm_session_overridden {
                    warm_session_runs.clear();
                    warm_session_overridden = true;
                }
                warm_session_runs.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args { baseline_receipt, one_token_runs, short_decode_runs, warm_session_runs, receipt_out })
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: dense_qwen_cuda_repeated_comparator_receipt [--baseline-receipt PATH] [--one-token-run PATH ...] [--short-decode-run PATH ...] [--warm-session-run PATH ...] [--receipt-out PATH]"
    );
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_runs(paths: &[PathBuf]) -> Result<Vec<(PathBuf, Value)>, Box<dyn Error>> {
    paths.iter().map(|path| Ok((path.clone(), read_json(path)?))).collect()
}

fn assert_repeated_sources(runs: &ProfileRuns) -> Result<(), Box<dyn Error>> {
    assert_profile_sources(
        "one_token",
        &runs.one_token,
        "dense_gguf_qwen_one_token_strict_cuda_proof",
        "dense_gguf_qwen_one_token_strict_cuda_proof_recorded",
    )?;
    assert_profile_sources(
        "short_decode_8",
        &runs.short_decode,
        "dense_gguf_qwen_short_decode_strict_cuda_proof",
        "dense_gguf_qwen_short_decode_strict_cuda_proof_recorded",
    )?;
    assert_profile_sources(
        "warm_session_3_turns",
        &runs.warm_session,
        "dense_gguf_qwen_warm_session_strict_cuda_proof",
        "dense_gguf_qwen_warm_session_strict_cuda_proof_recorded",
    )?;

    let anchor = runs.one_token.first().ok_or("one_token runs must not be empty")?.1.clone();
    let model_sha = str_at(&anchor, "/model/sha256")?;
    let prompt_template = str_at(&anchor, "/tokenizer_prompt_authority/prompt_template")?;
    let prompt_hash = str_at(&anchor, "/tokenizer_prompt_authority/rendered_prompt_sha256")?;
    let token_hash = str_at(&anchor, "/tokenizer_prompt_authority/prompt_token_ids_sha256")?;

    for (_, receipt) in runs.one_token.iter().chain(runs.short_decode.iter()) {
        if str_at(receipt, "/model/sha256")? != model_sha {
            return Err("all one-token and short-decode runs must use the same model sha256".into());
        }
        if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != prompt_template {
            return Err(
                "all one-token and short-decode runs must use the same prompt template".into()
            );
        }
        if str_at(receipt, "/tokenizer_prompt_authority/rendered_prompt_sha256")? != prompt_hash {
            return Err(
                "all one-token and short-decode runs must use the same rendered prompt hash".into(),
            );
        }
        if str_at(receipt, "/tokenizer_prompt_authority/prompt_token_ids_sha256")? != token_hash {
            return Err(
                "all one-token and short-decode runs must use the same prompt token hash".into()
            );
        }
    }

    for (_, receipt) in &runs.warm_session {
        if str_at(receipt, "/model/sha256")? != model_sha {
            return Err("all warm-session runs must use the same model sha256".into());
        }
        if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != prompt_template {
            return Err("all warm-session runs must use the same prompt template".into());
        }
        if str_at(receipt, "/tokenizer_prompt_authority/turns/0/rendered_prompt_sha256")?
            != prompt_hash
        {
            return Err(
                "warm-session first turns must use the one-token rendered prompt hash".into()
            );
        }
        if str_at(receipt, "/tokenizer_prompt_authority/turns/0/prompt_token_ids_sha256")?
            != token_hash
        {
            return Err("warm-session first turns must use the one-token prompt token hash".into());
        }
    }

    Ok(())
}

fn assert_profile_sources(
    profile: &str,
    runs: &[(PathBuf, Value)],
    expected_artifact_kind: &str,
    expected_claim: &str,
) -> Result<(), Box<dyn Error>> {
    if runs.len() < 3 {
        return Err(format!("{profile} requires at least 3 runs").into());
    }
    let mut paths = std::collections::BTreeSet::new();
    for (path, receipt) in runs {
        if !paths.insert(path_label(path)) {
            return Err(format!("{profile} run paths must be unique").into());
        }
        assert_source_receipt(receipt, expected_artifact_kind, expected_claim)?;
    }
    Ok(())
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

fn build_receipt(
    args: &Args,
    baseline: &Value,
    runs: &ProfileRuns,
) -> Result<Value, Box<dyn Error>> {
    let profiles = vec![
        profile_from_runs("one_token", &runs.one_token)?,
        profile_from_runs("short_decode_8", &runs.short_decode)?,
        profile_from_runs("warm_session_3_turns", &runs.warm_session)?,
    ];
    let comparator_summary = comparator_summary(&profiles);

    Ok(json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_qwen_repeated_comparator",
        "artifact_path": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
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
        "model": baseline.pointer("/model").cloned().ok_or("baseline model missing")?,
        "tokenizer_prompt_authority": baseline
            .pointer("/tokenizer_prompt_authority")
            .cloned()
            .ok_or("baseline tokenizer_prompt_authority missing")?,
        "execution_plan": baseline
            .pointer("/execution_plan")
            .cloned()
            .ok_or("baseline execution_plan missing")?,
        "baseline_input": {
            "path": path_label(&args.baseline_receipt),
            "sha256": sha256_file(&args.baseline_receipt)?,
            "artifact_kind": str_at(baseline, "/artifact_kind")?
        },
        "profiles": profiles,
        "comparator_summary": comparator_summary,
        "transfer_timing": {
            "status": "not_measured_in_source_receipts",
            "source": "source strict CUDA proof receipts record H2D/D2H bytes but do not yet expose transfer event timing",
            "host_to_device_bytes_recorded": true,
            "device_to_host_bytes_recorded": true,
            "host_to_device_timing_recorded": false,
            "device_to_host_timing_recorded": false,
            "next_step": "CUDA-DENSE-PERF-003 transfer timing or profile-specific qualification review"
        },
        "hardware_context": hardware_context(runs)?,
        "cuda": baseline.pointer("/cuda").cloned().ok_or("baseline cuda block missing")?,
        "claim_boundaries": [
            "speedup_claim=false; repeated CPU/CUDA comparator evidence is not a speedup qualification.",
            "benchmark_qualified_speedup=false; no dense profile is accepted by this receipt.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof.",
            "full_cuda_residency_claimed=false; transfer timing and broader residency remain future gates."
        ]
    }))
}

fn profile_from_runs(profile: &str, runs: &[(PathBuf, Value)]) -> Result<Value, Box<dyn Error>> {
    let run_values = runs
        .iter()
        .enumerate()
        .map(|(index, (path, receipt))| run_from_receipt(profile, index + 1, path, receipt))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(json!({
        "profile": profile,
        "status": "repeated_same_artifact_cpu_cuda_comparator",
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "run_count": run_values.len(),
        "cpu_runs": run_values.len(),
        "cuda_runs": run_values.len(),
        "min_runs_per_backend": 3,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_authority": true,
        "deterministic_generation_policy": true,
        "generated_token_ids_match": run_values.iter().all(|run| {
            run.get("generated_token_ids_match").and_then(Value::as_bool) == Some(true)
        }),
        "first_divergence_report": "no generated-token divergence recorded across source receipts",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "transfer_timing_status": "not_measured_in_source_receipts",
        "cpu_total_ms": number_summary(&run_values, "/timing/cpu_total_ms"),
        "cuda_total_ms": number_summary(&run_values, "/timing/cuda_total_ms"),
        "first_token_ms": number_summary(&run_values, "/timing/first_token_ms"),
        "decode_total_ms": number_summary(&run_values, "/timing/decode_total_ms"),
        "kernel_time_ms": number_summary(&run_values, "/timing/kernel_time_ms"),
        "host_to_device_bytes": u64_summary(&run_values, "/timing/host_to_device_bytes"),
        "device_to_host_bytes": u64_summary(&run_values, "/timing/device_to_host_bytes"),
        "runs": run_values
    }))
}

fn run_from_receipt(
    profile: &str,
    index: usize,
    path: &Path,
    receipt: &Value,
) -> Result<Value, Box<dyn Error>> {
    let generated = generated_identity(profile, receipt)?;
    let prompt_token_count = match profile {
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

    let mut run = json!({
        "run_id": format!("run-{index:02}"),
        "profile": profile,
        "source_receipt_path": path_label(path),
        "source_receipt_sha256": sha256_file(path)?,
        "source_artifact_kind": str_at(receipt, "/artifact_kind")?,
        "model_sha256": str_at(receipt, "/model/sha256")?,
        "prompt_template": str_at(receipt, "/tokenizer_prompt_authority/prompt_template")?,
        "prompt_token_count": prompt_token_count,
        "generation_policy": "greedy",
        "deterministic_generation": true,
        "generated_tokens": generated_tokens,
        "generated_token_ids_sha256": generated.ids_sha256,
        "generated_token_ids_match": generated.ids_match,
        "first_divergence_report": generated.first_divergence_report,
        "top_k_compared": generated.top_k_compared,
        "fallback_used": bool_at(receipt, "/fallback_used")?,
        "quality_passed": bool_at(receipt, "/quality_gate/passed")?,
        "parity_passed": bool_at(receipt, "/parity/passed")?,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "timing": {
            "cpu_total_ms": number_at(receipt, "/timing/cpu_reference_total_ms")?,
            "cuda_total_ms": number_at(receipt, "/timing/total_ms")?,
            "first_token_ms": number_at(receipt, "/timing/first_token_ms")?,
            "decode_total_ms": decode_total_ms(receipt)?,
            "kernel_time_ms": number_at(receipt, "/timing/kernel_time_ms")?,
            "kernel_invocations": u64_at(receipt, "/timing/kernel_invocations")?,
            "kernel_launches": u64_at(receipt, "/timing/kernel_launches")?,
            "host_to_device_bytes": u64_at(receipt, "/timing/host_to_device_bytes")?,
            "device_to_host_bytes": u64_at(receipt, "/timing/device_to_host_bytes")?,
            "host_to_device_ms": null,
            "host_to_device_ms_source": "not_measured_in_source_receipt",
            "device_to_host_ms": null,
            "device_to_host_ms_source": "not_measured_in_source_receipt"
        }
    });

    if profile == "warm_session_3_turns" {
        run["turns_count"] = json!(u64_at(receipt, "/warm_session_proof/turns_count")?);
    }
    Ok(run)
}

struct GeneratedIdentity {
    ids_sha256: String,
    ids_match: bool,
    first_divergence_report: String,
    top_k_compared: bool,
}

fn generated_identity(profile: &str, receipt: &Value) -> Result<GeneratedIdentity, Box<dyn Error>> {
    match profile {
        "one_token" => {
            let cpu = u64_at(receipt, "/one_token_proof/cpu_selected_token_id")?;
            let cuda = u64_at(receipt, "/one_token_proof/cuda_selected_token_id")?;
            let ids_sha256 =
                str_at(receipt, "/one_token_proof/cpu_logits_top_k_sha256")?.to_owned();
            Ok(GeneratedIdentity {
                ids_sha256,
                ids_match: cpu == cuda,
                first_divergence_report: if cpu == cuda {
                    "none; cpu_selected_token_id matches cuda_selected_token_id".to_owned()
                } else {
                    format!("cpu_selected_token_id={cpu}, cuda_selected_token_id={cuda}")
                },
                top_k_compared: bool_at(receipt, "/one_token_proof/top_k_compared")?,
            })
        }
        "short_decode_8" => Ok(GeneratedIdentity {
            ids_sha256: str_at(receipt, "/short_decode_proof/cpu_generated_token_ids_sha256")?
                .to_owned(),
            ids_match: bool_at(receipt, "/short_decode_proof/generated_token_ids_match")?,
            first_divergence_report: divergence_report(
                receipt.pointer("/short_decode_proof/first_token_divergence_index"),
            ),
            top_k_compared: bool_at(receipt, "/short_decode_proof/top_k_compared")?,
        }),
        "warm_session_3_turns" => Ok(GeneratedIdentity {
            ids_sha256: str_at(receipt, "/warm_session_proof/cpu_generated_token_ids_sha256")?
                .to_owned(),
            ids_match: bool_at(receipt, "/warm_session_proof/generated_token_ids_match")?,
            first_divergence_report: divergence_report(
                receipt.pointer("/warm_session_proof/first_token_divergence"),
            ),
            top_k_compared: bool_at(receipt, "/warm_session_proof/top_k_compared")?,
        }),
        _ => Err(format!("unknown profile {profile}").into()),
    }
}

fn divergence_report(value: Option<&Value>) -> String {
    match value {
        Some(Value::Null) | None => "none".to_owned(),
        Some(value) => format!("first divergence: {value}"),
    }
}

fn comparator_summary(profiles: &[Value]) -> Value {
    let total_runs = profiles.iter().map(|profile| u64_value(profile, "run_count")).sum::<u64>();
    json!({
        "status": "repeated_comparator_only",
        "profiles_recorded": profiles.len(),
        "min_runs_per_backend": 3,
        "total_cpu_runs": total_runs,
        "total_cuda_runs": total_runs,
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
            "profile-specific speedup thresholds remain unreviewed",
            "PERF-002 records repeated comparator evidence only"
        ],
        "next_step": "CUDA-DENSE-PERF-003 profile-specific speedup qualification review or transfer timing"
    })
}

fn hardware_context(runs: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let receipts = runs
        .one_token
        .iter()
        .chain(runs.short_decode.iter())
        .chain(runs.warm_session.iter())
        .map(|(_, receipt)| receipt)
        .collect::<Vec<_>>();
    let powers = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/power_draw_watts"))
        .collect::<Result<Vec<_>, _>>()?;
    let temperatures = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/temperature_c"))
        .collect::<Result<Vec<_>, _>>()?;
    let first_receipt =
        receipts.first().ok_or("hardware context requires at least one source receipt")?;
    let vram = u64_at(first_receipt, "/cuda/vram_bytes")?;

    Ok(json!({
        "vram_bytes": vram,
        "power_draw_watts_min": min_f64(&powers),
        "power_draw_watts_max": max_f64(&powers),
        "temperature_c_min": min_f64(&temperatures),
        "temperature_c_max": max_f64(&temperatures),
        "source": "NVML fields recorded in source strict CUDA proof receipts"
    }))
}

fn decode_total_ms(receipt: &Value) -> Result<f64, Box<dyn Error>> {
    receipt
        .pointer("/timing/decode_total_ms")
        .or_else(|| receipt.pointer("/timing/decode_ms"))
        .and_then(Value::as_f64)
        .ok_or_else(|| "/timing/decode_total_ms or /timing/decode_ms must be a number".into())
}

fn number_summary(runs: &[Value], pointer: &str) -> Value {
    let values = runs
        .iter()
        .filter_map(|run| run.pointer(pointer).and_then(Value::as_f64))
        .collect::<Vec<_>>();
    json!({
        "count": values.len(),
        "min": min_f64(&values),
        "mean": mean_f64(&values),
        "max": max_f64(&values)
    })
}

fn u64_summary(runs: &[Value], pointer: &str) -> Value {
    let values = runs
        .iter()
        .filter_map(|run| run.pointer(pointer).and_then(Value::as_u64))
        .collect::<Vec<_>>();
    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<u64>() as f64 / values.len() as f64
    };
    json!({
        "count": values.len(),
        "min": min,
        "mean": mean,
        "max": max
    })
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::min).unwrap_or(0.0)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
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
