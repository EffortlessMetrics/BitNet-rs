#![recursion_limit = "256"]

use bitnet_bench_receipts::validate_strict_cuda_warm_session_benchmark_receipt_json;
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json";
const DEFAULT_STRICT_ASK_REFERENCE: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json";

#[derive(Debug)]
struct Args {
    cuda_warm_session_receipts: Vec<PathBuf>,
    strict_ask_reference_receipt: PathBuf,
    receipt_out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.cuda_warm_session_receipts.len() < 2 {
        return Err("at least two CUDA warm-session receipts are required".into());
    }

    let receipts = read_receipts(&args.cuda_warm_session_receipts)?;
    for receipt in &receipts {
        assert_warm_session_receipt(receipt)?;
    }
    for receipt in receipts.iter().skip(1) {
        assert_same_session_inputs(&receipts[0], receipt)?;
    }

    let receipt = build_receipt(&args, &receipts)?;
    validate_strict_cuda_warm_session_benchmark_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut cuda_warm_session_receipts = Vec::new();
    let mut strict_ask_reference_receipt = PathBuf::from(DEFAULT_STRICT_ASK_REFERENCE);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cuda-warm-session-receipt" => {
                cuda_warm_session_receipts.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--strict-ask-reference-receipt" => {
                strict_ask_reference_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args { cuda_warm_session_receipts, strict_ask_reference_receipt, receipt_out })
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: strict_cuda_warm_session_benchmark_receipt --cuda-warm-session-receipt PATH [repeat for N runs] [--receipt-out PATH]"
    );
}

fn read_receipts(paths: &[PathBuf]) -> Result<Vec<Value>, Box<dyn Error>> {
    paths.iter().map(|path| read_json(path)).collect()
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn build_receipt(args: &Args, receipts: &[Value]) -> Result<Value, Box<dyn Error>> {
    let first = &receipts[0];
    let runs_per_backend = receipts.len() as u64;
    let turn_count = u64_at(first, "/session/turn_count")?;
    let runs = build_runs(&args.cuda_warm_session_receipts, receipts)?;
    let kernel_stats = aggregate_kernel_stats(&runs)?;
    let cuda_execution_residency = aggregate_cuda_residency(first, &kernel_stats);
    let execution_plan = execution_plan_from_source(first)?;
    let first_run_generated_tokens = u64_at(&runs[0], "/generated_tokens_total")?;
    let first_run_prompt_tokens = u64_at(&runs[0], "/prompt_tokens_total")?;

    Ok(json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_warm_session_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
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
        "execution_plan": execution_plan,
        "proof_inputs": {
            "cuda_warm_session_receipts": path_labels(&args.cuda_warm_session_receipts),
            "strict_ask_reference_receipt": path_label(&args.strict_ask_reference_receipt)
        },
        "model": {
            "repo": str_at(first, "/model/repo")?,
            "file": str_at(first, "/model/file")?,
            "sha256": str_at(first, "/model/sha256")?,
            "format": str_at(first, "/model/format")?,
            "architecture": str_at(first, "/model/architecture")?,
            "loader_mode": "strict_real_gguf",
            "source_loader_mode": str_at(first, "/model/loader_mode")?,
            "fallback_loader_used": bool_at(first, "/model/fallback_loader_used")?
        },
        "tokenizer": {
            "source": str_at(first, "/tokenizer/source")?,
            "strict": bool_at(first, "/tokenizer/strict")?,
            "type": str_at(first, "/tokenizer/type")?,
            "model_family": str_at(first, "/tokenizer/model_family")?,
            "pretokenizer_authority": str_at(first, "/tokenizer/pretokenizer_authority")?
        },
        "generation": {
            "prompt_template": str_at(first, "/generation/prompt_template")?,
            "mode": str_at(first, "/generation/mode")?,
            "deterministic": bool_at(first, "/generation/deterministic")?,
            "temperature": number_at(first, "/generation/temperature")?,
            "max_new_tokens": u64_at(first, "/generation/max_new_tokens")?
        },
        "session_contract": {
            "runs_per_backend": runs_per_backend,
            "turn_count": turn_count,
            "same_model": true,
            "same_tokenizer": true,
            "same_prompts": true,
            "same_sampling_policy": true,
            "fallback_free": true,
            "model_loaded_once": bool_at(first, "/session/model_loaded_once")?,
            "tokenizer_loaded_once": bool_at(first, "/session/tokenizer_loaded_once")?,
            "cuda_context_initialized_once": bool_at(first, "/session/cuda_context_initialized_once")?,
            "qk256_weights_uploaded_once": bool_at(first, "/session/qk256_weights_uploaded_once")?,
            "per_token_weight_upload": bool_at(first, "/session/per_token_weight_upload")?,
            "kv_cache_reuse_policy": str_at(first, "/session/kv_cache_reuse_policy")?,
            "kv_cache_reuse_claimed": false,
            "speedup_claim": false
        },
        "workload": {
            "profile": "strict_cuda_warm_session_2_turns",
            "turn_count": turn_count,
            "generated_tokens_total": first_run_generated_tokens,
            "prompt_tokens_total": first_run_prompt_tokens,
            "quality_passed": true,
            "prompts": workload_prompts(first)?,
            "answers": workload_answers(first)?
        },
        "benchmark": {
            "profile": "strict_cuda_warm_session_2_turns",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "runs_per_backend": runs_per_backend,
            "turns_per_run": turn_count,
            "cuda_median_total_session_ms": median_f64(values_at(&runs, "/total_session_ms")?),
            "cuda_median_kernel_time_ms": median_f64(values_at(&runs, "/kernel_time_ms")?),
            "cuda_median_generated_tokens_per_second": median_f64(values_at(&runs, "/generated_tokens_per_second")?),
            "cuda_median_host_to_device_bytes": median_u64_value(u64_values_at(&runs, "/host_to_device_bytes")?),
            "cuda_median_device_to_host_bytes": median_u64_value(u64_values_at(&runs, "/device_to_host_bytes")?),
            "quality_passed": true,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false
        },
        "summary": {
            "backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "runs": runs_per_backend,
            "quality_passed": true,
            "fallback_used": false,
            "total_session_ms": metric_summary(values_at(&runs, "/total_session_ms")?),
            "model_load_ms": metric_summary(values_at(&runs, "/model_load_ms")?),
            "tokenizer_load_ms": metric_summary(values_at(&runs, "/tokenizer_load_ms")?),
            "cuda_probe_ms": metric_summary(values_at(&runs, "/cuda_probe_ms")?),
            "kernel_time_ms": metric_summary(values_at(&runs, "/kernel_time_ms")?),
            "generated_tokens_per_second": metric_summary(values_at(&runs, "/generated_tokens_per_second")?),
            "host_to_device_bytes": u64_summary(u64_values_at(&runs, "/host_to_device_bytes")?),
            "device_to_host_bytes": u64_summary(u64_values_at(&runs, "/device_to_host_bytes")?),
            "memory_hwm_bytes": u64_summary(u64_values_at(&runs, "/memory_hwm_bytes")?)
        },
        "runs": runs,
        "cuda": {
            "available": bool_at(first, "/cuda/available")?,
            "device_count": u64_at(first, "/cuda/device_count")?,
            "device_index": u64_at(first, "/cuda/device_index")?,
            "device_name": str_at(first, "/cuda/device_name")?,
            "compute_capability": str_at(first, "/cuda/compute_capability")?,
            "driver_version": str_at(first, "/cuda/driver_version")?,
            "cuda_runtime_version": str_at(first, "/cuda/cuda_runtime_version")?,
            "cuda_toolkit_version": str_at(first, "/cuda/cuda_toolkit_version")?,
            "nvrtc_version": str_at(first, "/cuda/nvrtc_version")?,
            "vram_bytes": u64_at(first, "/cuda/vram_bytes")?,
            "memory_hwm_bytes": receipts.iter().filter_map(|receipt| receipt.pointer("/cuda/memory_hwm_bytes").and_then(Value::as_u64)).max().unwrap_or(1),
            "memory_hwm_source": str_at(first, "/cuda/memory_hwm_source")?,
            "cuda_kernel_invocations": kernel_stats["invocations"].as_u64().unwrap_or(0),
            "power_limit_watts": first.pointer("/cuda/power_limit_watts").cloned().unwrap_or(Value::Null),
            "power_draw_watts": first.pointer("/cuda/power_draw_watts").cloned().unwrap_or(Value::Null),
            "power_draw_watts_summary": optional_number_summary(receipts, "/cuda/power_draw_watts"),
            "temperature_c": first.pointer("/cuda/temperature_c").cloned().unwrap_or(Value::Null),
            "temperature_c_summary": optional_number_summary(receipts, "/cuda/temperature_c")
        },
        "kernel_stats": [kernel_stats],
        "cuda_execution_residency": cuda_execution_residency,
        "claim_boundaries": [
            "speedup_claim=false; repeated strict CUDA warm-session timing is baseline evidence only until explicit benchmark review upgrades a specific profile.",
            "Every run uses the same official Microsoft I2_S model, explicit tokenizer, bitnetcpp-answer prompt template, deterministic policy, and fallback_used=false boundary.",
            "This receipt qualifies only strict_cuda_warm_session_2_turns; it does not claim broad chat quality, production server readiness, full CUDA residency, or general speedup.",
            "Dense regular-LLM CUDA evidence remains separate from BitNet packed QK256 evidence."
        ],
        "artifact_path": path_label(&args.receipt_out)
    }))
}

fn build_runs(paths: &[PathBuf], receipts: &[Value]) -> Result<Vec<Value>, Box<dyn Error>> {
    receipts
        .iter()
        .zip(paths)
        .enumerate()
        .map(|(index, (receipt, path))| run_record(index + 1, path, receipt))
        .collect()
}

fn run_record(repeat_index: usize, path: &Path, receipt: &Value) -> Result<Value, Box<dyn Error>> {
    let generated_tokens_total = sum_turn_u64(receipt, "generated_tokens")?;
    let total_session_ms = number_at(receipt, "/timing/total_session_ms")?;
    let generated_tokens_per_second = if total_session_ms > 0.0 {
        generated_tokens_total as f64 / (total_session_ms / 1000.0)
    } else {
        0.0
    };

    Ok(json!({
        "profile": "strict_cuda_warm_session_2_turns",
        "backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "status": "measured",
        "repeat_index": repeat_index,
        "source_receipt_path": path_label(path),
        "selected_backend": str_at(receipt, "/selected_backend")?,
        "kernel_id": str_at(receipt, "/kernel_stats/0/kernel_id")?,
        "quality_passed": bool_at(receipt, "/quality_summary/passed")? && bool_at(receipt, "/strict_session_validation/passed")?,
        "fallback_used": bool_at(receipt, "/fallback_used")?,
        "model_loaded_once": bool_at(receipt, "/session/model_loaded_once")?,
        "tokenizer_loaded_once": bool_at(receipt, "/session/tokenizer_loaded_once")?,
        "cuda_context_initialized_once": bool_at(receipt, "/session/cuda_context_initialized_once")?,
        "qk256_weights_uploaded_once": bool_at(receipt, "/session/qk256_weights_uploaded_once")?,
        "per_token_weight_upload": bool_at(receipt, "/session/per_token_weight_upload")?,
        "turn_count": u64_at(receipt, "/session/turn_count")?,
        "generated_tokens_total": generated_tokens_total,
        "prompt_tokens_total": sum_turn_u64(receipt, "prompt_tokens")?,
        "total_session_ms": total_session_ms,
        "model_load_ms": number_at(receipt, "/timing/model_load_ms")?,
        "tokenizer_load_ms": number_at(receipt, "/timing/tokenizer_load_ms")?,
        "cuda_probe_ms": number_at(receipt, "/timing/cuda_probe_ms")?,
        "kernel_time_ms": number_at(receipt, "/timing/cuda_kernel_time_ms")?,
        "generated_tokens_per_second": generated_tokens_per_second,
        "kernel_invocations": u64_at(receipt, "/kernel_stats/0/invocations")?,
        "host_to_device_bytes": u64_at(receipt, "/timing/host_to_device_bytes")?,
        "device_to_host_bytes": u64_at(receipt, "/timing/device_to_host_bytes")?,
        "memory_hwm_bytes": u64_at(receipt, "/cuda/memory_hwm_bytes")?,
        "execution_plan": execution_plan_from_source(receipt)?,
        "turns": turn_records(receipt)?
    }))
}

fn turn_records(receipt: &Value) -> Result<Vec<Value>, Box<dyn Error>> {
    let turns = receipt.get("turns").and_then(Value::as_array).ok_or("turns must be an array")?;
    turns
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            Ok(json!({
                "turn_index": index + 1,
                "source_turn_index": u64_at(turn, "/turn_index")?,
                "prompt": str_at(turn, "/prompt")?,
                "answer_trimmed": str_at(turn, "/answer")?.trim(),
                "generated_tokens": u64_at(turn, "/generated_tokens")?,
                "prompt_tokens": u64_at(turn, "/prompt_tokens")?,
                "quality_passed": bool_at(turn, "/quality/garbage_filter_passed")?,
                "fallback_used": bool_at(turn, "/backend/fallback_used")?,
                "execution_plan": turn.pointer("/execution_plan").cloned().unwrap_or(Value::Null),
                "kernel_time_ms": number_at(turn, "/cuda_execution_residency/host_device_transfer_accounting/kernel_time_ms")?,
                "host_to_device_bytes": u64_at(turn, "/cuda_execution_residency/host_device_transfer_accounting/host_to_device_bytes")?,
                "device_to_host_bytes": u64_at(turn, "/cuda_execution_residency/host_device_transfer_accounting/device_to_host_bytes")?
            }))
        })
        .collect()
}

fn execution_plan_from_source(source: &Value) -> Result<Value, Box<dyn Error>> {
    source
        .pointer("/execution_plan")
        .filter(|plan| plan.is_object())
        .cloned()
        .ok_or_else(|| "CUDA warm-session source receipt must include execution_plan".into())
}

fn workload_prompts(receipt: &Value) -> Result<Vec<Value>, Box<dyn Error>> {
    let turns = receipt.get("turns").and_then(Value::as_array).ok_or("turns must be an array")?;
    turns
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            let prompt = str_at(turn, "/prompt")?;
            let expected_answer_scope =
                if prompt.contains("2+2") { "exact_trimmed_4" } else { "quality_gate" };
            Ok(json!({
                "turn_index": index + 1,
                "prompt": prompt,
                "expected_answer_scope": expected_answer_scope
            }))
        })
        .collect()
}

fn workload_answers(receipt: &Value) -> Result<Vec<String>, Box<dyn Error>> {
    let turns = receipt.get("turns").and_then(Value::as_array).ok_or("turns must be an array")?;
    turns.iter().map(|turn| Ok(str_at(turn, "/answer")?.to_string())).collect()
}

fn aggregate_kernel_stats(runs: &[Value]) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "kernel_id": "qk256_gemv_cuda",
        "invocations": sum_u64(runs, "/kernel_invocations")?,
        "fallback_invocations": 0,
        "kernel_launches": sum_u64(runs, "/kernel_invocations")?,
        "kernel_time_ms": values_at(runs, "/kernel_time_ms")?.iter().sum::<f64>(),
        "host_to_device_bytes": sum_u64(runs, "/host_to_device_bytes")?,
        "device_to_host_bytes": sum_u64(runs, "/device_to_host_bytes")?
    }))
}

fn aggregate_cuda_residency(first: &Value, kernel_stats: &Value) -> Value {
    let mut residency = first
        .pointer("/cuda_execution_residency")
        .cloned()
        .unwrap_or_else(|| json!({ "schema_version": "1.0.0" }));
    if let Some(object) = residency.as_object_mut() {
        object.insert("speedup_claim".to_string(), json!(false));
        object.insert("full_cuda_residency_claimed".to_string(), json!(false));
        object.insert(
            "host_device_transfer_accounting".to_string(),
            json!({
                "status": "qk256_measured",
                "host_to_device_bytes": kernel_stats["host_to_device_bytes"].clone(),
                "device_to_host_bytes": kernel_stats["device_to_host_bytes"].clone(),
                "kernel_time_ms": kernel_stats["kernel_time_ms"].clone(),
                "note": "Repeated strict warm-session aggregate of QK256 activation/output transfer bytes and CUDA event kernel time; not a full transformer residency claim."
            }),
        );
    }
    residency
}

fn assert_warm_session_receipt(receipt: &Value) -> Result<(), Box<dyn Error>> {
    if str_at(receipt, "/artifact_kind")? != "bitnet_cuda_warm_session" {
        return Err("source receipt must be bitnet_cuda_warm_session".into());
    }
    if str_at(receipt, "/selected_backend")? != "nvidia-rtx-5070-ti-cuda" {
        return Err("source receipt must select nvidia-rtx-5070-ti-cuda".into());
    }
    if str_at(receipt, "/runtime_api")? != "cuda" {
        return Err("source receipt must use runtime_api=cuda".into());
    }
    if bool_at(receipt, "/fallback_used")? {
        return Err("source receipt must be fallback-free".into());
    }
    if bool_at(receipt, "/speedup_claim")? {
        return Err("source receipt must not claim speedup".into());
    }
    if !bool_at(receipt, "/quality_summary/passed")?
        || !bool_at(receipt, "/strict_session_validation/passed")?
    {
        return Err("source receipt must pass quality and strict session validation".into());
    }
    if !bool_at(receipt, "/session/model_loaded_once")?
        || !bool_at(receipt, "/session/tokenizer_loaded_once")?
        || !bool_at(receipt, "/session/cuda_context_initialized_once")?
        || !bool_at(receipt, "/session/qk256_weights_uploaded_once")?
    {
        return Err("source receipt must prove warm-session reuse".into());
    }
    if bool_at(receipt, "/session/per_token_weight_upload")? {
        return Err("source receipt must not upload weights per token".into());
    }
    if str_at(receipt, "/kernel_stats/0/kernel_id")? != "qk256_gemv_cuda" {
        return Err("source receipt must use qk256_gemv_cuda".into());
    }
    if u64_at(receipt, "/kernel_stats/0/fallback_invocations")? != 0 {
        return Err("source receipt must have zero kernel fallback invocations".into());
    }
    Ok(())
}

fn assert_same_session_inputs(first: &Value, other: &Value) -> Result<(), Box<dyn Error>> {
    for pointer in [
        "/model/repo",
        "/model/file",
        "/model/sha256",
        "/tokenizer/source",
        "/tokenizer/type",
        "/tokenizer/pretokenizer_authority",
        "/generation/prompt_template",
        "/generation/mode",
        "/generation/max_new_tokens",
        "/session/turn_count",
    ] {
        if first.pointer(pointer) != other.pointer(pointer) {
            return Err(format!("source receipts must match at {pointer}").into());
        }
    }
    if bool_at(first, "/generation/deterministic")? != bool_at(other, "/generation/deterministic")?
        || number_at(first, "/generation/temperature")?
            != number_at(other, "/generation/temperature")?
        || bool_at(first, "/tokenizer/strict")? != bool_at(other, "/tokenizer/strict")?
    {
        return Err(
            "source receipts must share tokenizer and deterministic generation policy".into()
        );
    }

    let first_turns = first
        .get("turns")
        .and_then(Value::as_array)
        .ok_or("first receipt turns must be an array")?;
    let other_turns = other
        .get("turns")
        .and_then(Value::as_array)
        .ok_or("other receipt turns must be an array")?;
    if first_turns.len() != other_turns.len() {
        return Err("source receipts must have the same turn count".into());
    }
    for (index, (left, right)) in first_turns.iter().zip(other_turns).enumerate() {
        if str_at(left, "/prompt")? != str_at(right, "/prompt")? {
            return Err(format!("source receipts differ at turn {index} prompt").into());
        }
        if str_at(left, "/answer")?.trim() != str_at(right, "/answer")?.trim() {
            return Err(format!("source receipts differ at turn {index} answer").into());
        }
        if left.pointer("/generated_token_ids") != right.pointer("/generated_token_ids") {
            return Err(format!("source receipts differ at turn {index} generated ids").into());
        }
    }
    Ok(())
}

fn metric_summary(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let samples = values.len();
    let sum: f64 = values.iter().sum();
    json!({
        "samples": samples,
        "min": values.first().copied().unwrap_or(0.0),
        "max": values.last().copied().unwrap_or(0.0),
        "mean": if samples == 0 { 0.0 } else { sum / samples as f64 },
        "median": median_sorted_f64(&values)
    })
}

fn u64_summary(mut values: Vec<u64>) -> Value {
    values.sort_unstable();
    let samples = values.len();
    let sum: u64 = values.iter().sum();
    json!({
        "samples": samples,
        "min": values.first().copied().unwrap_or(0),
        "max": values.last().copied().unwrap_or(0),
        "mean": if samples == 0 { 0.0 } else { sum as f64 / samples as f64 },
        "median": median_sorted_u64(&values)
    })
}

fn optional_number_summary(receipts: &[Value], pointer: &str) -> Value {
    let values = receipts
        .iter()
        .filter_map(|receipt| receipt.pointer(pointer).and_then(Value::as_f64))
        .collect::<Vec<_>>();
    if values.is_empty() { Value::Null } else { metric_summary(values) }
}

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    median_sorted_f64(&values)
}

fn median_sorted_f64(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        len if len % 2 == 1 => values[len / 2],
        len => (values[len / 2 - 1] + values[len / 2]) / 2.0,
    }
}

fn median_u64_value(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    match values.len() {
        0 => 0,
        len => values[len / 2],
    }
}

fn median_sorted_u64(values: &[u64]) -> f64 {
    match values.len() {
        0 => 0.0,
        len if len % 2 == 1 => values[len / 2] as f64,
        len => (values[len / 2 - 1] as f64 + values[len / 2] as f64) / 2.0,
    }
}

fn values_at(runs: &[Value], pointer: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    runs.iter().map(|run| number_at(run, pointer)).collect()
}

fn u64_values_at(runs: &[Value], pointer: &str) -> Result<Vec<u64>, Box<dyn Error>> {
    runs.iter().map(|run| u64_at(run, pointer)).collect()
}

fn sum_u64(runs: &[Value], pointer: &str) -> Result<u64, Box<dyn Error>> {
    Ok(u64_values_at(runs, pointer)?.iter().sum())
}

fn sum_turn_u64(receipt: &Value, field: &str) -> Result<u64, Box<dyn Error>> {
    let turns = receipt.get("turns").and_then(Value::as_array).ok_or("turns must be an array")?;
    turns
        .iter()
        .map(|turn| {
            turn.get(field)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("turn.{field} must be an unsigned integer").into())
        })
        .sum()
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
        .ok_or_else(|| format!("{pointer} must be a boolean").into())
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

fn path_labels(paths: &[PathBuf]) -> Vec<String> {
    paths.iter().map(|path| path_label(path)).collect()
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
