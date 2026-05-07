use bitnet_bench_receipts::validate_strict_bitnet_cuda_benchmark_receipt_json;
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json";
const CPU_SELECTOR_REASON: &str =
    "current CLI does not expose a strict end-to-end selector for this CPU SIMD mode";

#[derive(Debug)]
struct Args {
    cuda_receipt: PathBuf,
    cpu_avx512_receipt: PathBuf,
    cpu_scalar_receipt: Option<PathBuf>,
    cpu_avx2_receipt: Option<PathBuf>,
    receipt_out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let cuda = read_json(&args.cuda_receipt)?;
    let cpu_avx512 = read_json(&args.cpu_avx512_receipt)?;
    let cpu_scalar = read_optional_json(args.cpu_scalar_receipt.as_deref())?;
    let cpu_avx2 = read_optional_json(args.cpu_avx2_receipt.as_deref())?;

    assert_same_strict_decode_inputs(&cuda, &cpu_avx512)?;
    if let Some(cpu_scalar) = &cpu_scalar {
        assert_same_strict_decode_inputs(&cuda, cpu_scalar)?;
    }
    if let Some(cpu_avx2) = &cpu_avx2 {
        assert_same_strict_decode_inputs(&cuda, cpu_avx2)?;
    }

    let receipt = build_receipt(&args, &cuda, &cpu_avx512, cpu_scalar.as_ref(), cpu_avx2.as_ref())?;
    validate_strict_bitnet_cuda_benchmark_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut cuda_receipt = None;
    let mut cpu_avx512_receipt = None;
    let mut cpu_scalar_receipt = None;
    let mut cpu_avx2_receipt = None;
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cuda-receipt" => cuda_receipt = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
            "--cpu-avx512-receipt" => {
                cpu_avx512_receipt = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--cpu-scalar-receipt" => {
                cpu_scalar_receipt = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--cpu-avx2-receipt" => {
                cpu_avx2_receipt = Some(PathBuf::from(next_value(&mut iter, &arg)?));
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
        cuda_receipt: cuda_receipt.ok_or("--cuda-receipt is required")?,
        cpu_avx512_receipt: cpu_avx512_receipt.ok_or("--cpu-avx512-receipt is required")?,
        cpu_scalar_receipt,
        cpu_avx2_receipt,
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
        "Usage: strict_bitnet_cuda_benchmark_receipt --cuda-receipt PATH --cpu-avx512-receipt PATH [--cpu-scalar-receipt PATH] [--cpu-avx2-receipt PATH] [--receipt-out PATH]"
    );
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_optional_json(path: Option<&Path>) -> Result<Option<Value>, Box<dyn Error>> {
    path.map(read_json).transpose()
}

fn build_receipt(
    args: &Args,
    cuda: &Value,
    cpu_avx512: &Value,
    cpu_scalar: Option<&Value>,
    cpu_avx2: Option<&Value>,
) -> Result<Value, Box<dyn Error>> {
    let cpu_total_ms = total_ms(cpu_avx512)?;
    let cuda_total_ms = total_ms(cuda)?;
    let cpu_cuda_ratio = if cuda_total_ms > 0.0 { cpu_total_ms / cuda_total_ms } else { 0.0 };
    let cuda_kernel_invocations = u64_at(cuda, "/cuda/cuda_kernel_invocations")
        .or_else(|_| u64_at(cuda, "/cuda_kernel_invocations"))?;
    let weights_uploaded_once = bool_at(cuda, "/bitnet/weights_uploaded_once")?;
    let per_token_weight_upload = bool_at(cuda, "/bitnet/per_token_weight_upload")?;

    Ok(json!({
        "schema": 1,
        "artifact_kind": "strict_bitnet_cuda_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "claim": "strict_bitnet_cuda_benchmark_baseline",
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "proof_inputs": {
            "cuda_short_decode_receipt": path_label(&args.cuda_receipt),
            "cpu_avx512_short_decode_receipt": path_label(&args.cpu_avx512_receipt),
            "cpu_scalar_short_decode_receipt": args.cpu_scalar_receipt.as_ref().map(|path| path_label(path)),
            "cpu_avx2_short_decode_receipt": args.cpu_avx2_receipt.as_ref().map(|path| path_label(path))
        },
        "model": {
            "repo": str_at(cuda, "/model/repo")?,
            "file": str_at(cuda, "/model/file")?,
            "sha256": str_at(cuda, "/model/sha256")?,
            "format": str_at(cuda, "/model/format")?,
            "architecture": str_at(cuda, "/model/architecture")?,
            "loader_mode": "strict",
            "source_loader_mode": str_at(cuda, "/model/loader_mode")?,
            "fallback_loader_used": bool_at(cuda, "/model/fallback_loader_used")?
        },
        "loader": {
            "mode": str_at(cuda, "/loader/mode")?,
            "strict_mode": strict_loader_mode(cuda)?,
            "minimal_loader_fallback_used": bool_at(cuda, "/loader/minimal_loader_fallback_used")?,
            "mock_tensors_used": bool_at(cuda, "/loader/mock_tensors_used")?
        },
        "tokenizer": {
            "source": str_at(cuda, "/tokenizer/source")?,
            "strict": bool_at(cuda, "/tokenizer/strict")?,
            "type": str_at(cuda, "/tokenizer/type")?
        },
        "bitnet": {
            "quantization": str_at(cuda, "/bitnet/quantization")?,
            "kernel_family": str_at(cuda, "/kernel/family")?,
            "layout": str_at(cuda, "/kernel/layout")?,
            "weights_uploaded_once": weights_uploaded_once,
            "per_token_weight_upload": per_token_weight_upload
        },
        "workload": {
            "profile": "short_decode_8",
            "prompt": str_at(cuda, "/prompt")?,
            "prompt_tokens": execution_u64(cuda, "prompt_tokens")?,
            "generated_tokens": execution_u64(cuda, "generated_tokens")?,
            "generated_text": str_at(cuda, "/text")?,
            "generated_token_ids": cuda.pointer("/tokens/ids").cloned().unwrap_or(Value::Null),
            "cpu_cuda_output_match": generated_output_matches(cuda, cpu_avx512)
        },
        "sampling": {
            "greedy": bool_at(cuda, "/gen_policy/greedy")?,
            "deterministic": bool_at(cuda, "/gen_policy/deterministic")?,
            "temperature": number_at(cuda, "/gen_policy/temperature")?,
            "seed": u64_at(cuda, "/gen_policy/seed")?
        },
        "comparison_contract": {
            "same_model": same_model(cuda, cpu_avx512),
            "same_tokenizer": same_tokenizer(cuda, cpu_avx512),
            "same_prompt": str_at(cuda, "/prompt")? == str_at(cpu_avx512, "/prompt")?,
            "same_generated_token_count": execution_u64(cuda, "generated_tokens")? == execution_u64(cpu_avx512, "generated_tokens")?,
            "same_strict_loader_mode": strict_loader_mode(cuda)? && strict_loader_mode(cpu_avx512)?,
            "same_sampling_policy": same_sampling_policy(cuda, cpu_avx512),
            "fallback_free": !bool_at(cuda, "/fallback_used")? && !bool_at(cpu_avx512, "/fallback_used")?
        },
        "cuda": {
            "available": bool_at(cuda, "/cuda/available")?,
            "device_count": u64_at(cuda, "/cuda/device_count")?,
            "device_index": u64_at(cuda, "/cuda/device_index")?,
            "device_name": str_at(cuda, "/cuda/device_name")?,
            "compute_capability": str_at(cuda, "/cuda/compute_capability")?,
            "driver_version": str_at(cuda, "/cuda/driver_version")?,
            "cuda_runtime_version": str_at(cuda, "/cuda/cuda_runtime_version")?,
            "cuda_toolkit_version": str_at(cuda, "/cuda/cuda_toolkit_version")?,
            "nvrtc_version": str_at(cuda, "/cuda/nvrtc_version")?,
            "vram_bytes": u64_at(cuda, "/cuda/vram_bytes")?,
            "memory_hwm_bytes": u64_at(cuda, "/cuda/memory_hwm_bytes")?,
            "memory_hwm_source": str_at(cuda, "/cuda/memory_hwm_source")?,
            "cuda_kernel_invocations": cuda_kernel_invocations
        },
        "benchmark": {
            "profile": "short_decode_8",
            "prompt_profile": "short_decode_37_prompt_tokens_8_generated_tokens",
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "cpu_avx512_total_ms": cpu_total_ms,
            "cuda_total_ms": cuda_total_ms,
            "cpu_avx512_first_token_ms": first_token_ms(cpu_avx512)?,
            "cuda_first_token_ms": first_token_ms(cuda)?,
            "cpu_avx512_tokens_per_second": tokens_per_second(cpu_avx512)?,
            "cuda_tokens_per_second": tokens_per_second(cuda)?,
            "cpu_avx512_total_ms_div_cuda_total_ms": cpu_cuda_ratio,
            "cuda_total_ms_div_cpu_avx512_total_ms": if cpu_total_ms > 0.0 { cuda_total_ms / cpu_total_ms } else { 0.0 },
            "cuda_kernel_invocations": cuda_kernel_invocations,
            "cpu_cuda_output_match": generated_output_matches(cuda, cpu_avx512),
            "speedup_claim": false
        },
        "profiles": [
            profile_or_not_run(cpu_scalar, "amd-9950x3d-cpu-scalar")?,
            profile_or_not_run(cpu_avx2, "amd-9950x3d-cpu-avx2")?,
            measured_profile(cpu_avx512, "amd-9950x3d-cpu-avx512", "cpu")?,
            measured_profile(cuda, "nvidia-rtx-5070-ti-cuda", "cuda")?
        ],
        "kernel_stats": cuda.pointer("/kernel_stats").cloned().unwrap_or_else(|| json!([])),
        "execution_coverage": cuda.pointer("/execution_coverage").cloned().unwrap_or(Value::Null),
        "claim_boundaries": [
            "speedup_claim=false; this receipt records a same-model baseline only.",
            "CPU scalar and AVX2 strict end-to-end profiles are present as explicit not_run entries unless matching strict receipts are supplied.",
            format!(
                "The source CUDA receipt records weights_uploaded_once={weights_uploaded_once} and per_token_weight_upload={per_token_weight_upload}; speedup_claim remains false until a refreshed same-model benchmark proves it."
            )
        ],
        "artifact_path": path_label(&args.receipt_out)
    }))
}

fn assert_same_strict_decode_inputs(cuda: &Value, cpu: &Value) -> Result<(), Box<dyn Error>> {
    if !same_model(cuda, cpu) {
        return Err("model identity differs between CUDA and CPU receipts".into());
    }
    if !same_tokenizer(cuda, cpu) {
        return Err("tokenizer identity differs between CUDA and CPU receipts".into());
    }
    if str_at(cuda, "/prompt")? != str_at(cpu, "/prompt")? {
        return Err("prompt differs between CUDA and CPU receipts".into());
    }
    if execution_u64(cuda, "generated_tokens")? != execution_u64(cpu, "generated_tokens")? {
        return Err("generated token count differs between CUDA and CPU receipts".into());
    }
    if !strict_loader_mode(cuda)? || !strict_loader_mode(cpu)? {
        return Err("both receipts must use strict loader mode".into());
    }
    if bool_at(cuda, "/fallback_used")? || bool_at(cpu, "/fallback_used")? {
        return Err("fallback_used must be false for both receipts".into());
    }
    Ok(())
}

fn same_model(left: &Value, right: &Value) -> bool {
    str_at(left, "/model/repo").ok() == str_at(right, "/model/repo").ok()
        && str_at(left, "/model/file").ok() == str_at(right, "/model/file").ok()
        && str_at(left, "/model/sha256").ok() == str_at(right, "/model/sha256").ok()
}

fn same_tokenizer(left: &Value, right: &Value) -> bool {
    str_at(left, "/tokenizer/source").ok() == str_at(right, "/tokenizer/source").ok()
        && bool_at(left, "/tokenizer/strict").ok() == bool_at(right, "/tokenizer/strict").ok()
        && str_at(left, "/tokenizer/type").ok() == str_at(right, "/tokenizer/type").ok()
}

fn same_sampling_policy(left: &Value, right: &Value) -> bool {
    bool_at(left, "/gen_policy/greedy").ok() == bool_at(right, "/gen_policy/greedy").ok()
        && bool_at(left, "/gen_policy/deterministic").ok()
            == bool_at(right, "/gen_policy/deterministic").ok()
        && number_at(left, "/gen_policy/temperature").ok()
            == number_at(right, "/gen_policy/temperature").ok()
}

fn strict_loader_mode(receipt: &Value) -> Result<bool, Box<dyn Error>> {
    Ok(bool_at(receipt, "/loader/minimal_fallback_disabled")?
        && !bool_at(receipt, "/loader/minimal_loader_fallback_used")?
        && !bool_at(receipt, "/loader/mock_tensors_used")?
        && !bool_at(receipt, "/model/fallback_loader_used")?)
}

fn generated_output_matches(left: &Value, right: &Value) -> bool {
    str_at(left, "/text").ok() == str_at(right, "/text").ok()
        && left.pointer("/tokens/ids") == right.pointer("/tokens/ids")
}

fn profile_or_not_run(receipt: Option<&Value>, backend: &str) -> Result<Value, Box<dyn Error>> {
    receipt.map_or_else(
        || {
            Ok(json!({
                "backend": backend,
                "runtime_api": "cpu",
                "status": "not_run",
                "reason": CPU_SELECTOR_REASON
            }))
        },
        |receipt| measured_profile(receipt, backend, "cpu"),
    )
}

fn measured_profile(
    receipt: &Value,
    backend: &str,
    runtime_api: &str,
) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured",
        "selected_backend": str_at(receipt, "/selected_backend")?,
        "kernel_id": str_at(receipt, "/kernel/kernel_id")?,
        "kernel_implementation": str_at(receipt, "/kernel/implementation")?,
        "total_ms": total_ms(receipt)?,
        "first_token_ms": first_token_ms(receipt)?,
        "tokens_per_second": tokens_per_second(receipt)?,
        "prompt_tokens": execution_u64(receipt, "prompt_tokens")?,
        "generated_tokens": execution_u64(receipt, "generated_tokens")?,
        "fallback_used": bool_at(receipt, "/fallback_used")?
    }))
}

fn total_ms(value: &Value) -> Result<f64, Box<dyn Error>> {
    number_at(value, "/latency/total_ms").or_else(|_| number_at(value, "/timing/total_ms"))
}

fn first_token_ms(value: &Value) -> Result<f64, Box<dyn Error>> {
    number_at(value, "/latency/decode_first_ms")
        .or_else(|_| number_at(value, "/timing/first_token_ms"))
}

fn tokens_per_second(value: &Value) -> Result<f64, Box<dyn Error>> {
    number_at(value, "/throughput/tokens_per_second")
}

fn execution_u64(value: &Value, field: &str) -> Result<u64, Box<dyn Error>> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .or_else(|| value.pointer(&format!("/execution/{field}")).and_then(Value::as_u64))
        .ok_or_else(|| format!("{field} must be an unsigned integer").into())
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

fn timestamp_label() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn path_label(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}
