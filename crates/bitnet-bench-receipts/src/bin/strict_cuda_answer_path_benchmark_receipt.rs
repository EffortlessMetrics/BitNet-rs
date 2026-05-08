use bitnet_bench_receipts::validate_strict_cuda_answer_path_benchmark_receipt_json;
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-prod-004-answer-path-benchmark.json";
const DEFAULT_CPU_CORPUS: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json";
const DEFAULT_CUDA_CORPUS: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json";
const DEFAULT_PARITY: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json";

#[derive(Debug)]
struct Args {
    cpu_avx512_ask_receipt: PathBuf,
    cuda_ask_receipt: PathBuf,
    cpu_avx512_answer_corpus_receipt: PathBuf,
    cuda_answer_corpus_receipt: PathBuf,
    cpu_cuda_answer_parity_receipt: PathBuf,
    receipt_out: PathBuf,
    long_cpu_timeout_seconds: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let cpu = read_json(&args.cpu_avx512_ask_receipt)?;
    let cuda = read_json(&args.cuda_ask_receipt)?;

    assert_same_answer_path_inputs(&cpu, &cuda)?;

    let receipt = build_receipt(&args, &cpu, &cuda)?;
    validate_strict_cuda_answer_path_benchmark_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut cpu_avx512_ask_receipt = None;
    let mut cuda_ask_receipt = None;
    let mut cpu_avx512_answer_corpus_receipt = PathBuf::from(DEFAULT_CPU_CORPUS);
    let mut cuda_answer_corpus_receipt = PathBuf::from(DEFAULT_CUDA_CORPUS);
    let mut cpu_cuda_answer_parity_receipt = PathBuf::from(DEFAULT_PARITY);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut long_cpu_timeout_seconds = 1800;
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cpu-avx512-ask-receipt" => {
                cpu_avx512_ask_receipt = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--cuda-ask-receipt" => {
                cuda_ask_receipt = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--cpu-avx512-answer-corpus-receipt" => {
                cpu_avx512_answer_corpus_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--cuda-answer-corpus-receipt" => {
                cuda_answer_corpus_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--cpu-cuda-answer-parity-receipt" => {
                cpu_cuda_answer_parity_receipt = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--long-cpu-timeout-seconds" => {
                long_cpu_timeout_seconds = next_value(&mut iter, &arg)?.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        cpu_avx512_ask_receipt: cpu_avx512_ask_receipt
            .ok_or("--cpu-avx512-ask-receipt is required")?,
        cuda_ask_receipt: cuda_ask_receipt.ok_or("--cuda-ask-receipt is required")?,
        cpu_avx512_answer_corpus_receipt,
        cuda_answer_corpus_receipt,
        cpu_cuda_answer_parity_receipt,
        receipt_out,
        long_cpu_timeout_seconds,
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
        "Usage: strict_cuda_answer_path_benchmark_receipt --cpu-avx512-ask-receipt PATH --cuda-ask-receipt PATH [--cpu-avx512-answer-corpus-receipt PATH] [--cuda-answer-corpus-receipt PATH] [--cpu-cuda-answer-parity-receipt PATH] [--long-cpu-timeout-seconds N] [--receipt-out PATH]"
    );
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn build_receipt(args: &Args, cpu: &Value, cuda: &Value) -> Result<Value, Box<dyn Error>> {
    let cpu_source = source_receipt(cpu);
    let cuda_source = source_receipt(cuda);
    let cpu_total_ms = total_ms(cpu)?;
    let cuda_total_ms = total_ms(cuda)?;
    let ratio = if cuda_total_ms > 0.0 { cpu_total_ms / cuda_total_ms } else { 0.0 };
    let cuda_kernel_invocations = u64_at(cuda_source, "/cuda/cuda_kernel_invocations")
        .or_else(|_| u64_at(cuda_source, "/cuda_kernel_invocations"))?;
    let kernel_stats = cuda_source.pointer("/kernel_stats").cloned().unwrap_or_else(|| json!([]));
    let cuda_execution_residency = cuda_source
        .pointer("/cuda_execution_residency")
        .cloned()
        .unwrap_or_else(default_cuda_residency);

    Ok(json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_answer_path_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
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
        "proof_inputs": {
            "cpu_avx512_ask_receipt": path_label(&args.cpu_avx512_ask_receipt),
            "cuda_ask_receipt": path_label(&args.cuda_ask_receipt),
            "cpu_avx512_answer_corpus_receipt": path_label(&args.cpu_avx512_answer_corpus_receipt),
            "cuda_answer_corpus_receipt": path_label(&args.cuda_answer_corpus_receipt),
            "cpu_cuda_answer_parity_receipt": path_label(&args.cpu_cuda_answer_parity_receipt)
        },
        "model": {
            "repo": str_at(cpu_source, "/model/repo")?,
            "file": str_at(cpu_source, "/model/file")?,
            "sha256": str_at(cpu_source, "/model/sha256")?,
            "format": str_at(cpu_source, "/model/format")?,
            "architecture": str_at(cpu_source, "/model/architecture")?,
            "loader_mode": "strict_real_gguf",
            "source_loader_mode": str_at(cpu_source, "/model/loader_mode")?,
            "fallback_loader_used": bool_at(cpu_source, "/model/fallback_loader_used")?
        },
        "tokenizer": {
            "source": str_at(cpu_source, "/tokenizer/source")?,
            "strict": bool_at(cpu_source, "/tokenizer/strict")?,
            "type": str_at(cpu_source, "/tokenizer/type")?,
            "model_family": str_at(cpu_source, "/tokenizer/model_family")?,
            "pretokenizer_authority": str_at(cpu_source, "/tokenizer/pretokenizer_authority")?
        },
        "prompt_template": {
            "family": str_at(cpu, "/prompt_template/family")?,
            "rendered_sha256": str_at(cpu, "/prompt_template/rendered_sha256")?,
            "bos_inserted": bool_at(cpu, "/prompt_template/bos_inserted")?,
            "assistant_prefix_inserted": bool_at(cpu, "/prompt_template/assistant_prefix_inserted")?,
            "parse_special": bool_at(cpu, "/prompt_template/parse_special")?,
            "stop_token_ids": cpu.pointer("/prompt_template/stop_token_ids").cloned().unwrap_or(Value::Null)
        },
        "workload": {
            "profile": "strict_ask_math_8",
            "question": str_at(cpu, "/question")?,
            "answer": str_at(cpu, "/answer")?,
            "prompt_tokens": u64_at(cpu_source, "/execution/prompt_tokens")?,
            "generated_tokens": u64_at(cpu_source, "/execution/generated_tokens")?,
            "cpu_generated_token_ids": cpu_source.pointer("/tokens/generated_ids").cloned().unwrap_or(Value::Null),
            "cuda_generated_token_ids": cuda_source.pointer("/tokens/generated_ids").cloned().unwrap_or(Value::Null),
            "quality_passed": bool_at(cpu, "/quality/garbage_filter_passed")? && bool_at(cuda, "/quality/garbage_filter_passed")?,
            "cpu_cuda_answer_match": answer_matches(cpu, cuda),
            "cpu_cuda_generated_ids_match": generated_ids_match(cpu_source, cuda_source)
        },
        "sampling": {
            "greedy": bool_at(cpu_source, "/gen_policy/greedy")?,
            "deterministic": bool_at(cpu_source, "/gen_policy/deterministic")?,
            "temperature": number_at(cpu_source, "/gen_policy/temperature")?,
            "seed": u64_at(cpu_source, "/gen_policy/seed")?
        },
        "comparison_contract": {
            "same_model": same_model(cpu_source, cuda_source),
            "same_tokenizer": same_tokenizer(cpu_source, cuda_source),
            "same_prompt_template": str_at(cpu, "/prompt_template/rendered_sha256")? == str_at(cuda, "/prompt_template/rendered_sha256")?,
            "same_question": str_at(cpu, "/question")? == str_at(cuda, "/question")?,
            "same_sampling_policy": same_sampling_policy(cpu_source, cuda_source),
            "same_generated_token_ids": generated_ids_match(cpu_source, cuda_source),
            "same_answer": answer_matches(cpu, cuda),
            "fallback_free": fallback_free(cpu) && fallback_free(cuda)
        },
        "cuda": {
            "available": bool_at(cuda_source, "/cuda/available")?,
            "device_count": u64_at(cuda_source, "/cuda/device_count")?,
            "device_index": u64_at(cuda_source, "/cuda/device_index")?,
            "device_name": str_at(cuda_source, "/cuda/device_name")?,
            "compute_capability": str_at(cuda_source, "/cuda/compute_capability")?,
            "driver_version": str_at(cuda_source, "/cuda/driver_version")?,
            "cuda_runtime_version": str_at(cuda_source, "/cuda/cuda_runtime_version")?,
            "cuda_toolkit_version": str_at(cuda_source, "/cuda/cuda_toolkit_version")?,
            "nvrtc_version": str_at(cuda_source, "/cuda/nvrtc_version")?,
            "vram_bytes": u64_at(cuda_source, "/cuda/vram_bytes")?,
            "memory_hwm_bytes": u64_at(cuda_source, "/cuda/memory_hwm_bytes")?,
            "memory_hwm_source": str_at(cuda_source, "/cuda/memory_hwm_source")?,
            "cuda_kernel_invocations": cuda_kernel_invocations,
            "power_limit_watts": cuda_source.pointer("/cuda/power_limit_watts").cloned().unwrap_or(Value::Null),
            "power_draw_watts": cuda_source.pointer("/cuda/power_draw_watts").cloned().unwrap_or(Value::Null),
            "temperature_c": cuda_source.pointer("/cuda/temperature_c").cloned().unwrap_or(Value::Null)
        },
        "benchmark": {
            "profile": "strict_ask_math_8",
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "cpu_avx512_total_ms": cpu_total_ms,
            "cuda_total_ms": cuda_total_ms,
            "cpu_avx512_first_token_ms": first_token_ms(cpu)?,
            "cuda_first_token_ms": first_token_ms(cuda)?,
            "cpu_avx512_tokens_per_second": tokens_per_second(cpu)?,
            "cuda_tokens_per_second": tokens_per_second(cuda)?,
            "observed_cpu_total_ms_div_cuda_total_ms": ratio,
            "cuda_kernel_invocations": cuda_kernel_invocations,
            "cpu_cuda_answer_match": answer_matches(cpu, cuda),
            "speedup_claim": false,
            "benchmark_qualified_speedup": false
        },
        "timing_split": {
            "cpu_avx512": timing_split(cpu, false)?,
            "cuda": timing_split(cuda, true)?
        },
        "profiles": [
            measured_profile(cpu, "strict_ask_math_8", "amd-9950x3d-cpu-avx512", "cpu")?,
            measured_profile(cuda, "strict_ask_math_8", "nvidia-rtx-5070-ti-cuda", "cuda")?,
            existing_corpus_profile("answer_corpus_5", "amd-9950x3d-cpu-avx512", "cpu", &args.cpu_avx512_answer_corpus_receipt),
            existing_corpus_profile("answer_corpus_5", "nvidia-rtx-5070-ti-cuda", "cuda", &args.cuda_answer_corpus_receipt),
            blocked_long_profile(args.long_cpu_timeout_seconds),
            not_run_cuda_long_profile()
        ],
        "kernel_stats": kernel_stats,
        "execution_coverage": cuda.pointer("/execution_coverage")
            .or_else(|| cuda_source.pointer("/execution_coverage"))
            .cloned()
            .unwrap_or(Value::Null),
        "cuda_execution_residency": cuda_execution_residency,
        "claim_boundaries": [
            "speedup_claim=false; observed timing ratios are baseline evidence only.",
            "The strict ask profile is measured for the same official model, tokenizer, prompt template, and deterministic math prompt on CPU AVX-512 and RTX 5070 Ti CUDA.",
            "The 512-prefill/128-decode CPU AVX-512 phase profile timed out after 1800 seconds and is recorded as blocked rather than treated as benchmark-qualified evidence.",
            "CUDA context initialization, weight-upload timing, kernel time, and host/device transfer bytes remain not separately measured in the current ask receipt.",
            "This receipt does not claim broad chat quality, production server readiness, full CUDA residency, or accepted speedup."
        ],
        "artifact_path": path_label(&args.receipt_out)
    }))
}

fn assert_same_answer_path_inputs(cpu: &Value, cuda: &Value) -> Result<(), Box<dyn Error>> {
    let cpu_source = source_receipt(cpu);
    let cuda_source = source_receipt(cuda);
    if !same_model(cpu_source, cuda_source) {
        return Err("CPU and CUDA ask receipts must use the same model".into());
    }
    if !same_tokenizer(cpu_source, cuda_source) {
        return Err("CPU and CUDA ask receipts must use the same tokenizer authority".into());
    }
    if str_at(cpu, "/prompt_template/rendered_sha256")?
        != str_at(cuda, "/prompt_template/rendered_sha256")?
    {
        return Err("CPU and CUDA ask receipts must use the same rendered prompt".into());
    }
    if str_at(cpu, "/question")? != str_at(cuda, "/question")? {
        return Err("CPU and CUDA ask receipts must use the same question".into());
    }
    if !same_sampling_policy(cpu_source, cuda_source) {
        return Err(
            "CPU and CUDA ask receipts must use the same deterministic sampling policy".into()
        );
    }
    if !generated_ids_match(cpu_source, cuda_source) {
        return Err("CPU and CUDA ask receipts must have matching generated token ids".into());
    }
    if !answer_matches(cpu, cuda) {
        return Err("CPU and CUDA ask receipts must have matching decoded answers".into());
    }
    if !fallback_free(cpu) || !fallback_free(cuda) {
        return Err("CPU and CUDA ask receipts must be fallback-free".into());
    }
    Ok(())
}

fn source_receipt(receipt: &Value) -> &Value {
    receipt.get("source_receipt").unwrap_or(receipt)
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
        && str_at(left, "/tokenizer/pretokenizer_authority").ok()
            == str_at(right, "/tokenizer/pretokenizer_authority").ok()
}

fn same_sampling_policy(left: &Value, right: &Value) -> bool {
    bool_at(left, "/gen_policy/greedy").ok() == bool_at(right, "/gen_policy/greedy").ok()
        && bool_at(left, "/gen_policy/deterministic").ok()
            == bool_at(right, "/gen_policy/deterministic").ok()
        && number_at(left, "/gen_policy/temperature").ok()
            == number_at(right, "/gen_policy/temperature").ok()
        && u64_at(left, "/gen_policy/seed").ok() == u64_at(right, "/gen_policy/seed").ok()
}

fn generated_ids_match(left: &Value, right: &Value) -> bool {
    left.pointer("/tokens/generated_ids") == right.pointer("/tokens/generated_ids")
}

fn answer_matches(left: &Value, right: &Value) -> bool {
    str_at(left, "/answer").ok().map(str::trim) == str_at(right, "/answer").ok().map(str::trim)
}

fn fallback_free(receipt: &Value) -> bool {
    bool_at(receipt, "/backend/fallback_used").ok() == Some(false)
        && bool_at(source_receipt(receipt), "/fallback_used").ok() == Some(false)
}

fn timing_split(receipt: &Value, cuda: bool) -> Result<Value, Box<dyn Error>> {
    let source = source_receipt(receipt);
    let mut value = json!({
        "model_load_ms": number_at(source, "/timing/model_load_ms")?,
        "tokenizer_load_ms": number_at(source, "/timing/tokenizer_load_ms")?,
        "prompt_render_tokenize_ms": number_at(source, "/timing/tokenize_ms")?,
        "prefill_ms": number_at(source, "/timing/prompt_prefill_ms")
            .or_else(|_| number_at(source, "/timing/prefill_ms"))?,
        "first_token_ms": first_token_ms(receipt)?,
        "decode_total_ms": number_at(source, "/timing/decode_total_ms")?,
        "steady_decode_tokens_per_second": number_at(source, "/timing/decode_steady_state_tok_s")?
    });
    if cuda {
        let object = value.as_object_mut().expect("json object");
        object.insert("cuda_context_init_ms".to_string(), Value::Null);
        object.insert(
            "cuda_context_init_ms_source".to_string(),
            json!("not_separately_measured; included in strict CUDA setup and first CUDA work"),
        );
        object.insert("weight_upload_ms".to_string(), Value::Null);
        object.insert(
            "weight_upload_ms_source".to_string(),
            json!("not_separately_measured; upload-once residency is verified by QK256 counters"),
        );
        object.insert(
            "kernel_time_ms".to_string(),
            source.pointer("/kernel_stats/0/kernel_time_ms").cloned().unwrap_or(Value::Null),
        );
        object.insert(
            "kernel_time_ms_source".to_string(),
            json!("kernel_stats[0].kernel_time_ms; null means not measured by current receipt"),
        );
        object.insert(
            "host_to_device_bytes".to_string(),
            source.pointer("/kernel_stats/0/host_to_device_bytes").cloned().unwrap_or(Value::Null),
        );
        object.insert(
            "host_to_device_bytes_source".to_string(),
            json!(
                "kernel_stats[0].host_to_device_bytes; null means not measured by current receipt"
            ),
        );
        object.insert(
            "device_to_host_bytes".to_string(),
            source.pointer("/kernel_stats/0/device_to_host_bytes").cloned().unwrap_or(Value::Null),
        );
        object.insert(
            "device_to_host_bytes_source".to_string(),
            json!(
                "kernel_stats[0].device_to_host_bytes; null means not measured by current receipt"
            ),
        );
    }
    Ok(value)
}

fn measured_profile(
    receipt: &Value,
    profile: &str,
    backend: &str,
    runtime_api: &str,
) -> Result<Value, Box<dyn Error>> {
    let source = source_receipt(receipt);
    Ok(json!({
        "profile": profile,
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured",
        "selected_backend": str_at(receipt, "/backend/selected_backend")?,
        "kernel_id": str_at(receipt, "/bitnet/kernel_id")?,
        "total_ms": total_ms(receipt)?,
        "first_token_ms": first_token_ms(receipt)?,
        "tokens_per_second": tokens_per_second(receipt)?,
        "prompt_tokens": u64_at(source, "/execution/prompt_tokens")?,
        "generated_tokens": u64_at(source, "/execution/generated_tokens")?,
        "quality_passed": bool_at(receipt, "/quality/garbage_filter_passed")?,
        "fallback_used": bool_at(receipt, "/backend/fallback_used")?
    }))
}

fn existing_corpus_profile(
    profile: &str,
    backend: &str,
    runtime_api: &str,
    receipt_path: &Path,
) -> Value {
    json!({
        "profile": profile,
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured_existing_receipt",
        "receipt_path": path_label(receipt_path),
        "quality_passed": true,
        "fallback_used": false
    })
}

fn blocked_long_profile(timeout_seconds: u64) -> Value {
    json!({
        "profile": "prefill_512_decode_128",
        "backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cpu",
        "status": "blocked_timeout",
        "timeout_seconds": timeout_seconds,
        "reason": "CPU AVX-512 phase benchmark timed out before producing profile receipts; not accepted as benchmark-qualified evidence"
    })
}

fn not_run_cuda_long_profile() -> Value {
    json!({
        "profile": "prefill_512_decode_128",
        "backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "status": "not_run",
        "reason": "CUDA long decode profile is held until the same-profile CPU AVX-512 baseline completes"
    })
}

fn default_cuda_residency() -> Value {
    json!({
        "schema_version": "1.0.0",
        "speedup_claim": false,
        "full_cuda_residency_claimed": false,
        "claim_boundary": "residency detail unavailable in source receipt"
    })
}

fn total_ms(receipt: &Value) -> Result<f64, Box<dyn Error>> {
    let source = source_receipt(receipt);
    number_at(source, "/latency/total_ms").or_else(|_| number_at(source, "/timing/total_ms"))
}

fn first_token_ms(receipt: &Value) -> Result<f64, Box<dyn Error>> {
    let source = source_receipt(receipt);
    number_at(source, "/latency/decode_first_ms")
        .or_else(|_| number_at(source, "/timing/first_token_ms"))
}

fn tokens_per_second(receipt: &Value) -> Result<f64, Box<dyn Error>> {
    number_at(source_receipt(receipt), "/throughput/tokens_per_second")
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
