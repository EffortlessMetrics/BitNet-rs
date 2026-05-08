#![recursion_limit = "256"]

use bitnet_bench_receipts::validate_strict_cuda_repeated_ask_benchmark_receipt_json;
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json";
const DEFAULT_CPU_CORPUS: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json";
const DEFAULT_CUDA_CORPUS: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json";
const DEFAULT_PARITY: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json";

#[derive(Debug)]
struct Args {
    cpu_avx512_ask_receipts: Vec<PathBuf>,
    cuda_ask_receipts: Vec<PathBuf>,
    cpu_avx512_answer_corpus_receipt: PathBuf,
    cuda_answer_corpus_receipt: PathBuf,
    cpu_cuda_answer_parity_receipt: PathBuf,
    receipt_out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.cpu_avx512_ask_receipts.len() != args.cuda_ask_receipts.len() {
        return Err("CPU and CUDA receipt counts must match".into());
    }
    if args.cpu_avx512_ask_receipts.len() < 2 {
        return Err("at least two CPU and CUDA receipts are required".into());
    }

    let cpu_receipts = read_receipts(&args.cpu_avx512_ask_receipts)?;
    let cuda_receipts = read_receipts(&args.cuda_ask_receipts)?;
    for (cpu, cuda) in cpu_receipts.iter().zip(cuda_receipts.iter()) {
        assert_same_answer_path_inputs(cpu, cuda)?;
    }

    let receipt = build_receipt(&args, &cpu_receipts, &cuda_receipts)?;
    validate_strict_cuda_repeated_ask_benchmark_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut cpu_avx512_ask_receipts = Vec::new();
    let mut cuda_ask_receipts = Vec::new();
    let mut cpu_avx512_answer_corpus_receipt = PathBuf::from(DEFAULT_CPU_CORPUS);
    let mut cuda_answer_corpus_receipt = PathBuf::from(DEFAULT_CUDA_CORPUS);
    let mut cpu_cuda_answer_parity_receipt = PathBuf::from(DEFAULT_PARITY);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cpu-avx512-ask-receipt" => {
                cpu_avx512_ask_receipts.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--cuda-ask-receipt" => {
                cuda_ask_receipts.push(PathBuf::from(next_value(&mut iter, &arg)?));
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
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        cpu_avx512_ask_receipts,
        cuda_ask_receipts,
        cpu_avx512_answer_corpus_receipt,
        cuda_answer_corpus_receipt,
        cpu_cuda_answer_parity_receipt,
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
        "Usage: strict_cuda_repeated_ask_benchmark_receipt --cpu-avx512-ask-receipt PATH --cuda-ask-receipt PATH [repeat both flags for N runs] [--receipt-out PATH]"
    );
}

fn read_receipts(paths: &[PathBuf]) -> Result<Vec<Value>, Box<dyn Error>> {
    paths.iter().map(|path| read_json(path)).collect()
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn build_receipt(
    args: &Args,
    cpu_receipts: &[Value],
    cuda_receipts: &[Value],
) -> Result<Value, Box<dyn Error>> {
    let first_cpu = &cpu_receipts[0];
    let first_cuda = &cuda_receipts[0];
    let first_cpu_source = source_receipt(first_cpu);
    let first_cuda_source = source_receipt(first_cuda);
    let runs_per_backend = cpu_receipts.len() as u64;
    let cpu_runs =
        build_runs(&args.cpu_avx512_ask_receipts, cpu_receipts, "amd-9950x3d-cpu-avx512", "cpu")?;
    let cuda_runs =
        build_runs(&args.cuda_ask_receipts, cuda_receipts, "nvidia-rtx-5070-ti-cuda", "cuda")?;
    let cpu_total_summary = metric_summary(
        cpu_runs.iter().map(|run| number_at(run, "/total_ms")).collect::<Result<Vec<_>, _>>()?,
    );
    let cuda_total_summary = metric_summary(
        cuda_runs.iter().map(|run| number_at(run, "/total_ms")).collect::<Result<Vec<_>, _>>()?,
    );
    let median_ratio = if number_at(&cuda_total_summary, "/median")? > 0.0 {
        number_at(&cpu_total_summary, "/median")? / number_at(&cuda_total_summary, "/median")?
    } else {
        0.0
    };
    let kernel_stats = aggregate_cuda_kernel_stats(&cuda_runs)?;
    let cuda_execution_residency = aggregate_cuda_residency(first_cuda, &kernel_stats);
    let execution_plan = execution_plan_from_source(first_cuda_source)?;

    Ok(json!({
        "schema": 1,
        "artifact_kind": "strict_cuda_repeated_ask_benchmark",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
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
        "execution_plan": execution_plan,
        "proof_inputs": {
            "cpu_avx512_ask_receipts": path_labels(&args.cpu_avx512_ask_receipts),
            "cuda_ask_receipts": path_labels(&args.cuda_ask_receipts),
            "cpu_avx512_answer_corpus_receipt": path_label(&args.cpu_avx512_answer_corpus_receipt),
            "cuda_answer_corpus_receipt": path_label(&args.cuda_answer_corpus_receipt),
            "cpu_cuda_answer_parity_receipt": path_label(&args.cpu_cuda_answer_parity_receipt)
        },
        "model": {
            "repo": str_at(first_cpu_source, "/model/repo")?,
            "file": str_at(first_cpu_source, "/model/file")?,
            "sha256": str_at(first_cpu_source, "/model/sha256")?,
            "format": str_at(first_cpu_source, "/model/format")?,
            "architecture": str_at(first_cpu_source, "/model/architecture")?,
            "loader_mode": "strict_real_gguf",
            "source_loader_mode": str_at(first_cpu_source, "/model/loader_mode")?,
            "fallback_loader_used": bool_at(first_cpu_source, "/model/fallback_loader_used")?
        },
        "tokenizer": {
            "source": str_at(first_cpu_source, "/tokenizer/source")?,
            "strict": bool_at(first_cpu_source, "/tokenizer/strict")?,
            "type": str_at(first_cpu_source, "/tokenizer/type")?,
            "model_family": str_at(first_cpu_source, "/tokenizer/model_family")?,
            "pretokenizer_authority": str_at(first_cpu_source, "/tokenizer/pretokenizer_authority")?
        },
        "prompt_template": {
            "family": str_at(first_cpu, "/prompt_template/family")?,
            "rendered_sha256": str_at(first_cpu, "/prompt_template/rendered_sha256")?,
            "bos_inserted": bool_at(first_cpu, "/prompt_template/bos_inserted")?,
            "assistant_prefix_inserted": bool_at(first_cpu, "/prompt_template/assistant_prefix_inserted")?,
            "parse_special": bool_at(first_cpu, "/prompt_template/parse_special")?,
            "stop_token_ids": first_cpu.pointer("/prompt_template/stop_token_ids").cloned().unwrap_or(Value::Null)
        },
        "workload": {
            "profile": "strict_ask_math_8",
            "question": str_at(first_cpu, "/question")?,
            "answer": str_at(first_cpu, "/answer")?,
            "prompt_tokens": u64_at(first_cpu_source, "/execution/prompt_tokens")?,
            "generated_tokens": u64_at(first_cpu_source, "/execution/generated_tokens")?,
            "cpu_generated_token_ids": first_cpu_source.pointer("/tokens/generated_ids").cloned().unwrap_or(Value::Null),
            "cuda_generated_token_ids": first_cuda_source.pointer("/tokens/generated_ids").cloned().unwrap_or(Value::Null),
            "quality_passed": cpu_receipts.iter().all(quality_passed) && cuda_receipts.iter().all(quality_passed),
            "cpu_cuda_answer_match": cpu_receipts.iter().zip(cuda_receipts).all(|(cpu, cuda)| answer_matches(cpu, cuda)),
            "cpu_cuda_generated_ids_match": cpu_receipts.iter().zip(cuda_receipts).all(|(cpu, cuda)| generated_ids_match(source_receipt(cpu), source_receipt(cuda)))
        },
        "sampling": {
            "greedy": bool_at(first_cpu_source, "/gen_policy/greedy")?,
            "deterministic": bool_at(first_cpu_source, "/gen_policy/deterministic")?,
            "temperature": number_at(first_cpu_source, "/gen_policy/temperature")?,
            "seed": u64_at(first_cpu_source, "/gen_policy/seed")?
        },
        "repeat_policy": {
            "runs_per_backend": runs_per_backend,
            "cold_warm_split": "process-level repeated strict ask; every source receipt records a full strict ask process including model load and backend setup",
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
            "runs_per_backend": runs_per_backend,
            "cpu_avx512_median_total_ms": number_at(&cpu_total_summary, "/median")?,
            "cuda_median_total_ms": number_at(&cuda_total_summary, "/median")?,
            "observed_median_cpu_total_ms_div_cuda_total_ms": median_ratio,
            "cpu_cuda_answer_match": true,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false
        },
        "summary": {
            "cpu_avx512": backend_summary(&cpu_runs, "amd-9950x3d-cpu-avx512", "cpu", false)?,
            "cuda": backend_summary(&cuda_runs, "nvidia-rtx-5070-ti-cuda", "cuda", true)?
        },
        "runs": cpu_runs.into_iter().chain(cuda_runs).collect::<Vec<_>>(),
        "pair_contracts": pair_contracts(cpu_receipts, cuda_receipts)?,
        "cuda": {
            "available": bool_at(first_cuda_source, "/cuda/available")?,
            "device_count": u64_at(first_cuda_source, "/cuda/device_count")?,
            "device_index": u64_at(first_cuda_source, "/cuda/device_index")?,
            "device_name": str_at(first_cuda_source, "/cuda/device_name")?,
            "compute_capability": str_at(first_cuda_source, "/cuda/compute_capability")?,
            "driver_version": str_at(first_cuda_source, "/cuda/driver_version")?,
            "cuda_runtime_version": str_at(first_cuda_source, "/cuda/cuda_runtime_version")?,
            "cuda_toolkit_version": str_at(first_cuda_source, "/cuda/cuda_toolkit_version")?,
            "nvrtc_version": str_at(first_cuda_source, "/cuda/nvrtc_version")?,
            "vram_bytes": u64_at(first_cuda_source, "/cuda/vram_bytes")?,
            "memory_hwm_bytes": cuda_receipts.iter().filter_map(|receipt| source_receipt(receipt).pointer("/cuda/memory_hwm_bytes").and_then(Value::as_u64)).max().unwrap_or(1),
            "memory_hwm_source": str_at(first_cuda_source, "/cuda/memory_hwm_source")?,
            "cuda_kernel_invocations": kernel_stats["invocations"].as_u64().unwrap_or(0),
            "power_limit_watts": first_cuda_source.pointer("/cuda/power_limit_watts").cloned().unwrap_or(Value::Null),
            "power_draw_watts": first_cuda_source.pointer("/cuda/power_draw_watts").cloned().unwrap_or(Value::Null),
            "temperature_c": first_cuda_source.pointer("/cuda/temperature_c").cloned().unwrap_or(Value::Null)
        },
        "kernel_stats": [kernel_stats],
        "cuda_execution_residency": cuda_execution_residency,
        "claim_boundaries": [
            "speedup_claim=false; repeated strict ask timing is baseline evidence only until explicit benchmark review upgrades a specific profile.",
            "All repeated runs use the same official Microsoft I2_S model, explicit tokenizer, bitnetcpp-answer prompt template, deterministic sampling policy, and fallback-free CPU/CUDA paths.",
            "This receipt qualifies only strict_ask_math_8 process-level repeated asks; it does not claim broad chat quality, production server readiness, full CUDA residency, or general speedup.",
            "Dense regular-LLM CUDA evidence remains separate from BitNet packed QK256 evidence."
        ],
        "artifact_path": path_label(&args.receipt_out)
    }))
}

fn build_runs(
    paths: &[PathBuf],
    receipts: &[Value],
    backend: &str,
    runtime_api: &str,
) -> Result<Vec<Value>, Box<dyn Error>> {
    receipts
        .iter()
        .zip(paths)
        .enumerate()
        .map(|(index, (receipt, path))| run_record(index + 1, path, receipt, backend, runtime_api))
        .collect()
}

fn run_record(
    repeat_index: usize,
    path: &Path,
    receipt: &Value,
    backend: &str,
    runtime_api: &str,
) -> Result<Value, Box<dyn Error>> {
    let source = source_receipt(receipt);
    let mut run = json!({
        "profile": "strict_ask_math_8",
        "backend": backend,
        "runtime_api": runtime_api,
        "status": "measured",
        "repeat_index": repeat_index,
        "source_receipt_path": path_label(path),
        "selected_backend": str_at(receipt, "/backend/selected_backend")?,
        "kernel_id": str_at(receipt, "/bitnet/kernel_id")?,
        "total_ms": total_ms(receipt)?,
        "first_token_ms": first_token_ms(receipt)?,
        "decode_total_ms": number_at(source, "/timing/decode_total_ms")?,
        "tokens_per_second": tokens_per_second(receipt)?,
        "prompt_tokens": u64_at(source, "/execution/prompt_tokens")?,
        "generated_tokens": u64_at(source, "/execution/generated_tokens")?,
        "answer_trimmed": str_at(receipt, "/answer")?.trim(),
        "generated_token_ids": source.pointer("/tokens/generated_ids").cloned().unwrap_or(Value::Null),
        "quality_passed": bool_at(receipt, "/quality/garbage_filter_passed")?,
        "fallback_used": bool_at(receipt, "/backend/fallback_used")?
    });
    if runtime_api == "cuda" {
        let object = run.as_object_mut().expect("run record object");
        object.insert("execution_plan".to_string(), execution_plan_from_source(source)?);
        object.insert(
            "kernel_invocations".to_string(),
            json!(u64_at(source, "/kernel_stats/0/invocations")?),
        );
        object.insert(
            "kernel_time_ms".to_string(),
            json!(number_at(source, "/kernel_stats/0/kernel_time_ms")?),
        );
        object.insert(
            "host_to_device_bytes".to_string(),
            json!(u64_at(source, "/kernel_stats/0/host_to_device_bytes")?),
        );
        object.insert(
            "device_to_host_bytes".to_string(),
            json!(u64_at(source, "/kernel_stats/0/device_to_host_bytes")?),
        );
    }
    Ok(run)
}

fn backend_summary(
    runs: &[Value],
    backend: &str,
    runtime_api: &str,
    cuda: bool,
) -> Result<Value, Box<dyn Error>> {
    let mut summary = json!({
        "backend": backend,
        "runtime_api": runtime_api,
        "runs": runs.len(),
        "quality_passed": runs.iter().all(|run| bool_at(run, "/quality_passed").unwrap_or(false)),
        "fallback_used": runs.iter().any(|run| bool_at(run, "/fallback_used").unwrap_or(true)),
        "total_ms": metric_summary(values_at(runs, "/total_ms")?),
        "first_token_ms": metric_summary(values_at(runs, "/first_token_ms")?),
        "decode_total_ms": metric_summary(values_at(runs, "/decode_total_ms")?),
        "tokens_per_second": metric_summary(values_at(runs, "/tokens_per_second")?)
    });
    if cuda {
        let object = summary.as_object_mut().expect("summary object");
        object.insert(
            "kernel_time_ms".to_string(),
            metric_summary(values_at(runs, "/kernel_time_ms")?),
        );
        object.insert(
            "host_to_device_bytes".to_string(),
            u64_summary(u64_values_at(runs, "/host_to_device_bytes")?),
        );
        object.insert(
            "device_to_host_bytes".to_string(),
            u64_summary(u64_values_at(runs, "/device_to_host_bytes")?),
        );
    }
    Ok(summary)
}

fn pair_contracts(
    cpu_receipts: &[Value],
    cuda_receipts: &[Value],
) -> Result<Vec<Value>, Box<dyn Error>> {
    cpu_receipts
        .iter()
        .zip(cuda_receipts)
        .enumerate()
        .map(|(index, (cpu, cuda))| {
            Ok(json!({
                "repeat_index": index + 1,
                "same_model": same_model(source_receipt(cpu), source_receipt(cuda)),
                "same_tokenizer": same_tokenizer(source_receipt(cpu), source_receipt(cuda)),
                "same_prompt_template": str_at(cpu, "/prompt_template/rendered_sha256")? == str_at(cuda, "/prompt_template/rendered_sha256")?,
                "same_question": str_at(cpu, "/question")? == str_at(cuda, "/question")?,
                "same_sampling_policy": same_sampling_policy(source_receipt(cpu), source_receipt(cuda)),
                "same_generated_token_ids": generated_ids_match(source_receipt(cpu), source_receipt(cuda)),
                "same_answer": answer_matches(cpu, cuda),
                "fallback_free": fallback_free(cpu) && fallback_free(cuda)
            }))
        })
        .collect()
}

fn aggregate_cuda_kernel_stats(cuda_runs: &[Value]) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "kernel_id": "qk256_gemv_cuda",
        "invocations": sum_u64(cuda_runs, "/kernel_invocations")?,
        "fallback_invocations": 0,
        "kernel_launches": sum_u64(cuda_runs, "/kernel_invocations")?,
        "kernel_time_ms": values_at(cuda_runs, "/kernel_time_ms")?.iter().sum::<f64>(),
        "host_to_device_bytes": sum_u64(cuda_runs, "/host_to_device_bytes")?,
        "device_to_host_bytes": sum_u64(cuda_runs, "/device_to_host_bytes")?
    }))
}

fn aggregate_cuda_residency(first_cuda: &Value, kernel_stats: &Value) -> Value {
    let mut residency = source_receipt(first_cuda)
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
                "note": "Repeated strict ask aggregate of QK256 activation/output transfer bytes and CUDA event kernel time; not a full transformer residency claim."
            }),
        );
    }
    residency
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
        "median": median(&values)
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
        "median": median_u64(&values)
    })
}

fn median(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        len if len % 2 == 1 => values[len / 2],
        len => (values[len / 2 - 1] + values[len / 2]) / 2.0,
    }
}

fn median_u64(values: &[u64]) -> f64 {
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

fn execution_plan_from_source(source: &Value) -> Result<Value, Box<dyn Error>> {
    source
        .pointer("/execution_plan")
        .filter(|plan| plan.is_object())
        .cloned()
        .ok_or_else(|| "CUDA source receipt must include execution_plan".into())
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

fn quality_passed(receipt: &Value) -> bool {
    bool_at(receipt, "/quality/garbage_filter_passed").unwrap_or(false)
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

fn path_labels(paths: &[PathBuf]) -> Vec<String> {
    paths.iter().map(|path| path_label(path)).collect()
}

fn path_label(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}
