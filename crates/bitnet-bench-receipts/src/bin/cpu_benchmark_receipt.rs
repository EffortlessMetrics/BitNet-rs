use bitnet_bench_receipts::validate_strict_cpu_benchmark_receipt_json;
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, QK256_SCALAR_GEMV_KERNEL_ID, gemv_qk256_with_kernel_selection,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const PROFILE_NAMES: [&str; 5] = ["micro", "layer", "prefill", "first_token", "decode"];

#[derive(Debug)]
struct Args {
    receipt_out: Option<PathBuf>,
    requested_kernel: Option<&'static str>,
    strict: bool,
    model_repo: String,
    model_file: String,
    model_sha256: String,
    quant_format: String,
    tokenizer_source: String,
    selected_backend: String,
    prompt_tokens: u64,
    generated_tokens: u64,
    batch_size: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            receipt_out: None,
            requested_kernel: None,
            strict: false,
            model_repo: "fixture/bitnet-qk256".to_string(),
            model_file: "fixture-qk256-i2_s.gguf".to_string(),
            model_sha256: "fixture-not-a-model-hash".to_string(),
            quant_format: "QK256/I2_S".to_string(),
            tokenizer_source: "fixture".to_string(),
            selected_backend: "cpu".to_string(),
            prompt_tokens: 32,
            generated_tokens: 8,
            batch_size: 1,
        }
    }
}

#[derive(Debug)]
struct MeasuredProfile {
    profile: &'static str,
    execution_phase: &'static str,
    requested_kernel: &'static str,
    selected_kernel: &'static str,
    fallback_used: bool,
    fallback_reason: Option<String>,
    rows: usize,
    cols: usize,
    iterations: u64,
    wall_time_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    bandwidth_gbps: f64,
    tokens_per_second: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let receipt = build_receipt(&args)?;
    validate_strict_cpu_benchmark_receipt_json(&receipt)?;

    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = args.receipt_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--receipt-out" => args.receipt_out = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
            "--kernel" | "--requested-kernel" => {
                let value = next_value(&mut iter, &arg)?;
                args.requested_kernel = match value.as_str() {
                    "auto" => None,
                    "qk256-scalar-gemv" => Some(QK256_SCALAR_GEMV_KERNEL_ID),
                    "qk256-avx2-gemv" => Some("qk256-avx2-gemv"),
                    other => return Err(format!("unsupported requested kernel: {other}").into()),
                };
            }
            "--strict" => args.strict = true,
            "--model-repo" => args.model_repo = next_value(&mut iter, &arg)?,
            "--model-file" => args.model_file = next_value(&mut iter, &arg)?,
            "--model-sha256" => args.model_sha256 = next_value(&mut iter, &arg)?,
            "--quant-format" => args.quant_format = next_value(&mut iter, &arg)?,
            "--tokenizer-source" => args.tokenizer_source = next_value(&mut iter, &arg)?,
            "--selected-backend" => args.selected_backend = next_value(&mut iter, &arg)?,
            "--prompt-tokens" => args.prompt_tokens = next_value(&mut iter, &arg)?.parse()?,
            "--generated-tokens" => args.generated_tokens = next_value(&mut iter, &arg)?.parse()?,
            "--batch-size" => args.batch_size = next_value(&mut iter, &arg)?.parse()?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(args)
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: cpu_benchmark_receipt [--receipt-out PATH] [--kernel auto|qk256-scalar-gemv|qk256-avx2-gemv] [--strict]\n\
         Options: --model-repo, --model-file, --model-sha256, --quant-format, --tokenizer-source,\n\
         --selected-backend, --prompt-tokens, --generated-tokens, --batch-size"
    );
}

fn build_receipt(args: &Args) -> Result<serde_json::Value, Box<dyn Error>> {
    let measured = [
        measure_profile("micro", "micro_kernel", 1, 256, 32, args.requested_kernel, args.strict)?,
        measure_profile("layer", "layer_forward", 32, 512, 12, args.requested_kernel, args.strict)?,
        measure_profile(
            "prefill",
            "prefill",
            16,
            512,
            args.prompt_tokens.max(1),
            args.requested_kernel,
            args.strict,
        )?,
        measure_profile(
            "first_token",
            "first_token",
            1,
            1024,
            16,
            args.requested_kernel,
            args.strict,
        )?,
        measure_profile(
            "decode",
            "decode_steady_state",
            1,
            1024,
            args.generated_tokens.max(1),
            args.requested_kernel,
            args.strict,
        )?,
    ];

    let selected_kernel = measured
        .first()
        .map(|profile| profile.selected_kernel)
        .unwrap_or(QK256_SCALAR_GEMV_KERNEL_ID);
    let fallback_used = measured.iter().any(|profile| profile.fallback_used);
    let cpu_features = cpu_features();

    let profiles: Vec<_> = measured
        .iter()
        .map(|profile| {
            json!({
                "profile": profile.profile,
                "execution_phase": profile.execution_phase,
                "status": "measured",
                "requested_kernel": profile.requested_kernel,
                "selected_kernel": profile.selected_kernel,
                "fallback_used": profile.fallback_used,
                "fallback_reason": profile.fallback_reason.as_deref(),
                "shape": {
                    "rows": profile.rows,
                    "cols": profile.cols,
                    "iterations": profile.iterations
                },
                "wall_time_ms": profile.wall_time_ms,
                "median_ms": profile.median_ms,
                "p95_ms": profile.p95_ms,
                "bandwidth_gbps": profile.bandwidth_gbps,
                "tokens_per_second": profile.tokens_per_second
            })
        })
        .collect();

    Ok(json!({
        "schema": 1,
        "artifact_kind": "cpu_benchmark",
        "machine_id": args.selected_backend,
        "hardware_lane": args.selected_backend,
        "timestamp_utc": timestamp_label(),
        "requested_backend": "cpu",
        "selected_backend": args.selected_backend,
        "runtime_api": "cpu",
        "claim": "cpu_benchmark_receipt",
        "speedup_claim": false,
        "fallback_used": fallback_used,
        "fallback_reason": null,
        "model": {
            "repo": args.model_repo,
            "file": args.model_file,
            "sha256": args.model_sha256,
            "family": "bitnet",
            "quant_format": args.quant_format
        },
        "tokenizer": {
            "source": args.tokenizer_source,
            "strict": true
        },
        "kernel": {
            "requested_kernel": args.requested_kernel.unwrap_or("auto"),
            "selected_kernel": selected_kernel,
            "oracle_kernel": QK256_SCALAR_GEMV_KERNEL_ID,
            "fallback_used": fallback_used,
            "fallback_reason": null,
            "dequantizes_before_compute": false
        },
        "cpu": {
            "model": cpu_model_label(),
            "arch": env::consts::ARCH,
            "features": cpu_features,
            "threads": available_threads(),
            "avx512": cpu_has_feature("avx512f"),
            "power_mode": "unknown",
            "temperature_c": null,
            "frequency_mhz": null
        },
        "workload": {
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
            "batch_size": args.batch_size
        },
        "profiles": profiles,
        "profile_order": PROFILE_NAMES,
        "artifact_path": args.receipt_out.as_ref().map(|path| path.display().to_string())
    }))
}

fn measure_profile(
    profile: &'static str,
    execution_phase: &'static str,
    rows: usize,
    cols: usize,
    iterations: u64,
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<MeasuredProfile, Box<dyn Error>> {
    let (packed, row_stride) = create_qk256_weights(rows, cols);
    let activations = create_activation_vector(cols);
    let mut output = vec![0.0f32; rows];
    let selection = gemv_qk256_with_kernel_selection(
        &packed,
        &activations,
        &mut output,
        rows,
        cols,
        row_stride,
        requested_kernel,
        strict,
    )?;

    let mut samples = Vec::with_capacity(iterations as usize);
    let wall_start = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        gemv_qk256_with_kernel_selection(
            &packed,
            &activations,
            &mut output,
            rows,
            cols,
            row_stride,
            requested_kernel,
            strict,
        )?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let wall_time_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    samples.sort_by(f64::total_cmp);

    let bytes_per_iteration = packed.len()
        + activations.len() * std::mem::size_of::<f32>()
        + output.len() * std::mem::size_of::<f32>();
    let total_seconds = (wall_time_ms / 1_000.0).max(f64::EPSILON);

    Ok(MeasuredProfile {
        profile,
        execution_phase,
        requested_kernel: selection.requested_kernel.unwrap_or("auto"),
        selected_kernel: selection.selected_kernel,
        fallback_used: selection.fallback_used,
        fallback_reason: selection.fallback_reason,
        rows,
        cols,
        iterations,
        wall_time_ms,
        median_ms: percentile(&samples, 0.50),
        p95_ms: percentile(&samples, 0.95),
        bandwidth_gbps: (bytes_per_iteration as f64 * iterations as f64)
            / total_seconds
            / 1_000_000_000.0,
        tokens_per_second: iterations as f64 / total_seconds,
    })
}

fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let index = ((samples.len() - 1) as f64 * p).ceil() as usize;
    samples[index.min(samples.len() - 1)]
}

fn create_qk256_weights(rows: usize, cols: usize) -> (Vec<u8>, usize) {
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride = blocks_per_row * QK256_PACKED_BYTES;
    let packed =
        (0..rows * row_stride).map(|i| ((i.wrapping_mul(0x55) + i / 7) & 0xFF) as u8).collect();
    (packed, row_stride)
}

fn create_activation_vector(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let x = (i as f32 - cols as f32 / 2.0) / (cols as f32 / 6.0);
            x * (-x * x / 2.0).exp()
        })
        .collect()
}

fn timestamp_label() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn cpu_model_label() -> String {
    env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or(proc_cpuinfo_model_label())
        .unwrap_or_else(|| env::consts::ARCH.to_string())
}

fn proc_cpuinfo_model_label() -> Option<String> {
    #[cfg(not(windows))]
    {
        fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("model name").and_then(|rest| {
                    rest.split_once(':').map(|(_, value)| value.trim().to_string())
                })
            })
        })
    }
    #[cfg(windows)]
    {
        None
    }
}

fn cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    for feature in ["sse2", "avx", "avx2", "fma", "avx512f"] {
        if cpu_has_feature(feature) {
            features.push(feature);
        }
    }
    if features.is_empty() {
        features.push(env::consts::ARCH);
    }
    features
}

fn cpu_has_feature(feature: &str) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match feature {
            "sse2" => std::arch::is_x86_feature_detected!("sse2"),
            "avx" => std::arch::is_x86_feature_detected!("avx"),
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            _ => false,
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = feature;
        false
    }
}
