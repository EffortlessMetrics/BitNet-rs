#![allow(clippy::items_after_test_module)]

use bitnet_bench_receipts::validate_strict_cpu_benchmark_receipt_json;
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, QK256_SCALAR_GEMM_KERNEL_ID, QK256_SCALAR_GEMV_KERNEL_ID,
    Qk256KernelSelection, gemv_qk256_with_kernel_selection, qk256_gemm_scalar,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const PROFILE_NAMES: [&str; 5] = ["micro", "layer", "prefill", "first_token", "decode"];
const TILING_PARALLELISM_DEGREES: [usize; 3] = [2, 4, 8];
const TILING_ROW_BLOCKS: [usize; 4] = [2, 4, 8, 16];
const TILING_COL_BLOCKS: [usize; 4] = [64, 128, 256, 512];
const TILING_THREAD_COUNTS: [usize; 5] = [1, 2, 4, 6, 8];

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
    include_i2s_tiling_matrix: bool,
    include_i2s_applied_thread_matrix: bool,
    include_embedding_quantization_evidence: bool,
    tensor_boundary_audit: Option<PathBuf>,
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
            include_i2s_tiling_matrix: false,
            include_i2s_applied_thread_matrix: false,
            include_embedding_quantization_evidence: false,
            tensor_boundary_audit: None,
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

#[derive(Debug)]
struct KernelMicrobenchProfile {
    profile: &'static str,
    operation: &'static str,
    execution_phase: &'static str,
    requested_kernel: &'static str,
    selected_kernel: &'static str,
    fallback_used: bool,
    fallback_reason: Option<String>,
    rows: usize,
    cols: usize,
    tokens: usize,
    iterations: u64,
    wall_time_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    bandwidth_gbps: f64,
    tokens_per_second: f64,
}

#[derive(Debug, Clone, Copy)]
struct TilingCandidate {
    parallelism_degree: usize,
    row_block: usize,
    col_block: usize,
    thread_count: usize,
}

#[derive(Debug)]
struct TilingMatrixRun {
    candidate: TilingCandidate,
    profile: KernelMicrobenchProfile,
}

#[derive(Debug)]
struct AppliedThreadMatrixRun {
    candidate: TilingCandidate,
    profile: KernelMicrobenchProfile,
    applied_thread_count: usize,
    thread_partition: &'static str,
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
            "--include-i2s-tiling-matrix" => args.include_i2s_tiling_matrix = true,
            "--include-i2s-applied-thread-matrix" => {
                args.include_i2s_applied_thread_matrix = true;
            }
            "--include-embedding-quantization-evidence" => {
                args.include_embedding_quantization_evidence = true;
            }
            "--tensor-boundary-audit" => {
                args.tensor_boundary_audit = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
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
         --selected-backend, --prompt-tokens, --generated-tokens, --batch-size, --include-i2s-tiling-matrix,\n\
         --include-i2s-applied-thread-matrix, --include-embedding-quantization-evidence, --tensor-boundary-audit PATH"
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
    let microbench_profiles = measure_i2s_microbench(args.requested_kernel, args.strict)?;
    let tiling_matrix_runs = if args.include_i2s_tiling_matrix {
        measure_i2s_tiling_thread_matrix(args.requested_kernel, args.strict)?
    } else {
        Vec::new()
    };
    let applied_thread_matrix_runs = if args.include_i2s_applied_thread_matrix {
        measure_i2s_applied_thread_matrix(args.requested_kernel, args.strict)?
    } else {
        Vec::new()
    };

    let selected_kernel = measured
        .first()
        .map(|profile| profile.selected_kernel)
        .unwrap_or(QK256_SCALAR_GEMV_KERNEL_ID);
    let fallback_used = measured.iter().any(|profile| profile.fallback_used)
        || microbench_profiles.iter().any(|profile| profile.fallback_used)
        || tiling_matrix_runs.iter().any(|run| run.profile.fallback_used)
        || applied_thread_matrix_runs.iter().any(|run| run.profile.fallback_used);
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

    let i2s_microbench_profiles: Vec<_> = microbench_profiles
        .iter()
        .map(|profile| {
            json!({
                "profile": profile.profile,
                "operation": profile.operation,
                "execution_phase": profile.execution_phase,
                "status": "measured",
                "requested_kernel": profile.requested_kernel,
                "selected_kernel": profile.selected_kernel,
                "fallback_used": profile.fallback_used,
                "fallback_reason": profile.fallback_reason.as_deref(),
                "shape": {
                    "rows": profile.rows,
                    "cols": profile.cols,
                    "tokens": profile.tokens,
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
    let i2s_tiling_thread_matrix = if args.include_i2s_tiling_matrix {
        Some(build_i2s_tiling_thread_matrix(&tiling_matrix_runs, args.quant_format.as_str()))
    } else {
        None
    };
    let i2s_applied_thread_matrix = if args.include_i2s_applied_thread_matrix {
        Some(build_i2s_applied_thread_matrix(
            &applied_thread_matrix_runs,
            args.quant_format.as_str(),
        ))
    } else {
        None
    };
    let embedding_quantization_evidence = if args.include_embedding_quantization_evidence {
        let audit_path = args
            .tensor_boundary_audit
            .as_ref()
            .ok_or("--include-embedding-quantization-evidence requires --tensor-boundary-audit")?;
        Some(build_embedding_quantization_evidence(audit_path)?)
    } else {
        None
    };

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
            "kernel_family": "i2_s_qk256",
            "requested_kernel": args.requested_kernel.unwrap_or("auto"),
            "selected_kernel": selected_kernel,
            "oracle_kernel": QK256_SCALAR_GEMV_KERNEL_ID,
            "gemm_oracle_kernel": QK256_SCALAR_GEMM_KERNEL_ID,
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
        "i2s_microbench": {
            "work_item": "CPU-BITNET-PERF-001",
            "artifact_kind": "cpu_bitnet_i2s_microbench",
            "claim": "i2_s_gemv_gemm_microbench_receipt",
            "kernel_family": "i2_s_qk256",
            "quantization": args.quant_format,
            "speedup_claim": false,
            "fallback_used": fallback_used,
            "fallback_reason": null,
            "profiles": i2s_microbench_profiles,
            "claim_boundary": [
                "Records QK256/I2_S GEMV and GEMM microbench timing only.",
                "Does not claim answer quality, sustained decode throughput, Arc/NPU execution, acceleration, or QK256 semantic changes."
            ]
        },
        "i2s_tiling_thread_matrix": i2s_tiling_thread_matrix,
        "i2s_applied_thread_matrix": i2s_applied_thread_matrix,
        "embedding_quantization_evidence": embedding_quantization_evidence,
        "profiles": profiles,
        "profile_order": PROFILE_NAMES,
        "artifact_path": args.receipt_out.as_ref().map(|path| path.display().to_string())
    }))
}

fn build_i2s_applied_thread_matrix(
    runs: &[AppliedThreadMatrixRun],
    quant_format: &str,
) -> serde_json::Value {
    let measured_runs: Vec<_> = runs
        .iter()
        .map(|run| {
            let profile = &run.profile;
            json!({
                "profile": profile.profile,
                "operation": profile.operation,
                "execution_phase": profile.execution_phase,
                "status": "measured",
                "candidate": {
                    "parallelism_degree": run.candidate.parallelism_degree,
                    "row_block": run.candidate.row_block,
                    "col_block": run.candidate.col_block,
                    "thread_count": run.candidate.thread_count,
                    "thread_count_applied": true,
                    "thread_count_policy": "applied_scoped_threads",
                    "applied_thread_count": run.applied_thread_count,
                    "thread_partition": run.thread_partition,
                    "thread_count_note": "This sample applies scoped worker threads inside the synthetic QK256/I2_S microbench only; it does not change the full BitNet decode path."
                },
                "requested_kernel": profile.requested_kernel,
                "selected_kernel": profile.selected_kernel,
                "fallback_used": profile.fallback_used,
                "fallback_reason": profile.fallback_reason.as_deref(),
                "shape": {
                    "rows": profile.rows,
                    "cols": profile.cols,
                    "tokens": profile.tokens,
                    "iterations": profile.iterations,
                    "cols_rounded_to_qk256_block": profile.cols != run.candidate.col_block
                },
                "wall_time_ms": profile.wall_time_ms,
                "median_ms": profile.median_ms,
                "p95_ms": profile.p95_ms,
                "bandwidth_gbps": profile.bandwidth_gbps,
                "tokens_per_second": profile.tokens_per_second,
                "speedup_claim": false
            })
        })
        .collect();
    json!({
        "work_item": "CPU-BITNET-PERF-003",
        "artifact_kind": "cpu_bitnet_i2s_applied_thread_matrix",
        "claim": "i2_s_applied_thread_matrix_receipt",
        "kernel_family": "i2_s_qk256",
        "quantization": quant_format,
        "speedup_claim": false,
        "fallback_used": runs.iter().any(|run| run.profile.fallback_used),
        "fallback_reason": null,
        "candidate_grid": {
            "parallelism_degrees": TILING_PARALLELISM_DEGREES,
            "row_blocks": TILING_ROW_BLOCKS,
            "col_blocks": TILING_COL_BLOCKS,
            "thread_counts": TILING_THREAD_COUNTS,
            "candidate_count": TILING_PARALLELISM_DEGREES.len()
                * TILING_ROW_BLOCKS.len()
                * TILING_COL_BLOCKS.len()
                * TILING_THREAD_COUNTS.len()
        },
        "coverage": {
            "status": "sampled_applied_thread_baseline",
            "measured_candidate_count": measured_runs.len(),
            "full_matrix_candidate_count": TILING_PARALLELISM_DEGREES.len()
                * TILING_ROW_BLOCKS.len()
                * TILING_COL_BLOCKS.len()
                * TILING_THREAD_COUNTS.len(),
            "thread_counts_applied": true,
            "thread_count_policy": "applied_scoped_threads",
            "thread_partitions": ["rows", "tokens"],
            "reason": "This receipt applies sampled thread-count candidates to synthetic QK256/I2_S GEMV/GEMM microbenches without upgrading any profile to a speedup or full-decode claim."
        },
        "measured_runs": measured_runs,
        "claim_boundary": [
            "Records sampled Lunar Lake QK256/I2_S GEMV/GEMM timings with scoped worker threads applied inside the synthetic microbench.",
            "Does not claim answer quality, sustained decode throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness.",
            "Does not claim the full BitNet runtime applies this worker-thread policy outside this benchmark receipt."
        ]
    })
}

fn build_embedding_quantization_evidence(
    tensor_boundary_audit: &PathBuf,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let audit: serde_json::Value = serde_json::from_slice(&fs::read(tensor_boundary_audit)?)?;
    let selected_embedding = audit
        .pointer("/tensor_boundary/selected_embedding")
        .and_then(serde_json::Value::as_object)
        .ok_or("tensor boundary audit missing tensor_boundary.selected_embedding")?;
    let tensor_type = selected_embedding
        .get("tensor_type")
        .and_then(serde_json::Value::as_str)
        .ok_or("selected embedding missing tensor_type")?;
    let shape = selected_embedding
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("selected embedding missing shape")?;
    if shape.is_empty() {
        return Err("selected embedding shape must not be empty".into());
    }
    let current_artifact_contains_q6_k_embedding = tensor_type.eq_ignore_ascii_case("Q6_K");
    let evidence_status = if current_artifact_contains_q6_k_embedding {
        "q6_k_embedding_present_in_current_artifact"
    } else {
        "q6_k_embedding_not_present_in_current_canonical_artifact"
    };

    Ok(json!({
        "work_item": "CPU-BITNET-EMBD-001",
        "artifact_kind": "cpu_bitnet_embedding_quantization_evidence",
        "claim": "bitnet_embedding_quantization_evidence_receipt",
        "source_tensor_boundary_audit": tensor_boundary_audit.display().to_string(),
        "target_quantization": "Q6_K",
        "fallback_used": false,
        "fallback_reason": null,
        "speedup_claim": false,
        "answer_quality_claim": false,
        "acceleration_claim": false,
        "qk256_semantic_change_claim": false,
        "current_embedding": selected_embedding,
        "current_embedding_quantization": tensor_type,
        "current_artifact_contains_q6_k_embedding": current_artifact_contains_q6_k_embedding,
        "q6_k_embedding_proven": current_artifact_contains_q6_k_embedding,
        "evidence_status": evidence_status,
        "loader_scope": {
            "q6_k_tensor_type_known": true,
            "q6_k_dense_standard_dequantizer_present": true,
            "q6_k_embedding_operating_path": if current_artifact_contains_q6_k_embedding {
                "current_artifact"
            } else {
                "not_applied_to_current_bitnet_artifact"
            },
            "note": "BitNet-rs can name and dequantize GGUF Q6_K tensors for supported dense-standard adapters, but this receipt does not prove a Q6_K embedding variant for the canonical BitNet b1.58 I2_S GGUF unless the committed tensor boundary actually contains Q6_K."
        },
        "recommended_next_step": if current_artifact_contains_q6_k_embedding {
            "Add answer and phase receipts that use the Q6_K embedding artifact before claiming performance value."
        } else {
            "Acquire or generate a canonical BitNet b1.58 Q6_K embedding variant, then add embedding lookup/dequant parity and answer/phase receipts before claiming embedding-quantization support."
        },
        "claim_boundary": [
            "Records BitNet embedding tensor quantization evidence from the committed 258V tensor boundary audit.",
            "Does not claim answer quality, speedup, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness.",
            "Does not claim Q6_K embedding quantization is active unless the current canonical BitNet artifact records a Q6_K embedding tensor."
        ]
    }))
}

fn build_i2s_tiling_thread_matrix(
    runs: &[TilingMatrixRun],
    quant_format: &str,
) -> serde_json::Value {
    let measured_runs: Vec<_> = runs
        .iter()
        .map(|run| {
            let profile = &run.profile;
            json!({
                "profile": profile.profile,
                "operation": profile.operation,
                "execution_phase": profile.execution_phase,
                "status": "measured",
                "candidate": {
                    "parallelism_degree": run.candidate.parallelism_degree,
                    "row_block": run.candidate.row_block,
                    "col_block": run.candidate.col_block,
                    "thread_count": run.candidate.thread_count,
                    "thread_count_applied": false,
                    "thread_count_policy": "recorded_not_applied",
                    "thread_count_note": "Current QK256/I2_S CPU microbench records thread-count candidates; worker-thread scheduling is a later tuning step."
                },
                "requested_kernel": profile.requested_kernel,
                "selected_kernel": profile.selected_kernel,
                "fallback_used": profile.fallback_used,
                "fallback_reason": profile.fallback_reason.as_deref(),
                "shape": {
                    "rows": profile.rows,
                    "cols": profile.cols,
                    "tokens": profile.tokens,
                    "iterations": profile.iterations,
                    "cols_rounded_to_qk256_block": profile.cols != run.candidate.col_block
                },
                "wall_time_ms": profile.wall_time_ms,
                "median_ms": profile.median_ms,
                "p95_ms": profile.p95_ms,
                "bandwidth_gbps": profile.bandwidth_gbps,
                "tokens_per_second": profile.tokens_per_second,
                "speedup_claim": false
            })
        })
        .collect();
    json!({
        "work_item": "CPU-BITNET-PERF-002",
        "artifact_kind": "cpu_bitnet_i2s_tiling_thread_matrix",
        "claim": "i2_s_tiling_thread_matrix_receipt",
        "kernel_family": "i2_s_qk256",
        "quantization": quant_format,
        "speedup_claim": false,
        "fallback_used": runs.iter().any(|run| run.profile.fallback_used),
        "fallback_reason": null,
        "candidate_grid": {
            "parallelism_degrees": TILING_PARALLELISM_DEGREES,
            "row_blocks": TILING_ROW_BLOCKS,
            "col_blocks": TILING_COL_BLOCKS,
            "thread_counts": TILING_THREAD_COUNTS,
            "candidate_count": TILING_PARALLELISM_DEGREES.len()
                * TILING_ROW_BLOCKS.len()
                * TILING_COL_BLOCKS.len()
                * TILING_THREAD_COUNTS.len()
        },
        "coverage": {
            "status": "sampled_baseline",
            "measured_candidate_count": measured_runs.len(),
            "full_matrix_candidate_count": TILING_PARALLELISM_DEGREES.len()
                * TILING_ROW_BLOCKS.len()
                * TILING_COL_BLOCKS.len()
                * TILING_THREAD_COUNTS.len(),
            "thread_counts_recorded_not_applied": true,
            "reason": "This receipt captures the Lunar Lake tiling/thread search surface and sampled QK256/I2_S timings without upgrading any profile to a speedup claim."
        },
        "measured_runs": measured_runs,
        "claim_boundary": [
            "Records a Lunar Lake QK256/I2_S tiling/thread candidate matrix and sampled GEMV/GEMM timings.",
            "Does not claim answer quality, sustained decode throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness."
        ]
    })
}

fn measure_i2s_tiling_thread_matrix(
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<Vec<TilingMatrixRun>, Box<dyn Error>> {
    let candidates = [
        TilingCandidate { parallelism_degree: 2, row_block: 2, col_block: 64, thread_count: 1 },
        TilingCandidate { parallelism_degree: 4, row_block: 4, col_block: 128, thread_count: 2 },
        TilingCandidate { parallelism_degree: 8, row_block: 8, col_block: 256, thread_count: 4 },
        TilingCandidate { parallelism_degree: 8, row_block: 16, col_block: 512, thread_count: 8 },
    ];
    let mut runs = Vec::with_capacity(candidates.len() * 2);
    for candidate in candidates {
        runs.push(TilingMatrixRun {
            candidate,
            profile: measure_gemv_microbench(
                "i2s_qk256_tiling_matrix_gemv",
                "decode_gemv_tiling_sample",
                candidate.parallelism_degree * candidate.row_block,
                candidate.col_block.max(QK256_BLOCK),
                8,
                requested_kernel,
                strict,
            )?,
        });
        runs.push(TilingMatrixRun {
            candidate,
            profile: measure_gemm_microbench(
                "i2s_qk256_tiling_matrix_gemm",
                "prefill_gemm_tiling_sample",
                candidate.parallelism_degree,
                candidate.parallelism_degree * candidate.row_block,
                candidate.col_block.max(QK256_BLOCK),
                4,
            )?,
        });
    }
    Ok(runs)
}

fn measure_i2s_applied_thread_matrix(
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<Vec<AppliedThreadMatrixRun>, Box<dyn Error>> {
    let candidates = [
        TilingCandidate { parallelism_degree: 2, row_block: 2, col_block: 64, thread_count: 1 },
        TilingCandidate { parallelism_degree: 4, row_block: 4, col_block: 128, thread_count: 2 },
        TilingCandidate { parallelism_degree: 8, row_block: 8, col_block: 256, thread_count: 4 },
        TilingCandidate { parallelism_degree: 8, row_block: 16, col_block: 512, thread_count: 8 },
    ];
    let mut runs = Vec::with_capacity(candidates.len() * 2);
    for candidate in candidates {
        let rows = candidate.parallelism_degree * candidate.row_block;
        let cols = candidate.col_block.max(QK256_BLOCK);
        let (profile, applied_thread_count) = measure_threaded_gemv_microbench(
            "i2s_qk256_applied_thread_matrix_gemv",
            "decode_gemv_applied_thread_sample",
            rows,
            cols,
            8,
            requested_kernel,
            strict,
            candidate.thread_count,
        )?;
        runs.push(AppliedThreadMatrixRun {
            candidate,
            profile,
            applied_thread_count,
            thread_partition: "rows",
        });

        let tokens = candidate.parallelism_degree;
        let rows = candidate.parallelism_degree * candidate.row_block;
        let cols = candidate.col_block.max(QK256_BLOCK);
        let (profile, applied_thread_count) = measure_threaded_gemm_microbench(
            "i2s_qk256_applied_thread_matrix_gemm",
            "prefill_gemm_applied_thread_sample",
            tokens,
            rows,
            cols,
            4,
            candidate.thread_count,
        )?;
        runs.push(AppliedThreadMatrixRun {
            candidate,
            profile,
            applied_thread_count,
            thread_partition: "tokens",
        });
    }
    Ok(runs)
}

fn measure_i2s_microbench(
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<Vec<KernelMicrobenchProfile>, Box<dyn Error>> {
    Ok(vec![
        measure_gemv_microbench(
            "i2s_qk256_gemv_decode_microbench",
            "decode_gemv_micro_kernel",
            64,
            1024,
            64,
            requested_kernel,
            strict,
        )?,
        measure_gemm_microbench(
            "i2s_qk256_gemm_prefill_microbench",
            "prefill_gemm_micro_kernel",
            16,
            64,
            1024,
            16,
        )?,
    ])
}

#[allow(clippy::too_many_arguments)]
fn measure_threaded_gemv_microbench(
    profile: &'static str,
    execution_phase: &'static str,
    rows: usize,
    cols: usize,
    iterations: u64,
    requested_kernel: Option<&'static str>,
    strict: bool,
    thread_count: usize,
) -> Result<(KernelMicrobenchProfile, usize), Box<dyn Error>> {
    let (packed, row_stride) = create_qk256_weights(rows, cols);
    let activations = create_activation_vector(cols);
    let mut output = vec![0.0f32; rows];
    let (selection, applied_thread_count) = run_threaded_gemv_qk256(
        &packed,
        &activations,
        &mut output,
        rows,
        cols,
        row_stride,
        requested_kernel,
        strict,
        thread_count,
    )?;

    let mut samples = Vec::with_capacity(iterations as usize);
    let wall_start = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        run_threaded_gemv_qk256(
            &packed,
            &activations,
            &mut output,
            rows,
            cols,
            row_stride,
            requested_kernel,
            strict,
            thread_count,
        )?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let wall_time_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    samples.sort_by(f64::total_cmp);

    let bytes_per_iteration = packed.len()
        + activations.len() * std::mem::size_of::<f32>()
        + output.len() * std::mem::size_of::<f32>();
    let total_seconds = (wall_time_ms / 1_000.0).max(f64::EPSILON);

    Ok((
        KernelMicrobenchProfile {
            profile,
            operation: "gemv",
            execution_phase,
            requested_kernel: selection.requested_kernel.unwrap_or("auto"),
            selected_kernel: selection.selected_kernel,
            fallback_used: selection.fallback_used,
            fallback_reason: selection.fallback_reason,
            rows,
            cols,
            tokens: 1,
            iterations,
            wall_time_ms,
            median_ms: percentile(&samples, 0.50),
            p95_ms: percentile(&samples, 0.95),
            bandwidth_gbps: (bytes_per_iteration as f64 * iterations as f64)
                / total_seconds
                / 1_000_000_000.0,
            tokens_per_second: iterations as f64 / total_seconds,
        },
        applied_thread_count,
    ))
}

#[allow(clippy::too_many_arguments)]
fn measure_threaded_gemm_microbench(
    profile: &'static str,
    execution_phase: &'static str,
    tokens: usize,
    rows: usize,
    cols: usize,
    iterations: u64,
    thread_count: usize,
) -> Result<(KernelMicrobenchProfile, usize), Box<dyn Error>> {
    let (packed, _row_stride) = create_qk256_weights(rows, cols);
    let activations = create_activation_matrix(tokens, cols);
    let mut output = vec![0.0f32; tokens * rows];
    let applied_thread_count = run_threaded_gemm_qk256(
        &packed,
        &activations,
        &mut output,
        tokens,
        rows,
        cols,
        thread_count,
    )?;

    let mut samples = Vec::with_capacity(iterations as usize);
    let wall_start = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        run_threaded_gemm_qk256(
            &packed,
            &activations,
            &mut output,
            tokens,
            rows,
            cols,
            thread_count,
        )?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let wall_time_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    samples.sort_by(f64::total_cmp);

    let bytes_per_iteration = packed.len()
        + activations.len() * std::mem::size_of::<f32>()
        + output.len() * std::mem::size_of::<f32>();
    let total_seconds = (wall_time_ms / 1_000.0).max(f64::EPSILON);

    Ok((
        KernelMicrobenchProfile {
            profile,
            operation: "gemm",
            execution_phase,
            requested_kernel: QK256_SCALAR_GEMM_KERNEL_ID,
            selected_kernel: QK256_SCALAR_GEMM_KERNEL_ID,
            fallback_used: false,
            fallback_reason: None,
            rows,
            cols,
            tokens,
            iterations,
            wall_time_ms,
            median_ms: percentile(&samples, 0.50),
            p95_ms: percentile(&samples, 0.95),
            bandwidth_gbps: (bytes_per_iteration as f64 * iterations as f64)
                / total_seconds
                / 1_000_000_000.0,
            tokens_per_second: (tokens as f64 * iterations as f64) / total_seconds,
        },
        applied_thread_count,
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_threaded_gemv_qk256(
    packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride: usize,
    requested_kernel: Option<&'static str>,
    strict: bool,
    thread_count: usize,
) -> Result<(Qk256KernelSelection, usize), Box<dyn Error>> {
    let applied_thread_count = applied_thread_count(rows, thread_count);
    let rows_per_thread = rows.div_ceil(applied_thread_count);
    let mut selections = Vec::with_capacity(applied_thread_count);
    let scope_result: Result<(), String> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, output_chunk) in output.chunks_mut(rows_per_thread).enumerate() {
            let row_start = chunk_index * rows_per_thread;
            let row_count = output_chunk.len();
            if row_count == 0 {
                continue;
            }
            let byte_start = row_start * row_stride;
            let byte_end = byte_start + row_count * row_stride;
            let packed_chunk = &packed[byte_start..byte_end];
            handles.push(scope.spawn(move || {
                gemv_qk256_with_kernel_selection(
                    packed_chunk,
                    activations,
                    output_chunk,
                    row_count,
                    cols,
                    row_stride,
                    requested_kernel,
                    strict,
                )
                .map_err(|err| err.to_string())
            }));
        }

        for handle in handles {
            let selection =
                handle.join().map_err(|_| "GEMV worker thread panicked".to_string())??;
            selections.push(selection);
        }
        Ok(())
    });
    scope_result.map_err(std::io::Error::other)?;

    let selection =
        selections.into_iter().next().ok_or("threaded GEMV produced no worker selection")?;
    Ok((selection, applied_thread_count))
}

fn run_threaded_gemm_qk256(
    packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    tokens: usize,
    rows: usize,
    cols: usize,
    thread_count: usize,
) -> Result<usize, Box<dyn Error>> {
    let applied_thread_count = applied_thread_count(tokens, thread_count);
    let tokens_per_thread = tokens.div_ceil(applied_thread_count);
    let scope_result: Result<(), String> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (activation_chunk, output_chunk) in activations
            .chunks(tokens_per_thread * cols)
            .zip(output.chunks_mut(tokens_per_thread * rows))
        {
            let token_count = activation_chunk.len() / cols;
            if token_count == 0 {
                continue;
            }
            handles.push(scope.spawn(move || {
                qk256_gemm_scalar(packed, activation_chunk, output_chunk, token_count, rows, cols)
                    .map_err(|err| err.to_string())
            }));
        }

        for handle in handles {
            handle.join().map_err(|_| "GEMM worker thread panicked".to_string())??;
        }
        Ok(())
    });
    scope_result.map_err(std::io::Error::other)?;
    Ok(applied_thread_count)
}

fn applied_thread_count(work_items: usize, requested_threads: usize) -> usize {
    requested_threads.max(1).min(work_items.max(1))
}

fn measure_gemv_microbench(
    profile: &'static str,
    execution_phase: &'static str,
    rows: usize,
    cols: usize,
    iterations: u64,
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<KernelMicrobenchProfile, Box<dyn Error>> {
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

    Ok(KernelMicrobenchProfile {
        profile,
        operation: "gemv",
        execution_phase,
        requested_kernel: selection.requested_kernel.unwrap_or("auto"),
        selected_kernel: selection.selected_kernel,
        fallback_used: selection.fallback_used,
        fallback_reason: selection.fallback_reason,
        rows,
        cols,
        tokens: 1,
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

fn measure_gemm_microbench(
    profile: &'static str,
    execution_phase: &'static str,
    tokens: usize,
    rows: usize,
    cols: usize,
    iterations: u64,
) -> Result<KernelMicrobenchProfile, Box<dyn Error>> {
    let (packed, _row_stride) = create_qk256_weights(rows, cols);
    let activations = create_activation_matrix(tokens, cols);
    let mut output = vec![0.0f32; tokens * rows];
    qk256_gemm_scalar(&packed, &activations, &mut output, tokens, rows, cols)?;

    let mut samples = Vec::with_capacity(iterations as usize);
    let wall_start = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        qk256_gemm_scalar(&packed, &activations, &mut output, tokens, rows, cols)?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let wall_time_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    samples.sort_by(f64::total_cmp);

    let bytes_per_iteration = packed.len()
        + activations.len() * std::mem::size_of::<f32>()
        + output.len() * std::mem::size_of::<f32>();
    let total_seconds = (wall_time_ms / 1_000.0).max(f64::EPSILON);

    Ok(KernelMicrobenchProfile {
        profile,
        operation: "gemm",
        execution_phase,
        requested_kernel: QK256_SCALAR_GEMM_KERNEL_ID,
        selected_kernel: QK256_SCALAR_GEMM_KERNEL_ID,
        fallback_used: false,
        fallback_reason: None,
        rows,
        cols,
        tokens,
        iterations,
        wall_time_ms,
        median_ms: percentile(&samples, 0.50),
        p95_ms: percentile(&samples, 0.95),
        bandwidth_gbps: (bytes_per_iteration as f64 * iterations as f64)
            / total_seconds
            / 1_000_000_000.0,
        tokens_per_second: (tokens as f64 * iterations as f64) / total_seconds,
    })
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

fn create_activation_matrix(tokens: usize, cols: usize) -> Vec<f32> {
    (0..tokens)
        .flat_map(|token| {
            create_activation_vector(cols)
                .into_iter()
                .enumerate()
                .map(move |(col, value)| value + ((token + col) % 17) as f32 * 0.0001)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_records_i2s_gemv_and_gemm_microbench_profiles() -> Result<(), Box<dyn Error>> {
        let receipt = build_receipt(&Args::default())?;
        let profiles = receipt["i2s_microbench"]["profiles"].as_array().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing microbench profiles")
        })?;

        assert!(profiles.iter().any(|profile| {
            profile["operation"] == "gemv"
                && profile["selected_kernel"].as_str().is_some_and(|kernel| {
                    kernel == QK256_SCALAR_GEMV_KERNEL_ID || kernel == "qk256-avx2-gemv"
                })
        }));
        assert!(profiles.iter().any(|profile| {
            profile["operation"] == "gemm"
                && profile["selected_kernel"] == QK256_SCALAR_GEMM_KERNEL_ID
        }));
        assert_eq!(receipt["i2s_microbench"]["speedup_claim"], false);
        assert_eq!(receipt["i2s_microbench"]["fallback_used"], false);
        Ok(())
    }

    #[test]
    fn receipt_records_i2s_tiling_thread_matrix_when_requested() -> Result<(), Box<dyn Error>> {
        let receipt = build_receipt(&Args { include_i2s_tiling_matrix: true, ..Args::default() })?;
        let matrix = receipt["i2s_tiling_thread_matrix"].as_object().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing tiling matrix")
        })?;
        let measured_runs = matrix["measured_runs"].as_array().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing tiling measured runs")
        })?;

        assert_eq!(matrix["work_item"], "CPU-BITNET-PERF-002");
        assert_eq!(matrix["speedup_claim"], false);
        assert_eq!(matrix["fallback_used"], false);
        assert!(measured_runs.iter().any(|run| run["operation"] == "gemv"));
        assert!(measured_runs.iter().any(|run| run["operation"] == "gemm"));
        Ok(())
    }

    #[test]
    fn receipt_records_i2s_applied_thread_matrix_when_requested() -> Result<(), Box<dyn Error>> {
        let receipt =
            build_receipt(&Args { include_i2s_applied_thread_matrix: true, ..Args::default() })?;
        let matrix = receipt["i2s_applied_thread_matrix"].as_object().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing applied thread matrix")
        })?;
        let measured_runs = matrix["measured_runs"].as_array().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing applied thread measured runs",
            )
        })?;

        assert_eq!(matrix["work_item"], "CPU-BITNET-PERF-003");
        assert_eq!(matrix["speedup_claim"], false);
        assert_eq!(matrix["fallback_used"], false);
        assert!(measured_runs.iter().all(|run| run["candidate"]["thread_count_applied"] == true));
        assert!(measured_runs.iter().any(|run| {
            run["operation"] == "gemv" && run["candidate"]["thread_partition"] == "rows"
        }));
        assert!(measured_runs.iter().any(|run| {
            run["operation"] == "gemm" && run["candidate"]["thread_partition"] == "tokens"
        }));
        Ok(())
    }
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
