//! RTX 5070 Ti CUDA benchmark receipt harness.
//!
//! This bench target is intentionally narrow: it benchmarks the parity-tested
//! tiny CUDA vector-add fixture and records unsupported follow-up profiles as
//! not-run instead of fabricating benchmark measurements.

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, bail};
use bitnet_device_probe::probe_nvidia_cuda;
use bitnet_kernels::gpu::{
    CUDA_TINY_VECTOR_ADD_FIXTURE_ID, CUDA_TINY_VECTOR_ADD_INPUT_LEN,
    CUDA_TINY_VECTOR_ADD_KERNEL_ID, CUDA_TINY_VECTOR_ADD_PARITY_TOLERANCE, CudaKernel,
    CudaKernelInvocationStats, compare_cuda_tiny_vector_add_outputs, cuda_tiny_vector_add_inputs,
    expected_cuda_tiny_vector_add,
};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::{driver::CudaContext, nvrtc::compile_ptx};
use serde_json::json;
use std::fs;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const RUN_ENV: &str = "BITNET_RUN_RTX5070TI_CUDA_BENCHMARK";
const RECEIPT_ENV: &str = "BITNET_RTX5070TI_CUDA_BENCHMARK_RECEIPT";
const ARTIFACT_PATH_ENV: &str = "BITNET_RTX5070TI_CUDA_BENCHMARK_ARTIFACT_PATH";
const TIMESTAMP_ENV: &str = "BITNET_RTX5070TI_CUDA_BENCHMARK_TIMESTAMP_UTC";
const ITERATIONS_ENV: &str = "BITNET_RTX5070TI_CUDA_BENCHMARK_ITERATIONS";
const DEVICE_INDEX_ENV: &str = "BITNET_RTX5070TI_CUDA_DEVICE_INDEX";

const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
const HARDWARE_LANE: &str = "nvidia_rtx_5070_ti_cuda";
const REQUESTED_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";
const SELECTED_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";
const REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";
const RUNTIME_API: &str = "cuda";
const CLAIM: &str = "cuda_benchmark_baseline";
const PROFILE: &str = "cuda_tiny_smoke";

const CUDA_TINY_VECTOR_ADD_SRC: &str = r#"
extern "C" __global__
void cuda_tiny_vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

fn main() -> Result<()> {
    if !env_flag(RUN_ENV) {
        println!("skipping RTX5070TI-008 benchmark; set {RUN_ENV}=1 to run it");
        return Ok(());
    }

    let device_index = env_usize(DEVICE_INDEX_ENV, 0)?;
    let iterations = env_usize(ITERATIONS_ENV, 10)?;
    if iterations == 0 {
        bail!("{ITERATIONS_ENV} must be positive");
    }

    let artifact_path =
        std::env::var(ARTIFACT_PATH_ENV).or_else(|_| std::env::var(RECEIPT_ENV)).unwrap_or_else(
            |_| "ci/hardware/windows-9950x3d-rtx5070ti/<date>/cuda-benchmark.json".to_string(),
        );
    let timestamp_utc =
        std::env::var(TIMESTAMP_ENV).unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());

    let receipt = run_benchmark(device_index, iterations, timestamp_utc, artifact_path)?;

    if let Ok(path) = std::env::var(RECEIPT_ENV) {
        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create benchmark receipt directory {parent:?}"))?;
        }
        fs::write(&path, serde_json::to_string_pretty(&receipt)?)
            .with_context(|| format!("write benchmark receipt to {path}"))?;
    }

    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn run_benchmark(
    device_index: usize,
    iterations: usize,
    timestamp_utc: String,
    artifact_path: String,
) -> Result<serde_json::Value> {
    let (a, b) = cuda_tiny_vector_add_inputs();
    let expected = measure_cpu_reference(&a, &b, iterations)?;
    let cuda = measure_cuda_vector_add(device_index, &a, &b, iterations)?;
    let comparison = compare_cuda_tiny_vector_add_outputs(
        &expected.output,
        &cuda.output,
        CUDA_TINY_VECTOR_ADD_PARITY_TOLERANCE,
    )?;

    if !comparison.passed {
        bail!(
            "CUDA benchmark fixture failed parity: max_abs_error={} mean_abs_error={}",
            comparison.max_abs_error,
            comparison.mean_abs_error
        );
    }

    if !is_rtx5070ti_device_name(&cuda.device_info.name) {
        bail!(
            "RTX5070TI-008 requires an NVIDIA GeForce RTX 5070 Ti CUDA device; found '{}'",
            cuda.device_info.name
        );
    }

    let compute_capability = format!(
        "{}.{}",
        cuda.device_info.compute_capability.0, cuda.device_info.compute_capability.1
    );
    if compute_capability != "12.0" {
        bail!(
            "RTX5070TI-008 expects compute capability 12.0 for RTX 5070 Ti; found {compute_capability}"
        );
    }

    let probe = probe_nvidia_cuda(Some(device_index));
    let kernel_stats = CudaKernelInvocationStats {
        kernel_id: CUDA_TINY_VECTOR_ADD_KERNEL_ID.to_string(),
        invocations: cuda.iterations as u64,
        fallback_invocations: 0,
        host_to_device_bytes: cuda.host_to_device_bytes,
        device_to_host_bytes: cuda.device_to_host_bytes,
        kernel_launches: cuda.iterations as u64,
        kernel_time_ms: Some(cuda.timing.cuda_kernel_ms),
        selected_device_index: cuda.device_info.device_id,
        selected_device_name: cuda.device_info.name.clone(),
        compute_capability: compute_capability.clone(),
    };

    let speedup_vs_cpu = if cuda.timing.cuda_total_ms > 0.0 {
        expected.cpu_reference_ms / cuda.timing.cuda_total_ms
    } else {
        0.0
    };

    Ok(json!({
        "schema": 1,
        "artifact_kind": "cuda_benchmark",
        "machine_id": MACHINE_ID,
        "hardware_lane": HARDWARE_LANE,
        "timestamp_utc": timestamp_utc,
        "requested_backend": REQUESTED_BACKEND,
        "selected_backend": SELECTED_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "runtime_api": RUNTIME_API,
        "claim": CLAIM,
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "cuda": {
            "available": probe.available,
            "device_count": probe.device_count,
            "selected_device_index": cuda.device_info.device_id,
            "selected_device_name": cuda.device_info.name,
            "compute_capability": compute_capability,
            "driver_version": probe.driver_version,
            "cuda_runtime_version": probe.cuda_runtime_version,
            "cuda_toolkit_version": probe.cuda_toolkit_version,
            "nvrtc_version": probe.nvrtc_version,
            "nvml_available": probe.nvml_available,
            "vram_bytes": cuda.device_info.total_memory,
            "power_limit_watts": probe.power_limit_watts,
            "power_draw_watts": probe.power_draw_watts,
            "temperature_c": probe.temperature_c
        },
        "machine": {
            "cpu": "AMD Ryzen 9 9950X3D",
            "gpu": cuda.device_info.name,
            "driver_version": probe.driver_version,
            "cuda_version": probe.cuda_runtime_version,
            "compute_capability": compute_capability,
            "vram_bytes": cuda.device_info.total_memory,
            "temperature_c": probe.temperature_c,
            "power_draw_watts": probe.power_draw_watts
        },
        "benchmark": {
            "profile": PROFILE,
            "kernel_id": CUDA_TINY_VECTOR_ADD_KERNEL_ID,
            "fixture_id": CUDA_TINY_VECTOR_ADD_FIXTURE_ID,
            "input_len": CUDA_TINY_VECTOR_ADD_INPUT_LEN,
            "iterations": iterations,
            "cold_warm": {
                "compile_ms": duration_ms(cuda.compile),
                "first_iteration_total_ms": duration_ms(cuda.first_iteration_total),
                "warm_iterations": iterations
            },
            "cpu_reference_backend": REFERENCE_BACKEND,
            "cuda_backend": SELECTED_BACKEND,
            "cpu_reference_ms": expected.cpu_reference_ms,
            "cuda_total_ms": cuda.timing.cuda_total_ms,
            "cuda_kernel_ms": cuda.timing.cuda_kernel_ms,
            "host_to_device_ms": cuda.timing.host_to_device_ms,
            "device_to_host_ms": cuda.timing.device_to_host_ms,
            "allocation_ms": cuda.timing.allocation_ms,
            "speedup_vs_cpu": speedup_vs_cpu,
            "max_abs_error": comparison.max_abs_error,
            "mean_abs_error": comparison.mean_abs_error,
            "passed": comparison.passed
        },
        "profiles": [
            {
                "profile": "cuda_tiny_smoke",
                "status": "measured",
                "kernel_id": CUDA_TINY_VECTOR_ADD_KERNEL_ID,
                "cuda_total_ms": cuda.timing.cuda_total_ms
            },
            {
                "profile": "cuda_transfer_h2d_d2h",
                "status": "measured",
                "source_profile": "cuda_tiny_smoke",
                "host_to_device_ms": cuda.timing.host_to_device_ms,
                "device_to_host_ms": cuda.timing.device_to_host_ms
            },
            {
                "profile": "cuda_fp32_matmul_small",
                "status": "not_run",
                "reason": "pending parity-tested CUDA benchmark primitive"
            },
            {
                "profile": "cuda_i2s_matmul_small",
                "status": "not_run",
                "reason": "pending parity-tested CUDA benchmark primitive"
            },
            {
                "profile": "cuda_i2s_matmul_medium",
                "status": "not_run",
                "reason": "pending parity-tested CUDA benchmark primitive"
            }
        ],
        "kernel_stats": [kernel_stats_json(&kernel_stats)],
        "artifact_path": artifact_path
    }))
}

fn measure_cpu_reference(a: &[f32], b: &[f32], iterations: usize) -> Result<CpuReference> {
    let mut output = Vec::new();
    let start = Instant::now();
    for _ in 0..iterations {
        output = expected_cuda_tiny_vector_add(a, b)?;
    }
    Ok(CpuReference { output, cpu_reference_ms: duration_ms(start.elapsed()) / iterations as f64 })
}

fn measure_cuda_vector_add(
    device_index: usize,
    a: &[f32],
    b: &[f32],
    iterations: usize,
) -> Result<CudaBenchmark> {
    const THREADS_PER_BLOCK: u32 = 256;

    let ctx = CudaContext::new(device_index)
        .with_context(|| format!("create CUDA context for device {device_index}"))?;
    let stream = ctx.default_stream();

    let compile_start = Instant::now();
    let ptx = compile_ptx(CUDA_TINY_VECTOR_ADD_SRC).context("compile tiny vector-add PTX")?;
    let module = ctx.load_module(ptx).context("load tiny vector-add module")?;
    let function = module
        .load_function(CUDA_TINY_VECTOR_ADD_KERNEL_ID)
        .context("load tiny vector-add function")?;
    let compile = compile_start.elapsed();

    let first = run_cuda_iteration(&stream, &function, a, b, THREADS_PER_BLOCK)?;

    let mut accumulator = IterationAccumulator::default();
    let mut output = first.output.clone();
    for _ in 0..iterations {
        let iteration = run_cuda_iteration(&stream, &function, a, b, THREADS_PER_BLOCK)?;
        accumulator.add(&iteration);
        output = iteration.output;
    }

    let device_info = CudaKernel::get_device_info(device_index)?;
    let per_iteration_h2d_bytes = ((a.len() + b.len()) * size_of::<f32>()) as u64;
    let per_iteration_d2h_bytes = (a.len() * size_of::<f32>()) as u64;

    Ok(CudaBenchmark {
        device_info,
        output,
        compile,
        first_iteration_total: first.total,
        timing: accumulator.average(iterations),
        iterations,
        host_to_device_bytes: per_iteration_h2d_bytes * iterations as u64,
        device_to_host_bytes: per_iteration_d2h_bytes * iterations as u64,
    })
}

fn run_cuda_iteration(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    a: &[f32],
    b: &[f32],
    threads_per_block: u32,
) -> Result<CudaIteration> {
    if a.len() != b.len() {
        bail!("tiny vector-add input lengths differ: lhs={} rhs={}", a.len(), b.len());
    }
    if a.len() > i32::MAX as usize {
        bail!("tiny vector-add input is too large: len={}", a.len());
    }

    let total_start = Instant::now();

    let h2d_start = Instant::now();
    let a_dev = stream.memcpy_stod(a).context("copy input A to CUDA device")?;
    let b_dev = stream.memcpy_stod(b).context("copy input B to CUDA device")?;
    let host_to_device = h2d_start.elapsed();

    let allocation_start = Instant::now();
    let mut c_dev: CudaSlice<f32> = stream.alloc_zeros(a.len()).context("allocate CUDA output")?;
    let allocation = allocation_start.elapsed();

    let grid = (a.len() as u32).div_ceil(threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(function);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut c_dev);
    let n_arg = a.len() as i32;
    builder.arg(&n_arg);

    let kernel_start = Instant::now();
    unsafe { builder.launch(cfg) }.context("launch tiny vector-add CUDA kernel")?;
    stream.synchronize().context("synchronize tiny vector-add CUDA kernel")?;
    let kernel = kernel_start.elapsed();

    let d2h_start = Instant::now();
    let output = stream.memcpy_dtov(&c_dev).context("copy CUDA output to host")?;
    let device_to_host = d2h_start.elapsed();

    Ok(CudaIteration {
        output,
        total: total_start.elapsed(),
        host_to_device,
        device_to_host,
        kernel,
        allocation,
    })
}

#[derive(Debug)]
struct CpuReference {
    output: Vec<f32>,
    cpu_reference_ms: f64,
}

#[derive(Debug)]
struct CudaBenchmark {
    device_info: bitnet_kernels::gpu::CudaDeviceInfo,
    output: Vec<f32>,
    compile: Duration,
    first_iteration_total: Duration,
    timing: CudaTiming,
    iterations: usize,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
}

#[derive(Debug)]
struct CudaIteration {
    output: Vec<f32>,
    total: Duration,
    host_to_device: Duration,
    device_to_host: Duration,
    kernel: Duration,
    allocation: Duration,
}

#[derive(Debug, Default)]
struct IterationAccumulator {
    total: Duration,
    host_to_device: Duration,
    device_to_host: Duration,
    kernel: Duration,
    allocation: Duration,
}

impl IterationAccumulator {
    fn add(&mut self, iteration: &CudaIteration) {
        self.total += iteration.total;
        self.host_to_device += iteration.host_to_device;
        self.device_to_host += iteration.device_to_host;
        self.kernel += iteration.kernel;
        self.allocation += iteration.allocation;
    }

    fn average(&self, iterations: usize) -> CudaTiming {
        CudaTiming {
            cuda_total_ms: duration_ms(self.total) / iterations as f64,
            cuda_kernel_ms: duration_ms(self.kernel) / iterations as f64,
            host_to_device_ms: duration_ms(self.host_to_device) / iterations as f64,
            device_to_host_ms: duration_ms(self.device_to_host) / iterations as f64,
            allocation_ms: duration_ms(self.allocation) / iterations as f64,
        }
    }
}

#[derive(Debug)]
struct CudaTiming {
    cuda_total_ms: f64,
    cuda_kernel_ms: f64,
    host_to_device_ms: f64,
    device_to_host_ms: f64,
    allocation_ms: f64,
}

fn kernel_stats_json(stats: &CudaKernelInvocationStats) -> serde_json::Value {
    json!({
        "kernel_id": stats.kernel_id,
        "invocations": stats.invocations,
        "fallback_invocations": stats.fallback_invocations,
        "host_to_device_bytes": stats.host_to_device_bytes,
        "device_to_host_bytes": stats.device_to_host_bytes,
        "kernel_launches": stats.kernel_launches,
        "kernel_time_ms": stats.kernel_time_ms,
        "selected_device_index": stats.selected_device_index,
        "selected_device_name": stats.selected_device_name,
        "compute_capability": stats.compute_capability
    })
}

fn is_rtx5070ti_device_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.contains("nvidia")
        && normalized.contains("geforce")
        && normalized.contains("rtx")
        && normalized.contains("5070")
        && normalized.contains("ti")
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => {
            value.parse::<usize>().with_context(|| format!("{name} must be a positive integer"))
        }
        Err(_) => Ok(default),
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
