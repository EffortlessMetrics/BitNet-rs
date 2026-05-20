use std::{
    env,
    error::Error,
    io,
    path::PathBuf,
    time::{Duration, Instant},
};

use bitnet_kernels::a770_opencl_fixture::{
    A770_MATMUL_I2S_ACTIVATIONS, A770_MATMUL_I2S_K, A770_MATMUL_I2S_M, A770_MATMUL_I2S_N,
    a770_matmul_i2s_cpu_reference, pack_a770_matmul_i2s_weights,
};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, ClMem};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;

const RECEIPT_ENV: &str = "BITNET_A770_OPENCL_PARITY_RECEIPT";
const TOLERANCE: f32 = 1.0e-6;

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let receipt = run_a770_matmul_i2s(args.mode)?;
    if let Some(path) = args.receipt_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, receipt.to_json())?;
    }
    println!("{}", receipt.to_json());
    Ok(())
}

#[derive(Debug)]
struct CliArgs {
    receipt_path: Option<PathBuf>,
    mode: RunMode,
}

#[derive(Debug)]
enum RunMode {
    Parity,
    Benchmark(BenchmarkConfig),
}

#[derive(Clone, Copy, Debug)]
struct BenchmarkConfig {
    iterations: usize,
    warmup_iterations: usize,
}

fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut receipt = env::var_os(RECEIPT_ENV).map(PathBuf::from);
    let mut benchmark = false;
    let mut iterations = 30usize;
    let mut warmup_iterations = 5usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => {
                let path =
                    args.next().ok_or_else(|| io_error("--receipt requires a path argument"))?;
                receipt = Some(PathBuf::from(path));
            }
            "--benchmark" => benchmark = true,
            "--iterations" => {
                let value = args.next().ok_or_else(|| io_error("--iterations requires a value"))?;
                iterations = parse_positive_usize("--iterations", &value)?;
            }
            "--warmup" => {
                let value = args.next().ok_or_else(|| io_error("--warmup requires a value"))?;
                warmup_iterations = parse_usize("--warmup", &value)?;
            }
            "--help" | "-h" => {
                println!(concat!(
                    "Usage: a770-opencl-parity [--receipt <path>] [--benchmark] ",
                    "[--iterations <N>] [--warmup <N>]\n\n",
                    "Runs selected Intel Arc A770 OpenCL matmul_i2s parity against a CPU reference.\n",
                    "With --benchmark, records a diagnostic kernel baseline receipt without speed claims."
                ));
                std::process::exit(0);
            }
            other => return Err(io_error(format!("unknown argument {other:?}"))),
        }
    }
    let mode = if benchmark {
        RunMode::Benchmark(BenchmarkConfig { iterations, warmup_iterations })
    } else {
        RunMode::Parity
    };
    Ok(CliArgs { receipt_path: receipt, mode })
}

fn parse_positive_usize(flag: &str, value: &str) -> Result<usize, Box<dyn Error>> {
    let parsed = parse_usize(flag, value)?;
    if parsed == 0 {
        return Err(io_error(format!("{flag} must be greater than zero")));
    }
    Ok(parsed)
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, Box<dyn Error>> {
    value.parse::<usize>().map_err(|err| io_error(format!("invalid {flag} value {value:?}: {err}")))
}

#[derive(Debug)]
struct A770OpenClParityReceipt {
    runtime_device: String,
    platform_index: usize,
    device_index: usize,
    platform_name: String,
    vendor: String,
    driver_version: String,
    matrix_m: usize,
    matrix_n: usize,
    matrix_k: usize,
    packed_weight_bytes: usize,
    activation_values: usize,
    max_abs_error: f32,
    mean_abs_error: f32,
    benchmark: Option<BenchmarkSummary>,
}

#[derive(Debug)]
struct BenchmarkSummary {
    iterations: usize,
    warmup_iterations: usize,
    cpu_reference_total_ms: f64,
    cpu_reference_avg_ms: f64,
    opencl_kernel_total_ms: f64,
    opencl_kernel_avg_ms: f64,
    cpu_reference_samples_ms: Vec<f64>,
    opencl_kernel_samples_ms: Vec<f64>,
    initial_host_to_device_bytes: usize,
    device_to_host_bytes: usize,
    kernel_invocations: usize,
}

impl A770OpenClParityReceipt {
    fn to_json(&self) -> String {
        let (work_item, proof_family, proof_stage) = if self.benchmark.is_some() {
            (
                "A770-008",
                "a770_opencl_matmul_i2s_benchmark_baseline",
                "diagnostic_benchmark_candidate",
            )
        } else {
            ("A770-006R", "a770_opencl_matmul_i2s_cpu_parity", "cpu_opencl_parity_tested")
        };
        let benchmark_json = self.benchmark_json();
        format!(
            concat!(
                "{{\n",
                "  \"campaign\": \"intel-a770\",\n",
                "  \"work_item\": \"{}\",\n",
                "  \"proof_family\": \"{}\",\n",
                "  \"proof_stage\": \"{}\",\n",
                "  \"requested_backend\": \"intel-arc-a770\",\n",
                "  \"selected_backend\": \"intel-arc-a770-opencl\",\n",
                "  \"runtime_api\": \"opencl\",\n",
                "  \"runtime_device\": \"{}\",\n",
                "  \"platform_index\": {},\n",
                "  \"device_index\": {},\n",
                "  \"platform_name\": \"{}\",\n",
                "  \"vendor\": \"{}\",\n",
                "  \"driver_version\": \"{}\",\n",
                "  \"kernel_source\": \"bitnet_kernels::kernels::MATMUL_I2S_SRC\",\n",
                "  \"kernel_name\": \"matmul_i2s\",\n",
                "  \"fixture_contract\": \"a770_matmul_i2s_explicit_activation_and_packed_weight_operands\",\n",
                "  \"operand_a\": \"int8_activations_row_major_m_by_k\",\n",
                "  \"operand_b\": \"packed_i2s_weights_k_by_n_four_weights_per_byte\",\n",
                "  \"matrix_m\": {},\n",
                "  \"matrix_n\": {},\n",
                "  \"matrix_k\": {},\n",
                "  \"packed_weight_bytes\": {},\n",
                "  \"activation_values\": {},\n",
                "  \"tolerance\": {},\n",
                "  \"max_abs_error\": {},\n",
                "  \"mean_abs_error\": {},\n",
                "  \"benchmark\": {},\n",
                "  \"benchmark_candidate\": {},\n",
                "  \"benchmark_profile\": {},\n",
                "  \"benchmark_claim_allowed\": false,\n",
                "  \"speedup_claim\": false,\n",
                "  \"passed\": true,\n",
                "  \"opencl_execution\": true,\n",
                "  \"cpu_reference\": true,\n",
                "  \"cpu_opencl_parity\": true,\n",
                "  \"fallback_used\": false,\n",
                "  \"cpu_fallback_allowed\": false,\n",
                "  \"bitnet_inference\": false,\n",
                "  \"qk256_decode\": false,\n",
                "  \"claim_allowed\": false,\n",
                "  \"diagnostic_only\": true,\n",
                "  \"performance_claim\": false,\n",
                "  \"full_residency_claim\": false,\n",
                "  \"model_family\": null,\n",
                "  \"must_not_claim\": [\n",
                "    \"Official BitNet QK256 production semantics are proven\",\n",
                "    \"BitNet inference works on A770\",\n",
                "    \"A770 trusted partial acceleration is claim-grade\",\n",
                "    \"Full A770 residency is proven\",\n",
                "    \"A770 performance speedup is proven\"\n",
                "  ]\n",
                "}}\n"
            ),
            work_item,
            proof_family,
            proof_stage,
            json_escape(&self.runtime_device),
            self.platform_index,
            self.device_index,
            json_escape(&self.platform_name),
            json_escape(&self.vendor),
            json_escape(&self.driver_version),
            self.matrix_m,
            self.matrix_n,
            self.matrix_k,
            self.packed_weight_bytes,
            self.activation_values,
            TOLERANCE,
            self.max_abs_error,
            self.mean_abs_error,
            benchmark_json,
            self.benchmark.is_some(),
            if self.benchmark.is_some() { "\"matmul_i2s_kernel_baseline\"" } else { "null" },
        )
    }

    fn benchmark_json(&self) -> String {
        let Some(benchmark) = &self.benchmark else {
            return "null".to_string();
        };
        format!(
            concat!(
                "{{\n",
                "    \"scope\": \"kernel_only_after_initial_upload\",\n",
                "    \"iterations\": {},\n",
                "    \"warmup_iterations\": {},\n",
                "    \"cpu_reference\": {{\n",
                "      \"implementation\": \"scalar_fixture_reference\",\n",
                "      \"total_ms\": {},\n",
                "      \"avg_ms\": {},\n",
                "      \"samples_ms\": {}\n",
                "    }},\n",
                "    \"opencl_kernel\": {{\n",
                "      \"total_ms\": {},\n",
                "      \"avg_ms\": {},\n",
                "      \"samples_ms\": {},\n",
                "      \"kernel_invocations\": {}\n",
                "    }},\n",
                "    \"transfer_bytes\": {{\n",
                "      \"initial_host_to_device\": {},\n",
                "      \"device_to_host\": {}\n",
                "    }},\n",
                "    \"cpu_avx2_applicable\": false,\n",
                "    \"profile_timing_applicable\": true,\n",
                "    \"quality_passed\": false,\n",
                "    \"quality_gate\": \"not_applicable_toy_fixture\",\n",
                "    \"performance_claim_allowed\": false\n",
                "  }}"
            ),
            benchmark.iterations,
            benchmark.warmup_iterations,
            format_f64(benchmark.cpu_reference_total_ms),
            format_f64(benchmark.cpu_reference_avg_ms),
            format_f64_array(&benchmark.cpu_reference_samples_ms),
            format_f64(benchmark.opencl_kernel_total_ms),
            format_f64(benchmark.opencl_kernel_avg_ms),
            format_f64_array(&benchmark.opencl_kernel_samples_ms),
            benchmark.kernel_invocations,
            benchmark.initial_host_to_device_bytes,
            benchmark.device_to_host_bytes
        )
    }
}

fn run_a770_matmul_i2s(mode: RunMode) -> Result<A770OpenClParityReceipt, Box<dyn Error>> {
    let selected = find_a770_device()?;
    let context = Context::from_device(&selected.device)
        .map_err(|err| io_error(format!("failed to create OpenCL context: {err}")))?;
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .map_err(|err| io_error(format!("failed to create OpenCL command queue: {err}")))?;
    let program = Program::create_and_build_from_source(
        &context,
        bitnet_kernels::kernels::MATMUL_I2S_SRC,
        "",
    )
    .map_err(|err| io_error(format!("failed to build bitnet-kernels MATMUL_I2S_SRC: {err}")))?;
    let kernel = Kernel::create(&program, "matmul_i2s")
        .map_err(|err| io_error(format!("failed to create matmul_i2s kernel: {err}")))?;

    const M: usize = 2;
    const N: usize = 3;
    const K: usize = 8;

    debug_assert_eq!(M, A770_MATMUL_I2S_M);
    debug_assert_eq!(N, A770_MATMUL_I2S_N);
    debug_assert_eq!(K, A770_MATMUL_I2S_K);

    let packed_weights = pack_a770_matmul_i2s_weights().map_err(io_error)?;
    let expected = a770_matmul_i2s_cpu_reference();
    let mut actual = vec![0.0f32; expected.len()];

    let mut buf_a = unsafe {
        Buffer::<i8>::create(
            &context,
            CL_MEM_READ_ONLY,
            A770_MATMUL_I2S_ACTIVATIONS.len(),
            std::ptr::null_mut(),
        )
        .map_err(|err| io_error(format!("failed to create input A buffer: {err}")))?
    };
    let mut buf_b = unsafe {
        Buffer::<u8>::create(&context, CL_MEM_READ_ONLY, packed_weights.len(), std::ptr::null_mut())
            .map_err(|err| io_error(format!("failed to create input B buffer: {err}")))?
    };
    let buf_out = unsafe {
        Buffer::<f32>::create(&context, CL_MEM_WRITE_ONLY, actual.len(), std::ptr::null_mut())
            .map_err(|err| io_error(format!("failed to create output buffer: {err}")))?
    };

    unsafe {
        queue
            .enqueue_write_buffer(&mut buf_a, CL_BLOCKING, 0, &A770_MATMUL_I2S_ACTIVATIONS, &[])
            .map_err(|err| io_error(format!("failed to write input A buffer: {err}")))?;
        queue
            .enqueue_write_buffer(&mut buf_b, CL_BLOCKING, 0, &packed_weights, &[])
            .map_err(|err| io_error(format!("failed to write input B buffer: {err}")))?;
    }

    let benchmark = match mode {
        RunMode::Parity => None,
        RunMode::Benchmark(config) => Some(run_benchmark(
            config,
            &queue,
            &kernel,
            &buf_a,
            &buf_b,
            &buf_out,
            &mut actual,
            packed_weights.len(),
        )?),
    };

    let matrix_m = M as u32;
    let matrix_n = N as u32;
    let matrix_k = K as u32;
    let event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buf_a.get())
            .set_arg(&buf_b.get())
            .set_arg(&buf_out.get())
            .set_arg(&matrix_m)
            .set_arg(&matrix_n)
            .set_arg(&matrix_k)
            .set_global_work_sizes(&[M, N])
            .enqueue_nd_range(&queue)
            .map_err(|err| io_error(format!("failed to enqueue matmul_i2s kernel: {err}")))?
    };
    event.wait().map_err(|err| io_error(format!("matmul_i2s kernel wait failed: {err}")))?;

    unsafe {
        queue
            .enqueue_read_buffer(&buf_out, CL_BLOCKING, 0, &mut actual, &[])
            .map_err(|err| io_error(format!("failed to read output buffer: {err}")))?;
    }

    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    for (expected_value, actual_value) in expected.iter().zip(&actual) {
        let delta = (expected_value - actual_value).abs();
        max_abs_error = max_abs_error.max(delta);
        sum_abs_error += delta;
    }
    let mean_abs_error = sum_abs_error / expected.len() as f32;
    if max_abs_error > TOLERANCE {
        return Err(io_error(format!(
            "A770 OpenCL matmul_i2s parity exceeded tolerance: {max_abs_error}"
        )));
    }

    Ok(A770OpenClParityReceipt {
        runtime_device: selected.device_name,
        platform_index: selected.platform_index,
        device_index: selected.device_index,
        platform_name: selected.platform_name,
        vendor: selected.vendor,
        driver_version: selected.driver_version,
        matrix_m: M,
        matrix_n: N,
        matrix_k: K,
        packed_weight_bytes: packed_weights.len(),
        activation_values: A770_MATMUL_I2S_ACTIVATIONS.len(),
        max_abs_error,
        mean_abs_error,
        benchmark,
    })
}

fn run_benchmark(
    config: BenchmarkConfig,
    queue: &CommandQueue,
    kernel: &Kernel,
    buf_a: &Buffer<i8>,
    buf_b: &Buffer<u8>,
    buf_out: &Buffer<f32>,
    actual: &mut [f32],
    packed_weight_bytes: usize,
) -> Result<BenchmarkSummary, Box<dyn Error>> {
    for _ in 0..config.warmup_iterations {
        enqueue_matmul_i2s(queue, kernel, buf_a, buf_b, buf_out)?;
    }

    let mut cpu_samples = Vec::with_capacity(config.iterations);
    let mut cpu_total = Duration::ZERO;
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _ = a770_matmul_i2s_cpu_reference();
        let elapsed = start.elapsed();
        cpu_total += elapsed;
        cpu_samples.push(duration_ms(elapsed));
    }

    let mut opencl_samples = Vec::with_capacity(config.iterations);
    let mut opencl_total = Duration::ZERO;
    for _ in 0..config.iterations {
        let start = Instant::now();
        enqueue_matmul_i2s(queue, kernel, buf_a, buf_b, buf_out)?;
        let elapsed = start.elapsed();
        opencl_total += elapsed;
        opencl_samples.push(duration_ms(elapsed));
    }

    unsafe {
        queue
            .enqueue_read_buffer(buf_out, CL_BLOCKING, 0, actual, &[])
            .map_err(|err| io_error(format!("failed to read benchmark output buffer: {err}")))?;
    }

    Ok(BenchmarkSummary {
        iterations: config.iterations,
        warmup_iterations: config.warmup_iterations,
        cpu_reference_total_ms: duration_ms(cpu_total),
        cpu_reference_avg_ms: duration_ms(cpu_total) / config.iterations as f64,
        opencl_kernel_total_ms: duration_ms(opencl_total),
        opencl_kernel_avg_ms: duration_ms(opencl_total) / config.iterations as f64,
        cpu_reference_samples_ms: cpu_samples,
        opencl_kernel_samples_ms: opencl_samples,
        initial_host_to_device_bytes: A770_MATMUL_I2S_ACTIVATIONS.len() + packed_weight_bytes,
        device_to_host_bytes: std::mem::size_of_val(actual),
        kernel_invocations: config.iterations,
    })
}

fn enqueue_matmul_i2s(
    queue: &CommandQueue,
    kernel: &Kernel,
    buf_a: &Buffer<i8>,
    buf_b: &Buffer<u8>,
    buf_out: &Buffer<f32>,
) -> Result<(), Box<dyn Error>> {
    let matrix_m = A770_MATMUL_I2S_M as u32;
    let matrix_n = A770_MATMUL_I2S_N as u32;
    let matrix_k = A770_MATMUL_I2S_K as u32;
    let event = unsafe {
        ExecuteKernel::new(kernel)
            .set_arg(&buf_a.get())
            .set_arg(&buf_b.get())
            .set_arg(&buf_out.get())
            .set_arg(&matrix_m)
            .set_arg(&matrix_n)
            .set_arg(&matrix_k)
            .set_global_work_sizes(&[A770_MATMUL_I2S_M, A770_MATMUL_I2S_N])
            .enqueue_nd_range(queue)
            .map_err(|err| io_error(format!("failed to enqueue matmul_i2s kernel: {err}")))?
    };
    event.wait().map_err(|err| io_error(format!("matmul_i2s kernel wait failed: {err}")))?;
    Ok(())
}

#[derive(Debug)]
struct SelectedA770Device {
    platform_index: usize,
    device_index: usize,
    platform_name: String,
    device_name: String,
    vendor: String,
    driver_version: String,
    device: Device,
}

fn find_a770_device() -> Result<SelectedA770Device, Box<dyn Error>> {
    let platforms = get_platforms()
        .map_err(|err| io_error(format!("failed to enumerate OpenCL platforms: {err}")))?;
    for (platform_index, platform) in platforms.iter().enumerate() {
        let platform_name = platform.name().unwrap_or_else(|_| "unknown".to_owned());
        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
        for (device_index, device_id) in devices.iter().enumerate() {
            let device = Device::new(*device_id);
            let device_name = device.name().unwrap_or_default();
            let vendor = device.vendor().unwrap_or_default();
            if !is_intel_vendor(&vendor) || !is_a770_device_name(&device_name) {
                continue;
            }
            let driver_version = device.driver_version().unwrap_or_default();
            return Ok(SelectedA770Device {
                platform_index,
                device_index,
                platform_name,
                device_name,
                vendor,
                driver_version,
                device,
            });
        }
    }
    Err(io_error("Intel Arc A770 OpenCL device was not visible"))
}

fn is_intel_vendor(value: &str) -> bool {
    value.to_ascii_lowercase().contains("intel")
}

fn is_a770_device_name(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("a770")) || lower.contains("56a0")
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn format_f64(value: f64) -> String {
    format!("{value:.9}")
}

fn format_f64_array(values: &[f64]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&format_f64(*value));
    }
    output.push(']');
    output
}

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_receipt(benchmark: Option<BenchmarkSummary>) -> A770OpenClParityReceipt {
        A770OpenClParityReceipt {
            runtime_device: "Intel(R) Arc(TM) A770 Graphics".to_string(),
            platform_index: 0,
            device_index: 1,
            platform_name: "Intel OpenCL".to_string(),
            vendor: "Intel(R) Corporation".to_string(),
            driver_version: "test".to_string(),
            matrix_m: A770_MATMUL_I2S_M,
            matrix_n: A770_MATMUL_I2S_N,
            matrix_k: A770_MATMUL_I2S_K,
            packed_weight_bytes: 6,
            activation_values: A770_MATMUL_I2S_ACTIVATIONS.len(),
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            benchmark,
        }
    }

    #[test]
    fn benchmark_receipt_stays_claim_closed() {
        let receipt = sample_receipt(Some(BenchmarkSummary {
            iterations: 2,
            warmup_iterations: 1,
            cpu_reference_total_ms: 0.2,
            cpu_reference_avg_ms: 0.1,
            opencl_kernel_total_ms: 0.4,
            opencl_kernel_avg_ms: 0.2,
            cpu_reference_samples_ms: vec![0.09, 0.11],
            opencl_kernel_samples_ms: vec![0.19, 0.21],
            initial_host_to_device_bytes: 22,
            device_to_host_bytes: 24,
            kernel_invocations: 2,
        }));
        let json = receipt.to_json();

        assert!(json.contains("\"work_item\": \"A770-008\""));
        assert!(json.contains("\"benchmark_candidate\": true"));
        assert!(json.contains("\"benchmark_claim_allowed\": false"));
        assert!(json.contains("\"speedup_claim\": false"));
        assert!(json.contains("\"claim_allowed\": false"));
        assert!(json.contains("\"quality_gate\": \"not_applicable_toy_fixture\""));
        assert!(json.contains("\"Official BitNet QK256 production semantics are proven\""));
    }

    #[test]
    fn parity_receipt_keeps_existing_work_item() {
        let json = sample_receipt(None).to_json();

        assert!(json.contains("\"work_item\": \"A770-006R\""));
        assert!(json.contains("\"benchmark\": null"));
        assert!(json.contains("\"benchmark_candidate\": false"));
    }

    #[test]
    fn benchmark_iterations_must_be_positive() {
        let err = parse_positive_usize("--iterations", "0").expect_err("zero rejected");
        assert!(err.to_string().contains("--iterations must be greater than zero"));
    }
}
