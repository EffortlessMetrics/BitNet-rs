use std::{env, error::Error, io, path::PathBuf};

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
    let receipt_path = receipt_path_from_args()?;
    let receipt = run_a770_matmul_i2s_parity()?;
    if let Some(path) = receipt_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, receipt.to_json())?;
    }
    println!("{}", receipt.to_json());
    Ok(())
}

fn receipt_path_from_args() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut receipt = env::var_os(RECEIPT_ENV).map(PathBuf::from);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => {
                let path =
                    args.next().ok_or_else(|| io_error("--receipt requires a path argument"))?;
                receipt = Some(PathBuf::from(path));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: a770-opencl-parity [--receipt <path>]\n\nRuns selected Intel Arc A770 OpenCL matmul_i2s parity against a CPU reference."
                );
                std::process::exit(0);
            }
            other => return Err(io_error(format!("unknown argument {other:?}"))),
        }
    }
    Ok(receipt)
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
}

impl A770OpenClParityReceipt {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"campaign\": \"intel-a770\",\n",
                "  \"work_item\": \"A770-006\",\n",
                "  \"proof_family\": \"a770_opencl_matmul_i2s_cpu_parity\",\n",
                "  \"proof_stage\": \"cpu_opencl_parity_tested\",\n",
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
                "  \"matrix_m\": {},\n",
                "  \"matrix_n\": {},\n",
                "  \"matrix_k\": {},\n",
                "  \"packed_weight_bytes\": {},\n",
                "  \"activation_values\": {},\n",
                "  \"tolerance\": {},\n",
                "  \"max_abs_error\": {},\n",
                "  \"mean_abs_error\": {},\n",
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
            self.mean_abs_error
        )
    }
}

fn run_a770_matmul_i2s_parity() -> Result<A770OpenClParityReceipt, Box<dyn Error>> {
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

    let weights = [
        1i8, 0, -1, 1, 0, -1, 1, 0, //
        -1, 1, 0, 0, 1, -1, 0, 1,
    ];
    let activations = [
        1u8, 2, 3, //
        4, 5, 6, //
        7, 8, 9, //
        2, 4, 6, //
        3, 6, 9, //
        5, 7, 11, //
        13, 17, 19, //
        23, 29, 31,
    ];
    let packed_weights = pack_i2s_weights(&weights)?;
    let expected = cpu_matmul_i2s_reference(&weights, &activations, M, N, K);
    let mut actual = vec![0.0f32; expected.len()];

    let mut buf_a = unsafe {
        Buffer::<i8>::create(&context, CL_MEM_READ_ONLY, packed_weights.len(), std::ptr::null_mut())
            .map_err(|err| io_error(format!("failed to create input A buffer: {err}")))?
    };
    let mut buf_b = unsafe {
        Buffer::<u8>::create(&context, CL_MEM_READ_ONLY, activations.len(), std::ptr::null_mut())
            .map_err(|err| io_error(format!("failed to create input B buffer: {err}")))?
    };
    let buf_out = unsafe {
        Buffer::<f32>::create(&context, CL_MEM_WRITE_ONLY, actual.len(), std::ptr::null_mut())
            .map_err(|err| io_error(format!("failed to create output buffer: {err}")))?
    };

    unsafe {
        queue
            .enqueue_write_buffer(&mut buf_a, CL_BLOCKING, 0, &packed_weights, &[])
            .map_err(|err| io_error(format!("failed to write input A buffer: {err}")))?;
        queue
            .enqueue_write_buffer(&mut buf_b, CL_BLOCKING, 0, &activations, &[])
            .map_err(|err| io_error(format!("failed to write input B buffer: {err}")))?;
    }

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
        activation_values: activations.len(),
        max_abs_error,
        mean_abs_error,
    })
}

fn pack_i2s_weights(weights: &[i8]) -> Result<Vec<i8>, Box<dyn Error>> {
    weights
        .chunks(4)
        .map(|chunk| -> Result<i8, Box<dyn Error>> {
            let mut packed = 0u8;
            for (sub, value) in chunk.iter().enumerate() {
                packed |= encode_i2s_weight(*value)? << (sub * 2);
            }
            Ok(packed as i8)
        })
        .collect()
}

fn encode_i2s_weight(value: i8) -> Result<u8, Box<dyn Error>> {
    match value {
        1 => Ok(0x01),
        -1 => Ok(0x03),
        0 => Ok(0x00),
        other => Err(io_error(format!("unsupported i2s fixture weight {other}"))),
    }
}

fn cpu_matmul_i2s_reference(
    weights: &[i8],
    activations: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for depth in 0..k {
                let weight = weights[row * k + depth] as f32;
                let activation = activations[depth * n + col] as f32;
                sum += weight * activation;
            }
            output[row * n + col] = sum;
        }
    }
    output
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

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}
