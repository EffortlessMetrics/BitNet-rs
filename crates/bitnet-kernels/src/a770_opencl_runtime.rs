//! Selected-device Intel Arc A770 OpenCL runtime helpers.
//!
//! These helpers are the first bridge from A770 fixture parity into reusable
//! runtime code. They intentionally cover only the grouped QK256 I2_S x
//! prequantized I8_S scaled GEMV contract. Activation quantization remains a
//! CPU-side input to this helper, so this is not a full residency claim.

use bitnet_common::{KernelError, Result};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, ClMem};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use std::fmt::Display;

const QK256_I2S_I8S_SCALED_GEMV_SRC: &str = r#"
__kernel void qk256_i2s_i8s_scaled_gemv(
    __global const char* q,
    __global const uchar* qs,
    __global float* y,
    const uint rows,
    const uint cols,
    const uint row_stride_bytes,
    const int activation_sum,
    const float activation_scale,
    const float weight_scale
) {
    const uint row = get_global_id(0);
    if (row >= rows) return;

    int int_dot = 0;
    const uint row_base = row * row_stride_bytes;

    for (uint col = 0; col < cols; col++) {
        const uint block = col / 256;
        const uint offset = col - block * 256;
        const uint chunk = offset / 128;
        const uint lane = (offset - chunk * 128) / 32;
        const uint gp = offset & 31;
        const uint byte_index = row_base + block * 64 + chunk * 32 + gp;
        const uchar packed = qs[byte_index];
        const uchar code = (packed >> (6 - lane * 2)) & 0x03;
        int_dot += ((int)code) * ((int)q[col]);
    }

    y[row] = (((float)(int_dot - activation_sum)) / activation_scale) * weight_scale;
}
"#;

/// Runtime request for the selected-device A770 QK256 scaled GEMV.
#[derive(Debug)]
pub struct A770OpenClQk256ScaledGemv<'a> {
    /// Prequantized I8_S activation row.
    pub activations_i8: &'a [i8],
    /// GGML grouped QK256 I2_S weight bytes.
    pub packed_qk256: &'a [u8],
    /// Number of output rows.
    pub rows: usize,
    /// Number of input columns.
    pub cols: usize,
    /// Packed byte stride for each output row.
    pub row_stride_bytes: usize,
    /// Sum of the prequantized I8_S activation row.
    pub activation_sum: i32,
    /// I8_S activation scale.
    pub activation_scale: f32,
    /// BitNet.cpp inline weight scale.
    pub weight_scale: f32,
}

/// Runtime result for the selected-device A770 QK256 scaled GEMV.
#[derive(Debug, Clone, PartialEq)]
pub struct A770OpenClQk256ScaledGemvResult {
    /// Output values in row order.
    pub output: Vec<f32>,
    /// OpenCL platform index selected for execution.
    pub platform_index: usize,
    /// OpenCL device index selected for execution.
    pub device_index: usize,
    /// OpenCL platform name.
    pub platform_name: String,
    /// Selected OpenCL device name.
    pub runtime_device: String,
    /// Selected OpenCL device vendor.
    pub vendor: String,
    /// Selected OpenCL driver version.
    pub driver_version: String,
    /// Host-to-device bytes uploaded for this invocation.
    pub host_to_device_bytes: usize,
    /// Device-to-host bytes read for this invocation.
    pub device_to_host_bytes: usize,
    /// Number of OpenCL kernel invocations.
    pub kernel_invocations: usize,
}

/// Run grouped QK256 I2_S x prequantized I8_S scaled GEMV on the selected A770.
pub fn run_a770_qk256_i8s_scaled_gemv(
    request: A770OpenClQk256ScaledGemv<'_>,
) -> Result<A770OpenClQk256ScaledGemvResult> {
    validate_request(&request)?;
    let selected = find_a770_device()?;
    let context = Context::from_device(&selected.device).map_err(gpu_err("create context"))?;
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .map_err(gpu_err("create command queue"))?;
    let program =
        Program::create_and_build_from_source(&context, QK256_I2S_I8S_SCALED_GEMV_SRC, "")
            .map_err(gpu_err("build qk256_i2s_i8s_scaled_gemv program"))?;
    let kernel =
        Kernel::create(&program, "qk256_i2s_i8s_scaled_gemv").map_err(gpu_err("create kernel"))?;

    let mut output = vec![0.0f32; request.rows];
    let mut buf_q = unsafe {
        Buffer::<i8>::create(
            &context,
            CL_MEM_READ_ONLY,
            request.activations_i8.len(),
            std::ptr::null_mut(),
        )
        .map_err(gpu_err("create activation buffer"))?
    };
    let mut buf_qs = unsafe {
        Buffer::<u8>::create(
            &context,
            CL_MEM_READ_ONLY,
            request.rows * request.row_stride_bytes,
            std::ptr::null_mut(),
        )
        .map_err(gpu_err("create packed weight buffer"))?
    };
    let buf_out = unsafe {
        Buffer::<f32>::create(&context, CL_MEM_WRITE_ONLY, output.len(), std::ptr::null_mut())
            .map_err(gpu_err("create output buffer"))?
    };

    let weight_bytes = &request.packed_qk256[..request.rows * request.row_stride_bytes];
    unsafe {
        queue
            .enqueue_write_buffer(&mut buf_q, CL_BLOCKING, 0, request.activations_i8, &[])
            .map_err(gpu_err("write activation buffer"))?;
        queue
            .enqueue_write_buffer(&mut buf_qs, CL_BLOCKING, 0, weight_bytes, &[])
            .map_err(gpu_err("write packed weight buffer"))?;
    }

    let rows = request.rows as u32;
    let cols = request.cols as u32;
    let row_stride_bytes = request.row_stride_bytes as u32;
    let event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buf_q.get())
            .set_arg(&buf_qs.get())
            .set_arg(&buf_out.get())
            .set_arg(&rows)
            .set_arg(&cols)
            .set_arg(&row_stride_bytes)
            .set_arg(&request.activation_sum)
            .set_arg(&request.activation_scale)
            .set_arg(&request.weight_scale)
            .set_global_work_sizes(&[request.rows])
            .enqueue_nd_range(&queue)
            .map_err(gpu_err("enqueue qk256_i2s_i8s_scaled_gemv kernel"))?
    };
    event.wait().map_err(gpu_err("wait for qk256_i2s_i8s_scaled_gemv kernel"))?;

    unsafe {
        queue
            .enqueue_read_buffer(&buf_out, CL_BLOCKING, 0, &mut output, &[])
            .map_err(gpu_err("read output buffer"))?;
    }

    Ok(A770OpenClQk256ScaledGemvResult {
        output,
        platform_index: selected.platform_index,
        device_index: selected.device_index,
        platform_name: selected.platform_name,
        runtime_device: selected.device_name,
        vendor: selected.vendor,
        driver_version: selected.driver_version,
        host_to_device_bytes: std::mem::size_of_val(request.activations_i8) + weight_bytes.len(),
        device_to_host_bytes: std::mem::size_of::<f32>() * request.rows,
        kernel_invocations: 1,
    })
}

fn validate_request(request: &A770OpenClQk256ScaledGemv<'_>) -> Result<()> {
    if request.rows == 0 || request.cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "A770 QK256 OpenCL request dimensions must be non-zero".to_string(),
        }
        .into());
    }
    if !request.cols.is_multiple_of(256) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL cols must be a multiple of 256, got {}",
                request.cols
            ),
        }
        .into());
    }
    let expected_stride = (request.cols / 256) * 64;
    if request.row_stride_bytes != expected_stride {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL row_stride_bytes {} != expected {} for cols={}",
                request.row_stride_bytes, expected_stride, request.cols
            ),
        }
        .into());
    }
    if request.activations_i8.len() < request.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL activation length {} < cols {}",
                request.activations_i8.len(),
                request.cols
            ),
        }
        .into());
    }
    let expected_bytes = request.rows * request.row_stride_bytes;
    if request.packed_qk256.len() < expected_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL packed bytes {} < expected {}",
                request.packed_qk256.len(),
                expected_bytes
            ),
        }
        .into());
    }
    if !request.activation_scale.is_finite() || request.activation_scale == 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL activation scale must be finite and non-zero: {}",
                request.activation_scale
            ),
        }
        .into());
    }
    if !request.weight_scale.is_finite() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL weight scale is not finite: {}",
                request.weight_scale
            ),
        }
        .into());
    }
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

fn find_a770_device() -> Result<SelectedA770Device> {
    let platforms = get_platforms().map_err(gpu_err("enumerate OpenCL platforms"))?;
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
    Err(KernelError::GpuError {
        reason: "Intel Arc A770 OpenCL device was not visible".to_string(),
    }
    .into())
}

fn is_intel_vendor(value: &str) -> bool {
    value.to_ascii_lowercase().contains("intel")
}

fn is_a770_device_name(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("a770")) || lower.contains("56a0")
}

fn gpu_err<E: Display>(context: &'static str) -> impl FnOnce(E) -> KernelError {
    move |err| KernelError::GpuError { reason: format!("A770 OpenCL {context}: {err}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(
        activations_i8: &'a [i8],
        packed_qk256: &'a [u8],
    ) -> A770OpenClQk256ScaledGemv<'a> {
        A770OpenClQk256ScaledGemv {
            activations_i8,
            packed_qk256,
            rows: 1,
            cols: 256,
            row_stride_bytes: 64,
            activation_sum: 0,
            activation_scale: 1.0,
            weight_scale: 0.25,
        }
    }

    #[test]
    fn validation_rejects_non_qk256_cols() {
        let q = vec![0; 128];
        let weights = vec![0; 64];
        let mut req = request(&q, &weights);
        req.cols = 128;
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn validation_rejects_bad_stride() {
        let q = vec![0; 256];
        let weights = vec![0; 64];
        let mut req = request(&q, &weights);
        req.row_stride_bytes = 32;
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn validation_accepts_minimal_qk256_contract() -> Result<()> {
        let q = vec![0; 256];
        let weights = vec![0; 64];
        validate_request(&request(&q, &weights))
    }
}
