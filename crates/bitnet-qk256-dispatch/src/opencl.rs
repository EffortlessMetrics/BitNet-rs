use bitnet_common::{BitNetError, KernelError, Result};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, ClMem};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use std::ptr::null_mut;
use std::sync::{Arc, OnceLock};

pub const QK256_OPENCL_KERNEL_NAME: &str = "qk256_gemm_no_scale";

pub const QK256_OPENCL_KERNEL_SRC: &str = r#"
inline int qk256_nearest_int_reference(const float fval) {
    const float val = fval + 12582912.0f;
    const int bits = as_int(val);
    return (bits & 0x007fffff) - 0x00400000;
}

__kernel void qk256_gemm_no_scale(
    __global const uchar* qs,
    __global const float* input,
    __global float* output,
    const uint input_rows,
    const uint rows,
    const uint cols,
    const uint row_stride_bytes,
    const float scale
) {
    const uint row = get_global_id(0);
    const uint input_row = get_global_id(1);
    if (row >= rows || input_row >= input_rows) {
        return;
    }

    __global const uchar* row_bytes = qs + ((ulong)row * (ulong)row_stride_bytes);
    __global const float* x = input + ((ulong)input_row * (ulong)cols);

    float max_abs = 0.00001f;
    for (uint col = 0; col < cols; ++col) {
        max_abs = fmax(max_abs, fabs(x[col]));
    }

    const float act_scale = 127.0f / max_abs;
    int integer_dot = 0;
    int act_sum = 0;
    for (uint col = 0; col < cols; ++col) {
        const uint group128 = col / 128u;
        const uint within = col - (group128 * 128u);
        const uint lane = within / 32u;
        const uint pos = within - (lane * 32u);
        const uchar packed = row_bytes[(group128 * 32u) + pos];
        const uchar code = (packed >> (6u - (lane * 2u))) & 3u;
        const int q = max(-128, min(127, qk256_nearest_int_reference(x[col] * act_scale)));
        integer_dot += ((int)code) * q;
        act_sum += q;
    }

    output[((ulong)input_row * (ulong)rows) + (ulong)row] =
        (((float)(integer_dot - act_sum)) / act_scale) * scale;
}
"#;

pub fn gemm_qk256_opencl(
    qs_data: &[u8],
    input: &[f32],
    output: &mut [f32],
    input_rows: usize,
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    scale: f32,
) -> Result<()> {
    validate_args(qs_data, input, output, input_rows, rows, cols, row_stride_bytes)?;

    let runtime = qk256_runtime()?;
    runtime.run(qs_data, input, output, input_rows, rows, cols, row_stride_bytes, scale)
}

fn validate_args(
    qs_data: &[u8],
    input: &[f32],
    output: &[f32],
    input_rows: usize,
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    if input_rows == 0 || rows == 0 || cols == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "QK256 OpenCL dimensions must be non-zero".to_string(),
        }));
    }
    let expected_qs = checked_len("rows * row_stride_bytes", rows, row_stride_bytes)?;
    let expected_input = checked_len("input_rows * cols", input_rows, cols)?;
    let expected_output = checked_len("input_rows * rows", input_rows, rows)?;

    if qs_data.len() < expected_qs {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("QK256 OpenCL qs length {} < expected {}", qs_data.len(), expected_qs),
        }));
    }
    if input.len() < expected_input {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "QK256 OpenCL input length {} < expected {}",
                input.len(),
                expected_input
            ),
        }));
    }
    if output.len() != expected_output {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "QK256 OpenCL output length {} != expected {}",
                output.len(),
                expected_output
            ),
        }));
    }
    u32::try_from(input_rows).map_err(|_| too_large("input_rows", input_rows))?;
    u32::try_from(rows).map_err(|_| too_large("rows", rows))?;
    u32::try_from(cols).map_err(|_| too_large("cols", cols))?;
    u32::try_from(row_stride_bytes).map_err(|_| too_large("row_stride_bytes", row_stride_bytes))?;
    Ok(())
}

fn checked_len(name: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("QK256 OpenCL length overflow for {name}: {lhs} * {rhs}"),
        })
    })
}

fn too_large(field: &str, value: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!("QK256 OpenCL {field}={value} exceeds u32 range"),
    })
}

struct OpenClQk256Runtime {
    context: Context,
    queue: CommandQueue,
    program: Program,
}

// SAFETY: The OpenCL command queue serializes submitted work; the cached runtime
// owns immutable context/program handles after initialization.
unsafe impl Send for OpenClQk256Runtime {}
unsafe impl Sync for OpenClQk256Runtime {}

static QK256_RUNTIME: OnceLock<Arc<OpenClQk256Runtime>> = OnceLock::new();

fn qk256_runtime() -> Result<Arc<OpenClQk256Runtime>> {
    if let Some(runtime) = QK256_RUNTIME.get() {
        return Ok(runtime.clone());
    }

    let runtime = Arc::new(OpenClQk256Runtime::new()?);
    if QK256_RUNTIME.set(runtime.clone()).is_ok() {
        Ok(runtime)
    } else {
        Ok(QK256_RUNTIME
            .get()
            .expect("QK256 runtime must be set after racing initialization")
            .clone())
    }
}

impl OpenClQk256Runtime {
    fn new() -> Result<Self> {
        let platforms = get_platforms().map_err(|e| {
            BitNetError::Kernel(KernelError::GpuError {
                reason: format!("failed to query OpenCL platforms: {e}"),
            })
        })?;

        for platform in &platforms {
            let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
            for device_id in device_ids {
                let device = Device::new(device_id);
                let vendor = device.vendor().unwrap_or_default();
                if !vendor.to_ascii_lowercase().contains("intel") {
                    continue;
                }

                let context = Context::from_device(&device).map_err(|e| {
                    BitNetError::Kernel(KernelError::GpuError {
                        reason: format!("failed to create OpenCL context: {e}"),
                    })
                })?;
                let queue = CommandQueue::create_default_with_properties(
                    &context,
                    CL_QUEUE_PROFILING_ENABLE,
                    0,
                )
                .map_err(|e| {
                    BitNetError::Kernel(KernelError::GpuError {
                        reason: format!("failed to create OpenCL command queue: {e}"),
                    })
                })?;
                let program = Program::create_and_build_from_source(
                    &context,
                    QK256_OPENCL_KERNEL_SRC,
                    "-cl-std=CL1.2",
                )
                .map_err(|e| {
                    BitNetError::Kernel(KernelError::GpuError {
                        reason: format!("failed to build QK256 OpenCL program: {e}"),
                    })
                })?;

                return Ok(Self { context, queue, program });
            }
        }

        Err(BitNetError::Kernel(KernelError::DeviceUnavailable {
            reason: "no Intel GPU OpenCL device found for QK256 dispatch".to_string(),
        }))
    }

    fn run(
        &self,
        qs_data: &[u8],
        input: &[f32],
        output: &mut [f32],
        input_rows: usize,
        rows: usize,
        cols: usize,
        row_stride_bytes: usize,
        scale: f32,
    ) -> Result<()> {
        let mut qs_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, qs_data.len(), null_mut())
        }
        .map_err(buffer_error("QK256 weights"))?;
        let mut input_buf = unsafe {
            Buffer::<f32>::create(&self.context, CL_MEM_READ_ONLY, input.len(), null_mut())
        }
        .map_err(buffer_error("QK256 input"))?;
        let output_buf = unsafe {
            Buffer::<f32>::create(&self.context, CL_MEM_WRITE_ONLY, output.len(), null_mut())
        }
        .map_err(buffer_error("QK256 output"))?;

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut qs_buf, CL_BLOCKING, 0, qs_data, &[])
                .map_err(transfer_error("write QK256 weights"))?;
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(transfer_error("write QK256 input"))?;
        }

        let kernel = Kernel::create(&self.program, QK256_OPENCL_KERNEL_NAME).map_err(|e| {
            BitNetError::Kernel(KernelError::LaunchFailed {
                kernel: QK256_OPENCL_KERNEL_NAME.to_string(),
                reason: format!("failed to create kernel: {e}"),
            })
        })?;

        let input_rows_u32 = input_rows as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let row_stride_u32 = row_stride_bytes as u32;

        let event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&qs_buf.get())
                .set_arg(&input_buf.get())
                .set_arg(&output_buf.get())
                .set_arg(&input_rows_u32)
                .set_arg(&rows_u32)
                .set_arg(&cols_u32)
                .set_arg(&row_stride_u32)
                .set_arg(&scale)
                .set_global_work_sizes(&[rows, input_rows])
                .enqueue_nd_range(&self.queue)
                .map_err(|e| {
                    BitNetError::Kernel(KernelError::LaunchFailed {
                        kernel: QK256_OPENCL_KERNEL_NAME.to_string(),
                        reason: format!("enqueue failed: {e}"),
                    })
                })?
        };

        event.wait().map_err(|e| {
            BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("QK256 OpenCL wait failed: {e}"),
            })
        })?;

        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &[])
                .map_err(transfer_error("read QK256 output"))?;
        }

        Ok(())
    }
}

fn buffer_error(name: &'static str) -> impl Fn(opencl3::error_codes::ClError) -> BitNetError {
    move |e| {
        BitNetError::Kernel(KernelError::GpuError {
            reason: format!("failed to allocate {name} buffer: {e}"),
        })
    }
}

fn transfer_error(name: &'static str) -> impl Fn(opencl3::error_codes::ClError) -> BitNetError {
    move |e| {
        BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("{name} transfer failed: {e}"),
        })
    }
}
