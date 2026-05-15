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
use std::sync::{Mutex, OnceLock};

pub const QK256_OPENCL_KERNEL_NAME: &str = "qk256_gemm_no_scale";

pub const QK256_OPENCL_KERNEL_SRC: &str = r#"
__kernel void qk256_gemm_no_scale(
    __global const uchar* qs,
    __global const float* input,
    __global float* output,
    const uint input_rows,
    const uint rows,
    const uint cols,
    const uint row_stride_bytes
) {
    const uint row = get_global_id(0);
    const uint input_row = get_global_id(1);
    if (row >= rows || input_row >= input_rows) {
        return;
    }

    __global const uchar* row_bytes = qs + ((ulong)row * (ulong)row_stride_bytes);
    __global const float* x = input + ((ulong)input_row * (ulong)cols);

    float acc = 0.0f;
    for (uint col = 0; col < cols; ++col) {
        const uchar packed = row_bytes[col >> 2];
        const uchar code = (packed >> ((col & 3u) * 2u)) & 3u;
        float w;
        if (code == 0u) {
            w = -2.0f;
        } else if (code == 1u) {
            w = -1.0f;
        } else if (code == 2u) {
            w = 1.0f;
        } else {
            w = 2.0f;
        }
        acc += w * x[col];
    }

    output[((ulong)input_row * (ulong)rows) + (ulong)row] = acc;
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
) -> Result<()> {
    validate_args(qs_data, input, output, input_rows, rows, cols, row_stride_bytes)?;

    let runtime = qk256_runtime()?;
    let runtime = runtime.lock().map_err(|_| {
        BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "QK256 OpenCL runtime cache lock was poisoned".to_string(),
        })
    })?;
    runtime.run(qs_data, input, output, input_rows, rows, cols, row_stride_bytes)
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

// SAFETY: OpenCL handles are reference-counted driver objects. The cached runtime
// is only accessed through a Mutex, so the single command queue is not used
// concurrently by Rust callers.
unsafe impl Send for OpenClQk256Runtime {}

static QK256_RUNTIME: OnceLock<Mutex<OpenClQk256Runtime>> = OnceLock::new();

fn qk256_runtime() -> Result<&'static Mutex<OpenClQk256Runtime>> {
    if let Some(runtime) = QK256_RUNTIME.get() {
        return Ok(runtime);
    }

    let runtime = Mutex::new(OpenClQk256Runtime::new()?);
    if QK256_RUNTIME.set(runtime).is_ok() {
        Ok(QK256_RUNTIME.get().expect("QK256 runtime must be set after initialization"))
    } else {
        Ok(QK256_RUNTIME.get().expect("QK256 runtime must be set after racing initialization"))
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
