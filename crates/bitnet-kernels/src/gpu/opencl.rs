//! OpenCL-based GPU kernel provider for Intel Arc GPUs.
//!
//! This module implements the [`KernelProvider`] trait using OpenCL 3.0
//! via the `opencl3` crate, targeting Intel's Compute Runtime for Arc GPUs.

use crate::KernelProvider;
use bitnet_common::{KernelError, QuantizationType, Result};
use log::{debug, info, warn};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{cl_device_id, CL_BLOCKING};

/// OpenCL kernel provider for Intel Arc GPUs.
///
/// Manages an OpenCL context, command queue, and compiled programs
/// for BitNet compute operations.
pub struct OpenClKernel {
    /// Platform name for logging
    platform_name: String,
    /// Device name for logging
    device_name: String,
    /// Whether the kernel is ready for compute
    ready: bool,
    // OpenCL handles stored as opaque wrapper to avoid exposing opencl3 in public API
    context: Option<OpenClContext>,
}

/// Internal OpenCL context wrapper holding all runtime handles.
struct OpenClContext {
    _platform: opencl3::platform::Platform,
    _device_id: cl_device_id,
    context: Context,
    queue: CommandQueue,
    matmul_program: Option<Program>,
    rmsnorm_program: Option<Program>,
    _quantize_program: Option<Program>,
    _elementwise_program: Option<Program>,
}

// SAFETY: OpenCL handles are thread-safe when used with proper synchronization.
// The CommandQueue serializes operations internally.
unsafe impl Send for OpenClContext {}
unsafe impl Sync for OpenClContext {}

impl std::fmt::Debug for OpenClKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenClKernel")
            .field("platform_name", &self.platform_name)
            .field("device_name", &self.device_name)
            .field("ready", &self.ready)
            .finish()
    }
}

impl OpenClKernel {
    /// Attempt to create a new OpenCL kernel provider.
    ///
    /// Searches for Intel GPU devices via OpenCL platform enumeration.
    /// Returns `Ok(Self)` if a suitable device is found, error otherwise.
    pub fn new() -> Result<Self> {
        match Self::try_init() {
            Ok(kernel) => {
                info!(
                    "OpenCL kernel initialized: {} on {}",
                    kernel.device_name, kernel.platform_name
                );
                Ok(kernel)
            }
            Err(e) => {
                debug!("OpenCL initialization failed: {}", e);
                Err(e)
            }
        }
    }

    fn try_init() -> Result<Self> {
        let platforms = get_platforms().map_err(|e| {
            KernelError::GpuError {
                reason: format!("Failed to get OpenCL platforms: {}", e),
            }
        })?;

        if platforms.is_empty() {
            return Err(KernelError::GpuError {
                reason: "No OpenCL platforms found".into(),
            }
            .into());
        }

        // Search for Intel GPU device
        for platform in &platforms {
            let platform_name = platform.name().unwrap_or_default();
            debug!("Checking OpenCL platform: {}", platform_name);

            let device_ids = platform
                .get_devices(CL_DEVICE_TYPE_GPU)
                .unwrap_or_default();

            for device_id in device_ids {
                let device = Device::new(device_id);
                let device_name = device.name().unwrap_or_default();
                let vendor = device.vendor().unwrap_or_default();

                debug!("Found OpenCL device: {} (vendor: {})", device_name, vendor);

                // Accept Intel GPUs
                if vendor.to_lowercase().contains("intel") {
                    info!("Selected Intel GPU: {}", device_name);

                    let context = Context::from_device(&device).map_err(|e| {
                        KernelError::GpuError {
                            reason: format!("Failed to create OpenCL context: {}", e),
                        }
                    })?;

                    let queue =
                        CommandQueue::create_default_with_properties(
                            &context,
                            CL_QUEUE_PROFILING_ENABLE,
                            0,
                        )
                        .map_err(|e| KernelError::GpuError {
                            reason: format!("Failed to create command queue: {}", e),
                        })?;

                    // Compile kernel programs (non-fatal if compilation fails)
                    let matmul_program = Self::compile_program(
                        &context,
                        crate::kernels::MATMUL_I2S_SRC,
                        "matmul_i2s",
                    );
                    let rmsnorm_program = Self::compile_program(
                        &context,
                        crate::kernels::RMSNORM_SRC,
                        "rmsnorm",
                    );
                    let quantize_program = Self::compile_program(
                        &context,
                        crate::kernels::QUANTIZE_I2S_SRC,
                        "quantize_i2s",
                    );
                    let elementwise_program = Self::compile_program(
                        &context,
                        crate::kernels::ELEMENTWISE_SRC,
                        "elementwise",
                    );

                    let opencl_ctx = OpenClContext {
                        _platform: *platform,
                        _device_id: device_id,
                        context,
                        queue,
                        matmul_program,
                        rmsnorm_program,
                        _quantize_program: quantize_program,
                        _elementwise_program: elementwise_program,
                    };

                    return Ok(Self {
                        platform_name,
                        device_name,
                        ready: true,
                        context: Some(opencl_ctx),
                    });
                }
            }
        }

        Err(KernelError::GpuError {
            reason: "No Intel GPU device found via OpenCL".into(),
        }
        .into())
    }

    fn compile_program(
        context: &Context,
        source: &str,
        name: &str,
    ) -> Option<Program> {
        match Program::create_and_build_from_source(context, source, "") {
            Ok(program) => {
                info!("Compiled OpenCL program: {}", name);
                Some(program)
            }
            Err(e) => {
                warn!("Failed to compile OpenCL program '{}': {}", name, e);
                None
            }
        }
    }

    /// Get the device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get the platform name.
    pub fn platform_name(&self) -> &str {
        &self.platform_name
    }
}

impl KernelProvider for OpenClKernel {
    fn name(&self) -> &'static str {
        "opencl-intel"
    }

    fn is_available(&self) -> bool {
        self.ready && self.context.is_some()
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let ctx = self.context.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: "OpenCL context not initialized".into(),
        })?;

        let program = ctx.matmul_program.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: "matmul_i2s program not compiled".into(),
        })?;

        use opencl3::kernel::{ExecuteKernel, Kernel};

        // Create buffers
        let mut buf_a = unsafe {
            Buffer::<i8>::create(
                &ctx.context,
                CL_MEM_READ_ONLY,
                a.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| KernelError::GpuError {
                reason: format!("Buffer A create: {}", e),
            })?
        };
        let mut buf_b = unsafe {
            Buffer::<u8>::create(
                &ctx.context,
                CL_MEM_READ_ONLY,
                b.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| KernelError::GpuError {
                reason: format!("Buffer B create: {}", e),
            })?
        };
        let buf_c = unsafe {
            Buffer::<f32>::create(
                &ctx.context,
                CL_MEM_WRITE_ONLY,
                c.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| KernelError::GpuError {
                reason: format!("Buffer C create: {}", e),
            })?
        };

        // Upload data (blocking writes)
        unsafe {
            ctx.queue
                .enqueue_write_buffer(&mut buf_a, CL_BLOCKING, 0, a, &[])
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Write A: {}", e),
                })?;
            ctx.queue
                .enqueue_write_buffer(&mut buf_b, CL_BLOCKING, 0, b, &[])
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Write B: {}", e),
                })?;
        }

        // Create and run kernel
        let kernel = Kernel::create(program, "matmul_i2s").map_err(|e| {
            KernelError::GpuError {
                reason: format!("Kernel create: {}", e),
            }
        })?;

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        let event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&buf_a.get())
                .set_arg(&buf_b.get())
                .set_arg(&buf_c.get())
                .set_arg(&m_u32)
                .set_arg(&n_u32)
                .set_arg(&k_u32)
                .set_global_work_sizes(&[m, n])
                .enqueue_nd_range(&ctx.queue)
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Enqueue: {}", e),
                })?
        };

        event.wait().map_err(|e| KernelError::GpuError {
            reason: format!("Kernel wait: {}", e),
        })?;

        // Read results (blocking)
        unsafe {
            ctx.queue
                .enqueue_read_buffer(&buf_c, CL_BLOCKING, 0, c, &[])
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Read C: {}", e),
                })?;
        }

        Ok(())
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        // Fall back to CPU quantization for correctness validation.
        // GPU quantization will be optimized in a follow-up PR.
        warn!("OpenCL quantize: falling back to CPU for correctness validation");
        crate::cpu::FallbackKernel.quantize(input, output, scales, qtype)
    }
}

impl OpenClKernel {
    /// Run optimized parallel RMSNorm on the GPU.
    ///
    /// Each work-group processes one row using tree reduction for the
    /// sum-of-squares, then normalises and scales in parallel.
    ///
    /// `input`  — `[rows × hidden_dim]` flattened row-major
    /// `weight` — `[hidden_dim]` per-element learnable scale
    /// `output` — `[rows × hidden_dim]`
    #[allow(dead_code)]
    pub fn rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        rows: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let ctx = self.context.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: "OpenCL context not initialized".into(),
        })?;

        let program = ctx.rmsnorm_program.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: "rmsnorm program not compiled".into(),
        })?;

        use opencl3::kernel::{ExecuteKernel, Kernel};

        let mut buf_input = unsafe {
            Buffer::<f32>::create(
                &ctx.context,
                CL_MEM_READ_ONLY,
                input.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| KernelError::GpuError {
                reason: format!("Buffer input create: {}", e),
            })?
        };
        let mut buf_weight = unsafe {
            Buffer::<f32>::create(
                &ctx.context,
                CL_MEM_READ_ONLY,
                weight.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| KernelError::GpuError {
                reason: format!("Buffer weight create: {}", e),
            })?
        };
        let buf_output = unsafe {
            Buffer::<f32>::create(
                &ctx.context,
                CL_MEM_WRITE_ONLY,
                output.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| KernelError::GpuError {
                reason: format!("Buffer output create: {}", e),
            })?
        };

        unsafe {
            ctx.queue
                .enqueue_write_buffer(&mut buf_input, CL_BLOCKING, 0, input, &[])
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Write input: {}", e),
                })?;
            ctx.queue
                .enqueue_write_buffer(&mut buf_weight, CL_BLOCKING, 0, weight, &[])
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Write weight: {}", e),
                })?;
        }

        let kernel = Kernel::create(program, "rms_norm_parallel").map_err(|e| {
            KernelError::GpuError {
                reason: format!("Kernel create: {}", e),
            }
        })?;

        let n_u32 = hidden_dim as u32;
        let local_size = 256usize;
        let global_size = rows * local_size;

        let event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&buf_input.get())
                .set_arg(&buf_weight.get())
                .set_arg(&buf_output.get())
                .set_arg(&n_u32)
                .set_arg(&eps)
                .set_global_work_sizes(&[global_size])
                .set_local_work_sizes(&[local_size])
                .enqueue_nd_range(&ctx.queue)
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Enqueue: {}", e),
                })?
        };

        event.wait().map_err(|e| KernelError::GpuError {
            reason: format!("Kernel wait: {}", e),
        })?;

        unsafe {
            ctx.queue
                .enqueue_read_buffer(&buf_output, CL_BLOCKING, 0, output, &[])
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Read output: {}", e),
                })?;
        }

        Ok(())
    }
}
