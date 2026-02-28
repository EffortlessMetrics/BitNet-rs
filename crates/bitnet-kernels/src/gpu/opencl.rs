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

/// Runtime env var lookup (unlike `option_env!` which is compile-time).
fn option_env_dynamic(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

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
    matmul_tiled_program: Option<Program>,
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
                    let matmul_tiled_program = Self::compile_program(
                        &context,
                        crate::kernels::MATMUL_I2S_TILED_SRC,
                        "matmul_i2s_tiled",
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
                        matmul_tiled_program,
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
        // Check BITNET_OPENCL_FORCE_SOURCE to skip SPIR-V
        let force_source = std::env::var("BITNET_OPENCL_FORCE_SOURCE")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Try SPIR-V first (if pre-compiled and not forced to source)
        if !force_source {
            if let Some(program) = Self::try_spirv_program(context, name) {
                return Some(program);
            }
        }

        // Fallback: runtime source compilation
        match Program::create_and_build_from_source(context, source, "") {
            Ok(program) => {
                info!("Compiled OpenCL program from source: {}", name);
                Some(program)
            }
            Err(e) => {
                warn!("Failed to compile OpenCL program '{}': {}", name, e);
                None
            }
        }
    }

    /// Attempt to load a pre-compiled SPIR-V program for the given kernel name.
    ///
    /// Looks for `BITNET_SPV_<NAME>` env var set by build.rs pointing to the
    /// `.spv` file.  Uses `clCreateProgramWithIL` when available.
    fn try_spirv_program(context: &Context, name: &str) -> Option<Program> {
        let env_key = format!("BITNET_SPV_{}", name.to_uppercase());
        let spv_path = match option_env_dynamic(&env_key) {
            Some(p) => std::path::PathBuf::from(p),
            None => return None,
        };

        if !spv_path.exists() {
            debug!("SPIR-V file not found at {}", spv_path.display());
            return None;
        }

        let spv_bytes = match std::fs::read(&spv_path) {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed to read SPIR-V file {}: {}", spv_path.display(), e);
                return None;
            }
        };

        match Program::create_and_build_from_il(context, &spv_bytes, "") {
            Ok(program) => {
                info!(
                    "Loaded SPIR-V program for '{}' from {}",
                    name,
                    spv_path.display()
                );
                Some(program)
            }
            Err(e) => {
                debug!(
                    "SPIR-V load failed for '{}' (falling back to source): {}",
                    name, e
                );
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

        // Prefer tiled kernel when available; fall back to naive
        let (program, kernel_name) = if let Some(ref tiled) = ctx.matmul_tiled_program {
            (tiled, "matmul_i2s_tiled")
        } else if let Some(ref naive) = ctx.matmul_program {
            (naive, "matmul_i2s")
        } else {
            return Err(KernelError::GpuError {
                reason: "No matmul program compiled".into(),
            }
            .into());
        };

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
        let kernel = Kernel::create(program, kernel_name).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Kernel create: {}", e),
            }
        })?;

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        // Tiled kernel uses work-group-aligned global sizes
        let tile_size = 16usize;
        let (global_m, global_n) = if kernel_name == "matmul_i2s_tiled" {
            let gm = ((m + tile_size - 1) / tile_size) * tile_size;
            let gn = ((n + tile_size - 1) / tile_size) * tile_size;
            (gm, gn)
        } else {
            (m, n)
        };

        let event = unsafe {
            let mut exec = ExecuteKernel::new(&kernel);
            exec.set_arg(&buf_a.get())
                .set_arg(&buf_b.get())
                .set_arg(&buf_c.get())
                .set_arg(&m_u32)
                .set_arg(&n_u32)
                .set_arg(&k_u32)
                .set_global_work_sizes(&[global_m, global_n]);

            if kernel_name == "matmul_i2s_tiled" {
                exec.set_local_work_sizes(&[tile_size, tile_size]);
            }

            exec.enqueue_nd_range(&ctx.queue)
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
