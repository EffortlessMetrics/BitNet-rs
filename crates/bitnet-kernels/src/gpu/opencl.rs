//! OpenCL-based GPU kernel provider for Intel Arc GPUs.
//!
//! This module implements the [`KernelProvider`] trait using the
//! `bitnet-opencl` microcrate, targeting Intel's Compute Runtime for Arc GPUs.

use crate::KernelProvider;
use bitnet_common::{KernelError, QuantizationType, Result};
use bitnet_opencl::{
    AccessMode, OpenClBuffer, OpenClContext, OpenClQueue, ProgramCache,
};
use log::{debug, info, warn};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::ClMem;

/// OpenCL kernel provider for Intel Arc GPUs.
///
/// Manages an OpenCL context, command queue, and compiled programs
/// for BitNet compute operations via the `bitnet-opencl` abstraction layer.
pub struct OpenClKernel {
    /// Platform name for logging
    platform_name: String,
    /// Device name for logging
    device_name: String,
    /// Whether the kernel is ready for compute
    ready: bool,
    /// OpenCL runtime handles via bitnet-opencl abstractions.
    runtime: Option<OpenClRuntime>,
}

/// Internal OpenCL runtime holding all handles via bitnet-opencl abstractions.
struct OpenClRuntime {
    ctx: OpenClContext,
    queue: OpenClQueue,
    programs: ProgramCache,
}

// SAFETY: OpenCL handles are thread-safe when used with proper synchronization.
// The CommandQueue serializes operations internally.
unsafe impl Send for OpenClRuntime {}
unsafe impl Sync for OpenClRuntime {}

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
        let ctx =
            OpenClContext::new_intel().map_err(|e| KernelError::GpuError {
                reason: format!("OpenCL init: {}", e),
            })?;

        let platform_name = ctx.platform_name().to_string();
        let device_name = ctx.device_name().to_string();

        let queue =
            ctx.create_queue().map_err(|e| KernelError::GpuError {
                reason: format!("Failed to create command queue: {}", e),
            })?;

        let mut programs = ProgramCache::new();

        // Compile kernel programs (non-fatal if compilation fails)
        let _ = programs.compile_and_insert(
            &ctx,
            crate::kernels::MATMUL_I2S_SRC,
            "matmul_i2s",
        );
        let _ = programs.compile_and_insert(
            &ctx,
            crate::kernels::QUANTIZE_I2S_SRC,
            "quantize_i2s",
        );
        let _ = programs.compile_and_insert(
            &ctx,
            crate::kernels::ELEMENTWISE_SRC,
            "elementwise",
        );

        let runtime = OpenClRuntime {
            ctx,
            queue,
            programs,
        };

        Ok(Self {
            platform_name,
            device_name,
            ready: true,
            runtime: Some(runtime),
        })
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
        self.ready && self.runtime.is_some()
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
        let rt =
            self.runtime.as_ref().ok_or_else(|| KernelError::GpuError {
                reason: "OpenCL context not initialized".into(),
            })?;

        let program =
            rt.programs.get("matmul_i2s").ok_or_else(|| {
                KernelError::GpuError {
                    reason: "matmul_i2s program not compiled".into(),
                }
            })?;

        // Create buffers via bitnet-opencl
        let mut buf_a = OpenClBuffer::<i8>::new(
            &rt.ctx,
            a.len(),
            AccessMode::ReadOnly,
        )
        .map_err(|e| KernelError::GpuError {
            reason: format!("Buffer A: {}", e),
        })?;
        let mut buf_b = OpenClBuffer::<u8>::new(
            &rt.ctx,
            b.len(),
            AccessMode::ReadOnly,
        )
        .map_err(|e| KernelError::GpuError {
            reason: format!("Buffer B: {}", e),
        })?;
        let buf_c = OpenClBuffer::<f32>::new(
            &rt.ctx,
            c.len(),
            AccessMode::WriteOnly,
        )
        .map_err(|e| KernelError::GpuError {
            reason: format!("Buffer C: {}", e),
        })?;

        // Upload data via bitnet-opencl
        buf_a.write(&rt.queue, a).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Write A: {}", e),
            }
        })?;
        buf_b.write(&rt.queue, b).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Write B: {}", e),
            }
        })?;

        // Create and run kernel (direct opencl3 kernel execution)
        let kernel = Kernel::create(&program.inner, "matmul_i2s")
            .map_err(|e| KernelError::GpuError {
                reason: format!("Kernel create: {}", e),
            })?;

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        let event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&buf_a.inner.get())
                .set_arg(&buf_b.inner.get())
                .set_arg(&buf_c.inner.get())
                .set_arg(&m_u32)
                .set_arg(&n_u32)
                .set_arg(&k_u32)
                .set_global_work_sizes(&[m, n])
                .enqueue_nd_range(&rt.queue.inner)
                .map_err(|e| KernelError::GpuError {
                    reason: format!("Enqueue: {}", e),
                })?
        };

        event.wait().map_err(|e| KernelError::GpuError {
            reason: format!("Kernel wait: {}", e),
        })?;

        // Read results via bitnet-opencl
        buf_c.read(&rt.queue, c).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Read C: {}", e),
            }
        })?;

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
        warn!(
            "OpenCL quantize: falling back to CPU for correctness validation"
        );
        crate::cpu::FallbackKernel.quantize(input, output, scales, qtype)
    }
}
