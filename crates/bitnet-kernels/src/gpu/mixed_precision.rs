//! Mixed precision support for GPU kernels

use bitnet_common::{KernelError, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg, sys::CUdevice_attribute,
};
use cudarc::nvrtc::compile_ptx;
use half::{bf16, f16};
use std::sync::Arc;

/// Mixed precision configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain floating point (BF16)
    BF16,
    /// Automatic precision selection
    Auto,
}

/// Mixed precision kernel provider
pub struct MixedPrecisionKernel {
    /// CUDA device index this kernel operates on
    device_id: usize,
    /// CUDA context for the selected device
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    /// Stream used for kernel launches
    stream: Arc<CudaStream>,
    /// Loaded CUDA module containing mixed precision kernels
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    /// FP16 matrix multiplication kernel
    matmul_fp16_fn: CudaFunction,
    /// BF16 matrix multiplication kernel (may be unavailable on some devices)
    matmul_bf16_fn: Option<CudaFunction>,
    /// Whether the device supports native FP16 operations
    supports_fp16: bool,
    /// Whether the device supports native BF16 operations
    supports_bf16: bool,
    /// Current precision mode
    precision_mode: PrecisionMode,
}

impl MixedPrecisionKernel {
    /// Create a new mixed precision kernel provider
    pub fn new(device_id: usize) -> Result<Self> {
        log::info!("Creating mixed precision kernel for device {}", device_id);

        // Create CUDA context and stream for the specified device
        let ctx = CudaContext::new(device_id).map_err(|e| KernelError::GpuError {
            reason: format!(
                "Failed to create CUDA context for device {}: {:?}",
                device_id, e
            ),
        })?;
        let stream = ctx.default_stream();

        // Compile and load the mixed precision kernels
        let ptx = compile_ptx(include_str!("kernels/mixed_precision_kernels.cu"))
            .map_err(|e| KernelError::GpuError {
                reason: format!("Failed to compile PTX for mixed precision kernels: {:?}", e),
            })?;
        let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to load mixed precision CUDA module: {:?}", e),
        })?;

        // Load kernel functions
        let matmul_fp16_fn = module
            .load_function("bitnet_matmul_fp16")
            .map_err(|e| KernelError::GpuError {
                reason: format!("Failed to load FP16 matmul kernel: {:?}", e),
            })?;
        let matmul_bf16_fn = match module.load_function("bitnet_matmul_bf16") {
            Ok(f) => Some(f),
            Err(e) => {
                log::warn!("Failed to load BF16 matmul kernel: {:?}", e);
                None
            }
        };

        // Query device capabilities
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| KernelError::GpuError {
                reason: format!("Failed to get compute capability major: {:?}", e),
            })?;
        let supports_fp16 = major >= 6;
        let supports_bf16 = major >= 8;

        Ok(Self {
            device_id,
            ctx,
            stream,
            module,
            matmul_fp16_fn,
            matmul_bf16_fn,
            supports_fp16,
            supports_bf16,
            precision_mode: PrecisionMode::Auto,
        })
    }

    /// Set precision mode
    pub fn set_precision_mode(&mut self, mode: PrecisionMode) {
        self.precision_mode = mode;
        log::info!("Set precision mode to {:?}", mode);
    }

    /// Get current precision mode
    pub fn precision_mode(&self) -> PrecisionMode {
        self.precision_mode
    }

    /// Check if FP16 is supported on this device
    pub fn supports_fp16(&self) -> bool {
        self.supports_fp16
    }

    /// Check if BF16 is supported on this device
    pub fn supports_bf16(&self) -> bool {
        self.supports_bf16
    }

    /// Matrix multiplication with FP16 precision
    pub fn matmul_fp16(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.supports_fp16 {
            return Err(KernelError::GpuError {
                reason: "FP16 not supported on this device".to_string(),
            }
            .into());
        }

        log::debug!("Launching FP16 matmul on device {}", self.device_id);

        let func = &self.matmul_fp16_fn;

        // Convert inputs to FP16
        let a_half: Vec<u16> = a
            .iter()
            .map(|&x| f16::from_f32(x).to_bits())
            .collect();
        let b_half: Vec<u16> = b
            .iter()
            .map(|&x| f16::from_f32(x).to_bits())
            .collect();

        // Transfer data to device
        let a_dev = self.stream.memcpy_stod(&a_half).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer A to device: {:?}", e),
        })?;
        let b_dev = self.stream.memcpy_stod(&b_half).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer B to device: {:?}", e),
        })?;
        let mut c_dev: CudaSlice<u16> = self
            .stream
            .alloc_zeros(m * n)
            .map_err(|e| {
                KernelError::GpuError {
                    reason: format!("Failed to allocate C on device: {:?}", e),
                }
            })?;

        // Configure launch parameters
        const BLOCK: u32 = 16;
        let grid_x = (n as u32).div_ceil(BLOCK);
        let grid_y = (m as u32).div_ceil(BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (BLOCK, BLOCK, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let mut builder = self.stream.launch_builder(func);
        builder.arg(&a_dev);
        builder.arg(&b_dev);
        builder.arg(&mut c_dev);
        let m_arg = m as i32;
        let n_arg = n as i32;
        let k_arg = k as i32;
        builder.arg(&m_arg);
        builder.arg(&n_arg);
        builder.arg(&k_arg);

        unsafe { builder.launch(cfg) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP16 matmul kernel: {:?}", e),
        })?;

        // Copy result back and convert to f32
        let c_half: Vec<u16> = self.stream.memcpy_dtov(&c_dev).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Failed to copy result back to host: {:?}", e),
            }
        })?;
        for (dst, bits) in c.iter_mut().zip(c_half.iter()) {
            *dst = f16::from_bits(*bits).to_f32();
        }

        Ok(())
    }

    /// Matrix multiplication with BF16 precision
    pub fn matmul_bf16(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.supports_bf16 {
            return Err(KernelError::GpuError {
                reason: "BF16 not supported on this device".to_string(),
            }
            .into());
        }

        log::debug!("Launching BF16 matmul on device {}", self.device_id);

        let func = match &self.matmul_bf16_fn {
            Some(f) => f,
            None => {
                return Err(KernelError::GpuError {
                    reason: "BF16 kernel not loaded".to_string(),
                }
                .into())
            }
        };

        // Convert inputs to BF16
        let a_bf16: Vec<u16> = a
            .iter()
            .map(|&x| bf16::from_f32(x).to_bits())
            .collect();
        let b_bf16: Vec<u16> = b
            .iter()
            .map(|&x| bf16::from_f32(x).to_bits())
            .collect();

        // Transfer data
        let a_dev = self.stream.memcpy_stod(&a_bf16).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer A to device: {:?}", e),
        })?;
        let b_dev = self.stream.memcpy_stod(&b_bf16).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer B to device: {:?}", e),
        })?;
        let mut c_dev: CudaSlice<u16> = self
            .stream
            .alloc_zeros(m * n)
            .map_err(|e| {
                KernelError::GpuError {
                    reason: format!("Failed to allocate C on device: {:?}", e),
                }
            })?;

        // Launch configuration
        const BLOCK: u32 = 16;
        let grid_x = (n as u32).div_ceil(BLOCK);
        let grid_y = (m as u32).div_ceil(BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (BLOCK, BLOCK, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let mut builder = self.stream.launch_builder(func);
        builder.arg(&a_dev);
        builder.arg(&b_dev);
        builder.arg(&mut c_dev);
        let m_arg = m as i32;
        let n_arg = n as i32;
        let k_arg = k as i32;
        builder.arg(&m_arg);
        builder.arg(&n_arg);
        builder.arg(&k_arg);

        unsafe { builder.launch(cfg) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch BF16 matmul kernel: {:?}", e),
        })?;

        // Copy result back
        let c_bf16: Vec<u16> = self.stream.memcpy_dtov(&c_dev).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Failed to copy result back to host: {:?}", e),
            }
        })?;
        for (dst, bits) in c.iter_mut().zip(c_bf16.iter()) {
            *dst = bf16::from_bits(*bits).to_f32();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_creation() {
        let kernel = MixedPrecisionKernel::new(0);
        assert!(kernel.is_ok());

        if let Ok(mut kernel) = kernel {
            assert_eq!(kernel.precision_mode(), PrecisionMode::Auto);

            kernel.set_precision_mode(PrecisionMode::FP16);
            assert_eq!(kernel.precision_mode(), PrecisionMode::FP16);
        }
    }

    #[test]
    fn test_precision_support() {
        let kernel = MixedPrecisionKernel::new(0).unwrap();

        // These will return false until API is fixed
        assert!(!kernel.supports_fp16());
        assert!(!kernel.supports_bf16());
    }
}
