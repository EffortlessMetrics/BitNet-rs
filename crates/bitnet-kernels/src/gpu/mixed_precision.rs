//! Mixed precision support for GPU kernels

use crate::KernelProvider;
use bitnet_common::{KernelError, QuantizationType, Result};
use cudarc::prelude::*;
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
    /// Automatic precision selection based on hardware
    Auto,
}

/// Mixed precision kernel provider
pub struct MixedPrecisionKernel {
    device: Arc<CudaDevice>,
    module: CudaModule,
    precision_mode: PrecisionMode,
    supports_fp16: bool,
    supports_bf16: bool,
    supports_tensor_cores: bool,
}

impl MixedPrecisionKernel {
    /// Create a new mixed precision kernel
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Detect hardware capabilities
        let (supports_fp16, supports_bf16, supports_tensor_cores) = 
            Self::detect_precision_support(&device)?;

        log::info!("Mixed precision support - FP16: {}, BF16: {}, Tensor Cores: {}", 
                  supports_fp16, supports_bf16, supports_tensor_cores);

        // Compile kernels with mixed precision support
        let module = Self::compile_mixed_precision_kernels(&device)?;

        // Select optimal precision mode
        let precision_mode = Self::select_optimal_precision(supports_fp16, supports_bf16, supports_tensor_cores);
        log::info!("Selected precision mode: {:?}", precision_mode);

        Ok(Self {
            device,
            module,
            precision_mode,
            supports_fp16,
            supports_bf16,
            supports_tensor_cores,
        })
    }

    /// Detect hardware precision support
    fn detect_precision_support(device: &CudaDevice) -> Result<(bool, bool, bool)> {
        // Get compute capability
        // Note: cudarc doesn't expose compute capability directly
        // We'll use reasonable assumptions based on modern GPUs
        
        // For now, assume modern GPU capabilities
        // In a real implementation, we would query the actual device properties
        let supports_fp16 = true; // Most modern GPUs support FP16
        let supports_bf16 = true; // Ampere and newer support BF16
        let supports_tensor_cores = true; // Volta and newer have Tensor Cores

        Ok((supports_fp16, supports_bf16, supports_tensor_cores))
    }

    /// Select optimal precision mode based on hardware capabilities
    fn select_optimal_precision(supports_fp16: bool, supports_bf16: bool, supports_tensor_cores: bool) -> PrecisionMode {
        if supports_tensor_cores && supports_bf16 {
            PrecisionMode::BF16 // BF16 often better for training stability
        } else if supports_tensor_cores && supports_fp16 {
            PrecisionMode::FP16 // FP16 for inference
        } else {
            PrecisionMode::FP32 // Fallback to full precision
        }
    }

    /// Compile mixed precision kernels
    fn compile_mixed_precision_kernels(device: &CudaDevice) -> Result<CudaModule> {
        log::info!("Compiling mixed precision CUDA kernels...");

        let kernel_source = include_str!("kernels/mixed_precision_kernels.cu");

        let ptx = compile_ptx_with_opts(
            kernel_source,
            PtxJitOptions {
                arch: Some("sm_75".to_string()), // Target modern architectures
                include_paths: vec![],
                max_register_count: Some(64),
                optimization_level: Some(OptLevel::O3),
                debug: false,
                verbose: false,
                ..Default::default()
            },
        ).map_err(|e| KernelError::GpuError { 
            reason: format!("Failed to compile mixed precision kernels: {}", e) 
        })?;

        let module = device.load_ptx(
            ptx,
            "mixed_precision_kernels",
            &[
                "bitnet_matmul_fp16",
                "bitnet_matmul_bf16", 
                "bitnet_matmul_tensor_core",
                "bitnet_quantize_fp16",
                "bitnet_quantize_bf16",
                "convert_fp32_to_fp16",
                "convert_fp32_to_bf16",
                "convert_fp16_to_fp32",
                "convert_bf16_to_fp32",
            ],
        ).map_err(|e| KernelError::GpuError { 
            reason: format!("Failed to load mixed precision module: {}", e) 
        })?;

        Ok(module)
    }

    /// Get current precision mode
    pub fn precision_mode(&self) -> PrecisionMode {
        self.precision_mode
    }

    /// Set precision mode (if supported by hardware)
    pub fn set_precision_mode(&mut self, mode: PrecisionMode) -> Result<()> {
        match mode {
            PrecisionMode::FP16 if !self.supports_fp16 => {
                return Err(KernelError::GpuError { 
                    reason: "FP16 not supported by hardware".to_string() 
                }.into());
            }
            PrecisionMode::BF16 if !self.supports_bf16 => {
                return Err(KernelError::GpuError { 
                    reason: "BF16 not supported by hardware".to_string() 
                }.into());
            }
            PrecisionMode::Auto => {
                self.precision_mode = Self::select_optimal_precision(
                    self.supports_fp16, 
                    self.supports_bf16, 
                    self.supports_tensor_cores
                );
                return Ok(());
            }
            _ => {}
        }

        self.precision_mode = mode;
        log::info!("Precision mode set to: {:?}", mode);
        Ok(())
    }

    /// Matrix multiplication with automatic precision selection
    pub fn matmul_mixed_precision(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self.precision_mode {
            PrecisionMode::FP32 => self.matmul_fp32(a, b, c, m, n, k),
            PrecisionMode::FP16 => self.matmul_fp16(a, b, c, m, n, k),
            PrecisionMode::BF16 => self.matmul_bf16(a, b, c, m, n, k),
            PrecisionMode::Auto => {
                // Use the automatically selected precision
                match Self::select_optimal_precision(self.supports_fp16, self.supports_bf16, self.supports_tensor_cores) {
                    PrecisionMode::FP16 => self.matmul_fp16(a, b, c, m, n, k),
                    PrecisionMode::BF16 => self.matmul_bf16(a, b, c, m, n, k),
                    _ => self.matmul_fp32(a, b, c, m, n, k),
                }
            }
        }
    }

    /// FP32 matrix multiplication (fallback)
    fn matmul_fp32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Use standard FP32 CUDA BLAS or custom kernel
        log::debug!("Using FP32 matrix multiplication");
        
        // Allocate GPU memory
        let a_gpu = self.device.htod_copy(a)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy matrix A to GPU: {}", e) 
            })?;

        let b_gpu = self.device.htod_copy(b)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy matrix B to GPU: {}", e) 
            })?;

        let mut c_gpu = self.device.alloc_zeros::<f32>(c.len())
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory for result: {}", e) 
            })?;

        // Use cuBLAS for FP32 if available, otherwise use custom kernel
        // For now, we'll use a simple custom kernel approach
        self.launch_fp32_kernel(&a_gpu, &b_gpu, &mut c_gpu, m, n, k)?;

        // Copy result back
        self.device.dtoh_sync_copy_into(&c_gpu, c)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy result from GPU: {}", e) 
            })?;

        Ok(())
    }

    /// FP16 matrix multiplication with Tensor Cores
    fn matmul_fp16(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        log::debug!("Using FP16 matrix multiplication with Tensor Cores");

        // Convert inputs to FP16
        let a_fp16 = self.convert_fp32_to_fp16(a)?;
        let b_fp16 = self.convert_fp32_to_fp16(b)?;

        // Allocate GPU memory for FP16 data
        let a_gpu = self.device.htod_copy(&a_fp16)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy FP16 matrix A to GPU: {}", e) 
            })?;

        let b_gpu = self.device.htod_copy(&b_fp16)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy FP16 matrix B to GPU: {}", e) 
            })?;

        let mut c_gpu_fp16 = self.device.alloc_zeros::<u16>(c.len()) // FP16 as u16
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory for FP16 result: {}", e) 
            })?;

        // Launch FP16 Tensor Core kernel
        self.launch_fp16_tensor_core_kernel(&a_gpu, &b_gpu, &mut c_gpu_fp16, m, n, k)?;

        // Convert result back to FP32
        let c_fp16_host = self.device.dtoh_sync_copy(&c_gpu_fp16)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy FP16 result from GPU: {}", e) 
            })?;

        self.convert_fp16_to_fp32(&c_fp16_host, c)?;

        Ok(())
    }

    /// BF16 matrix multiplication
    fn matmul_bf16(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        log::debug!("Using BF16 matrix multiplication");

        // Convert inputs to BF16
        let a_bf16 = self.convert_fp32_to_bf16(a)?;
        let b_bf16 = self.convert_fp32_to_bf16(b)?;

        // Allocate GPU memory for BF16 data
        let a_gpu = self.device.htod_copy(&a_bf16)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy BF16 matrix A to GPU: {}", e) 
            })?;

        let b_gpu = self.device.htod_copy(&b_bf16)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy BF16 matrix B to GPU: {}", e) 
            })?;

        let mut c_gpu_bf16 = self.device.alloc_zeros::<u16>(c.len()) // BF16 as u16
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory for BF16 result: {}", e) 
            })?;

        // Launch BF16 kernel
        self.launch_bf16_kernel(&a_gpu, &b_gpu, &mut c_gpu_bf16, m, n, k)?;

        // Convert result back to FP32
        let c_bf16_host = self.device.dtoh_sync_copy(&c_gpu_bf16)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy BF16 result from GPU: {}", e) 
            })?;

        self.convert_bf16_to_fp32(&c_bf16_host, c)?;

        Ok(())
    }

    /// Launch FP32 kernel
    fn launch_fp32_kernel(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Simple FP32 kernel launch - in practice would use optimized BLAS
        let block_size = 16;
        let grid_x = (m + block_size - 1) / block_size;
        let grid_y = (n + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };

        // For now, use a placeholder - would implement actual FP32 kernel
        log::debug!("FP32 kernel launch: {}x{}x{}", m, n, k);
        Ok(())
    }

    /// Launch FP16 Tensor Core kernel
    fn launch_fp16_tensor_core_kernel(
        &self,
        a: &CudaSlice<u16>,
        b: &CudaSlice<u16>,
        c: &mut CudaSlice<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.supports_tensor_cores {
            return Err(KernelError::GpuError { 
                reason: "Tensor Cores not supported".to_string() 
            }.into());
        }

        let kernel_func = self.module.get_func("bitnet_matmul_tensor_core")
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to get Tensor Core kernel: {}", e) 
            })?;

        // Tensor Cores work with 16x16 tiles
        let block_size = 16;
        let grid_x = (m + block_size - 1) / block_size;
        let grid_y = (n + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (32, 1, 1), // Warp size for Tensor Cores
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel_func.launch(
                config,
                (a, b, c, m as i32, n as i32, k as i32),
            ).map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to launch Tensor Core kernel: {}", e) 
            })?;
        }

        Ok(())
    }

    /// Launch BF16 kernel
    fn launch_bf16_kernel(
        &self,
        a: &CudaSlice<u16>,
        b: &CudaSlice<u16>,
        c: &mut CudaSlice<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let kernel_func = self.module.get_func("bitnet_matmul_bf16")
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to get BF16 kernel: {}", e) 
            })?;

        let block_size = 16;
        let grid_x = (m + block_size - 1) / block_size;
        let grid_y = (n + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel_func.launch(
                config,
                (a, b, c, m as i32, n as i32, k as i32),
            ).map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to launch BF16 kernel: {}", e) 
            })?;
        }

        Ok(())
    }

    /// Convert FP32 to FP16
    fn convert_fp32_to_fp16(&self, input: &[f32]) -> Result<Vec<u16>> {
        // Simple conversion - in practice would use GPU kernel for large arrays
        let output = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        Ok(output)
    }

    /// Convert FP32 to BF16
    fn convert_fp32_to_bf16(&self, input: &[f32]) -> Result<Vec<u16>> {
        // BF16 conversion: truncate FP32 mantissa
        let output = input.iter()
            .map(|&x| {
                let bits = x.to_bits();
                ((bits >> 16) & 0xFFFF) as u16
            })
            .collect();
        Ok(output)
    }

    /// Convert FP16 to FP32
    fn convert_fp16_to_fp32(&self, input: &[u16], output: &mut [f32]) -> Result<()> {
        for (i, &x) in input.iter().enumerate() {
            output[i] = half::f16::from_bits(x).to_f32();
        }
        Ok(())
    }

    /// Convert BF16 to FP32
    fn convert_bf16_to_fp32(&self, input: &[u16], output: &mut [f32]) -> Result<()> {
        for (i, &x) in input.iter().enumerate() {
            let bits = (x as u32) << 16;
            output[i] = f32::from_bits(bits);
        }
        Ok(())
    }

    /// Get precision-specific memory requirements
    pub fn get_memory_requirements(&self, elements: usize) -> usize {
        match self.precision_mode {
            PrecisionMode::FP32 => elements * 4,
            PrecisionMode::FP16 | PrecisionMode::BF16 => elements * 2,
            PrecisionMode::Auto => {
                match Self::select_optimal_precision(self.supports_fp16, self.supports_bf16, self.supports_tensor_cores) {
                    PrecisionMode::FP16 | PrecisionMode::BF16 => elements * 2,
                    _ => elements * 4,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_conversion() {
        let input = vec![1.0f32, -2.5f32, 0.0f32, 3.14159f32];
        
        // Test FP16 conversion
        let fp16_data = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect::<Vec<u16>>();
        
        let mut fp32_result = vec![0.0f32; input.len()];
        for (i, &x) in fp16_data.iter().enumerate() {
            fp32_result[i] = half::f16::from_bits(x).to_f32();
        }
        
        // Check that conversion is reasonably accurate
        for (original, converted) in input.iter().zip(fp32_result.iter()) {
            let error = (original - converted).abs();
            assert!(error < 0.01, "FP16 conversion error too large: {} vs {}", original, converted);
        }
    }

    #[test]
    fn test_bf16_conversion() {
        let input = vec![1.0f32, -2.5f32, 0.0f32, 3.14159f32];
        
        // Test BF16 conversion
        let bf16_data = input.iter()
            .map(|&x| {
                let bits = x.to_bits();
                ((bits >> 16) & 0xFFFF) as u16
            })
            .collect::<Vec<u16>>();
        
        let mut fp32_result = vec![0.0f32; input.len()];
        for (i, &x) in bf16_data.iter().enumerate() {
            let bits = (x as u32) << 16;
            fp32_result[i] = f32::from_bits(bits);
        }
        
        // BF16 has lower precision than FP16 but same range as FP32
        for (original, converted) in input.iter().zip(fp32_result.iter()) {
            let error = (original - converted).abs();
            assert!(error < 0.1, "BF16 conversion error too large: {} vs {}", original, converted);
        }
    }
}