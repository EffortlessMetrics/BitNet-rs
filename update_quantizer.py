import re

file_path = 'crates/bitnet-quantization/src/device_aware_quantizer.rs'

with open(file_path, 'r') as f:
    content = f.read()

# 1. Add imports
imports_search = r"use bitnet_common::{Device, QuantizationError, Result};"
imports_replace = """use bitnet_common::{Device, QuantizationError, Result};
#[cfg(any(feature = "gpu", feature = "cuda"))]
use bitnet_kernels::{CudaKernel, KernelProvider};
use std::sync::Arc;"""

content = content.replace(imports_search, imports_replace)

# 2. Update GPUQuantizer struct
struct_search = r"""pub struct GPUQuantizer {
    #[allow(dead_code)]
    tolerance_config: ToleranceConfig,
    #[allow(dead_code)]
    device_id: usize,
}"""
struct_replace = """pub struct GPUQuantizer {
    #[allow(dead_code)]
    tolerance_config: ToleranceConfig,
    #[allow(dead_code)]
    device_id: usize,
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    kernel: Option<Arc<CudaKernel>>,
}"""

content = content.replace(struct_search, struct_replace)

# 3. Update GPUQuantizer impl
impl_search = r"""impl GPUQuantizer {
    pub fn new(tolerance_config: ToleranceConfig, device_id: usize) -> Self {
        Self { tolerance_config, device_id }
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub fn quantize_i2s(&self, data: &[f32]) -> Result<QuantizedTensor> {
        debug!("Performing I2S quantization on GPU:{}", self.device_id);

        // For now, fall back to CPU implementation
        // In a real implementation, this would use CUDA kernels
        let cpu_quantizer = CPUQuantizer::new(self.tolerance_config.clone());
        cpu_quantizer.quantize_i2s(data)
    }"""

impl_replace = r"""impl GPUQuantizer {
    pub fn new(tolerance_config: ToleranceConfig, device_id: usize) -> Self {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let kernel = match CudaKernel::new_with_device(device_id) {
                Ok(k) => Some(Arc::new(k)),
                Err(e) => {
                    warn!("Failed to initialize CUDA kernel for device {}: {:?}, GPU quantization will fallback to CPU", device_id, e);
                    None
                }
            };
            Self { tolerance_config, device_id, kernel }
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            Self { tolerance_config, device_id }
        }
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub fn quantize_i2s(&self, data: &[f32]) -> Result<QuantizedTensor> {
        debug!("Performing I2S quantization on GPU:{}", self.device_id);

        if let Some(kernel) = &self.kernel {
            let len = data.len();
            // Block size 32 to match CPU implementation and common CUDA warp size
            let block_size = 32;
            let num_blocks = len.div_ceil(block_size);

            let mut output = vec![0u8; len.div_ceil(4)]; // 2 bits per element = 4 elements per byte
            let mut scales = vec![0.0f32; num_blocks];

            match kernel.quantize(data, &mut output, &mut scales, QuantizationType::I2S) {
                Ok(_) => {
                    return Ok(QuantizedTensor::new(
                        output,
                        QuantizationType::I2S,
                        vec![len],
                        scales,
                        block_size,
                    ));
                }
                Err(e) => {
                    warn!("CUDA quantization failed: {:?}, falling back to CPU", e);
                }
            }
        } else {
            warn!("CUDA kernel not available, falling back to CPU");
        }

        // Fall back to CPU implementation
        let cpu_quantizer = CPUQuantizer::new(self.tolerance_config.clone());
        cpu_quantizer.quantize_i2s(data)
    }"""

content = content.replace(impl_search, impl_replace)

# 4. Update GPUQuantizer::dequantize_i2s to use kernel (optional, but good for completeness if kernel supports it)
# The kernel interface in bitnet-kernels/src/lib.rs does NOT list a dequantize method in KernelProvider.
# It only has matmul_i2s and quantize.
# So we will leave dequantize_i2s as CPU fallback for now, as intended by the current kernel capabilities.

with open(file_path, 'w') as f:
    f.write(content)

print("Successfully updated device_aware_quantizer.rs")
