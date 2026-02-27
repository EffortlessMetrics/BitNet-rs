import re

file_path = 'crates/bitnet-quantization/src/device_aware_quantizer.rs'

with open(file_path, 'r') as f:
    content = f.read()

# 1. Update struct definition and add Debug impl
search_struct = r"""/// GPU quantizer implementation
#[derive(Debug, Clone)]
pub struct GPUQuantizer {
    #[allow(dead_code)]
    tolerance_config: ToleranceConfig,
    #[allow(dead_code)]
    device_id: usize,
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    kernel: Option<Arc<CudaKernel>>,
}

impl GPUQuantizer {"""

replace_struct = r"""/// GPU quantizer implementation
#[derive(Clone)]
pub struct GPUQuantizer {
    #[allow(dead_code)]
    tolerance_config: ToleranceConfig,
    #[allow(dead_code)]
    device_id: usize,
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    kernel: Option<Arc<CudaKernel>>,
}

impl std::fmt::Debug for GPUQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("GPUQuantizer");
        d.field("tolerance_config", &self.tolerance_config)
            .field("device_id", &self.device_id);
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        if self.kernel.is_some() {
            let _ = d.field("kernel", &"Some(CudaKernel)");
        } else {
            let _ = d.field("kernel", &"None");
        }
        d.finish()
    }
}

impl GPUQuantizer {"""

# Replace struct definition
content = content.replace(search_struct, replace_struct)

# 2. Fix QuantizationType mismatch
search_call = r"match kernel.quantize(data, &mut output, &mut scales, QuantizationType::I2S) {"
replace_call = r"match kernel.quantize(data, &mut output, &mut scales, bitnet_common::QuantizationType::I2S) {"

content = content.replace(search_call, replace_call)

with open(file_path, 'w') as f:
    f.write(content)

print("Applied manual fix")
