import re

file_path = 'crates/bitnet-quantization/src/device_aware_quantizer.rs'

with open(file_path, 'r') as f:
    content = f.read()

# 1. Remove Derive Debug from GPUQuantizer
struct_def = r"#[derive(Debug, Clone)]\npub struct GPUQuantizer"
struct_def_new = r"#[derive(Clone)]\npub struct GPUQuantizer"
content = content.replace(struct_def, struct_def_new)

# 2. Add manual Debug impl for GPUQuantizer
# We'll add it after the struct definition
struct_end = r"    #[cfg(any(feature = \"gpu\", feature = \"cuda\"))]\n    kernel: Option<Arc<CudaKernel>>,\n}"
debug_impl = r"""    #[cfg(any(feature = "gpu", feature = "cuda"))]
    kernel: Option<Arc<CudaKernel>>,
}

impl std::fmt::Debug for GPUQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("GPUQuantizer");
        d.field("tolerance_config", &self.tolerance_config)
         .field("device_id", &self.device_id);
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        d.field("kernel", &if self.kernel.is_some() { "Some(CudaKernel)" } else { "None" });
        d.finish()
    }
}"""

# Use a more robust replace for the struct end/impl insertion
if struct_end in content:
    content = content.replace(struct_end, debug_impl)
else:
    print("Could not find struct end to insert Debug impl")
    # Fallback to replace the struct definition block completely if regex fails
    # (Doing a simpler replacement strategy might be safer)

# 3. Fix QuantizationType mismatch
# We need to map local QuantizationType to bitnet_common::QuantizationType
# The call site is: match kernel.quantize(data, &mut output, &mut scales, QuantizationType::I2S)
# We need to change QuantizationType::I2S to bitnet_common::QuantizationType::I2S
call_site = r"match kernel.quantize(data, &mut output, &mut scales, QuantizationType::I2S)"
call_site_fixed = r"match kernel.quantize(data, &mut output, &mut scales, bitnet_common::QuantizationType::I2S)"
content = content.replace(call_site, call_site_fixed)

with open(file_path, 'w') as f:
    f.write(content)
