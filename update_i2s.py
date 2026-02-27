import re

file_path = 'crates/bitnet-quantization/src/i2s.rs'

with open(file_path, 'r') as f:
    content = f.read()

# Fix import in quantize_cuda_with_limits
search = r"use bitnet_kernels::gpu::cuda::CudaKernel;"
replace = r"use bitnet_kernels::{gpu::cuda::CudaKernel, KernelProvider};"

if search in content:
    content = content.replace(search, replace)
    with open(file_path, 'w') as f:
        f.write(content)
    print("Updated i2s.rs")
else:
    print("Could not find import to replace in i2s.rs")
