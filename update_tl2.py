import re

file_path = 'crates/bitnet-quantization/src/tl2.rs'

with open(file_path, 'r') as f:
    content = f.read()

# Fix import in quantize_cuda
search = r"use bitnet_kernels::gpu::cuda::CudaKernel;"
replace = r"use bitnet_kernels::{gpu::cuda::CudaKernel, KernelProvider};"

if search in content:
    content = content.replace(search, replace)
    with open(file_path, 'w') as f:
        f.write(content)
    print("Updated tl2.rs")
else:
    print("Could not find import to replace in tl2.rs")
