import re

file_path = 'crates/bitnet-kernels/src/gpu/kernels/bitnet_kernels.cu'

with open(file_path, 'r') as f:
    content = f.read()

# Fix TL1
tl1_search = r"""extern "C" __global__ void bitnet_quantize_tl1(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // TL1 uses lookup tables - simplified version for CUDA
    // In practice, this would use precomputed lookup tables
    float value = input[idx];
    float scale = 1.0f; // Simplified scaling

    // Quantize to 2-bit using lookup table approach
    float normalized = value / scale;
    uint8_t quantized;

    if (normalized > 0.75f) {
        quantized = 3;
    } else if (normalized > 0.25f) {
        quantized = 2;
    } else if (normalized > -0.25f) {
        quantized = 1;
    } else {
        quantized = 0;
    }

    // Pack into output
    int pack_idx = idx / 4;
    int bit_offset = (idx % 4) * 2;
    atomicOr((unsigned int*)&output[pack_idx], (unsigned int)(quantized << bit_offset));

    // Store scale
    if (idx == 0) {
        scales[0] = scale;
    }
}"""

tl1_replace = r"""extern "C" __global__ void bitnet_quantize_tl1(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Store scale (only once)
    if (idx == 0) {
        scales[0] = 1.0f;
    }

    // Only every 4th thread acts to pack 4 values into 1 byte
    if (idx % 4 == 0 && idx < N) {
        uint8_t packed_byte = 0;
        float scale = 1.0f;

        for (int j = 0; j < 4; j++) {
            if (idx + j < N) {
                float value = input[idx + j];
                float normalized = value / scale;
                uint8_t quantized;

                if (normalized > 0.75f) {
                    quantized = 3;
                } else if (normalized > 0.25f) {
                    quantized = 2;
                } else if (normalized > -0.25f) {
                    quantized = 1;
                } else {
                    quantized = 0;
                }

                packed_byte |= (quantized << (j * 2));
            }
        }
        output[idx / 4] = packed_byte;
    }
}"""

# Fix TL2
tl2_search = r"""extern "C" __global__ void bitnet_quantize_tl2(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // TL2 uses different lookup tables optimized for x86
    float value = input[idx];
    float scale = 1.0f; // Simplified scaling

    // Quantize using TL2 lookup approach
    float normalized = value / scale;
    uint8_t quantized;

    // Different quantization thresholds for TL2
    if (normalized > 0.5f) {
        quantized = 2;
    } else if (normalized > -0.5f) {
        quantized = 1;
    } else {
        quantized = 0;
    }

    // Pack into output
    int pack_idx = idx / 4;
    int bit_offset = (idx % 4) * 2;
    atomicOr((unsigned int*)&output[pack_idx], (unsigned int)(quantized << bit_offset));

    // Store scale
    if (idx == 0) {
        scales[0] = scale;
    }
}"""

tl2_replace = r"""extern "C" __global__ void bitnet_quantize_tl2(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Store scale
    if (idx == 0) {
        scales[0] = 1.0f;
    }

    // Only every 4th thread acts
    if (idx % 4 == 0 && idx < N) {
        uint8_t packed_byte = 0;
        float scale = 1.0f;

        for (int j = 0; j < 4; j++) {
            if (idx + j < N) {
                float value = input[idx + j];
                float normalized = value / scale;
                uint8_t quantized;

                // Different quantization thresholds for TL2
                if (normalized > 0.5f) {
                    quantized = 2;
                } else if (normalized > -0.5f) {
                    quantized = 1;
                } else {
                    quantized = 0;
                }

                packed_byte |= (quantized << (j * 2));
            }
        }
        output[idx / 4] = packed_byte;
    }
}"""

content = content.replace(tl1_search, tl1_replace)
content = content.replace(tl2_search, tl2_replace)

with open(file_path, 'w') as f:
    f.write(content)

print("Updated kernels")
