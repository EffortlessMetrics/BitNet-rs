#include <cuda_runtime.h>
#include <math.h>

// I2S quantization: block size 32, values -1/0/1 encoded as 3/0/1 respectively
extern "C" __global__ void bitnet_quantize_i2s(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scales,
    int n
) {
    const int BLOCK_SIZE = 32;
    int block_start = blockIdx.x * BLOCK_SIZE;
    if (block_start >= n) return;
    int block_end = block_start + BLOCK_SIZE;
    if (block_end > n) block_end = n;

    float scale;
    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int i = block_start; i < block_end; ++i) {
            float v = fabsf(input[i]);
            if (v > max_val) max_val = v;
        }
        scale = max_val > 1e-8f ? max_val / 1.5f : 1.0f;
        scales[blockIdx.x] = scale;
    }
    __syncthreads();
    scale = scales[blockIdx.x];

    for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        float normalized = input[i] / scale;
        unsigned char q;
        if (normalized > 0.5f) {
            q = 1;
        } else if (normalized < -0.5f) {
            q = 3;
        } else {
            q = 0;
        }
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        atomicOr(&output[byte_idx], q << bit_offset);
    }
}

// TL1 quantization: block size 64 with lookup table
extern "C" __global__ void bitnet_quantize_tl1(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scales,
    int n
) {
    const int BLOCK_SIZE = 64;
    int block_start = blockIdx.x * BLOCK_SIZE;
    if (block_start >= n) return;
    int block_end = block_start + BLOCK_SIZE;
    if (block_end > n) block_end = n;

    float scale;
    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int i = block_start; i < block_end; ++i) {
            float v = fabsf(input[i]);
            if (v > max_val) max_val = v;
        }
        scale = max_val > 1e-8f ? max_val / 1.5f : 1.0f;
        scales[blockIdx.x] = scale;
    }
    __syncthreads();
    scale = scales[blockIdx.x];

    const float lut[4] = {-1.0f, -0.33f, 0.33f, 1.0f};
    for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        float normalized = input[i] / scale;
        int best_idx = 0;
        float best_dist = fabsf(normalized - lut[0]);
        for (int j = 1; j < 4; ++j) {
            float dist = fabsf(normalized - lut[j]);
            if (dist < best_dist) { best_dist = dist; best_idx = j; }
        }
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        atomicOr(&output[byte_idx], ((unsigned char)best_idx) << bit_offset);
    }
}

// TL2 quantization: block size 128 with different lookup table
extern "C" __global__ void bitnet_quantize_tl2(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scales,
    int n
) {
    const int BLOCK_SIZE = 128;
    int block_start = blockIdx.x * BLOCK_SIZE;
    if (block_start >= n) return;
    int block_end = block_start + BLOCK_SIZE;
    if (block_end > n) block_end = n;

    float scale;
    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int i = block_start; i < block_end; ++i) {
            float v = fabsf(input[i]);
            if (v > max_val) max_val = v;
        }
        scale = max_val > 1e-8f ? max_val / 1.5f : 1.0f;
        scales[blockIdx.x] = scale;
    }
    __syncthreads();
    scale = scales[blockIdx.x];

    const float lut[4] = {-1.2f, -0.4f, 0.4f, 1.2f};
    for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        float normalized = input[i] / scale;
        int best_idx = 0;
        float best_dist = fabsf(normalized - lut[0]);
        for (int j = 1; j < 4; ++j) {
            float dist = fabsf(normalized - lut[j]);
            if (dist < best_dist) { best_dist = dist; best_idx = j; }
        }
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        atomicOr(&output[byte_idx], ((unsigned char)best_idx) << bit_offset);
    }
}

