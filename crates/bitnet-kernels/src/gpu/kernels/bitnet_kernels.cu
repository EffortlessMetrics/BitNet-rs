/**
 * CUDA kernels for BitNet operations
 * 
 * This file contains optimized CUDA kernels for:
 * - Matrix multiplication with I2S quantization
 * - Quantization operations (I2S, TL1, TL2)
 * - Dequantization operations
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// Constants
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define SHARED_MEM_BANK_SIZE 32

/**
 * I2S Matrix Multiplication Kernel
 * Performs C = A * B where A is int8, B is uint8 (2-bit packed), C is float32
 */
extern "C" __global__ void bitnet_matmul_i2s(
    const int8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Block dimensions
    const int BLOCK_SIZE = 16;
    
    // Global indices
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    // Shared memory for tiles
    __shared__ int8_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint8_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tiles into shared memory
        int a_row = row;
        int a_col = t * BLOCK_SIZE + tx;
        int b_row = t * BLOCK_SIZE + ty;
        int b_col = col;
        
        // Load A tile
        if (a_row < M && a_col < K) {
            As[ty][tx] = A[a_row * K + a_col];
        } else {
            As[ty][tx] = 0;
        }
        
        // Load B tile (handle 2-bit unpacking)
        if (b_row < K && b_col < N) {
            // B is packed with 4 2-bit values per byte
            int b_idx = b_row * N + b_col;
            uint8_t packed = B[b_idx / 4];
            int shift = (b_idx % 4) * 2;
            uint8_t unpacked = (packed >> shift) & 0x3;
            
            // Convert 2-bit unsigned to signed (-1, 0, 1)
            int8_t signed_val;
            switch (unpacked) {
                case 0: signed_val = -1; break;
                case 1: signed_val = 0; break;
                case 2: signed_val = 1; break;
                default: signed_val = 0; break;
            }
            Bs[ty][tx] = signed_val;
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += (float)As[ty][k] * (float)Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * I2S Quantization Kernel
 * Quantizes float32 values to 2-bit signed integers with scaling
 */
extern "C" __global__ void bitnet_quantize_i2s(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = 128; // Process 128 elements per block for scale computation
    int block_idx = idx / block_size;
    int local_idx = idx % block_size;
    
    if (idx >= N) return;
    
    // Shared memory for reduction
    __shared__ float shared_max[32]; // Assuming max 1024 threads per block
    
    // Find maximum absolute value in the block for scaling
    float local_max = 0.0f;
    for (int i = block_idx * block_size; i < min((block_idx + 1) * block_size, N); i += blockDim.x) {
        if (i + threadIdx.x < N) {
            local_max = fmaxf(local_max, fabsf(input[i + threadIdx.x]));
        }
    }
    
    // Reduce to find block maximum
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    float scale = shared_max[0];
    if (scale == 0.0f) scale = 1.0f; // Avoid division by zero
    
    // Store scale (only first thread in block)
    if (threadIdx.x == 0) {
        scales[block_idx] = scale;
    }
    
    // Quantize values
    if (idx < N) {
        float normalized = input[idx] / scale;
        int8_t quantized;
        
        if (normalized > 0.5f) {
            quantized = 1;
        } else if (normalized < -0.5f) {
            quantized = -1;
        } else {
            quantized = 0;
        }
        
        // Pack 4 2-bit values into one byte
        int pack_idx = idx / 4;
        int bit_offset = (idx % 4) * 2;
        
        // Convert signed to unsigned for packing
        uint8_t unsigned_val;
        switch (quantized) {
            case -1: unsigned_val = 0; break;
            case 0: unsigned_val = 1; break;
            case 1: unsigned_val = 2; break;
            default: unsigned_val = 1; break;
        }
        
        // Atomic update to pack bits
        atomicOr(&output[pack_idx], unsigned_val << bit_offset);
    }
}

/**
 * TL1 Quantization Kernel (ARM-optimized lookup table)
 * Optimized for ARM NEON-like operations but implemented in CUDA
 */
extern "C" __global__ void bitnet_quantize_tl1(
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
    atomicOr(&output[pack_idx], quantized << bit_offset);
    
    // Store scale
    if (idx == 0) {
        scales[0] = scale;
    }
}

/**
 * TL2 Quantization Kernel (x86-optimized lookup table)
 * Optimized for x86 AVX-like operations but implemented in CUDA
 */
extern "C" __global__ void bitnet_quantize_tl2(
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
    atomicOr(&output[pack_idx], quantized << bit_offset);
    
    // Store scale
    if (idx == 0) {
        scales[0] = scale;
    }
}

/**
 * Dequantization Kernel
 * Converts quantized values back to float32
 */
extern "C" __global__ void bitnet_dequantize(
    const uint8_t* __restrict__ input,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int N,
    int quantization_type // 0=I2S, 1=TL1, 2=TL2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    // Unpack 2-bit value
    int pack_idx = idx / 4;
    int bit_offset = (idx % 4) * 2;
    uint8_t packed = input[pack_idx];
    uint8_t quantized = (packed >> bit_offset) & 0x3;
    
    // Get scale
    float scale = scales[idx / 128]; // Assuming block size of 128
    
    // Dequantize based on type
    float dequantized;
    switch (quantization_type) {
        case 0: // I2S
            switch (quantized) {
                case 0: dequantized = -1.0f; break;
                case 1: dequantized = 0.0f; break;
                case 2: dequantized = 1.0f; break;
                default: dequantized = 0.0f; break;
            }
            break;
        case 1: // TL1
            dequantized = (float)quantized / 3.0f * 2.0f - 1.0f;
            break;
        case 2: // TL2
            dequantized = (float)quantized / 2.0f * 2.0f - 1.0f;
            break;
        default:
            dequantized = 0.0f;
            break;
    }
    
    output[idx] = dequantized * scale;
}

/**
 * Mixed Precision Matrix Multiplication (FP16)
 * High-performance matrix multiplication using Tensor Cores when available
 */
extern "C" __global__ void bitnet_matmul_fp16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    // Use Tensor Cores if available (compute capability 7.0+)
    #if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    
    // Tensor Core fragment declarations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Compute tile indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Perform matrix multiplication using Tensor Cores
    for (int i = 0; i < K; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;
        
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
    #else
    // Fallback for older architectures
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        __half sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    #endif
}