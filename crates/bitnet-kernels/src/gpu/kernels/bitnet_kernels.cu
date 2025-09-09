/**
 * CUDA kernels for BitNet operations
 * 
 * This file contains optimized CUDA kernels for:
 * - Matrix multiplication with I2S quantization
 * - Quantization operations (I2S, TL1, TL2)
 * - Dequantization operations
 */

// Type definitions for NVRTC compilation
typedef signed char int8_t;
typedef unsigned char uint8_t;

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
    
    // Fixed block dimensions for shared memory
    const int BLOCK_SIZE = 16;
    
    // Global indices
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    // Shared memory for tiles (use fixed size)
    __shared__ int8_t As[16][16];
    __shared__ uint8_t Bs[16][16];
    
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
        
        // Load B tile (B is unpacked u8, same as CPU)
        if (b_row < K && b_col < N) {
            // B is already unpacked u8 values, just load directly
            uint8_t b_val = B[b_row * N + b_col];
            // Convert u8 to signed for computation (matches CPU)
            Bs[ty][tx] = (int8_t)b_val;
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
 * I2S Quantization Kernel - CPU-compatible version
 * Quantizes float32 values to 2-bit signed integers with identical CPU scaling
 * 
 * Key fixes for CPU parity:
 * - Uses per-block max computation (32-element blocks like CPU)
 * - Identical scaling formula: max_val / 1.5  
 * - Deterministic bit packing with proper synchronization
 * - CPU-compatible quantization thresholds and bit encoding
 */
extern "C" __global__ void bitnet_quantize_i2s(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    const int BLOCK_SIZE = 32; // Match CPU block size exactly
    
    // Use a grid where each block processes multiple CPU-sized blocks
    int tid = threadIdx.x;
    int cpu_blocks_per_cuda_block = (blockDim.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Each CUDA block processes multiple CPU blocks
    for (int cpu_block = 0; cpu_block < cpu_blocks_per_cuda_block; cpu_block++) {
        int block_start = (blockIdx.x * cpu_blocks_per_cuda_block + cpu_block) * BLOCK_SIZE;
        if (block_start >= N) break;
        
        int block_end = min(block_start + BLOCK_SIZE, N);
        int block_idx = block_start / BLOCK_SIZE;
        
        // Step 1: Find maximum value in this CPU block
        float local_max = 0.0f;
        if (block_start + tid < block_end) {
            local_max = fabsf(input[block_start + tid]);
        }
        
        // Reduction to find block maximum
        __shared__ float shared_max[1024];
        shared_max[tid] = local_max;
        __syncthreads();
        
        // Parallel reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            }
            __syncthreads();
        }
        
        float block_max = shared_max[0];
        
        // Step 2: Compute scale using CPU-identical formula
        float block_scale = (block_max > 1e-8f) ? (block_max / 1.5f) : 1.0f;
        
        // Store scale (only first thread)
        if (tid == 0 && block_idx < (N + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            scales[block_idx] = block_scale;
        }
        __syncthreads();
        
        // Step 3: Quantize values in this block
        if (block_start + tid < block_end) {
            float normalized = input[block_start + tid] / block_scale;
            
            // CPU-identical quantization thresholds
            int8_t quantized;
            if (normalized > 0.5f) {
                quantized = 1;   // +1
            } else if (normalized < -0.5f) {
                quantized = 3;   // -1 (CPU uses 3i8 for -1 in 2-bit representation)
            } else {
                quantized = 0;   // 0
            }
            
            // Step 4: Deterministic bit packing
            int global_idx = block_start + tid;
            int pack_idx = global_idx / 4;
            int bit_offset = (global_idx % 4) * 2;
            
            // Only the first thread of each group of 4 handles packing
            if ((tid % 4) == 0 && pack_idx < (N + 3) / 4) {
                uint8_t packed_byte = 0;
                
                // Pack up to 4 values into this byte
                for (int j = 0; j < 4 && (pack_idx * 4 + j) < N; j++) {
                    int src_idx = pack_idx * 4 + j;
                    if (src_idx < block_end && src_idx >= block_start) {
                        float norm_val = input[src_idx] / block_scale;
                        
                        uint8_t val;
                        if (norm_val > 0.5f) {
                            val = 1;  // +1
                        } else if (norm_val < -0.5f) {
                            val = 3;  // -1 (CPU format)
                        } else {
                            val = 0;  // 0
                        }
                        
                        packed_byte |= (val << (j * 2));
                    }
                }
                
                output[pack_idx] = packed_byte;
            }
        }
        __syncthreads();
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
    atomicOr((unsigned int*)&output[pack_idx], (unsigned int)(quantized << bit_offset));
    
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
    atomicOr((unsigned int*)&output[pack_idx], (unsigned int)(quantized << bit_offset));
    
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
    
    // Get scale - use CPU-compatible block size of 32
    float scale = scales[idx / 32]; // Match CPU block size
    
    // Dequantize based on type
    float dequantized;
    switch (quantization_type) {
        case 0: // I2S - CPU-compatible format
            switch (quantized) {
                case 0: dequantized = 0.0f; break;  // 0 -> 0
                case 1: dequantized = 1.0f; break;  // 1 -> +1
                case 3: dequantized = -1.0f; break; // 3 -> -1 (CPU format)
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
 * Mixed Precision Matrix Multiplication (FP32)
 * Standard matrix multiplication for completeness
 */
extern "C" __global__ void bitnet_matmul_fp32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}