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
 * I2S Quantization Kernel - CPU-compatible version
 * Quantizes float32 values to 2-bit signed integers with identical CPU scaling
 * 
 * Key changes for CPU parity:
 * - Uses global max computation (32-element blocks like CPU)
 * - Identical scaling formula: max_val / 1.5
 * - Deterministic bit packing without atomics
 * - CPU-compatible quantization thresholds
 */
extern "C" __global__ void bitnet_quantize_i2s(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int N
) {
    const int BLOCK_SIZE = 32; // Match CPU block size exactly
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = global_idx / BLOCK_SIZE;
    int local_idx = global_idx % BLOCK_SIZE;
    
    if (global_idx >= N) return;
    
    // Step 1: Global max computation using two-phase reduction
    // Phase 1 - Find local max across entire input
    __shared__ float shared_max[1024]; // Max threads per block
    float thread_max = 0.0f;
    
    // Each thread processes multiple elements if needed
    for (int i = global_idx; i < N; i += gridDim.x * blockDim.x) {
        thread_max = fmaxf(thread_max, fabsf(input[i]));
    }
    
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Phase 2 - Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // Global max is now in shared_max[0]
    float global_max = shared_max[0];
    
    // Step 2: Compute block scale using CPU-identical formula
    int start_idx = block_idx * BLOCK_SIZE;
    int end_idx = min(start_idx + BLOCK_SIZE, N);
    
    float block_max = 0.0f;
    if (start_idx + local_idx < end_idx) {
        block_max = fabsf(input[start_idx + local_idx]);
    }
    
    // Find max in this specific block
    __shared__ float block_shared_max[32]; // One per local index
    if (local_idx < 32) {
        block_shared_max[local_idx] = block_max;
    }
    __syncthreads();
    
    // Reduce block max
    for (int s = 16; s > 0; s >>= 1) {
        if (local_idx < s && local_idx + s < 32) {
            block_shared_max[local_idx] = fmaxf(block_shared_max[local_idx], block_shared_max[local_idx + s]);
        }
        __syncthreads();
    }
    
    float block_scale;
    if (local_idx == 0) {
        float max_val = block_shared_max[0];
        // CPU-identical scaling formula
        block_scale = (max_val > 1e-8f) ? (max_val / 1.5f) : 1.0f;
        scales[block_idx] = block_scale;
    }
    __syncthreads();
    
    // Broadcast scale to all threads in this block
    block_scale = scales[block_idx];
    
    // Step 3: Quantize with CPU-identical thresholds
    if (global_idx < N) {
        float normalized = input[global_idx] / block_scale;
        int8_t quantized;
        
        // CPU-identical quantization boundaries
        if (normalized > 0.5f) {
            quantized = 1;   // +1
        } else if (normalized < -0.5f) {
            quantized = 3;   // -1 (CPU uses 3i8 for -1)
        } else {
            quantized = 0;   // 0
        }
        
        // Step 4: Deterministic bit packing (no atomics)
        int pack_idx = global_idx / 4;
        int bit_offset = (global_idx % 4) * 2;
        
        // Convert to CPU-compatible unsigned values
        uint8_t unsigned_val;
        switch (quantized) {
            case 3:  unsigned_val = 3; break; // -1 -> 3 (match CPU)
            case 0:  unsigned_val = 0; break; // 0 -> 0  
            case 1:  unsigned_val = 1; break; // +1 -> 1
            default: unsigned_val = 0; break;
        }
        
        // Deterministic packing: only one thread writes per byte position
        if ((global_idx % 4) == 0 && pack_idx < (N + 3) / 4) {
            // Clear the byte first
            output[pack_idx] = 0;
            
            // Pack all 4 values for this byte position
            for (int j = 0; j < 4 && (pack_idx * 4 + j) < N; j++) {
                int src_idx = pack_idx * 4 + j;
                float norm_val = input[src_idx] / block_scale;
                
                uint8_t val;
                if (norm_val > 0.5f) {
                    val = 1;  // +1
                } else if (norm_val < -0.5f) {
                    val = 3;  // -1 (CPU format)
                } else {
                    val = 0;  // 0
                }
                
                output[pack_idx] |= (val << (j * 2));
            }
        }
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