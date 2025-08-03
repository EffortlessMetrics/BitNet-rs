/**
 * Mixed precision CUDA kernels for BitNet operations
 * 
 * This file contains optimized CUDA kernels for:
 * - FP16 matrix multiplication with Tensor Cores
 * - BF16 matrix multiplication
 * - Precision conversion utilities
 * - Memory-optimized operations
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

// Tensor Core matrix multiplication using WMMA API
extern "C" __global__ void bitnet_matmul_tensor_core(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    // Only available on compute capability 7.0+
    #if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    
    // WMMA fragment declarations for 16x16x16 operations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    
    // Calculate warp and lane indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;
    
    // Each warp computes a 16x16 tile
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension in chunks of 16
    for (int i = 0; i < K; i += 16) {
        int aRow = cRow;
        int aCol = i;
        int bRow = i;
        int bCol = cCol;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrix fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiplication and accumulation
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store the result
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
    #endif
}

// FP16 matrix multiplication (fallback for older architectures)
extern "C" __global__ void bitnet_matmul_fp16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        __half sum = __float2half(0.0f);
        
        for (int k = 0; k < K; ++k) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        
        C[row * N + col] = sum;
    }
}

// BF16 matrix multiplication
extern "C" __global__ void bitnet_matmul_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    #if __CUDA_ARCH__ >= 800  // BF16 support requires Ampere or newer
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        __nv_bfloat16 sum = __float2bfloat16(0.0f);
        
        for (int k = 0; k < K; ++k) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        
        C[row * N + col] = sum;
    }
    #endif
}

// Optimized FP16 quantization
extern "C" __global__ void bitnet_quantize_fp16(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    __half* __restrict__ scales,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = 128;
    int block_idx = idx / block_size;
    
    if (idx >= N) return;
    
    // Shared memory for reduction
    __shared__ __half shared_max[32];
    
    // Find maximum absolute value in the block
    __half local_max = __float2half(0.0f);
    for (int i = block_idx * block_size; i < min((block_idx + 1) * block_size, N); i += blockDim.x) {
        if (i + threadIdx.x < N) {
            __half val = input[i + threadIdx.x];
            local_max = __hmax(local_max, __habs(val));
        }
    }
    
    // Reduction in shared memory
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            shared_max[threadIdx.x] = __hmax(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    __half scale = shared_max[0];
    if (__heq(scale, __float2half(0.0f))) {
        scale = __float2half(1.0f);
    }
    
    // Store scale
    if (threadIdx.x == 0) {
        scales[block_idx] = scale;
    }
    
    // Quantize values
    if (idx < N) {
        __half normalized = __hdiv(input[idx], scale);
        int8_t quantized;
        
        float norm_f = __half2float(normalized);
        if (norm_f > 0.5f) {
            quantized = 1;
        } else if (norm_f < -0.5f) {
            quantized = -1;
        } else {
            quantized = 0;
        }
        
        // Pack into output
        int pack_idx = idx / 4;
        int bit_offset = (idx % 4) * 2;
        
        uint8_t unsigned_val;
        switch (quantized) {
            case -1: unsigned_val = 0; break;
            case 0: unsigned_val = 1; break;
            case 1: unsigned_val = 2; break;
            default: unsigned_val = 1; break;
        }
        
        atomicOr(&output[pack_idx], unsigned_val << bit_offset);
    }
}

// BF16 quantization
extern "C" __global__ void bitnet_quantize_bf16(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    __nv_bfloat16* __restrict__ scales,
    int N
) {
    #if __CUDA_ARCH__ >= 800
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = 128;
    int block_idx = idx / block_size;
    
    if (idx >= N) return;
    
    // Similar to FP16 quantization but using BF16 operations
    __shared__ __nv_bfloat16 shared_max[32];
    
    __nv_bfloat16 local_max = __float2bfloat16(0.0f);
    for (int i = block_idx * block_size; i < min((block_idx + 1) * block_size, N); i += blockDim.x) {
        if (i + threadIdx.x < N) {
            __nv_bfloat16 val = input[i + threadIdx.x];
            local_max = __hmax(local_max, __habs(val));
        }
    }
    
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            shared_max[threadIdx.x] = __hmax(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    __nv_bfloat16 scale = shared_max[0];
    if (__heq(scale, __float2bfloat16(0.0f))) {
        scale = __float2bfloat16(1.0f);
    }
    
    if (threadIdx.x == 0) {
        scales[block_idx] = scale;
    }
    
    if (idx < N) {
        __nv_bfloat16 normalized = __hdiv(input[idx], scale);
        int8_t quantized;
        
        float norm_f = __bfloat162float(normalized);
        if (norm_f > 0.5f) {
            quantized = 1;
        } else if (norm_f < -0.5f) {
            quantized = -1;
        } else {
            quantized = 0;
        }
        
        int pack_idx = idx / 4;
        int bit_offset = (idx % 4) * 2;
        
        uint8_t unsigned_val;
        switch (quantized) {
            case -1: unsigned_val = 0; break;
            case 0: unsigned_val = 1; break;
            case 1: unsigned_val = 2; break;
            default: unsigned_val = 1; break;
        }
        
        atomicOr(&output[pack_idx], unsigned_val << bit_offset);
    }
    #endif
}

// Precision conversion kernels
extern "C" __global__ void convert_fp32_to_fp16(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = __float2half(input[idx]);
    }
}

extern "C" __global__ void convert_fp32_to_bf16(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N
) {
    #if __CUDA_ARCH__ >= 800
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = __float2bfloat16(input[idx]);
    }
    #endif
}

extern "C" __global__ void convert_fp16_to_fp32(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = __half2float(input[idx]);
    }
}

extern "C" __global__ void convert_bf16_to_fp32(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    #if __CUDA_ARCH__ >= 800
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = __bfloat162float(input[idx]);
    }
    #endif
}

// Memory-optimized batch operations
extern "C" __global__ void batch_convert_and_quantize_fp16(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    __half* __restrict__ scales,
    int batch_size,
    int elements_per_batch
) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || idx >= elements_per_batch) return;
    
    int global_idx = batch_idx * elements_per_batch + idx;
    
    // Convert to FP16 and quantize in one pass
    __half fp16_val = __float2half(input[global_idx]);
    
    // Simple quantization (can be made more sophisticated)
    int8_t quantized;
    float val = __half2float(fp16_val);
    
    if (val > 0.5f) {
        quantized = 1;
    } else if (val < -0.5f) {
        quantized = -1;
    } else {
        quantized = 0;
    }
    
    // Pack into output
    int pack_idx = global_idx / 4;
    int bit_offset = (global_idx % 4) * 2;
    
    uint8_t unsigned_val;
    switch (quantized) {
        case -1: unsigned_val = 0; break;
        case 0: unsigned_val = 1; break;
        case 1: unsigned_val = 2; break;
        default: unsigned_val = 1; break;
    }
    
    atomicOr(&output[pack_idx], unsigned_val << bit_offset);
    
    // Store scale (simplified - one scale per batch)
    if (idx == 0) {
        scales[batch_idx] = __float2half(1.0f); // Simplified scaling
    }
}

// Memory bandwidth optimized copy with conversion
extern "C" __global__ void optimized_copy_convert_fp16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int N
) {
    // Use vectorized loads/stores for better memory bandwidth
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < N) {
        // Load 4 float values at once
        float4 src_vals = reinterpret_cast<const float4*>(src)[idx / 4];
        
        // Convert to FP16
        __half4 dst_vals;
        dst_vals.x = __float2half(src_vals.x);
        dst_vals.y = __float2half(src_vals.y);
        dst_vals.z = __float2half(src_vals.z);
        dst_vals.w = __float2half(src_vals.w);
        
        // Store 4 FP16 values at once
        reinterpret_cast<__half4*>(dst)[idx / 4] = dst_vals;
    } else {
        // Handle remaining elements
        for (int i = idx; i < N && i < idx + 4; ++i) {
            dst[i] = __float2half(src[i]);
        }
    }
}