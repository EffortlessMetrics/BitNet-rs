/**
 * HIP matrix multiplication kernels for BitNet inference on AMD GPUs.
 *
 * Tiled matmul: C = A * B where A is int8, B is uint8 (I2S packed), C is f32.
 * Uses shared-memory tiling with 16×16 blocks for memory coalescing.
 */

#include <hip/hip_runtime.h>

#define TILE_SIZE 16

extern "C" __global__ void bitnet_matmul_i2s(
    const signed char* __restrict__ A,
    const unsigned char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int row = hipBlockIdx_y * TILE_SIZE + ty;
    int col = hipBlockIdx_x * TILE_SIZE + tx;

    __shared__ signed char As[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned char Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0;
        Bs[ty][tx] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += (float)As[ty][k] * (float)Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/**
 * Simple matmul without tiling – useful for small matrices or as fallback.
 */
extern "C" __global__ void bitnet_matmul_simple(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        acc += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = acc;
}
