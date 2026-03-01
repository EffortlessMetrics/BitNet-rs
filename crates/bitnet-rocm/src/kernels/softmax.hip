/**
 * HIP numerically-stable softmax kernel for AMD GPUs.
 *
 * Two-pass algorithm:
 *   1. Warp-reduction to find row max.
 *   2. Subtract max, exponentiate, warp-reduction for sum, normalise.
 */

#include <hip/hip_runtime.h>

#define WARP_SIZE 64

/// Warp-level reduction: maximum
__device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

/// Warp-level reduction: sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

/**
 * Row-wise softmax.  Each block processes one row of length `cols`.
 * Grid: (batch, 1, 1)   Block: (cols_padded_to_warp, 1, 1)
 */
extern "C" __global__ void softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = hipBlockIdx_x;
    if (row >= rows) return;

    int tid = hipThreadIdx_x;

    // -- Pass 1: find row max --
    float local_max = -1e30f;
    for (int c = tid; c < cols; c += hipBlockDim_x) {
        local_max = fmaxf(local_max, input[row * cols + c]);
    }

    __shared__ float smax[64];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) smax[wid] = warp_max;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < (hipBlockDim_x + WARP_SIZE - 1) / WARP_SIZE)
            ? smax[tid] : -1e30f;
        warp_max = warp_reduce_max(v);
        if (tid == 0) smax[0] = warp_max;
    }
    __syncthreads();
    float row_max = smax[0];

    // -- Pass 2: exp and sum --
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += hipBlockDim_x) {
        float e = expf(input[row * cols + c] - row_max);
        output[row * cols + c] = e;
        local_sum += e;
    }

    __shared__ float ssum[64];
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) ssum[wid] = warp_sum;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < (hipBlockDim_x + WARP_SIZE - 1) / WARP_SIZE)
            ? ssum[tid] : 0.0f;
        warp_sum = warp_reduce_sum(v);
        if (tid == 0) ssum[0] = warp_sum;
    }
    __syncthreads();
    float row_sum = ssum[0];

    // -- Pass 3: normalise --
    float inv_sum = 1.0f / (row_sum + 1e-8f);
    for (int c = tid; c < cols; c += hipBlockDim_x) {
        output[row * cols + c] *= inv_sum;
    }
}
