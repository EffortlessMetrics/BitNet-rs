/**
 * HIP RMS normalisation kernel for AMD GPUs.
 *
 * RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
 *
 * Each block handles one row of the input tensor.
 */

#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__device__ float warp_reduce_sum_rms(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

extern "C" __global__ void rmsnorm_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int rows, int cols, float eps
) {
    int row = hipBlockIdx_x;
    if (row >= rows) return;

    int tid = hipThreadIdx_x;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += hipBlockDim_x) {
        float v = input[row * cols + c];
        sum_sq += v * v;
    }

    // Block-wide reduction via shared memory
    __shared__ float shared[64];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    float warp_sum = warp_reduce_sum_rms(sum_sq);
    if (lane == 0) shared[wid] = warp_sum;
    __syncthreads();

    if (tid < WARP_SIZE) {
        float v = (tid < (hipBlockDim_x + WARP_SIZE - 1) / WARP_SIZE)
            ? shared[tid] : 0.0f;
        warp_sum = warp_reduce_sum_rms(v);
        if (tid == 0) shared[0] = warp_sum;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / (float)cols + eps);

    // Normalise and scale
    for (int c = tid; c < cols; c += hipBlockDim_x) {
        output[row * cols + c] =
            input[row * cols + c] * rms * weight[c];
    }
}
