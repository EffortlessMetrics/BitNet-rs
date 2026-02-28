/**
 * HIP element-wise activation and arithmetic kernels for AMD GPUs.
 *
 * Activations: SiLU, GELU (tanh approximation)
 * Arithmetic:  vector add, vector multiply
 */

#include <hip/hip_runtime.h>

// ---------- activations ----------

extern "C" __global__ void silu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= N) return;

    float x = input[idx];
    float sigmoid = 1.0f / (1.0f + expf(-x));
    output[idx] = x * sigmoid;
}

extern "C" __global__ void gelu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= N) return;

    float x = input[idx];
    // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float SQRT_2_OVER_PI = 0.7978845608f;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

// ---------- arithmetic ----------

extern "C" __global__ void vec_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N
) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= N) return;

    out[idx] = a[idx] + b[idx];
}

extern "C" __global__ void vec_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N
) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= N) return;

    out[idx] = a[idx] * b[idx];
}
