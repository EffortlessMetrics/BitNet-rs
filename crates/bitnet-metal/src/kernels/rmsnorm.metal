// Metal Shading Language (MSL) — RMS normalization kernel
// output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)

#include <metal_stdlib>
using namespace metal;

struct RmsNormParams {
    uint n;
    float eps;
};

kernel void rmsnorm(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]])
{
    uint row = tg;
    uint row_start = row * params.n;
    threadgroup float shared_sq_sum[256];

    // Phase 1: sum of squares
    float local_sq = 0.0f;
    for (uint i = tid; i < params.n; i += 256) {
        float val = input[row_start + i];
        local_sq += val * val;
    }
    shared_sq_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sq_sum[0] / float(params.n) + params.eps);

    // Phase 2: normalize and scale
    for (uint i = tid; i < params.n; i += 256) {
        output[row_start + i] = (input[row_start + i] / rms) * weight[i];
    }
}
