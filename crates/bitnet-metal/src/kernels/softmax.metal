// Metal Shading Language (MSL) — row-wise softmax kernel
// Each threadgroup processes one row of length N.

#include <metal_stdlib>
using namespace metal;

struct SoftmaxParams {
    uint n;
};

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]])
{
    uint row = tg;
    uint row_start = row * params.n;
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    // Phase 1: row max
    float local_max = -MAXFLOAT;
    for (uint i = tid; i < params.n; i += 256) {
        float val = input[row_start + i];
        local_max = max(local_max, val);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // Phase 2: exp + sum
    float local_sum = 0.0f;
    for (uint i = tid; i < params.n; i += 256) {
        float e = exp(input[row_start + i] - row_max);
        output[row_start + i] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared_sum[0];

    // Phase 3: normalize
    for (uint i = tid; i < params.n; i += 256) {
        output[row_start + i] /= total;
    }
}
