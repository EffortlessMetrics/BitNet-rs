#include <metal_stdlib>
using namespace metal;

/// RMS normalization: output = (input / rms) * weight
/// where rms = sqrt(mean(input^2) + eps)
/// dims.x = batch size, dims.y = hidden dimension
constant uint RMSNORM_THREADGROUP_SIZE = 256;

kernel void rmsnorm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint batch = gid.y;
    const uint hidden_dim = dims.y;

    if (batch >= dims.x) {
        return;
    }

    const uint row_offset = batch * hidden_dim;

    threadgroup float shared_data[256];

    // Phase 1: Compute sum of squares via threadgroup reduction
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        const float val = input[row_offset + i];
        local_sum_sq += val * val;
    }
    shared_data[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms = rsqrt(shared_data[0] / float(hidden_dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Normalize and scale
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        output[row_offset + i] = input[row_offset + i] * rms * weight[i];
    }
}
