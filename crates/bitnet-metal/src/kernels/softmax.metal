#include <metal_stdlib>
using namespace metal;

/// Numerically stable softmax over the last dimension.
/// dims.x = batch size, dims.y = sequence length (number of elements per row).
/// Uses threadgroup reduction for max and sum computation.
constant uint SOFTMAX_THREADGROUP_SIZE = 256;

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint batch = gid.y;
    const uint seq_len = dims.y;

    if (batch >= dims.x) {
        return;
    }

    const uint row_offset = batch * seq_len;

    threadgroup float shared_data[256];

    // Phase 1: Find max value via threadgroup reduction
    float local_max = -INFINITY;
    for (uint i = tid; i < seq_len; i += tg_size) {
        local_max = max(local_max, input[row_offset + i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute sum of exp(x - max) via threadgroup reduction
    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += tg_size) {
        local_sum += exp(input[row_offset + i] - row_max);
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize
    const float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < seq_len; i += tg_size) {
        output[row_offset + i] = exp(input[row_offset + i] - row_max) * inv_sum;
    }
}
