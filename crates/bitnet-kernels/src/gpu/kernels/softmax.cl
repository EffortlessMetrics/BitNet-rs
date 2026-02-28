/// Numerically stable softmax for BitNet inference.
///
/// Implements the three-pass algorithm:
///   1. Find max value (parallel reduction in local memory)
///   2. Subtract max, exponentiate, and sum (parallel reduction)
///   3. Divide each element by the sum
///
/// Each work-group processes one row of the input tensor.

__kernel void softmax(
    __global const float* input,
    __global float* output,
    const uint N
) {
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);

    __local float scratch[256];

    const uint row_offset = row * N;

    // --- Pass 1: find row maximum via parallel reduction ---
    float local_max = -INFINITY;
    for (uint i = lid; i < N; i += local_size) {
        float v = input[row_offset + i];
        local_max = fmax(local_max, v);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Pass 2: exponentiate (shifted) and compute sum ---
    float local_sum = 0.0f;
    for (uint i = lid; i < N; i += local_size) {
        float e = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = e;  // store intermediate exp values
        local_sum += e;
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Pass 3: normalise ---
    float inv_sum = 1.0f / row_sum;
    for (uint i = lid; i < N; i += local_size) {
        output[row_offset + i] *= inv_sum;
    }
}