// Numerically stable softmax with work-group parallel reduction.
//
// Uses a two-pass algorithm:
//   1. Parallel reduction to find row-max (prevents overflow in exp).
//   2. Parallel computation of exp(x − max) and reduction of the sum.
//   3. Normalisation by 1/sum.
//
// Each work-group processes one row of length N.
// Local memory `scratch` must be allocated to at least
// get_local_size(0) * sizeof(float).

__kernel void softmax_stable(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const uint N)
{
    const uint lid    = get_local_id(0);
    const uint wg     = get_local_size(0);
    const uint row    = get_group_id(0);
    const uint offset = row * N;

    // Phase 1: parallel max reduction.
    float local_max = -INFINITY;
    for (uint i = lid; i < N; i += wg) {
        local_max = fmax(local_max, input[offset + i]);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: exp and parallel sum reduction.
    float local_sum = 0.0f;
    for (uint i = lid; i < N; i += wg) {
        float e = exp(input[offset + i] - max_val);
        output[offset + i] = e;
        local_sum += e;
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float total_sum = scratch[0];

    // Phase 3: normalise.
    const float inv_sum = 1.0f / total_sum;
    for (uint i = lid; i < N; i += wg) {
        output[offset + i] *= inv_sum;
    }
}

// Per-row softmax with float4 vectorization.
// Requires N % 4 == 0.  For non-aligned lengths, use softmax_stable.
__kernel void softmax_stable_vec4(
    __global const float4* input,
    __global float4* output,
    __local float* scratch,
    const uint N)
{
    const uint lid    = get_local_id(0);
    const uint wg     = get_local_size(0);
    const uint row    = get_group_id(0);
    const uint N4     = N / 4;
    const uint offset = row * N4;

    // Phase 1: find max.
    float local_max = -INFINITY;
    for (uint i = lid; i < N4; i += wg) {
        float4 v = input[offset + i];
        local_max = fmax(local_max, fmax(fmax(v.s0, v.s1), fmax(v.s2, v.s3)));
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: exp + sum.
    float local_sum = 0.0f;
    for (uint i = lid; i < N4; i += wg) {
        float4 v = input[offset + i];
        float4 e = exp(v - (float4)(max_val));
        output[offset + i] = e;
        local_sum += e.s0 + e.s1 + e.s2 + e.s3;
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = wg / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float total = scratch[0];

    // Phase 3: normalise.
    const float4 inv = (float4)(1.0f / total);
    for (uint i = lid; i < N4; i += wg) {
        output[offset + i] *= inv;
    }
}
