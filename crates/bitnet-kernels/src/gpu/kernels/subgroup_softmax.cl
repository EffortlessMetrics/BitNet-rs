/// Subgroup-optimized softmax for Intel Arc GPUs.
///
/// Uses cl_intel_subgroups for fast max and sum reduction within subgroups.
/// Numerically stable: subtracts max before exponentiation.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 16
#endif

/// Subgroup reduce helpers (inlined for softmax use).
inline float sg_reduce_max(float val) {
    for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = intel_sub_group_shuffle_down(val, val, offset);
        val = fmax(val, other);
    }
    return intel_sub_group_shuffle(val, 0);
}

inline float sg_reduce_sum(float val) {
    for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset >>= 1) {
        val += intel_sub_group_shuffle_down(val, val, offset);
    }
    return intel_sub_group_shuffle(val, 0);
}

/// Row-wise softmax: output[row][i] = exp(input[row][i] - max) / sum(exp)
///
/// Each workgroup handles one row of length N.
/// Phase 1: find row max (subgroup reduce -> local reduce)
/// Phase 2: compute exp(x - max) and sum (subgroup reduce -> local reduce)
/// Phase 3: normalize by sum
__kernel void subgroup_softmax(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int N
) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int wg_size = get_local_size(0);
    const int sg_id = get_sub_group_id();
    const int sg_local = get_sub_group_local_id();
    const int num_subgroups = (wg_size + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

    __global const float* row_in = input + row * N;
    __global float* row_out = output + row * N;

    // Phase 1: find row maximum
    float local_max = -INFINITY;
    for (int i = lid; i < N; i += wg_size) {
        local_max = fmax(local_max, row_in[i]);
    }
    float sg_max = sg_reduce_max(local_max);
    if (sg_local == 0) {
        scratch[sg_id] = sg_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float row_max = -INFINITY;
    if (sg_id == 0) {
        float v = (sg_local < num_subgroups) ? scratch[sg_local] : -INFINITY;
        row_max = sg_reduce_max(v);
        if (sg_local == 0) {
            scratch[0] = row_max;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    row_max = scratch[0];

    // Phase 2: compute exp(x - max) and accumulate sum
    float local_sum = 0.0f;
    for (int i = lid; i < N; i += wg_size) {
        float e = exp(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    float sg_sum = sg_reduce_sum(local_sum);
    if (sg_local == 0) {
        scratch[sg_id] = sg_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float row_sum = 0.0f;
    if (sg_id == 0) {
        float v = (sg_local < num_subgroups) ? scratch[sg_local] : 0.0f;
        row_sum = sg_reduce_sum(v);
        if (sg_local == 0) {
            scratch[0] = row_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    row_sum = scratch[0];

    // Phase 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = lid; i < N; i += wg_size) {
        row_out[i] *= inv_sum;
    }
}

/// Masked softmax for causal attention.
/// Positions where mask[i] == 0 are set to -inf before softmax.
__kernel void subgroup_softmax_masked(
    __global const float* input,
    __global const int* mask,
    __global float* output,
    __local float* scratch,
    const int N
) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int wg_size = get_local_size(0);
    const int sg_id = get_sub_group_id();
    const int sg_local = get_sub_group_local_id();
    const int num_subgroups = (wg_size + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

    __global const float* row_in = input + row * N;
    __global const int* row_mask = mask + row * N;
    __global float* row_out = output + row * N;

    // Phase 1: masked max
    float local_max = -INFINITY;
    for (int i = lid; i < N; i += wg_size) {
        float v = row_mask[i] ? row_in[i] : -INFINITY;
        local_max = fmax(local_max, v);
    }
    float sg_max = sg_reduce_max(local_max);
    if (sg_local == 0) scratch[sg_id] = sg_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    float row_max = -INFINITY;
    if (sg_id == 0) {
        float v = (sg_local < num_subgroups) ? scratch[sg_local] : -INFINITY;
        row_max = sg_reduce_max(v);
        if (sg_local == 0) scratch[0] = row_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    row_max = scratch[0];

    // Phase 2: masked exp + sum
    float local_sum = 0.0f;
    for (int i = lid; i < N; i += wg_size) {
        float e = row_mask[i] ? exp(row_in[i] - row_max) : 0.0f;
        row_out[i] = e;
        local_sum += e;
    }
    float sg_sum = sg_reduce_sum(local_sum);
    if (sg_local == 0) scratch[sg_id] = sg_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    float row_sum = 0.0f;
    if (sg_id == 0) {
        float v = (sg_local < num_subgroups) ? scratch[sg_local] : 0.0f;
        row_sum = sg_reduce_sum(v);
        if (sg_local == 0) scratch[0] = row_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    row_sum = scratch[0];

    // Phase 3: normalize
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int i = lid; i < N; i += wg_size) {
        row_out[i] *= inv_sum;
    }
}
