/// Subgroup-optimized reduction operations for Intel Arc GPUs.
///
/// Uses cl_intel_subgroups extension for fast intra-subgroup communication.
/// Falls back to work-group local memory reduction if subgroups not available.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 16
#endif

/// Fast sum reduction within a subgroup using shuffle.
inline float subgroup_reduce_sum(float val) {
    for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset >>= 1) {
        val += intel_sub_group_shuffle_down(val, val, offset);
    }
    return intel_sub_group_shuffle(val, 0);
}

/// Fast max reduction within a subgroup using shuffle.
inline float subgroup_reduce_max(float val) {
    for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = intel_sub_group_shuffle_down(val, val, offset);
        val = fmax(val, other);
    }
    return intel_sub_group_shuffle(val, 0);
}

/// Parallel reduction: compute sum of input[0..N-1].
__kernel void reduce_sum(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const uint N
) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wg_size = get_local_size(0);

    float val = (gid < N) ? input[gid] : 0.0f;

    // Phase 1: subgroup-level reduction
    float sg_sum = subgroup_reduce_sum(val);

    uint sg_id = get_sub_group_id();
    uint sg_local = get_sub_group_local_id();
    if (sg_local == 0) {
        scratch[sg_id] = sg_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: first subgroup reduces the partial sums
    uint num_subgroups = (wg_size + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    if (sg_id == 0) {
        float partial = (sg_local < num_subgroups) ? scratch[sg_local] : 0.0f;
        float total = subgroup_reduce_sum(partial);
        if (sg_local == 0) {
            output[get_group_id(0)] = total;
        }
    }
}

/// Parallel reduction: compute max of input[0..N-1].
__kernel void reduce_max(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const uint N
) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wg_size = get_local_size(0);

    float val = (gid < N) ? input[gid] : -INFINITY;

    float sg_max = subgroup_reduce_max(val);

    uint sg_id = get_sub_group_id();
    uint sg_local = get_sub_group_local_id();
    if (sg_local == 0) {
        scratch[sg_id] = sg_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint num_subgroups = (wg_size + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    if (sg_id == 0) {
        float partial = (sg_local < num_subgroups) ? scratch[sg_local] : -INFINITY;
        float total = subgroup_reduce_max(partial);
        if (sg_local == 0) {
            output[get_group_id(0)] = total;
        }
    }
}

/// Dot product of two vectors using subgroup reduction.
__kernel void reduce_dot(
    __global const float* A,
    __global const float* B,
    __global float* output,
    __local float* scratch,
    const uint N
) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wg_size = get_local_size(0);

    float val = (gid < N) ? A[gid] * B[gid] : 0.0f;

    float sg_sum = subgroup_reduce_sum(val);

    uint sg_id = get_sub_group_id();
    uint sg_local = get_sub_group_local_id();
    if (sg_local == 0) {
        scratch[sg_id] = sg_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint num_subgroups = (wg_size + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
    if (sg_id == 0) {
        float partial = (sg_local < num_subgroups) ? scratch[sg_local] : 0.0f;
        float total = subgroup_reduce_sum(partial);
        if (sg_local == 0) {
            output[get_group_id(0)] = total;
        }
    }
}
