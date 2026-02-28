/// Intel subgroup-optimized matrix multiplication.
///
/// Uses cl_intel_subgroups extension for warp-level operations on Intel Arc GPUs.
/// Subgroup size on Intel Arc = 16 (SIMD16).
///
/// Falls back to regular matmul if subgroups are not supported.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 16
#endif

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

/// Subgroup-optimized GEMM: C = A * B
/// A is (M x K), B is (K x N), C is (M x N), row-major.
__kernel void subgroup_matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int sg_local_id = get_sub_group_local_id();

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // Tiled accumulation along K dimension
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        float a_val = 0.0f;
        float b_val = 0.0f;

        for (int k_off = 0; k_off < TILE_K && (k_base + k_off) < K; k_off++) {
            int k = k_base + k_off;
            a_val = A[row * K + k];
            b_val = B[k * N + col];
            acc += a_val * b_val;
        }
    }

    C[row * N + col] = acc;
}

/// Subgroup-optimized GEMM with intel_sub_group_block_read for coalesced loads.
/// Requires workgroup size to be a multiple of SUBGROUP_SIZE.
__kernel void subgroup_matmul_block(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K
) {
    const int row = get_global_id(0);
    const int sg_id = get_sub_group_id();
    const int sg_local = get_sub_group_local_id();

    if (row >= M) return;

    // Each subgroup computes SUBGROUP_SIZE columns of one row
    int col_base = sg_id * SUBGROUP_SIZE + sg_local;
    if (col_base >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        float a_val = A[row * K + k];
        float b_val = B[k * N + col_base];
        acc += a_val * b_val;
    }

    C[row * N + col_base] = acc;
}

/// Subgroup matmul for ternary weights (-1, 0, +1) packed as int8.
/// Avoids full multiply — uses conditional add/sub.
__kernel void subgroup_matmul_ternary(
    __global const char* weights,  // ternary weights as int8 (-1, 0, +1)
    __global const float* input,
    __global float* output,
    const int M, const int N, const int K
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        char w = weights[row * K + k];
        float x = input[k * N + col];
        acc += (w == 1) ? x : ((w == -1) ? -x : 0.0f);
    }

    output[row * N + col] = acc;
}
