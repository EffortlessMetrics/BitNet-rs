// Optimized matmul with float4 vectorization and local memory tiling.
//
// C = A × B where:
//   A is M×K (row-major float)
//   B is K×N (row-major float)
//   C is M×N (row-major float)
//
// Uses TILE_SIZE×TILE_SIZE local memory tiles to reduce global traffic
// by a factor of TILE_SIZE.  Inner loop uses fma() for fused
// multiply-add on supporting hardware.

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__kernel void matmul_vec4(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const uint M,
    const uint N,
    const uint K)
{
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const uint lid_x = get_local_id(0);
    const uint lid_y = get_local_id(1);
    const uint row = get_group_id(1) * TILE_SIZE + lid_y;
    const uint col = get_group_id(0) * TILE_SIZE + lid_x;

    float acc = 0.0f;

    const uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < num_tiles; t++) {
        const uint a_col = t * TILE_SIZE + lid_x;
        const uint b_row = t * TILE_SIZE + lid_y;

        tileA[lid_y][lid_x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        tileB[lid_y][lid_x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < TILE_SIZE; k++) {
            acc = fma(tileA[lid_y][k], tileB[k][lid_x], acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// float4-vectorized variant – each work-item computes a 1×4 strip of C.
// Requires N % 4 == 0 for best performance; falls back to scalar for
// the final partial strip.
__kernel void matmul_vec4_wide(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float4* restrict C,
    const uint M,
    const uint N,
    const uint K)
{
    const uint row = get_global_id(1);
    // Each work-item covers 4 consecutive columns.
    const uint col4 = get_global_id(0);
    const uint col  = col4 * 4;

    if (row >= M || col >= N) return;

    float4 acc = (float4)(0.0f);

    uint k = 0;
    // Process 4 K-elements at a time when possible.
    for (; k + 3 < K; k += 4) {
        float4 a = vload4(0, &A[row * K + k]);
        float4 b0 = vload4(0, &B[(k + 0) * N + col]);
        float4 b1 = vload4(0, &B[(k + 1) * N + col]);
        float4 b2 = vload4(0, &B[(k + 2) * N + col]);
        float4 b3 = vload4(0, &B[(k + 3) * N + col]);

        acc = fma((float4)(a.s0), b0, acc);
        acc = fma((float4)(a.s1), b1, acc);
        acc = fma((float4)(a.s2), b2, acc);
        acc = fma((float4)(a.s3), b3, acc);
    }
    // Remainder loop.
    for (; k < K; k++) {
        float a_val = A[row * K + k];
        float4 b_val = vload4(0, &B[k * N + col]);
        acc = fma((float4)(a_val), b_val, acc);
    }

    // Bounds-safe store: write full vector only if 4 cols fit.
    if (col + 4 <= N) {
        C[row * (N / 4) + col4] = acc;
    } else {
        __global float* Cf = (__global float*)C;
        uint base = row * N + col;
        if (col + 0 < N) Cf[base + 0] = acc.s0;
        if (col + 1 < N) Cf[base + 1] = acc.s1;
        if (col + 2 < N) Cf[base + 2] = acc.s2;
        if (col + 3 < N) Cf[base + 3] = acc.s3;
    }
}
