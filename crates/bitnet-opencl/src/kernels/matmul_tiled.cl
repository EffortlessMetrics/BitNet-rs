// Tiled matrix multiplication using __local memory for work-group shared data.
//
// C = A × B where:
//   A is M×K (row-major, i8 ternary weights {-1, 0, 1})
//   B is K×N (row-major, f32 activations)
//   C is M×N (row-major, f32 output)
//
// Each work-group is TILE×TILE threads. Tiles of A and B are staged
// in __local memory to reduce global memory traffic by a factor of TILE.

#ifndef TILE
#define TILE 16
#endif

__kernel void matmul_tiled(
    __global const char* A,
    __global const float* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K)
{
    const uint row = get_local_id(0);
    const uint col = get_local_id(1);
    const uint global_row = get_group_id(0) * TILE + row;
    const uint global_col = get_group_id(1) * TILE + col;

    __local char  A_tile[TILE][TILE];
    __local float B_tile[TILE][TILE];

    float acc = 0.0f;

    const uint num_tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        // Cooperative load of A tile into local memory
        uint a_col = t * TILE + col;
        if (global_row < M && a_col < K)
            A_tile[row][col] = A[global_row * K + a_col];
        else
            A_tile[row][col] = 0;

        // Cooperative load of B tile into local memory
        uint b_row = t * TILE + row;
        if (b_row < K && global_col < N)
            B_tile[row][col] = B[b_row * N + global_col];
        else
            B_tile[row][col] = 0.0f;

        // Synchronize to ensure all data is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate partial dot product from tiles
        for (uint e = 0; e < TILE; e++) {
            acc += (float)A_tile[row][e] * B_tile[e][col];
        }

        // Synchronize before loading next tiles
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}
