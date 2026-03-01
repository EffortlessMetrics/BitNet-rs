// Tiled matrix multiplication kernels for Intel Arc A770 (Xe-HPG)
//
// Two kernels:
// - tiled_matmul_f32: 16×16 tiled GEMM using shared local memory
// - quantized_gemv_i2s: I2_S 2-bit packed weight GEMV with per-row scales

#define TILE_SIZE 16

// Tiled GEMM: C = A × B using 16×16 tiles in local memory.
// Grid: (ceil(N/16)*16, ceil(M/16)*16), Local: (16, 16)
__kernel void tiled_matmul_f32(
    __global const float* A,        // [M, K]
    __global const float* B,        // [K, N]
    __global float* C,              // [M, N]
    const int M,
    const int N,
    const int K,
    __local float* tile_A,          // [TILE_SIZE * TILE_SIZE]
    __local float* tile_B           // [TILE_SIZE * TILE_SIZE]
) {
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    const int lr  = get_local_id(1);
    const int lc  = get_local_id(0);

    float acc = 0.0f;

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A into local memory
        int a_col = t * TILE_SIZE + lc;
        if (row < M && a_col < K) {
            tile_A[lr * TILE_SIZE + lc] = A[row * K + a_col];
        } else {
            tile_A[lr * TILE_SIZE + lc] = 0.0f;
        }

        // Load tile of B into local memory
        int b_row = t * TILE_SIZE + lr;
        if (b_row < K && col < N) {
            tile_B[lr * TILE_SIZE + lc] = B[b_row * N + col];
        } else {
            tile_B[lr * TILE_SIZE + lc] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_A[lr * TILE_SIZE + k] * tile_B[k * TILE_SIZE + lc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Quantized GEMV: y = (packed_W · x) * scales
// W is I2_S packed (4 ternary values per byte), scales is per-row.
// Grid: (M), each work-item computes one output element.
__kernel void quantized_gemv_i2s(
    __global const char* packed_W,  // [M, K/4] packed 2-bit weights
    __global const float* x,        // [K]
    __global const float* scales,   // [M] per-row scale factors
    __global float* y,              // [M]
    const int M,
    const int K
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    const int packed_K = (K + 3) / 4;
    float acc = 0.0f;

    for (int j = 0; j < packed_K; j++) {
        uchar packed = (uchar)packed_W[row * packed_K + j];

        for (int sub = 0; sub < 4; sub++) {
            int col = j * 4 + sub;
            if (col >= K) break;

            uchar bits = (packed >> (sub * 2)) & 0x03;

            // Ternary decode: 0x01 -> +1, 0x03 -> -1, else 0
            float w;
            if (bits == 0x01) {
                w = 1.0f;
            } else if (bits == 0x03) {
                w = -1.0f;
            } else {
                w = 0.0f;
            }

            acc += w * x[col];
        }
    }

    y[row] = acc * scales[row];
}
