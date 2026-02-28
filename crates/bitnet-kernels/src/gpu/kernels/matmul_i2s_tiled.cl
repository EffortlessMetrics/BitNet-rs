/// Tiled matrix multiplication for I2S (2-bit signed) quantized weights.
///
/// Optimized for Intel Arc GPUs with:
///   1. Tiled computation using __local (shared) memory
///   2. Float4 vectorization for better memory throughput
///   3. Branchless ternary weight decode via bitwise ops
///   4. Configurable work-group sizes for Intel Arc EU mapping
///
/// Computes C = A * B where:
///   A is an [M x K] matrix of int8 activations
///   B is a [K x N] matrix of uint8 packed 2-bit weights (4 values per byte)
///   C is an [M x N] matrix of float32 results
///
/// The original matmul_i2s kernel is retained as a fallback for non-tiled
/// dispatch or when dimensions are not tile-aligned.

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef LOCAL_SIZE_X
#define LOCAL_SIZE_X 16
#endif

#ifndef LOCAL_SIZE_Y
#define LOCAL_SIZE_Y 16
#endif

/// Branchless ternary decode for a single 2-bit value.
/// Encoding: 0b00=0, 0b01=+1, 0b11=-1, 0b10=0
///
/// Method: use bit 0 as magnitude, bit 1 as sign.
///   magnitude = bits & 1
///   sign      = (bits >> 1) & 1
///   weight    = magnitude - 2 * magnitude * sign
inline int decode_ternary(uchar bits) {
    int mag  = bits & 1;
    int sign = (bits >> 1) & 1;
    return mag - 2 * mag * sign;
}

/// Tiled I2S matmul kernel using local memory.
///
/// Each work-group computes a TILE_SIZE x TILE_SIZE sub-block of C by
/// iterating over K in TILE_SIZE-wide strips, loading tiles of A and
/// decoded tiles of B into __local memory, then computing partial sums
/// from the fast local arrays.
__kernel void matmul_i2s_tiled(
    __global const char*  A,   // [M x K] int8 activations
    __global const uchar* B,   // [K/4 x N] packed 2-bit weights
    __global float*       C,   // [M x N] output
    const uint M,
    const uint N,
    const uint K
) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const uint localRow = get_local_id(0);
    const uint localCol = get_local_id(1);
    const uint globalRow = get_group_id(0) * TILE_SIZE + localRow;
    const uint globalCol = get_group_id(1) * TILE_SIZE + localCol;

    float acc = 0.0f;
    const uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load tile of A into local memory (int8 -> float)
        uint aCol = t * TILE_SIZE + localCol;
        if (globalRow < M && aCol < K) {
            tileA[localRow][localCol] = (float)A[globalRow * K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }

        // Load tile of B into local memory (decode packed 2-bit -> float)
        uint bRow = t * TILE_SIZE + localRow;
        if (bRow < K && globalCol < N) {
            uint byte_idx = (bRow / 4) * N + globalCol;
            uint sub = bRow & 3;
            uchar packed = B[byte_idx];
            uchar bits = (packed >> (sub * 2)) & 0x03;
            tileB[localRow][localCol] = (float)decode_ternary(bits);
        } else {
            tileB[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = acc;
    }
}

/// Float4-vectorized tiled matmul for aligned dimensions.
///
/// Processes 4 columns simultaneously using float4, which maps well to
/// Intel Arc's 256-bit SIMD execution units.
__kernel void matmul_i2s_tiled_vec4(
    __global const char*  A,   // [M x K] int8 activations
    __global const uchar* B,   // [K/4 x N] packed 2-bit weights
    __global float*       C,   // [M x N] output
    const uint M,
    const uint N,
    const uint K
) {
    const uint localRow = get_local_id(0);
    const uint localCol = get_local_id(1);
    const uint globalRow = get_group_id(0) * TILE_SIZE + localRow;
    const uint globalCol4 = (get_group_id(1) * TILE_SIZE + localCol) * 4;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float4 tileB4[TILE_SIZE][TILE_SIZE];

    float4 acc = (float4)(0.0f);
    const uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + localCol;
        if (globalRow < M && aCol < K) {
            tileA[localRow][localCol] = (float)A[globalRow * K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }

        uint bRow = t * TILE_SIZE + localRow;
        float4 bVal = (float4)(0.0f);
        if (bRow < K) {
            for (uint v = 0; v < 4; v++) {
                uint col = globalCol4 + v;
                if (col < N) {
                    uint byte_idx = (bRow / 4) * N + col;
                    uint sub = bRow & 3;
                    uchar packed = B[byte_idx];
                    uchar bits = (packed >> (sub * 2)) & 0x03;
                    switch (v) {
                        case 0: bVal.x = (float)decode_ternary(bits); break;
                        case 1: bVal.y = (float)decode_ternary(bits); break;
                        case 2: bVal.z = (float)decode_ternary(bits); break;
                        case 3: bVal.w = (float)decode_ternary(bits); break;
                    }
                }
            }
        }
        tileB4[localRow][localCol] = bVal;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < TILE_SIZE; k++) {
            float aElem = tileA[localRow][k];
            acc += aElem * tileB4[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < M && globalCol4 + 3 < N) {
        vstore4(acc, 0, &C[globalRow * N + globalCol4]);
    } else if (globalRow < M) {
        if (globalCol4     < N) C[globalRow * N + globalCol4]     = acc.x;
        if (globalCol4 + 1 < N) C[globalRow * N + globalCol4 + 1] = acc.y;
        if (globalCol4 + 2 < N) C[globalRow * N + globalCol4 + 2] = acc.z;
        if (globalCol4 + 3 < N) C[globalRow * N + globalCol4 + 3] = acc.w;
    }
}
