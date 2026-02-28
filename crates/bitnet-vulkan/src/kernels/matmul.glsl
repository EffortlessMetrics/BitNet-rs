#version 450

// Tiled matrix multiply using shared memory.
// C[M,N] = A[M,K] * B[K,N]
//
// Each workgroup computes a 16x16 tile of C using shared-memory tiling
// to reduce global memory bandwidth.

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer InputA { float a[]; };
layout(set = 0, binding = 1) readonly buffer InputB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float c[]; };

layout(push_constant) uniform Params {
    uint M, N, K;
};

shared float tileA[16][16];
shared float tileB[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint localRow = gl_LocalInvocationID.y;
    uint localCol = gl_LocalInvocationID.x;

    float sum = 0.0;
    uint numTiles = (K + 15u) / 16u;

    for (uint t = 0u; t < numTiles; t++) {
        // Load tile of A into shared memory
        uint aCol = t * 16u + localCol;
        if (row < M && aCol < K) {
            tileA[localRow][localCol] = a[row * K + aCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        // Load tile of B into shared memory
        uint bRow = t * 16u + localRow;
        if (bRow < K && col < N) {
            tileB[localRow][localCol] = b[bRow * N + col];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        // Synchronize to ensure tile is fully loaded
        barrier();

        // Accumulate partial dot product from this tile
        for (uint k = 0u; k < 16u; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        // Synchronize before loading next tile
        barrier();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
