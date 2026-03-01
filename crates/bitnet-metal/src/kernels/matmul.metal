#include <metal_stdlib>
using namespace metal;

/// Matrix multiplication: C = A * B
/// dims.x = M (rows of A), dims.y = N (cols of B), dims.z = K (cols of A / rows of B)
kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint M = dims.x;
    const uint N = dims.y;
    const uint K = dims.z;

    const uint row = gid.y;
    const uint col = gid.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

/// Tiled matrix multiplication using threadgroup shared memory.
/// TILE_SIZE must match the threadgroup dimensions (e.g., 16×16).
constant uint TILE_SIZE = 16;

kernel void matmul_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint M = dims.x;
    const uint N = dims.y;
    const uint K = dims.z;

    threadgroup float a_tile[16 * 16];
    threadgroup float b_tile[16 * 16];

    const uint row = tgid.y * TILE_SIZE + tid.y;
    const uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;
    const uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile from A
        const uint a_col = t * TILE_SIZE + tid.x;
        if (row < M && a_col < K) {
            a_tile[tid.y * TILE_SIZE + tid.x] = a[row * K + a_col];
        } else {
            a_tile[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load tile from B
        const uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < K && col < N) {
            b_tile[tid.y * TILE_SIZE + tid.x] = b[b_row * N + col];
        } else {
            b_tile[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += a_tile[tid.y * TILE_SIZE + i] * b_tile[i * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
