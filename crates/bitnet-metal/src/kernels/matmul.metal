// Metal Shading Language (MSL) — matrix multiplication kernel
// C = A × B  where A is M×K, B is K×N, C is M×N (row-major f32)

#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint m;
    uint n;
    uint k;
};

kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result   [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    float sum = 0.0f;
    for (uint i = 0; i < params.k; i++) {
        sum += a[row * params.k + i] * b[i * params.n + col];
    }
    result[row * params.n + col] = sum;
}
