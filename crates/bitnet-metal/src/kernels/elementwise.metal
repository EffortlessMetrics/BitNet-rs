#include <metal_stdlib>
using namespace metal;

/// Element-wise addition: output = a + b
kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    output[gid] = a[gid] + b[gid];
}

/// Element-wise multiplication: output = a * b
kernel void mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    output[gid] = a[gid] * b[gid];
}

/// SiLU (Swish) activation: output = x * sigmoid(x)
kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    const float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}

/// GELU activation (approximate): output = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    const float x = input[gid];
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

/// Fused SiLU-gated multiplication: output = silu(gate) * up
/// Common in LLM feed-forward blocks (gate projection × up projection).
kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    const float g = gate[gid];
    const float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * up[gid];
}

/// Scalar multiply: output = input * scalar
kernel void scalar_mul(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    output[gid] = input[gid] * scalar;
}
