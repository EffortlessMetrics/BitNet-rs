// OpenCL activation kernels for GPU-accelerated transformer inference.
// SiLU, SiLU-gate (fused), and GELU activations.

/// SiLU (Sigmoid Linear Unit): output[i] = input[i] * sigmoid(input[i])
__kernel void silu_activation(
    __global const float* input,
    __global float* output,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        float x = input[gid];
        float sig = 1.0f / (1.0f + exp(-x));
        output[gid] = x * sig;
    }
}

/// Fused SiLU-gate: output[i] = silu(gate[i]) * value[i]
/// Common in LLaMA-style FFN: the up-projection is split into gate and value halves.
/// gate: [n], value: [n], output: [n]
__kernel void silu_gate_fused(
    __global const float* gate,
    __global const float* value,
    __global float* output,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        float g = gate[gid];
        float sig = 1.0f / (1.0f + exp(-g));
        output[gid] = (g * sig) * value[gid];
    }
}

/// GELU (Gaussian Error Linear Unit) approximation using tanh.
/// output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__kernel void gelu_activation(
    __global const float* input,
    __global float* output,
    const uint n)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        float x = input[gid];
        float c = 0.7978845608f; // sqrt(2/pi)
        float inner = c * (x + 0.044715f * x * x * x);
        output[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}
