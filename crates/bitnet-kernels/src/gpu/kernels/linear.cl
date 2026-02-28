// OpenCL linear projection kernels for GPU-accelerated transformer inference.

/// Dense linear forward pass: output[row] = sum(weights[row][k] * input[k]) + bias[row]
/// weights: [out_features * in_features], input: [in_features], output: [out_features]
__kernel void linear_forward(
    __global const float* weights,
    __global const float* input,
    __global const float* bias,
    __global float* output,
    const uint in_features,
    const uint out_features)
{
    uint row = get_global_id(0);
    if (row < out_features) {
        float acc = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            acc += weights[row * in_features + k] * input[k];
        }
        output[row] = acc + bias[row];
    }
}

/// Dense linear forward pass without bias.
/// weights: [out_features * in_features], input: [in_features], output: [out_features]
__kernel void linear_forward_nobias(
    __global const float* weights,
    __global const float* input,
    __global float* output,
    const uint in_features,
    const uint out_features)
{
    uint row = get_global_id(0);
    if (row < out_features) {
        float acc = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            acc += weights[row * in_features + k] * input[k];
        }
        output[row] = acc;
    }
}

/// Tiled batched linear: output[b][row] = sum(weights[row][k] * input[b][k]) + bias[row]
/// Processes TILE_SIZE elements per work-item for better memory coalescing.
/// weights: [out_features * in_features], input: [batch * in_features],
/// output: [batch * out_features]
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
__kernel void linear_forward_batched(
    __global const float* weights,
    __global const float* input,
    __global const float* bias,
    __global float* output,
    const uint in_features,
    const uint out_features,
    const uint batch_size)
{
    uint row = get_global_id(0);
    uint b   = get_global_id(1);
    if (row < out_features && b < batch_size) {
        float acc = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            acc += weights[row * in_features + k] * input[b * in_features + k];
        }
        output[b * out_features + row] = acc + bias[row];
    }
}
