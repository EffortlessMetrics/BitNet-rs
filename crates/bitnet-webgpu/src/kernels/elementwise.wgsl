// Elementwise operations compute shader for WebGPU inference.
// Supports: add, mul, SiLU, and GELU activations.
// Each entry point operates on flattened arrays of length `n`.

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Elementwise addition: output = input_a + input_b
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    output[idx] = input_a[idx] + input_b[idx];
}

// Elementwise multiplication: output = input_a * input_b
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    output[idx] = input_a[idx] * input_b[idx];
}

// SiLU activation: output = input_a * sigmoid(input_a)
// Also known as swish: x * σ(x) where σ(x) = 1 / (1 + exp(-x))
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    let x = input_a[idx];
    let sigmoid = 1.0 / (1.0 + exp(-x));
    output[idx] = x * sigmoid;
}

// GELU activation (tanh approximation):
// output = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    let x = input_a[idx];
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let coeff: f32 = 0.044715;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
