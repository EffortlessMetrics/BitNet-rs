@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn gelu_approx(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        let x = input[i];
        let c = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        output[i] = 0.5 * x * (1.0 + tanh(c));
    }
}
