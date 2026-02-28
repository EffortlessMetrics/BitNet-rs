@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn mish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        let x = input[i];
        var sp: f32;
        if x > 20.0 {
            sp = x;
        } else {
            sp = log(1.0 + exp(x));
        }
        output[i] = x * tanh(sp);
    }
}
