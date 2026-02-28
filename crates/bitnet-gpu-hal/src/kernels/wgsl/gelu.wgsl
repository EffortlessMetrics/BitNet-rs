@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736
        + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - poly * exp(-a * a));
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        let x = input[i];
        let cdf = 0.5 * (1.0 + erf_approx(x * 0.7071067811865476));
        output[i] = x * cdf;
    }
}
