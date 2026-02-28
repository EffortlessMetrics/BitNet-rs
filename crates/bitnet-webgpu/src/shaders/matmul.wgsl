// Matrix multiplication: C = A * B
// A is M×K, B is K×N, C is M×N

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params { m: u32, n: u32, k: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) { return; }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        sum = sum + a[row * params.k + i] * b[i * params.n + col];
    }
    result[row * params.n + col] = sum;
}
