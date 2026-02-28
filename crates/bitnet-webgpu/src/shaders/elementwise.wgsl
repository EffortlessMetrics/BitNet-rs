// Element-wise operations: add, mul, ReLU, SiLU
// op == 0: add, op == 1: mul, op == 2: relu, op == 3: silu

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params { len: u32, op: u32 }
@group(0) @binding(3) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }

    switch params.op {
        case 0u: { // add
            result[idx] = a[idx] + b[idx];
        }
        case 1u: { // mul
            result[idx] = a[idx] * b[idx];
        }
        case 2u: { // relu
            result[idx] = max(a[idx], 0.0);
        }
        case 3u: { // silu
            result[idx] = silu(a[idx]);
        }
        default: {
            result[idx] = a[idx];
        }
    }
}
