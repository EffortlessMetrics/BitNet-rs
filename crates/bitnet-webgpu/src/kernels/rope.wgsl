// Rotary position embeddings (RoPE) compute shader for WebGPU inference.
// Applies rotary embeddings in-place to query/key tensors.
// Each thread handles one (cos, sin) rotation for a pair of elements.

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    theta_base: f32,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.seq_len * params.n_heads * half_dim;

    let idx = gid.x;
    if (idx >= total_pairs) {
        return;
    }

    // Decompose linear index into (pos, head, d)
    let d = idx % half_dim;
    let remainder = idx / half_dim;
    let head = remainder % params.n_heads;
    let pos = remainder / params.n_heads;

    // Compute rotation angle: theta = pos / (theta_base ^ (2d / head_dim))
    let freq_exp = f32(2u * d) / f32(params.head_dim);
    let inv_freq = 1.0 / pow(params.theta_base, freq_exp);
    let angle = f32(pos) * inv_freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // Index into the flattened (seq_len, n_heads, head_dim) tensor
    let base = (pos * params.n_heads + head) * params.head_dim;
    let i0 = base + d;
    let i1 = base + d + half_dim;

    // Apply rotation to Q
    let q0 = q[i0];
    let q1 = q[i1];
    q[i0] = q0 * cos_val - q1 * sin_val;
    q[i1] = q0 * sin_val + q1 * cos_val;

    // Apply rotation to K
    let k0 = k[i0];
    let k1 = k[i1];
    k[i0] = k0 * cos_val - k1 * sin_val;
    k[i1] = k0 * sin_val + k1 * cos_val;
}
