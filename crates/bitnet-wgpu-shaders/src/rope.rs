//! Rotary Position Embedding (`RoPE`) WGSL compute shader.
//!
//! Applies rotary embeddings in-place to query/key tensors for
//! transformer attention. Each thread handles one (position, `head_dim/2`) pair.

/// Rotary Position Embedding.
///
/// Applies `RoPE` to an input tensor of shape [`seq_len`, `num_heads`, `head_dim`].
/// The rotation is applied to consecutive pairs: (x[2i], x[2i+1]).
///
/// freq(i) = 1 / (theta ^ (2i / `head_dim`))
/// x'[2i]   = x[2i]   * cos(pos * freq) - x[2i+1] * sin(pos * freq)
/// x'[2i+1] = x[2i]   * sin(pos * freq) + x[2i+1] * cos(pos * freq)
///
/// Workgroup size: [256, 1, 1]. Dispatch with enough threads to cover
/// `seq_len` * `num_heads` * (`head_dim` / 2).
pub const ROPE_SRC: &str = r"
struct RopeParams {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RopeParams;

@compute @workgroup_size(256, 1, 1)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = params.head_dim / 2u;
    let total = params.seq_len * params.num_heads * half_dim;
    let idx = gid.x;
    if idx >= total { return; }

    // Decompose linear index into (seq_pos, head, pair_idx)
    let pair_idx = idx % half_dim;
    let remaining = idx / half_dim;
    let head = remaining % params.num_heads;
    let seq_pos = remaining / params.num_heads;

    let pos = f32(positions[seq_pos]);
    let freq_exp = 2.0 * f32(pair_idx) / f32(params.head_dim);
    let freq = 1.0 / pow(params.theta, freq_exp);
    let angle = pos * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    // Offset into the flattened [seq_len, num_heads, head_dim] tensor
    let base = (seq_pos * params.num_heads + head) * params.head_dim + pair_idx * 2u;
    let x0 = input[base];
    let x1 = input[base + 1u];

    output[base] = x0 * cos_a - x1 * sin_a;
    output[base + 1u] = x0 * sin_a + x1 * cos_a;
}
";
