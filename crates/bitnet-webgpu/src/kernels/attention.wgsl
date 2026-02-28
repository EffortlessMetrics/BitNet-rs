// Attention: Q·Kᵀ / √d_k → softmax → · V
// Single-head attention for one query position against seq_len key/value positions.

@group(0) @binding(0) var<storage, read> query: array<f32>;   // [head_dim]
@group(0) @binding(1) var<storage, read> key: array<f32>;     // [seq_len × head_dim]
@group(0) @binding(2) var<storage, read> value: array<f32>;   // [seq_len × head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [head_dim]

struct Params { head_dim: u32, seq_len: u32 }
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scores: array<f32, 256>;
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let hd = params.head_dim;
    let sl = params.seq_len;
    let scale = 1.0 / sqrt(f32(hd));

    // Phase 1: compute attention scores (Q · K^T) / sqrt(d_k)
    var idx = tid;
    loop {
        if (idx >= sl) { break; }
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dot = dot + query[d] * key[idx * hd + d];
        }
        scores[idx] = dot * scale;
        idx = idx + 256u;
    }
    workgroupBarrier();

    // Phase 2: softmax over scores — find max
    var local_max: f32 = -3.402823e+38;
    idx = tid;
    loop {
        if (idx >= sl) { break; }
        if (scores[idx] > local_max) { local_max = scores[idx]; }
        idx = idx + 256u;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {
            shared_max[tid] = shared_max[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let max_score = shared_max[0];
    workgroupBarrier();

    // Phase 3: exp and sum
    var local_sum: f32 = 0.0;
    idx = tid;
    loop {
        if (idx >= sl) { break; }
        let e = exp(scores[idx] - max_score);
        scores[idx] = e;
        local_sum = local_sum + e;
        idx = idx + 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let total = shared_sum[0];
    workgroupBarrier();

    // Normalize scores
    idx = tid;
    loop {
        if (idx >= sl) { break; }
        scores[idx] = scores[idx] / total;
        idx = idx + 256u;
    }
    workgroupBarrier();

    // Phase 4: weighted sum of values
    let d_out = wid.x; // each workgroup handles one output dimension
    if (d_out < hd) {
        var acc: f32 = 0.0;
        for (var s: u32 = 0u; s < sl; s = s + 1u) {
            acc = acc + scores[s] * value[s * hd + d_out];
        }
        output[d_out] = acc;
// Scaled dot-product attention compute shader for WebGPU inference.
// Computes: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
// Fused kernel: computes QK^T scores, applies causal mask, softmax, then V mul.
// Each workgroup handles one (batch, head, query_pos) slice.

struct Params {
    seq_len: u32,
    kv_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> scores: array<f32, 256>;
var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.y;
    let query_pos = wid.x;
    let local_id = lid.x;

    if (query_pos >= params.seq_len || head >= params.n_heads) {
        return;
    }

    let scale = 1.0 / sqrt(f32(params.head_dim));
    let q_offset = (query_pos * params.n_heads + head) * params.head_dim;

    // Step 1: Compute QK^T scores with causal mask
    // Each thread computes score for one or more KV positions
    var local_max: f32 = NEG_INF;
    for (var kv: u32 = local_id; kv < params.kv_len; kv = kv + WG_SIZE) {
        let k_offset = (kv * params.n_heads + head) * params.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot = dot + q[q_offset + d] * k[k_offset + d];
        }
        dot = dot * scale;

        // Causal mask: mask future positions
        if (kv > query_pos) {
            dot = NEG_INF;
        }

        scores[kv % WG_SIZE] = dot;
        local_max = max(local_max, dot);
    }
    shared_val[local_id] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_id < s) {
            shared_val[local_id] = max(
                shared_val[local_id],
                shared_val[local_id + s],
            );
        }
        workgroupBarrier();
    }
    let row_max = shared_val[0];
    workgroupBarrier();

    // Step 2: Compute exp(score - max) and sum
    var local_sum: f32 = 0.0;
    for (var kv: u32 = local_id; kv < params.kv_len; kv = kv + WG_SIZE) {
        let k_offset = (kv * params.n_heads + head) * params.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot = dot + q[q_offset + d] * k[k_offset + d];
        }
        dot = dot * scale;
        if (kv > query_pos) {
            dot = NEG_INF;
        }

        let exp_val = exp(dot - row_max);
        scores[kv % WG_SIZE] = exp_val;
        local_sum = local_sum + exp_val;
    }
    shared_val[local_id] = local_sum;
    workgroupBarrier();

    // Reduce for sum
    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_id < s) {
            shared_val[local_id] = shared_val[local_id]
                                 + shared_val[local_id + s];
        }
        workgroupBarrier();
    }
    let total_sum = shared_val[0];
    let inv_sum = 1.0 / total_sum;
    workgroupBarrier();

    // Step 3: Weighted sum over V
    let out_offset = (query_pos * params.n_heads + head) * params.head_dim;
    for (var d: u32 = local_id; d < params.head_dim; d = d + WG_SIZE) {
        var acc: f32 = 0.0;
        for (var kv: u32 = 0u; kv < params.kv_len; kv = kv + 1u) {
            let k_offset2 = (kv * params.n_heads + head) * params.head_dim;
            // Recompute normalized attention weight for this kv position
            var dot2: f32 = 0.0;
            for (var dd: u32 = 0u; dd < params.head_dim; dd = dd + 1u) {
                dot2 = dot2 + q[q_offset + dd] * k[k_offset2 + dd];
            }
            dot2 = dot2 * scale;
            if (kv > query_pos) {
                dot2 = NEG_INF;
            }
            let w = exp(dot2 - row_max) * inv_sum;
            acc = acc + w * v[k_offset2 + d];
        }
        output[out_offset + d] = acc;
    }
}
