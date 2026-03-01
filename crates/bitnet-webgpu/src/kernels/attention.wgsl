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
    }
}
