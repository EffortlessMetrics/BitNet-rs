// RMSNorm: output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input²) + eps)
// RMS normalization compute shader for WebGPU inference.
// Computes: output[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
// Each workgroup processes one row of length `n`.

struct Params {
    rows: u32,
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params { n: u32, eps: f32 }
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.n;

    // Phase 1: compute sum of squares
    var local_sum: f32 = 0.0;
    var idx = tid;
    loop {
        if (idx >= n) { break; }
        let v = input[idx];
        local_sum = local_sum + v * v;
        idx = idx + 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Parallel reduction
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let rms = sqrt(shared_sum[0] / f32(n) + params.eps);
    workgroupBarrier();

    // Phase 2: normalize and scale
    idx = tid;
    loop {
        if (idx >= n) { break; }
        output[idx] = (input[idx] / rms) * weight[idx];
        idx = idx + 256u;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    let local_id = lid.x;

    if (row >= params.rows) {
        return;
    }

    let row_offset = row * params.n;

    // Compute sum of squares (parallel reduction)
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = local_id; i < params.n; i = i + WG_SIZE) {
        let val = input[row_offset + i];
        local_sum_sq = local_sum_sq + val * val;
    }
    shared_data[local_id] = local_sum_sq;
    workgroupBarrier();

    // Tree reduction for sum of squares
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            shared_data[local_id] = shared_data[local_id]
                                  + shared_data[local_id + stride];
        }
        workgroupBarrier();
    }

    let rms = sqrt(shared_data[0] / f32(params.n) + params.eps);
    let inv_rms = 1.0 / rms;
    workgroupBarrier();

    // Normalize and apply weight
    for (var i: u32 = local_id; i < params.n; i = i + WG_SIZE) {
        output[row_offset + i] = input[row_offset + i] * inv_rms
                                * weight[i];
    }
}
