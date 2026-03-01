// Row-wise softmax: output[i] = exp(input[i] - max) / Σexp(input - max)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params { n: u32 }
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
// Numerically stable softmax compute shader for WebGPU inference.
// Three-pass approach: (1) find max, (2) compute exp sum, (3) normalize.
// Operates on rows of length `n` from a flattened (rows × n) input.

struct Params {
    rows: u32,
    n: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let row_start = row * params.n;

    // Find row max
    var local_max: f32 = -3.402823e+38;
    var idx = tid;
    loop {
        if (idx >= params.n) { break; }
        let val = input[row_start + idx];
        if (val > local_max) { local_max = val; }
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
    let row_max = shared_max[0];
    workgroupBarrier();

    // Compute exp and sum
    var local_sum: f32 = 0.0;
    idx = tid;
    loop {
        if (idx >= params.n) { break; }
        let e = exp(input[row_start + idx] - row_max);
        output[row_start + idx] = e;
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

    // Normalize
    idx = tid;
    loop {
        if (idx >= params.n) { break; }
        output[row_start + idx] = output[row_start + idx] / total;
        idx = idx + 256u;
    let local_id = lid.x;

    if (row >= params.rows) {
        return;
    }

    let row_offset = row * params.n;

    // Pass 1: Find max value in row (parallel reduction)
    var local_max: f32 = -3.402823e+38; // -FLT_MAX
    for (var i: u32 = local_id; i < params.n; i = i + WG_SIZE) {
        local_max = max(local_max, input[row_offset + i]);
    }
    shared_data[local_id] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            shared_data[local_id] = max(
                shared_data[local_id],
                shared_data[local_id + stride],
            );
        }
        workgroupBarrier();
    }
    let row_max = shared_data[0];
    workgroupBarrier();

    // Pass 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = local_id; i < params.n; i = i + WG_SIZE) {
        let val = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = val;
        local_sum = local_sum + val;
    }
    shared_data[local_id] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            shared_data[local_id] = shared_data[local_id]
                                  + shared_data[local_id + stride];
        }
        workgroupBarrier();
    }
    let row_sum = shared_data[0];
    workgroupBarrier();

    // Pass 3: Normalize
    let inv_sum = 1.0 / row_sum;
    for (var i: u32 = local_id; i < params.n; i = i + WG_SIZE) {
        output[row_offset + i] = output[row_offset + i] * inv_sum;
    }
}
