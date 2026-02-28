// Row-wise softmax: result[i] = exp(x[i] - max) / sum(exp(x - max))
// Operates on a single row of length N.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params { n: u32 }
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let row_start = row * params.n;

    // Phase 1: find row max
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

    // Phase 2: compute exp and sum
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

    // Phase 3: normalize
    idx = tid;
    loop {
        if (idx >= params.n) { break; }
        output[row_start + idx] = output[row_start + idx] / total;
        idx = idx + 256u;
    }
}
