// RMSNorm: output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input²) + eps)

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
    }
}
