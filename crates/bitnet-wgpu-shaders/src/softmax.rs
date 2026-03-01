//! Softmax WGSL compute shaders.
//!
//! Row-wise softmax and log-softmax with shared-memory reductions for
//! numerical stability (online max subtraction).

/// Row-wise softmax with online max subtraction.
///
/// Each workgroup processes one row. Workgroup size: [256, 1, 1].
/// Uses shared memory for parallel max-reduction and sum-reduction.
///
/// `params.rows` = number of rows, `params.cols` = row width.
/// Input layout is row-major: `input[row * cols + col]`.
pub const SOFTMAX_SRC: &str = r"
struct SoftmaxParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let row_offset = row * params.cols;

    // Phase 1: find row maximum via parallel reduction
    var local_max: f32 = -3.402823466e+38;  // -FLT_MAX
    var col = tid;
    while col < params.cols {
        local_max = max(local_max, input[row_offset + col]);
        col = col + WG_SIZE;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    var stride: u32 = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Phase 2: compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    col = tid;
    while col < params.cols {
        let val = exp(input[row_offset + col] - row_max);
        output[row_offset + col] = val;
        local_sum = local_sum + val;
        col = col + WG_SIZE;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    stride = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = shared_sum[0];
    workgroupBarrier();

    // Phase 3: normalize
    let inv_sum = 1.0 / row_sum;
    col = tid;
    while col < params.cols {
        output[row_offset + col] = output[row_offset + col] * inv_sum;
        col = col + WG_SIZE;
    }
}
";

/// Numerically stable log-softmax: log(softmax(x)).
///
/// Computed as `x[i] - max - log(sum(exp(x - max)))` to avoid
/// separate softmax + log passes. Same layout/dispatch as `softmax`.
pub const LOG_SOFTMAX_SRC: &str = r"
struct SoftmaxParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn log_softmax(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let row_offset = row * params.cols;

    // Phase 1: find row maximum
    var local_max: f32 = -3.402823466e+38;
    var col = tid;
    while col < params.cols {
        local_max = max(local_max, input[row_offset + col]);
        col = col + WG_SIZE;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Phase 2: compute sum of exp(x - max)
    var local_sum: f32 = 0.0;
    col = tid;
    while col < params.cols {
        local_sum = local_sum + exp(input[row_offset + col] - row_max);
        col = col + WG_SIZE;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    stride = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let log_sum_exp = log(shared_sum[0]);
    workgroupBarrier();

    // Phase 3: log_softmax = x - max - log(sum_exp)
    col = tid;
    while col < params.cols {
        output[row_offset + col] = input[row_offset + col] - row_max - log_sum_exp;
        col = col + WG_SIZE;
    }
}
";
