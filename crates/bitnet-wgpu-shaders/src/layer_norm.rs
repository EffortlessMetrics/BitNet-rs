//! Layer normalization WGSL compute shaders.
//!
//! Provides standard `LayerNorm` (with gamma/beta) and RMS normalization
//! (LLaMA-style, gamma only). Each workgroup processes one row.

/// Layer normalization with learnable gamma (weight) and beta (bias).
///
/// y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
///
/// Each workgroup handles one row. Workgroup size: [256, 1, 1].
pub const LAYER_NORM_SRC: &str = r"
struct LayerNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: LayerNormParams;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sq_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn layer_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let row_offset = row * params.cols;

    // Phase 1: compute sum and squared sum for mean and variance
    var local_sum: f32 = 0.0;
    var local_sq_sum: f32 = 0.0;
    var col = tid;
    while col < params.cols {
        let val = input[row_offset + col];
        local_sum = local_sum + val;
        local_sq_sum = local_sq_sum + val * val;
        col = col + WG_SIZE;
    }
    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;
    workgroupBarrier();

    // Tree reduction
    var stride: u32 = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
            shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let n = f32(params.cols);
    let mean = shared_sum[0] / n;
    let variance = shared_sq_sum[0] / n - mean * mean;
    let inv_std = 1.0 / sqrt(variance + params.eps);
    workgroupBarrier();

    // Phase 2: normalize with gamma and beta
    col = tid;
    while col < params.cols {
        let normalized = (input[row_offset + col] - mean) * inv_std;
        output[row_offset + col] = gamma[col] * normalized + beta[col];
        col = col + WG_SIZE;
    }
}
";

/// RMS normalization (LLaMA-style): y[i] = gamma[i] * x[i] / rms(x).
///
/// rms(x) = sqrt(mean(x²) + eps). No mean subtraction or beta.
/// Each workgroup handles one row. Workgroup size: [256, 1, 1].
pub const RMS_NORM_SRC: &str = r"
struct RmsNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RmsNormParams;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sq_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rms_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let row_offset = row * params.cols;

    // Phase 1: compute sum of squares
    var local_sq_sum: f32 = 0.0;
    var col = tid;
    while col < params.cols {
        let val = input[row_offset + col];
        local_sq_sum = local_sq_sum + val * val;
        col = col + WG_SIZE;
    }
    shared_sq_sum[tid] = local_sq_sum;
    workgroupBarrier();

    // Tree reduction for sum of squares
    var stride: u32 = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let rms = sqrt(shared_sq_sum[0] / f32(params.cols) + params.eps);
    let inv_rms = 1.0 / rms;
    workgroupBarrier();

    // Phase 2: scale by gamma / rms
    col = tid;
    while col < params.cols {
        output[row_offset + col] = gamma[col] * input[row_offset + col] * inv_rms;
        col = col + WG_SIZE;
    }
}
";
