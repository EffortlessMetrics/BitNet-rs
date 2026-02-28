//! OpenCL C kernel source strings for BitNet operations.
//!
//! Each constant holds the full OpenCL C source for one kernel entry-point.
//! These are compiled at runtime via `clCreateProgramWithSource` (or the
//! equivalent SPIR-V path on oneAPI).

/// OpenCL C source for the QK256 dequantization + GEMV kernel.
///
/// The kernel processes 256-element quantised blocks (2 bits per weight,
/// packed 4 per byte with a per-block `half` scale factor).
pub const QK256_GEMV_SOURCE: &str = r#"
__kernel void qk256_gemv(
    __global const uchar* packed_weights,
    __global const half*  scales,
    __global const float* input,
    __global float*       output,
    const uint seq_len,
    const uint n_out,
    const uint k)
{
    const uint gid = get_global_id(0);
    if (gid >= n_out) return;

    const uint blocks_per_row = k / 256;
    float acc = 0.0f;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        const uint base = gid * blocks_per_row * 64 + blk * 64;
        half scale = scales[gid * blocks_per_row + blk];

        for (uint i = 0; i < 64; ++i) {
            uchar packed = packed_weights[base + i];
            for (uint j = 0; j < 4; ++j) {
                int w = ((packed >> (j * 2)) & 0x3) - 1;
                uint idx = blk * 256 + i * 4 + j;
                acc += (float)w * input[idx] * (float)scale;
            }
        }
    }
    output[gid] = acc;
}
"#;

/// OpenCL C source for the RMSNorm kernel.
pub const RMSNORM_SOURCE: &str = r#"
__kernel void rmsnorm(
    __global const float* input,
    __global const float* gamma,
    __global float*       output,
    const uint hidden_dim,
    const float eps)
{
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint wg  = get_local_size(0);

    __local float partial_sum;
    if (lid == 0) partial_sum = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_sum = 0.0f;
    for (uint i = lid; i < hidden_dim; i += wg) {
        float x = input[row * hidden_dim + i];
        local_sum += x * x;
    }

    /* Simple serial reduction (placeholder for tree reduce) */
    atomic_add_local(&partial_sum, local_sum);
    barrier(CLK_LOCAL_MEM_FENCE);

    float rms = sqrt(partial_sum / (float)hidden_dim + eps);
    float inv_rms = 1.0f / rms;

    for (uint i = lid; i < hidden_dim; i += wg) {
        output[row * hidden_dim + i] =
            input[row * hidden_dim + i] * inv_rms * gamma[i];
    }
}
"#;

/// OpenCL C source for the scaled dot-product attention kernel (naive).
pub const ATTENTION_SOURCE: &str = r#"
__kernel void attention(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global float*       output,
    const uint n_heads,
    const uint seq_len_q,
    const uint seq_len_kv,
    const uint head_dim,
    const float scale,
    const uint causal)
{
    const uint head = get_group_id(1);
    const uint qi   = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (qi >= seq_len_q) return;

    const uint q_offset = head * seq_len_q  * head_dim;
    const uint k_offset = head * seq_len_kv * head_dim;
    const uint v_offset = k_offset;

    /* Compute attention scores: softmax(Q·Kᵀ / scale) · V */
    float max_score = -INFINITY;
    for (uint ki = 0; ki < seq_len_kv; ++ki) {
        if (causal && ki > qi) break;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += q[q_offset + qi * head_dim + d]
                 * k[k_offset + ki * head_dim + d];
        }
        float s = dot * scale;
        if (s > max_score) max_score = s;
    }

    float sum_exp = 0.0f;
    for (uint ki = 0; ki < seq_len_kv; ++ki) {
        if (causal && ki > qi) break;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += q[q_offset + qi * head_dim + d]
                 * k[k_offset + ki * head_dim + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }

    for (uint d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        for (uint ki = 0; ki < seq_len_kv; ++ki) {
            if (causal && ki > qi) break;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; ++dd) {
                dot += q[q_offset + qi * head_dim + dd]
                     * k[k_offset + ki * head_dim + dd];
            }
            float w = exp(dot * scale - max_score) / sum_exp;
            out_val += w * v[v_offset + ki * head_dim + d];
        }
        output[q_offset + qi * head_dim + d] = out_val;
    }
}
"#;

/// All kernel source strings, keyed by name.
pub fn all_kernel_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("qk256_gemv", QK256_GEMV_SOURCE),
        ("rmsnorm", RMSNORM_SOURCE),
        ("attention", ATTENTION_SOURCE),
    ]
}
