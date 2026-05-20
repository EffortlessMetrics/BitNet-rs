//! OpenCL kernel source strings for Intel GPU acceleration.
//!
//! These constants contain the `.cl` kernel sources that are compiled at runtime
//! by the OpenCL driver. They are always available (not feature-gated) so that
//! tests can validate kernel correctness without requiring GPU hardware.

/// OpenCL kernel source for ternary (I2_S) matrix multiplication.
///
/// Computes C = A x B where:
/// - A is an [M x K] matrix of int8 activations (`char`)
/// - B is a [K/4 x N] matrix of packed 2-bit I2_S weights (`uchar`)
/// - C is the `float` output
///
/// Ternary encoding: 0b00 = 0, 0b01 = +1, 0b11 = -1.
pub const MATMUL_I2S_SRC: &str = r#"
__kernel void matmul_i2s(
    __global const char* A,
    __global const uchar* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const uint k_packed = K / 4;

    for (uint kp = 0; kp < k_packed; kp++) {
        uchar packed = B[kp * N + col];

        for (uint sub = 0; sub < 4; sub++) {
            uint k_idx = kp * 4 + sub;
            if (k_idx >= K) break;

            uchar bits = (packed >> (sub * 2)) & 0x03;

            int w;
            if (bits == 0x01) {
                w = 1;
            } else if (bits == 0x03) {
                w = -1;
            } else {
                w = 0;
            }

            char a_val = A[row * K + k_idx];
            sum += (float)a_val * (float)w;
        }
    }

    C[row * N + col] = sum;
}
"#;

/// OpenCL kernel source for a QK256 grouped-layout I2_S × I8_S scaled GEMV
/// fixture.
///
/// This fixture consumes already-quantized I8_S activations plus activation
/// scale/sum metadata and GGML grouped QK256 I2_S weight bytes. It exercises
/// the production BitNet scale/sum formula, but it does not prove GPU-resident
/// activation quantization, transformer dispatch, or full BitNet inference.
pub const QK256_I2S_I8S_SCALED_GEMV_SRC: &str = r#"
__kernel void qk256_i2s_i8s_scaled_gemv(
    __global const char* q,
    __global const uchar* qs,
    __global float* y,
    const uint rows,
    const uint cols,
    const uint row_stride_bytes,
    const int activation_sum,
    const float activation_scale,
    const float weight_scale
) {
    const uint row = get_global_id(0);
    if (row >= rows) return;

    int int_dot = 0;
    const uint row_base = row * row_stride_bytes;

    for (uint col = 0; col < cols; col++) {
        const uint block = col / 256;
        const uint offset = col - block * 256;
        const uint chunk = offset / 128;
        const uint lane = (offset - chunk * 128) / 32;
        const uint gp = offset & 31;
        const uint byte_index = row_base + block * 64 + chunk * 32 + gp;
        const uchar packed = qs[byte_index];
        const uchar code = (packed >> (6 - lane * 2)) & 0x03;
        int_dot += ((int)code) * ((int)q[col]);
    }

    y[row] = (((float)(int_dot - activation_sum)) / activation_scale) * weight_scale;
}
"#;

/// OpenCL kernel source for I2_S quantization.
///
/// Quantizes `float` activations into 2-bit ternary values packed 4-per-byte,
/// computing per-block scales from the absolute maximum.
///
/// Ternary encoding: +1 → 0b01 (1), −1 → 0b11 (3), 0 → 0b00 (0).
pub const QUANTIZE_I2S_SRC: &str = r#"
__kernel void quantize_i2s(
    __global const float* input,
    __global uchar* output,
    __global float* scales,
    const uint N,
    const uint block_size
) {
    uint block_id = get_global_id(0);
    uint block_start = block_id * block_size;
    if (block_start >= N) return;

    uint block_end = min(block_start + block_size, N);

    // Step 1: compute absmax for this block
    float absmax = 0.0f;
    for (uint i = block_start; i < block_end; i++) {
        absmax = fmax(absmax, fabs(input[i]));
    }

    // Step 2: compute scale, guard against zero
    float scale;
    if (absmax > 0.0f) {
        scale = absmax / 1.5f;
    } else {
        scale = 1.0f;
    }
    scales[block_id] = scale;

    // Step 3: quantize and pack 4 values per byte
    for (uint i = block_start; i < block_end; i += 4) {
        uchar packed = 0;
        for (uint j = 0; j < 4 && (i + j) < block_end; j++) {
            float normalized = input[i + j] / scale;
            uchar ternary;
            if (normalized > 0.5f) {
                ternary = 1;
            } else if (normalized < -0.5f) {
                ternary = 3;
            } else {
                ternary = 0;
            }
            packed |= (ternary << (j * 2));
        }
        output[(i - block_start) / 4 + (block_start / 4)] = packed;
    }
}
"#;

/// OpenCL kernel sources for elementwise operations (vec_add, silu, rms_norm, softmax).
pub const ELEMENTWISE_SRC: &str = r#"
__kernel void vec_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const uint N
) {
    uint i = get_global_id(0);
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

__kernel void silu(
    __global const float* input,
    __global float* output,
    const uint N
) {
    uint i = get_global_id(0);
    if (i < N) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + exp(-x));
        output[i] = x * sigmoid;
    }
}

__kernel void rms_norm(
    __global const float* input,
    __global const float* weight,
    __global float* output,
    const uint N,
    const float eps
) {
    // Compute mean of squares
    float sum_sq = 0.0f;
    for (uint i = 0; i < N; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = rsqrt(sum_sq / (float)N + eps);

    uint i = get_global_id(0);
    if (i < N) {
        output[i] = input[i] * rms * weight[i];
    }
}

__kernel void softmax(
    __global const float* input,
    __global float* output,
    const uint N
) {
    // Find max for numerical stability
    float max_val = input[0];
    for (uint i = 1; i < N; i++) {
        max_val = fmax(max_val, input[i]);
    }

    // Compute exp sum
    float sum = 0.0f;
    for (uint i = 0; i < N; i++) {
        sum += exp(input[i] - max_val);
    }

    uint i = get_global_id(0);
    if (i < N) {
        output[i] = exp(input[i] - max_val) / sum;
    }
}
"#;

/// OpenCL kernel source for scaled dot-product attention.
///
/// Three kernels:
/// - `attention_scores`: computes QK^T / sqrt(d_k) with optional causal masking
/// - `attention_softmax`: numerically stable row-wise softmax with tree reduction
/// - `attention_weighted_sum`: computes attention_weights × V
pub const ATTENTION_SRC: &str = include_str!("gpu/kernels/attention.cl");

/// OpenCL kernel source for RMSNorm and LayerNorm normalization.
///
/// Two kernels optimized for Intel Arc A770 (Xe-HPG) with tree reductions:
/// - `rmsnorm`: RMSNorm(x) = x * rsqrt(mean(x²) + eps) * weight
/// - `layernorm`: LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
pub const NORMALIZATION_SRC: &str = include_str!("gpu/kernels/normalization.cl");

/// OpenCL kernel source for activation functions and softmax.
///
/// Eight kernels:
/// - `silu`: SiLU (Swish) activation x * σ(x) for LLaMA FFN
/// - `silu_mul`: fused SiLU + elementwise multiply (gate * up pattern)
/// - `gelu`: GELU activation (tanh approximation)
/// - `relu`: ReLU activation
/// - `elementwise_add`: vector addition
/// - `elementwise_mul`: vector multiplication
/// - `scale`: scalar multiplication (in-place)
/// - `softmax_full`: numerically stable softmax with tree reduction
pub const ACTIVATIONS_SRC: &str = include_str!("gpu/kernels/activations.cl");

/// OpenCL kernel source for Rotary Position Embedding (RoPE).
///
/// Two kernels optimized for Intel Arc A770:
/// - `rope_apply`: real-time sin/cos computation per position
/// - `rope_apply_cached`: pre-computed frequency table for repeated calls
///
/// Supports KV cache continuation via `position_offset`.
pub const ROPE_SRC: &str = include_str!("gpu/kernels/rope.cl");

/// OpenCL kernel source for tiled GEMM and quantized GEMV.
///
/// Two kernels optimized for Intel Arc A770 Xe-HPG:
/// - `tiled_matmul_f32`: 16×16 tiled GEMM using shared local memory
/// - `quantized_gemv_i2s`: I2_S 2-bit packed weight GEMV with per-row scales
pub const TILED_MATMUL_SRC: &str = include_str!("gpu/kernels/tiled_matmul.cl");
