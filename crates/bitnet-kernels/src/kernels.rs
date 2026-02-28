//! OpenCL kernel source strings for Intel GPU acceleration.
//!
//! These constants contain the `.cl` kernel sources that are compiled at runtime
//! by the OpenCL driver. They are always available (not feature-gated) so that
//! tests can validate kernel correctness without requiring GPU hardware.

/// OpenCL kernel source for ternary (I2_S) matrix multiplication.
///
/// Computes C = A × B where:
/// - A is packed 2-bit ternary weights (`char`, 4 values per byte)
/// - B is activation vectors (`uchar`)
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
    uint row = get_global_id(0);
    uint col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        // A is packed: 4 ternary values per byte
        uint byte_idx = (row * K + i) / 4;
        uint sub = (row * K + i) % 4;
        uchar packed = (uchar)A[byte_idx];
        uchar bits = (packed >> (sub * 2)) & 0x03;

        // Decode ternary: 0x01 -> +1, 0x03 -> -1, else 0
        int w;
        if (bits == 0x01) {
            w = 1;
        } else if (bits == 0x03) {
            w = -1;
        } else {
            w = 0;
        }

        sum += (float)w * (float)B[i * N + col];
    }

    C[row * N + col] = sum;
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
