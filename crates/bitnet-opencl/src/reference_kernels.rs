//! CPU reference implementations matching GPU kernel behavior exactly.
//!
//! These implementations prioritize correctness over performance and serve
//! as the ground truth for validating GPU kernel outputs.

/// Matrix multiplication: C = A * B.
/// A is (m × k), B is (k × n), C is (m × n). All row-major.
pub fn ref_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");
    assert_eq!(c.len(), m * n, "C dimensions mismatch");

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
}

/// RMS normalization: output[i] = input[i] * weight[i] / rms(input).
pub fn ref_rmsnorm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(weight.len(), n);
    assert_eq!(output.len(), n);

    let mean_sq: f64 =
        input.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n as f64;
    let rms = (mean_sq + eps as f64).sqrt();

    for i in 0..n {
        output[i] = ((input[i] as f64 / rms) * weight[i] as f64) as f32;
    }
}

/// Rotary position embedding applied in-place to q and k.
pub fn ref_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    theta: f32,
) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), head_dim);

    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / (theta as f64).powf(2.0 * i as f64 / head_dim as f64);
        let angle = pos as f64 * freq;
        let cos_val = angle.cos() as f32;
        let sin_val = angle.sin() as f32;

        // Rotate q
        let q0 = q[i];
        let q1 = q[i + half];
        q[i] = q0 * cos_val - q1 * sin_val;
        q[i + half] = q0 * sin_val + q1 * cos_val;

        // Rotate k
        let k0 = k[i];
        let k1 = k[i + half];
        k[i] = k0 * cos_val - k1 * sin_val;
        k[i + half] = k0 * sin_val + k1 * cos_val;
    }
}

/// Softmax: output[i] = exp(input[i]) / sum(exp(input)).
/// Numerically stable via max subtraction.
pub fn ref_softmax(input: &[f32], output: &mut [f32], n: usize) {
    assert!(input.len() >= n);
    assert!(output.len() >= n);

    let max_val = input[..n]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f64;
    for i in 0..n {
        let e = ((input[i] - max_val) as f64).exp();
        output[i] = e as f32;
        sum += e;
    }
    for i in 0..n {
        output[i] = (output[i] as f64 / sum) as f32;
    }
}

/// Single-head attention: output = softmax(q·K^T / sqrt(d)) · V.
pub fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    assert_eq!(output.len(), head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: q · K^T
    let mut scores = vec![0.0f32; seq_len];
    for s in 0..seq_len {
        let mut dot = 0.0f64;
        for d in 0..head_dim {
            dot += q[d] as f64 * k[s * head_dim + d] as f64;
        }
        scores[s] = (dot as f32) * scale;
    }

    // Softmax
    let mut weights = vec![0.0f32; seq_len];
    ref_softmax(&scores, &mut weights, seq_len);

    // Weighted sum of V
    for d in 0..head_dim {
        let mut sum = 0.0f64;
        for s in 0..seq_len {
            sum += weights[s] as f64 * v[s * head_dim + d] as f64;
        }
        output[d] = sum as f32;
    }
}

/// Embedding table lookup.
pub fn ref_embedding(
    tokens: &[u32],
    table: &[f32],
    output: &mut [f32],
    dim: usize,
) {
    let vocab_size = table.len() / dim;
    assert_eq!(output.len(), tokens.len() * dim);

    for (t_idx, &tok) in tokens.iter().enumerate() {
        assert!(
            (tok as usize) < vocab_size,
            "token {} out of vocab range {}",
            tok,
            vocab_size
        );
        let src = &table[tok as usize * dim..(tok as usize + 1) * dim];
        let dst = &mut output[t_idx * dim..(t_idx + 1) * dim];
        dst.copy_from_slice(src);
    }
}

/// SiLU activation: x * sigmoid(x).
pub fn ref_silu(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        let sigmoid = 1.0 / (1.0 + (-x as f64).exp());
        output[i] = (x as f64 * sigmoid) as f32;
    }
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
pub fn ref_gelu(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    for (i, &x) in input.iter().enumerate() {
        let x64 = x as f64;
        let inner = sqrt_2_over_pi * (x64 + 0.044715 * x64 * x64 * x64);
        output[i] = (0.5 * x64 * (1.0 + inner.tanh())) as f32;
    }
}

/// Layer normalization with weight and bias.
pub fn ref_layernorm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(weight.len(), n);
    assert_eq!(bias.len(), n);
    assert_eq!(output.len(), n);

    let mean: f64 = input.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var: f64 = input
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std_inv = 1.0 / (var + eps as f64).sqrt();

    for i in 0..n {
        let norm = (input[i] as f64 - mean) * std_inv;
        output[i] = (norm * weight[i] as f64 + bias[i] as f64) as f32;
    }
}

/// Dequantize I2_S packed ternary values.
/// Each u32 contains 16 ternary values (2 bits each): 0b00=0, 0b01=+1, 0b10=-1.
pub fn ref_dequant_i2s(
    packed: &[u32],
    scale: f32,
    output: &mut [f32],
) {
    let mut out_idx = 0;
    for &word in packed {
        for bit_pos in 0..16 {
            if out_idx >= output.len() {
                return;
            }
            let val = (word >> (bit_pos * 2)) & 0x3;
            output[out_idx] = match val {
                0b00 => 0.0,
                0b01 => scale,
                0b10 => -scale,
                _ => 0.0, // 0b11 unused
            };
            out_idx += 1;
        }
    }
}

/// Dequantize QK256 format: 256-element blocks with per-block f16 scales.
/// Each u32 contains 16 ternary values (2 bits each).
pub fn ref_dequant_qk256(
    packed: &[u32],
    scales: &[u16],
    output: &mut [f32],
) {
    let words_per_block = 16; // 256 values / 16 per word
    let mut out_idx = 0;

    for (block_idx, &scale_bits) in scales.iter().enumerate() {
        let scale = f16_to_f32_simple(scale_bits);
        let block_start = block_idx * words_per_block;
        let block_end = (block_start + words_per_block).min(packed.len());

        for &word in &packed[block_start..block_end] {
            for bit_pos in 0..16 {
                if out_idx >= output.len() {
                    return;
                }
                let val = (word >> (bit_pos * 2)) & 0x3;
                output[out_idx] = match val {
                    0b00 => 0.0,
                    0b01 => scale,
                    0b10 => -scale,
                    _ => 0.0,
                };
                out_idx += 1;
            }
        }
    }
}

/// Simple f16-to-f32 conversion (same as in testing module).
fn f16_to_f32_simple(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        let val = (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let f_exp = (exp as i32) - 15 + 127;
        let f_bits = (sign << 31) | ((f_exp as u32) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}
