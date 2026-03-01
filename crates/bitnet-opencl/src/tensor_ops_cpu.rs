// Pure CPU reference implementations of tensor operations.
//
// These serve as ground truth for GPU kernel validation.
// Prioritises clarity over performance — no SIMD optimisation.

use crate::tensor_ops::{Tensor, TensorError, TensorResult, TensorShape};

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

fn require_same_shape(a: &Tensor, b: &Tensor) -> TensorResult<()> {
    if a.shape != b.shape {
        return Err(TensorError::ShapeMismatch(format!("expected {}, got {}", a.shape, b.shape,)));
    }
    Ok(())
}

fn require_2d(t: &Tensor) -> TensorResult<(usize, usize)> {
    let d = t.shape.dims();
    if d.len() != 2 {
        return Err(TensorError::ShapeMismatch(format!("expected 2-D tensor, got {}-D", d.len(),)));
    }
    Ok((d[0], d[1]))
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

/// Matrix multiplication: `[M, K] × [K, N] → [M, N]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    let (m, k1) = require_2d(a)?;
    let (k2, n) = require_2d(b)?;
    if k1 != k2 {
        return Err(TensorError::ShapeMismatch(format!(
            "matmul inner dims mismatch: {k1} vs {k2}",
        )));
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k1 {
                sum += a.data[i * k1 + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Tensor::new(TensorShape::new(&[m, n]), out)
}

/// Element-wise addition.
pub fn add(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    require_same_shape(a, b)?;
    let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(a.shape.clone(), data)
}

/// Element-wise multiplication.
pub fn mul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    require_same_shape(a, b)?;
    let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Tensor::new(a.shape.clone(), data)
}

/// Softmax along the last dimension (for a 2-D tensor, `dim` must equal 1;
/// for 1-D, `dim` must be 0).
pub fn softmax(input: &Tensor, dim: usize) -> TensorResult<Tensor> {
    if input.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    let ndim = input.shape.ndim();
    if dim >= ndim {
        return Err(TensorError::InvalidDimension { dim, ndim });
    }

    // Only last-dimension softmax is implemented.
    if dim != ndim - 1 {
        return Err(TensorError::ShapeMismatch("softmax only supports the last dimension".into()));
    }

    let inner = *input.shape.dims().last().unwrap();
    let outer = input.numel() / inner;
    let mut out = input.data.clone();

    for o in 0..outer {
        let start = o * inner;
        let end = start + inner;
        let row = &mut out[start..end];

        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }

    Tensor::new(input.shape.clone(), out)
}

/// RMS normalization: `x / rms(x) * weight`.
pub fn rmsnorm(input: &Tensor, weight: &Tensor, eps: f32) -> TensorResult<Tensor> {
    if input.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    let ndim = input.shape.ndim();
    let inner = input.shape.dim(ndim - 1)?;
    if weight.numel() != inner {
        return Err(TensorError::ShapeMismatch(format!(
            "weight length {} != last dim {inner}",
            weight.numel(),
        )));
    }

    let outer = input.numel() / inner;
    let mut out = vec![0.0f32; input.numel()];

    for o in 0..outer {
        let start = o * inner;
        let row = &input.data[start..start + inner];

        #[allow(clippy::cast_precision_loss)]
        let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / inner as f32;
        let rms = (ms + eps).sqrt();

        for (i, &x) in row.iter().enumerate() {
            out[start + i] = (x / rms) * weight.data[i];
        }
    }
    Tensor::new(input.shape.clone(), out)
}

/// Rotary position embedding (real-valued pair rotation).
///
/// `input` shape: `[..., seq_len, dim]` where `dim` is even.
/// `freqs` shape: `[seq_len, dim/2]` — angles in radians.
pub fn rope(input: &Tensor, freqs: &Tensor) -> TensorResult<Tensor> {
    let ndim = input.shape.ndim();
    if ndim < 2 {
        return Err(TensorError::ShapeMismatch("rope requires at least 2-D input".into()));
    }
    let dim = input.shape.dim(ndim - 1)?;
    let seq = input.shape.dim(ndim - 2)?;
    if dim % 2 != 0 {
        return Err(TensorError::ShapeMismatch("rope requires even last dimension".into()));
    }
    let half = dim / 2;
    let (fs, fh) = require_2d(freqs)?;
    if fs != seq || fh != half {
        return Err(TensorError::ShapeMismatch(format!(
            "freqs shape [{fs}, {fh}] doesn't match [{seq}, {half}]",
        )));
    }

    let outer = input.numel() / (seq * dim);
    let mut out = input.data.clone();

    for o in 0..outer {
        for s in 0..seq {
            let base = o * seq * dim + s * dim;
            for h in 0..half {
                let cos_v = freqs.data[s * half + h].cos();
                let sin_v = freqs.data[s * half + h].sin();
                let x0 = input.data[base + h];
                let x1 = input.data[base + half + h];
                out[base + h] = x0.mul_add(cos_v, -(x1 * sin_v));
                out[base + half + h] = x0.mul_add(sin_v, x1 * cos_v);
            }
        }
    }
    Tensor::new(input.shape.clone(), out)
}

/// `SiLU` activation: `x * sigmoid(x)`.
pub fn silu(input: &Tensor) -> TensorResult<Tensor> {
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        })
        .collect();
    Tensor::new(input.shape.clone(), data)
}

/// Scaled dot-product attention.
///
/// `q`, `k`, `v` shapes: `[batch, heads, seq, head_dim]` (4-D) or
/// `[seq, head_dim]` (2-D, single-head shortcut).
///
/// `mask` (optional): additive mask broadcastable to `[seq_q, seq_k]`.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> TensorResult<Tensor> {
    // Support simple 2-D case: [seq, head_dim].
    let (sq, dk) = require_2d(q)?;
    let (sk, dk2) = require_2d(k)?;
    let (sv, dv) = require_2d(v)?;

    if dk != dk2 {
        return Err(TensorError::ShapeMismatch("q and k head_dim mismatch".into()));
    }
    if sk != sv {
        return Err(TensorError::ShapeMismatch("k and v seq_len mismatch".into()));
    }

    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / (dk as f32).sqrt();

    // scores = q @ k^T  → [sq, sk]
    let mut scores = vec![0.0f32; sq * sk];
    for i in 0..sq {
        for j in 0..sk {
            let mut dot = 0.0f32;
            for p in 0..dk {
                dot += q.data[i * dk + p] * k.data[j * dk + p];
            }
            scores[i * sk + j] = dot * scale;
        }
    }

    // Apply mask (additive).
    if let Some(m) = mask {
        let (mr, mc) = require_2d(m)?;
        if mr != sq || mc != sk {
            return Err(TensorError::ShapeMismatch(format!(
                "mask shape [{mr}, {mc}] doesn't match [{sq}, {sk}]",
            )));
        }
        for (s, mv) in scores.iter_mut().zip(&m.data) {
            *s += mv;
        }
    }

    // Row-wise softmax over scores.
    for i in 0..sq {
        let row = &mut scores[i * sk..(i + 1) * sk];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }

    // out = scores @ v  → [sq, dv]
    let mut out = vec![0.0f32; sq * dv];
    for i in 0..sq {
        for j in 0..dv {
            let mut sum = 0.0f32;
            for p in 0..sk {
                sum += scores[i * sk + p] * v.data[p * dv + j];
            }
            out[i * dv + j] = sum;
        }
    }

    Tensor::new(TensorShape::new(&[sq, dv]), out)
}

/// Embedding lookup: each id in `input_ids` selects a row from `table`.
///
/// `table` shape: `[vocab_size, embed_dim]`.
/// Returns `[len(input_ids), embed_dim]`.
pub fn embedding(input_ids: &[u32], table: &Tensor) -> TensorResult<Tensor> {
    let (vocab, embed) = require_2d(table)?;
    let mut out = Vec::with_capacity(input_ids.len() * embed);
    for &id in input_ids {
        let id = id as usize;
        if id >= vocab {
            return Err(TensorError::ShapeMismatch(format!(
                "embedding id {id} >= vocab size {vocab}",
            )));
        }
        let start = id * embed;
        out.extend_from_slice(&table.data[start..start + embed]);
    }
    Tensor::new(TensorShape::new(&[input_ids.len(), embed]), out)
}

/// Linear projection: `input @ weight^T [+ bias]`.
///
/// `input` shape: `[batch, in_features]`.
/// `weight` shape: `[out_features, in_features]`.
/// `bias` shape (optional): `[out_features]`.
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> TensorResult<Tensor> {
    let (batch, in_f) = require_2d(input)?;
    let (out_f, in_f2) = require_2d(weight)?;
    if in_f != in_f2 {
        return Err(TensorError::ShapeMismatch(format!(
            "linear in_features mismatch: {in_f} vs {in_f2}",
        )));
    }
    if let Some(b) = bias
        && b.numel() != out_f
    {
        return Err(TensorError::ShapeMismatch(format!(
            "bias length {} != out_features {out_f}",
            b.numel(),
        )));
    }

    // out = input @ weight^T
    let mut out = vec![0.0f32; batch * out_f];
    for i in 0..batch {
        for j in 0..out_f {
            let mut sum = 0.0f32;
            for p in 0..in_f {
                sum += input.data[i * in_f + p] * weight.data[j * in_f + p];
            }
            if let Some(b) = bias {
                sum += b.data[j];
            }
            out[i * out_f + j] = sum;
        }
    }
    Tensor::new(TensorShape::new(&[batch, out_f]), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2d(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
        Tensor::new(TensorShape::new(&[rows, cols]), data).unwrap()
    }

    #[test]
    fn matmul_identity() {
        let a = t2d(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let eye = t2d(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let c = matmul(&a, &eye).unwrap();
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn silu_zero() {
        let t = Tensor::new(TensorShape::new(&[3]), vec![0.0, 0.0, 0.0]).unwrap();
        let r = silu(&t).unwrap();
        assert!(r.data.iter().all(|&v| v == 0.0));
    }
}
