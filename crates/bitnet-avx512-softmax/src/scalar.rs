//! Scalar fallback implementations of all softmax variants.
//!
//! These are used on CPUs that lack AVX-512F support and also serve as the
//! reference implementation for correctness testing.

/// Numerically-stable softmax in-place (max-subtract trick).
pub fn softmax_inplace(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv_sum = 1.0 / sum;
    for x in xs.iter_mut() {
        *x *= inv_sum;
    }
}

/// Online (single-pass) softmax.
///
/// Uses the online normalisation algorithm:
/// maintain running `max` and `sum` and rescale when a new maximum is found.
pub fn online_softmax(logits: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;

    // Forward pass: compute running max and compensated sum.
    for &v in logits {
        if v > max {
            // Rescale accumulated sum for the new maximum.
            sum = sum.mul_add((max - v).exp(), 1.0);
            max = v;
        } else {
            sum += (v - max).exp();
        }
    }

    // Backward pass: compute probabilities.
    let inv_sum = 1.0 / sum;
    logits.iter().map(|&v| (v - max).exp() * inv_sum).collect()
}

/// Log-softmax: `log(softmax(x))` computed in a numerically stable way.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    logits.iter().map(|&v| v - log_sum_exp).collect()
}

/// Temperature-scaled softmax.
pub fn temperature_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let mut scaled: Vec<f32> = logits.iter().map(|&v| v * inv_t).collect();
    softmax_inplace(&mut scaled);
    scaled
}

/// Masked softmax: positions where `mask[i] == false` receive zero probability.
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let mut masked: Vec<f32> = logits
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m { v } else { f32::NEG_INFINITY })
        .collect();
    softmax_inplace(&mut masked);
    // Replace NaN (from -inf/-inf) with 0.
    for v in &mut masked {
        if v.is_nan() {
            *v = 0.0;
        }
    }
    masked
}

/// Batched softmax in-place: apply [`softmax_inplace`] to each row.
pub fn batch_softmax_inplace(logits: &mut [f32], row_len: usize) {
    for row in logits.chunks_exact_mut(row_len) {
        softmax_inplace(row);
    }
}
