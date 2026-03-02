//! Portable scalar softmax implementations.
//!
//! These are used as the default backend on non-`aarch64` targets and serve as
//! the reference implementations against which NEON results are validated.

/// In-place softmax using the max-subtraction trick.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Log-softmax via the log-sum-exp trick.
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = x.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
    x.iter().map(|&v| v - log_sum_exp).collect()
}

/// Temperature-scaled softmax.
pub fn temperature_softmax(x: &[f32], temperature: f32) -> Vec<f32> {
    let inv_temp = 1.0 / temperature;
    let scaled: Vec<f32> = x.iter().map(|&v| v * inv_temp).collect();
    let mut result = scaled;
    softmax_inplace(&mut result);
    result
}

/// Online (single-pass) softmax.
///
/// Pass 1: compute running max and normalisation factor.
/// Pass 2: normalise.
pub fn online_softmax(x: &[f32]) -> Vec<f32> {
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;

    for &v in x {
        if v > running_max {
            // Rescale accumulated sum when max changes.
            running_sum = running_sum.mul_add((running_max - v).exp(), 1.0);
            running_max = v;
        } else {
            running_sum += (v - running_max).exp();
        }
    }

    let inv_sum = 1.0 / running_sum;
    x.iter().map(|&v| (v - running_max).exp() * inv_sum).collect()
}
