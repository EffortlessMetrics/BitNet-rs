//! Probability-domain primitives for decode-time sampling.
//!
//! This crate intentionally provides small pure functions that can be reused
//! across sampling implementations.

/// Renormalize a probability vector in place so it sums to exactly 1.0.
///
/// Returns `true` when normalization was applied, and `false` if `probs`
/// could not be normalized (empty or non-positive sum).
pub fn renormalize_in_place(probs: &mut [f32]) -> bool {
    if probs.is_empty() {
        return false;
    }

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return false;
    }

    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    true
}

/// Sample a categorical distribution using a pre-generated random value.
///
/// `random_value` is expected in `[0.0, 1.0)`. Values outside this range are
/// safely clamped.
///
/// Returns `None` when `probabilities` is empty.
pub fn sample_categorical(probabilities: &[f32], random_value: f32) -> Option<usize> {
    if probabilities.is_empty() {
        return None;
    }

    let rv = random_value.clamp(0.0, 1.0 - f32::EPSILON);

    let mut cumulative = 0.0_f32;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumulative += prob;
        if rv <= cumulative {
            return Some(i);
        }
    }

    // In case of floating-point accumulation error, pick the last token.
    Some(probabilities.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn renormalize_success() {
        let mut probs = vec![2.0_f32, 3.0, 5.0];
        assert!(renormalize_in_place(&mut probs));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn renormalize_rejects_empty_or_zero_sum() {
        let mut empty: Vec<f32> = vec![];
        assert!(!renormalize_in_place(&mut empty));

        let mut zeros = vec![0.0_f32, 0.0];
        assert!(!renormalize_in_place(&mut zeros));
    }

    #[test]
    fn categorical_returns_last_on_rounding_tail() {
        let probs = vec![0.1_f32, 0.2, 0.3, 0.4];
        assert_eq!(sample_categorical(&probs, 0.999_999_94), Some(3));
    }

    proptest! {
        #[test]
        fn categorical_always_returns_valid_index(
            values in prop::collection::vec(0.0f32..10.0f32, 1..128),
            random in 0.0f32..1.0f32,
        ) {
            let mut probs = values;
            prop_assume!(renormalize_in_place(&mut probs));
            let idx = sample_categorical(&probs, random).unwrap();
            prop_assert!(idx < probs.len());
        }
    }
}
