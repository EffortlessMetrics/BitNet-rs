//! Token sampling utilities.
//!
//! Top-k, top-p, temperature, repetition penalty helpers.

/// Apply temperature scaling to logits.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || temperature == 1.0 {
        return;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }
}

/// Apply top-k filtering: keep only top k logits, set rest to -inf.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].1;
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-p (nucleus) filtering.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let keep: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|&(i, _)| i).collect();
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Apply repetition penalty to already-generated tokens.
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &tok in generated {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Simple softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 {
        probs.fill(0.0);
        return probs;
    }
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

/// Argmax: index of largest value.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Weighted random sample from logits (returns index).
pub fn sample_from_logits(logits: &[f32], random: f32) -> usize {
    let probs = softmax(logits);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if random <= cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 0.5).abs() < 0.01);
        assert!((logits[2] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_temperature_one_noop() {
        let mut logits = vec![1.0, 2.0];
        let orig = logits.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, orig);
    }

    #[test]
    fn test_top_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[4], 4.0);
    }

    #[test]
    fn test_top_k_all() {
        let mut logits = vec![1.0, 2.0];
        apply_top_k(&mut logits, 10); // k > len, no-op
        assert_eq!(logits, vec![1.0, 2.0]);
    }

    #[test]
    fn test_top_p() {
        let mut logits = vec![1.0, 10.0, 0.1, 0.1];
        apply_top_p(&mut logits, 0.9);
        // Token 1 (10.0) should survive
        assert!(logits[1].is_finite());
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, -1.0, 3.0];
        apply_repetition_penalty(&mut logits, &[1, 2], 2.0);
        assert!((logits[1] - 1.0).abs() < 0.01); // positive / 2
        assert!((logits[2] - (-2.0)).abs() < 0.01); // negative * 2
    }

    #[test]
    fn test_softmax() {
        let p = softmax(&[0.0, 0.0, 0.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[10.0, 5.0, 3.0]), 0);
    }

    #[test]
    fn test_sample_deterministic() {
        let logits = vec![f32::NEG_INFINITY, 100.0, f32::NEG_INFINITY];
        let idx = sample_from_logits(&logits, 0.5);
        assert_eq!(idx, 1); // all mass on index 1
    }

    #[test]
    fn test_sample_boundary() {
        let logits = vec![0.0, 0.0]; // equal probs
        let idx = sample_from_logits(&logits, 0.0);
        assert!(idx <= 1);
    }

    #[test]
    fn test_penalty_noop() {
        let mut logits = vec![1.0, 2.0];
        let orig = logits.clone();
        apply_repetition_penalty(&mut logits, &[0], 1.0);
        assert_eq!(logits, orig);
    }
}
