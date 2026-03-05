//! Decoding strategy utilities.
//!
//! Top-k, top-p, temperature, and repetition penalty application.

/// Apply temperature scaling to logits.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || temperature == 1.0 {
        return;
    }
    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
}

/// Apply top-k filtering: keep only the k highest logits.
/// Returns indices of kept tokens.
pub fn top_k_filter(logits: &mut [f32], k: usize) -> Vec<usize> {
    if k == 0 || k >= logits.len() {
        return (0..logits.len()).collect();
    }

    // Find kth largest value
    let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = sorted[k - 1].1;
    let mut kept = Vec::new();

    for (i, v) in logits.iter_mut().enumerate() {
        if *v >= threshold && kept.len() < k {
            kept.push(i);
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
    kept
}

/// Apply top-p (nucleus) filtering.
/// Keeps tokens until cumulative probability exceeds p.
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    // Softmax to get probabilities
    let probs = softmax(logits);

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut keep_set = std::collections::HashSet::new();

    for (idx, prob) in &indexed {
        cumulative += prob;
        keep_set.insert(*idx);
        if cumulative >= p {
            break;
        }
    }

    for (i, v) in logits.iter_mut().enumerate() {
        if !keep_set.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply repetition penalty to logits for previously generated tokens.
pub fn apply_repetition_penalty(logits: &mut [f32], previous_tokens: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    for &token in previous_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Compute argmax of logits.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Softmax over logits, returning probabilities.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
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

/// Sample from probability distribution (deterministic with seed).
pub fn sample_from_probs(probs: &[f32], random_value: f32) -> usize {
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= random_value {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert!((logits[0] - 2.0).abs() < 0.01);
        assert!((logits[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_temperature_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 1.0);
        assert!((logits[0] - 1.0).abs() < 0.01); // unchanged
    }

    #[test]
    fn test_top_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let kept = top_k_filter(&mut logits, 2);
        assert_eq!(kept.len(), 2);
        assert!(logits[0] == f32::NEG_INFINITY); // filtered out
    }

    #[test]
    fn test_top_k_all() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let kept = top_k_filter(&mut logits, 10);
        assert_eq!(kept.len(), 3); // all kept
    }

    #[test]
    fn test_top_p() {
        let mut logits = vec![10.0, 1.0, 0.1, 0.01];
        top_p_filter(&mut logits, 0.9);
        assert!(logits[0] > f32::NEG_INFINITY); // kept
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, -1.0, 3.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 1.5);
        assert!(logits[0] < 2.0); // positive reduced
        assert!(logits[1] < -1.0); // negative made more negative
        assert!((logits[2] - 3.0).abs() < 0.01); // untouched
    }

    #[test]
    fn test_repetition_no_penalty() {
        let mut logits = vec![2.0, 3.0];
        apply_repetition_penalty(&mut logits, &[0], 1.0);
        assert!((logits[0] - 2.0).abs() < 0.01); // unchanged
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 3.0]), 0);
    }

    #[test]
    fn test_softmax() {
        let probs = softmax(&[0.0, 0.0, 0.0]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 0.01);
        // Uniform
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_sample() {
        let probs = vec![0.1, 0.7, 0.2];
        assert_eq!(sample_from_probs(&probs, 0.05), 0);
        assert_eq!(sample_from_probs(&probs, 0.5), 1);
        assert_eq!(sample_from_probs(&probs, 0.9), 2);
    }

    #[test]
    fn test_softmax_sum() {
        let probs = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_argmax_single() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn test_sample_edge() {
        let probs = vec![1.0];
        assert_eq!(sample_from_probs(&probs, 0.5), 0);
    }
}
