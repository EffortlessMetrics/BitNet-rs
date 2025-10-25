//! Parity metrics for cross-validation
//!
//! This module provides statistical metrics for comparing logits and probability
//! distributions between Rust and C++ implementations.
//!
//! # Metrics
//!
//! - **MSE**: Mean Squared Error for raw logit comparison
//! - **Max Absolute Difference**: Maximum element-wise difference
//! - **KL Divergence**: Kullback-Leibler divergence for probability distributions
//! - **Top-K Agreement**: Fraction of top-K tokens that match between two distributions
//!
//! # Example
//!
//! ```rust
//! use bitnet_crossval::metrics::{mse_row, kl_divergence, topk_agree};
//!
//! let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
//! let cpp_logits = vec![1.1, 2.0, 2.9, 4.0];
//!
//! let error = mse_row(&rust_logits, &cpp_logits);
//! let kl_div = kl_divergence(&rust_logits, &cpp_logits);
//! let agreement = topk_agree(&rust_logits, &cpp_logits, 2);
//!
//! println!("MSE: {:.6}", error);
//! println!("KL divergence: {:.6}", kl_div);
//! println!("Top-2 agreement: {:.2}%", agreement * 100.0);
//! ```

/// Mean Squared Error between two float arrays
///
/// Computes: MSE = (1/n) * Σ(aᵢ - bᵢ)²
///
/// # Arguments
///
/// * `a` - First array of floats
/// * `b` - Second array of floats
///
/// # Returns
///
/// The mean squared error between the two arrays
///
/// # Panics
///
/// Panics if the arrays have different lengths
///
/// # Example
///
/// ```rust
/// use bitnet_crossval::metrics::mse_row;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.1, 2.0, 2.9];
///
/// let mse = mse_row(&a, &b);
/// assert!(mse < 0.01); // Very close
/// ```
pub fn mse_row(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have equal length for MSE: a.len()={}, b.len()={}",
        a.len(),
        b.len()
    );

    if a.is_empty() {
        return 0.0;
    }

    let sum_squared_diff: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    sum_squared_diff / a.len() as f32
}

/// Maximum absolute difference between two float arrays
///
/// Computes: max|aᵢ - bᵢ| for all i
///
/// # Arguments
///
/// * `a` - First array of floats
/// * `b` - Second array of floats
///
/// # Returns
///
/// The maximum absolute difference between corresponding elements
///
/// # Panics
///
/// Panics if the arrays have different lengths
///
/// # Example
///
/// ```rust
/// use bitnet_crossval::metrics::max_abs;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.1, 2.5, 2.9];
///
/// let max_diff = max_abs(&a, &b);
/// assert!((max_diff - 0.5).abs() < 1e-6); // max diff at index 1
/// ```
pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have equal length for max_abs: a.len()={}, b.len()={}",
        a.len(),
        b.len()
    );

    if a.is_empty() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// KL divergence between two probability distributions
///
/// Computes: KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
///
/// The input arrays are treated as unnormalized logits and are converted to
/// probabilities using softmax before computing KL divergence.
///
/// # Arguments
///
/// * `p` - First distribution (unnormalized logits)
/// * `q` - Second distribution (unnormalized logits)
///
/// # Returns
///
/// The KL divergence from Q to P (in nats, using natural logarithm)
///
/// # Panics
///
/// Panics if the arrays have different lengths
///
/// # Numerical Stability
///
/// - Uses log-sum-exp trick for softmax to avoid overflow
/// - Clamps small probabilities to avoid log(0)
/// - Returns 0.0 for empty arrays
///
/// # Example
///
/// ```rust
/// use bitnet_crossval::metrics::kl_divergence;
///
/// let p = vec![1.0, 2.0, 3.0];
/// let q = vec![1.0, 2.0, 3.0];
///
/// let kl_div = kl_divergence(&p, &q);
/// assert!(kl_div < 1e-6); // Identical distributions have KL ≈ 0
/// ```
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(
        p.len(),
        q.len(),
        "Arrays must have equal length for KL divergence: p.len()={}, q.len()={}",
        p.len(),
        q.len()
    );

    if p.is_empty() {
        return 0.0;
    }

    // Convert logits to probabilities using stable softmax
    let p_probs = softmax(p);
    let q_probs = softmax(q);

    // Compute KL divergence with numerical stability
    // KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
    const EPSILON: f32 = 1e-10; // Prevent log(0)

    p_probs
        .iter()
        .zip(q_probs.iter())
        .map(|(&p_i, &q_i)| {
            // Clamp to avoid log(0)
            let p_i = p_i.max(EPSILON);
            let q_i = q_i.max(EPSILON);

            // KL contribution for this element
            p_i * (p_i / q_i).ln()
        })
        .sum()
}

/// Top-K agreement: fraction of top-K indices that match
///
/// Computes the fraction of top-K tokens (by logit value) that appear in both arrays.
///
/// # Arguments
///
/// * `a` - First array of logits
/// * `b` - Second array of logits
/// * `k` - Number of top elements to compare
///
/// # Returns
///
/// A value in [0.0, 1.0] representing the fraction of top-K indices that match.
/// Returns 1.0 if all top-K indices match, 0.0 if none match.
///
/// # Panics
///
/// Panics if:
/// - Arrays have different lengths
/// - `k` is 0
/// - `k` is greater than array length
///
/// # Example
///
/// ```rust
/// use bitnet_crossval::metrics::topk_agree;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![1.5, 2.0, 2.5, 4.0];
///
/// // Top-2 in a: [3, 2] (indices by value: 4.0, 3.0)
/// // Top-2 in b: [3, 0] (indices by value: 4.0, 2.5)
/// let agreement = topk_agree(&a, &b, 2);
/// // Should have 50% agreement (index 3 matches)
/// assert!((agreement - 0.5).abs() < 0.01);
/// ```
pub fn topk_agree(a: &[f32], b: &[f32], k: usize) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have equal length for topk_agree: a.len()={}, b.len()={}",
        a.len(),
        b.len()
    );
    assert!(k > 0, "k must be greater than 0");
    assert!(k <= a.len(), "k must be <= array length: k={}, len={}", k, a.len());

    let topk_a = topk_indices(a, k);
    let topk_b = topk_indices(b, k);

    // Count how many indices appear in both top-K sets
    let matches = topk_a.iter().filter(|&&idx| topk_b.contains(&idx)).count();

    matches as f32 / k as f32
}

/// Get top-K indices from logits (sorted by value, descending)
///
/// Returns the indices of the K largest values in the array.
///
/// # Arguments
///
/// * `row` - Array of logits
/// * `k` - Number of top elements to return
///
/// # Returns
///
/// A vector of indices corresponding to the K largest values, sorted by value (descending).
/// In case of ties, indices are sorted by their position (ascending).
///
/// # Panics
///
/// Panics if `k` is greater than the array length
///
/// # Example
///
/// ```rust
/// use bitnet_crossval::metrics::topk_indices;
///
/// let logits = vec![1.0, 4.0, 2.0, 3.0];
/// let top2 = topk_indices(&logits, 2);
///
/// assert_eq!(top2, vec![1, 3]); // indices of values 4.0 and 3.0
/// ```
pub fn topk_indices(row: &[f32], k: usize) -> Vec<usize> {
    assert!(k <= row.len(), "k must be <= array length: k={}, len={}", k, row.len());

    let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();

    // Sort by value (descending), with stable ordering for ties
    indexed.sort_by(|(idx_a, val_a), (idx_b, val_b)| {
        // First compare by value (descending)
        match val_b.partial_cmp(val_a) {
            Some(std::cmp::Ordering::Equal) => {
                // For ties, use index (ascending) for stable ordering
                idx_a.cmp(idx_b)
            }
            Some(ordering) => ordering,
            // Handle NaN by treating as less than any other value
            None => {
                if val_a.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    });

    // Take top K indices
    indexed.iter().take(k).map(|(idx, _)| *idx).collect()
}

/// Numerically stable softmax
///
/// Converts logits to probabilities using the softmax function:
/// softmax(x)ᵢ = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))
///
/// Uses the log-sum-exp trick for numerical stability.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) for all elements
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

    // Compute sum of exponentials
    let sum_exp: f32 = exp_values.iter().sum();

    // Normalize to get probabilities
    exp_values.iter().map(|&e| e / sum_exp).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_mse_row_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(mse_row(&a, &b) < EPSILON);
    }

    #[test]
    fn test_mse_row_simple() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let mse = mse_row(&a, &b);
        // MSE = ((1-0)^2 + (1-0)^2) / 2 = 2/2 = 1.0
        assert!((mse - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_mse_row_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(mse_row(&a, &b), 0.0);
    }

    #[test]
    #[should_panic(expected = "Arrays must have equal length")]
    fn test_mse_row_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        mse_row(&a, &b);
    }

    #[test]
    fn test_max_abs_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(max_abs(&a, &b) < EPSILON);
    }

    #[test]
    fn test_max_abs_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.5, 2.9];
        let max_diff = max_abs(&a, &b);
        // Max diff is at index 1: |2.0 - 2.5| = 0.5
        assert!((max_diff - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_max_abs_negative() {
        let a = vec![-5.0, 0.0, 5.0];
        let b = vec![-3.0, 0.0, 3.0];
        let max_diff = max_abs(&a, &b);
        // Max diff is at index 0: |-5.0 - (-3.0)| = 2.0
        assert!((max_diff - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_max_abs_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(max_abs(&a, &b), 0.0);
    }

    #[test]
    #[should_panic(expected = "Arrays must have equal length")]
    fn test_max_abs_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        max_abs(&a, &b);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![1.0, 2.0, 3.0];
        let q = vec![1.0, 2.0, 3.0];
        let kl = kl_divergence(&p, &q);
        // KL divergence should be ~0 for identical distributions
        assert!(kl < 1e-5, "KL divergence was {}", kl);
    }

    #[test]
    fn test_kl_divergence_uniform() {
        // Uniform distributions should have KL = 0
        let p = vec![1.0, 1.0, 1.0, 1.0];
        let q = vec![1.0, 1.0, 1.0, 1.0];
        let kl = kl_divergence(&p, &q);
        assert!(kl < 1e-5);
    }

    #[test]
    fn test_kl_divergence_non_zero() {
        // Different distributions should have KL > 0
        let p = vec![10.0, 0.0]; // Strongly prefers first element
        let q = vec![0.0, 10.0]; // Strongly prefers second element
        let kl = kl_divergence(&p, &q);
        assert!(kl > 1.0, "KL divergence should be significant: {}", kl);
    }

    #[test]
    fn test_kl_divergence_empty() {
        let p: Vec<f32> = vec![];
        let q: Vec<f32> = vec![];
        assert_eq!(kl_divergence(&p, &q), 0.0);
    }

    #[test]
    #[should_panic(expected = "Arrays must have equal length")]
    fn test_kl_divergence_length_mismatch() {
        let p = vec![1.0, 2.0];
        let q = vec![1.0];
        kl_divergence(&p, &q);
    }

    #[test]
    fn test_topk_indices_simple() {
        let logits = vec![1.0, 4.0, 2.0, 3.0];
        let top2 = topk_indices(&logits, 2);
        assert_eq!(top2, vec![1, 3]); // indices of 4.0 and 3.0
    }

    #[test]
    fn test_topk_indices_all() {
        let logits = vec![3.0, 1.0, 2.0];
        let top3 = topk_indices(&logits, 3);
        assert_eq!(top3, vec![0, 2, 1]); // all indices, sorted by value
    }

    #[test]
    fn test_topk_indices_single() {
        let logits = vec![1.0, 4.0, 2.0, 3.0];
        let top1 = topk_indices(&logits, 1);
        assert_eq!(top1, vec![1]); // index of maximum value
    }

    #[test]
    fn test_topk_indices_with_ties() {
        let logits = vec![1.0, 3.0, 3.0, 2.0];
        let top2 = topk_indices(&logits, 2);
        // When tied at 3.0, should prefer lower index (stable sort)
        assert_eq!(top2, vec![1, 2]);
    }

    #[test]
    #[should_panic(expected = "k must be <= array length")]
    fn test_topk_indices_k_too_large() {
        let logits = vec![1.0, 2.0];
        topk_indices(&logits, 3);
    }

    #[test]
    fn test_topk_agree_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let agreement = topk_agree(&a, &b, 2);
        assert!((agreement - 1.0).abs() < EPSILON); // 100% agreement
    }

    #[test]
    fn test_topk_agree_partial() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 0.5, 3.5, 0.0];

        // Top-2 in a: indices [3, 2] (values 4.0, 3.0)
        // Top-2 in b: indices [0, 2] (values 4.0, 3.5)
        // Match: index 2 only -> 1/2 = 0.5
        let agreement = topk_agree(&a, &b, 2);
        assert!((agreement - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_topk_agree_no_match() {
        let a = vec![10.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 10.0, 0.0, 0.0];

        // Top-1 in a: index 0
        // Top-1 in b: index 1
        // No match -> 0/1 = 0.0
        let agreement = topk_agree(&a, &b, 1);
        assert!(agreement.abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "k must be greater than 0")]
    fn test_topk_agree_k_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        topk_agree(&a, &b, 0);
    }

    #[test]
    #[should_panic(expected = "Arrays must have equal length")]
    fn test_topk_agree_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        topk_agree(&a, &b, 1);
    }

    #[test]
    fn test_softmax_simple() {
        let logits = vec![0.0, 0.0, 0.0];
        let probs = softmax(&logits);

        // Uniform logits -> uniform probabilities
        for p in &probs {
            assert!((p - 1.0 / 3.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with very large values that would overflow without stability
        let logits = vec![1000.0, 1001.0, 999.0];
        let probs = softmax(&logits);

        // Should still sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);

        // Highest logit should have highest probability
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 1); // index of 1001.0
    }

    #[test]
    fn test_softmax_empty() {
        let logits: Vec<f32> = vec![];
        let probs = softmax(&logits);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_comprehensive_parity_example() {
        // Real-world example: comparing two nearly-identical logit arrays
        let rust_logits = vec![2.1, 3.5, 1.8, 4.2, 2.9];
        let cpp_logits = vec![2.0, 3.6, 1.9, 4.1, 3.0];

        let mse = mse_row(&rust_logits, &cpp_logits);
        let max_diff = max_abs(&rust_logits, &cpp_logits);
        let kl_div = kl_divergence(&rust_logits, &cpp_logits);
        let top2_agreement = topk_agree(&rust_logits, &cpp_logits, 2);

        // These should all indicate high similarity
        assert!(mse < 0.02, "MSE too high: {}", mse);
        assert!(max_diff < 0.2, "Max diff too high: {}", max_diff);
        assert!(kl_div < 0.01, "KL divergence too high: {}", kl_div);
        assert!(top2_agreement >= 0.5, "Top-2 agreement too low: {}", top2_agreement);
    }
}
