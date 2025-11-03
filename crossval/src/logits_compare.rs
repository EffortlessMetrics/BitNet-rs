//! Per-position logits comparison for cross-validation
//!
//! This module provides utilities to compare logits between Rust and C++
//! implementations at each token position, identifying divergence points.

use serde::{Deserialize, Serialize};

/// Result of per-position logits comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitsDivergence {
    /// First token position where logits diverged (None if all match)
    pub first_divergence_token: Option<usize>,

    /// Cosine similarity for each token position
    pub per_token_cosine_sim: Vec<f32>,

    /// L2 distance for each token position
    pub per_token_l2_dist: Vec<f32>,

    /// Maximum absolute difference across all positions and logits
    pub max_absolute_diff: f32,
}

/// Default tolerance for cosine similarity (matches existing parity threshold)
pub const COSINE_SIMILARITY_THRESHOLD: f32 = 1e-4;

/// Compare logits at each token position between Rust and C++ implementations
///
/// # Arguments
///
/// * `rs_logits` - Logits from Rust implementation (outer vec = positions, inner vec = vocab)
/// * `cpp_logits` - Logits from C++ implementation (outer vec = positions, inner vec = vocab)
///
/// # Returns
///
/// A `LogitsDivergence` struct with per-position metrics and divergence point
///
/// # Example
///
/// ```no_run
/// use bitnet_crossval::logits_compare::compare_per_position_logits;
///
/// let rs_logits = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
/// let cpp_logits = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
///
/// let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);
/// assert!(divergence.first_divergence_token.is_none()); // No divergence
/// ```
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence {
    let n_positions = rs_logits.len().min(cpp_logits.len());

    let mut per_token_cosine_sim = Vec::with_capacity(n_positions);
    let mut per_token_l2_dist = Vec::with_capacity(n_positions);
    let mut max_absolute_diff = 0.0f32;
    let mut first_divergence_token = None;

    for pos in 0..n_positions {
        let rs_vec = &rs_logits[pos];
        let cpp_vec = &cpp_logits[pos];

        // Skip if vector sizes don't match
        if rs_vec.len() != cpp_vec.len() {
            per_token_cosine_sim.push(0.0);
            per_token_l2_dist.push(f32::INFINITY);
            if first_divergence_token.is_none() {
                first_divergence_token = Some(pos);
            }
            continue;
        }

        // Calculate cosine similarity
        let cosine_sim = cosine_similarity(rs_vec, cpp_vec);
        per_token_cosine_sim.push(cosine_sim);

        // Calculate L2 distance
        let l2_dist = l2_distance(rs_vec, cpp_vec);
        per_token_l2_dist.push(l2_dist);

        // Track max absolute difference
        let max_diff_at_pos =
            rs_vec.iter().zip(cpp_vec.iter()).map(|(r, c)| (r - c).abs()).fold(0.0f32, f32::max);

        if max_diff_at_pos > max_absolute_diff {
            max_absolute_diff = max_diff_at_pos;
        }

        // Check if this is the first divergence (cosine similarity too low)
        if first_divergence_token.is_none() && (1.0 - cosine_sim) > COSINE_SIMILARITY_THRESHOLD {
            first_divergence_token = Some(pos);
        }
    }

    LogitsDivergence {
        first_divergence_token,
        per_token_cosine_sim,
        per_token_l2_dist,
        max_absolute_diff,
    }
}

/// Calculate cosine similarity between two vectors
///
/// Returns 1.0 for identical vectors, 0.0 for orthogonal vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Avoid division by zero
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Calculate L2 (Euclidean) distance between two vectors
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have cosine similarity of 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Orthogonal vectors should have cosine similarity of 0.0");
    }

    #[test]
    fn test_l2_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = l2_distance(&a, &b);
        assert!(dist.abs() < 1e-6, "Identical vectors should have L2 distance of 0.0");
    }

    #[test]
    fn test_l2_distance_simple() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6, "Expected L2 distance of 5.0 (3-4-5 triangle)");
    }

    #[test]
    fn test_compare_per_position_logits_no_divergence() {
        let rs_logits = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6], vec![0.7, 0.8, 0.9]];
        let cpp_logits = rs_logits.clone();

        let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);

        assert!(divergence.first_divergence_token.is_none());
        assert_eq!(divergence.per_token_cosine_sim.len(), 3);
        assert_eq!(divergence.per_token_l2_dist.len(), 3);
        assert!(divergence.max_absolute_diff < 1e-6);

        // All cosine similarities should be 1.0
        for sim in &divergence.per_token_cosine_sim {
            assert!((sim - 1.0).abs() < 1e-6);
        }

        // All L2 distances should be 0.0
        for dist in &divergence.per_token_l2_dist {
            assert!(dist.abs() < 1e-6);
        }
    }

    #[test]
    fn test_compare_per_position_logits_with_divergence() {
        let rs_logits = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6], // This will match
            vec![0.7, 0.8, 0.9], // This will diverge significantly
        ];
        let mut cpp_logits = rs_logits.clone();

        // Make position 2 diverge significantly
        cpp_logits[2] = vec![1.0, 1.0, 1.0];

        let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);

        assert_eq!(divergence.first_divergence_token, Some(2));
        assert_eq!(divergence.per_token_cosine_sim.len(), 3);
        assert_eq!(divergence.per_token_l2_dist.len(), 3);

        // First two positions should match closely
        assert!((divergence.per_token_cosine_sim[0] - 1.0).abs() < 1e-6);
        assert!((divergence.per_token_cosine_sim[1] - 1.0).abs() < 1e-6);

        // Third position should have lower similarity
        assert!(divergence.per_token_cosine_sim[2] < 1.0);
    }

    #[test]
    fn test_compare_per_position_logits_size_mismatch() {
        let rs_logits = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let cpp_logits = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5], // Different size
        ];

        let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);

        assert_eq!(divergence.first_divergence_token, Some(1));
        assert_eq!(divergence.per_token_l2_dist[1], f32::INFINITY);
    }

    #[test]
    fn test_compare_per_position_logits_empty() {
        let rs_logits: Vec<Vec<f32>> = vec![];
        let cpp_logits: Vec<Vec<f32>> = vec![];

        let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);

        assert!(divergence.first_divergence_token.is_none());
        assert!(divergence.per_token_cosine_sim.is_empty());
        assert!(divergence.per_token_l2_dist.is_empty());
        assert_eq!(divergence.max_absolute_diff, 0.0);
    }
}
