//! Attention pattern analysis.
//!
//! Analyze attention weight distributions for debugging,
//! sparsity detection, and pattern classification.

/// Attention pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternKind {
    /// Focuses on recent tokens.
    Local,
    /// Broadly distributed attention.
    Global,
    /// Mostly zeros with few high values.
    Sparse,
    /// Nearly uniform distribution.
    Uniform,
    /// Strongly focused on one position.
    Peaked,
}

/// Statistics for a single attention head.
#[derive(Debug, Clone)]
pub struct HeadStats {
    pub head_idx: usize,
    pub entropy: f32,
    pub sparsity: f32,
    pub max_weight: f32,
    pub max_position: usize,
    pub pattern: PatternKind,
}

/// Compute entropy of a probability distribution.
pub fn entropy(weights: &[f32]) -> f32 {
    let mut h = 0.0f64;
    for &w in weights {
        if w > 1e-10 {
            h -= (w as f64) * (w as f64).ln();
        }
    }
    h as f32
}

/// Compute sparsity (fraction of near-zero values).
pub fn sparsity(weights: &[f32], threshold: f32) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let near_zero = weights.iter().filter(|&&w| w.abs() < threshold).count();
    near_zero as f32 / weights.len() as f32
}

/// Find the max weight and its position.
pub fn find_peak(weights: &[f32]) -> (f32, usize) {
    if weights.is_empty() {
        return (0.0, 0);
    }
    let mut max_val = weights[0];
    let mut max_idx = 0;
    for (i, &w) in weights.iter().enumerate().skip(1) {
        if w > max_val {
            max_val = w;
            max_idx = i;
        }
    }
    (max_val, max_idx)
}

/// Classify the attention pattern.
pub fn classify_pattern(weights: &[f32]) -> PatternKind {
    if weights.is_empty() {
        return PatternKind::Uniform;
    }

    let n = weights.len();
    let (max_w, max_pos) = find_peak(weights);
    let sp = sparsity(weights, 0.01);
    let ent = entropy(weights);
    let max_entropy = (n as f32).ln();

    // Very peaked: one value dominates
    if max_w > 0.5 {
        return PatternKind::Peaked;
    }

    // Very sparse
    if sp > 0.8 {
        return PatternKind::Sparse;
    }

    // Nearly uniform
    if max_entropy > 0.0 && ent / max_entropy > 0.9 {
        return PatternKind::Uniform;
    }

    // Local: peak is near the end (recent tokens)
    if n > 4 && max_pos > n * 3 / 4 {
        return PatternKind::Local;
    }

    PatternKind::Global
}

/// Analyze a single attention head.
pub fn analyze_head(head_idx: usize, weights: &[f32]) -> HeadStats {
    let (max_weight, max_position) = find_peak(weights);
    HeadStats {
        head_idx,
        entropy: entropy(weights),
        sparsity: sparsity(weights, 0.01),
        max_weight,
        max_position,
        pattern: classify_pattern(weights),
    }
}

/// Analyze all heads in a layer.
pub fn analyze_layer(heads: &[&[f32]]) -> Vec<HeadStats> {
    heads.iter().enumerate().map(|(i, weights)| analyze_head(i, weights)).collect()
}

/// Summary statistics across all heads.
#[derive(Debug, Clone)]
pub struct LayerSummary {
    pub num_heads: usize,
    pub avg_entropy: f32,
    pub avg_sparsity: f32,
    pub pattern_counts: Vec<(PatternKind, usize)>,
}

pub fn summarize_layer(stats: &[HeadStats]) -> LayerSummary {
    let n = stats.len();
    let avg_entropy =
        if n > 0 { stats.iter().map(|s| s.entropy).sum::<f32>() / n as f32 } else { 0.0 };
    let avg_sparsity =
        if n > 0 { stats.iter().map(|s| s.sparsity).sum::<f32>() / n as f32 } else { 0.0 };

    let mut counts = std::collections::HashMap::new();
    for s in stats {
        *counts.entry(s.pattern).or_insert(0) += 1;
    }
    let pattern_counts: Vec<_> = counts.into_iter().collect();

    LayerSummary { num_heads: n, avg_entropy, avg_sparsity, pattern_counts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let w = vec![0.25, 0.25, 0.25, 0.25];
        let e = entropy(&w);
        let expected = (4.0f32).ln();
        assert!((e - expected).abs() < 0.01);
    }

    #[test]
    fn test_entropy_peaked() {
        let w = vec![1.0, 0.0, 0.0, 0.0];
        let e = entropy(&w);
        assert!(e.abs() < 0.01);
    }

    #[test]
    fn test_sparsity_dense() {
        let w = vec![0.5, 0.3, 0.1, 0.1];
        assert!(sparsity(&w, 0.01) < 0.1);
    }

    #[test]
    fn test_sparsity_sparse() {
        let w = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
        assert!(sparsity(&w, 0.01) > 0.7);
    }

    #[test]
    fn test_find_peak() {
        let w = vec![0.1, 0.5, 0.3, 0.1];
        let (max, pos) = find_peak(&w);
        assert_eq!(pos, 1);
        assert!((max - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_classify_peaked() {
        let w = vec![0.0, 0.0, 0.9, 0.1];
        assert_eq!(classify_pattern(&w), PatternKind::Peaked);
    }

    #[test]
    fn test_classify_uniform() {
        let n = 100;
        let w = vec![1.0 / n as f32; n];
        assert_eq!(classify_pattern(&w), PatternKind::Uniform);
    }

    #[test]
    fn test_classify_sparse() {
        let mut w = vec![0.0; 100];
        w[50] = 0.3;
        w[51] = 0.3;
        w[52] = 0.4;
        assert_eq!(classify_pattern(&w), PatternKind::Sparse);
    }

    #[test]
    fn test_analyze_head() {
        let w = vec![0.25, 0.25, 0.25, 0.25];
        let stats = analyze_head(0, &w);
        assert_eq!(stats.head_idx, 0);
        assert!(stats.entropy > 0.0);
    }

    #[test]
    fn test_analyze_layer() {
        let h0 = vec![0.5, 0.5];
        let h1 = vec![0.9, 0.1];
        let stats = analyze_layer(&[&h0, &h1]);
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn test_summarize() {
        let stats = vec![analyze_head(0, &vec![0.25; 4]), analyze_head(1, &vec![0.25; 4])];
        let summary = summarize_layer(&stats);
        assert_eq!(summary.num_heads, 2);
        assert!(summary.avg_entropy > 0.0);
    }

    #[test]
    fn test_empty_weights() {
        assert_eq!(classify_pattern(&[]), PatternKind::Uniform);
        assert_eq!(sparsity(&[], 0.01), 0.0);
    }
}
