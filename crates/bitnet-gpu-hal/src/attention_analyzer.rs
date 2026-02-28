//! Attention pattern analyzer for debugging and optimization.
//!
//! Provides tools to capture, analyze, classify, and visualize attention
//! patterns from transformer attention heads. Useful for identifying
//! prunable heads, diagnosing attention collapse, and guiding kernel
//! optimization decisions.

use std::fmt;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Captured attention weights for a single head in a single layer.
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    /// Layer index in the transformer stack.
    pub layer_idx: usize,
    /// Head index within the layer.
    pub head_idx: usize,
    /// Attention weight matrix – `weights[i][j]` is the attention query
    /// position `i` pays to key position `j`.  Each row should sum to ~1.0
    /// (softmax output).
    pub weights: Vec<Vec<f32>>,
    /// Sequence length (rows == cols == seq_len for self-attention).
    pub seq_len: usize,
}

impl AttentionPattern {
    /// Create a new pattern, validating dimensions.
    pub fn new(
        layer_idx: usize,
        head_idx: usize,
        weights: Vec<Vec<f32>>,
    ) -> Result<Self, AnalyzerError> {
        if weights.is_empty() {
            return Err(AnalyzerError::EmptyWeights);
        }
        let seq_len = weights.len();
        for (i, row) in weights.iter().enumerate() {
            if row.len() != seq_len {
                return Err(AnalyzerError::DimensionMismatch {
                    row: i,
                    expected: seq_len,
                    got: row.len(),
                });
            }
        }
        Ok(Self { layer_idx, head_idx, weights, seq_len })
    }
}

/// Classification of the dominant attention pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// Attention concentrated along the main diagonal (local attention).
    Diagonal,
    /// Attention concentrated in one or more vertical stripes (global tokens).
    Vertical,
    /// Attention in block-diagonal form (chunked local attention).
    BlockDiagonal,
    /// Majority of weights below threshold (sparse attention).
    Sparse,
    /// Weights spread fairly uniformly (dense / uniform attention).
    Dense,
    /// Repeating pattern with a detectable period.
    Periodic,
    /// No single dominant pattern.
    Mixed,
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Diagonal => write!(f, "diagonal"),
            Self::Vertical => write!(f, "vertical"),
            Self::BlockDiagonal => write!(f, "block-diagonal"),
            Self::Sparse => write!(f, "sparse"),
            Self::Dense => write!(f, "dense"),
            Self::Periodic => write!(f, "periodic"),
            Self::Mixed => write!(f, "mixed"),
        }
    }
}

/// Summary statistics for one attention pattern.
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Shannon entropy averaged over all query positions.
    pub entropy: f32,
    /// Fraction of weights below the sparsity threshold.
    pub sparsity: f32,
    /// Maximum single weight in the entire matrix.
    pub max_weight: f32,
    /// Mean weight across the matrix.
    pub mean_weight: f32,
    /// Effective context length: average number of positions receiving
    /// meaningful attention (weight > threshold).
    pub effective_context_length: f32,
}

/// Importance metadata for a single attention head.
#[derive(Debug, Clone)]
pub struct HeadImportance {
    pub head_idx: usize,
    /// Score in `[0, 1]` – higher means more important.
    pub importance_score: f32,
    /// `true` when the head can likely be pruned without quality loss.
    pub prunable: bool,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the analyzer pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalyzerError {
    EmptyWeights,
    DimensionMismatch { row: usize, expected: usize, got: usize },
    NoPatterns,
}

impl fmt::Display for AnalyzerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyWeights => write!(f, "attention weight matrix is empty"),
            Self::DimensionMismatch { row, expected, got } => {
                write!(f, "row {row}: expected {expected} cols, got {got}")
            }
            Self::NoPatterns => write!(f, "no patterns captured for analysis"),
        }
    }
}

impl std::error::Error for AnalyzerError {}

// ---------------------------------------------------------------------------
// SparsityAnalyzer
// ---------------------------------------------------------------------------

/// Threshold-based and top-k sparsity analysis.
pub struct SparsityAnalyzer {
    /// Weights below this value are considered "zero" for sparsity counting.
    pub threshold: f32,
}

impl SparsityAnalyzer {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Fraction of weights below `self.threshold`.
    pub fn compute_sparsity(&self, pattern: &AttentionPattern) -> f32 {
        let total = (pattern.seq_len * pattern.seq_len) as f32;
        if total == 0.0 {
            return 0.0;
        }
        let sparse_count = pattern
            .weights
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&w| w < self.threshold)
            .count() as f32;
        sparse_count / total
    }

    /// Number of positions per query that exceed the threshold (top-k proxy).
    pub fn active_positions_per_query(&self, pattern: &AttentionPattern) -> Vec<usize> {
        pattern
            .weights
            .iter()
            .map(|row| row.iter().filter(|&&w| w >= self.threshold).count())
            .collect()
    }

    /// Average number of active positions across all queries.
    pub fn mean_active_positions(&self, pattern: &AttentionPattern) -> f32 {
        let counts = self.active_positions_per_query(pattern);
        if counts.is_empty() {
            return 0.0;
        }
        counts.iter().sum::<usize>() as f32 / counts.len() as f32
    }
}

impl Default for SparsityAnalyzer {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ---------------------------------------------------------------------------
// EntropyCalculator
// ---------------------------------------------------------------------------

/// Computes Shannon entropy of attention distributions.
pub struct EntropyCalculator;

impl EntropyCalculator {
    /// Entropy for a single probability distribution (one query row).
    /// Returns value in nats (natural log).  Clamps to avoid log(0).
    pub fn row_entropy(row: &[f32]) -> f32 {
        let eps = 1e-10_f32;
        -row.iter()
            .map(|&p| {
                let p = p.max(eps);
                p * p.ln()
            })
            .sum::<f32>()
    }

    /// Mean entropy across all query positions in a pattern.
    pub fn mean_entropy(pattern: &AttentionPattern) -> f32 {
        if pattern.weights.is_empty() {
            return 0.0;
        }
        let sum: f32 = pattern.weights.iter().map(|row| Self::row_entropy(row)).sum();
        sum / pattern.weights.len() as f32
    }

    /// Maximum possible entropy for a uniform distribution of length `n`.
    pub fn max_entropy(n: usize) -> f32 {
        if n <= 1 {
            return 0.0;
        }
        (n as f32).ln()
    }

    /// Normalized entropy in `[0, 1]` (actual / max).
    pub fn normalized_entropy(pattern: &AttentionPattern) -> f32 {
        let max_e = Self::max_entropy(pattern.seq_len);
        if max_e == 0.0 {
            return 0.0;
        }
        (Self::mean_entropy(pattern) / max_e).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// HeadPruner
// ---------------------------------------------------------------------------

/// Identifies pruneable attention heads based on importance scores.
pub struct HeadPruner {
    /// Heads with importance below this threshold are considered prunable.
    pub prune_threshold: f32,
}

impl HeadPruner {
    pub fn new(prune_threshold: f32) -> Self {
        Self { prune_threshold }
    }

    /// Score a single head.  Importance is a composite of entropy and
    /// non-sparsity (higher entropy + lower sparsity → more important).
    pub fn score_head(&self, pattern: &AttentionPattern) -> HeadImportance {
        let norm_entropy = EntropyCalculator::normalized_entropy(pattern);
        let sparsity = SparsityAnalyzer::default().compute_sparsity(pattern);
        // Heads that are very sparse AND low-entropy are likely redundant.
        let importance = 0.6 * norm_entropy + 0.4 * (1.0 - sparsity);
        HeadImportance {
            head_idx: pattern.head_idx,
            importance_score: importance.clamp(0.0, 1.0),
            prunable: importance < self.prune_threshold,
        }
    }

    /// Score and rank all heads, sorted descending by importance.
    pub fn rank_heads(&self, patterns: &[AttentionPattern]) -> Vec<HeadImportance> {
        let mut scores: Vec<HeadImportance> = patterns.iter().map(|p| self.score_head(p)).collect();
        scores.sort_by(|a, b| {
            b.importance_score.partial_cmp(&a.importance_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores
    }

    /// Return only heads flagged as prunable.
    pub fn prunable_heads(&self, patterns: &[AttentionPattern]) -> Vec<HeadImportance> {
        self.rank_heads(patterns).into_iter().filter(|h| h.prunable).collect()
    }
}

impl Default for HeadPruner {
    fn default() -> Self {
        Self::new(0.3)
    }
}

// ---------------------------------------------------------------------------
// PatternClassifier
// ---------------------------------------------------------------------------

/// Classifies an attention pattern into a [`PatternType`].
pub struct PatternClassifier {
    /// Diagonal detection: fraction of total weight on the main diagonal.
    pub diagonal_threshold: f32,
    /// Vertical detection: fraction of total column weight in any single column.
    pub vertical_threshold: f32,
    /// Sparsity fraction above which a pattern is labelled Sparse.
    pub sparsity_threshold: f32,
    /// Entropy ratio above which a pattern is labelled Dense.
    pub dense_entropy_ratio: f32,
}

impl PatternClassifier {
    pub fn new() -> Self {
        Self {
            diagonal_threshold: 0.5,
            vertical_threshold: 0.4,
            sparsity_threshold: 0.8,
            dense_entropy_ratio: 0.9,
        }
    }

    /// Classify a pattern.
    pub fn classify(&self, pattern: &AttentionPattern) -> PatternType {
        if pattern.seq_len == 0 {
            return PatternType::Mixed;
        }

        // Check diagonal dominance.
        if self.is_diagonal(pattern) {
            return PatternType::Diagonal;
        }

        // Check vertical stripes.
        if self.is_vertical(pattern) {
            return PatternType::Vertical;
        }

        // Check sparsity early – a highly sparse matrix should not be
        // misclassified as block-diagonal or dense.
        let sparsity = SparsityAnalyzer::default().compute_sparsity(pattern);
        if sparsity >= self.sparsity_threshold {
            return PatternType::Sparse;
        }

        // Check dense / uniform before block-diagonal so that a uniformly
        // spread pattern is not accidentally matched by block structure.
        let norm_entropy = EntropyCalculator::normalized_entropy(pattern);
        if norm_entropy >= self.dense_entropy_ratio {
            return PatternType::Dense;
        }

        // Check block-diagonal.
        if self.is_block_diagonal(pattern) {
            return PatternType::BlockDiagonal;
        }

        PatternType::Mixed
    }

    /// True if attention is local (diagonal-dominated).
    pub fn is_local(&self, pattern: &AttentionPattern) -> bool {
        self.is_diagonal(pattern)
    }

    /// True if attention is global (vertical-dominated).
    pub fn is_global(&self, pattern: &AttentionPattern) -> bool {
        self.is_vertical(pattern)
    }

    /// True if the pattern appears positional (diagonal or block-diagonal).
    pub fn is_positional(&self, pattern: &AttentionPattern) -> bool {
        self.is_diagonal(pattern) || self.is_block_diagonal(pattern)
    }

    // -- internal helpers ---------------------------------------------------

    fn is_diagonal(&self, pattern: &AttentionPattern) -> bool {
        let n = pattern.seq_len;
        if n == 0 {
            return false;
        }
        let total: f32 = pattern.weights.iter().flat_map(|r| r.iter()).sum();
        if total == 0.0 {
            return false;
        }
        let diag_sum: f32 = (0..n).map(|i| pattern.weights[i][i]).sum();
        diag_sum / total >= self.diagonal_threshold
    }

    fn is_vertical(&self, pattern: &AttentionPattern) -> bool {
        let n = pattern.seq_len;
        if n == 0 {
            return false;
        }
        let total: f32 = pattern.weights.iter().flat_map(|r| r.iter()).sum();
        if total == 0.0 {
            return false;
        }
        for col in 0..n {
            let col_sum: f32 = (0..n).map(|row| pattern.weights[row][col]).sum();
            if col_sum / total >= self.vertical_threshold {
                return true;
            }
        }
        false
    }

    fn is_block_diagonal(&self, pattern: &AttentionPattern) -> bool {
        let n = pattern.seq_len;
        if n < 4 {
            return false;
        }
        // Try block sizes 2..=n/2 and check if blocks capture most weight.
        let total: f32 = pattern.weights.iter().flat_map(|r| r.iter()).sum();
        if total == 0.0 {
            return false;
        }
        for block_size in 2..=n / 2 {
            if n % block_size != 0 {
                continue;
            }
            let mut block_sum = 0.0_f32;
            for block_start in (0..n).step_by(block_size) {
                for i in block_start..block_start + block_size {
                    for j in block_start..block_start + block_size {
                        block_sum += pattern.weights[i][j];
                    }
                }
            }
            if block_sum / total > self.diagonal_threshold {
                return true;
            }
        }
        false
    }
}

impl Default for PatternClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AttentionVisualizer
// ---------------------------------------------------------------------------

/// Generates ASCII text visualizations of attention patterns.
pub struct AttentionVisualizer {
    /// Number of buckets per axis when down-sampling large patterns.
    pub max_display_size: usize,
    /// Characters used for intensity levels (low → high).
    pub palette: Vec<char>,
}

impl AttentionVisualizer {
    pub fn new() -> Self {
        Self { max_display_size: 40, palette: vec![' ', '░', '▒', '▓', '█'] }
    }

    /// Render a pattern as a multi-line ASCII string.
    pub fn render(&self, pattern: &AttentionPattern) -> String {
        let n = pattern.seq_len;
        if n == 0 {
            return String::from("(empty)");
        }

        let display_n = n.min(self.max_display_size);
        let bucket = |idx: usize| -> usize { idx * n / display_n };

        let mut lines = Vec::with_capacity(display_n + 2);
        lines.push(format!("Layer {} Head {} ({}×{})", pattern.layer_idx, pattern.head_idx, n, n));

        for row in 0..display_n {
            let ri = bucket(row);
            let mut chars = String::with_capacity(display_n);
            for col in 0..display_n {
                let ci = bucket(col);
                let w = pattern.weights[ri][ci];
                let idx = ((w * (self.palette.len() - 1) as f32).round() as usize)
                    .min(self.palette.len() - 1);
                chars.push(self.palette[idx]);
            }
            lines.push(chars);
        }
        lines.join("\n")
    }

    /// Render a compact one-line summary.
    pub fn summary_line(&self, pattern: &AttentionPattern) -> String {
        let stats = compute_stats(pattern);
        format!(
            "L{}H{} entropy={:.3} sparsity={:.1}% max={:.3}",
            pattern.layer_idx,
            pattern.head_idx,
            stats.entropy,
            stats.sparsity * 100.0,
            stats.max_weight,
        )
    }
}

impl Default for AttentionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Compute [`AttentionStats`] for a single pattern.
pub fn compute_stats(pattern: &AttentionPattern) -> AttentionStats {
    let entropy = EntropyCalculator::mean_entropy(pattern);
    let analyzer = SparsityAnalyzer::default();
    let sparsity = analyzer.compute_sparsity(pattern);
    let all_weights: Vec<f32> = pattern.weights.iter().flat_map(|r| r.iter().copied()).collect();
    let max_weight = all_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_weight = if all_weights.is_empty() {
        0.0
    } else {
        all_weights.iter().sum::<f32>() / all_weights.len() as f32
    };
    let effective_context_length = analyzer.mean_active_positions(pattern);

    AttentionStats { entropy, sparsity, max_weight, mean_weight, effective_context_length }
}

// ---------------------------------------------------------------------------
// AttentionAnalyzerEngine
// ---------------------------------------------------------------------------

/// Optimization suggestion produced by the engine.
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub description: String,
    pub priority: SuggestionPriority,
}

/// Priority of an optimization suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SuggestionPriority {
    Low,
    Medium,
    High,
}

/// Main orchestrator: capture patterns, then analyze / classify / suggest.
pub struct AttentionAnalyzerEngine {
    patterns: Vec<AttentionPattern>,
    classifier: PatternClassifier,
    pruner: HeadPruner,
    visualizer: AttentionVisualizer,
}

impl AttentionAnalyzerEngine {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            classifier: PatternClassifier::new(),
            pruner: HeadPruner::default(),
            visualizer: AttentionVisualizer::new(),
        }
    }

    /// Customize the prune threshold.
    pub fn with_prune_threshold(mut self, t: f32) -> Self {
        self.pruner.prune_threshold = t;
        self
    }

    /// Capture a new pattern for analysis.
    pub fn capture(&mut self, pattern: AttentionPattern) {
        self.patterns.push(pattern);
    }

    /// Number of captured patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Classify all captured patterns.
    pub fn classify_all(&self) -> Vec<(usize, usize, PatternType)> {
        self.patterns
            .iter()
            .map(|p| (p.layer_idx, p.head_idx, self.classifier.classify(p)))
            .collect()
    }

    /// Get stats for all captured patterns.
    pub fn stats_all(&self) -> Vec<AttentionStats> {
        self.patterns.iter().map(compute_stats).collect()
    }

    /// Rank heads by importance.
    pub fn rank_heads(&self) -> Vec<HeadImportance> {
        self.pruner.rank_heads(&self.patterns)
    }

    /// Identify prunable heads.
    pub fn prunable_heads(&self) -> Vec<HeadImportance> {
        self.pruner.prunable_heads(&self.patterns)
    }

    /// Render all patterns as ASCII visualizations.
    pub fn visualize_all(&self) -> Vec<String> {
        self.patterns.iter().map(|p| self.visualizer.render(p)).collect()
    }

    /// Generate optimization suggestions based on analysis.
    pub fn suggest_optimizations(&self) -> Result<Vec<OptimizationSuggestion>, AnalyzerError> {
        if self.patterns.is_empty() {
            return Err(AnalyzerError::NoPatterns);
        }

        let mut suggestions = Vec::new();

        // Suggestion: prune redundant heads.
        let prunable = self.prunable_heads();
        if !prunable.is_empty() {
            suggestions.push(OptimizationSuggestion {
                description: format!(
                    "{} head(s) appear prunable (importance < {:.2})",
                    prunable.len(),
                    self.pruner.prune_threshold,
                ),
                priority: SuggestionPriority::High,
            });
        }

        // Suggestion: use sparse kernels when sparsity is high.
        let sparse_count =
            self.classify_all().iter().filter(|(_, _, t)| *t == PatternType::Sparse).count();
        if sparse_count > 0 {
            suggestions.push(OptimizationSuggestion {
                description: format!(
                    "{sparse_count} head(s) are sparse – consider sparse attention kernels"
                ),
                priority: SuggestionPriority::Medium,
            });
        }

        // Suggestion: local/sliding-window kernels for diagonal patterns.
        let diag_count =
            self.classify_all().iter().filter(|(_, _, t)| *t == PatternType::Diagonal).count();
        if diag_count > 0 {
            suggestions.push(OptimizationSuggestion {
                description: format!(
                    "{diag_count} head(s) are diagonal – sliding-window kernel recommended"
                ),
                priority: SuggestionPriority::Medium,
            });
        }

        // Suggestion: global-token caching for vertical patterns.
        let vert_count =
            self.classify_all().iter().filter(|(_, _, t)| *t == PatternType::Vertical).count();
        if vert_count > 0 {
            suggestions.push(OptimizationSuggestion {
                description: format!("{vert_count} head(s) are vertical – cache global token(s)"),
                priority: SuggestionPriority::Low,
            });
        }

        Ok(suggestions)
    }

    /// Clear captured patterns.
    pub fn clear(&mut self) {
        self.patterns.clear();
    }
}

impl Default for AttentionAnalyzerEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a uniform attention pattern (every position equal weight).
    fn uniform_pattern(n: usize, layer: usize, head: usize) -> AttentionPattern {
        let w = 1.0 / n as f32;
        let weights = vec![vec![w; n]; n];
        AttentionPattern::new(layer, head, weights).unwrap()
    }

    /// Build a perfectly diagonal pattern (identity attention).
    fn diagonal_pattern(n: usize, layer: usize, head: usize) -> AttentionPattern {
        let mut weights = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            weights[i][i] = 1.0;
        }
        AttentionPattern::new(layer, head, weights).unwrap()
    }

    /// Build a vertical pattern where column 0 gets all weight.
    fn vertical_pattern(n: usize, layer: usize, head: usize) -> AttentionPattern {
        let mut weights = vec![vec![0.0_f32; n]; n];
        for row in &mut weights {
            row[0] = 1.0;
        }
        AttentionPattern::new(layer, head, weights).unwrap()
    }

    /// Build a block-diagonal pattern with given block size.
    fn block_diagonal_pattern(
        n: usize,
        block_size: usize,
        layer: usize,
        head: usize,
    ) -> AttentionPattern {
        let w = 1.0 / block_size as f32;
        let mut weights = vec![vec![0.0_f32; n]; n];
        for block_start in (0..n).step_by(block_size) {
            for i in block_start..block_start + block_size {
                for j in block_start..block_start + block_size {
                    weights[i][j] = w;
                }
            }
        }
        AttentionPattern::new(layer, head, weights).unwrap()
    }

    /// Build a sparse pattern (most weights near zero, a few high).
    fn sparse_pattern(n: usize, layer: usize, head: usize) -> AttentionPattern {
        let mut weights = vec![vec![0.001_f32; n]; n];
        // Put most weight on a single element per row.
        for i in 0..n {
            let total_small = 0.001 * (n - 1) as f32;
            weights[i][i % n] = 1.0 - total_small;
        }
        AttentionPattern::new(layer, head, weights).unwrap()
    }

    /// All-zero pattern.
    fn zero_pattern(n: usize, layer: usize, head: usize) -> AttentionPattern {
        let weights = vec![vec![0.0_f32; n]; n];
        AttentionPattern::new(layer, head, weights).unwrap()
    }

    /// Focused pattern: every row focuses entirely on position 0.
    fn focused_pattern(n: usize, layer: usize, head: usize) -> AttentionPattern {
        vertical_pattern(n, layer, head)
    }

    // -----------------------------------------------------------------------
    // AttentionPattern construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_pattern_new_valid() {
        let p = AttentionPattern::new(0, 0, vec![vec![1.0]]).unwrap();
        assert_eq!(p.seq_len, 1);
    }

    #[test]
    fn test_pattern_new_empty() {
        let err = AttentionPattern::new(0, 0, vec![]).unwrap_err();
        assert_eq!(err, AnalyzerError::EmptyWeights);
    }

    #[test]
    fn test_pattern_new_dimension_mismatch() {
        let err = AttentionPattern::new(0, 0, vec![vec![1.0, 0.0], vec![1.0]]).unwrap_err();
        assert!(matches!(err, AnalyzerError::DimensionMismatch { row: 1, .. }));
    }

    #[test]
    fn test_pattern_preserves_metadata() {
        let p = uniform_pattern(4, 3, 7);
        assert_eq!(p.layer_idx, 3);
        assert_eq!(p.head_idx, 7);
        assert_eq!(p.seq_len, 4);
    }

    #[test]
    fn test_pattern_single_element() {
        let p = AttentionPattern::new(0, 0, vec![vec![1.0]]).unwrap();
        assert_eq!(p.weights[0][0], 1.0);
    }

    // -----------------------------------------------------------------------
    // PatternType Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_pattern_type_display() {
        assert_eq!(format!("{}", PatternType::Diagonal), "diagonal");
        assert_eq!(format!("{}", PatternType::Vertical), "vertical");
        assert_eq!(format!("{}", PatternType::BlockDiagonal), "block-diagonal");
        assert_eq!(format!("{}", PatternType::Sparse), "sparse");
        assert_eq!(format!("{}", PatternType::Dense), "dense");
        assert_eq!(format!("{}", PatternType::Periodic), "periodic");
        assert_eq!(format!("{}", PatternType::Mixed), "mixed");
    }

    // -----------------------------------------------------------------------
    // SparsityAnalyzer
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparsity_uniform() {
        let p = uniform_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.01);
        // 1/4 = 0.25, all above 0.01
        assert_eq!(sa.compute_sparsity(&p), 0.0);
    }

    #[test]
    fn test_sparsity_all_zero() {
        let p = zero_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.01);
        assert_eq!(sa.compute_sparsity(&p), 1.0);
    }

    #[test]
    fn test_sparsity_diagonal() {
        let p = diagonal_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.01);
        // 12 of 16 are zero → 0.75
        assert!((sa.compute_sparsity(&p) - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_sparsity_custom_threshold() {
        let p = uniform_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.5);
        // 0.25 < 0.5 → all sparse
        assert_eq!(sa.compute_sparsity(&p), 1.0);
    }

    #[test]
    fn test_active_positions_per_query_uniform() {
        let p = uniform_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.01);
        assert_eq!(sa.active_positions_per_query(&p), vec![4, 4, 4, 4]);
    }

    #[test]
    fn test_active_positions_per_query_diagonal() {
        let p = diagonal_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.01);
        assert_eq!(sa.active_positions_per_query(&p), vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_mean_active_positions() {
        let p = diagonal_pattern(4, 0, 0);
        let sa = SparsityAnalyzer::new(0.01);
        assert!((sa.mean_active_positions(&p) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparsity_default_threshold() {
        let sa = SparsityAnalyzer::default();
        assert_eq!(sa.threshold, 0.01);
    }

    // -----------------------------------------------------------------------
    // EntropyCalculator
    // -----------------------------------------------------------------------

    #[test]
    fn test_entropy_uniform_is_max() {
        let p = uniform_pattern(8, 0, 0);
        let entropy = EntropyCalculator::mean_entropy(&p);
        let max_e = EntropyCalculator::max_entropy(8);
        assert!((entropy - max_e).abs() < 0.01);
    }

    #[test]
    fn test_entropy_focused_is_low() {
        let p = focused_pattern(8, 0, 0);
        let entropy = EntropyCalculator::mean_entropy(&p);
        assert!(entropy < 0.1, "focused entropy should be near 0, got {entropy}");
    }

    #[test]
    fn test_entropy_diagonal_single_peak() {
        let p = diagonal_pattern(4, 0, 0);
        let entropy = EntropyCalculator::mean_entropy(&p);
        // Each row has a single 1.0 → entropy ≈ 0.
        assert!(entropy < 0.01);
    }

    #[test]
    fn test_max_entropy_single_token() {
        assert_eq!(EntropyCalculator::max_entropy(1), 0.0);
    }

    #[test]
    fn test_max_entropy_two_tokens() {
        let expected = (2.0_f32).ln();
        assert!((EntropyCalculator::max_entropy(2) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_normalized_entropy_uniform() {
        let p = uniform_pattern(8, 0, 0);
        let ne = EntropyCalculator::normalized_entropy(&p);
        assert!((ne - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_normalized_entropy_focused() {
        let p = focused_pattern(8, 0, 0);
        let ne = EntropyCalculator::normalized_entropy(&p);
        assert!(ne < 0.05);
    }

    #[test]
    fn test_entropy_bounds() {
        // Property: entropy is always in [0, log(n)].
        for n in 2..=16 {
            let p = uniform_pattern(n, 0, 0);
            let e = EntropyCalculator::mean_entropy(&p);
            let max_e = EntropyCalculator::max_entropy(n);
            assert!(e >= -0.001, "entropy < 0 for n={n}");
            assert!(e <= max_e + 0.01, "entropy > max for n={n}");
        }
    }

    #[test]
    fn test_entropy_empty_pattern_no_panic() {
        // Create a 1-element pattern; entropy of a single position is 0.
        let p = AttentionPattern::new(0, 0, vec![vec![1.0]]).unwrap();
        let e = EntropyCalculator::mean_entropy(&p);
        assert!(e.abs() < 0.01);
    }

    #[test]
    fn test_row_entropy_two_element_uniform() {
        let e = EntropyCalculator::row_entropy(&[0.5, 0.5]);
        let expected = (2.0_f32).ln();
        assert!((e - expected).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // HeadPruner
    // -----------------------------------------------------------------------

    #[test]
    fn test_pruner_focused_head_is_prunable() {
        let p = focused_pattern(8, 0, 0);
        let pruner = HeadPruner::new(0.5);
        let h = pruner.score_head(&p);
        assert!(h.prunable, "focused head should be prunable");
    }

    #[test]
    fn test_pruner_uniform_head_not_prunable() {
        let p = uniform_pattern(8, 0, 0);
        let pruner = HeadPruner::new(0.5);
        let h = pruner.score_head(&p);
        assert!(!h.prunable, "uniform head should NOT be prunable");
    }

    #[test]
    fn test_pruner_rank_heads_ordering() {
        let patterns =
            vec![uniform_pattern(8, 0, 0), focused_pattern(8, 0, 1), diagonal_pattern(8, 0, 2)];
        let pruner = HeadPruner::new(0.3);
        let ranked = pruner.rank_heads(&patterns);
        // Should be descending by importance.
        for pair in ranked.windows(2) {
            assert!(pair[0].importance_score >= pair[1].importance_score);
        }
    }

    #[test]
    fn test_pruner_prunable_heads_subset() {
        let patterns = vec![uniform_pattern(8, 0, 0), focused_pattern(8, 0, 1)];
        let pruner = HeadPruner::new(0.5);
        let prunable = pruner.prunable_heads(&patterns);
        assert!(prunable.iter().all(|h| h.prunable));
    }

    #[test]
    fn test_pruner_default_threshold() {
        let pruner = HeadPruner::default();
        assert!((pruner.prune_threshold - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_head_importance_score_bounds() {
        for n in [2, 4, 8, 16] {
            let p = uniform_pattern(n, 0, 0);
            let pruner = HeadPruner::new(0.3);
            let h = pruner.score_head(&p);
            assert!((0.0..=1.0).contains(&h.importance_score), "score out of [0,1] for n={n}");
        }
    }

    // -----------------------------------------------------------------------
    // PatternClassifier
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_diagonal() {
        let p = diagonal_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert_eq!(c.classify(&p), PatternType::Diagonal);
    }

    #[test]
    fn test_classify_vertical() {
        let p = vertical_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert_eq!(c.classify(&p), PatternType::Vertical);
    }

    #[test]
    fn test_classify_block_diagonal() {
        let p = block_diagonal_pattern(8, 4, 0, 0);
        let c = PatternClassifier::new();
        assert_eq!(c.classify(&p), PatternType::BlockDiagonal);
    }

    #[test]
    fn test_classify_dense_uniform() {
        let p = uniform_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert_eq!(c.classify(&p), PatternType::Dense);
    }

    #[test]
    fn test_classify_sparse() {
        // All weights below the default sparsity threshold (0.01).
        let n = 8;
        let weights = vec![vec![0.001_f32; n]; n];
        let p = AttentionPattern::new(0, 0, weights).unwrap();
        let c = PatternClassifier::new();
        let t = c.classify(&p);
        assert_eq!(t, PatternType::Sparse);
    }

    #[test]
    fn test_is_local_diagonal() {
        let p = diagonal_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert!(c.is_local(&p));
    }

    #[test]
    fn test_is_global_vertical() {
        let p = vertical_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert!(c.is_global(&p));
    }

    #[test]
    fn test_is_positional_diagonal() {
        let p = diagonal_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert!(c.is_positional(&p));
    }

    #[test]
    fn test_is_positional_block_diagonal() {
        let p = block_diagonal_pattern(8, 4, 0, 0);
        let c = PatternClassifier::new();
        assert!(c.is_positional(&p));
    }

    #[test]
    fn test_uniform_not_local() {
        let p = uniform_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert!(!c.is_local(&p));
    }

    #[test]
    fn test_diagonal_not_global() {
        let p = diagonal_pattern(8, 0, 0);
        let c = PatternClassifier::new();
        assert!(!c.is_global(&p));
    }

    #[test]
    fn test_classify_mixed_zero() {
        // All-zero → no dominant pattern → Mixed (diagonal check fails because total=0).
        let p = zero_pattern(4, 0, 0);
        let c = PatternClassifier::new();
        let t = c.classify(&p);
        // All zeros → sparsity is 1.0 (all below 0.01) → Sparse
        assert_eq!(t, PatternType::Sparse);
    }

    #[test]
    fn test_classify_single_token() {
        let p = AttentionPattern::new(0, 0, vec![vec![1.0]]).unwrap();
        let c = PatternClassifier::new();
        // Single token: diagonal sum/total == 1.0 → Diagonal
        assert_eq!(c.classify(&p), PatternType::Diagonal);
    }

    #[test]
    fn test_classifier_default() {
        let c = PatternClassifier::default();
        assert!((c.diagonal_threshold - 0.5).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // AttentionVisualizer
    // -----------------------------------------------------------------------

    #[test]
    fn test_visualizer_renders_header() {
        let p = uniform_pattern(4, 2, 5);
        let vis = AttentionVisualizer::new();
        let output = vis.render(&p);
        assert!(output.contains("Layer 2 Head 5"));
    }

    #[test]
    fn test_visualizer_renders_correct_lines() {
        let p = uniform_pattern(4, 0, 0);
        let vis = AttentionVisualizer::new();
        let output = vis.render(&p);
        // 1 header + 4 rows = 5 lines
        assert_eq!(output.lines().count(), 5);
    }

    #[test]
    fn test_visualizer_empty_pattern() {
        // Cannot create 0-length via new (would fail), but render handles it.
        let p = AttentionPattern { layer_idx: 0, head_idx: 0, weights: vec![], seq_len: 0 };
        let vis = AttentionVisualizer::new();
        assert_eq!(vis.render(&p), "(empty)");
    }

    #[test]
    fn test_visualizer_diagonal_intensity() {
        let p = diagonal_pattern(4, 0, 0);
        let vis = AttentionVisualizer::new();
        let output = vis.render(&p);
        // Diagonal entries should use the highest intensity character '█'.
        assert!(output.contains('█'));
    }

    #[test]
    fn test_summary_line_format() {
        let p = uniform_pattern(4, 1, 2);
        let vis = AttentionVisualizer::new();
        let line = vis.summary_line(&p);
        assert!(line.starts_with("L1H2"));
        assert!(line.contains("entropy="));
        assert!(line.contains("sparsity="));
        assert!(line.contains("max="));
    }

    #[test]
    fn test_visualizer_default() {
        let vis = AttentionVisualizer::default();
        assert_eq!(vis.max_display_size, 40);
        assert_eq!(vis.palette.len(), 5);
    }

    // -----------------------------------------------------------------------
    // compute_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_max_weight_uniform() {
        let p = uniform_pattern(4, 0, 0);
        let stats = compute_stats(&p);
        assert!((stats.max_weight - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_stats_max_weight_diagonal() {
        let p = diagonal_pattern(4, 0, 0);
        let stats = compute_stats(&p);
        assert!((stats.max_weight - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_stats_mean_weight_uniform() {
        let p = uniform_pattern(4, 0, 0);
        let stats = compute_stats(&p);
        assert!((stats.mean_weight - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_stats_mean_weight_diagonal() {
        let p = diagonal_pattern(4, 0, 0);
        let stats = compute_stats(&p);
        // 4 ones + 12 zeros → mean = 4/16 = 0.25
        assert!((stats.mean_weight - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_stats_effective_context_uniform() {
        let p = uniform_pattern(4, 0, 0);
        let stats = compute_stats(&p);
        // All weights 0.25 > 0.01 → 4 active per query
        assert!((stats.effective_context_length - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_stats_effective_context_diagonal() {
        let p = diagonal_pattern(4, 0, 0);
        let stats = compute_stats(&p);
        // Only 1 active per query
        assert!((stats.effective_context_length - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_stats_sparsity_matches_analyzer() {
        let p = diagonal_pattern(8, 0, 0);
        let stats = compute_stats(&p);
        let sa = SparsityAnalyzer::default();
        assert!((stats.sparsity - sa.compute_sparsity(&p)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // AttentionAnalyzerEngine
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_capture_and_count() {
        let mut engine = AttentionAnalyzerEngine::new();
        assert_eq!(engine.pattern_count(), 0);
        engine.capture(uniform_pattern(4, 0, 0));
        assert_eq!(engine.pattern_count(), 1);
        engine.capture(diagonal_pattern(4, 0, 1));
        assert_eq!(engine.pattern_count(), 2);
    }

    #[test]
    fn test_engine_classify_all() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(diagonal_pattern(8, 0, 0));
        engine.capture(uniform_pattern(8, 0, 1));
        let classes = engine.classify_all();
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0].2, PatternType::Diagonal);
        assert_eq!(classes[1].2, PatternType::Dense);
    }

    #[test]
    fn test_engine_stats_all() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(uniform_pattern(4, 0, 0));
        let stats = engine.stats_all();
        assert_eq!(stats.len(), 1);
        assert!((stats[0].max_weight - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_engine_rank_heads() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(uniform_pattern(8, 0, 0));
        engine.capture(focused_pattern(8, 0, 1));
        let ranked = engine.rank_heads();
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].importance_score >= ranked[1].importance_score);
    }

    #[test]
    fn test_engine_prunable_heads() {
        let mut engine = AttentionAnalyzerEngine::new().with_prune_threshold(0.5);
        engine.capture(uniform_pattern(8, 0, 0));
        engine.capture(focused_pattern(8, 0, 1));
        let prunable = engine.prunable_heads();
        // Focused head should be prunable.
        assert!(!prunable.is_empty());
        assert!(prunable.iter().all(|h| h.head_idx == 1));
    }

    #[test]
    fn test_engine_visualize_all() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(uniform_pattern(4, 0, 0));
        let vis = engine.visualize_all();
        assert_eq!(vis.len(), 1);
        assert!(vis[0].contains("Layer 0 Head 0"));
    }

    #[test]
    fn test_engine_suggest_no_patterns_error() {
        let engine = AttentionAnalyzerEngine::new();
        let err = engine.suggest_optimizations().unwrap_err();
        assert_eq!(err, AnalyzerError::NoPatterns);
    }

    #[test]
    fn test_engine_suggest_prunable() {
        let mut engine = AttentionAnalyzerEngine::new().with_prune_threshold(0.5);
        engine.capture(focused_pattern(8, 0, 0));
        let suggestions = engine.suggest_optimizations().unwrap();
        assert!(suggestions.iter().any(|s| s.description.contains("prunable")));
    }

    #[test]
    fn test_engine_suggest_sparse_kernel() {
        let n = 8;
        let weights = vec![vec![0.001_f32; n]; n];
        let p = AttentionPattern::new(0, 0, weights).unwrap();
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(p);
        let suggestions = engine.suggest_optimizations().unwrap();
        assert!(suggestions.iter().any(|s| s.description.contains("sparse")));
    }

    #[test]
    fn test_engine_suggest_sliding_window() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(diagonal_pattern(8, 0, 0));
        let suggestions = engine.suggest_optimizations().unwrap();
        assert!(suggestions.iter().any(|s| s.description.contains("sliding-window")));
    }

    #[test]
    fn test_engine_suggest_global_cache() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(vertical_pattern(8, 0, 0));
        let suggestions = engine.suggest_optimizations().unwrap();
        assert!(suggestions.iter().any(|s| s.description.contains("global token")));
    }

    #[test]
    fn test_engine_clear() {
        let mut engine = AttentionAnalyzerEngine::new();
        engine.capture(uniform_pattern(4, 0, 0));
        engine.clear();
        assert_eq!(engine.pattern_count(), 0);
    }

    #[test]
    fn test_engine_default() {
        let engine = AttentionAnalyzerEngine::default();
        assert_eq!(engine.pattern_count(), 0);
    }

    #[test]
    fn test_engine_with_prune_threshold() {
        let engine = AttentionAnalyzerEngine::new().with_prune_threshold(0.9);
        assert!((engine.pruner.prune_threshold - 0.9).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_head_single_token() {
        let p = AttentionPattern::new(0, 0, vec![vec![1.0]]).unwrap();
        let stats = compute_stats(&p);
        assert!((stats.max_weight - 1.0).abs() < 1e-5);
        assert!((stats.mean_weight - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_large_pattern_no_panic() {
        let n = 128;
        let p = uniform_pattern(n, 0, 0);
        let _ = compute_stats(&p);
        let c = PatternClassifier::new();
        let _ = c.classify(&p);
    }

    #[test]
    fn test_error_display() {
        let e = AnalyzerError::EmptyWeights;
        assert_eq!(format!("{e}"), "attention weight matrix is empty");
        let e2 = AnalyzerError::DimensionMismatch { row: 2, expected: 4, got: 3 };
        assert!(format!("{e2}").contains("row 2"));
        let e3 = AnalyzerError::NoPatterns;
        assert!(format!("{e3}").contains("no patterns"));
    }

    #[test]
    fn test_suggestion_priority_ordering() {
        assert!(SuggestionPriority::Low < SuggestionPriority::Medium);
        assert!(SuggestionPriority::Medium < SuggestionPriority::High);
    }

    // -----------------------------------------------------------------------
    // Property-style tests: entropy bounds for various sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_entropy_bounds_focused() {
        for n in [2, 4, 8, 16, 32] {
            let p = focused_pattern(n, 0, 0);
            let e = EntropyCalculator::mean_entropy(&p);
            assert!(e < 0.1, "focused entropy too high for n={n}: {e}");
        }
    }

    #[test]
    fn test_entropy_bounds_uniform() {
        for n in [2, 4, 8, 16, 32] {
            let p = uniform_pattern(n, 0, 0);
            let e = EntropyCalculator::mean_entropy(&p);
            let max_e = EntropyCalculator::max_entropy(n);
            assert!(
                (e - max_e).abs() < 0.05,
                "uniform entropy far from max for n={n}: {e} vs {max_e}"
            );
        }
    }

    #[test]
    fn test_normalized_entropy_always_in_unit_range() {
        let patterns = [
            uniform_pattern(8, 0, 0),
            diagonal_pattern(8, 0, 0),
            focused_pattern(8, 0, 0),
            sparse_pattern(8, 0, 0),
        ];
        for p in &patterns {
            let ne = EntropyCalculator::normalized_entropy(p);
            assert!((0.0..=1.0).contains(&ne), "normalized entropy {ne} out of [0,1]");
        }
    }

    // -----------------------------------------------------------------------
    // Pattern detection robustness
    // -----------------------------------------------------------------------

    #[test]
    fn test_diagonal_detected_various_sizes() {
        let c = PatternClassifier::new();
        for n in [2, 4, 8, 16] {
            let p = diagonal_pattern(n, 0, 0);
            assert_eq!(c.classify(&p), PatternType::Diagonal, "diagonal not detected for n={n}");
        }
    }

    #[test]
    fn test_vertical_detected_various_sizes() {
        let c = PatternClassifier::new();
        // Start at n=3; n=2 is ambiguous (diagonal ratio = 0.5).
        for n in [3, 4, 8, 16] {
            let p = vertical_pattern(n, 0, 0);
            assert_eq!(c.classify(&p), PatternType::Vertical, "vertical not detected for n={n}");
        }
    }

    #[test]
    fn test_block_diagonal_detected_multiple_block_sizes() {
        let c = PatternClassifier::new();
        // 8 with blocks of 2 and 4.
        for bs in [2, 4] {
            let p = block_diagonal_pattern(8, bs, 0, 0);
            let t = c.classify(&p);
            assert!(
                t == PatternType::BlockDiagonal || t == PatternType::Diagonal,
                "block-diagonal not detected for block_size={bs}, got {t}"
            );
        }
    }
}
