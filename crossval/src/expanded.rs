//! Expanded cross-validation framework for improved parity testing
//!
//! This module adds:
//! - Per-layer activation comparison with layer-type-aware thresholds
//! - Automated divergence bisection (binary search for first divergent layer)
//! - Batch comparison across multiple prompts
//! - Visual diff report generation (Markdown)
//! - Regression test snapshot generation from successful comparisons

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Layer types & per-type thresholds
// ---------------------------------------------------------------------------

/// Transformer layer types with distinct numerical tolerance profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Self-attention sublayer (QKV projections + output projection)
    Attention,
    /// Feed-forward network sublayer (gate / up / down projections)
    Ffn,
    /// Normalization sublayer (RMSNorm or LayerNorm)
    Norm,
    /// Embedding or output projection
    Embedding,
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attention => write!(f, "attention"),
            Self::Ffn => write!(f, "ffn"),
            Self::Norm => write!(f, "norm"),
            Self::Embedding => write!(f, "embedding"),
        }
    }
}

impl LayerType {
    /// Infer layer type from a layer name string.
    ///
    /// Uses common naming conventions from GGUF / transformer models:
    /// - names containing `attn`, `self_attn`, `attention` → Attention
    /// - names containing `ffn`, `mlp`, `feed_forward` → Ffn
    /// - names containing `norm`, `ln`, `layernorm`, `rmsnorm` → Norm
    /// - names containing `embed`, `tok_embd`, `lm_head` → Embedding
    pub fn from_layer_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("norm") || lower.contains("ln") || lower.contains("rmsnorm") {
            Self::Norm
        } else if lower.contains("embed") || lower.contains("embd") || lower.contains("lm_head") {
            Self::Embedding
        } else if lower.contains("attn") || lower.contains("attention") || lower.contains("self_") {
            Self::Attention
        } else if lower.contains("ffn") || lower.contains("mlp") || lower.contains("feed_forward") {
            Self::Ffn
        } else if lower.contains("output") {
            Self::Embedding
        } else {
            // Default to FFN since it's the most common sublayer
            Self::Ffn
        }
    }
}

/// Cosine similarity thresholds keyed by layer type.
///
/// Norm layers require tighter agreement (they are scalar multipliers),
/// while FFN layers can tolerate slightly more numerical noise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerThresholds {
    /// Per-type minimum cosine similarity (values below trigger divergence)
    pub thresholds: BTreeMap<LayerType, f32>,
}

impl Default for LayerThresholds {
    fn default() -> Self {
        let mut thresholds = BTreeMap::new();
        // Norm layers: tight tolerance (scalar scale factors)
        thresholds.insert(LayerType::Norm, 0.9999);
        // Attention layers: medium tolerance (softmax amplifies small diffs)
        thresholds.insert(LayerType::Attention, 0.999);
        // FFN layers: slightly relaxed (accumulation noise in wide matmuls)
        thresholds.insert(LayerType::Ffn, 0.998);
        // Embedding: tight (lookup, no accumulation)
        thresholds.insert(LayerType::Embedding, 0.9999);
        Self { thresholds }
    }
}

impl LayerThresholds {
    /// Get the threshold for a given layer type, falling back to 0.999.
    pub fn get(&self, layer_type: LayerType) -> f32 {
        self.thresholds.get(&layer_type).copied().unwrap_or(0.999)
    }
}

// ---------------------------------------------------------------------------
// Per-layer activation snapshot & comparison
// ---------------------------------------------------------------------------

/// A snapshot of activations captured at a specific layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSnapshot {
    /// Human-readable layer name (e.g., "layers.0.attn_norm")
    pub layer_name: String,
    /// Inferred layer type
    pub layer_type: LayerType,
    /// Layer index in the model (0-based)
    pub layer_index: usize,
    /// Flattened activation values (row-major)
    pub values: Vec<f32>,
}

/// Result of comparing activations at a single layer between two implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerComparisonResult {
    /// Layer name
    pub layer_name: String,
    /// Inferred layer type
    pub layer_type: LayerType,
    /// Layer index
    pub layer_index: usize,
    /// Cosine similarity between the two activation vectors
    pub cosine_similarity: f32,
    /// L2 (Euclidean) distance
    pub l2_distance: f32,
    /// Maximum absolute element-wise difference
    pub max_abs_diff: f32,
    /// Mean squared error
    pub mse: f32,
    /// Whether this layer passes the layer-type-aware threshold
    pub passed: bool,
}

/// Compare activations layer-by-layer between Rust and C++ snapshots.
///
/// The two snapshot vectors must be aligned: `rust_snapshots[i]` and
/// `cpp_snapshots[i]` correspond to the same logical layer.
pub fn compare_layer_activations(
    rust_snapshots: &[ActivationSnapshot],
    cpp_snapshots: &[ActivationSnapshot],
    thresholds: &LayerThresholds,
) -> Vec<LayerComparisonResult> {
    let n = rust_snapshots.len().min(cpp_snapshots.len());
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let rs = &rust_snapshots[i];
        let cpp = &cpp_snapshots[i];
        let cos_sim = cosine_similarity(&rs.values, &cpp.values);
        let l2 = l2_distance(&rs.values, &cpp.values);
        let max_abs = max_abs_diff(&rs.values, &cpp.values);
        let mse = mean_squared_error(&rs.values, &cpp.values);
        let threshold = thresholds.get(rs.layer_type);

        results.push(LayerComparisonResult {
            layer_name: rs.layer_name.clone(),
            layer_type: rs.layer_type,
            layer_index: rs.layer_index,
            cosine_similarity: cos_sim,
            l2_distance: l2,
            max_abs_diff: max_abs,
            mse,
            passed: cos_sim >= threshold,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Divergence bisection
// ---------------------------------------------------------------------------

/// Result of binary-search divergence bisection across layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisectionResult {
    /// Index of the first layer that diverges (None = all layers match)
    pub first_divergent_layer: Option<usize>,
    /// Total layers examined
    pub total_layers: usize,
    /// Number of probe comparisons performed (≤ log2(N) + 1)
    pub probes: usize,
    /// Comparison result at the divergent layer (if found)
    pub divergent_detail: Option<LayerComparisonResult>,
}

/// Binary-search for the first divergent layer.
///
/// Instead of comparing every layer sequentially (O(N)), this performs a
/// binary search on the layer index to find the earliest divergence in
/// O(log N) comparisons. Each "probe" compares a single layer pair.
///
/// The algorithm assumes that once divergence appears at layer `k`, all
/// subsequent layers also diverge (monotonicity assumption for forward-pass
/// error propagation).
pub fn bisect_divergence(
    rust_snapshots: &[ActivationSnapshot],
    cpp_snapshots: &[ActivationSnapshot],
    thresholds: &LayerThresholds,
) -> BisectionResult {
    let n = rust_snapshots.len().min(cpp_snapshots.len());
    if n == 0 {
        return BisectionResult {
            first_divergent_layer: None,
            total_layers: 0,
            probes: 0,
            divergent_detail: None,
        };
    }

    let mut probes = 0;
    let mut lo: usize = 0;
    let mut hi: usize = n;

    // Binary search: find smallest index where divergence occurs
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        probes += 1;
        let cmp = compare_single_layer(&rust_snapshots[mid], &cpp_snapshots[mid], thresholds);
        if cmp.passed {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // lo == hi == first divergent index (or n if none diverged)
    if lo < n {
        probes += 1;
        let detail = compare_single_layer(&rust_snapshots[lo], &cpp_snapshots[lo], thresholds);
        BisectionResult {
            first_divergent_layer: Some(lo),
            total_layers: n,
            probes,
            divergent_detail: Some(detail),
        }
    } else {
        BisectionResult {
            first_divergent_layer: None,
            total_layers: n,
            probes,
            divergent_detail: None,
        }
    }
}

fn compare_single_layer(
    rs: &ActivationSnapshot,
    cpp: &ActivationSnapshot,
    thresholds: &LayerThresholds,
) -> LayerComparisonResult {
    let cos_sim = cosine_similarity(&rs.values, &cpp.values);
    let l2 = l2_distance(&rs.values, &cpp.values);
    let max_abs = max_abs_diff(&rs.values, &cpp.values);
    let mse = mean_squared_error(&rs.values, &cpp.values);
    let threshold = thresholds.get(rs.layer_type);

    LayerComparisonResult {
        layer_name: rs.layer_name.clone(),
        layer_type: rs.layer_type,
        layer_index: rs.layer_index,
        cosine_similarity: cos_sim,
        l2_distance: l2,
        max_abs_diff: max_abs,
        mse,
        passed: cos_sim >= threshold,
    }
}

// ---------------------------------------------------------------------------
// Batch comparison
// ---------------------------------------------------------------------------

/// A single prompt entry for batch comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptEntry {
    /// Human-readable label for the prompt
    pub label: String,
    /// The prompt text
    pub prompt: String,
}

/// Per-prompt result inside a batch comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptComparisonResult {
    /// Label from the `PromptEntry`
    pub label: String,
    /// Per-layer comparison results for this prompt
    pub layer_results: Vec<LayerComparisonResult>,
    /// Index of the first divergent layer (None if all pass)
    pub first_divergent_layer: Option<usize>,
    /// Whether all layers passed
    pub all_passed: bool,
}

/// Aggregate result of batch comparison across multiple prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchComparisonResult {
    /// Per-prompt results
    pub prompt_results: Vec<PromptComparisonResult>,
    /// Number of prompts that passed all layer checks
    pub passed_count: usize,
    /// Total prompts
    pub total_count: usize,
    /// Overall pass rate [0.0, 1.0]
    pub pass_rate: f64,
}

/// Activation provider trait — callers supply a closure that captures
/// activations for a given prompt.
///
/// The closure receives a prompt string and returns a vector of
/// `ActivationSnapshot` for each layer.
pub type ActivationCaptureFn = dyn Fn(&str) -> anyhow::Result<Vec<ActivationSnapshot>>;

/// Compare multiple prompts in batch.
///
/// For each prompt, `rust_capture` and `cpp_capture` are called to obtain
/// layer activations, which are then compared using `thresholds`.
pub fn batch_compare(
    prompts: &[PromptEntry],
    rust_capture: &ActivationCaptureFn,
    cpp_capture: &ActivationCaptureFn,
    thresholds: &LayerThresholds,
) -> anyhow::Result<BatchComparisonResult> {
    let mut prompt_results = Vec::with_capacity(prompts.len());

    for entry in prompts {
        let rs_snaps = rust_capture(&entry.prompt)?;
        let cpp_snaps = cpp_capture(&entry.prompt)?;

        let layer_results = compare_layer_activations(&rs_snaps, &cpp_snaps, thresholds);
        let first_divergent = layer_results.iter().position(|r| !r.passed);
        let all_passed = first_divergent.is_none();

        prompt_results.push(PromptComparisonResult {
            label: entry.label.clone(),
            layer_results,
            first_divergent_layer: first_divergent,
            all_passed,
        });
    }

    let passed_count = prompt_results.iter().filter(|r| r.all_passed).count();
    let total_count = prompt_results.len();
    let pass_rate = if total_count > 0 { passed_count as f64 / total_count as f64 } else { 1.0 };

    Ok(BatchComparisonResult { prompt_results, passed_count, total_count, pass_rate })
}

// ---------------------------------------------------------------------------
// Markdown diff report
// ---------------------------------------------------------------------------

/// Generate a Markdown-formatted diff report from layer comparison results.
pub fn generate_diff_report(
    model_name: &str,
    results: &[LayerComparisonResult],
    thresholds: &LayerThresholds,
) -> String {
    let mut md = String::new();

    md.push_str("# Cross-Validation Diff Report\n\n");
    md.push_str(&format!("**Model:** `{}`\n\n", model_name));

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let status = if passed == total { "✅ PASS" } else { "❌ FAIL" };
    md.push_str(&format!("**Status:** {} ({}/{} layers passed)\n\n", status, passed, total));

    // Thresholds table
    md.push_str("## Thresholds\n\n");
    md.push_str("| Layer Type | Min Cosine Similarity |\n");
    md.push_str("|------------|----------------------|\n");
    for (lt, &thresh) in &thresholds.thresholds {
        md.push_str(&format!("| {} | {:.4} |\n", lt, thresh));
    }
    md.push('\n');

    // Results table
    md.push_str("## Per-Layer Results\n\n");
    md.push_str("| # | Layer | Type | Cosine Sim | L2 Dist | Max Abs | MSE | Status |\n");
    md.push_str("|---|-------|------|-----------|---------|---------|-----|--------|\n");

    for r in results {
        let icon = if r.passed { "✅" } else { "❌" };
        md.push_str(&format!(
            "| {} | {} | {} | {:.6} | {:.6} | {:.6} | {:.2e} | {} |\n",
            r.layer_index,
            r.layer_name,
            r.layer_type,
            r.cosine_similarity,
            r.l2_distance,
            r.max_abs_diff,
            r.mse,
            icon,
        ));
    }
    md.push('\n');

    // Divergence summary
    if let Some(first_fail) = results.iter().find(|r| !r.passed) {
        md.push_str("## First Divergence\n\n");
        md.push_str(&format!(
            "Layer **{}** (index {}, type: {}) diverged with cosine similarity \
             {:.6} (threshold: {:.4}).\n\n",
            first_fail.layer_name,
            first_fail.layer_index,
            first_fail.layer_type,
            first_fail.cosine_similarity,
            thresholds.get(first_fail.layer_type),
        ));
    }

    md
}

/// Generate a Markdown batch report covering multiple prompts.
pub fn generate_batch_report(
    model_name: &str,
    batch: &BatchComparisonResult,
    thresholds: &LayerThresholds,
) -> String {
    let mut md = String::new();

    md.push_str("# Batch Cross-Validation Report\n\n");
    md.push_str(&format!("**Model:** `{}`\n\n", model_name));
    md.push_str(&format!(
        "**Overall:** {}/{} prompts passed ({:.1}%)\n\n",
        batch.passed_count,
        batch.total_count,
        batch.pass_rate * 100.0,
    ));

    // Summary table
    md.push_str("## Prompt Summary\n\n");
    md.push_str("| Prompt | Layers | First Divergence | Status |\n");
    md.push_str("|--------|--------|-----------------|--------|\n");

    for pr in &batch.prompt_results {
        let status = if pr.all_passed { "✅" } else { "❌" };
        let diverge = match pr.first_divergent_layer {
            Some(idx) => format!("layer {}", idx),
            None => "—".to_string(),
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            pr.label,
            pr.layer_results.len(),
            diverge,
            status,
        ));
    }
    md.push('\n');

    // Detailed per-prompt reports for failures
    for pr in batch.prompt_results.iter().filter(|r| !r.all_passed) {
        md.push_str(&format!("### {} (FAILED)\n\n", pr.label));
        md.push_str(&generate_diff_report(model_name, &pr.layer_results, thresholds));
        md.push('\n');
    }

    md
}

// ---------------------------------------------------------------------------
// Regression test snapshot generation
// ---------------------------------------------------------------------------

/// A regression snapshot captured from a successful comparison.
///
/// This can be serialized to JSON and stored alongside the test suite.
/// Future runs compare against these snapshots to detect regressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSnapshot {
    /// Schema version
    pub version: u32,
    /// Timestamp of snapshot creation (RFC 3339)
    pub timestamp: String,
    /// Model identifier
    pub model: String,
    /// Prompt used
    pub prompt: String,
    /// Per-layer cosine similarities (golden reference)
    pub layer_cosine_sims: Vec<f32>,
    /// Per-layer MSE values (golden reference)
    pub layer_mse: Vec<f32>,
    /// Layer names (for debugging)
    pub layer_names: Vec<String>,
    /// Layer types (for threshold lookup)
    pub layer_types: Vec<LayerType>,
}

impl RegressionSnapshot {
    /// Create a regression snapshot from a passing set of layer comparison results.
    ///
    /// Returns `None` if any layer failed (only successful runs become snapshots).
    pub fn from_results(
        model: &str,
        prompt: &str,
        results: &[LayerComparisonResult],
    ) -> Option<Self> {
        if results.iter().any(|r| !r.passed) {
            return None;
        }

        Some(Self {
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            model: model.to_string(),
            prompt: prompt.to_string(),
            layer_cosine_sims: results.iter().map(|r| r.cosine_similarity).collect(),
            layer_mse: results.iter().map(|r| r.mse).collect(),
            layer_names: results.iter().map(|r| r.layer_name.clone()).collect(),
            layer_types: results.iter().map(|r| r.layer_type).collect(),
        })
    }

    /// Write the snapshot to a JSON file.
    pub fn write_to_file(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a snapshot from a JSON file.
    pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Validate current results against this golden snapshot.
    ///
    /// Returns a list of layer indices where the current cosine similarity
    /// regressed by more than `tolerance` compared to the snapshot.
    pub fn check_regression(
        &self,
        current_results: &[LayerComparisonResult],
        tolerance: f32,
    ) -> Vec<RegressionViolation> {
        let mut violations = Vec::new();

        for (i, (golden, cr)) in
            self.layer_cosine_sims.iter().zip(current_results.iter()).enumerate()
        {
            let current = cr.cosine_similarity;
            let delta = golden - current;

            if delta > tolerance {
                violations.push(RegressionViolation {
                    layer_index: i,
                    layer_name: cr.layer_name.clone(),
                    golden_cosine_sim: *golden,
                    current_cosine_sim: current,
                    delta,
                });
            }
        }

        violations
    }
}

/// A single regression violation at a specific layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionViolation {
    /// Layer index where regression was detected
    pub layer_index: usize,
    /// Layer name
    pub layer_name: String,
    /// Cosine similarity from the golden snapshot
    pub golden_cosine_sim: f32,
    /// Cosine similarity from the current run
    pub current_cosine_sim: f32,
    /// Delta (golden − current); positive means regression
    pub delta: f32,
}

// ---------------------------------------------------------------------------
// Helper math functions (internal)
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    sum / a.len() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LayerType inference ----

    #[test]
    fn test_layer_type_from_name_attention() {
        assert_eq!(LayerType::from_layer_name("layers.0.self_attn"), LayerType::Attention);
        assert_eq!(LayerType::from_layer_name("blk.3.attn_q"), LayerType::Attention);
        assert_eq!(LayerType::from_layer_name("attention_output"), LayerType::Attention);
    }

    #[test]
    fn test_layer_type_from_name_ffn() {
        assert_eq!(LayerType::from_layer_name("layers.0.ffn_gate"), LayerType::Ffn);
        assert_eq!(LayerType::from_layer_name("blk.5.mlp_up"), LayerType::Ffn);
        assert_eq!(LayerType::from_layer_name("feed_forward.w1"), LayerType::Ffn);
    }

    #[test]
    fn test_layer_type_from_name_norm() {
        assert_eq!(LayerType::from_layer_name("layers.0.attn_norm"), LayerType::Norm);
        assert_eq!(LayerType::from_layer_name("blk.0.ln_1"), LayerType::Norm);
        assert_eq!(LayerType::from_layer_name("rmsnorm_final"), LayerType::Norm);
    }

    #[test]
    fn test_layer_type_from_name_embedding() {
        assert_eq!(LayerType::from_layer_name("tok_embd.weight"), LayerType::Embedding);
        assert_eq!(LayerType::from_layer_name("lm_head.weight"), LayerType::Embedding);
        assert_eq!(LayerType::from_layer_name("embed_tokens"), LayerType::Embedding);
        assert_eq!(LayerType::from_layer_name("output.weight"), LayerType::Embedding);
    }

    #[test]
    fn test_layer_type_display() {
        assert_eq!(format!("{}", LayerType::Attention), "attention");
        assert_eq!(format!("{}", LayerType::Ffn), "ffn");
        assert_eq!(format!("{}", LayerType::Norm), "norm");
        assert_eq!(format!("{}", LayerType::Embedding), "embedding");
    }

    // ---- Thresholds ----

    #[test]
    fn test_default_thresholds() {
        let th = LayerThresholds::default();
        assert!(th.get(LayerType::Norm) > th.get(LayerType::Attention));
        assert!(th.get(LayerType::Attention) > th.get(LayerType::Ffn));
        assert_eq!(th.get(LayerType::Norm), 0.9999);
        assert_eq!(th.get(LayerType::Ffn), 0.998);
    }

    // ---- Helper math ----

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_zero() {
        let a = vec![1.0, 2.0];
        assert!(l2_distance(&a, &a) < 1e-6);
    }

    #[test]
    fn test_l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(mean_squared_error(&a, &a) < 1e-6);
    }

    #[test]
    fn test_max_abs_diff_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.5, 2.9];
        assert!((max_abs_diff(&a, &b) - 0.5).abs() < 1e-6);
    }

    // ---- Per-layer activation comparison ----

    fn make_snapshot(name: &str, idx: usize, values: Vec<f32>) -> ActivationSnapshot {
        ActivationSnapshot {
            layer_name: name.to_string(),
            layer_type: LayerType::from_layer_name(name),
            layer_index: idx,
            values,
        }
    }

    #[test]
    fn test_compare_layer_activations_identical() {
        let snaps = vec![
            make_snapshot("layers.0.attn_norm", 0, vec![1.0, 2.0, 3.0]),
            make_snapshot("layers.0.self_attn", 1, vec![4.0, 5.0, 6.0]),
            make_snapshot("layers.0.ffn_gate", 2, vec![7.0, 8.0, 9.0]),
        ];
        let th = LayerThresholds::default();
        let results = compare_layer_activations(&snaps, &snaps, &th);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.passed);
            assert!((r.cosine_similarity - 1.0).abs() < 1e-6);
            assert!(r.l2_distance < 1e-6);
        }
    }

    #[test]
    fn test_compare_layer_activations_divergent() {
        let rs = vec![make_snapshot("layers.0.ffn_gate", 0, vec![1.0, 0.0, 0.0])];
        let cpp = vec![make_snapshot("layers.0.ffn_gate", 0, vec![0.0, 1.0, 0.0])];

        let th = LayerThresholds::default();
        let results = compare_layer_activations(&rs, &cpp, &th);

        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
        assert!(results[0].cosine_similarity.abs() < 1e-6); // orthogonal
    }

    // ---- Divergence bisection ----

    #[test]
    fn test_bisect_all_pass() {
        let snaps: Vec<ActivationSnapshot> = (0..8)
            .map(|i| make_snapshot(&format!("layers.{}.ffn_gate", i), i, vec![1.0, 2.0, 3.0]))
            .collect();
        let th = LayerThresholds::default();
        let result = bisect_divergence(&snaps, &snaps, &th);

        assert!(result.first_divergent_layer.is_none());
        assert_eq!(result.total_layers, 8);
        // Binary search should use at most ceil(log2(8)) + 1 = 4 probes
        assert!(result.probes <= 4);
    }

    #[test]
    fn test_bisect_diverge_at_layer_4() {
        let n = 8;
        let rs: Vec<ActivationSnapshot> = (0..n)
            .map(|i| make_snapshot(&format!("layers.{}.ffn_gate", i), i, vec![1.0, 2.0, 3.0]))
            .collect();
        let mut cpp = rs.clone();
        // Make layer 4+ diverge
        for i in 4..n {
            cpp[i].values = vec![0.0, 0.0, 1.0];
        }

        let th = LayerThresholds::default();
        let result = bisect_divergence(&rs, &cpp, &th);

        assert_eq!(result.first_divergent_layer, Some(4));
        assert!(result.divergent_detail.is_some());
        // Should be logarithmic probes
        assert!(result.probes <= 5);
    }

    #[test]
    fn test_bisect_diverge_at_first_layer() {
        let rs = vec![make_snapshot("layers.0.ffn_gate", 0, vec![1.0, 0.0, 0.0])];
        let cpp = vec![make_snapshot("layers.0.ffn_gate", 0, vec![0.0, 1.0, 0.0])];

        let th = LayerThresholds::default();
        let result = bisect_divergence(&rs, &cpp, &th);

        assert_eq!(result.first_divergent_layer, Some(0));
    }

    #[test]
    fn test_bisect_empty() {
        let th = LayerThresholds::default();
        let result = bisect_divergence(&[], &[], &th);
        assert!(result.first_divergent_layer.is_none());
        assert_eq!(result.probes, 0);
    }

    // ---- Batch comparison ----

    #[test]
    fn test_batch_compare_all_pass() {
        let prompts = vec![
            PromptEntry { label: "math".to_string(), prompt: "2+2".to_string() },
            PromptEntry { label: "greeting".to_string(), prompt: "Hello".to_string() },
        ];

        let capture = |_prompt: &str| -> anyhow::Result<Vec<ActivationSnapshot>> {
            Ok(vec![make_snapshot("layers.0.ffn_gate", 0, vec![1.0, 2.0, 3.0])])
        };

        let th = LayerThresholds::default();
        let result = batch_compare(&prompts, &capture, &capture, &th).unwrap();

        assert_eq!(result.total_count, 2);
        assert_eq!(result.passed_count, 2);
        assert!((result.pass_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_compare_partial_fail() {
        let prompts = vec![
            PromptEntry { label: "good".to_string(), prompt: "ok".to_string() },
            PromptEntry { label: "bad".to_string(), prompt: "fail".to_string() },
        ];

        let rs_capture = |_prompt: &str| -> anyhow::Result<Vec<ActivationSnapshot>> {
            Ok(vec![make_snapshot("layers.0.ffn_gate", 0, vec![1.0, 0.0, 0.0])])
        };
        let cpp_capture = |prompt: &str| -> anyhow::Result<Vec<ActivationSnapshot>> {
            if prompt == "fail" {
                // Orthogonal = divergent
                Ok(vec![make_snapshot("layers.0.ffn_gate", 0, vec![0.0, 1.0, 0.0])])
            } else {
                Ok(vec![make_snapshot("layers.0.ffn_gate", 0, vec![1.0, 0.0, 0.0])])
            }
        };

        let th = LayerThresholds::default();
        let result = batch_compare(&prompts, &rs_capture, &cpp_capture, &th).unwrap();

        assert_eq!(result.total_count, 2);
        assert_eq!(result.passed_count, 1);
        assert!((result.pass_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_batch_compare_empty() {
        let th = LayerThresholds::default();
        let capture = |_: &str| -> anyhow::Result<Vec<ActivationSnapshot>> { Ok(vec![]) };
        let result = batch_compare(&[], &capture, &capture, &th).unwrap();
        assert_eq!(result.total_count, 0);
        assert!((result.pass_rate - 1.0).abs() < 1e-6);
    }

    // ---- Markdown report ----

    #[test]
    fn test_generate_diff_report_all_pass() {
        let results = vec![
            LayerComparisonResult {
                layer_name: "norm_0".to_string(),
                layer_type: LayerType::Norm,
                layer_index: 0,
                cosine_similarity: 1.0,
                l2_distance: 0.0,
                max_abs_diff: 0.0,
                mse: 0.0,
                passed: true,
            },
            LayerComparisonResult {
                layer_name: "attn_0".to_string(),
                layer_type: LayerType::Attention,
                layer_index: 1,
                cosine_similarity: 0.9995,
                l2_distance: 0.01,
                max_abs_diff: 0.001,
                mse: 1e-5,
                passed: true,
            },
        ];

        let th = LayerThresholds::default();
        let report = generate_diff_report("test-model.gguf", &results, &th);

        assert!(report.contains("# Cross-Validation Diff Report"));
        assert!(report.contains("test-model.gguf"));
        assert!(report.contains("✅ PASS"));
        assert!(report.contains("2/2 layers passed"));
        assert!(report.contains("norm_0"));
        assert!(report.contains("attn_0"));
        assert!(!report.contains("## First Divergence"));
    }

    #[test]
    fn test_generate_diff_report_with_failure() {
        let results = vec![
            LayerComparisonResult {
                layer_name: "norm_0".to_string(),
                layer_type: LayerType::Norm,
                layer_index: 0,
                cosine_similarity: 1.0,
                l2_distance: 0.0,
                max_abs_diff: 0.0,
                mse: 0.0,
                passed: true,
            },
            LayerComparisonResult {
                layer_name: "attn_0".to_string(),
                layer_type: LayerType::Attention,
                layer_index: 1,
                cosine_similarity: 0.5,
                l2_distance: 2.0,
                max_abs_diff: 1.0,
                mse: 0.5,
                passed: false,
            },
        ];

        let th = LayerThresholds::default();
        let report = generate_diff_report("model.gguf", &results, &th);

        assert!(report.contains("❌ FAIL"));
        assert!(report.contains("1/2 layers passed"));
        assert!(report.contains("## First Divergence"));
        assert!(report.contains("attn_0"));
    }

    #[test]
    fn test_generate_batch_report() {
        let batch = BatchComparisonResult {
            prompt_results: vec![
                PromptComparisonResult {
                    label: "math".to_string(),
                    layer_results: vec![LayerComparisonResult {
                        layer_name: "ffn_0".to_string(),
                        layer_type: LayerType::Ffn,
                        layer_index: 0,
                        cosine_similarity: 1.0,
                        l2_distance: 0.0,
                        max_abs_diff: 0.0,
                        mse: 0.0,
                        passed: true,
                    }],
                    first_divergent_layer: None,
                    all_passed: true,
                },
                PromptComparisonResult {
                    label: "chat".to_string(),
                    layer_results: vec![LayerComparisonResult {
                        layer_name: "ffn_0".to_string(),
                        layer_type: LayerType::Ffn,
                        layer_index: 0,
                        cosine_similarity: 0.5,
                        l2_distance: 2.0,
                        max_abs_diff: 1.0,
                        mse: 0.5,
                        passed: false,
                    }],
                    first_divergent_layer: Some(0),
                    all_passed: false,
                },
            ],
            passed_count: 1,
            total_count: 2,
            pass_rate: 0.5,
        };

        let th = LayerThresholds::default();
        let report = generate_batch_report("model.gguf", &batch, &th);

        assert!(report.contains("# Batch Cross-Validation Report"));
        assert!(report.contains("1/2 prompts passed"));
        assert!(report.contains("50.0%"));
        assert!(report.contains("math"));
        assert!(report.contains("chat"));
        // Should include detailed report for failing prompt
        assert!(report.contains("chat (FAILED)"));
    }

    // ---- Regression snapshots ----

    #[test]
    fn test_regression_snapshot_from_passing_results() {
        let results = vec![
            LayerComparisonResult {
                layer_name: "norm_0".to_string(),
                layer_type: LayerType::Norm,
                layer_index: 0,
                cosine_similarity: 0.9999,
                l2_distance: 0.001,
                max_abs_diff: 0.0005,
                mse: 1e-7,
                passed: true,
            },
            LayerComparisonResult {
                layer_name: "ffn_0".to_string(),
                layer_type: LayerType::Ffn,
                layer_index: 1,
                cosine_similarity: 0.999,
                l2_distance: 0.01,
                max_abs_diff: 0.005,
                mse: 1e-5,
                passed: true,
            },
        ];

        let snap = RegressionSnapshot::from_results("model.gguf", "test prompt", &results);
        assert!(snap.is_some());

        let snap = snap.unwrap();
        assert_eq!(snap.version, 1);
        assert_eq!(snap.model, "model.gguf");
        assert_eq!(snap.prompt, "test prompt");
        assert_eq!(snap.layer_cosine_sims.len(), 2);
        assert_eq!(snap.layer_names, vec!["norm_0", "ffn_0"]);
    }

    #[test]
    fn test_regression_snapshot_rejected_on_failure() {
        let results = vec![LayerComparisonResult {
            layer_name: "ffn_0".to_string(),
            layer_type: LayerType::Ffn,
            layer_index: 0,
            cosine_similarity: 0.5,
            l2_distance: 2.0,
            max_abs_diff: 1.0,
            mse: 0.5,
            passed: false,
        }];

        let snap = RegressionSnapshot::from_results("model.gguf", "test", &results);
        assert!(snap.is_none());
    }

    #[test]
    fn test_regression_check_no_violation() {
        let snap = RegressionSnapshot {
            version: 1,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            model: "model.gguf".to_string(),
            prompt: "test".to_string(),
            layer_cosine_sims: vec![0.999, 0.998],
            layer_mse: vec![1e-5, 1e-4],
            layer_names: vec!["norm_0".to_string(), "ffn_0".to_string()],
            layer_types: vec![LayerType::Norm, LayerType::Ffn],
        };

        let current = vec![
            LayerComparisonResult {
                layer_name: "norm_0".to_string(),
                layer_type: LayerType::Norm,
                layer_index: 0,
                cosine_similarity: 0.9995,
                l2_distance: 0.001,
                max_abs_diff: 0.0005,
                mse: 1e-5,
                passed: true,
            },
            LayerComparisonResult {
                layer_name: "ffn_0".to_string(),
                layer_type: LayerType::Ffn,
                layer_index: 1,
                cosine_similarity: 0.999,
                l2_distance: 0.01,
                max_abs_diff: 0.005,
                mse: 1e-4,
                passed: true,
            },
        ];

        let violations = snap.check_regression(&current, 0.01);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_regression_check_with_violation() {
        let snap = RegressionSnapshot {
            version: 1,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            model: "model.gguf".to_string(),
            prompt: "test".to_string(),
            layer_cosine_sims: vec![0.999],
            layer_mse: vec![1e-5],
            layer_names: vec!["ffn_0".to_string()],
            layer_types: vec![LayerType::Ffn],
        };

        let current = vec![LayerComparisonResult {
            layer_name: "ffn_0".to_string(),
            layer_type: LayerType::Ffn,
            layer_index: 0,
            cosine_similarity: 0.95, // Significant regression
            l2_distance: 0.5,
            max_abs_diff: 0.1,
            mse: 0.01,
            passed: true,
        }];

        let violations = snap.check_regression(&current, 0.01);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].layer_index, 0);
        assert!((violations[0].delta - 0.049).abs() < 0.001);
    }

    #[test]
    fn test_regression_snapshot_roundtrip() {
        let snap = RegressionSnapshot {
            version: 1,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            model: "model.gguf".to_string(),
            prompt: "test".to_string(),
            layer_cosine_sims: vec![0.999, 0.998],
            layer_mse: vec![1e-5, 1e-4],
            layer_names: vec!["norm_0".to_string(), "ffn_0".to_string()],
            layer_types: vec![LayerType::Norm, LayerType::Ffn],
        };

        let json = serde_json::to_string_pretty(&snap).unwrap();
        let loaded: RegressionSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, snap.version);
        assert_eq!(loaded.model, snap.model);
        assert_eq!(loaded.layer_cosine_sims, snap.layer_cosine_sims);
        assert_eq!(loaded.layer_types, snap.layer_types);
    }

    #[test]
    fn test_regression_snapshot_file_io() {
        let snap = RegressionSnapshot {
            version: 1,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            model: "model.gguf".to_string(),
            prompt: "test".to_string(),
            layer_cosine_sims: vec![0.999],
            layer_mse: vec![1e-5],
            layer_names: vec!["ffn_0".to_string()],
            layer_types: vec![LayerType::Ffn],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.json");

        snap.write_to_file(&path).unwrap();
        let loaded = RegressionSnapshot::load_from_file(&path).unwrap();

        assert_eq!(loaded.model, "model.gguf");
        assert_eq!(loaded.layer_cosine_sims, vec![0.999]);
    }
}
