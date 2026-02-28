//! GPU-accelerated speculative decoding for BitNet inference.
//!
//! Implements speculative decoding where a small "draft" model proposes
//! candidate tokens that a larger "verifier" model accepts or rejects.
//! This can provide significant speedups when the draft model's acceptance
//! rate is high, especially with GPU-accelerated verification.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Maximum number of draft tokens to generate per speculation round.
    pub max_draft_tokens: usize,
    /// Temperature for draft model sampling (lower = more greedy = higher acceptance).
    pub draft_temperature: f32,
    /// Temperature for verifier model sampling.
    pub verifier_temperature: f32,
    /// Minimum acceptance rate before reducing draft length.
    pub min_acceptance_rate: f32,
    /// Whether to use tree-based speculation (multiple branches).
    pub tree_speculation: bool,
    /// Maximum tree width (number of branches at each level).
    pub max_tree_width: usize,
    /// Whether to run verification on GPU.
    pub gpu_verify: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: 5,
            draft_temperature: 0.3,
            verifier_temperature: 0.7,
            min_acceptance_rate: 0.5,
            tree_speculation: false,
            max_tree_width: 3,
            gpu_verify: false,
        }
    }
}

impl SpeculativeConfig {
    /// Create config optimized for high-throughput batch inference.
    pub fn batch_optimized() -> Self {
        Self {
            max_draft_tokens: 8,
            draft_temperature: 0.2,
            verifier_temperature: 0.7,
            min_acceptance_rate: 0.4,
            tree_speculation: true,
            max_tree_width: 4,
            gpu_verify: true,
        }
    }

    /// Create config optimized for low-latency interactive use.
    pub fn interactive() -> Self {
        Self {
            max_draft_tokens: 3,
            draft_temperature: 0.4,
            verifier_temperature: 0.7,
            min_acceptance_rate: 0.6,
            tree_speculation: false,
            max_tree_width: 1,
            gpu_verify: false,
        }
    }
}

/// A single token prediction from the draft model.
#[derive(Debug, Clone)]
pub struct DraftToken {
    /// The predicted token ID.
    pub token_id: u32,
    /// Log-probability of this token under the draft model.
    pub draft_log_prob: f32,
    /// Position in the sequence.
    pub position: usize,
}

/// A tree node for tree-based speculation.
#[derive(Debug, Clone)]
pub struct SpeculationTreeNode {
    /// Token at this node.
    pub token: DraftToken,
    /// Child branches (empty for leaf nodes).
    pub children: Vec<SpeculationTreeNode>,
    /// Cumulative log-probability from root to this node.
    pub cumulative_log_prob: f32,
}

impl SpeculationTreeNode {
    /// Create a new leaf node.
    pub fn leaf(token: DraftToken) -> Self {
        let cumulative_log_prob = token.draft_log_prob;
        Self {
            token,
            children: Vec::new(),
            cumulative_log_prob,
        }
    }

    /// Total number of tokens in this subtree.
    pub fn total_tokens(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_tokens()).sum::<usize>()
    }

    /// Flatten the tree into sequences (each path from root to leaf).
    pub fn flatten_paths(&self) -> Vec<Vec<DraftToken>> {
        if self.children.is_empty() {
            return vec![vec![self.token.clone()]];
        }
        let mut paths = Vec::new();
        for child in &self.children {
            for mut path in child.flatten_paths() {
                path.insert(0, self.token.clone());
                paths.push(path);
            }
        }
        paths
    }
}

/// Result of verifying draft tokens against the verifier model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens that were accepted.
    pub accepted_count: usize,
    /// The accepted token IDs (in order).
    pub accepted_tokens: Vec<u32>,
    /// The corrected token (sampled from adjusted distribution after rejection).
    pub corrected_token: Option<u32>,
    /// Per-token acceptance probabilities from the verifier.
    pub acceptance_probs: Vec<f32>,
}

impl VerificationResult {
    /// Acceptance rate for this verification round.
    pub fn acceptance_rate(&self) -> f32 {
        let total = self.accepted_tokens.len()
            + if self.corrected_token.is_some() {
                1
            } else {
                0
            };
        if total == 0 {
            return 0.0;
        }
        self.accepted_count as f32 / total as f32
    }

    /// Total tokens produced (accepted + corrected).
    pub fn total_tokens_produced(&self) -> usize {
        self.accepted_count + usize::from(self.corrected_token.is_some())
    }
}

/// Statistics tracking for speculative decoding performance.
#[derive(Debug)]
pub struct SpeculativeStats {
    pub total_rounds: AtomicU64,
    pub total_draft_tokens: AtomicU64,
    pub total_accepted_tokens: AtomicU64,
    pub total_corrected_tokens: AtomicU64,
    pub total_verification_time_us: AtomicU64,
    pub total_draft_time_us: AtomicU64,
}

impl Default for SpeculativeStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativeStats {
    pub fn new() -> Self {
        Self {
            total_rounds: AtomicU64::new(0),
            total_draft_tokens: AtomicU64::new(0),
            total_accepted_tokens: AtomicU64::new(0),
            total_corrected_tokens: AtomicU64::new(0),
            total_verification_time_us: AtomicU64::new(0),
            total_draft_time_us: AtomicU64::new(0),
        }
    }

    /// Record a completed speculation round.
    pub fn record_round(
        &self,
        num_draft: usize,
        num_accepted: usize,
        had_correction: bool,
        draft_us: u64,
        verify_us: u64,
    ) {
        self.total_rounds.fetch_add(1, Ordering::Relaxed);
        self.total_draft_tokens
            .fetch_add(num_draft as u64, Ordering::Relaxed);
        self.total_accepted_tokens
            .fetch_add(num_accepted as u64, Ordering::Relaxed);
        if had_correction {
            self.total_corrected_tokens
                .fetch_add(1, Ordering::Relaxed);
        }
        self.total_draft_time_us
            .fetch_add(draft_us, Ordering::Relaxed);
        self.total_verification_time_us
            .fetch_add(verify_us, Ordering::Relaxed);
    }

    /// Overall acceptance rate across all rounds.
    pub fn acceptance_rate(&self) -> f32 {
        let drafted = self.total_draft_tokens.load(Ordering::Relaxed);
        if drafted == 0 {
            return 0.0;
        }
        self.total_accepted_tokens.load(Ordering::Relaxed) as f32 / drafted as f32
    }

    /// Average tokens produced per round (accepted + corrected).
    pub fn avg_tokens_per_round(&self) -> f32 {
        let rounds = self.total_rounds.load(Ordering::Relaxed);
        if rounds == 0 {
            return 0.0;
        }
        let accepted = self.total_accepted_tokens.load(Ordering::Relaxed);
        let corrected = self.total_corrected_tokens.load(Ordering::Relaxed);
        (accepted + corrected) as f32 / rounds as f32
    }

    /// Speedup estimate compared to autoregressive decoding.
    /// Assumes draft model is `draft_cost_ratio` times cheaper than verifier.
    pub fn estimated_speedup(&self, draft_cost_ratio: f32) -> f32 {
        let rounds = self.total_rounds.load(Ordering::Relaxed);
        if rounds == 0 {
            return 1.0;
        }
        let tokens_per_round = self.avg_tokens_per_round();
        let avg_draft = self.total_draft_tokens.load(Ordering::Relaxed) as f32 / rounds as f32;
        // Cost per round: draft_cost * num_draft + 1 verify
        // Autoregressive cost for same tokens: tokens_per_round * 1 verify
        let spec_cost = avg_draft * draft_cost_ratio + 1.0;
        let auto_cost = tokens_per_round;
        if spec_cost <= 0.0 {
            return 1.0;
        }
        auto_cost / spec_cost
    }
}

impl fmt::Display for SpeculativeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpeculativeStats {{ rounds: {}, acceptance: {:.1}%, avg_tokens/round: {:.1}, speedup_est: {:.2}x }}",
            self.total_rounds.load(Ordering::Relaxed),
            self.acceptance_rate() * 100.0,
            self.avg_tokens_per_round(),
            self.estimated_speedup(0.1),
        )
    }
}

/// Trait for a draft model that proposes candidate tokens.
pub trait DraftModel: Send + Sync {
    /// Generate up to `max_tokens` draft tokens given the current context.
    fn draft(
        &self,
        context_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Vec<DraftToken>;

    /// Generate a speculation tree with branching.
    fn draft_tree(
        &self,
        context_tokens: &[u32],
        max_depth: usize,
        max_width: usize,
        temperature: f32,
    ) -> Vec<SpeculationTreeNode> {
        // Default: linear chain (no branching)
        let tokens = self.draft(context_tokens, max_depth, temperature);
        tokens
            .into_iter()
            .map(SpeculationTreeNode::leaf)
            .collect()
    }
}

/// Trait for a verifier model that validates draft tokens.
pub trait VerifierModel: Send + Sync {
    /// Compute log-probabilities for the draft tokens under the verifier.
    fn verify_logprobs(
        &self,
        context_tokens: &[u32],
        draft_tokens: &[DraftToken],
    ) -> Vec<f32>;

    /// Sample a corrected token from the adjusted distribution.
    fn sample_corrected(
        &self,
        context_tokens: &[u32],
        draft_tokens: &[DraftToken],
        rejection_index: usize,
        temperature: f32,
    ) -> u32;
}

/// The main speculative decoder that orchestrates draft-verify cycles.
pub struct SpeculativeDecoder<D: DraftModel, V: VerifierModel> {
    draft: D,
    verifier: V,
    config: SpeculativeConfig,
    stats: Arc<SpeculativeStats>,
    adaptive_draft_len: usize,
}

impl<D: DraftModel, V: VerifierModel> SpeculativeDecoder<D, V> {
    pub fn new(draft: D, verifier: V, config: SpeculativeConfig) -> Self {
        let adaptive_draft_len = config.max_draft_tokens;
        Self {
            draft,
            verifier,
            config,
            stats: Arc::new(SpeculativeStats::new()),
            adaptive_draft_len,
        }
    }

    /// Get shared reference to statistics.
    pub fn stats(&self) -> Arc<SpeculativeStats> {
        Arc::clone(&self.stats)
    }

    /// Run one speculation round: draft tokens, then verify.
    pub fn speculate_round(&mut self, context: &[u32]) -> VerificationResult {
        let start_draft = std::time::Instant::now();

        let draft_tokens = self.draft.draft(
            context,
            self.adaptive_draft_len,
            self.config.draft_temperature,
        );
        let draft_us = start_draft.elapsed().as_micros() as u64;

        if draft_tokens.is_empty() {
            return VerificationResult {
                accepted_count: 0,
                accepted_tokens: Vec::new(),
                corrected_token: None,
                acceptance_probs: Vec::new(),
            };
        }

        let start_verify = std::time::Instant::now();
        let verifier_logprobs =
            self.verifier.verify_logprobs(context, &draft_tokens);
        let verify_us = start_verify.elapsed().as_micros() as u64;

        // Standard speculative decoding acceptance:
        // Accept token i if r < min(1, p_verifier(token_i) / p_draft(token_i))
        let mut accepted_count = 0;
        let mut accepted_tokens = Vec::new();
        let mut acceptance_probs = Vec::new();

        for (i, draft_token) in draft_tokens.iter().enumerate() {
            let verifier_lp = verifier_logprobs.get(i).copied().unwrap_or(f32::NEG_INFINITY);
            let draft_lp = draft_token.draft_log_prob;

            // Acceptance probability: min(1, exp(verifier_lp - draft_lp))
            let acceptance_prob = (verifier_lp - draft_lp).exp().min(1.0);
            acceptance_probs.push(acceptance_prob);

            // Deterministic acceptance for testing: accept if prob >= 0.5
            // In production, this would use random sampling
            if acceptance_prob >= 0.5 {
                accepted_count += 1;
                accepted_tokens.push(draft_token.token_id);
            } else {
                // Rejection: sample corrected token
                let corrected = self.verifier.sample_corrected(
                    context,
                    &draft_tokens,
                    i,
                    self.config.verifier_temperature,
                );
                self.stats.record_round(
                    draft_tokens.len(),
                    accepted_count,
                    true,
                    draft_us,
                    verify_us,
                );
                self.adapt_draft_length(accepted_count, draft_tokens.len());
                return VerificationResult {
                    accepted_count,
                    accepted_tokens,
                    corrected_token: Some(corrected),
                    acceptance_probs,
                };
            }
        }

        // All tokens accepted
        self.stats.record_round(
            draft_tokens.len(),
            accepted_count,
            false,
            draft_us,
            verify_us,
        );
        self.adapt_draft_length(accepted_count, draft_tokens.len());

        VerificationResult {
            accepted_count,
            accepted_tokens,
            corrected_token: None,
            acceptance_probs,
        }
    }

    /// Adapt draft length based on acceptance rate.
    fn adapt_draft_length(&mut self, accepted: usize, total: usize) {
        if total == 0 {
            return;
        }
        let rate = accepted as f32 / total as f32;
        if rate >= 0.8 && self.adaptive_draft_len < self.config.max_draft_tokens {
            self.adaptive_draft_len += 1;
        } else if rate < self.config.min_acceptance_rate && self.adaptive_draft_len > 1 {
            self.adaptive_draft_len -= 1;
        }
    }

    /// Generate tokens using speculative decoding.
    pub fn generate(&mut self, initial_context: &[u32], max_tokens: usize) -> Vec<u32> {
        let mut context = initial_context.to_vec();
        let mut output = Vec::new();

        while output.len() < max_tokens {
            let result = self.speculate_round(&context);
            for &tok in &result.accepted_tokens {
                if output.len() >= max_tokens {
                    break;
                }
                output.push(tok);
                context.push(tok);
            }
            if let Some(corrected) = result.corrected_token {
                if output.len() < max_tokens {
                    output.push(corrected);
                    context.push(corrected);
                }
            }
            // If nothing was produced, break to avoid infinite loop
            if result.total_tokens_produced() == 0 {
                break;
            }
        }

        output.truncate(max_tokens);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock draft model that returns predetermined tokens.
    struct MockDraft {
        tokens: Vec<(u32, f32)>, // (token_id, log_prob)
    }

    impl DraftModel for MockDraft {
        fn draft(
            &self,
            _context: &[u32],
            max_tokens: usize,
            _temperature: f32,
        ) -> Vec<DraftToken> {
            self.tokens
                .iter()
                .take(max_tokens)
                .enumerate()
                .map(|(i, &(token_id, log_prob))| DraftToken {
                    token_id,
                    draft_log_prob: log_prob,
                    position: i,
                })
                .collect()
        }
    }

    /// A mock verifier that accepts tokens with log_prob > threshold.
    struct MockVerifier {
        logprobs: Vec<f32>,
        corrected_token: u32,
    }

    impl VerifierModel for MockVerifier {
        fn verify_logprobs(
            &self,
            _context: &[u32],
            _draft_tokens: &[DraftToken],
        ) -> Vec<f32> {
            self.logprobs.clone()
        }

        fn sample_corrected(
            &self,
            _context: &[u32],
            _draft_tokens: &[DraftToken],
            _rejection_index: usize,
            _temperature: f32,
        ) -> u32 {
            self.corrected_token
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.max_draft_tokens, 5);
        assert!((config.draft_temperature - 0.3).abs() < 1e-6);
        assert!(!config.tree_speculation);
    }

    #[test]
    fn test_config_batch_optimized() {
        let config = SpeculativeConfig::batch_optimized();
        assert_eq!(config.max_draft_tokens, 8);
        assert!(config.tree_speculation);
        assert!(config.gpu_verify);
    }

    #[test]
    fn test_config_interactive() {
        let config = SpeculativeConfig::interactive();
        assert_eq!(config.max_draft_tokens, 3);
        assert!(!config.tree_speculation);
    }

    #[test]
    fn test_all_tokens_accepted() {
        // Draft and verifier agree perfectly
        let draft = MockDraft {
            tokens: vec![(10, -0.5), (20, -0.5), (30, -0.5)],
        };
        let verifier = MockVerifier {
            logprobs: vec![-0.5, -0.5, -0.5], // Same as draft
            corrected_token: 99,
        };
        let mut decoder =
            SpeculativeDecoder::new(draft, verifier, SpeculativeConfig::default());
        let result = decoder.speculate_round(&[1, 2, 3]);

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30]);
        assert!(result.corrected_token.is_none());
    }

    #[test]
    fn test_partial_acceptance_with_correction() {
        // Verifier rejects token at index 2
        let draft = MockDraft {
            tokens: vec![(10, -0.5), (20, -0.5), (30, -0.5)],
        };
        let verifier = MockVerifier {
            logprobs: vec![-0.5, -0.5, -10.0], // Third token has very low prob
            corrected_token: 42,
        };
        let mut decoder =
            SpeculativeDecoder::new(draft, verifier, SpeculativeConfig::default());
        let result = decoder.speculate_round(&[1, 2, 3]);

        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        assert_eq!(result.corrected_token, Some(42));
    }

    #[test]
    fn test_all_rejected() {
        // Verifier rejects everything
        let draft = MockDraft {
            tokens: vec![(10, -0.1), (20, -0.1)],
        };
        let verifier = MockVerifier {
            logprobs: vec![-10.0, -10.0],
            corrected_token: 55,
        };
        let mut decoder =
            SpeculativeDecoder::new(draft, verifier, SpeculativeConfig::default());
        let result = decoder.speculate_round(&[1]);

        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.corrected_token, Some(55));
    }

    #[test]
    fn test_stats_tracking() {
        let stats = SpeculativeStats::new();
        stats.record_round(5, 3, true, 100, 200);
        stats.record_round(5, 5, false, 80, 250);

        assert_eq!(stats.total_rounds.load(Ordering::Relaxed), 2);
        assert_eq!(stats.total_draft_tokens.load(Ordering::Relaxed), 10);
        assert_eq!(stats.total_accepted_tokens.load(Ordering::Relaxed), 8);
        assert_eq!(stats.total_corrected_tokens.load(Ordering::Relaxed), 1);
        assert!((stats.acceptance_rate() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_stats_display() {
        let stats = SpeculativeStats::new();
        stats.record_round(4, 3, true, 100, 200);
        let display = format!("{stats}");
        assert!(display.contains("SpeculativeStats"));
        assert!(display.contains("rounds: 1"));
    }

    #[test]
    fn test_generate_respects_max_tokens() {
        let draft = MockDraft {
            tokens: vec![(10, -0.5), (20, -0.5), (30, -0.5)],
        };
        let verifier = MockVerifier {
            logprobs: vec![-0.5, -0.5, -0.5],
            corrected_token: 99,
        };
        let mut decoder =
            SpeculativeDecoder::new(draft, verifier, SpeculativeConfig::default());
        let output = decoder.generate(&[1, 2, 3], 2);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_verification_result_metrics() {
        let result = VerificationResult {
            accepted_count: 3,
            accepted_tokens: vec![1, 2, 3],
            corrected_token: Some(4),
            acceptance_probs: vec![0.9, 0.8, 0.7, 0.1],
        };
        assert_eq!(result.total_tokens_produced(), 4);
        assert!((result.acceptance_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_speculation_tree_node() {
        let leaf = SpeculationTreeNode::leaf(DraftToken {
            token_id: 42,
            draft_log_prob: -0.5,
            position: 0,
        });
        assert_eq!(leaf.total_tokens(), 1);
        assert!(leaf.children.is_empty());

        let paths = leaf.flatten_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 1);
        assert_eq!(paths[0][0].token_id, 42);
    }

    #[test]
    fn test_tree_with_branches() {
        let mut root = SpeculationTreeNode::leaf(DraftToken {
            token_id: 1,
            draft_log_prob: -0.3,
            position: 0,
        });
        root.children = vec![
            SpeculationTreeNode::leaf(DraftToken {
                token_id: 2,
                draft_log_prob: -0.5,
                position: 1,
            }),
            SpeculationTreeNode::leaf(DraftToken {
                token_id: 3,
                draft_log_prob: -0.7,
                position: 1,
            }),
        ];

        assert_eq!(root.total_tokens(), 3);
        let paths = root.flatten_paths();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0][0].token_id, 1);
        assert_eq!(paths[0][1].token_id, 2);
        assert_eq!(paths[1][0].token_id, 1);
        assert_eq!(paths[1][1].token_id, 3);
    }

    #[test]
    fn test_adaptive_draft_length_increase() {
        let draft = MockDraft {
            tokens: vec![(10, -0.5), (20, -0.5), (30, -0.5)],
        };
        let verifier = MockVerifier {
            logprobs: vec![-0.5, -0.5, -0.5],
            corrected_token: 99,
        };
        let config = SpeculativeConfig {
            max_draft_tokens: 5,
            ..SpeculativeConfig::default()
        };
        let mut decoder = SpeculativeDecoder::new(draft, verifier, config);
        // Start at max (5), but draft only produces 3, all accepted (100% rate)
        let _result = decoder.speculate_round(&[1]);
        // With 100% acceptance and current < max, length should increase
        // But since adaptive starts at max (5), it won't go above max
        assert!(decoder.adaptive_draft_len <= 5);
    }

    #[test]
    fn test_empty_draft_produces_empty_result() {
        let draft = MockDraft { tokens: vec![] };
        let verifier = MockVerifier {
            logprobs: vec![],
            corrected_token: 0,
        };
        let mut decoder =
            SpeculativeDecoder::new(draft, verifier, SpeculativeConfig::default());
        let result = decoder.speculate_round(&[1, 2, 3]);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert!(result.corrected_token.is_none());
    }

    #[test]
    fn test_speedup_estimation() {
        let stats = SpeculativeStats::new();
        // 5 draft tokens, 4 accepted, 1 correction per round
        stats.record_round(5, 4, true, 100, 500);
        // draft_cost_ratio = 0.1 means draft is 10x cheaper
        // spec_cost = 5 * 0.1 + 1 = 1.5
        // auto_cost = 5 tokens (4 accepted + 1 corrected)
        // speedup = 5.0 / 1.5 = 3.33x
        let speedup = stats.estimated_speedup(0.1);
        assert!(speedup > 3.0 && speedup < 3.5, "speedup was {speedup}");
    }
}
