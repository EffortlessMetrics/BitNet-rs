//! OpenCL stop-sequence detection for text generation termination.
//!
//! Detects conditions that should terminate autoregressive decoding:
//! end-of-sequence tokens, user-defined stop strings (as token sequences),
//! maximum token limits, infinite-loop repetition, and content policy
//! violations.
//!
//! # CPU reference
//!
//! All implementations are pure CPU reference code — no OpenCL runtime
//! required.  An Aho–Corasick automaton on token IDs provides efficient
//! multi-pattern matching for stop sequences and content filters.
//!
//! # GPU path (future)
//!
//! When the `oneapi` feature is enabled, the automaton state-transition
//! table can be uploaded to GPU memory for parallel stop-sequence
//! checking across a batch of sequences.

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A named token sequence that triggers generation stop.
#[derive(Debug, Clone)]
pub struct StopSequence {
    /// Human-readable name for this stop sequence.
    pub name: String,
    /// Token IDs that form the stop pattern.
    pub tokens: Vec<u32>,
}

impl StopSequence {
    /// Create a new stop sequence.
    pub fn new(name: impl Into<String>, tokens: Vec<u32>) -> Self {
        Self { name: name.into(), tokens }
    }
}

impl fmt::Display for StopSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StopSequence({:?}, {} tokens)", self.name, self.tokens.len())
    }
}

/// Reason why generation was stopped.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// End-of-sequence token encountered.
    Eos,
    /// A user-defined stop sequence was matched.
    StopSequence(String),
    /// Maximum token count reached.
    MaxTokens,
    /// Infinite repetition loop detected.
    RepeatLoop {
        /// Length of the repeating unit.
        period: usize,
        /// Number of consecutive repetitions observed.
        count: usize,
    },
    /// Content filter blocked a pattern.
    ContentBlock(String),
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eos => write!(f, "EOS"),
            Self::StopSequence(name) => {
                write!(f, "StopSequence({name})")
            }
            Self::MaxTokens => write!(f, "MaxTokens"),
            Self::RepeatLoop { period, count } => {
                write!(f, "RepeatLoop(period={period}, count={count})")
            }
            Self::ContentBlock(name) => {
                write!(f, "ContentBlock({name})")
            }
        }
    }
}

/// Statistics about detection operations.
#[derive(Debug, Clone, Default)]
pub struct DetectionStats {
    /// Total tokens checked.
    pub tokens_checked: u64,
    /// Number of times each reason triggered (keyed by category).
    pub triggers: HashMap<String, u64>,
}

impl DetectionStats {
    /// Record a trigger for the given reason.
    pub fn record(&mut self, reason: &StopReason) {
        let key = match reason {
            StopReason::Eos => "eos".to_string(),
            StopReason::StopSequence(n) => format!("stop:{n}"),
            StopReason::MaxTokens => "max_tokens".to_string(),
            StopReason::RepeatLoop { .. } => "repeat_loop".to_string(),
            StopReason::ContentBlock(n) => format!("content:{n}"),
        };
        *self.triggers.entry(key).or_insert(0) += 1;
    }
}

impl fmt::Display for DetectionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DetectionStats(checked={}, triggers={})",
            self.tokens_checked,
            self.triggers.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Aho–Corasick automaton (internal)
// ---------------------------------------------------------------------------

/// A node in the Aho–Corasick trie.
#[derive(Debug, Clone)]
struct AcNode {
    children: HashMap<u32, usize>,
    failure: usize,
    /// Pattern indices that complete at this node.
    output: Vec<usize>,
}

/// Aho–Corasick automaton for multi-pattern token-ID matching.
#[derive(Debug, Clone)]
struct AhoCorasick {
    nodes: Vec<AcNode>,
    current: usize,
}

impl AhoCorasick {
    /// Build an automaton from token patterns. Empty patterns are skipped.
    fn new(patterns: &[Vec<u32>]) -> Self {
        let mut nodes = vec![AcNode { children: HashMap::new(), failure: 0, output: Vec::new() }];

        // Phase 1: build trie
        for (pi, pattern) in patterns.iter().enumerate() {
            if pattern.is_empty() {
                continue;
            }
            let mut state = 0;
            for &token in pattern {
                let next = nodes[state].children.get(&token).copied();
                state = match next {
                    Some(s) => s,
                    None => {
                        let s = nodes.len();
                        nodes.push(AcNode {
                            children: HashMap::new(),
                            failure: 0,
                            output: Vec::new(),
                        });
                        nodes[state].children.insert(token, s);
                        s
                    }
                };
            }
            nodes[state].output.push(pi);
        }

        // Phase 2: failure links via BFS
        let mut queue = VecDeque::new();
        let root_children: Vec<usize> = nodes[0].children.values().copied().collect();
        for child in root_children {
            nodes[child].failure = 0;
            queue.push_back(child);
        }

        while let Some(u) = queue.pop_front() {
            let transitions: Vec<(u32, usize)> =
                nodes[u].children.iter().map(|(&t, &s)| (t, s)).collect();

            for (tok, v) in transitions {
                queue.push_back(v);

                let mut f = nodes[u].failure;
                loop {
                    if let Some(&target) = nodes[f].children.get(&tok) {
                        nodes[v].failure = if target == v { 0 } else { target };
                        break;
                    }
                    if f == 0 {
                        nodes[v].failure = 0;
                        break;
                    }
                    f = nodes[f].failure;
                }

                let fail_out = nodes[nodes[v].failure].output.clone();
                nodes[v].output.extend(fail_out);
            }
        }

        Self { nodes, current: 0 }
    }

    /// Feed one token; return pattern indices that match ending here.
    fn feed(&mut self, token: u32) -> &[usize] {
        loop {
            if let Some(&next) = self.nodes[self.current].children.get(&token) {
                self.current = next;
                return &self.nodes[self.current].output;
            }
            if self.current == 0 {
                return &[];
            }
            self.current = self.nodes[self.current].failure;
        }
    }

    /// Reset to the start state.
    fn reset(&mut self) {
        self.current = 0;
    }
}

// ---------------------------------------------------------------------------
// StopCondition trait
// ---------------------------------------------------------------------------

/// Common interface for all stop-condition detectors.
pub trait StopCondition {
    /// Process a token; return `Some(reason)` when generation should stop.
    fn feed_token(&mut self, token: u32) -> Option<StopReason>;

    /// Reset internal state for a new generation.
    fn reset(&mut self);

    /// Human-readable detector name.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// StopDetector — multi-pattern Aho–Corasick stop sequences
// ---------------------------------------------------------------------------

/// Aho–Corasick stop-sequence detector.
///
/// Efficiently matches multiple stop sequences against a token stream
/// and reports the first full match.
pub struct StopDetector {
    automaton: AhoCorasick,
    sequences: Vec<StopSequence>,
}

impl StopDetector {
    /// Create a detector for the given stop sequences.
    pub fn new(sequences: Vec<StopSequence>) -> Self {
        let patterns: Vec<Vec<u32>> = sequences.iter().map(|s| s.tokens.clone()).collect();
        let automaton = AhoCorasick::new(&patterns);
        Self { automaton, sequences }
    }
}

impl StopCondition for StopDetector {
    fn feed_token(&mut self, token: u32) -> Option<StopReason> {
        let idx = self.automaton.feed(token).first().copied();
        idx.map(|i| StopReason::StopSequence(self.sequences[i].name.clone()))
    }

    fn reset(&mut self) {
        self.automaton.reset();
    }

    fn name(&self) -> &str {
        "StopDetector"
    }
}

// ---------------------------------------------------------------------------
// EosDetector
// ---------------------------------------------------------------------------

/// Detects the end-of-sequence token.
pub struct EosDetector {
    eos_token_id: u32,
}

impl EosDetector {
    /// Create a detector for the given EOS token ID.
    pub fn new(eos_token_id: u32) -> Self {
        Self { eos_token_id }
    }
}

impl StopCondition for EosDetector {
    fn feed_token(&mut self, token: u32) -> Option<StopReason> {
        if token == self.eos_token_id { Some(StopReason::Eos) } else { None }
    }

    fn reset(&mut self) {
        // Stateless — nothing to reset.
    }

    fn name(&self) -> &str {
        "EosDetector"
    }
}

// ---------------------------------------------------------------------------
// MaxTokensGuard
// ---------------------------------------------------------------------------

/// Stops generation after a fixed number of tokens.
///
/// The guard triggers when the cumulative count of fed tokens reaches
/// `max_tokens`.  A `max_tokens` of 0 triggers on the very first token.
pub struct MaxTokensGuard {
    max_tokens: usize,
    count: usize,
}

impl MaxTokensGuard {
    /// Create a guard that stops after `max_tokens` tokens.
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens, count: 0 }
    }

    /// Current token count.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl StopCondition for MaxTokensGuard {
    fn feed_token(&mut self, _token: u32) -> Option<StopReason> {
        self.count += 1;
        if self.count >= self.max_tokens { Some(StopReason::MaxTokens) } else { None }
    }

    fn reset(&mut self) {
        self.count = 0;
    }

    fn name(&self) -> &str {
        "MaxTokensGuard"
    }
}

// ---------------------------------------------------------------------------
// RepeatDetector
// ---------------------------------------------------------------------------

/// Detects infinite repetition loops in the token stream.
///
/// Maintains a sliding window of recent tokens and checks whether the
/// tail consists of a short pattern repeated at least `min_repeats`
/// times.
pub struct RepeatDetector {
    window: Vec<u32>,
    window_capacity: usize,
    min_period: usize,
    min_repeats: usize,
}

impl RepeatDetector {
    /// Create a repeat detector.
    ///
    /// * `window_capacity` — maximum recent tokens to keep.
    /// * `min_period` — shortest repeating-unit length (clamped ≥ 1).
    /// * `min_repeats` — minimum consecutive repeats to trigger
    ///   (clamped ≥ 2).
    pub fn new(window_capacity: usize, min_period: usize, min_repeats: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_capacity),
            window_capacity,
            min_period: min_period.max(1),
            min_repeats: min_repeats.max(2),
        }
    }

    /// Check the current window for a repeating suffix.
    fn check_repeat(&self) -> Option<(usize, usize)> {
        let n = self.window.len();
        let max_period = n / self.min_repeats;

        for period in self.min_period..=max_period {
            let pattern = &self.window[n - period..];
            let mut repeats = 1usize;
            let mut pos = n - period;
            while pos >= period {
                pos -= period;
                if self.window[pos..pos + period] == *pattern {
                    repeats += 1;
                    if repeats >= self.min_repeats {
                        return Some((period, repeats));
                    }
                } else {
                    break;
                }
            }
        }
        None
    }
}

impl StopCondition for RepeatDetector {
    fn feed_token(&mut self, token: u32) -> Option<StopReason> {
        if self.window.len() >= self.window_capacity {
            self.window.remove(0);
        }
        self.window.push(token);

        self.check_repeat().map(|(period, count)| StopReason::RepeatLoop { period, count })
    }

    fn reset(&mut self) {
        self.window.clear();
    }

    fn name(&self) -> &str {
        "RepeatDetector"
    }
}

// ---------------------------------------------------------------------------
// ContentFilter
// ---------------------------------------------------------------------------

/// Blocks specific token patterns from appearing in the output.
///
/// Uses an Aho–Corasick automaton (same core as [`StopDetector`]) but
/// semantically represents content-policy violations rather than
/// user-requested stop strings.
pub struct ContentFilter {
    automaton: AhoCorasick,
    pattern_names: Vec<String>,
}

impl ContentFilter {
    /// Create a content filter.
    ///
    /// Each entry is `(name, token_pattern)`.
    pub fn new(patterns: Vec<(String, Vec<u32>)>) -> Self {
        let (names, seqs): (Vec<_>, Vec<_>) = patterns.into_iter().unzip();
        let automaton = AhoCorasick::new(&seqs);
        Self { automaton, pattern_names: names }
    }
}

impl StopCondition for ContentFilter {
    fn feed_token(&mut self, token: u32) -> Option<StopReason> {
        let idx = self.automaton.feed(token).first().copied();
        idx.map(|i| StopReason::ContentBlock(self.pattern_names[i].clone()))
    }

    fn reset(&mut self) {
        self.automaton.reset();
    }

    fn name(&self) -> &str {
        "ContentFilter"
    }
}

// ---------------------------------------------------------------------------
// DetectorChain
// ---------------------------------------------------------------------------

/// Composes multiple stop detectors with priority ordering.
///
/// Detectors are checked in insertion order on every token.  The first
/// trigger (highest priority) wins and is returned.
pub struct DetectorChain {
    detectors: Vec<Box<dyn StopCondition>>,
    stats: DetectionStats,
}

impl DetectorChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self { detectors: Vec::new(), stats: DetectionStats::default() }
    }

    /// Append a detector (lower priority than previously added ones).
    pub fn add(&mut self, detector: Box<dyn StopCondition>) {
        self.detectors.push(detector);
    }

    /// Builder-style append.
    #[must_use]
    pub fn with(mut self, detector: Box<dyn StopCondition>) -> Self {
        self.detectors.push(detector);
        self
    }

    /// Feed a token to **all** detectors; return the first trigger.
    pub fn feed_token(&mut self, token: u32) -> Option<StopReason> {
        self.stats.tokens_checked += 1;

        let mut result: Option<StopReason> = None;
        for det in &mut self.detectors {
            if let Some(reason) = det.feed_token(token)
                && result.is_none()
            {
                result = Some(reason);
            }
        }

        if let Some(ref reason) = result {
            self.stats.record(reason);
        }
        result
    }

    /// Reset all detectors and clear stats.
    pub fn reset(&mut self) {
        for det in &mut self.detectors {
            det.reset();
        }
        self.stats = DetectionStats::default();
    }

    /// Borrow detection statistics.
    pub fn stats(&self) -> &DetectionStats {
        &self.stats
    }

    /// Number of detectors in the chain.
    pub fn len(&self) -> usize {
        self.detectors.len()
    }

    /// Whether the chain contains no detectors.
    pub fn is_empty(&self) -> bool {
        self.detectors.is_empty()
    }
}

impl Default for DetectorChain {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // -- StopSequence -------------------------------------------------------

    #[test]
    fn test_stop_sequence_new() {
        let ss = StopSequence::new("eof", vec![1, 2, 3]);
        assert_eq!(ss.name, "eof");
        assert_eq!(ss.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_stop_sequence_display() {
        let ss = StopSequence::new("nl", vec![10, 13]);
        let s = format!("{ss}");
        assert!(s.contains("nl"));
        assert!(s.contains("2 tokens"));
    }

    #[test]
    fn test_stop_sequence_empty_tokens() {
        let ss = StopSequence::new("empty", vec![]);
        assert!(ss.tokens.is_empty());
        assert!(format!("{ss}").contains("0 tokens"));
    }

    // -- StopReason ----------------------------------------------------------

    #[test]
    fn test_stop_reason_display() {
        assert_eq!(format!("{}", StopReason::Eos), "EOS");
        assert_eq!(format!("{}", StopReason::MaxTokens), "MaxTokens");
        let sr = StopReason::RepeatLoop { period: 3, count: 5 };
        assert!(format!("{sr}").contains("period=3"));
    }

    #[test]
    fn test_stop_reason_eq() {
        assert_eq!(StopReason::Eos, StopReason::Eos);
        assert_ne!(StopReason::Eos, StopReason::MaxTokens);
        assert_eq!(StopReason::StopSequence("a".into()), StopReason::StopSequence("a".into()),);
        assert_ne!(StopReason::StopSequence("a".into()), StopReason::StopSequence("b".into()),);
    }

    // -- DetectionStats ------------------------------------------------------

    #[test]
    fn test_stats_default() {
        let stats = DetectionStats::default();
        assert_eq!(stats.tokens_checked, 0);
        assert!(stats.triggers.is_empty());
    }

    #[test]
    fn test_stats_record() {
        let mut stats = DetectionStats::default();
        stats.record(&StopReason::Eos);
        stats.record(&StopReason::Eos);
        stats.record(&StopReason::MaxTokens);
        assert_eq!(stats.triggers["eos"], 2);
        assert_eq!(stats.triggers["max_tokens"], 1);
    }

    #[test]
    fn test_stats_display() {
        let stats = DetectionStats { tokens_checked: 42, triggers: HashMap::new() };
        let s = format!("{stats}");
        assert!(s.contains("checked=42"));
    }

    // -- StopDetector --------------------------------------------------------

    #[test]
    fn test_single_stop_sequence() {
        let mut det = StopDetector::new(vec![StopSequence::new("end", vec![10, 20, 30])]);
        assert!(det.feed_token(10).is_none());
        assert!(det.feed_token(20).is_none());
        let r = det.feed_token(30);
        assert_eq!(r, Some(StopReason::StopSequence("end".into())));
    }

    #[test]
    fn test_multi_pattern_detection() {
        let mut det = StopDetector::new(vec![
            StopSequence::new("a", vec![1, 2]),
            StopSequence::new("b", vec![3, 4]),
        ]);
        assert!(det.feed_token(3).is_none());
        let r = det.feed_token(4);
        assert_eq!(r, Some(StopReason::StopSequence("b".into())));
    }

    #[test]
    fn test_stop_detector_no_match() {
        let mut det = StopDetector::new(vec![StopSequence::new("x", vec![1, 2, 3])]);
        for tok in [4, 5, 6, 7, 8] {
            assert!(det.feed_token(tok).is_none());
        }
    }

    #[test]
    fn test_stop_detector_partial_match_then_diverge() {
        let mut det = StopDetector::new(vec![StopSequence::new("seq", vec![1, 2, 3])]);
        // Start matching then diverge.
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(2).is_none());
        assert!(det.feed_token(9).is_none()); // diverge
        // Full match should still work after reset via automaton.
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(2).is_none());
        assert!(det.feed_token(3).is_some());
    }

    #[test]
    fn test_stop_detector_match_at_start() {
        let mut det = StopDetector::new(vec![StopSequence::new("one", vec![42])]);
        let r = det.feed_token(42);
        assert_eq!(r, Some(StopReason::StopSequence("one".into())));
    }

    #[test]
    fn test_stop_detector_overlapping_prefixes() {
        // Patterns share prefix [1, 2].
        let mut det = StopDetector::new(vec![
            StopSequence::new("short", vec![1, 2]),
            StopSequence::new("long", vec![1, 2, 3]),
        ]);
        assert!(det.feed_token(1).is_none());
        // Token 2 completes "short".
        let r = det.feed_token(2);
        assert_eq!(r, Some(StopReason::StopSequence("short".into())));
    }

    #[test]
    fn test_stop_detector_reset() {
        let mut det = StopDetector::new(vec![StopSequence::new("s", vec![1, 2])]);
        det.feed_token(1);
        det.reset();
        // After reset partial state is gone.
        assert!(det.feed_token(2).is_none());
    }

    #[test]
    fn test_stop_detector_shared_prefix_patterns() {
        // [A, B, C] and [A, B, D]
        let mut det = StopDetector::new(vec![
            StopSequence::new("abc", vec![1, 2, 3]),
            StopSequence::new("abd", vec![1, 2, 4]),
        ]);
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(2).is_none());
        let r = det.feed_token(4);
        assert_eq!(r, Some(StopReason::StopSequence("abd".into())));
    }

    #[test]
    fn test_stop_detector_repeated_token_pattern() {
        let mut det = StopDetector::new(vec![StopSequence::new("aaa", vec![7, 7, 7])]);
        assert!(det.feed_token(7).is_none());
        assert!(det.feed_token(7).is_none());
        assert!(det.feed_token(7).is_some());
    }

    #[test]
    fn test_stop_detector_empty_pattern_ignored() {
        let mut det = StopDetector::new(vec![
            StopSequence::new("empty", vec![]),
            StopSequence::new("real", vec![5]),
        ]);
        // Empty pattern never matches; real pattern works.
        let r = det.feed_token(5);
        assert_eq!(r, Some(StopReason::StopSequence("real".into())));
    }

    // -- EosDetector ---------------------------------------------------------

    #[test]
    fn test_eos_detect() {
        let mut det = EosDetector::new(2);
        assert!(det.feed_token(1).is_none());
        assert_eq!(det.feed_token(2), Some(StopReason::Eos));
    }

    #[test]
    fn test_eos_no_match() {
        let mut det = EosDetector::new(99);
        for tok in 0..50 {
            assert!(det.feed_token(tok).is_none());
        }
    }

    #[test]
    fn test_eos_at_first_token() {
        let mut det = EosDetector::new(0);
        assert_eq!(det.feed_token(0), Some(StopReason::Eos));
    }

    #[test]
    fn test_eos_reset() {
        let mut det = EosDetector::new(5);
        det.reset(); // no-op but must not panic
        assert_eq!(det.feed_token(5), Some(StopReason::Eos));
    }

    #[test]
    fn test_eos_detect_among_many() {
        let mut det = EosDetector::new(100);
        for tok in 0..100 {
            assert!(det.feed_token(tok).is_none());
        }
        assert_eq!(det.feed_token(100), Some(StopReason::Eos));
    }

    // -- MaxTokensGuard ------------------------------------------------------

    #[test]
    fn test_max_tokens_exact_limit() {
        let mut g = MaxTokensGuard::new(3);
        assert!(g.feed_token(0).is_none());
        assert!(g.feed_token(0).is_none());
        assert_eq!(g.feed_token(0), Some(StopReason::MaxTokens));
    }

    #[test]
    fn test_max_tokens_below_limit() {
        let mut g = MaxTokensGuard::new(10);
        for tok in 0..9 {
            assert!(g.feed_token(tok).is_none());
        }
    }

    #[test]
    fn test_max_tokens_zero() {
        let mut g = MaxTokensGuard::new(0);
        // max=0 ⟹ first token triggers (count=1 >= 0).
        assert_eq!(g.feed_token(0), Some(StopReason::MaxTokens));
    }

    #[test]
    fn test_max_tokens_one() {
        let mut g = MaxTokensGuard::new(1);
        assert_eq!(g.feed_token(99), Some(StopReason::MaxTokens));
    }

    #[test]
    fn test_max_tokens_reset() {
        let mut g = MaxTokensGuard::new(2);
        g.feed_token(0);
        g.feed_token(0);
        g.reset();
        assert_eq!(g.count(), 0);
        assert!(g.feed_token(0).is_none());
    }

    #[test]
    fn test_max_tokens_count() {
        let mut g = MaxTokensGuard::new(100);
        assert_eq!(g.count(), 0);
        g.feed_token(0);
        assert_eq!(g.count(), 1);
        g.feed_token(0);
        assert_eq!(g.count(), 2);
    }

    // -- RepeatDetector ------------------------------------------------------

    #[test]
    fn test_repeat_simple_period_two() {
        // [1,2, 1,2, 1,2] → period=2, count≥3
        let mut det = RepeatDetector::new(64, 1, 3);
        let tokens = [1, 2, 1, 2, 1, 2];
        let mut triggered = false;
        for &tok in &tokens {
            if det.feed_token(tok).is_some() {
                triggered = true;
            }
        }
        assert!(triggered);
    }

    #[test]
    fn test_repeat_period_three() {
        let mut det = RepeatDetector::new(64, 1, 3);
        let tokens = [10, 20, 30, 10, 20, 30, 10, 20, 30];
        let result = tokens.iter().filter_map(|&t| det.feed_token(t)).next();
        assert!(matches!(result, Some(StopReason::RepeatLoop { period: 3, .. })));
    }

    #[test]
    fn test_repeat_no_repeat() {
        let mut det = RepeatDetector::new(64, 1, 3);
        for tok in 0..20 {
            assert!(det.feed_token(tok).is_none());
        }
    }

    #[test]
    fn test_repeat_single_token() {
        let mut det = RepeatDetector::new(64, 1, 3);
        // [5,5,5] → period=1, count=3
        assert!(det.feed_token(5).is_none());
        assert!(det.feed_token(5).is_none());
        let r = det.feed_token(5);
        assert!(matches!(r, Some(StopReason::RepeatLoop { period: 1, count: 3 })));
    }

    #[test]
    fn test_repeat_min_period_respected() {
        // Period-2 pattern [1,2,1,2,…] should NOT trigger when
        // min_period=3 because period 2 < min_period.
        let mut det = RepeatDetector::new(64, 3, 3);
        let tokens = [1, 2, 1, 2, 1, 2, 1, 2];
        for &tok in &tokens {
            assert!(det.feed_token(tok).is_none());
        }
    }

    #[test]
    fn test_repeat_min_repeats_respected() {
        // min_repeats=4 means [1,2,1,2,1,2] (3 reps) won't trigger.
        let mut det = RepeatDetector::new(64, 1, 4);
        let tokens = [1, 2, 1, 2, 1, 2];
        for &tok in &tokens {
            assert!(det.feed_token(tok).is_none());
        }
        // Fourth repetition triggers.
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(2).is_some());
    }

    #[test]
    fn test_repeat_reset() {
        let mut det = RepeatDetector::new(64, 1, 3);
        det.feed_token(1);
        det.feed_token(1);
        det.reset();
        // After reset, previous tokens are gone.
        assert!(det.feed_token(1).is_none());
    }

    #[test]
    fn test_repeat_late_onset() {
        // Non-repeating prefix followed by repeating suffix.
        let mut det = RepeatDetector::new(64, 1, 3);
        for tok in [10, 20, 30, 40, 50] {
            assert!(det.feed_token(tok).is_none());
        }
        // Now start repeating.
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(1).is_some());
    }

    #[test]
    fn test_repeat_window_overflow() {
        // Small window: old tokens are evicted.
        let mut det = RepeatDetector::new(6, 1, 3);
        // Fill with noise.
        for tok in 100..110 {
            det.feed_token(tok);
        }
        // Now repeat — earlier noise is evicted.
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(1).is_some());
    }

    // -- ContentFilter -------------------------------------------------------

    #[test]
    fn test_content_filter_block_single() {
        let mut cf = ContentFilter::new(vec![("bad".into(), vec![66, 77])]);
        assert!(cf.feed_token(66).is_none());
        let r = cf.feed_token(77);
        assert_eq!(r, Some(StopReason::ContentBlock("bad".into())));
    }

    #[test]
    fn test_content_filter_block_multi() {
        let mut cf = ContentFilter::new(vec![("a".into(), vec![1, 2]), ("b".into(), vec![3, 4])]);
        assert!(cf.feed_token(3).is_none());
        let r = cf.feed_token(4);
        assert_eq!(r, Some(StopReason::ContentBlock("b".into())));
    }

    #[test]
    fn test_content_filter_no_block() {
        let mut cf = ContentFilter::new(vec![("x".into(), vec![9, 8, 7])]);
        for tok in [1, 2, 3, 4, 5] {
            assert!(cf.feed_token(tok).is_none());
        }
    }

    #[test]
    fn test_content_filter_partial_match() {
        let mut cf = ContentFilter::new(vec![("p".into(), vec![1, 2, 3])]);
        cf.feed_token(1);
        cf.feed_token(2);
        // Diverge — no block.
        assert!(cf.feed_token(9).is_none());
    }

    #[test]
    fn test_content_filter_reset() {
        let mut cf = ContentFilter::new(vec![("r".into(), vec![1, 2])]);
        cf.feed_token(1);
        cf.reset();
        // Partial state lost.
        assert!(cf.feed_token(2).is_none());
    }

    #[test]
    fn test_content_filter_first_token_block() {
        let mut cf = ContentFilter::new(vec![("single".into(), vec![42])]);
        let r = cf.feed_token(42);
        assert_eq!(r, Some(StopReason::ContentBlock("single".into())));
    }

    #[test]
    fn test_content_filter_overlapping_patterns() {
        let mut cf =
            ContentFilter::new(vec![("ab".into(), vec![1, 2]), ("abc".into(), vec![1, 2, 3])]);
        cf.feed_token(1);
        // Token 2 matches "ab" immediately.
        let r = cf.feed_token(2);
        assert_eq!(r, Some(StopReason::ContentBlock("ab".into())));
    }

    // -- DetectorChain -------------------------------------------------------

    #[test]
    fn test_chain_empty() {
        let mut chain = DetectorChain::new();
        assert!(chain.is_empty());
        assert!(chain.feed_token(0).is_none());
    }

    #[test]
    fn test_chain_single_detector() {
        let mut chain = DetectorChain::new();
        chain.add(Box::new(EosDetector::new(99)));
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.feed_token(99), Some(StopReason::Eos));
    }

    #[test]
    fn test_chain_eos_priority() {
        // EOS added first ⟹ highest priority.
        let mut chain = DetectorChain::new()
            .with(Box::new(EosDetector::new(1)))
            .with(Box::new(MaxTokensGuard::new(1)));
        // Token 1 is both EOS and triggers max_tokens.
        let r = chain.feed_token(1);
        assert_eq!(r, Some(StopReason::Eos));
    }

    #[test]
    fn test_chain_stop_seq_priority() {
        let mut chain = DetectorChain::new()
            .with(Box::new(StopDetector::new(vec![StopSequence::new("s", vec![5])])))
            .with(Box::new(MaxTokensGuard::new(1)));
        let r = chain.feed_token(5);
        assert_eq!(r, Some(StopReason::StopSequence("s".into())));
    }

    #[test]
    fn test_chain_max_tokens_fallback() {
        let mut chain = DetectorChain::new()
            .with(Box::new(EosDetector::new(999)))
            .with(Box::new(MaxTokensGuard::new(3)));
        assert!(chain.feed_token(1).is_none());
        assert!(chain.feed_token(2).is_none());
        assert_eq!(chain.feed_token(3), Some(StopReason::MaxTokens));
    }

    #[test]
    fn test_chain_stats_tracking() {
        let mut chain = DetectorChain::new().with(Box::new(EosDetector::new(9)));
        chain.feed_token(1);
        chain.feed_token(9);
        assert_eq!(chain.stats().tokens_checked, 2);
        assert_eq!(chain.stats().triggers["eos"], 1);
    }

    #[test]
    fn test_chain_reset() {
        let mut chain = DetectorChain::new().with(Box::new(MaxTokensGuard::new(2)));
        chain.feed_token(0);
        chain.feed_token(0);
        chain.reset();
        assert_eq!(chain.stats().tokens_checked, 0);
        // Guard counter is also reset.
        assert!(chain.feed_token(0).is_none());
    }

    #[test]
    fn test_chain_first_trigger_wins() {
        // Two stop detectors; first one wins.
        let mut chain = DetectorChain::new()
            .with(Box::new(StopDetector::new(vec![StopSequence::new("first", vec![1])])))
            .with(Box::new(StopDetector::new(vec![StopSequence::new("second", vec![1])])));
        let r = chain.feed_token(1);
        assert_eq!(r, Some(StopReason::StopSequence("first".into())));
    }

    #[test]
    fn test_chain_content_filter_blocks() {
        let mut chain = DetectorChain::new()
            .with(Box::new(ContentFilter::new(vec![("blocked".into(), vec![8, 9])])))
            .with(Box::new(MaxTokensGuard::new(100)));
        chain.feed_token(8);
        let r = chain.feed_token(9);
        assert_eq!(r, Some(StopReason::ContentBlock("blocked".into())));
    }

    #[test]
    fn test_chain_all_detectors_combined() {
        let mut chain = DetectorChain::new()
            .with(Box::new(EosDetector::new(0)))
            .with(Box::new(StopDetector::new(vec![StopSequence::new("stop", vec![10, 20])])))
            .with(Box::new(ContentFilter::new(vec![("bad".into(), vec![30, 40])])))
            .with(Box::new(RepeatDetector::new(64, 1, 3)))
            .with(Box::new(MaxTokensGuard::new(100)));

        // Normal tokens — no trigger.
        assert!(chain.feed_token(1).is_none());
        assert!(chain.feed_token(2).is_none());
        // EOS triggers.
        assert_eq!(chain.feed_token(0), Some(StopReason::Eos));
    }

    // -- Edge cases -----------------------------------------------------------

    #[test]
    fn test_edge_empty_token_stream() {
        let chain = DetectorChain::new().with(Box::new(EosDetector::new(0)));
        // Never feed anything.
        assert_eq!(chain.stats().tokens_checked, 0);
    }

    #[test]
    fn test_edge_max_token_id() {
        let mut det = EosDetector::new(u32::MAX);
        assert!(det.feed_token(0).is_none());
        assert_eq!(det.feed_token(u32::MAX), Some(StopReason::Eos));
    }

    #[test]
    fn test_edge_zero_token_id() {
        let mut det = StopDetector::new(vec![StopSequence::new("zero", vec![0])]);
        assert!(det.feed_token(0).is_some());
    }

    #[test]
    fn test_edge_single_token_pattern() {
        let mut det = StopDetector::new(vec![
            StopSequence::new("a", vec![1]),
            StopSequence::new("b", vec![2]),
            StopSequence::new("c", vec![3]),
        ]);
        assert!(det.feed_token(99).is_none());
        let r = det.feed_token(2);
        assert_eq!(r, Some(StopReason::StopSequence("b".into())));
    }

    #[test]
    fn test_edge_continue_after_stop() {
        let mut det = StopDetector::new(vec![StopSequence::new("s", vec![1, 2])]);
        det.feed_token(1);
        assert!(det.feed_token(2).is_some());
        // Feeding more tokens after a match should not crash.
        assert!(det.feed_token(3).is_none());
        // Can match again.
        det.feed_token(1);
        assert!(det.feed_token(2).is_some());
    }

    #[test]
    fn test_edge_very_long_sequence() {
        let mut det = StopDetector::new(vec![StopSequence::new("end", vec![99_999])]);
        for tok in 0..10_000 {
            assert!(det.feed_token(tok).is_none());
        }
        // Still works after many tokens.
        assert!(det.feed_token(99_999).is_some());
    }

    #[test]
    fn test_edge_aho_corasick_failure_links() {
        // Classic AC scenario: patterns [A,B,C] and [B,C,D].
        // After feeding A,B then failing on next token, the automaton
        // should retain the B prefix via failure link.
        let mut det = StopDetector::new(vec![
            StopSequence::new("abc", vec![1, 2, 3]),
            StopSequence::new("bcd", vec![2, 3, 4]),
        ]);
        // Feed [1, 2, 3, 4]: should match "abc" at token 3.
        assert!(det.feed_token(1).is_none());
        assert!(det.feed_token(2).is_none());
        let r = det.feed_token(3);
        assert_eq!(r, Some(StopReason::StopSequence("abc".into())));
        // After matching abc, the failure link for the abc-end state
        // should allow continuing to match bcd: token 4 completes it.
        let r2 = det.feed_token(4);
        assert_eq!(r2, Some(StopReason::StopSequence("bcd".into())));
    }

    // -- Property tests -------------------------------------------------------

    proptest! {
        #[test]
        fn prop_max_tokens_always_stops(
            max in 1usize..500,
            extra in 0usize..50,
        ) {
            let mut guard = MaxTokensGuard::new(max);
            let mut stopped = false;
            for i in 0..max + extra {
                if guard.feed_token(i as u32).is_some() {
                    stopped = true;
                    prop_assert!(i + 1 >= max);
                    break;
                }
            }
            prop_assert!(stopped);
        }

        #[test]
        fn prop_eos_always_detected(eos_id in 0u32..1000) {
            let mut det = EosDetector::new(eos_id);
            // Feed non-EOS tokens.
            for tok in 0..eos_id {
                prop_assert!(det.feed_token(tok).is_none());
            }
            prop_assert_eq!(
                det.feed_token(eos_id),
                Some(StopReason::Eos)
            );
        }

        #[test]
        fn prop_repeat_detector_catches_repeats(
            period in 1usize..5,
            extra_repeats in 0usize..3,
        ) {
            let min_repeats = 3;
            let total = min_repeats + extra_repeats;
            let mut det = RepeatDetector::new(128, 1, min_repeats);

            let pattern: Vec<u32> =
                (0..period).map(|i| i as u32 + 1).collect();
            let mut triggered = false;
            for _ in 0..total {
                for &tok in &pattern {
                    if det.feed_token(tok).is_some() {
                        triggered = true;
                    }
                }
            }
            prop_assert!(triggered);
        }

        #[test]
        fn prop_chain_respects_max_tokens(
            max in 1usize..200,
        ) {
            let mut chain = DetectorChain::new()
                .with(Box::new(MaxTokensGuard::new(max)));
            let mut count = 0u64;
            for i in 0..max + 10 {
                count += 1;
                if chain.feed_token(i as u32).is_some() {
                    break;
                }
            }
            prop_assert!(count <= max as u64 + 1);
        }

        #[test]
        fn prop_content_filter_catches_pattern(
            prefix_len in 0usize..20,
        ) {
            let blocked = vec![100, 200, 300];
            let mut cf = ContentFilter::new(vec![
                ("p".into(), blocked.clone()),
            ]);
            // Feed harmless prefix.
            for tok in 0..prefix_len as u32 {
                let _ = cf.feed_token(tok);
            }
            cf.reset();
            // Feed the blocked pattern.
            cf.feed_token(100);
            cf.feed_token(200);
            let r = cf.feed_token(300);
            prop_assert_eq!(
                r,
                Some(StopReason::ContentBlock("p".into()))
            );
        }
    }
}
