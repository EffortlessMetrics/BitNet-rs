//! Complete text generation loop for GPU (and CPU) inference.
//!
//! [`GenerationEngine`] orchestrates tokenisation, transformer forward
//! passes, sampling, KV-cache management, and stopping criteria into a
//! single `generate` / `generate_stream` API.  The current implementation
//! is **CPU-only** (MVP); a GPU dispatch interface is stubbed out so that
//! an `OpenCL` / CUDA backend can be slotted in later.

use std::collections::VecDeque;
use std::sync::mpsc;

use bitnet_generation::{StopCriteria, StopReason, check_stop};
use serde::{Deserialize, Serialize};

use crate::generation_stats::{GenerationStats, StatsCollector};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that may occur during text generation.
#[derive(Debug, thiserror::Error)]
pub enum GenerationError {
    /// The prompt was empty and no default behaviour is defined.
    #[error("prompt must not be empty")]
    EmptyPrompt,
    /// A model / backend error during the forward pass.
    #[error("forward pass failed: {0}")]
    ForwardPass(String),
    /// The streaming channel was closed unexpectedly.
    #[error("stream channel closed")]
    StreamClosed,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Full configuration for a generation request.
///
/// Combines sampling parameters, stopping criteria, and an optional
/// random seed for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Hard cap on the number of new tokens to produce.
    pub max_tokens: usize,
    /// Softmax temperature (0 = greedy).
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// String sequences that terminate generation.
    pub stop_sequences: Vec<String>,
    /// Optional seed for deterministic sampling.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            stop_sequences: Vec::new(),
            seed: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Stopping criteria
// ---------------------------------------------------------------------------

/// Stopping criteria for the generation loop.
///
/// Wraps [`StopCriteria`] from `bitnet-generation` with convenience
/// constructors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoppingCriteria {
    /// Maximum number of tokens to generate.
    pub max_length: usize,
    /// Token IDs that immediately end generation.
    pub stop_tokens: Vec<u32>,
    /// Decoded strings that end generation when detected.
    pub stop_strings: Vec<String>,
    /// The model EOS token, if known.
    pub eos_token: Option<u32>,
}

impl StoppingCriteria {
    /// Convert to the canonical `bitnet_generation::StopCriteria`.
    #[must_use]
    pub fn to_stop_criteria(&self) -> StopCriteria {
        StopCriteria {
            stop_token_ids: self.stop_tokens.clone(),
            stop_strings: self.stop_strings.clone(),
            max_tokens: self.max_length,
            eos_token_id: self.eos_token,
        }
    }
}

// ---------------------------------------------------------------------------
// Generation result
// ---------------------------------------------------------------------------

/// Result of a completed generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Decoded output text.
    pub output_text: String,
    /// Token IDs that were generated.
    pub tokens: Vec<u32>,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Timing / throughput statistics.
    pub stats: GenerationStats,
}

// ---------------------------------------------------------------------------
// Mock model backend (MVP)
// ---------------------------------------------------------------------------

/// Trait abstracting the model forward pass so that GPU backends can be
/// swapped in without changing the generation loop.
pub trait ModelBackend {
    /// Run a forward pass for one step and return raw logits.
    ///
    /// # Errors
    /// Returns an error string on failure.
    fn forward(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, String>;

    /// Vocabulary size of the model.
    fn vocab_size(&self) -> usize;
}

/// Trivial CPU backend used for testing.  Produces a fixed cyclic
/// sequence of token IDs via deterministic logit patterns.
pub struct MockModelBackend {
    vocab_size: usize,
    step: usize,
}

impl MockModelBackend {
    /// Create a mock backend with the given vocabulary size.
    #[must_use]
    pub const fn new(vocab_size: usize) -> Self {
        Self { vocab_size, step: 0 }
    }
}

impl ModelBackend for MockModelBackend {
    fn forward(&mut self, _token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut logits = vec![0.0f32; self.vocab_size];
        // Cycle through vocab so tests can observe varying tokens.
        let hot = self.step % self.vocab_size;
        logits[hot] = 10.0;
        self.step += 1;
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ---------------------------------------------------------------------------
// Mock tokenizer (MVP)
// ---------------------------------------------------------------------------

/// Minimal tokenizer interface.
pub trait Tokenizer {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Decode a single token ID into its string fragment.
    fn decode_token(&self, id: u32) -> String;
    /// Decode a sequence of token IDs.
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&id| self.decode_token(id)).collect()
    }
}

/// Identity tokenizer: each character → its Unicode codepoint (clamped
/// to `u32`). Good enough for unit tests.
pub struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars().map(|c| c as u32).collect()
    }

    fn decode_token(&self, id: u32) -> String {
        char::from_u32(id).map_or_else(|| format!("<{id}>"), |c| c.to_string())
    }
}

// ---------------------------------------------------------------------------
// Generation stream
// ---------------------------------------------------------------------------

/// A single token event emitted during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamToken {
    /// Vocabulary index.
    pub id: u32,
    /// Decoded text fragment.
    pub text: String,
    /// Ordinal index of this token in the generated sequence.
    pub index: usize,
}

/// Receives tokens one-by-one from a background generation loop.
pub struct GenerationStream {
    rx: mpsc::Receiver<StreamToken>,
    collected: Vec<StreamToken>,
}

impl GenerationStream {
    /// Block until the next token arrives, or `None` if generation is
    /// done.
    pub fn next_token(&mut self) -> Option<StreamToken> {
        match self.rx.recv() {
            Ok(tok) => {
                self.collected.push(tok.clone());
                Some(tok)
            }
            Err(_) => None,
        }
    }

    /// All tokens received so far.
    #[must_use]
    pub fn tokens_so_far(&self) -> &[StreamToken] {
        &self.collected
    }
}

impl Iterator for GenerationStream {
    type Item = StreamToken;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

// ---------------------------------------------------------------------------
// Generation engine
// ---------------------------------------------------------------------------

/// Orchestrates the complete text generation loop.
///
/// 1. Tokenize input prompt
/// 2. **Prefill**: process prompt tokens (parallel)
/// 3. **Decode**: autoregressive loop — forward → sample → append
/// 4. Check stopping criteria after every token
/// 5. Return decoded output with stats
pub struct GenerationEngine<M: ModelBackend, T: Tokenizer> {
    model: M,
    tokenizer: T,
    config: GenerationConfig,
    stopping: StoppingCriteria,
}

impl<M: ModelBackend, T: Tokenizer> GenerationEngine<M, T> {
    /// Create a new engine.
    pub const fn new(
        model: M,
        tokenizer: T,
        config: GenerationConfig,
        stopping: StoppingCriteria,
    ) -> Self {
        Self { model, tokenizer, config, stopping }
    }

    /// Replace the generation config.
    pub fn set_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }

    /// Replace stopping criteria.
    pub fn set_stopping(&mut self, stopping: StoppingCriteria) {
        self.stopping = stopping;
    }

    /// Run full (non-streaming) generation.
    ///
    /// # Errors
    /// Returns [`GenerationError`] on empty prompts or forward-pass
    /// failures.
    pub fn generate(&mut self, prompt: &str) -> Result<GenerationResult, GenerationError> {
        if prompt.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }

        let mut collector = StatsCollector::new();
        let criteria = self.build_criteria();

        // --- prefill -------------------------------------------------------
        collector.begin_prefill();
        let prompt_ids = self.tokenizer.encode(prompt);
        // Feed prompt through model (one batch call).
        self.model.forward(&prompt_ids).map_err(GenerationError::ForwardPass)?;
        collector.end_prefill();

        // --- decode --------------------------------------------------------
        let mut generated: Vec<u32> = Vec::new();
        let mut decoded_text = String::new();
        let mut kv_context: VecDeque<u32> = prompt_ids.iter().copied().collect();

        loop {
            let logits = self
                .model
                .forward(kv_context.make_contiguous())
                .map_err(GenerationError::ForwardPass)?;

            let token_id = self.sample(&logits);
            generated.push(token_id);
            kv_context.push_back(token_id);

            let fragment = self.tokenizer.decode_token(token_id);
            decoded_text.push_str(&fragment);

            collector.record_token();

            if let Some(reason) = check_stop(&criteria, token_id, &generated, &decoded_text) {
                let stats = collector.finish();
                return Ok(GenerationResult {
                    output_text: decoded_text,
                    tokens: generated,
                    stop_reason: reason,
                    stats,
                });
            }
        }
    }

    /// Start streaming generation, returning a [`GenerationStream`].
    ///
    /// # Errors
    /// Returns [`GenerationError::EmptyPrompt`] if the prompt is empty.
    pub fn generate_stream(&mut self, prompt: &str) -> Result<GenerationStream, GenerationError> {
        if prompt.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }

        let criteria = self.build_criteria();

        // Prefill.
        let prompt_ids = self.tokenizer.encode(prompt);
        self.model.forward(&prompt_ids).map_err(GenerationError::ForwardPass)?;

        let (tx, rx) = mpsc::channel();
        let mut kv_context: VecDeque<u32> = prompt_ids.iter().copied().collect();
        let mut generated: Vec<u32> = Vec::new();
        let mut decoded_text = String::new();
        let mut index = 0usize;

        // Decode loop (synchronous — runs on the caller's thread via the
        // mpsc channel; a real GPU backend would spawn a worker).
        while let Ok(logits) = self.model.forward(kv_context.make_contiguous()) {
            let token_id = self.sample(&logits);
            generated.push(token_id);
            kv_context.push_back(token_id);

            let text = self.tokenizer.decode_token(token_id);
            decoded_text.push_str(&text);

            let tok = StreamToken { id: token_id, text, index };
            index += 1;

            // If the receiver is gone, stop silently.
            if tx.send(tok).is_err() {
                break;
            }

            if check_stop(&criteria, token_id, &generated, &decoded_text).is_some() {
                break;
            }
        }

        // Drop the sender so the stream knows generation is done.
        drop(tx);

        Ok(GenerationStream { rx, collected: Vec::new() })
    }

    // --- helpers -----------------------------------------------------------

    fn build_criteria(&self) -> StopCriteria {
        let mut c = self.stopping.to_stop_criteria();
        // Merge stop sequences from config.
        for s in &self.config.stop_sequences {
            if !c.stop_strings.contains(s) {
                c.stop_strings.push(s.clone());
            }
        }
        // Honour max_tokens from config if stopping criteria has none.
        if c.max_tokens == 0 {
            c.max_tokens = self.config.max_tokens;
        }
        c
    }

    fn sample(&self, logits: &[f32]) -> u32 {
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            // Greedy.
            return argmax(logits);
        }
        // Simple temperature + top-k sampling.
        let mut scored: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if self.config.top_k > 0 {
            scored.truncate(self.config.top_k);
        }
        // Apply temperature.
        let temp = self.config.temperature;
        let max_logit = scored.first().map_or(0.0, |s| s.1);
        let exps: Vec<f64> =
            scored.iter().map(|&(_, l)| (f64::from((l - max_logit) / temp)).exp()).collect();
        let sum: f64 = exps.iter().sum();

        // Deterministic pick based on seed (or just take the top).
        #[allow(clippy::cast_precision_loss)]
        let target = self.config.seed.map_or(0.0, |seed| {
            let hash = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (hash as f64 / u64::MAX as f64) * sum
        });

        let mut cumulative = 0.0;
        for (i, &exp) in exps.iter().enumerate() {
            cumulative += exp;
            if cumulative >= target {
                #[allow(clippy::cast_possible_truncation)]
                return scored[i].0 as u32;
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        scored.last().map_or(0, |s| s.0 as u32)
    }
}

/// Index of the maximum value.
#[allow(clippy::cast_possible_truncation)]
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(max_tokens: usize) -> GenerationEngine<MockModelBackend, MockTokenizer> {
        GenerationEngine::new(
            MockModelBackend::new(256),
            MockTokenizer,
            GenerationConfig { max_tokens, temperature: 0.0, ..Default::default() },
            StoppingCriteria { max_length: max_tokens, ..Default::default() },
        )
    }

    #[test]
    fn generate_produces_tokens() {
        let mut e = engine(5);
        let res = e.generate("hi").unwrap();
        assert!(!res.tokens.is_empty());
        assert!(!res.output_text.is_empty());
    }

    #[test]
    fn max_tokens_limit_enforced() {
        let mut e = engine(3);
        let res = e.generate("hello").unwrap();
        assert!(res.tokens.len() <= 3);
        assert_eq!(res.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn empty_prompt_returns_error() {
        let mut e = engine(10);
        let res = e.generate("");
        assert!(res.is_err());
        assert!(matches!(res.unwrap_err(), GenerationError::EmptyPrompt),);
    }

    #[test]
    fn stop_token_terminates_generation() {
        let mut e = GenerationEngine::new(
            MockModelBackend::new(256),
            MockTokenizer,
            GenerationConfig { max_tokens: 100, temperature: 0.0, ..Default::default() },
            StoppingCriteria {
                max_length: 100,
                // The mock cycles 0,1,2,...  — stop on token 2.
                stop_tokens: vec![2],
                ..Default::default()
            },
        );
        let res = e.generate("ab").unwrap();
        assert_eq!(res.stop_reason, StopReason::StopTokenId(2));
    }

    #[test]
    fn eos_token_terminates_generation() {
        let mut e = GenerationEngine::new(
            MockModelBackend::new(256),
            MockTokenizer,
            GenerationConfig { max_tokens: 100, temperature: 0.0, ..Default::default() },
            StoppingCriteria { max_length: 100, eos_token: Some(1), ..Default::default() },
        );
        let res = e.generate("x").unwrap();
        assert_eq!(res.stop_reason, StopReason::EosToken);
    }

    #[test]
    fn stop_string_terminates_generation() {
        let mut e = GenerationEngine::new(
            MockModelBackend::new(128),
            MockTokenizer,
            GenerationConfig { max_tokens: 200, temperature: 0.0, ..Default::default() },
            StoppingCriteria {
                max_length: 200,
                // Character with codepoint 10 is '\n'.
                stop_strings: vec!["\n".to_string()],
                ..Default::default()
            },
        );
        let res = e.generate("a").unwrap();
        assert!(matches!(res.stop_reason, StopReason::StopString(_)));
    }

    #[test]
    fn config_stop_sequences_merged_into_criteria() {
        let mut e = GenerationEngine::new(
            MockModelBackend::new(128),
            MockTokenizer,
            GenerationConfig {
                max_tokens: 200,
                temperature: 0.0,
                stop_sequences: vec!["\n".to_string()],
                ..Default::default()
            },
            StoppingCriteria { max_length: 200, ..Default::default() },
        );
        let res = e.generate("a").unwrap();
        assert!(matches!(res.stop_reason, StopReason::StopString(_)));
    }

    #[test]
    fn stats_are_recorded() {
        let mut e = engine(5);
        let res = e.generate("hi").unwrap();
        assert!(res.stats.total_time_ms >= 0.0);
        assert_eq!(res.stats.tokens_generated, res.tokens.len());
        assert!(res.stats.prefill_time_ms >= 0.0);
    }

    #[test]
    fn greedy_sampling_is_deterministic() {
        let mut e1 = engine(5);
        let mut e2 = engine(5);
        let r1 = e1.generate("ab").unwrap();
        let r2 = e2.generate("ab").unwrap();
        assert_eq!(r1.tokens, r2.tokens);
    }

    #[test]
    fn generation_result_serializes_to_json() {
        let mut e = engine(3);
        let res = e.generate("hi").unwrap();
        let json = serde_json::to_string(&res).unwrap();
        assert!(json.contains("output_text"));
        assert!(json.contains("stop_reason"));
    }

    #[test]
    fn set_config_updates_engine() {
        let mut e = engine(10);
        e.set_config(GenerationConfig { max_tokens: 2, temperature: 0.0, ..Default::default() });
        e.set_stopping(StoppingCriteria { max_length: 2, ..Default::default() });
        let res = e.generate("hi").unwrap();
        assert!(res.tokens.len() <= 2);
    }

    #[test]
    fn argmax_returns_correct_index() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn argmax_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }
}
