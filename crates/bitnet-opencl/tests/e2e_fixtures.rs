//! Shared test fixtures for `bitnet-opencl` end-to-end tests.
//!
//! Provides [`MockModel`], [`MockTokenizer`], [`TestPipeline`], and
//! assertion helpers so every e2e test starts from a deterministic,
//! hardware-independent baseline.

#![allow(dead_code)]

use bitnet_engine_core::{GenerationConfig, SessionConfig, StreamEvent};
use bitnet_generation::{StopCriteria, StopReason};
use bitnet_opencl::{OpenClPipeline, OpenClPipelineConfig};

// ---------------------------------------------------------------------------
// XorShift PRNG (mirrors the one inside the crate for weight generation)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG for deterministic test data.
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFF_FFFF) as f32 / u32::MAX as f32
    }
}

// ---------------------------------------------------------------------------
// MockModel
// ---------------------------------------------------------------------------

/// A mock model with deterministic weights generated via xorshift PRNG.
pub struct MockModel {
    pub name: String,
    pub num_parameters: u64,
    pub num_layers: u32,
    pub vocab_size: u32,
    /// Flattened weight data (one f32 per parameter, capped to 4096 for tests).
    pub weights: Vec<f32>,
}

impl MockModel {
    /// Create a mock model with deterministic weights.
    pub fn new(name: &str, seed: u64) -> Self {
        let num_parameters: u64 = 1_000_000;
        let num_layers: u32 = 12;
        let vocab_size: u32 = 1000;
        let weight_count = 4096; // Cap for test speed
        let mut rng = XorShift64::new(seed);
        let weights: Vec<f32> = (0..weight_count).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        Self { name: name.to_string(), num_parameters, num_layers, vocab_size, weights }
    }
}

// ---------------------------------------------------------------------------
// MockTokenizer
// ---------------------------------------------------------------------------

/// A mock tokenizer with a fixed vocabulary of 1 000 tokens.
pub struct MockTokenizer {
    vocab: Vec<String>,
}

impl MockTokenizer {
    /// Create a tokenizer with `tok_0` … `tok_999`.
    pub fn new() -> Self {
        let vocab: Vec<String> = (0..1000).map(|i| format!("tok_{i}")).collect();
        Self { vocab }
    }

    /// Encode a string as a sequence of token IDs (round-robin mapping).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| u32::from(b) % self.vocab.len() as u32).collect()
    }

    /// Decode a token ID back to its string representation.
    pub fn decode(&self, id: u32) -> &str {
        let idx = id as usize % self.vocab.len();
        &self.vocab[idx]
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.vocab.len() as u32
    }
}

impl Default for MockTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TestPipeline — builder for quick test setup
// ---------------------------------------------------------------------------

/// Builder for constructing an [`OpenClPipeline`] with sensible test defaults.
pub struct TestPipeline {
    config: OpenClPipelineConfig,
    vocab_size: u32,
}

impl TestPipeline {
    /// Start a new builder with default configuration.
    pub fn new() -> Self {
        let config = OpenClPipelineConfig {
            session: SessionConfig {
                model_path: "mock://test-model.gguf".to_string(),
                tokenizer_path: "mock://tokenizer.json".to_string(),
                backend: "cpu".to_string(),
                max_context: 2048,
                seed: Some(42),
            },
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        Self { config, vocab_size: 1000 }
    }

    /// Override the model path.
    pub fn model_path(mut self, path: &str) -> Self {
        self.config.session.model_path = path.to_string();
        self
    }

    /// Override the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.session.seed = Some(seed);
        self
    }

    /// Override the backend identifier.
    pub fn backend(mut self, backend: &str) -> Self {
        self.config.session.backend = backend.to_string();
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.config.temperature = t;
        self
    }

    /// Set the top-k sampling parameter.
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    /// Set the top-p sampling parameter.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = p;
        self
    }

    /// Set the vocabulary size.
    pub fn vocab_size(mut self, v: u32) -> Self {
        self.vocab_size = v;
        self
    }

    /// Build the [`OpenClPipeline`].
    pub fn build(self) -> OpenClPipeline {
        OpenClPipeline::new(self.config, self.vocab_size)
    }
}

impl Default for TestPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// Extract token events from a stream event list.
pub fn token_events(events: &[StreamEvent]) -> Vec<&StreamEvent> {
    events.iter().filter(|e| matches!(e, StreamEvent::Token(_))).collect()
}

/// Extract the final Done event, panicking if absent.
pub fn done_event(events: &[StreamEvent]) -> &StreamEvent {
    events
        .iter()
        .rfind(|e| matches!(e, StreamEvent::Done { .. }))
        .expect("expected a Done event in stream")
}

/// Assert that a generation result is structurally valid:
/// - At least one token was produced
/// - Exactly one Done event at the end
/// - All intermediate events are Token variants
pub fn assert_generation_valid(events: &[StreamEvent]) {
    assert!(events.len() >= 2, "expected at least one token + Done, got {} events", events.len());

    // Last event must be Done.
    assert!(matches!(events.last(), Some(StreamEvent::Done { .. })), "last event must be Done");

    // All events except the last must be Token.
    for event in &events[..events.len() - 1] {
        assert!(
            matches!(event, StreamEvent::Token(_)),
            "non-terminal event must be Token, got {event:?}"
        );
    }
}

/// Assert that generation stopped for the expected reason.
pub fn assert_stop_reason(events: &[StreamEvent], expected: &StopReason) {
    match done_event(events) {
        StreamEvent::Done { reason, .. } => {
            assert_eq!(reason, expected, "unexpected stop reason");
        }
        _ => panic!("expected Done event"),
    }
}

/// Count the number of tokens produced (excludes the Done event).
pub fn token_count(events: &[StreamEvent]) -> usize {
    events.iter().filter(|e| matches!(e, StreamEvent::Token(_))).count()
}

/// Build a default [`GenerationConfig`] with the given max tokens.
pub fn gen_config(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_tokens,
        seed: Some(42),
        stop_criteria: StopCriteria { max_tokens, ..Default::default() },
    }
}

/// Build a [`GenerationConfig`] with a stop string.
pub fn gen_config_with_stop_string(max_tokens: usize, stop: &str) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_tokens,
        seed: Some(42),
        stop_criteria: StopCriteria {
            max_tokens,
            stop_strings: vec![stop.to_string()],
            ..Default::default()
        },
    }
}

/// Build a [`GenerationConfig`] with an EOS token id.
pub fn gen_config_with_eos(max_tokens: usize, eos_id: u32) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_tokens,
        seed: Some(42),
        stop_criteria: StopCriteria {
            max_tokens,
            eos_token_id: Some(eos_id),
            ..Default::default()
        },
    }
}
