//! Platform-agnostic types shared between native and wasm32 targets.
//!
//! These types carry no dependency on `wasm-bindgen` or `web-sys` and can be
//! tested with `cargo test` on the host.

use serde::{Deserialize, Serialize};

use crate::error::WasmError;

// ── Generation configuration ─────────────────────────────────────────

/// Configuration for text generation, usable on any platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to produce.
    pub max_new_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling.
    pub top_k: Option<usize>,
    /// Nucleus (top-p) sampling.
    pub top_p: Option<f32>,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Optional RNG seed for deterministic generation.
    pub seed: Option<u64>,
    /// Strings that halt generation when produced.
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            seed: None,
            stop_sequences: vec!["</s>".into(), "<|endoftext|>".into()],
        }
    }
}

impl GenerationConfig {
    /// Validate configuration values. Returns `Err` on invalid settings.
    pub fn validate(&self) -> Result<(), WasmError> {
        if self.max_new_tokens == 0 {
            return Err("max_new_tokens must be positive".into());
        }
        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".into());
        }
        if let Some(p) = self.top_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err("top_p must be between 0.0 and 1.0".into());
        }
        if self.repetition_penalty < 0.0 {
            return Err("repetition_penalty must be non-negative".into());
        }
        Ok(())
    }

    /// Create a greedy-decoding configuration.
    pub fn greedy(max_tokens: usize) -> Self {
        Self { max_new_tokens: max_tokens, temperature: 0.0, ..Self::default() }
    }
}

// ── Token event ──────────────────────────────────────────────────────

/// A single token emitted during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenEvent {
    /// The decoded text of this token.
    pub text: String,
    /// Zero-based position in the generated sequence.
    pub position: usize,
    /// `true` when this is the last token.
    pub is_final: bool,
}

// ── Generation stats ─────────────────────────────────────────────────

/// Summary statistics produced after a generation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GenerationStats {
    /// Number of tokens generated (excluding prompt).
    pub tokens_generated: usize,
    /// Wall-clock time for the generation in milliseconds.
    pub time_ms: f64,
    /// Throughput in tokens per second.
    pub tokens_per_second: f64,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
}

impl GenerationStats {
    /// Create stats from raw measurements.
    pub fn from_measurements(tokens_generated: usize, time_ms: f64, prompt_tokens: usize) -> Self {
        let tps = if time_ms > 0.0 { tokens_generated as f64 / (time_ms / 1000.0) } else { 0.0 };
        Self { tokens_generated, time_ms, tokens_per_second: tps, prompt_tokens }
    }
}

// ── Model metadata ───────────────────────────────────────────────────

/// Metadata about a loaded model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Format of the model file (e.g. "gguf").
    pub format: String,
    /// Size of the model data in bytes.
    pub size_bytes: usize,
    /// Quantization type, if detected.
    pub quantization: Option<String>,
    /// Number of parameters (estimated).
    pub num_parameters: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = GenerationConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn greedy_config_is_valid() {
        let cfg = GenerationConfig::greedy(16);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.max_new_tokens, 16);
    }

    #[test]
    fn zero_tokens_rejected() {
        let cfg = GenerationConfig { max_new_tokens: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn negative_temperature_rejected() {
        let cfg = GenerationConfig { temperature: -0.1, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn top_p_out_of_range_rejected() {
        let above = GenerationConfig { top_p: Some(1.1), ..Default::default() };
        assert!(above.validate().is_err());

        let below = GenerationConfig { top_p: Some(-0.1), ..Default::default() };
        assert!(below.validate().is_err());
    }

    #[test]
    fn top_p_boundaries_accepted() {
        let zero = GenerationConfig { top_p: Some(0.0), ..Default::default() };
        assert!(zero.validate().is_ok());

        let one = GenerationConfig { top_p: Some(1.0), ..Default::default() };
        assert!(one.validate().is_ok());
    }

    #[test]
    fn negative_repetition_penalty_rejected() {
        let cfg = GenerationConfig { repetition_penalty: -1.0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn token_event_serde_roundtrip() {
        let evt = TokenEvent { text: "hello".into(), position: 0, is_final: false };
        let json = serde_json::to_string(&evt).unwrap();
        let back: TokenEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(evt, back);
    }

    #[test]
    fn generation_stats_from_measurements() {
        let stats = GenerationStats::from_measurements(10, 1000.0, 5);
        assert_eq!(stats.tokens_generated, 10);
        assert!((stats.tokens_per_second - 10.0).abs() < 1e-6);
        assert_eq!(stats.prompt_tokens, 5);
    }

    #[test]
    fn generation_stats_zero_time() {
        let stats = GenerationStats::from_measurements(10, 0.0, 5);
        assert_eq!(stats.tokens_per_second, 0.0);
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = GenerationConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GenerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_new_tokens, cfg.max_new_tokens);
        assert_eq!(back.temperature, cfg.temperature);
    }

    #[test]
    fn model_metadata_default() {
        let meta = ModelMetadata::default();
        assert!(meta.format.is_empty());
        assert_eq!(meta.size_bytes, 0);
        assert!(meta.quantization.is_none());
    }
}
