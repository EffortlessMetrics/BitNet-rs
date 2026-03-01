//! # Inference Configuration Builder
//!
//! A high-level builder for composing inference configurations from presets
//! and individual parameter overrides. Wraps sampling, generation, and
//! hardware settings into a single validated bundle.
//!
//! ## Quick Start
//!
//! ```
//! use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};
//!
//! let config = InferenceConfigBuilder::new()
//!     .preset(InferencePreset::Balanced)
//!     .max_tokens(64)
//!     .temperature(0.8)
//!     .build()
//!     .expect("valid config");
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Preset enum
// ---------------------------------------------------------------------------

/// Named presets that configure all sub-configs with sensible defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferencePreset {
    /// Low temperature, greedy decoding, single-threaded. Optimised for
    /// latency and deterministic output.
    Fast,
    /// Moderate temperature and nucleus sampling. Good general-purpose
    /// default.
    Balanced,
    /// Higher temperature, broader sampling, repetition penalty. Tuned for
    /// creative / high-quality generation.
    Quality,
    /// Temperature 0, fixed seed, single thread. Bit-exact reproducible
    /// output.
    Deterministic,
    /// Minimal token budget, verbose-friendly defaults for development.
    Debug,
}

// ---------------------------------------------------------------------------
// Sub-config structs
// ---------------------------------------------------------------------------

/// Sampling parameters that control token selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for softmax (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling limit (0 = disabled).
    pub top_k: u32,
    /// Top-p nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = none).
    pub repetition_penalty: f32,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 50, top_p: 0.9, repetition_penalty: 1.0, seed: None }
    }
}

/// Parameters that control the generation loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to produce.
    pub max_tokens: u32,
    /// String-based stop sequences.
    pub stop_sequences: Vec<String>,
    /// Token IDs that trigger immediate stop.
    pub stop_token_ids: Vec<u32>,
    /// Whether to emit tokens incrementally via a stream.
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
            stream: false,
        }
    }
}

/// Hardware / runtime resource constraints.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Number of CPU threads (0 = auto-detect).
    pub num_threads: usize,
    /// Soft memory ceiling in MiB (0 = unlimited).
    pub memory_limit_mb: usize,
}

// ---------------------------------------------------------------------------
// Composite config
// ---------------------------------------------------------------------------

/// A fully-resolved inference configuration bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub sampling: SamplingConfig,
    pub generation: GenerationConfig,
    pub hardware: HardwareConfig,
}

impl InferenceConfig {
    /// Apply an [`InferencePreset`], replacing all sub-configs.
    fn apply_preset(preset: InferencePreset) -> Self {
        match preset {
            InferencePreset::Fast => Self {
                sampling: SamplingConfig {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                    seed: None,
                },
                generation: GenerationConfig { max_tokens: 64, ..Default::default() },
                hardware: HardwareConfig { num_threads: 1, memory_limit_mb: 0 },
            },
            InferencePreset::Balanced => Self {
                sampling: SamplingConfig {
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.9,
                    repetition_penalty: 1.05,
                    seed: None,
                },
                generation: GenerationConfig { max_tokens: 128, ..Default::default() },
                hardware: HardwareConfig::default(),
            },
            InferencePreset::Quality => Self {
                sampling: SamplingConfig {
                    temperature: 0.9,
                    top_k: 100,
                    top_p: 0.95,
                    repetition_penalty: 1.1,
                    seed: None,
                },
                generation: GenerationConfig { max_tokens: 256, ..Default::default() },
                hardware: HardwareConfig::default(),
            },
            InferencePreset::Deterministic => Self {
                sampling: SamplingConfig {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                    seed: Some(42),
                },
                generation: GenerationConfig { max_tokens: 128, ..Default::default() },
                hardware: HardwareConfig { num_threads: 1, memory_limit_mb: 0 },
            },
            InferencePreset::Debug => Self {
                sampling: SamplingConfig {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                    seed: Some(0),
                },
                generation: GenerationConfig { max_tokens: 8, ..Default::default() },
                hardware: HardwareConfig { num_threads: 1, memory_limit_mb: 256 },
            },
        }
    }

    /// Validate all sub-configs. Returns `Err` with a description on
    /// failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.sampling.temperature < 0.0 {
            return Err("temperature must be >= 0.0".into());
        }
        if self.sampling.top_p <= 0.0 || self.sampling.top_p > 1.0 {
            return Err("top_p must be in (0.0, 1.0]".into());
        }
        if self.sampling.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be > 0.0".into());
        }
        if self.generation.max_tokens == 0 {
            return Err("max_tokens must be > 0".into());
        }
        if self.hardware.num_threads > 1024 {
            return Err("num_threads must be <= 1024".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for [`InferenceConfig`].
///
/// Start with [`InferenceConfigBuilder::new()`], optionally apply a
/// [`InferencePreset`], override individual fields, and call
/// [`build()`](InferenceConfigBuilder::build) to obtain a validated config.
#[derive(Debug, Clone)]
pub struct InferenceConfigBuilder {
    config: InferenceConfig,
}

impl Default for InferenceConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceConfigBuilder {
    /// Create a builder starting from [`InferencePreset::Balanced`] defaults.
    pub fn new() -> Self {
        Self { config: InferenceConfig::apply_preset(InferencePreset::Balanced) }
    }

    /// Reset all parameters to the given preset.
    #[must_use]
    pub fn preset(mut self, preset: InferencePreset) -> Self {
        self.config = InferenceConfig::apply_preset(preset);
        self
    }

    // -- Sampling setters ---------------------------------------------------

    #[must_use]
    pub fn temperature(mut self, value: f32) -> Self {
        self.config.sampling.temperature = value;
        self
    }

    #[must_use]
    pub fn top_k(mut self, value: u32) -> Self {
        self.config.sampling.top_k = value;
        self
    }

    #[must_use]
    pub fn top_p(mut self, value: f32) -> Self {
        self.config.sampling.top_p = value;
        self
    }

    #[must_use]
    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.config.sampling.repetition_penalty = value;
        self
    }

    #[must_use]
    pub fn seed(mut self, value: u64) -> Self {
        self.config.sampling.seed = Some(value);
        self
    }

    // -- Generation setters -------------------------------------------------

    #[must_use]
    pub fn max_tokens(mut self, value: u32) -> Self {
        self.config.generation.max_tokens = value;
        self
    }

    #[must_use]
    pub fn stop_sequence(mut self, seq: impl Into<String>) -> Self {
        self.config.generation.stop_sequences.push(seq.into());
        self
    }

    #[must_use]
    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.config.generation.stop_sequences = seqs;
        self
    }

    #[must_use]
    pub fn stop_token_id(mut self, id: u32) -> Self {
        self.config.generation.stop_token_ids.push(id);
        self
    }

    #[must_use]
    pub fn stop_token_ids(mut self, ids: Vec<u32>) -> Self {
        self.config.generation.stop_token_ids = ids;
        self
    }

    #[must_use]
    pub fn stream(mut self, enabled: bool) -> Self {
        self.config.generation.stream = enabled;
        self
    }

    // -- Hardware setters ---------------------------------------------------

    #[must_use]
    pub fn num_threads(mut self, value: usize) -> Self {
        self.config.hardware.num_threads = value;
        self
    }

    #[must_use]
    pub fn memory_limit_mb(mut self, value: usize) -> Self {
        self.config.hardware.memory_limit_mb = value;
        self
    }

    // -- Terminal -----------------------------------------------------------

    /// Validate and return the final [`InferenceConfig`].
    pub fn build(self) -> Result<InferenceConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Builder construction & defaults ------------------------------------

    #[test]
    fn test_builder_default_is_balanced() {
        let config = InferenceConfigBuilder::new().build().unwrap();
        assert_eq!(config.sampling.temperature, 0.7);
        assert_eq!(config.sampling.top_k, 50);
        assert_eq!(config.sampling.top_p, 0.9);
        assert_eq!(config.sampling.repetition_penalty, 1.05);
        assert_eq!(config.generation.max_tokens, 128);
        assert!(!config.generation.stream);
    }

    #[test]
    fn test_builder_default_impl() {
        let a = InferenceConfigBuilder::new().build().unwrap();
        let b = InferenceConfigBuilder::default().build().unwrap();
        assert_eq!(a, b);
    }

    // -- Presets ------------------------------------------------------------

    #[test]
    fn test_preset_fast() {
        let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();
        assert_eq!(cfg.sampling.temperature, 0.0);
        assert_eq!(cfg.sampling.top_k, 1);
        assert_eq!(cfg.generation.max_tokens, 64);
        assert_eq!(cfg.hardware.num_threads, 1);
    }

    #[test]
    fn test_preset_balanced() {
        let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Balanced).build().unwrap();
        assert_eq!(cfg.sampling.temperature, 0.7);
        assert_eq!(cfg.sampling.repetition_penalty, 1.05);
    }

    #[test]
    fn test_preset_quality() {
        let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Quality).build().unwrap();
        assert_eq!(cfg.sampling.temperature, 0.9);
        assert_eq!(cfg.sampling.top_k, 100);
        assert_eq!(cfg.sampling.top_p, 0.95);
        assert_eq!(cfg.sampling.repetition_penalty, 1.1);
        assert_eq!(cfg.generation.max_tokens, 256);
    }

    #[test]
    fn test_preset_deterministic() {
        let cfg =
            InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();
        assert_eq!(cfg.sampling.temperature, 0.0);
        assert_eq!(cfg.sampling.seed, Some(42));
        assert_eq!(cfg.hardware.num_threads, 1);
    }

    #[test]
    fn test_preset_debug() {
        let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Debug).build().unwrap();
        assert_eq!(cfg.sampling.seed, Some(0));
        assert_eq!(cfg.generation.max_tokens, 8);
        assert_eq!(cfg.hardware.memory_limit_mb, 256);
    }

    // -- Validation errors --------------------------------------------------

    #[test]
    fn test_validation_negative_temperature() {
        let err = InferenceConfigBuilder::new().temperature(-0.1).build().unwrap_err();
        assert!(err.contains("temperature"), "{err}");
    }

    #[test]
    fn test_validation_top_p_zero() {
        let err = InferenceConfigBuilder::new().top_p(0.0).build().unwrap_err();
        assert!(err.contains("top_p"), "{err}");
    }

    #[test]
    fn test_validation_top_p_above_one() {
        let err = InferenceConfigBuilder::new().top_p(1.01).build().unwrap_err();
        assert!(err.contains("top_p"), "{err}");
    }

    #[test]
    fn test_validation_repetition_penalty_zero() {
        let err = InferenceConfigBuilder::new().repetition_penalty(0.0).build().unwrap_err();
        assert!(err.contains("repetition_penalty"), "{err}");
    }

    #[test]
    fn test_validation_max_tokens_zero() {
        let err = InferenceConfigBuilder::new().max_tokens(0).build().unwrap_err();
        assert!(err.contains("max_tokens"), "{err}");
    }

    #[test]
    fn test_validation_num_threads_too_large() {
        let err = InferenceConfigBuilder::new().num_threads(2000).build().unwrap_err();
        assert!(err.contains("num_threads"), "{err}");
    }

    #[test]
    fn test_validation_negative_repetition_penalty() {
        let err = InferenceConfigBuilder::new().repetition_penalty(-1.0).build().unwrap_err();
        assert!(err.contains("repetition_penalty"), "{err}");
    }

    // -- Serialization round-trip -------------------------------------------

    #[test]
    fn test_serde_round_trip_balanced() {
        let original = InferenceConfigBuilder::new().build().unwrap();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let restored: InferenceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_serde_round_trip_with_overrides() {
        let original = InferenceConfigBuilder::new()
            .preset(InferencePreset::Quality)
            .seed(99)
            .max_tokens(512)
            .stop_sequence("</s>")
            .stop_token_id(128009)
            .stream(true)
            .num_threads(4)
            .memory_limit_mb(1024)
            .build()
            .unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: InferenceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_serde_round_trip_deterministic() {
        let original =
            InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: InferenceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(original.sampling.seed, restored.sampling.seed);
    }

    // -- Builder chaining ---------------------------------------------------

    #[test]
    fn test_chaining_overrides_preset() {
        let cfg = InferenceConfigBuilder::new()
            .preset(InferencePreset::Fast)
            .temperature(0.5)
            .max_tokens(200)
            .build()
            .unwrap();
        assert_eq!(cfg.sampling.temperature, 0.5);
        assert_eq!(cfg.generation.max_tokens, 200);
        // Rest stays from Fast preset
        assert_eq!(cfg.hardware.num_threads, 1);
    }

    #[test]
    fn test_preset_resets_previous_overrides() {
        let cfg = InferenceConfigBuilder::new()
            .temperature(0.99)
            .max_tokens(999)
            .preset(InferencePreset::Deterministic)
            .build()
            .unwrap();
        assert_eq!(cfg.sampling.temperature, 0.0);
        assert_eq!(cfg.generation.max_tokens, 128);
    }

    #[test]
    fn test_multiple_stop_sequences() {
        let cfg = InferenceConfigBuilder::new()
            .stop_sequence("</s>")
            .stop_sequence("\n\nQ:")
            .stop_token_id(128009)
            .stop_token_id(128001)
            .build()
            .unwrap();
        assert_eq!(cfg.generation.stop_sequences.len(), 2);
        assert_eq!(cfg.generation.stop_token_ids.len(), 2);
    }

    #[test]
    fn test_stop_sequences_replaces() {
        let cfg = InferenceConfigBuilder::new()
            .stop_sequence("old")
            .stop_sequences(vec!["new1".into(), "new2".into()])
            .build()
            .unwrap();
        assert_eq!(cfg.generation.stop_sequences, vec!["new1", "new2"]);
    }

    #[test]
    fn test_stop_token_ids_replaces() {
        let cfg = InferenceConfigBuilder::new()
            .stop_token_id(1)
            .stop_token_ids(vec![2, 3])
            .build()
            .unwrap();
        assert_eq!(cfg.generation.stop_token_ids, vec![2, 3]);
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_temperature_zero_is_valid() {
        let cfg = InferenceConfigBuilder::new().temperature(0.0).build().unwrap();
        assert_eq!(cfg.sampling.temperature, 0.0);
    }

    #[test]
    fn test_top_p_exactly_one_is_valid() {
        let cfg = InferenceConfigBuilder::new().top_p(1.0).build().unwrap();
        assert_eq!(cfg.sampling.top_p, 1.0);
    }

    #[test]
    fn test_top_k_zero_disables() {
        let cfg = InferenceConfigBuilder::new().top_k(0).build().unwrap();
        assert_eq!(cfg.sampling.top_k, 0);
    }

    #[test]
    fn test_max_tokens_one_is_valid() {
        let cfg = InferenceConfigBuilder::new().max_tokens(1).build().unwrap();
        assert_eq!(cfg.generation.max_tokens, 1);
    }

    #[test]
    fn test_large_max_tokens() {
        let cfg = InferenceConfigBuilder::new().max_tokens(u32::MAX).build().unwrap();
        assert_eq!(cfg.generation.max_tokens, u32::MAX);
    }

    #[test]
    fn test_memory_limit_zero_means_unlimited() {
        let cfg = InferenceConfigBuilder::new().build().unwrap();
        assert_eq!(cfg.hardware.memory_limit_mb, 0);
    }

    #[test]
    fn test_seed_persists_through_chain() {
        let cfg = InferenceConfigBuilder::new()
            .seed(123)
            .temperature(0.5)
            .max_tokens(32)
            .build()
            .unwrap();
        assert_eq!(cfg.sampling.seed, Some(123));
    }

    #[test]
    fn test_stream_flag() {
        let cfg = InferenceConfigBuilder::new().stream(true).build().unwrap();
        assert!(cfg.generation.stream);
    }

    #[test]
    fn test_hardware_overrides() {
        let cfg =
            InferenceConfigBuilder::new().num_threads(8).memory_limit_mb(2048).build().unwrap();
        assert_eq!(cfg.hardware.num_threads, 8);
        assert_eq!(cfg.hardware.memory_limit_mb, 2048);
    }

    #[test]
    fn test_preset_enum_serde() {
        let json = serde_json::to_string(&InferencePreset::Quality).unwrap();
        let restored: InferencePreset = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, InferencePreset::Quality);
    }

    #[test]
    fn test_validate_passes_for_all_presets() {
        for preset in [
            InferencePreset::Fast,
            InferencePreset::Balanced,
            InferencePreset::Quality,
            InferencePreset::Deterministic,
            InferencePreset::Debug,
        ] {
            let result = InferenceConfigBuilder::new().preset(preset).build();
            assert!(result.is_ok(), "preset {preset:?} should produce a valid config");
        }
    }
}
