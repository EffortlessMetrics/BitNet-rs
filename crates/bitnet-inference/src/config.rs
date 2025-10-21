//! # Inference Configuration
//!
//! Configuration structures for inference engine and text generation.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

/// Type alias for logits callback function
pub type LogitsCallback = Arc<dyn Fn(usize, Vec<(u32, f32)>, u32) + Send + Sync>;

/// Configuration for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum context length to maintain
    pub max_context_length: usize,
    /// Number of threads for CPU inference
    pub num_threads: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable mixed precision inference
    pub mixed_precision: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_context_length: 2048,
            num_threads: num_cpus::get(),
            batch_size: 1,
            mixed_precision: false,
            memory_pool_size: 1024 * 1024 * 512, // 512MB
        }
    }
}

/// Configuration for text generation
#[non_exhaustive]
#[derive(Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: u32,
    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-k sampling limit (0 = disabled)
    pub top_k: u32,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty, higher = less repetition)
    pub repetition_penalty: f32,
    /// Stop sequences to end generation
    pub stop_sequences: Vec<String>,
    /// Token IDs that trigger immediate stop (checked before string matching)
    /// Useful for LLaMA-3 <|eot_id|> and other special tokens
    pub stop_token_ids: Vec<u32>,
    /// Precomputed HashSet for O(1) stop token ID lookups
    /// This is derived from stop_token_ids and not serialized.
    /// Use `with_stop_token_ids()` builder to set stop tokens, which automatically
    /// maintains this internal set for O(1) lookups via `is_stop_token()`.
    #[serde(skip)]
    stop_token_ids_set: HashSet<u32>,
    /// Window size for tail-based string matching (default: 64)
    /// Only decode the last N tokens when checking stop sequences to avoid O(n²) decode costs
    pub stop_string_window: usize,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Whether to skip special tokens in output
    pub skip_special_tokens: bool,
    /// EOS token ID for stopping generation (None = use tokenizer default)
    pub eos_token_id: Option<u32>,
    /// Number of decode steps to capture logits for (0 = disabled)
    pub logits_tap_steps: usize,
    /// Number of top tokens to capture per step
    pub logits_topk: usize,
    /// Optional callback for capturing logits at each step
    /// Parameters: (step, topk_tokens_and_logits, chosen_token_id)
    #[serde(skip)]
    pub logits_cb: Option<LogitsCallback>,
    /// Whether to add BOS token during tokenization (default: false for pre-formatted prompts)
    pub add_bos: bool,
}

impl std::fmt::Debug for GenerationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationConfig")
            .field("max_new_tokens", &self.max_new_tokens)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("stop_sequences", &self.stop_sequences)
            .field("stop_token_ids", &self.stop_token_ids)
            .field("stop_string_window", &self.stop_string_window)
            .field("seed", &self.seed)
            .field("skip_special_tokens", &self.skip_special_tokens)
            .field("eos_token_id", &self.eos_token_id)
            .field("logits_tap_steps", &self.logits_tap_steps)
            .field("logits_topk", &self.logits_topk)
            .field("logits_cb", &self.logits_cb.is_some())
            .field("add_bos", &self.add_bos)
            .finish()
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            stop_token_ids: vec![],
            stop_token_ids_set: HashSet::new(),
            stop_string_window: 64, // Default: decode only last 64 tokens for stop sequence matching
            seed: None,
            skip_special_tokens: true,
            eos_token_id: None,
            logits_tap_steps: 0,
            logits_topk: 10,
            logits_cb: None,
            add_bos: false, // Default to false for pre-formatted prompts
        }
    }
}

impl GenerationConfig {
    /// Create a greedy generation config (deterministic)
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 1, top_p: 1.0, ..Default::default() }
    }

    /// Create a creative generation config (high randomness)
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: 100,
            top_p: 0.95,
            repetition_penalty: 1.1,
            ..Default::default()
        }
    }

    /// Create a balanced generation config
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.max_new_tokens == 0 {
            return Err("max_new_tokens must be greater than 0".to_string());
        }

        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".to_string());
        }

        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in range (0.0, 1.0]".to_string());
        }

        if self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be positive".to_string());
        }

        Ok(())
    }

    /// Set random seed for reproducible generation
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add stop sequence
    #[must_use]
    pub fn with_stop_sequence(mut self, stop_seq: String) -> Self {
        self.stop_sequences.push(stop_seq);
        self
    }

    /// Set all stop sequences at once
    ///
    /// # Example
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let config = GenerationConfig::default()
    ///     .with_stop_sequences(vec!["</s>".to_string(), "\n\n".to_string()]);
    /// assert_eq!(config.stop_sequences.len(), 2);
    /// ```
    #[must_use]
    pub fn with_stop_sequences<I: IntoIterator<Item = String>>(mut self, sequences: I) -> Self {
        self.stop_sequences = sequences.into_iter().collect();
        self
    }

    /// Set maximum tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_new_tokens = max_tokens;
        self
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k
    #[must_use]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set repetition penalty
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set stop string window size
    #[must_use]
    pub fn with_stop_string_window(mut self, window: usize) -> Self {
        self.stop_string_window = window;
        self
    }

    /// Set whether to skip special tokens in output
    ///
    /// # Example
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let config = GenerationConfig::default()
    ///     .with_skip_special_tokens(false);
    /// assert!(!config.skip_special_tokens);
    /// ```
    #[must_use]
    pub fn with_skip_special_tokens(mut self, skip: bool) -> Self {
        self.skip_special_tokens = skip;
        self
    }

    /// Set whether to add BOS token during tokenization
    ///
    /// # Example
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let config = GenerationConfig::default()
    ///     .with_add_bos(true);
    /// assert!(config.add_bos);
    /// ```
    #[must_use]
    pub fn with_add_bos(mut self, add_bos: bool) -> Self {
        self.add_bos = add_bos;
        self
    }

    /// Add stop token IDs and rebuild the HashSet for O(1) lookups
    ///
    /// This is the preferred way to set stop token IDs as it automatically
    /// maintains the internal HashSet for O(1) lookups via `is_stop_token()`.
    ///
    /// # Example
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let config = GenerationConfig::default()
    ///     .with_stop_token_ids(vec![128009, 128001]); // LLaMA-3 EOT tokens
    ///
    /// assert!(config.is_stop_token(128009)); // O(1) lookup
    /// assert!(config.is_stop_token(128001));
    /// assert!(!config.is_stop_token(999));
    /// ```
    #[must_use]
    pub fn with_stop_token_ids(mut self, token_ids: Vec<u32>) -> Self {
        self.stop_token_ids = token_ids;
        self.rebuild_stop_token_set();
        self
    }

    /// Add a single stop token ID
    ///
    /// # Example
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let config = GenerationConfig::default()
    ///     .with_stop_token_id(128009); // LLaMA-3 <|eot_id|>
    ///
    /// assert!(config.is_stop_token(128009));
    /// ```
    #[must_use]
    pub fn with_stop_token_id(mut self, token_id: u32) -> Self {
        self.stop_token_ids.push(token_id);
        self.stop_token_ids_set.insert(token_id);
        self
    }

    /// Rebuild the stop token HashSet from the Vec
    ///
    /// Call this after:
    /// - Modifying `stop_token_ids` directly (discouraged - use builders instead)
    /// - Deserializing from JSON/YAML (HashSet is not serialized)
    ///
    /// # Example: Direct modification (not recommended)
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let mut config = GenerationConfig::default();
    ///
    /// // Direct modification without rebuild - is_stop_token() won't work!
    /// config.stop_token_ids = vec![128009];
    /// assert!(!config.is_stop_token(128009)); // ❌ Returns false!
    ///
    /// // Must call rebuild_stop_token_set() to sync the internal HashSet
    /// config.rebuild_stop_token_set();
    /// assert!(config.is_stop_token(128009)); // ✅ Now works!
    /// ```
    ///
    /// # Example: Deserialization (required)
    /// ```
    /// use bitnet_inference::config::GenerationConfig;
    ///
    /// let json = r#"{"max_new_tokens":100,"temperature":0.7,"top_k":50,"top_p":0.9,
    ///                "repetition_penalty":1.0,"stop_sequences":[],"stop_token_ids":[128009],
    ///                "stop_string_window":64,"seed":null,"skip_special_tokens":true,
    ///                "eos_token_id":null,"logits_tap_steps":0,"logits_topk":10,"add_bos":false}"#;
    ///
    /// let mut config: GenerationConfig = serde_json::from_str(json).unwrap();
    /// config.rebuild_stop_token_set(); // Required after deserialization
    ///
    /// assert!(config.is_stop_token(128009));
    /// ```
    pub fn rebuild_stop_token_set(&mut self) {
        self.stop_token_ids_set = self.stop_token_ids.iter().copied().collect();
    }

    /// Check if a token ID is a stop token (O(1) using HashSet)
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids_set.contains(&token_id)
    }
}

impl InferenceConfig {
    /// Create configuration optimized for CPU inference
    pub fn cpu_optimized() -> Self {
        Self {
            num_threads: num_cpus::get(),
            mixed_precision: false,
            batch_size: 1,
            ..Default::default()
        }
    }

    /// Create configuration optimized for GPU inference
    pub fn gpu_optimized() -> Self {
        Self {
            mixed_precision: true,
            batch_size: 4,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        }
    }

    /// Create configuration for memory-constrained environments
    pub fn memory_efficient() -> Self {
        Self {
            max_context_length: 1024,
            batch_size: 1,
            memory_pool_size: 1024 * 1024 * 256, // 256MB
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.max_context_length == 0 {
            return Err("max_context_length must be greater than 0".to_string());
        }

        if self.num_threads == 0 {
            return Err("num_threads must be greater than 0".to_string());
        }

        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }

        if self.memory_pool_size == 0 {
            return Err("memory_pool_size must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Set number of threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable mixed precision
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Set memory pool size
    pub fn with_memory_pool_size(mut self, size: usize) -> Self {
        self.memory_pool_size = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(config.stop_sequences.is_empty());
        assert!(config.seed.is_none());
        assert!(config.skip_special_tokens);
    }

    #[test]
    fn test_generation_config_presets() {
        let greedy = GenerationConfig::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let creative = GenerationConfig::creative();
        assert_eq!(creative.temperature, 0.9);
        assert_eq!(creative.top_k, 100);

        let balanced = GenerationConfig::balanced();
        assert_eq!(balanced.temperature, 0.7);
        assert_eq!(balanced.repetition_penalty, 1.05);
    }

    #[test]
    fn test_generation_config_validation() {
        let mut config = GenerationConfig::default();
        assert!(config.validate().is_ok());

        config.max_new_tokens = 0;
        assert!(config.validate().is_err());

        config.max_new_tokens = 100;
        config.temperature = -1.0;
        assert!(config.validate().is_err());

        config.temperature = 0.7;
        config.top_p = 0.0;
        assert!(config.validate().is_err());

        config.top_p = 1.5;
        assert!(config.validate().is_err());

        config.top_p = 0.9;
        config.repetition_penalty = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::default()
            .with_seed(42)
            .with_stop_sequence("</s>".to_string())
            .with_max_tokens(200)
            .with_temperature(0.8)
            .with_top_k(40)
            .with_top_p(0.95);

        assert_eq!(config.seed, Some(42));
        assert_eq!(config.stop_sequences, vec!["</s>"]);
        assert_eq!(config.max_new_tokens, 200);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.top_p, 0.95);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_context_length, 2048);
        assert_eq!(config.num_threads, num_cpus::get());
        assert_eq!(config.batch_size, 1);
        assert!(!config.mixed_precision);
        assert_eq!(config.memory_pool_size, 1024 * 1024 * 512);
    }

    #[test]
    fn test_inference_config_presets() {
        let cpu_config = InferenceConfig::cpu_optimized();
        assert!(!cpu_config.mixed_precision);
        assert_eq!(cpu_config.batch_size, 1);

        let gpu_config = InferenceConfig::gpu_optimized();
        assert!(gpu_config.mixed_precision);
        assert_eq!(gpu_config.batch_size, 4);

        let memory_config = InferenceConfig::memory_efficient();
        assert_eq!(memory_config.max_context_length, 1024);
        assert_eq!(memory_config.memory_pool_size, 1024 * 1024 * 256);
    }

    #[test]
    fn test_inference_config_validation() {
        let mut config = InferenceConfig::default();
        assert!(config.validate().is_ok());

        config.max_context_length = 0;
        assert!(config.validate().is_err());

        config.max_context_length = 2048;
        config.num_threads = 0;
        assert!(config.validate().is_err());

        config.num_threads = 4;
        config.batch_size = 0;
        assert!(config.validate().is_err());

        config.batch_size = 1;
        config.memory_pool_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::default()
            .with_threads(8)
            .with_batch_size(4)
            .with_mixed_precision(true)
            .with_memory_pool_size(1024 * 1024 * 1024);

        assert_eq!(config.num_threads, 8);
        assert_eq!(config.batch_size, 4);
        assert!(config.mixed_precision);
        assert_eq!(config.memory_pool_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_config_serialization() {
        let gen_config = GenerationConfig::default();
        let serialized = serde_json::to_string(&gen_config).unwrap();
        let deserialized: GenerationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(gen_config.max_new_tokens, deserialized.max_new_tokens);
        assert_eq!(gen_config.temperature, deserialized.temperature);

        let inf_config = InferenceConfig::default();
        let serialized = serde_json::to_string(&inf_config).unwrap();
        let deserialized: InferenceConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(inf_config.max_context_length, deserialized.max_context_length);
        assert_eq!(inf_config.num_threads, deserialized.num_threads);
    }

    #[test]
    fn test_stop_token_set_vs_fallback() {
        // Test 1: Using builder maintains O(1) lookup set
        let config = GenerationConfig::default().with_stop_token_ids(vec![128009, 128001]);

        assert!(config.is_stop_token(128009), "Builder should maintain HashSet for O(1) lookup");
        assert!(config.is_stop_token(128001), "Builder should maintain HashSet for O(1) lookup");
        assert!(!config.is_stop_token(999), "Non-stop token should return false");

        // Test 2: Direct modification without rebuild - is_stop_token() won't work
        // This pattern is intentionally wrong to demonstrate the foot-gun
        #[allow(clippy::field_reassign_with_default)]
        let mut config = GenerationConfig::default();
        #[allow(clippy::field_reassign_with_default)]
        {
            config.stop_token_ids = vec![128009];
        }

        assert!(
            !config.is_stop_token(128009),
            "Direct vec modification without rebuild should NOT enable O(1) lookup"
        );

        // Test 3: After rebuild, O(1) lookup works
        config.rebuild_stop_token_set();
        assert!(
            config.is_stop_token(128009),
            "After rebuild_stop_token_set(), O(1) lookup should work"
        );

        // Test 4: Verify with_stop_token_id also maintains the set
        let config =
            GenerationConfig::default().with_stop_token_id(128009).with_stop_token_id(128001);

        assert!(config.is_stop_token(128009));
        assert!(config.is_stop_token(128001));
        assert_eq!(config.stop_token_ids.len(), 2);
    }

    #[test]
    fn test_new_builder_methods() {
        // Test with_skip_special_tokens
        let config = GenerationConfig::default().with_skip_special_tokens(false);
        assert!(!config.skip_special_tokens);

        // Test with_add_bos
        let config = GenerationConfig::default().with_add_bos(true);
        assert!(config.add_bos);

        // Test with_stop_sequences
        let config = GenerationConfig::default()
            .with_stop_sequences(vec!["</s>".to_string(), "\n\n".to_string()]);
        assert_eq!(config.stop_sequences.len(), 2);
        assert_eq!(config.stop_sequences[0], "</s>");
        assert_eq!(config.stop_sequences[1], "\n\n");

        // Test chaining all new builders
        let config = GenerationConfig::default()
            .with_skip_special_tokens(false)
            .with_add_bos(true)
            .with_stop_sequences(vec!["END".to_string()])
            .with_stop_token_ids(vec![128009]);

        assert!(!config.skip_special_tokens);
        assert!(config.add_bos);
        assert_eq!(config.stop_sequences.len(), 1);
        assert!(config.is_stop_token(128009));
    }
}
