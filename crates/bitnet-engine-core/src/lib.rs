//! Orchestration contracts and session types for `BitNet` inference engines.
//!
//! Provides **traits and pure data types** that describe _what_ an inference
//! session must do, without prescribing _how_ it does it.  Concrete
//! implementations live in `bitnet-inference`.

pub use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Session trait
// ---------------------------------------------------------------------------

/// Minimal contract for an inference session.
///
/// Any backend (CPU, GPU, FFI) that can produce tokens from a text prompt
/// should implement this trait.
pub trait InferenceSession: Send + Sync {
    /// Generate tokens for the given `prompt` using the supplied `config`.
    ///
    /// Returns a complete list of [`StreamEvent`]s in the order they would
    /// have been streamed.  The final event must always be
    /// [`StreamEvent::Done`].
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<Vec<StreamEvent>>;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for creating an inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Filesystem path to the GGUF model file.
    pub model_path: String,
    /// Filesystem path to the tokenizer JSON file.
    pub tokenizer_path: String,
    /// Backend identifier (e.g. `"cpu"`, `"cuda"`, `"ffi"`).
    pub backend: String,
    /// Maximum context window in tokens (prompt + generation).
    pub max_context: usize,
    /// Optional random seed for reproducible sessions.
    pub seed: Option<u64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            backend: "cpu".to_string(),
            max_context: 2048,
            seed: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend info
// ---------------------------------------------------------------------------

/// Describes the backend driving a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Human-readable backend name (e.g. `"cpu-rust"`, `"cuda"`).
    pub backend_name: String,
    /// List of kernel identifiers used during inference.
    pub kernel_ids: Vec<String>,
    /// One-line human-readable summary for logs / receipts.
    pub backend_summary: String,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Runtime performance metrics for a completed session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Average tokens generated per second.
    pub tokens_per_second: f64,
    /// Latency from request to first token (milliseconds).
    pub time_to_first_token_ms: f64,
    /// Total number of tokens generated in the session.
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_config_default_values() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.backend, "cpu");
        assert_eq!(cfg.max_context, 2048);
        assert!(cfg.seed.is_none());
        assert!(cfg.model_path.is_empty());
    }

    #[test]
    fn backend_info_default_is_empty() {
        let info = BackendInfo::default();
        assert!(info.backend_name.is_empty());
        assert!(info.kernel_ids.is_empty());
    }

    #[test]
    fn session_metrics_default_is_zero() {
        let m = SessionMetrics::default();
        assert_eq!(m.total_tokens, 0);
        assert_eq!(m.tokens_per_second, 0.0);
        assert_eq!(m.time_to_first_token_ms, 0.0);
    }

    #[test]
    fn generation_config_re_exported() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
    }

    #[test]
    fn stop_reason_variants_accessible() {
        let _max = StopReason::MaxTokens;
        let _eos = StopReason::EosToken;
        let _id = StopReason::StopTokenId(42);
        let _str = StopReason::StopString("</s>".to_string());
    }
}
