//! Orchestration contracts and session types for `BitNet` inference engines.
//!
//! Provides **traits and pure data types** that describe _what_ an inference
//! session must do, without prescribing _how_ it does it.  Concrete
//! implementations live in `bitnet-inference`.

pub use bitnet_engine_state_core::{EngineState, EngineStateError, EngineStateTracker};
pub use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
};
pub use bitnet_session_config_core::{ConfigError, SessionConfig, VALID_BACKENDS};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Session trait
// ---------------------------------------------------------------------------

/// Minimal contract for an inference session.
///
/// Any backend (CPU, GPU, FFI) that can produce tokens from a text prompt
/// should implement this trait.
///
/// # Examples
///
/// ```
/// use bitnet_engine_core::{InferenceSession, GenerationConfig, StreamEvent, StopReason};
/// use anyhow::Result;
///
/// struct EchoSession;
///
/// impl InferenceSession for EchoSession {
///     fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<Vec<StreamEvent>> {
///         use bitnet_engine_core::{TokenEvent, GenerationStats};
///         Ok(vec![
///             StreamEvent::Token(TokenEvent { id: 0, text: prompt.to_string() }),
///             StreamEvent::Done {
///                 reason: StopReason::MaxTokens,
///                 stats: GenerationStats::default(),
///             },
///         ])
///     }
/// }
///
/// let session = EchoSession;
/// let events = session.generate("hello", &GenerationConfig::default()).unwrap();
/// assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
/// ```
pub trait InferenceSession: Send + Sync {
    /// Generate tokens for the given `prompt` using the supplied `config`.
    ///
    /// Returns a complete list of [`StreamEvent`]s in the order they would
    /// have been streamed.  The final event must always be
    /// [`StreamEvent::Done`].
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load, tokenization fails, or
    /// the inference backend encounters an unrecoverable error.
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<Vec<StreamEvent>>;
}

// ---------------------------------------------------------------------------
// Backend info
// ---------------------------------------------------------------------------

/// Describes the backend driving a session.
///
/// Attached to inference receipts and logs to identify which kernels ran.
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
///
/// Populated by the inference engine after generation finishes and written
/// to the inference receipt.
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
// Session ID
// ---------------------------------------------------------------------------

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A unique, non-empty identifier for an inference session.
///
/// IDs are generated monotonically within a process using an atomic counter.
///
/// # Examples
///
/// ```
/// use bitnet_engine_core::SessionId;
///
/// let id = SessionId::generate();
/// assert!(!id.as_str().is_empty());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Generate a new, unique [`SessionId`].
    pub fn generate() -> Self {
        let n = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(format!("session-{n}"))
    }

    /// Return the string representation of this session ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Concurrency configuration
// ---------------------------------------------------------------------------

/// Limits on how many inference sessions may run concurrently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Maximum number of sessions allowed to run at the same time.
    ///
    /// Must be at least 1.  The runtime enforces this limit and queues
    /// requests that exceed it.
    pub max_concurrent: usize,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self { max_concurrent: 4 }
    }
}

impl ConcurrencyConfig {
    /// Returns `true` if `active` is within the configured limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_engine_core::ConcurrencyConfig;
    ///
    /// let cfg = ConcurrencyConfig { max_concurrent: 4 };
    /// assert!(cfg.allows(0));
    /// assert!(cfg.allows(3));
    /// assert!(!cfg.allows(4));
    /// ```
    pub const fn allows(&self, active: usize) -> bool {
        active < self.max_concurrent
    }
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
        // Checking exact zero: these fields are initialized to literal 0.0
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(m.tokens_per_second, 0.0);
            assert_eq!(m.time_to_first_token_ms, 0.0);
        }
    }

    #[test]
    fn generation_config_re_exported() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
    }

    #[test]
    fn engine_state_tracker_re_exported() {
        let mut tracker = EngineStateTracker::new();
        assert_eq!(tracker.state(), &EngineState::Idle);
        tracker.start().expect("idle -> running");
        assert_eq!(tracker.state(), &EngineState::Running);
    }

    #[test]
    fn stop_reason_variants_accessible() {
        // Just verify the variants are constructible
        let _ = StopReason::MaxTokens;
        let _ = StopReason::EosToken;
        let _ = StopReason::StopTokenId(42);
        let _ = StopReason::StopString("</s>".to_string());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// `SessionConfig` round-trips through JSON without data loss.
        #[test]
        fn session_config_json_roundtrip(
            model_path in "[a-z0-9/_\\-]{0,64}",
            tokenizer_path in "[a-z0-9/_\\-]{0,64}",
            backend in "(cpu|cuda|gpu|ffi)",
            max_context in 1usize..=65536,
            seed in proptest::option::of(any::<u64>()),
        ) {
            let cfg = SessionConfig {
                model_path,
                tokenizer_path,
                backend,
                max_context,
                seed,
            };
            let json = serde_json::to_string(&cfg).expect("serialize");
            let back: SessionConfig = serde_json::from_str(&json).expect("deserialize");
            prop_assert_eq!(&cfg.model_path, &back.model_path);
            prop_assert_eq!(&cfg.tokenizer_path, &back.tokenizer_path);
            prop_assert_eq!(&cfg.backend, &back.backend);
            prop_assert_eq!(cfg.max_context, back.max_context);
            prop_assert_eq!(cfg.seed, back.seed);
        }

        /// `BackendInfo` round-trips through JSON.
        #[test]
        fn backend_info_json_roundtrip(
            backend_name in "[a-z0-9_\\-]{0,32}",
            kernel_ids in proptest::collection::vec("[a-z0-9_]{1,16}", 0..=8),
            backend_summary in "[a-z0-9 _\\-]{0,64}",
        ) {
            let info = BackendInfo {
                backend_name,
                kernel_ids,
                backend_summary,
            };
            let json = serde_json::to_string(&info).expect("serialize");
            let back: BackendInfo = serde_json::from_str(&json).expect("deserialize");
            prop_assert_eq!(&info.backend_name, &back.backend_name);
            prop_assert_eq!(&info.kernel_ids, &back.kernel_ids);
            prop_assert_eq!(&info.backend_summary, &back.backend_summary);
        }

        /// `SessionMetrics` non-negativity: metrics constructed from valid
        /// measurements must not yield negative tokens_per_second.
        #[test]
        fn session_metrics_non_negative(
            tps in 0.0f64..1_000_000.0,
            ttft in 0.0f64..60_000.0,
            total in 0usize..=100_000,
        ) {
            let m = SessionMetrics {
                tokens_per_second: tps,
                time_to_first_token_ms: ttft,
                total_tokens: total,
            };
            prop_assert!(m.tokens_per_second >= 0.0);
            prop_assert!(m.time_to_first_token_ms >= 0.0);
            prop_assert_eq!(m.total_tokens, total);
        }
    }
}
