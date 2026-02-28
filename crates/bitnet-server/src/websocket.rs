//! WebSocket streaming endpoint for GPU inference.
//!
//! Provides real-time token-by-token output via WebSocket connections.
//! Falls back to Server-Sent Events (SSE) for clients that don't
//! support WebSocket.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// WebSocket message types for inference streaming.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Client sends inference request.
    #[serde(rename = "request")]
    Request {
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        stream: Option<bool>,
    },
    /// Server sends a generated token.
    #[serde(rename = "token")]
    Token { text: String, token_id: u32, logprob: Option<f32>, finish_reason: Option<String> },
    /// Server sends generation metadata.
    #[serde(rename = "metadata")]
    Metadata {
        model: String,
        backend: String,
        tokens_per_second: f64,
        total_tokens: usize,
        prompt_tokens: usize,
    },
    /// Server sends error.
    #[serde(rename = "error")]
    Error { message: String, code: u32 },
    /// Ping for keepalive.
    #[serde(rename = "ping")]
    Ping,
    /// Pong for keepalive.
    #[serde(rename = "pong")]
    Pong,
}

/// Configuration for WebSocket streaming.
pub struct WsConfig {
    /// Maximum number of concurrent WebSocket connections.
    pub max_connections: usize,
    /// Idle timeout before disconnecting (seconds).
    pub idle_timeout_secs: u64,
    /// Maximum incoming message size in bytes.
    pub max_message_size: usize,
    /// Interval between heartbeat pings (seconds).
    pub heartbeat_interval_secs: u64,
    /// Maximum queued messages before applying backpressure.
    pub backpressure_limit: usize,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            idle_timeout_secs: 300,
            max_message_size: 64 * 1024,
            heartbeat_interval_secs: 30,
            backpressure_limit: 1024,
        }
    }
}

/// Manages active WebSocket connections.
pub struct WsConnectionManager {
    config: WsConfig,
    active_connections: Arc<AtomicUsize>,
}

impl WsConnectionManager {
    /// Create a new connection manager with the given configuration.
    pub fn new(config: WsConfig) -> Self {
        Self { config, active_connections: Arc::new(AtomicUsize::new(0)) }
    }

    /// Return the number of active connections.
    pub fn active_count(&self) -> usize {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Return `true` when the manager can accept another connection.
    pub fn can_accept(&self) -> bool {
        self.active_count() < self.config.max_connections
    }

    /// Register a new connection. Returns `false` if at capacity.
    pub fn register(&self) -> bool {
        let prev =
            self.active_connections.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                if cur < self.config.max_connections { Some(cur + 1) } else { None }
            });
        prev.is_ok()
    }

    /// Unregister a connection (on disconnect).
    pub fn unregister(&self) {
        self.active_connections.fetch_sub(1, Ordering::SeqCst);
    }

    /// Return the configured maximum number of connections.
    pub fn max_connections(&self) -> usize {
        self.config.max_connections
    }

    /// Return the configured backpressure limit.
    pub fn backpressure_limit(&self) -> usize {
        self.config.backpressure_limit
    }

    /// Return the configured idle timeout in seconds.
    pub fn idle_timeout_secs(&self) -> u64 {
        self.config.idle_timeout_secs
    }

    /// Return the configured heartbeat interval in seconds.
    pub fn heartbeat_interval_secs(&self) -> u64 {
        self.config.heartbeat_interval_secs
    }

    /// Return the configured max message size in bytes.
    pub fn max_message_size(&self) -> usize {
        self.config.max_message_size
    }
}

/// Check whether the queued message count exceeds the backpressure
/// threshold.
pub fn should_apply_backpressure(queued: usize, limit: usize) -> bool {
    queued >= limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_config_defaults_are_sane() {
        let cfg = WsConfig::default();
        assert_eq!(cfg.max_connections, 100);
        assert_eq!(cfg.idle_timeout_secs, 300);
        assert_eq!(cfg.max_message_size, 64 * 1024);
        assert_eq!(cfg.heartbeat_interval_secs, 30);
        assert_eq!(cfg.backpressure_limit, 1024);
    }

    #[test]
    fn connection_manager_starts_at_zero() {
        let mgr = WsConnectionManager::new(WsConfig::default());
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.can_accept());
    }

    #[test]
    fn register_and_unregister_adjusts_count() {
        let mgr = WsConnectionManager::new(WsConfig { max_connections: 2, ..WsConfig::default() });
        assert!(mgr.register());
        assert_eq!(mgr.active_count(), 1);
        assert!(mgr.register());
        assert_eq!(mgr.active_count(), 2);
        assert!(!mgr.register()); // at capacity
        assert!(!mgr.can_accept());

        mgr.unregister();
        assert_eq!(mgr.active_count(), 1);
        assert!(mgr.can_accept());
    }

    #[test]
    fn backpressure_applies_at_limit() {
        assert!(!should_apply_backpressure(0, 10));
        assert!(!should_apply_backpressure(9, 10));
        assert!(should_apply_backpressure(10, 10));
        assert!(should_apply_backpressure(11, 10));
    }
}
