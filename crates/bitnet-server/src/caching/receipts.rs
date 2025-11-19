//! KV Pool Receipt Structures
//!
//! This module defines receipt types for tracing and auditing KV cache operations,
//! including evictions, batch operations, and pool health snapshots.
//!
//! These structures are designed to integrate with BitNet.rs's existing receipts
//! infrastructure and support JSON serialization for logging and telemetry.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Receipt emitted when a single KV cache entry is evicted
///
/// Captures before/after pool state for traceability and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEvictionReceipt {
    /// Event type identifier
    pub event: String,

    /// Session being evicted
    pub session_id: String,

    /// Memory freed by this eviction (bytes)
    pub bytes_freed: usize,

    /// Pool state before eviction
    pub pool_used_before: usize,
    pub pool_total: usize,
    pub fragmentation_before: f64,
    pub sessions_before: u64,

    /// Pool state after eviction
    pub pool_used_after: usize,
    pub fragmentation_after: f64,
    pub sessions_after: u64,

    /// Timing and metadata
    pub timestamp: SystemTime,
    pub eviction_reason: EvictionReason,
}

/// Receipt emitted when multiple KV cache entries are evicted in a batch
///
/// Provides aggregate metrics for LRU sweeps and bulk evictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEvictionBatchReceipt {
    /// Event type identifier
    pub event: String,

    /// Number of sessions evicted
    pub count: usize,

    /// Total memory freed (bytes)
    pub total_bytes_freed: usize,

    /// Pool state before batch
    pub pool_used_before: usize,
    pub fragmentation_before: f64,

    /// Pool state after batch
    pub pool_used_after: usize,
    pub fragmentation_after: f64,

    /// Timing information
    pub timestamp: SystemTime,
    pub duration_ms: u64,
}

/// Receipt emitted periodically for pool health monitoring
///
/// Designed for dashboards, alerts, and capacity planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvPoolSnapshotReceipt {
    /// Event type identifier
    pub event: String,

    /// Current pool state
    pub pool_used: usize,
    pub pool_total: usize,
    pub utilization: f64,
    pub fragmentation: f64,

    /// Session statistics
    pub active_sessions: u64,
    pub total_sessions_created: u64,
    pub total_evictions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,

    /// Timing
    pub timestamp: SystemTime,
}

/// Reason for KV cache eviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum EvictionReason {
    /// Manual removal via API (e.g., DELETE /sessions/{id})
    Manual,

    /// LRU eviction due to capacity pressure
    Lru,

    /// TTL expiration (time-to-live exceeded)
    Ttl,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eviction_receipt_roundtrip() {
        let receipt = KvEvictionReceipt {
            event: "kv_eviction".to_string(),
            session_id: "sess_test123".to_string(),
            bytes_freed: 262144,
            pool_used_before: 1048576,
            pool_total: 10485760,
            fragmentation_before: 0.12,
            sessions_before: 8,
            pool_used_after: 786432,
            fragmentation_after: 0.08,
            sessions_after: 7,
            timestamp: SystemTime::UNIX_EPOCH,
            eviction_reason: EvictionReason::Lru,
        };

        // Round-trip through serde_json::Value to avoid lifetime issues
        let value = serde_json::to_value(&receipt).unwrap();
        let parsed: KvEvictionReceipt = serde_json::from_value(value).unwrap();

        assert_eq!(parsed.event, "kv_eviction");
        assert_eq!(parsed.session_id, "sess_test123");
        assert_eq!(parsed.bytes_freed, 262144);
        assert_eq!(parsed.eviction_reason, EvictionReason::Lru);
    }

    #[test]
    fn test_batch_receipt_roundtrip() {
        let receipt = KvEvictionBatchReceipt {
            event: "kv_eviction_batch".to_string(),
            count: 3,
            total_bytes_freed: 786432,
            pool_used_before: 9437184,
            fragmentation_before: 0.18,
            pool_used_after: 8650752,
            fragmentation_after: 0.09,
            timestamp: SystemTime::UNIX_EPOCH,
            duration_ms: 12,
        };

        let value = serde_json::to_value(&receipt).unwrap();
        let parsed: KvEvictionBatchReceipt = serde_json::from_value(value).unwrap();

        assert_eq!(parsed.event, "kv_eviction_batch");
        assert_eq!(parsed.count, 3);
        assert_eq!(parsed.total_bytes_freed, 786432);
    }

    #[test]
    fn test_snapshot_receipt_roundtrip() {
        let receipt = KvPoolSnapshotReceipt {
            event: "kv_pool_snapshot".to_string(),
            pool_used: 8650752,
            pool_total: 10485760,
            utilization: 0.825,
            fragmentation: 0.09,
            active_sessions: 5,
            total_sessions_created: 128,
            total_evictions: 123,
            cache_hits: 456,
            cache_misses: 32,
            timestamp: SystemTime::UNIX_EPOCH,
        };

        let value = serde_json::to_value(&receipt).unwrap();
        let parsed: KvPoolSnapshotReceipt = serde_json::from_value(value).unwrap();

        assert_eq!(parsed.event, "kv_pool_snapshot");
        assert_eq!(parsed.active_sessions, 5);
        assert_eq!(parsed.utilization, 0.825);
    }

    #[test]
    fn test_eviction_reason_roundtrip() {
        let reasons = vec![EvictionReason::Manual, EvictionReason::Lru, EvictionReason::Ttl];

        for reason in reasons {
            let value = serde_json::to_value(reason).unwrap();
            let parsed: EvictionReason = serde_json::from_value(value).unwrap();
            assert_eq!(parsed, reason);
        }
    }
}
