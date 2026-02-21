//! KV Pool Receipt Structures and Event Emission
//!
//! This module defines receipt types for tracing and auditing KV cache operations,
//! including evictions, batch operations, and pool health snapshots.
//!
//! Phase 2 adds trait-based event emission with dependency injection for
//! testability and production observability.

use crate::caching::CachingConfig;
use crate::caching::kv_cache::KVCacheStatistics;
#[cfg(any(test, feature = "tuning"))]
use crate::caching::performance_tuning::PerformanceReport;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Target for tracing-based receipt events.
pub const KV_RECEIPTS_TARGET: &str = "bitnet::kv::receipts";

// ────────────────────────────────────────────────────────────────────────────
// Phase 2: Runtime-wired eviction receipts
// ────────────────────────────────────────────────────────────────────────────

/// Receipt emitted when a single KV cache entry is evicted (Phase 2).
///
/// Captures full before/after statistics snapshots for detailed analysis.
#[derive(Debug, Clone, Serialize)]
pub struct KvEvictionReport {
    /// Stable identifier for the logical session.
    pub session_id: String,

    /// Offset of the evicted block within the arena.
    pub block_offset: usize,

    /// Size of the evicted block in bytes.
    pub block_size_bytes: usize,

    /// Snapshot of cache metrics before the eviction.
    pub before: KVCacheStatistics,

    /// Snapshot of cache metrics after the eviction.
    pub after: KVCacheStatistics,

    /// Optional performance summary at the time of eviction.
    #[cfg(any(test, feature = "tuning"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance: Option<PerformanceReport>,

    /// Wall-clock time the eviction was recorded.
    pub timestamp: SystemTime,
}

/// Trait for consuming KV cache lifecycle events.
///
/// Implementations can route events to tracing, structured logging,
/// metrics backends, or test channels.
pub trait KvEventSink: Send + Sync {
    /// Record a single eviction event.
    fn on_eviction(&self, event: KvEvictionReport);

    // Future extension points for Phase 3/4:
    // fn on_batch(&self, event: KvEvictionBatchReceipt);
    // fn on_performance(&self, report: PerformanceReport);
}

/// A sink that emits events via `tracing::info!`.
///
/// This is the default production sink when receipts are enabled.
pub struct TracingSink;

impl KvEventSink for TracingSink {
    fn on_eviction(&self, event: KvEvictionReport) {
        #[cfg(any(test, feature = "tuning"))]
        let has_perf = event.performance.is_some();
        #[cfg(not(any(test, feature = "tuning")))]
        let has_perf = false;

        tracing::info!(
            target: KV_RECEIPTS_TARGET,
            kv_event = "eviction",
            session_id = %event.session_id,
            block_offset = event.block_offset,
            block_size_bytes = event.block_size_bytes,
            before_sessions = event.before.total_sessions,
            after_sessions = event.after.total_sessions,
            before_mem_mb = event.before.used_memory_mb,
            after_mem_mb = event.after.used_memory_mb,
            has_perf = has_perf,
            "KV eviction recorded"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 1: Receipt types (for future batch/snapshot work)
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// Test-only helpers
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;
    use tokio::sync::mpsc;

    /// A channel-based sink for test assertions.
    ///
    /// Call `ChannelSink::channel(buffer)` to get a `(sink, receiver)` pair.
    #[derive(Clone)]
    pub struct ChannelSink {
        tx: mpsc::Sender<KvEvictionReport>,
    }

    impl KvEventSink for ChannelSink {
        fn on_eviction(&self, event: KvEvictionReport) {
            // Fire and forget; tests should size the buffer appropriately.
            let _ = self.tx.try_send(event);
        }
    }

    impl ChannelSink {
        /// Create a bounded channel sink with the given buffer capacity.
        pub fn channel(buffer: usize) -> (Self, mpsc::Receiver<KvEvictionReport>) {
            let (tx, rx) = mpsc::channel(buffer);
            (Self { tx }, rx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
