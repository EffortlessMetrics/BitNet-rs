//! GPU memory leak detector.
//!
//! Tracks all GPU buffer allocations with call-site info and reports
//! unreleased buffers when the tracker is dropped or checked.
//! Enable via `BITNET_GPU_LEAK_CHECK=1`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Unique identifier for a GPU buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocId(u64);

impl std::fmt::Display for AllocId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alloc#{}", self.0)
    }
}

/// Information about a single GPU buffer allocation.
#[derive(Debug, Clone)]
pub struct AllocRecord {
    /// Unique allocation ID.
    pub id: AllocId,
    /// Size in bytes.
    pub size: usize,
    /// Description of the call site (e.g. function name, file:line).
    pub call_site: String,
    /// Optional label for the buffer.
    pub label: Option<String>,
    /// Timestamp (monotonic counter) of when this allocation was made.
    pub alloc_order: u64,
}

/// Summary of allocation tracking statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeakSummary {
    /// Total number of allocations made.
    pub total_allocs: u64,
    /// Total number of frees performed.
    pub total_frees: u64,
    /// Number of currently leaked (unreleased) buffers.
    pub leaked_count: u64,
    /// Total bytes still allocated (leaked).
    pub leaked_bytes: u64,
}

impl std::fmt::Display for LeakSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "allocs={}, frees={}, leaked={} ({} bytes)",
            self.total_allocs, self.total_frees, self.leaked_count, self.leaked_bytes
        )
    }
}

/// Check if leak detection is enabled via environment variable.
pub fn is_leak_check_enabled() -> bool {
    matches!(
        std::env::var("BITNET_GPU_LEAK_CHECK").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Error type for leak detector operations.
#[derive(Debug, thiserror::Error)]
pub enum LeakDetectorError {
    #[error("allocation not found: {0}")]
    AllocNotFound(AllocId),
    #[error("double free detected: {0}")]
    DoubleFree(AllocId),
    #[error("leak check not enabled (set BITNET_GPU_LEAK_CHECK=1)")]
    NotEnabled,
}

static NEXT_ALLOC_ID: AtomicU64 = AtomicU64::new(1);

fn next_alloc_id() -> AllocId {
    AllocId(NEXT_ALLOC_ID.fetch_add(1, Ordering::Relaxed))
}

/// GPU memory leak detector that tracks allocations and reports leaks.
pub struct LeakDetector {
    live: Mutex<HashMap<AllocId, AllocRecord>>,
    total_allocs: AtomicU64,
    total_frees: AtomicU64,
    freed_ids: Mutex<Vec<AllocId>>,
    enabled: bool,
    alloc_counter: AtomicU64,
}

impl std::fmt::Debug for LeakDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let summary = self.summary();
        f.debug_struct("LeakDetector")
            .field("enabled", &self.enabled)
            .field("summary", &summary)
            .finish()
    }
}

impl LeakDetector {
    /// Create a new leak detector. If `force` is false, checks
    /// `BITNET_GPU_LEAK_CHECK` environment variable.
    pub fn new(force: bool) -> Result<Self, LeakDetectorError> {
        if !force && !is_leak_check_enabled() {
            return Err(LeakDetectorError::NotEnabled);
        }
        Ok(Self {
            live: Mutex::new(HashMap::new()),
            total_allocs: AtomicU64::new(0),
            total_frees: AtomicU64::new(0),
            freed_ids: Mutex::new(Vec::new()),
            enabled: true,
            alloc_counter: AtomicU64::new(0),
        })
    }

    /// Record a new GPU buffer allocation.
    pub fn record_alloc(
        &self,
        size: usize,
        call_site: &str,
        label: Option<&str>,
    ) -> AllocId {
        let id = next_alloc_id();
        let order = self.alloc_counter.fetch_add(1, Ordering::Relaxed);
        let record = AllocRecord {
            id,
            size,
            call_site: call_site.to_string(),
            label: label.map(|s| s.to_string()),
            alloc_order: order,
        };
        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        live.insert(id, record);
        self.total_allocs.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Record that a GPU buffer was freed.
    pub fn record_free(&self, id: AllocId) -> Result<(), LeakDetectorError> {
        // Check for double free
        {
            let freed = self.freed_ids.lock().unwrap_or_else(|e| e.into_inner());
            if freed.contains(&id) {
                return Err(LeakDetectorError::DoubleFree(id));
            }
        }

        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        if live.remove(&id).is_none() {
            return Err(LeakDetectorError::AllocNotFound(id));
        }
        self.total_frees.fetch_add(1, Ordering::Relaxed);

        let mut freed = self.freed_ids.lock().unwrap_or_else(|e| e.into_inner());
        freed.push(id);

        Ok(())
    }

    /// Get a summary of current allocation state.
    pub fn summary(&self) -> LeakSummary {
        let live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        let leaked_bytes: u64 = live.values().map(|r| r.size as u64).sum();
        LeakSummary {
            total_allocs: self.total_allocs.load(Ordering::Relaxed),
            total_frees: self.total_frees.load(Ordering::Relaxed),
            leaked_count: live.len() as u64,
            leaked_bytes,
        }
    }

    /// Get all currently live (unreleased) allocations.
    pub fn live_allocations(&self) -> Vec<AllocRecord> {
        let live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        let mut records: Vec<_> = live.values().cloned().collect();
        records.sort_by_key(|r| r.alloc_order);
        records
    }

    /// Check if there are any leaks. Returns the list of leaked records
    /// if any exist.
    pub fn check_leaks(&self) -> Option<Vec<AllocRecord>> {
        let records = self.live_allocations();
        if records.is_empty() {
            None
        } else {
            Some(records)
        }
    }

    /// Returns whether the detector is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Reset all tracking state.
    pub fn reset(&self) {
        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        live.clear();
        self.total_allocs.store(0, Ordering::Relaxed);
        self.total_frees.store(0, Ordering::Relaxed);
        let mut freed = self.freed_ids.lock().unwrap_or_else(|e| e.into_inner());
        freed.clear();
        self.alloc_counter.store(0, Ordering::Relaxed);
    }

    /// Format a human-readable leak report.
    pub fn leak_report(&self) -> String {
        let summary = self.summary();
        let mut report = format!("GPU Leak Report: {summary}\n");
        if let Some(leaks) = self.check_leaks() {
            for record in &leaks {
                report.push_str(&format!(
                    "  LEAKED: {} - {} bytes at {} (label: {})\n",
                    record.id,
                    record.size,
                    record.call_site,
                    record.label.as_deref().unwrap_or("<none>"),
                ));
            }
        }
        report
    }
}

impl Drop for LeakDetector {
    fn drop(&mut self) {
        if self.enabled {
            let summary = self.summary();
            if summary.leaked_count > 0 {
                tracing::warn!(
                    "GPU leak detector dropping with {} leaked buffers ({} bytes)",
                    summary.leaked_count,
                    summary.leaked_bytes,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> LeakDetector {
        LeakDetector::new(true).expect("forced detector should succeed")
    }

    #[test]
    fn test_new_requires_env_var() {
        temp_env::with_var("BITNET_GPU_LEAK_CHECK", None::<&str>, || {
            let result = LeakDetector::new(false);
            assert!(matches!(result, Err(LeakDetectorError::NotEnabled)));
        });
    }

    #[test]
    fn test_new_with_force() {
        let detector = LeakDetector::new(true);
        assert!(detector.is_ok());
        assert!(detector.unwrap().is_enabled());
    }

    #[test]
    fn test_record_alloc_and_free() {
        let d = make_detector();
        let id = d.record_alloc(1024, "test_fn", Some("weight_buf"));
        assert_eq!(d.summary().total_allocs, 1);
        assert_eq!(d.summary().leaked_count, 1);
        assert_eq!(d.summary().leaked_bytes, 1024);

        d.record_free(id).unwrap();
        assert_eq!(d.summary().total_frees, 1);
        assert_eq!(d.summary().leaked_count, 0);
        assert_eq!(d.summary().leaked_bytes, 0);
    }

    #[test]
    fn test_multiple_allocs_and_partial_free() {
        let d = make_detector();
        let id1 = d.record_alloc(100, "alloc_a", None);
        let _id2 = d.record_alloc(200, "alloc_b", Some("buf_b"));
        let id3 = d.record_alloc(300, "alloc_c", None);

        assert_eq!(d.summary().total_allocs, 3);
        assert_eq!(d.summary().leaked_bytes, 600);

        d.record_free(id1).unwrap();
        d.record_free(id3).unwrap();

        let summary = d.summary();
        assert_eq!(summary.total_frees, 2);
        assert_eq!(summary.leaked_count, 1);
        assert_eq!(summary.leaked_bytes, 200);
    }

    #[test]
    fn test_double_free_detected() {
        let d = make_detector();
        let id = d.record_alloc(512, "test", None);
        d.record_free(id).unwrap();
        let err = d.record_free(id).unwrap_err();
        assert!(matches!(err, LeakDetectorError::DoubleFree(_)));
    }

    #[test]
    fn test_free_unknown_id() {
        let d = make_detector();
        let bogus = AllocId(999_999);
        let err = d.record_free(bogus).unwrap_err();
        assert!(matches!(err, LeakDetectorError::AllocNotFound(_)));
    }

    #[test]
    fn test_check_leaks_none_when_clean() {
        let d = make_detector();
        let id = d.record_alloc(64, "test", None);
        d.record_free(id).unwrap();
        assert!(d.check_leaks().is_none());
    }

    #[test]
    fn test_check_leaks_some_when_leaking() {
        let d = make_detector();
        d.record_alloc(128, "leaky_fn", Some("leaked_buf"));
        let leaks = d.check_leaks().expect("should have leaks");
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].size, 128);
        assert_eq!(leaks[0].call_site, "leaky_fn");
    }

    #[test]
    fn test_live_allocations_ordered() {
        let d = make_detector();
        d.record_alloc(10, "first", None);
        d.record_alloc(20, "second", None);
        d.record_alloc(30, "third", None);

        let live = d.live_allocations();
        assert_eq!(live.len(), 3);
        assert_eq!(live[0].call_site, "first");
        assert_eq!(live[1].call_site, "second");
        assert_eq!(live[2].call_site, "third");
    }

    #[test]
    fn test_reset() {
        let d = make_detector();
        d.record_alloc(100, "a", None);
        d.record_alloc(200, "b", None);
        d.reset();

        let summary = d.summary();
        assert_eq!(summary.total_allocs, 0);
        assert_eq!(summary.total_frees, 0);
        assert_eq!(summary.leaked_count, 0);
        assert_eq!(summary.leaked_bytes, 0);
    }

    #[test]
    fn test_leak_report_format() {
        let d = make_detector();
        d.record_alloc(256, "matmul_kernel", Some("weight_matrix"));
        let report = d.leak_report();
        assert!(report.contains("GPU Leak Report"));
        assert!(report.contains("256 bytes"));
        assert!(report.contains("matmul_kernel"));
        assert!(report.contains("weight_matrix"));
    }

    #[test]
    fn test_summary_display() {
        let d = make_detector();
        d.record_alloc(1024, "test", None);
        let summary = d.summary();
        let display = format!("{summary}");
        assert!(display.contains("allocs=1"));
        assert!(display.contains("leaked=1"));
        assert!(display.contains("1024 bytes"));
    }

    #[test]
    fn test_alloc_id_display() {
        let id = AllocId(42);
        assert_eq!(format!("{id}"), "alloc#42");
    }
}
