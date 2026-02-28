//! GPU memory profiler and allocation tracker.
//!
//! Tracks all GPU memory allocations with per-category budgets,
//! fragmentation analysis, and timeline recording.
//!
//! Enable at runtime with `BITNET_GPU_MEMORY_PROFILE=1`.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ── Types ────────────────────────────────────────────────────────────────────

/// Category tag for a GPU memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationTag {
    /// Model weight tensors (typically long-lived).
    Weights,
    /// Activation buffers (per-inference, short-lived).
    Activations,
    /// Key-value cache (grows with context length).
    KvCache,
    /// Temporary scratch space.
    Scratch,
}

impl fmt::Display for AllocationTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weights => write!(f, "weights"),
            Self::Activations => write!(f, "activations"),
            Self::KvCache => write!(f, "kv_cache"),
            Self::Scratch => write!(f, "scratch"),
        }
    }
}

/// Record of a single GPU memory allocation.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Unique allocation identifier.
    pub id: u64,
    /// Size in bytes.
    pub size: u64,
    /// When this allocation was created (monotonic).
    pub timestamp: Instant,
    /// Category tag.
    pub tag: AllocationTag,
    /// Whether this allocation has been freed.
    pub freed: bool,
}

/// Point-in-time snapshot of memory state.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Total bytes currently allocated.
    pub total_allocated: u64,
    /// Peak bytes allocated since profiler creation.
    pub peak: u64,
    /// Fragmentation ratio in `[0.0, 1.0]`.
    ///
    /// Computed as `1.0 - (largest_contiguous / total_allocated)`.
    /// Returns `0.0` when nothing is allocated.
    pub fragmentation_ratio: f64,
    /// Per-category breakdown.
    pub by_category: HashMap<AllocationTag, u64>,
    /// Number of live (unfreed) allocations.
    pub live_count: usize,
}

/// A timestamped event in the allocation timeline.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Monotonic timestamp.
    pub timestamp: Instant,
    /// Allocation ID this event refers to.
    pub alloc_id: u64,
    /// `true` for allocation, `false` for free.
    pub is_alloc: bool,
    /// Size in bytes.
    pub size: u64,
    /// Category tag.
    pub tag: AllocationTag,
    /// Running total after this event.
    pub running_total: u64,
}

/// Allocation timeline for pattern analysis.
#[derive(Debug, Clone, Default)]
pub struct MemoryTimeline {
    /// Ordered sequence of allocation/free events.
    pub events: Vec<TimelineEvent>,
}

impl MemoryTimeline {
    /// Number of recorded events.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the timeline is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// Per-category memory budget.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    limits: HashMap<AllocationTag, u64>,
}

impl MemoryBudget {
    /// Create a new budget with no limits set.
    #[must_use]
    pub fn new() -> Self {
        Self { limits: HashMap::new() }
    }

    /// Set the byte limit for a category.
    pub fn set_limit(&mut self, tag: AllocationTag, limit: u64) {
        self.limits.insert(tag, limit);
    }

    /// Get the limit for a category, if any.
    #[must_use]
    pub fn get_limit(&self, tag: &AllocationTag) -> Option<u64> {
        self.limits.get(tag).copied()
    }

    /// Check whether allocating `size` bytes in `tag` would exceed budget.
    ///
    /// Returns `Ok(())` if within budget or no limit is set, otherwise
    /// returns `Err` with a descriptive message.
    ///
    /// # Errors
    ///
    /// Returns an error string if the allocation would exceed the
    /// configured budget for this category.
    pub fn check(&self, tag: &AllocationTag, current: u64, additional: u64) -> Result<(), String> {
        if let Some(limit) = self.limits.get(tag) {
            let total = current.saturating_add(additional);
            if total > *limit {
                return Err(format!("{tag} budget exceeded: {total} > {limit} bytes"));
            }
        }
        Ok(())
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::new()
    }
}

// ── Profiler internals ───────────────────────────────────────────────────────

/// Inner mutable state guarded by a mutex.
#[derive(Debug)]
struct ProfilerInner {
    allocations: HashMap<u64, AllocationRecord>,
    timeline: MemoryTimeline,
    per_category: HashMap<AllocationTag, u64>,
}

/// Thread-safe GPU memory profiler.
///
/// Tracks allocations, computes snapshots, enforces budgets, and records
/// a timeline of allocation events. Atomic counters are used for
/// frequently-read values; a mutex protects the detailed bookkeeping.
#[derive(Debug)]
pub struct MemoryProfiler {
    inner: Arc<Mutex<ProfilerInner>>,
    next_id: AtomicU64,
    total_allocated: AtomicU64,
    peak: AtomicU64,
    budget: Arc<Mutex<MemoryBudget>>,
    enabled: AtomicBool,
}

impl MemoryProfiler {
    /// Create a new profiler.
    ///
    /// If `check_env` is `true`, the profiler is only enabled when
    /// `BITNET_GPU_MEMORY_PROFILE=1` is set.
    #[must_use]
    pub fn new(check_env: bool) -> Self {
        let enabled = if check_env {
            std::env::var("BITNET_GPU_MEMORY_PROFILE")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false)
        } else {
            true
        };

        Self {
            inner: Arc::new(Mutex::new(ProfilerInner {
                allocations: HashMap::new(),
                timeline: MemoryTimeline::default(),
                per_category: HashMap::new(),
            })),
            next_id: AtomicU64::new(1),
            total_allocated: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            budget: Arc::new(Mutex::new(MemoryBudget::new())),
            enabled: AtomicBool::new(enabled),
        }
    }

    /// Whether the profiler is actively tracking.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enable or disable tracking at runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Set the memory budget.
    pub fn set_budget(&self, budget: MemoryBudget) {
        *self.budget.lock().expect("budget lock poisoned") = budget;
    }

    /// Track a new allocation.
    ///
    /// Returns the allocation id on success, or an error if the budget
    /// would be exceeded.
    ///
    /// # Errors
    ///
    /// Returns an error string when the allocation exceeds the configured
    /// budget for this tag category.
    pub fn track_alloc(&self, size: u64, tag: AllocationTag) -> Result<u64, String> {
        if !self.is_enabled() {
            return Ok(0);
        }

        // Budget check.
        {
            let budget = self.budget.lock().expect("budget lock poisoned");
            let current = self
                .inner
                .lock()
                .expect("inner lock poisoned")
                .per_category
                .get(&tag)
                .copied()
                .unwrap_or(0);
            budget.check(&tag, current, size)?;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();

        // Update atomic counters.
        let new_total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        loop {
            let old_peak = self.peak.load(Ordering::Relaxed);
            if new_total <= old_peak {
                break;
            }
            if self
                .peak
                .compare_exchange_weak(old_peak, new_total, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Update inner state.
        {
            let mut inner = self.inner.lock().expect("inner lock poisoned");
            inner
                .allocations
                .insert(id, AllocationRecord { id, size, timestamp: now, tag, freed: false });
            *inner.per_category.entry(tag).or_insert(0) += size;
            inner.timeline.events.push(TimelineEvent {
                timestamp: now,
                alloc_id: id,
                is_alloc: true,
                size,
                tag,
                running_total: new_total,
            });
        }

        Ok(id)
    }

    /// Track a free (deallocation).
    ///
    /// Returns `true` if the allocation was found and freed, `false`
    /// otherwise.
    pub fn track_free(&self, id: u64) -> bool {
        if !self.is_enabled() {
            return false;
        }

        let mut inner = self.inner.lock().expect("inner lock poisoned");
        let Some(record) = inner.allocations.get_mut(&id) else {
            return false;
        };
        if record.freed {
            return false;
        }
        record.freed = true;
        let size = record.size;
        let tag = record.tag;

        self.total_allocated.fetch_sub(size, Ordering::Relaxed);
        if let Some(cat) = inner.per_category.get_mut(&tag) {
            *cat = cat.saturating_sub(size);
        }

        let running = self.total_allocated.load(Ordering::Relaxed);
        inner.timeline.events.push(TimelineEvent {
            timestamp: Instant::now(),
            alloc_id: id,
            is_alloc: false,
            size,
            tag,
            running_total: running,
        });

        true
    }

    /// Capture a point-in-time snapshot.
    #[must_use]
    pub fn snapshot(&self) -> MemorySnapshot {
        let total = self.total_allocated.load(Ordering::Relaxed);
        let peak = self.peak.load(Ordering::Relaxed);

        let inner = self.inner.lock().expect("inner lock poisoned");
        let live_count = inner.allocations.values().filter(|a| !a.freed).count();
        let by_category = inner.per_category.clone();

        // Simple fragmentation heuristic: ratio of freed-but-tracked space.
        let (total_ever, freed_total): (u64, u64) = {
            let te: u64 = inner.allocations.values().map(|a| a.size).sum();
            let ft: u64 = inner.allocations.values().filter(|a| a.freed).map(|a| a.size).sum();
            (te, ft)
        };
        drop(inner);
        #[allow(clippy::cast_precision_loss)]
        let fragmentation_ratio =
            if total_ever > 0 { (freed_total as f64) / (total_ever as f64) } else { 0.0 };

        MemorySnapshot {
            total_allocated: total,
            peak,
            fragmentation_ratio,
            by_category,
            live_count,
        }
    }

    /// Return a clone of the allocation timeline.
    #[must_use]
    pub fn timeline(&self) -> MemoryTimeline {
        self.inner.lock().expect("inner lock poisoned").timeline.clone()
    }

    /// Format a human-readable report.
    #[must_use]
    pub fn report(&self) -> String {
        let snap = self.snapshot();
        let tl = self.timeline();

        let mut s = String::new();
        s.push_str("=== GPU Memory Profile ===\n");
        let _ = writeln!(s, "Total allocated: {} bytes", snap.total_allocated);
        let _ = writeln!(s, "Peak: {} bytes", snap.peak);
        let _ = writeln!(s, "Fragmentation: {:.2}%", snap.fragmentation_ratio * 100.0);
        let _ = writeln!(s, "Live allocations: {}", snap.live_count);
        let _ = writeln!(s, "Timeline events: {}", tl.len());

        for (tag, bytes) in &snap.by_category {
            let _ = writeln!(s, "  {tag}: {bytes} bytes");
        }

        s
    }

    /// Check whether a proposed allocation fits within the current budget.
    ///
    /// # Errors
    ///
    /// Returns an error string if the allocation would exceed the budget
    /// for the given category.
    pub fn check_budget(&self, tag: AllocationTag, size: u64) -> Result<(), String> {
        let budget = self.budget.lock().expect("budget lock poisoned");
        let current = self
            .inner
            .lock()
            .expect("inner lock poisoned")
            .per_category
            .get(&tag)
            .copied()
            .unwrap_or(0);
        budget.check(&tag, current, size)
    }

    /// Current total allocated bytes (atomic read).
    #[must_use]
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Peak allocated bytes (atomic read).
    #[must_use]
    pub fn peak_allocated(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

impl Clone for MemoryProfiler {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            next_id: AtomicU64::new(self.next_id.load(Ordering::Relaxed)),
            total_allocated: AtomicU64::new(self.total_allocated.load(Ordering::Relaxed)),
            peak: AtomicU64::new(self.peak.load(Ordering::Relaxed)),
            budget: Arc::clone(&self.budget),
            enabled: AtomicBool::new(self.is_enabled()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_profiler_returns_zero_stats() {
        let p = MemoryProfiler::new(false);
        let snap = p.snapshot();
        assert_eq!(snap.total_allocated, 0);
        assert_eq!(snap.peak, 0);
        assert_eq!(snap.live_count, 0);
        assert!(snap.fragmentation_ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn single_alloc_updates_counters() {
        let p = MemoryProfiler::new(false);
        let id = p.track_alloc(1024, AllocationTag::Weights).unwrap();
        assert!(id > 0);
        assert_eq!(p.total_allocated(), 1024);
        assert_eq!(p.peak_allocated(), 1024);
    }

    #[test]
    fn alloc_then_free() {
        let p = MemoryProfiler::new(false);
        let id = p.track_alloc(512, AllocationTag::Activations).unwrap();
        assert!(p.track_free(id));
        assert_eq!(p.total_allocated(), 0);
        assert_eq!(p.peak_allocated(), 512);
    }

    #[test]
    fn double_free_returns_false() {
        let p = MemoryProfiler::new(false);
        let id = p.track_alloc(256, AllocationTag::Scratch).unwrap();
        assert!(p.track_free(id));
        assert!(!p.track_free(id));
    }

    #[test]
    fn free_unknown_id_returns_false() {
        let p = MemoryProfiler::new(false);
        assert!(!p.track_free(9999));
    }

    #[test]
    fn peak_tracks_maximum() {
        let p = MemoryProfiler::new(false);
        let a = p.track_alloc(1000, AllocationTag::Weights).unwrap();
        let b = p.track_alloc(2000, AllocationTag::Weights).unwrap();
        // peak = 3000
        p.track_free(a);
        p.track_free(b);
        assert_eq!(p.peak_allocated(), 3000);
        assert_eq!(p.total_allocated(), 0);
    }
}
