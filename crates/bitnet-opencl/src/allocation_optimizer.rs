//! Allocation optimizer — analyses profiler data and suggests improvements.
//!
//! Examines allocation history to detect patterns, recommend pre-allocation
//! sizes, and identify fragmentation hot-spots.

use crate::memory_profiler::{AllocationTag, MemoryProfiler, MemoryTimeline};
use std::collections::HashMap;

// ── Pattern detection ────────────────────────────────────────────────────────

/// High-level allocation pattern for a category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationPattern {
    /// Total memory usage is increasing over time.
    Growing,
    /// Total memory usage is decreasing over time.
    Shrinking,
    /// Memory usage is roughly constant.
    Steady,
    /// Not enough data to determine a pattern.
    Unknown,
}

/// A single recommendation from the optimizer.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Human-readable description.
    pub message: String,
    /// Category this recommendation applies to, if any.
    pub tag: Option<AllocationTag>,
}

/// A gap (freed region) found in the allocation timeline.
#[derive(Debug, Clone)]
pub struct FragmentationGap {
    /// Allocation id that was freed, leaving the gap.
    pub alloc_id: u64,
    /// Size of the gap in bytes.
    pub size: u64,
    /// Category of the freed allocation.
    pub tag: AllocationTag,
}

// ── Optimizer ────────────────────────────────────────────────────────────────

/// Analyses a [`MemoryProfiler`] and produces optimisation suggestions.
#[derive(Debug)]
pub struct AllocationOptimizer {
    timeline: MemoryTimeline,
    /// Per-category allocation sizes observed (for pre-alloc hints).
    category_sizes: HashMap<AllocationTag, Vec<u64>>,
    /// Detected patterns per category.
    patterns: HashMap<AllocationTag, AllocationPattern>,
}

impl AllocationOptimizer {
    /// Build an optimizer from the current profiler state.
    #[must_use]
    pub fn from_profiler(profiler: &MemoryProfiler) -> Self {
        let timeline = profiler.timeline();
        let mut category_sizes: HashMap<AllocationTag, Vec<u64>> = HashMap::new();

        for ev in &timeline.events {
            if ev.is_alloc {
                category_sizes.entry(ev.tag).or_default().push(ev.size);
            }
        }

        let patterns =
            category_sizes.iter().map(|(tag, sizes)| (*tag, detect_pattern(sizes))).collect();

        Self { timeline, category_sizes, patterns }
    }

    /// Detected pattern for a specific category.
    #[must_use]
    pub fn pattern_for(&self, tag: &AllocationTag) -> AllocationPattern {
        self.patterns.get(tag).copied().unwrap_or(AllocationPattern::Unknown)
    }

    /// Suggest a pre-allocation size for `tag` based on observed history.
    ///
    /// Returns `None` if no data is available.
    #[must_use]
    pub fn suggested_prealloc(&self, tag: &AllocationTag) -> Option<u64> {
        let sizes = self.category_sizes.get(tag)?;
        if sizes.is_empty() {
            return None;
        }
        // Use the maximum observed size as a safe pre-allocation hint.
        Some(*sizes.iter().max().unwrap_or(&0))
    }

    /// Suggest a pool size for `tag`.
    ///
    /// Returns `None` if no data is available. The pool size is the sum
    /// of all observed allocations, representing the high-water mark of
    /// concurrent usage.
    #[must_use]
    pub fn suggested_pool_size(&self, tag: &AllocationTag) -> Option<u64> {
        let sizes = self.category_sizes.get(tag)?;
        if sizes.is_empty() {
            return None;
        }
        Some(sizes.iter().sum())
    }

    /// Identify fragmentation gaps (freed allocations in the timeline).
    #[must_use]
    pub fn fragmentation_gaps(&self) -> Vec<FragmentationGap> {
        self.timeline
            .events
            .iter()
            .filter(|e| !e.is_alloc)
            .map(|e| FragmentationGap { alloc_id: e.alloc_id, size: e.size, tag: e.tag })
            .collect()
    }

    /// Produce a list of recommendations.
    #[must_use]
    pub fn recommendations(&self) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        for (tag, pattern) in &self.patterns {
            match pattern {
                AllocationPattern::Growing => {
                    recs.push(Recommendation {
                        message: format!(
                            "{tag}: memory usage is growing — \
                             consider a larger pool or streaming \
                             deallocation"
                        ),
                        tag: Some(*tag),
                    });
                }
                AllocationPattern::Steady => {
                    if let Some(pool) = self.suggested_pool_size(tag) {
                        recs.push(Recommendation {
                            message: format!(
                                "{tag}: steady usage — \
                                 pre-allocate a {pool}-byte pool"
                            ),
                            tag: Some(*tag),
                        });
                    }
                }
                AllocationPattern::Shrinking | AllocationPattern::Unknown => {}
            }
        }

        let gaps = self.fragmentation_gaps();
        if !gaps.is_empty() {
            let total_gap: u64 = gaps.iter().map(|g| g.size).sum();
            recs.push(Recommendation {
                message: format!(
                    "fragmentation: {} freed gaps totalling \
                     {total_gap} bytes — consider compaction",
                    gaps.len()
                ),
                tag: None,
            });
        }

        recs
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Simple linear-trend heuristic over allocation sizes.
fn detect_pattern(sizes: &[u64]) -> AllocationPattern {
    if sizes.len() < 3 {
        return AllocationPattern::Unknown;
    }

    let n = sizes.len();
    let half = n / 2;
    #[allow(clippy::cast_precision_loss)]
    let first_avg: f64 = sizes[..half].iter().sum::<u64>() as f64 / half as f64;
    #[allow(clippy::cast_precision_loss)]
    let second_avg: f64 = sizes[half..].iter().sum::<u64>() as f64 / (n - half) as f64;

    let ratio = (second_avg - first_avg).abs() / first_avg.max(1.0);

    if ratio < 0.1 {
        AllocationPattern::Steady
    } else if second_avg > first_avg {
        AllocationPattern::Growing
    } else {
        AllocationPattern::Shrinking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_growing_pattern() {
        let sizes = vec![100, 100, 100, 200, 200, 300];
        assert_eq!(detect_pattern(&sizes), AllocationPattern::Growing);
    }

    #[test]
    fn detect_shrinking_pattern() {
        let sizes = vec![300, 300, 300, 100, 100, 100];
        assert_eq!(detect_pattern(&sizes), AllocationPattern::Shrinking);
    }

    #[test]
    fn detect_steady_pattern() {
        let sizes = vec![100, 100, 100, 100, 100, 100];
        assert_eq!(detect_pattern(&sizes), AllocationPattern::Steady);
    }

    #[test]
    fn detect_unknown_with_insufficient_data() {
        assert_eq!(detect_pattern(&[100, 200]), AllocationPattern::Unknown);
        assert_eq!(detect_pattern(&[]), AllocationPattern::Unknown);
    }
}
