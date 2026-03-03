//! Batch dispatch: plan multiple kernel launches as a single schedule.

use crate::config::LaunchConfig;

/// A single entry in a batch plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEntry {
    /// Logical name / tag for this kernel (for profiling / debugging).
    pub tag: String,
    /// Launch configuration for this kernel.
    pub config: LaunchConfig,
    /// Which CUDA stream ordinal this entry should run on (0 = default).
    pub stream_id: u32,
    /// Indices of entries that must complete before this one starts.
    pub depends_on: Vec<usize>,
}

/// An ordered batch of kernel launches.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    entries: Vec<BatchEntry>,
}

impl BatchPlan {
    /// Create an empty batch plan.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Create a plan with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self { entries: Vec::with_capacity(cap) }
    }

    /// Append a kernel and return its index (for dependency wiring).
    pub fn push(&mut self, entry: BatchEntry) -> usize {
        let idx = self.entries.len();
        self.entries.push(entry);
        idx
    }

    /// Convenience: append a simple kernel on stream 0 with no dependencies.
    pub fn push_simple(&mut self, tag: impl Into<String>, config: LaunchConfig) -> usize {
        self.push(BatchEntry { tag: tag.into(), config, stream_id: 0, depends_on: Vec::new() })
    }

    /// Number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the plan is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Immutable view of all entries.
    #[must_use]
    pub fn entries(&self) -> &[BatchEntry] {
        &self.entries
    }

    /// Validate that all dependency indices are in-range and acyclic (no
    /// forward refs — each entry may only depend on earlier entries).
    pub fn validate(&self) -> Result<(), BatchValidationError> {
        for (i, entry) in self.entries.iter().enumerate() {
            for &dep in &entry.depends_on {
                if dep >= i {
                    return Err(BatchValidationError::ForwardDependency { entry: i, dep });
                }
            }
            if !entry.config.is_valid() {
                return Err(BatchValidationError::InvalidConfig { entry: i });
            }
        }
        Ok(())
    }

    /// Total threads across all entries.
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        self.entries.iter().map(|e| e.config.total_threads()).sum()
    }

    /// Number of distinct streams used.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        for e in &self.entries {
            seen.insert(e.stream_id);
        }
        seen.len()
    }

    /// Return entries grouped by their `stream_id`, in original order.
    #[must_use]
    pub fn entries_by_stream(&self) -> Vec<(u32, Vec<&BatchEntry>)> {
        let mut map: std::collections::BTreeMap<u32, Vec<&BatchEntry>> =
            std::collections::BTreeMap::new();
        for e in &self.entries {
            map.entry(e.stream_id).or_default().push(e);
        }
        map.into_iter().collect()
    }
}

impl Default for BatchPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors detected during batch validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchValidationError {
    /// Entry `entry` depends on `dep` which is not a prior entry.
    ForwardDependency {
        /// Index of the entry with the bad dependency.
        entry: usize,
        /// Index of the dependency that violates ordering.
        dep: usize,
    },
    /// Entry `entry` has an invalid launch configuration.
    InvalidConfig {
        /// Index of the entry with the invalid config.
        entry: usize,
    },
}

impl std::fmt::Display for BatchValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardDependency { entry, dep } => {
                write!(f, "entry {entry} has forward dependency on {dep}")
            }
            Self::InvalidConfig { entry } => {
                write!(f, "entry {entry} has invalid launch config")
            }
        }
    }
}

impl std::error::Error for BatchValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config() -> LaunchConfig {
        LaunchConfig::for_elements(256)
    }

    #[test]
    fn empty_plan() {
        let plan = BatchPlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn push_and_len() {
        let mut plan = BatchPlan::new();
        let idx = plan.push_simple("kern_a", simple_config());
        assert_eq!(idx, 0);
        assert_eq!(plan.len(), 1);
        assert!(!plan.is_empty());
    }

    #[test]
    fn dependency_chain() {
        let mut plan = BatchPlan::new();
        let a = plan.push_simple("a", simple_config());
        let b = plan.push(BatchEntry {
            tag: "b".into(),
            config: simple_config(),
            stream_id: 0,
            depends_on: vec![a],
        });
        let _c = plan.push(BatchEntry {
            tag: "c".into(),
            config: simple_config(),
            stream_id: 0,
            depends_on: vec![a, b],
        });
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn forward_dependency_rejected() {
        let mut plan = BatchPlan::new();
        plan.push(BatchEntry {
            tag: "bad".into(),
            config: simple_config(),
            stream_id: 0,
            depends_on: vec![1],
        });
        plan.push_simple("second", simple_config());

        let err = plan.validate().unwrap_err();
        assert_eq!(err, BatchValidationError::ForwardDependency { entry: 0, dep: 1 });
    }

    #[test]
    fn total_threads_sum() {
        let mut plan = BatchPlan::new();
        plan.push_simple("a", LaunchConfig::for_elements(256));
        plan.push_simple("b", LaunchConfig::for_elements(512));
        assert_eq!(plan.total_threads(), 256 + 512);
    }

    #[test]
    fn stream_count_default() {
        let mut plan = BatchPlan::new();
        plan.push_simple("a", simple_config());
        plan.push_simple("b", simple_config());
        assert_eq!(plan.stream_count(), 1);
    }

    #[test]
    fn stream_count_multi() {
        let mut plan = BatchPlan::new();
        plan.push(BatchEntry {
            tag: "a".into(),
            config: simple_config(),
            stream_id: 0,
            depends_on: vec![],
        });
        plan.push(BatchEntry {
            tag: "b".into(),
            config: simple_config(),
            stream_id: 1,
            depends_on: vec![],
        });
        plan.push(BatchEntry {
            tag: "c".into(),
            config: simple_config(),
            stream_id: 2,
            depends_on: vec![],
        });
        assert_eq!(plan.stream_count(), 3);
    }

    #[test]
    fn entries_by_stream_grouped() {
        let mut plan = BatchPlan::new();
        plan.push(BatchEntry {
            tag: "s0_a".into(),
            config: simple_config(),
            stream_id: 0,
            depends_on: vec![],
        });
        plan.push(BatchEntry {
            tag: "s1_a".into(),
            config: simple_config(),
            stream_id: 1,
            depends_on: vec![],
        });
        plan.push(BatchEntry {
            tag: "s0_b".into(),
            config: simple_config(),
            stream_id: 0,
            depends_on: vec![],
        });
        let grouped = plan.entries_by_stream();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].0, 0);
        assert_eq!(grouped[0].1.len(), 2);
        assert_eq!(grouped[1].0, 1);
        assert_eq!(grouped[1].1.len(), 1);
    }

    #[test]
    fn with_capacity_starts_empty() {
        let plan = BatchPlan::with_capacity(16);
        assert!(plan.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let plan = BatchPlan::default();
        assert!(plan.is_empty());
    }

    #[test]
    fn validation_error_display() {
        let e = BatchValidationError::ForwardDependency { entry: 2, dep: 5 };
        assert_eq!(e.to_string(), "entry 2 has forward dependency on 5");
        let e2 = BatchValidationError::InvalidConfig { entry: 3 };
        assert_eq!(e2.to_string(), "entry 3 has invalid launch config");
    }
}
