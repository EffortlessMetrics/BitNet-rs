//! Runtime configuration hot-reload for Intel Arc A770 OpenCL backend.
//!
//! Allows changing kernel parameters, memory limits, batch sizes, and
//! optimization profiles without restarting the inference server. All
//! operations are CPU reference implementations that compile unconditionally.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// ConfigKey — tunable parameter identifiers
// ---------------------------------------------------------------------------

/// Identifiers for tunable A770 OpenCL kernel parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConfigKey {
    /// OpenCL local work-group size (max 1024 on A770).
    WorkgroupSize,
    /// Tile dimension for tiled matmul kernels.
    TileSize,
    /// Maximum batch size for batched inference.
    MaxBatchSize,
    /// Device memory limit in bytes.
    MemoryLimit,
    /// Number of buffers to prefetch ahead.
    PrefetchDepth,
    /// Whether to use FP16 arithmetic.
    UseFP16,
    /// Whether to use DP4A (dot-product-accumulate) instructions.
    UseDP4A,
    /// Verbosity / debug level (0 = off).
    DebugLevel,
}

impl fmt::Display for ConfigKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::WorkgroupSize => "workgroup_size",
            Self::TileSize => "tile_size",
            Self::MaxBatchSize => "max_batch_size",
            Self::MemoryLimit => "memory_limit",
            Self::PrefetchDepth => "prefetch_depth",
            Self::UseFP16 => "use_fp16",
            Self::UseDP4A => "use_dp4a",
            Self::DebugLevel => "debug_level",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// ConfigValue — type-safe parameter values
// ---------------------------------------------------------------------------

/// Type-safe value for a configuration parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    String(String),
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigSnapshot — versioned point-in-time configuration
// ---------------------------------------------------------------------------

/// An immutable, versioned point-in-time snapshot of all configuration values.
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    values: HashMap<ConfigKey, ConfigValue>,
    version: u64,
    timestamp: u64,
}

impl ConfigSnapshot {
    /// Create a new snapshot with the given values and version.
    pub fn new(values: HashMap<ConfigKey, ConfigValue>, version: u64) -> Self {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        Self { values, version, timestamp }
    }

    /// Create a snapshot with an explicit timestamp (for testing).
    pub fn with_timestamp(
        values: HashMap<ConfigKey, ConfigValue>,
        version: u64,
        timestamp: u64,
    ) -> Self {
        Self { values, version, timestamp }
    }

    pub fn values(&self) -> &HashMap<ConfigKey, ConfigValue> {
        &self.values
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    pub fn get(&self, key: &ConfigKey) -> Option<&ConfigValue> {
        self.values.get(key)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ConfigChange / ConfigDiff — change tracking between snapshots
// ---------------------------------------------------------------------------

/// A single configuration parameter change.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigChange {
    pub key: ConfigKey,
    pub old_value: Option<ConfigValue>,
    pub new_value: Option<ConfigValue>,
}

/// Diff between two configuration snapshots.
#[derive(Debug, Clone)]
pub struct ConfigDiff {
    pub added: Vec<ConfigChange>,
    pub removed: Vec<ConfigChange>,
    pub changed: Vec<ConfigChange>,
}

impl ConfigDiff {
    /// Compute the diff from `old` to `new`.
    pub fn compute(old: &ConfigSnapshot, new: &ConfigSnapshot) -> Self {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        // Keys in new but not in old → added.
        // Keys in both but with different values → changed.
        for (key, new_val) in new.values() {
            match old.get(key) {
                None => added.push(ConfigChange {
                    key: *key,
                    old_value: None,
                    new_value: Some(new_val.clone()),
                }),
                Some(old_val) if old_val != new_val => {
                    changed.push(ConfigChange {
                        key: *key,
                        old_value: Some(old_val.clone()),
                        new_value: Some(new_val.clone()),
                    });
                }
                _ => {} // unchanged
            }
        }

        // Keys in old but not in new → removed.
        for (key, old_val) in old.values() {
            if new.get(key).is_none() {
                removed.push(ConfigChange {
                    key: *key,
                    old_value: Some(old_val.clone()),
                    new_value: None,
                });
            }
        }

        // Sort for deterministic output.
        added.sort_by_key(|c| c.key);
        removed.sort_by_key(|c| c.key);
        changed.sort_by_key(|c| c.key);

        Self { added, removed, changed }
    }

    /// Returns `true` when there are no differences.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    /// Total number of individual changes.
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.removed.len() + self.changed.len()
    }
}

// ---------------------------------------------------------------------------
// ConfigValidator — A770 constraint enforcement
// ---------------------------------------------------------------------------

/// Validation error for a proposed configuration change.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub key: ConfigKey,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.message)
    }
}

/// Validates configuration values against Intel Arc A770 hardware constraints.
#[derive(Debug, Clone)]
pub struct ConfigValidator {
    /// Maximum allowed work-group size (1024 on A770).
    pub max_workgroup_size: u32,
    /// Maximum tile size (limited by shared local memory).
    pub max_tile_size: u32,
    /// Maximum device memory in bytes (16 GiB for A770).
    pub max_memory_bytes: u64,
    /// Maximum prefetch depth.
    pub max_prefetch_depth: u32,
    /// Maximum batch size.
    pub max_batch_size: u32,
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::a770()
    }
}

impl ConfigValidator {
    /// Default constraints for the Intel Arc A770.
    pub fn a770() -> Self {
        Self {
            max_workgroup_size: 1024,
            max_tile_size: 128,
            max_memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
            max_prefetch_depth: 16,
            max_batch_size: 256,
        }
    }

    /// Validate a single key-value pair. Returns `Ok(())` or a list of errors.
    pub fn validate(
        &self,
        key: ConfigKey,
        value: &ConfigValue,
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        match (key, value) {
            (ConfigKey::WorkgroupSize, ConfigValue::U32(v)) => {
                if *v == 0 {
                    errors.push(ValidationError {
                        key,
                        message: "workgroup size must be > 0".into(),
                    });
                }
                if *v > self.max_workgroup_size {
                    errors.push(ValidationError {
                        key,
                        message: format!(
                            "workgroup size {v} exceeds A770 max {}",
                            self.max_workgroup_size
                        ),
                    });
                }
                if *v > 0 && !v.is_power_of_two() {
                    errors.push(ValidationError {
                        key,
                        message: format!("workgroup size {v} is not a power of two"),
                    });
                }
            }
            (ConfigKey::TileSize, ConfigValue::U32(v)) => {
                if *v == 0 {
                    errors.push(ValidationError { key, message: "tile size must be > 0".into() });
                }
                if *v > self.max_tile_size {
                    errors.push(ValidationError {
                        key,
                        message: format!("tile size {v} exceeds max {}", self.max_tile_size),
                    });
                }
            }
            (ConfigKey::MaxBatchSize, ConfigValue::U32(v)) => {
                if *v == 0 {
                    errors.push(ValidationError { key, message: "batch size must be > 0".into() });
                }
                if *v > self.max_batch_size {
                    errors.push(ValidationError {
                        key,
                        message: format!("batch size {v} exceeds max {}", self.max_batch_size),
                    });
                }
            }
            (ConfigKey::MemoryLimit, ConfigValue::U64(v)) => {
                if *v > self.max_memory_bytes {
                    errors.push(ValidationError {
                        key,
                        message: format!(
                            "memory limit {} exceeds A770 max {}",
                            v, self.max_memory_bytes
                        ),
                    });
                }
            }
            (ConfigKey::PrefetchDepth, ConfigValue::U32(v)) => {
                if *v > self.max_prefetch_depth {
                    errors.push(ValidationError {
                        key,
                        message: format!(
                            "prefetch depth {v} exceeds max {}",
                            self.max_prefetch_depth
                        ),
                    });
                }
            }
            (ConfigKey::DebugLevel, ConfigValue::U32(v)) => {
                if *v > 5 {
                    errors.push(ValidationError {
                        key,
                        message: format!("debug level {v} exceeds max 5"),
                    });
                }
            }
            // Bool keys accept any Bool value.
            (ConfigKey::UseFP16 | ConfigKey::UseDP4A, ConfigValue::Bool(_)) => {}
            // Type mismatch.
            (_, _) => {
                errors.push(ValidationError {
                    key,
                    message: format!("unexpected value type for {key}"),
                });
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Validate an entire snapshot. Returns all errors found.
    pub fn validate_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), Vec<ValidationError>> {
        let mut all_errors = Vec::new();
        for (key, value) in snapshot.values() {
            if let Err(errs) = self.validate(*key, value) {
                all_errors.extend(errs);
            }
        }
        if all_errors.is_empty() { Ok(()) } else { Err(all_errors) }
    }
}

// ---------------------------------------------------------------------------
// RollbackHistory — stores previous snapshots for undo
// ---------------------------------------------------------------------------

/// Bounded history of configuration snapshots for rollback support.
#[derive(Debug)]
pub struct RollbackHistory {
    snapshots: Vec<ConfigSnapshot>,
    max_depth: usize,
}

impl RollbackHistory {
    /// Create a new history with the given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self { snapshots: Vec::new(), max_depth: max_depth.max(1) }
    }

    /// Push a snapshot onto the history stack.
    pub fn push(&mut self, snapshot: ConfigSnapshot) {
        if self.snapshots.len() >= self.max_depth {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);
    }

    /// Pop (undo) the most recent snapshot.
    pub fn pop(&mut self) -> Option<ConfigSnapshot> {
        self.snapshots.pop()
    }

    /// Peek at the most recent snapshot without removing it.
    pub fn peek(&self) -> Option<&ConfigSnapshot> {
        self.snapshots.last()
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Clear all stored history.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

// ---------------------------------------------------------------------------
// ConfigManager — primary entry point
// ---------------------------------------------------------------------------

/// Manages configuration state with versioned snapshots and rollback.
#[derive(Debug)]
pub struct ConfigManager {
    current: HashMap<ConfigKey, ConfigValue>,
    version: u64,
    validator: ConfigValidator,
    history: RollbackHistory,
}

impl ConfigManager {
    /// Create a new manager with default A770 validation and the given
    /// rollback history depth.
    pub fn new(history_depth: usize) -> Self {
        Self {
            current: HashMap::new(),
            version: 0,
            validator: ConfigValidator::a770(),
            history: RollbackHistory::new(history_depth),
        }
    }

    /// Create a manager with a custom validator.
    pub fn with_validator(validator: ConfigValidator, history_depth: usize) -> Self {
        Self {
            current: HashMap::new(),
            version: 0,
            validator,
            history: RollbackHistory::new(history_depth),
        }
    }

    /// Get the current value for a key.
    pub fn get(&self, key: &ConfigKey) -> Option<&ConfigValue> {
        self.current.get(key)
    }

    /// Set a single configuration value. Validates before applying.
    pub fn set(&mut self, key: ConfigKey, value: ConfigValue) -> Result<(), Vec<ValidationError>> {
        self.validator.validate(key, &value)?;
        // Save current state for rollback.
        self.history.push(self.snapshot());
        self.current.insert(key, value);
        self.version += 1;
        Ok(())
    }

    /// Remove a configuration key.
    pub fn remove(&mut self, key: &ConfigKey) -> Option<ConfigValue> {
        if self.current.contains_key(key) {
            self.history.push(self.snapshot());
            self.version += 1;
            self.current.remove(key)
        } else {
            None
        }
    }

    /// Apply a batch of changes atomically. If any value fails validation
    /// the entire batch is rejected.
    pub fn merge(
        &mut self,
        changes: HashMap<ConfigKey, ConfigValue>,
    ) -> Result<(), Vec<ValidationError>> {
        // Validate all first.
        let mut all_errors = Vec::new();
        for (key, value) in &changes {
            if let Err(errs) = self.validator.validate(*key, value) {
                all_errors.extend(errs);
            }
        }
        if !all_errors.is_empty() {
            return Err(all_errors);
        }
        // All valid — save snapshot and apply.
        self.history.push(self.snapshot());
        self.current.extend(changes);
        self.version += 1;
        Ok(())
    }

    /// Take an immutable snapshot of the current state.
    pub fn snapshot(&self) -> ConfigSnapshot {
        ConfigSnapshot::new(self.current.clone(), self.version)
    }

    /// Rollback to the previous configuration. Returns the diff that was
    /// undone, or `None` if there is no history.
    pub fn rollback(&mut self) -> Option<ConfigDiff> {
        let previous = self.history.pop()?;
        let current_snap = self.snapshot();
        let diff = ConfigDiff::compute(&current_snap, &previous);
        self.current = previous.values().clone();
        // Keep the version monotonically increasing.
        self.version += 1;
        Some(diff)
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// A770 default configuration preset.
    pub fn a770_defaults() -> HashMap<ConfigKey, ConfigValue> {
        let mut m = HashMap::new();
        m.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(256));
        m.insert(ConfigKey::TileSize, ConfigValue::U32(16));
        m.insert(ConfigKey::MaxBatchSize, ConfigValue::U32(32));
        m.insert(ConfigKey::MemoryLimit, ConfigValue::U64(16 * 1024 * 1024 * 1024));
        m.insert(ConfigKey::PrefetchDepth, ConfigValue::U32(4));
        m.insert(ConfigKey::UseFP16, ConfigValue::Bool(true));
        m.insert(ConfigKey::UseDP4A, ConfigValue::Bool(true));
        m.insert(ConfigKey::DebugLevel, ConfigValue::U32(0));
        m
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_manager() -> ConfigManager {
        ConfigManager::new(10)
    }

    fn default_snapshot() -> ConfigSnapshot {
        ConfigSnapshot::new(ConfigManager::a770_defaults(), 1)
    }

    // -----------------------------------------------------------------------
    // ConfigKey
    // -----------------------------------------------------------------------

    #[test]
    fn config_key_display() {
        assert_eq!(ConfigKey::WorkgroupSize.to_string(), "workgroup_size");
        assert_eq!(ConfigKey::TileSize.to_string(), "tile_size");
        assert_eq!(ConfigKey::MaxBatchSize.to_string(), "max_batch_size");
        assert_eq!(ConfigKey::MemoryLimit.to_string(), "memory_limit");
        assert_eq!(ConfigKey::PrefetchDepth.to_string(), "prefetch_depth");
        assert_eq!(ConfigKey::UseFP16.to_string(), "use_fp16");
        assert_eq!(ConfigKey::UseDP4A.to_string(), "use_dp4a");
        assert_eq!(ConfigKey::DebugLevel.to_string(), "debug_level");
    }

    #[test]
    fn config_key_equality_and_hash() {
        let mut map = HashMap::new();
        map.insert(ConfigKey::WorkgroupSize, 1);
        map.insert(ConfigKey::TileSize, 2);
        assert_eq!(map[&ConfigKey::WorkgroupSize], 1);
        assert_eq!(map[&ConfigKey::TileSize], 2);
    }

    #[test]
    fn config_key_ordering() {
        // Derived Ord should be consistent.
        let mut keys = vec![ConfigKey::DebugLevel, ConfigKey::WorkgroupSize, ConfigKey::TileSize];
        keys.sort();
        assert_eq!(keys[0], ConfigKey::WorkgroupSize);
    }

    // -----------------------------------------------------------------------
    // ConfigValue
    // -----------------------------------------------------------------------

    #[test]
    fn config_value_display() {
        assert_eq!(ConfigValue::U32(42).to_string(), "42");
        assert_eq!(ConfigValue::U64(100).to_string(), "100");
        assert_eq!(ConfigValue::Bool(true).to_string(), "true");
        assert_eq!(ConfigValue::String("hi".into()).to_string(), "hi");
    }

    #[test]
    fn config_value_equality() {
        assert_eq!(ConfigValue::U32(10), ConfigValue::U32(10));
        assert_ne!(ConfigValue::U32(10), ConfigValue::U32(20));
        assert_ne!(ConfigValue::U32(10), ConfigValue::U64(10));
    }

    #[test]
    fn config_value_clone() {
        let v = ConfigValue::String("test".into());
        let v2 = v.clone();
        assert_eq!(v, v2);
    }

    // -----------------------------------------------------------------------
    // ConfigSnapshot
    // -----------------------------------------------------------------------

    #[test]
    fn snapshot_creation_and_access() {
        let snap = default_snapshot();
        assert_eq!(snap.version(), 1);
        assert_eq!(snap.len(), 8);
        assert!(!snap.is_empty());
        assert_eq!(snap.get(&ConfigKey::WorkgroupSize), Some(&ConfigValue::U32(256)));
    }

    #[test]
    fn snapshot_empty() {
        let snap = ConfigSnapshot::new(HashMap::new(), 0);
        assert!(snap.is_empty());
        assert_eq!(snap.len(), 0);
        assert_eq!(snap.get(&ConfigKey::TileSize), None);
    }

    #[test]
    fn snapshot_with_explicit_timestamp() {
        let snap = ConfigSnapshot::with_timestamp(HashMap::new(), 5, 1234567890);
        assert_eq!(snap.version(), 5);
        assert_eq!(snap.timestamp(), 1234567890);
    }

    #[test]
    fn snapshot_versioning_monotonicity() {
        let s1 = ConfigSnapshot::new(HashMap::new(), 1);
        let s2 = ConfigSnapshot::new(HashMap::new(), 2);
        let s3 = ConfigSnapshot::new(HashMap::new(), 3);
        assert!(s1.version() < s2.version());
        assert!(s2.version() < s3.version());
    }

    // -----------------------------------------------------------------------
    // ConfigDiff
    // -----------------------------------------------------------------------

    #[test]
    fn diff_identical_snapshots() {
        let s1 = default_snapshot();
        let s2 = ConfigSnapshot::new(s1.values().clone(), 2);
        let diff = ConfigDiff::compute(&s1, &s2);
        assert!(diff.is_empty());
        assert_eq!(diff.total_changes(), 0);
    }

    #[test]
    fn diff_added_key() {
        let s1 = ConfigSnapshot::new(HashMap::new(), 1);
        let mut vals = HashMap::new();
        vals.insert(ConfigKey::TileSize, ConfigValue::U32(16));
        let s2 = ConfigSnapshot::new(vals, 2);
        let diff = ConfigDiff::compute(&s1, &s2);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].key, ConfigKey::TileSize);
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn diff_removed_key() {
        let mut vals = HashMap::new();
        vals.insert(ConfigKey::TileSize, ConfigValue::U32(16));
        let s1 = ConfigSnapshot::new(vals, 1);
        let s2 = ConfigSnapshot::new(HashMap::new(), 2);
        let diff = ConfigDiff::compute(&s1, &s2);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].key, ConfigKey::TileSize);
        assert!(diff.added.is_empty());
    }

    #[test]
    fn diff_changed_value() {
        let mut v1 = HashMap::new();
        v1.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(256));
        let mut v2 = HashMap::new();
        v2.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(512));
        let s1 = ConfigSnapshot::new(v1, 1);
        let s2 = ConfigSnapshot::new(v2, 2);
        let diff = ConfigDiff::compute(&s1, &s2);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].old_value, Some(ConfigValue::U32(256)));
        assert_eq!(diff.changed[0].new_value, Some(ConfigValue::U32(512)));
    }

    #[test]
    fn diff_mixed_add_remove_change() {
        let mut v1 = HashMap::new();
        v1.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(256));
        v1.insert(ConfigKey::TileSize, ConfigValue::U32(16));
        let mut v2 = HashMap::new();
        v2.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(512));
        v2.insert(ConfigKey::DebugLevel, ConfigValue::U32(1));
        let s1 = ConfigSnapshot::new(v1, 1);
        let s2 = ConfigSnapshot::new(v2, 2);
        let diff = ConfigDiff::compute(&s1, &s2);
        assert_eq!(diff.added.len(), 1); // DebugLevel
        assert_eq!(diff.removed.len(), 1); // TileSize
        assert_eq!(diff.changed.len(), 1); // WorkgroupSize
        assert_eq!(diff.total_changes(), 3);
    }

    // -----------------------------------------------------------------------
    // ConfigValidator
    // -----------------------------------------------------------------------

    #[test]
    fn validator_a770_defaults() {
        let v = ConfigValidator::a770();
        assert_eq!(v.max_workgroup_size, 1024);
        assert_eq!(v.max_tile_size, 128);
        assert_eq!(v.max_prefetch_depth, 16);
        assert_eq!(v.max_batch_size, 256);
    }

    #[test]
    fn validate_workgroup_within_limit() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::WorkgroupSize, &ConfigValue::U32(256)).is_ok());
        assert!(v.validate(ConfigKey::WorkgroupSize, &ConfigValue::U32(1024)).is_ok());
    }

    #[test]
    fn validate_workgroup_exceeds_limit() {
        let v = ConfigValidator::a770();
        let result = v.validate(ConfigKey::WorkgroupSize, &ConfigValue::U32(2048));
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("exceeds")));
    }

    #[test]
    fn validate_workgroup_zero() {
        let v = ConfigValidator::a770();
        let result = v.validate(ConfigKey::WorkgroupSize, &ConfigValue::U32(0));
        assert!(result.is_err());
    }

    #[test]
    fn validate_workgroup_not_power_of_two() {
        let v = ConfigValidator::a770();
        let result = v.validate(ConfigKey::WorkgroupSize, &ConfigValue::U32(100));
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("power of two")));
    }

    #[test]
    fn validate_tile_size_valid() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::TileSize, &ConfigValue::U32(64)).is_ok());
    }

    #[test]
    fn validate_tile_size_exceeds() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::TileSize, &ConfigValue::U32(256)).is_err());
    }

    #[test]
    fn validate_memory_limit_valid() {
        let v = ConfigValidator::a770();
        assert!(
            v.validate(ConfigKey::MemoryLimit, &ConfigValue::U64(8 * 1024 * 1024 * 1024),).is_ok()
        );
    }

    #[test]
    fn validate_memory_limit_exceeds() {
        let v = ConfigValidator::a770();
        assert!(
            v.validate(ConfigKey::MemoryLimit, &ConfigValue::U64(32 * 1024 * 1024 * 1024),)
                .is_err()
        );
    }

    #[test]
    fn validate_prefetch_depth_valid() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::PrefetchDepth, &ConfigValue::U32(4)).is_ok());
    }

    #[test]
    fn validate_prefetch_depth_exceeds() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::PrefetchDepth, &ConfigValue::U32(32)).is_err());
    }

    #[test]
    fn validate_debug_level_valid() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::DebugLevel, &ConfigValue::U32(3)).is_ok());
    }

    #[test]
    fn validate_debug_level_exceeds() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::DebugLevel, &ConfigValue::U32(6)).is_err());
    }

    #[test]
    fn validate_bool_keys() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::UseFP16, &ConfigValue::Bool(true)).is_ok());
        assert!(v.validate(ConfigKey::UseDP4A, &ConfigValue::Bool(false)).is_ok());
    }

    #[test]
    fn validate_type_mismatch() {
        let v = ConfigValidator::a770();
        // WorkgroupSize expects U32, not Bool.
        assert!(v.validate(ConfigKey::WorkgroupSize, &ConfigValue::Bool(true)).is_err());
    }

    #[test]
    fn validate_snapshot_all_valid() {
        let v = ConfigValidator::a770();
        let snap = ConfigSnapshot::new(ConfigManager::a770_defaults(), 1);
        assert!(v.validate_snapshot(&snap).is_ok());
    }

    #[test]
    fn validate_snapshot_with_error() {
        let v = ConfigValidator::a770();
        let mut vals = ConfigManager::a770_defaults();
        vals.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(9999));
        let snap = ConfigSnapshot::new(vals, 1);
        assert!(v.validate_snapshot(&snap).is_err());
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError { key: ConfigKey::WorkgroupSize, message: "too big".into() };
        assert_eq!(e.to_string(), "workgroup_size: too big");
    }

    // -----------------------------------------------------------------------
    // RollbackHistory
    // -----------------------------------------------------------------------

    #[test]
    fn rollback_history_push_pop() {
        let mut h = RollbackHistory::new(5);
        assert!(h.is_empty());
        h.push(ConfigSnapshot::new(HashMap::new(), 1));
        assert_eq!(h.len(), 1);
        let s = h.pop().unwrap();
        assert_eq!(s.version(), 1);
        assert!(h.is_empty());
    }

    #[test]
    fn rollback_history_respects_max_depth() {
        let mut h = RollbackHistory::new(3);
        for i in 0..5 {
            h.push(ConfigSnapshot::new(HashMap::new(), i));
        }
        assert_eq!(h.len(), 3);
        // Oldest entries were evicted; newest is version 4.
        assert_eq!(h.pop().unwrap().version(), 4);
        assert_eq!(h.pop().unwrap().version(), 3);
        assert_eq!(h.pop().unwrap().version(), 2);
        assert!(h.pop().is_none());
    }

    #[test]
    fn rollback_history_peek() {
        let mut h = RollbackHistory::new(5);
        h.push(ConfigSnapshot::new(HashMap::new(), 1));
        h.push(ConfigSnapshot::new(HashMap::new(), 2));
        assert_eq!(h.peek().unwrap().version(), 2);
        assert_eq!(h.len(), 2); // peek doesn't remove
    }

    #[test]
    fn rollback_history_clear() {
        let mut h = RollbackHistory::new(5);
        h.push(ConfigSnapshot::new(HashMap::new(), 1));
        h.push(ConfigSnapshot::new(HashMap::new(), 2));
        h.clear();
        assert!(h.is_empty());
    }

    #[test]
    fn rollback_history_max_depth_at_least_one() {
        let h = RollbackHistory::new(0);
        assert_eq!(h.max_depth(), 1);
    }

    // -----------------------------------------------------------------------
    // ConfigManager — basic get/set
    // -----------------------------------------------------------------------

    #[test]
    fn manager_set_and_get() {
        let mut mgr = make_manager();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(32)).unwrap();
        assert_eq!(mgr.get(&ConfigKey::TileSize), Some(&ConfigValue::U32(32)));
    }

    #[test]
    fn manager_set_invalid_rejected() {
        let mut mgr = make_manager();
        let result = mgr.set(ConfigKey::WorkgroupSize, ConfigValue::U32(9999));
        assert!(result.is_err());
        // Value should NOT have been applied.
        assert!(mgr.get(&ConfigKey::WorkgroupSize).is_none());
    }

    #[test]
    fn manager_version_increments() {
        let mut mgr = make_manager();
        assert_eq!(mgr.version(), 0);
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(16)).unwrap();
        assert_eq!(mgr.version(), 1);
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(32)).unwrap();
        assert_eq!(mgr.version(), 2);
    }

    #[test]
    fn manager_remove_existing() {
        let mut mgr = make_manager();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(16)).unwrap();
        let removed = mgr.remove(&ConfigKey::TileSize);
        assert_eq!(removed, Some(ConfigValue::U32(16)));
        assert!(mgr.get(&ConfigKey::TileSize).is_none());
    }

    #[test]
    fn manager_remove_missing() {
        let mut mgr = make_manager();
        assert!(mgr.remove(&ConfigKey::TileSize).is_none());
    }

    // -----------------------------------------------------------------------
    // ConfigManager — merge
    // -----------------------------------------------------------------------

    #[test]
    fn manager_merge_all_valid() {
        let mut mgr = make_manager();
        mgr.merge(ConfigManager::a770_defaults()).unwrap();
        assert_eq!(mgr.get(&ConfigKey::WorkgroupSize), Some(&ConfigValue::U32(256)));
        assert_eq!(mgr.get(&ConfigKey::UseFP16), Some(&ConfigValue::Bool(true)));
    }

    #[test]
    fn manager_merge_partial_invalid_rejected() {
        let mut mgr = make_manager();
        let mut changes = HashMap::new();
        changes.insert(ConfigKey::TileSize, ConfigValue::U32(16));
        changes.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(9999));
        let result = mgr.merge(changes);
        assert!(result.is_err());
        // Neither change should be applied.
        assert!(mgr.get(&ConfigKey::TileSize).is_none());
    }

    #[test]
    fn manager_merge_overlay() {
        let mut mgr = make_manager();
        mgr.merge(ConfigManager::a770_defaults()).unwrap();
        let mut overlay = HashMap::new();
        overlay.insert(ConfigKey::WorkgroupSize, ConfigValue::U32(512));
        mgr.merge(overlay).unwrap();
        assert_eq!(mgr.get(&ConfigKey::WorkgroupSize), Some(&ConfigValue::U32(512)));
        // Other defaults unchanged.
        assert_eq!(mgr.get(&ConfigKey::TileSize), Some(&ConfigValue::U32(16)));
    }

    // -----------------------------------------------------------------------
    // ConfigManager — snapshot
    // -----------------------------------------------------------------------

    #[test]
    fn manager_snapshot_reflects_state() {
        let mut mgr = make_manager();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(32)).unwrap();
        let snap = mgr.snapshot();
        assert_eq!(snap.version(), mgr.version());
        assert_eq!(snap.get(&ConfigKey::TileSize), Some(&ConfigValue::U32(32)));
    }

    // -----------------------------------------------------------------------
    // ConfigManager — rollback
    // -----------------------------------------------------------------------

    #[test]
    fn manager_rollback_single() {
        let mut mgr = make_manager();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(16)).unwrap();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(32)).unwrap();
        let diff = mgr.rollback().unwrap();
        assert!(!diff.is_empty());
        assert_eq!(mgr.get(&ConfigKey::TileSize), Some(&ConfigValue::U32(16)));
    }

    #[test]
    fn manager_rollback_chain() {
        let mut mgr = make_manager();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(8)).unwrap();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(16)).unwrap();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(32)).unwrap();
        mgr.rollback(); // → 16
        assert_eq!(mgr.get(&ConfigKey::TileSize), Some(&ConfigValue::U32(16)));
        mgr.rollback(); // → 8
        assert_eq!(mgr.get(&ConfigKey::TileSize), Some(&ConfigValue::U32(8)));
        mgr.rollback(); // → empty
        assert!(mgr.get(&ConfigKey::TileSize).is_none());
    }

    #[test]
    fn manager_rollback_beyond_history() {
        let mut mgr = make_manager();
        assert!(mgr.rollback().is_none());
    }

    #[test]
    fn manager_rollback_version_monotonic() {
        let mut mgr = make_manager();
        mgr.set(ConfigKey::TileSize, ConfigValue::U32(16)).unwrap();
        let v_before = mgr.version();
        mgr.rollback();
        assert!(mgr.version() > v_before);
    }

    // -----------------------------------------------------------------------
    // ConfigManager — with_validator
    // -----------------------------------------------------------------------

    #[test]
    fn manager_custom_validator() {
        let mut v = ConfigValidator::a770();
        v.max_workgroup_size = 512;
        let mut mgr = ConfigManager::with_validator(v, 5);
        // 512 is now the max.
        assert!(mgr.set(ConfigKey::WorkgroupSize, ConfigValue::U32(512)).is_ok());
        assert!(mgr.set(ConfigKey::WorkgroupSize, ConfigValue::U32(1024)).is_err());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn empty_config_snapshot() {
        let mgr = make_manager();
        let snap = mgr.snapshot();
        assert!(snap.is_empty());
        assert_eq!(snap.version(), 0);
    }

    #[test]
    fn diff_empty_to_empty() {
        let s1 = ConfigSnapshot::new(HashMap::new(), 0);
        let s2 = ConfigSnapshot::new(HashMap::new(), 1);
        let diff = ConfigDiff::compute(&s1, &s2);
        assert!(diff.is_empty());
    }

    #[test]
    fn a770_defaults_count() {
        let defaults = ConfigManager::a770_defaults();
        assert_eq!(defaults.len(), 8);
    }

    // -----------------------------------------------------------------------
    // Property-style: version monotonicity across operations
    // -----------------------------------------------------------------------

    #[test]
    fn property_version_never_decreases() {
        let mut mgr = make_manager();
        let mut prev_version = mgr.version();
        // Series of sets.
        for v in [64, 128, 256, 512] {
            mgr.set(ConfigKey::WorkgroupSize, ConfigValue::U32(v)).unwrap();
            assert!(mgr.version() > prev_version);
            prev_version = mgr.version();
        }
        // Rollbacks also increase version.
        for _ in 0..4 {
            if mgr.rollback().is_some() {
                assert!(mgr.version() > prev_version);
                prev_version = mgr.version();
            }
        }
    }

    #[test]
    fn property_snapshot_version_matches_manager() {
        let mut mgr = make_manager();
        for v in [64, 128, 256] {
            mgr.set(ConfigKey::WorkgroupSize, ConfigValue::U32(v)).unwrap();
            assert_eq!(mgr.snapshot().version(), mgr.version());
        }
    }

    #[test]
    fn validate_max_batch_size_valid() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::MaxBatchSize, &ConfigValue::U32(128)).is_ok());
    }

    #[test]
    fn validate_max_batch_size_exceeds() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::MaxBatchSize, &ConfigValue::U32(512)).is_err());
    }

    #[test]
    fn validate_max_batch_size_zero() {
        let v = ConfigValidator::a770();
        assert!(v.validate(ConfigKey::MaxBatchSize, &ConfigValue::U32(0)).is_err());
    }
}
