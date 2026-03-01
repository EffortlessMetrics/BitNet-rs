//! Configuration hot-reload with validation, diffing, and rollback.
//!
//! Provides [`HotReloader`] for watching config sources and atomically
//! swapping in validated updates, with automatic rollback on failure.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ── Config source ─────────────────────────────────────────────────────

/// Where a configuration value originates.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigSource {
    /// Local file path.
    File(PathBuf),
    /// Environment variable.
    Environment,
    /// Remote endpoint URL.
    Remote(String),
}

impl fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(p) => write!(f, "file:{}", p.display()),
            Self::Environment => write!(f, "env"),
            Self::Remote(url) => write!(f, "remote:{url}"),
        }
    }
}

// ── Reload config ─────────────────────────────────────────────────────

/// Settings that govern the hot-reload behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadConfig {
    /// Paths (or URLs) to watch for changes.
    pub watch_paths: Vec<ConfigSource>,
    /// How often the poller checks for changes (milliseconds).
    pub poll_interval_ms: u64,
    /// Whether the validator runs before applying a new config.
    pub validation_enabled: bool,
}

impl Default for ReloadConfig {
    fn default() -> Self {
        Self { watch_paths: Vec::new(), poll_interval_ms: 1000, validation_enabled: true }
    }
}

// ── ConfigWatcher trait ───────────────────────────────────────────────

/// Abstraction over how file/source changes are detected.
pub trait ConfigWatcher: Send + Sync {
    /// Check whether the watched source has changed since last call.
    fn has_changed(&self) -> bool;

    /// Read the current raw content of the watched source.
    fn read_content(&self) -> Result<String, ReloadError>;

    /// Return the source this watcher is observing.
    fn source(&self) -> &ConfigSource;
}

// ── PollingWatcher ────────────────────────────────────────────────────

/// Cross-platform poll-based watcher that detects changes by content
/// hash comparison.
pub struct PollingWatcher {
    source: ConfigSource,
    interval: Duration,
    last_hash: Mutex<u64>,
    content: Mutex<String>,
}

impl PollingWatcher {
    pub const fn new(source: ConfigSource, interval: Duration) -> Self {
        Self { source, interval, last_hash: Mutex::new(0), content: Mutex::new(String::new()) }
    }

    /// Return the poll interval.
    pub const fn interval(&self) -> Duration {
        self.interval
    }

    /// Inject content for testing without file I/O.
    pub fn set_content(&self, content: &str) {
        let hash = Self::hash_content(content);
        content.clone_into(&mut self.content.lock().unwrap());
        *self.last_hash.lock().unwrap() = hash;
    }

    /// Update the stored content and return whether it differs.
    pub fn update_content(&self, content: &str) -> bool {
        let new_hash = Self::hash_content(content);
        let mut last = self.last_hash.lock().unwrap();
        if *last == new_hash {
            return false;
        }
        *last = new_hash;
        drop(last);
        content.clone_into(&mut self.content.lock().unwrap());
        true
    }

    fn hash_content(s: &str) -> u64 {
        // FNV-1a 64-bit
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

impl ConfigWatcher for PollingWatcher {
    fn has_changed(&self) -> bool {
        let hash = Self::hash_content(&self.content.lock().unwrap());
        let last = self.last_hash.lock().unwrap();
        hash != *last
    }

    fn read_content(&self) -> Result<String, ReloadError> {
        let content = self.content.lock().unwrap();
        if content.is_empty() {
            return Err(ReloadError::SourceUnreachable(self.source.to_string()));
        }
        Ok(content.clone())
    }

    fn source(&self) -> &ConfigSource {
        &self.source
    }
}

// ── ConfigDiff ────────────────────────────────────────────────────────

/// Describes the difference between two configuration snapshots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
}

impl ConfigDiff {
    /// True when no keys were added, removed, or changed.
    pub const fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    /// Total number of differences.
    pub const fn len(&self) -> usize {
        self.added.len() + self.removed.len() + self.changed.len()
    }

    /// Compute the diff between two key-value maps.
    pub fn compute(old: &HashMap<String, String>, new: &HashMap<String, String>) -> Self {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        for (k, v) in new {
            match old.get(k) {
                None => added.push(k.clone()),
                Some(old_v) if old_v != v => changed.push(k.clone()),
                _ => {}
            }
        }
        for k in old.keys() {
            if !new.contains_key(k) {
                removed.push(k.clone());
            }
        }

        added.sort();
        removed.sort();
        changed.sort();

        Self { added, removed, changed }
    }
}

impl fmt::Display for ConfigDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+{} -{} ~{}", self.added.len(), self.removed.len(), self.changed.len())
    }
}

// ── ReloadCallback ────────────────────────────────────────────────────

/// Callback invoked when configuration changes are detected.
pub type ReloadCallback = Box<dyn Fn(&ConfigDiff) -> Result<(), ReloadError> + Send + Sync>;

// ── ConfigValidator ───────────────────────────────────────────────────

/// Validates a new configuration snapshot before it is applied.
pub trait ConfigValidator: Send + Sync {
    /// Return `Ok(())` if `new_config` is acceptable, or a descriptive
    /// error if not.
    fn validate(&self, new_config: &HashMap<String, String>) -> Result<(), ReloadError>;
}

/// Default validator that accepts everything.
pub struct NoopValidator;

impl ConfigValidator for NoopValidator {
    fn validate(&self, _new_config: &HashMap<String, String>) -> Result<(), ReloadError> {
        Ok(())
    }
}

/// Validator that rejects configs missing required keys.
pub struct RequiredKeysValidator {
    pub required: Vec<String>,
}

impl ConfigValidator for RequiredKeysValidator {
    fn validate(&self, new_config: &HashMap<String, String>) -> Result<(), ReloadError> {
        for key in &self.required {
            if !new_config.contains_key(key) {
                return Err(ReloadError::ValidationFailed(format!("missing required key: {key}")));
            }
        }
        Ok(())
    }
}

// ── RollbackManager ──────────────────────────────────────────────────

/// Keeps a snapshot of the previous config so that a failed update can
/// be rolled back atomically.
pub struct RollbackManager {
    previous: Mutex<Option<HashMap<String, String>>>,
    rollback_count: Mutex<u64>,
}

impl RollbackManager {
    pub const fn new() -> Self {
        Self { previous: Mutex::new(None), rollback_count: Mutex::new(0) }
    }

    /// Save `snapshot` as the rollback target.
    pub fn save(&self, snapshot: HashMap<String, String>) {
        *self.previous.lock().unwrap() = Some(snapshot);
    }

    /// Retrieve the saved snapshot (if any) and clear it.
    pub fn rollback(&self) -> Option<HashMap<String, String>> {
        let snap = self.previous.lock().unwrap().take();
        if snap.is_some() {
            *self.rollback_count.lock().unwrap() += 1;
        }
        snap
    }

    /// How many rollbacks have been performed.
    pub fn rollback_count(&self) -> u64 {
        *self.rollback_count.lock().unwrap()
    }

    /// Whether a rollback snapshot is available.
    pub fn has_snapshot(&self) -> bool {
        self.previous.lock().unwrap().is_some()
    }
}

impl Default for RollbackManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── AtomicConfig ─────────────────────────────────────────────────────

/// Thread-safe config holder with atomic swaps via `Arc<RwLock<T>>`.
pub struct AtomicConfig<T> {
    inner: Arc<RwLock<T>>,
    version: Mutex<u64>,
}

impl<T: Clone> AtomicConfig<T> {
    pub fn new(initial: T) -> Self {
        Self { inner: Arc::new(RwLock::new(initial)), version: Mutex::new(0) }
    }

    /// Read the current value.
    pub fn load(&self) -> T {
        self.inner.read().unwrap().clone()
    }

    /// Atomically replace the config, incrementing the version.
    pub fn store(&self, value: T) {
        *self.inner.write().unwrap() = value;
        *self.version.lock().unwrap() += 1;
    }

    /// Current config version (incremented on each `store`).
    pub fn version(&self) -> u64 {
        *self.version.lock().unwrap()
    }

    /// Get a clone of the inner `Arc` for sharing across threads.
    pub fn shared(&self) -> Arc<RwLock<T>> {
        Arc::clone(&self.inner)
    }
}

impl<T: Clone + Default> Default for AtomicConfig<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

// ── ReloadMetrics ────────────────────────────────────────────────────

/// Counters tracking hot-reload operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadMetrics {
    pub reload_count: u64,
    pub failure_count: u64,
    pub rollback_count: u64,
    pub last_reload_ms: Option<u128>,
    pub last_reload_instant: Option<u128>,
}

impl ReloadMetrics {
    pub const fn new() -> Self {
        Self {
            reload_count: 0,
            failure_count: 0,
            rollback_count: 0,
            last_reload_ms: None,
            last_reload_instant: None,
        }
    }

    /// Total attempts (successes + failures).
    pub const fn total_attempts(&self) -> u64 {
        self.reload_count + self.failure_count
    }

    /// Success rate as a fraction in [0.0, 1.0].
    pub fn success_rate(&self) -> f64 {
        let total = self.total_attempts();
        if total == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.reload_count as f64 / total as f64
        }
    }
}

impl Default for ReloadMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Errors ───────────────────────────────────────────────────────────

/// Errors that can occur during config reload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadError {
    /// The config source could not be reached.
    SourceUnreachable(String),
    /// New config failed validation.
    ValidationFailed(String),
    /// Error while parsing config content.
    ParseError(String),
    /// Rollback was triggered after a failed apply.
    RollbackTriggered(String),
    /// No rollback snapshot available.
    NoRollbackAvailable,
    /// The watcher is already running.
    AlreadyRunning,
    /// Generic I/O error description.
    Io(String),
}

impl fmt::Display for ReloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceUnreachable(s) => {
                write!(f, "source unreachable: {s}")
            }
            Self::ValidationFailed(s) => {
                write!(f, "validation failed: {s}")
            }
            Self::ParseError(s) => write!(f, "parse error: {s}"),
            Self::RollbackTriggered(s) => {
                write!(f, "rollback triggered: {s}")
            }
            Self::NoRollbackAvailable => {
                write!(f, "no rollback snapshot available")
            }
            Self::AlreadyRunning => write!(f, "watcher already running"),
            Self::Io(s) => write!(f, "I/O error: {s}"),
        }
    }
}

impl std::error::Error for ReloadError {}

// ── HotReloader ──────────────────────────────────────────────────────

/// Watches config sources and reloads on change, with validation and
/// rollback support.
pub struct HotReloader {
    config: AtomicConfig<HashMap<String, String>>,
    reload_config: ReloadConfig,
    validator: Box<dyn ConfigValidator>,
    rollback: RollbackManager,
    metrics: Mutex<ReloadMetrics>,
    callbacks: Mutex<Vec<ReloadCallback>>,
    running: Mutex<bool>,
}

impl HotReloader {
    /// Create a reloader with the given settings and validator.
    pub fn new(reload_config: ReloadConfig, validator: Box<dyn ConfigValidator>) -> Self {
        Self {
            config: AtomicConfig::new(HashMap::new()),
            reload_config,
            validator,
            rollback: RollbackManager::new(),
            metrics: Mutex::new(ReloadMetrics::new()),
            callbacks: Mutex::new(Vec::new()),
            running: Mutex::new(false),
        }
    }

    /// Create a reloader with default settings and no validation.
    pub fn with_defaults() -> Self {
        Self::new(ReloadConfig::default(), Box::new(NoopValidator))
    }

    /// Register a callback invoked after each successful reload.
    pub fn on_reload(&self, cb: ReloadCallback) {
        self.callbacks.lock().unwrap().push(cb);
    }

    /// Read the current config snapshot.
    pub fn current_config(&self) -> HashMap<String, String> {
        self.config.load()
    }

    /// Config version counter.
    pub fn version(&self) -> u64 {
        self.config.version()
    }

    /// Access reload metrics.
    pub fn metrics(&self) -> ReloadMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Access the reload configuration.
    pub const fn reload_config(&self) -> &ReloadConfig {
        &self.reload_config
    }

    /// Whether the watcher loop is running.
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Mark the reloader as running (used by external loops).
    pub fn set_running(&self, val: bool) {
        *self.running.lock().unwrap() = val;
    }

    /// Attempt to apply a new config, with validation and rollback.
    pub fn apply(&self, new_config: HashMap<String, String>) -> Result<ConfigDiff, ReloadError> {
        let start = Instant::now();
        let old = self.config.load();
        let diff = ConfigDiff::compute(&old, &new_config);

        if diff.is_empty() {
            return Ok(diff);
        }

        // Validate if enabled.
        if self.reload_config.validation_enabled
            && let Err(e) = self.validator.validate(&new_config)
        {
            self.metrics.lock().unwrap().failure_count += 1;
            return Err(e);
        }

        // Save rollback snapshot.
        self.rollback.save(old);

        // Swap in new config.
        self.config.store(new_config);

        // Fire callbacks.
        for cb in self.callbacks.lock().unwrap().iter() {
            if let Err(e) = cb(&diff) {
                // Rollback on callback failure.
                if let Some(prev) = self.rollback.rollback() {
                    self.config.store(prev);
                }
                let mut m = self.metrics.lock().unwrap();
                m.failure_count += 1;
                m.rollback_count += 1;
                drop(m);
                return Err(ReloadError::RollbackTriggered(e.to_string()));
            }
        }

        // Update metrics.
        let elapsed = start.elapsed();
        let mut m = self.metrics.lock().unwrap();
        m.reload_count += 1;
        m.last_reload_ms = Some(elapsed.as_millis());
        drop(m);

        Ok(diff)
    }

    /// Force a rollback to the previous config snapshot.
    pub fn force_rollback(&self) -> Result<(), ReloadError> {
        self.rollback.rollback().map_or(Err(ReloadError::NoRollbackAvailable), |prev| {
            self.config.store(prev);
            self.metrics.lock().unwrap().rollback_count += 1;
            Ok(())
        })
    }

    /// Parse `key=value` lines into a config map.
    pub fn parse_kv(content: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                map.insert(k.trim().to_owned(), v.trim().to_owned());
            }
        }
        map
    }

    /// Perform one poll cycle: read from watcher, parse, and apply.
    pub fn poll_once(&self, watcher: &dyn ConfigWatcher) -> Result<ConfigDiff, ReloadError> {
        let content = watcher.read_content()?;
        let new_config = Self::parse_kv(&content);
        self.apply(new_config)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ConfigSource ─────────────────────────────────────────────

    #[test]
    fn config_source_file_display() {
        let src = ConfigSource::File(PathBuf::from("/etc/app.conf"));
        assert!(src.to_string().contains("file:"));
    }

    #[test]
    fn config_source_env_display() {
        let src = ConfigSource::Environment;
        assert_eq!(src.to_string(), "env");
    }

    #[test]
    fn config_source_remote_display() {
        let src = ConfigSource::Remote("https://cfg.example".into());
        assert!(src.to_string().contains("remote:"));
    }

    #[test]
    fn config_source_equality() {
        let a = ConfigSource::File(PathBuf::from("a.toml"));
        let b = ConfigSource::File(PathBuf::from("a.toml"));
        assert_eq!(a, b);
    }

    #[test]
    fn config_source_inequality() {
        let a = ConfigSource::File(PathBuf::from("a.toml"));
        let b = ConfigSource::Environment;
        assert_ne!(a, b);
    }

    #[test]
    fn config_source_serde_roundtrip() {
        let src = ConfigSource::Remote("http://x".into());
        let json = serde_json::to_string(&src).unwrap();
        let back: ConfigSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }

    // ── ReloadConfig ─────────────────────────────────────────────

    #[test]
    fn reload_config_defaults() {
        let rc = ReloadConfig::default();
        assert!(rc.watch_paths.is_empty());
        assert_eq!(rc.poll_interval_ms, 1000);
        assert!(rc.validation_enabled);
    }

    #[test]
    fn reload_config_custom() {
        let rc = ReloadConfig {
            watch_paths: vec![ConfigSource::Environment],
            poll_interval_ms: 500,
            validation_enabled: false,
        };
        assert_eq!(rc.watch_paths.len(), 1);
        assert_eq!(rc.poll_interval_ms, 500);
        assert!(!rc.validation_enabled);
    }

    #[test]
    fn reload_config_serde_roundtrip() {
        let rc = ReloadConfig::default();
        let json = serde_json::to_string(&rc).unwrap();
        let back: ReloadConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.poll_interval_ms, rc.poll_interval_ms);
    }

    // ── PollingWatcher ───────────────────────────────────────────

    #[test]
    fn polling_watcher_detects_change() {
        let w = PollingWatcher::new(
            ConfigSource::File(PathBuf::from("test.conf")),
            Duration::from_millis(100),
        );
        w.set_content("key=val");
        assert!(!w.has_changed());
        assert!(w.update_content("key=val2"));
    }

    #[test]
    fn polling_watcher_no_change_on_same_content() {
        let w = PollingWatcher::new(ConfigSource::Environment, Duration::from_millis(50));
        w.set_content("a=1");
        assert!(!w.update_content("a=1"));
    }

    #[test]
    fn polling_watcher_read_content() {
        let w = PollingWatcher::new(
            ConfigSource::File(PathBuf::from("x.conf")),
            Duration::from_millis(50),
        );
        w.set_content("hello=world");
        assert_eq!(w.read_content().unwrap(), "hello=world");
    }

    #[test]
    fn polling_watcher_empty_content_error() {
        let w = PollingWatcher::new(
            ConfigSource::File(PathBuf::from("empty.conf")),
            Duration::from_millis(50),
        );
        assert!(w.read_content().is_err());
    }

    #[test]
    fn polling_watcher_source() {
        let src = ConfigSource::Remote("http://x".into());
        let w = PollingWatcher::new(src.clone(), Duration::from_millis(50));
        assert_eq!(*w.source(), src);
    }

    #[test]
    fn polling_watcher_interval() {
        let w = PollingWatcher::new(ConfigSource::Environment, Duration::from_millis(250));
        assert_eq!(w.interval(), Duration::from_millis(250));
    }

    // ── ConfigDiff ───────────────────────────────────────────────

    #[test]
    fn diff_empty_maps() {
        let diff = ConfigDiff::compute(&HashMap::new(), &HashMap::new());
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn diff_added_keys() {
        let old = HashMap::new();
        let new = HashMap::from([("a".into(), "1".into())]);
        let diff = ConfigDiff::compute(&old, &new);
        assert_eq!(diff.added, vec!["a"]);
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn diff_removed_keys() {
        let old = HashMap::from([("a".into(), "1".into())]);
        let new = HashMap::new();
        let diff = ConfigDiff::compute(&old, &new);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed, vec!["a"]);
    }

    #[test]
    fn diff_changed_keys() {
        let old = HashMap::from([("a".into(), "1".into())]);
        let new = HashMap::from([("a".into(), "2".into())]);
        let diff = ConfigDiff::compute(&old, &new);
        assert_eq!(diff.changed, vec!["a"]);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn diff_mixed() {
        let old = HashMap::from([
            ("keep".into(), "v".into()),
            ("del".into(), "x".into()),
            ("chg".into(), "old".into()),
        ]);
        let new = HashMap::from([
            ("keep".into(), "v".into()),
            ("add".into(), "y".into()),
            ("chg".into(), "new".into()),
        ]);
        let diff = ConfigDiff::compute(&old, &new);
        assert_eq!(diff.added, vec!["add"]);
        assert_eq!(diff.removed, vec!["del"]);
        assert_eq!(diff.changed, vec!["chg"]);
        assert_eq!(diff.len(), 3);
    }

    #[test]
    fn diff_display() {
        let diff = ConfigDiff {
            added: vec!["a".into()],
            removed: vec!["b".into(), "c".into()],
            changed: vec![],
        };
        assert_eq!(diff.to_string(), "+1 -2 ~0");
    }

    #[test]
    fn diff_sorted_output() {
        let old = HashMap::new();
        let new = HashMap::from([
            ("z".into(), "1".into()),
            ("a".into(), "2".into()),
            ("m".into(), "3".into()),
        ]);
        let diff = ConfigDiff::compute(&old, &new);
        assert_eq!(diff.added, vec!["a", "m", "z"]);
    }

    // ── NoopValidator ────────────────────────────────────────────

    #[test]
    fn noop_validator_accepts() {
        let v = NoopValidator;
        assert!(v.validate(&HashMap::new()).is_ok());
    }

    // ── RequiredKeysValidator ────────────────────────────────────

    #[test]
    fn required_keys_validator_passes() {
        let v = RequiredKeysValidator { required: vec!["host".into(), "port".into()] };
        let cfg =
            HashMap::from([("host".into(), "localhost".into()), ("port".into(), "8080".into())]);
        assert!(v.validate(&cfg).is_ok());
    }

    #[test]
    fn required_keys_validator_fails() {
        let v = RequiredKeysValidator { required: vec!["host".into()] };
        assert!(v.validate(&HashMap::new()).is_err());
    }

    #[test]
    fn required_keys_validator_error_message() {
        let v = RequiredKeysValidator { required: vec!["missing_key".into()] };
        let err = v.validate(&HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("missing_key"));
    }

    // ── RollbackManager ─────────────────────────────────────────

    #[test]
    fn rollback_manager_empty() {
        let rm = RollbackManager::new();
        assert!(!rm.has_snapshot());
        assert_eq!(rm.rollback_count(), 0);
    }

    #[test]
    fn rollback_manager_save_and_restore() {
        let rm = RollbackManager::new();
        rm.save(HashMap::from([("a".into(), "1".into())]));
        assert!(rm.has_snapshot());
        let snap = rm.rollback().unwrap();
        assert_eq!(snap["a"], "1");
        assert!(!rm.has_snapshot());
        assert_eq!(rm.rollback_count(), 1);
    }

    #[test]
    fn rollback_manager_double_rollback() {
        let rm = RollbackManager::new();
        rm.save(HashMap::new());
        rm.rollback();
        assert!(rm.rollback().is_none());
        assert_eq!(rm.rollback_count(), 1);
    }

    #[test]
    fn rollback_manager_default() {
        let rm = RollbackManager::default();
        assert!(!rm.has_snapshot());
    }

    // ── AtomicConfig ─────────────────────────────────────────────

    #[test]
    fn atomic_config_initial() {
        let ac = AtomicConfig::new(42u32);
        assert_eq!(ac.load(), 42);
        assert_eq!(ac.version(), 0);
    }

    #[test]
    fn atomic_config_store_increments_version() {
        let ac = AtomicConfig::new(0u32);
        ac.store(1);
        assert_eq!(ac.load(), 1);
        assert_eq!(ac.version(), 1);
        ac.store(2);
        assert_eq!(ac.version(), 2);
    }

    #[test]
    fn atomic_config_shared() {
        let ac = AtomicConfig::new(String::from("hello"));
        let shared = ac.shared();
        assert_eq!(*shared.read().unwrap(), "hello");
        ac.store(String::from("world"));
        assert_eq!(*shared.read().unwrap(), "world");
    }

    #[test]
    fn atomic_config_default() {
        let ac: AtomicConfig<Vec<u8>> = AtomicConfig::default();
        assert!(ac.load().is_empty());
    }

    #[test]
    fn atomic_config_complex_type() {
        let ac = AtomicConfig::new(HashMap::from([("k".to_owned(), "v".to_owned())]));
        assert_eq!(ac.load()["k"], "v");
    }

    // ── ReloadMetrics ────────────────────────────────────────────

    #[test]
    fn metrics_initial() {
        let m = ReloadMetrics::new();
        assert_eq!(m.reload_count, 0);
        assert_eq!(m.failure_count, 0);
        assert_eq!(m.total_attempts(), 0);
        assert!((m.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_success_rate() {
        let m = ReloadMetrics {
            reload_count: 3,
            failure_count: 1,
            rollback_count: 0,
            last_reload_ms: None,
            last_reload_instant: None,
        };
        assert!((m.success_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_default() {
        let m = ReloadMetrics::default();
        assert_eq!(m.total_attempts(), 0);
    }

    #[test]
    fn metrics_serde_roundtrip() {
        let m = ReloadMetrics {
            reload_count: 5,
            failure_count: 2,
            rollback_count: 1,
            last_reload_ms: Some(42),
            last_reload_instant: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: ReloadMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.reload_count, 5);
        assert_eq!(back.failure_count, 2);
    }

    // ── ReloadError ──────────────────────────────────────────────

    #[test]
    fn error_display_source_unreachable() {
        let e = ReloadError::SourceUnreachable("gone".into());
        assert!(e.to_string().contains("unreachable"));
    }

    #[test]
    fn error_display_validation_failed() {
        let e = ReloadError::ValidationFailed("bad key".into());
        assert!(e.to_string().contains("validation"));
    }

    #[test]
    fn error_display_parse() {
        let e = ReloadError::ParseError("syntax".into());
        assert!(e.to_string().contains("parse"));
    }

    #[test]
    fn error_display_rollback() {
        let e = ReloadError::RollbackTriggered("cb fail".into());
        assert!(e.to_string().contains("rollback"));
    }

    #[test]
    fn error_display_no_rollback() {
        let e = ReloadError::NoRollbackAvailable;
        assert!(e.to_string().contains("no rollback"));
    }

    #[test]
    fn error_display_already_running() {
        let e = ReloadError::AlreadyRunning;
        assert!(e.to_string().contains("already running"));
    }

    #[test]
    fn error_display_io() {
        let e = ReloadError::Io("disk full".into());
        assert!(e.to_string().contains("I/O"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ReloadError::Io("x".into()));
        assert!(!e.to_string().is_empty());
    }

    // ── HotReloader ──────────────────────────────────────────────

    #[test]
    fn reloader_defaults() {
        let r = HotReloader::with_defaults();
        assert!(r.current_config().is_empty());
        assert_eq!(r.version(), 0);
        assert!(!r.is_running());
    }

    #[test]
    fn reloader_apply_new_config() {
        let r = HotReloader::with_defaults();
        let cfg = HashMap::from([("a".into(), "1".into())]);
        let diff = r.apply(cfg).unwrap();
        assert_eq!(diff.added, vec!["a"]);
        assert_eq!(r.current_config()["a"], "1");
        assert_eq!(r.version(), 1);
    }

    #[test]
    fn reloader_apply_no_change() {
        let r = HotReloader::with_defaults();
        let diff = r.apply(HashMap::new()).unwrap();
        assert!(diff.is_empty());
        assert_eq!(r.version(), 0);
    }

    #[test]
    fn reloader_validation_rejects() {
        let rc = ReloadConfig { validation_enabled: true, ..ReloadConfig::default() };
        let v = RequiredKeysValidator { required: vec!["must_exist".into()] };
        let r = HotReloader::new(rc, Box::new(v));
        let err = r.apply(HashMap::from([("a".into(), "1".into())]));
        assert!(err.is_err());
        assert_eq!(r.metrics().failure_count, 1);
    }

    #[test]
    fn reloader_validation_disabled() {
        let rc = ReloadConfig { validation_enabled: false, ..ReloadConfig::default() };
        let v = RequiredKeysValidator { required: vec!["must_exist".into()] };
        let r = HotReloader::new(rc, Box::new(v));
        // Without validation the missing key is fine.
        let diff = r.apply(HashMap::from([("a".into(), "1".into())])).unwrap();
        assert_eq!(diff.added.len(), 1);
    }

    #[test]
    fn reloader_rollback_on_callback_failure() {
        let r = HotReloader::with_defaults();
        r.on_reload(Box::new(|_diff| Err(ReloadError::Io("injected".into()))));
        // Seed initial config so rollback has something.
        r.config.store(HashMap::from([("x".into(), "0".into())]));
        let err = r.apply(HashMap::from([("x".into(), "1".into())])).unwrap_err();
        assert!(matches!(err, ReloadError::RollbackTriggered(_)));
        assert_eq!(r.current_config()["x"], "0");
        assert_eq!(r.metrics().rollback_count, 1);
    }

    #[test]
    fn reloader_force_rollback() {
        let r = HotReloader::with_defaults();
        r.apply(HashMap::from([("a".into(), "1".into())])).unwrap();
        r.apply(HashMap::from([("a".into(), "2".into())])).unwrap();
        r.force_rollback().unwrap();
        assert_eq!(r.current_config()["a"], "1");
    }

    #[test]
    fn reloader_force_rollback_no_snapshot() {
        let r = HotReloader::with_defaults();
        assert!(r.force_rollback().is_err());
    }

    #[test]
    fn reloader_metrics_after_apply() {
        let r = HotReloader::with_defaults();
        r.apply(HashMap::from([("a".into(), "1".into())])).unwrap();
        let m = r.metrics();
        assert_eq!(m.reload_count, 1);
        assert!(m.last_reload_ms.is_some());
    }

    #[test]
    fn reloader_set_running() {
        let r = HotReloader::with_defaults();
        r.set_running(true);
        assert!(r.is_running());
        r.set_running(false);
        assert!(!r.is_running());
    }

    #[test]
    fn reloader_reload_config_accessor() {
        let rc = ReloadConfig { poll_interval_ms: 777, ..ReloadConfig::default() };
        let r = HotReloader::new(rc, Box::new(NoopValidator));
        assert_eq!(r.reload_config().poll_interval_ms, 777);
    }

    // ── HotReloader::parse_kv ────────────────────────────────────

    #[test]
    fn parse_kv_basic() {
        let map = HotReloader::parse_kv("a=1\nb=2\n");
        assert_eq!(map.len(), 2);
        assert_eq!(map["a"], "1");
        assert_eq!(map["b"], "2");
    }

    #[test]
    fn parse_kv_comments_and_blanks() {
        let map = HotReloader::parse_kv("# comment\n\na=1\n");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn parse_kv_whitespace_trimmed() {
        let map = HotReloader::parse_kv("  key  =  val  \n");
        assert_eq!(map["key"], "val");
    }

    #[test]
    fn parse_kv_empty() {
        let map = HotReloader::parse_kv("");
        assert!(map.is_empty());
    }

    #[test]
    fn parse_kv_no_equals() {
        let map = HotReloader::parse_kv("no_equals_here\n");
        assert!(map.is_empty());
    }

    #[test]
    fn parse_kv_equals_in_value() {
        let map = HotReloader::parse_kv("url=http://x?a=1\n");
        assert_eq!(map["url"], "http://x?a=1");
    }

    // ── poll_once ────────────────────────────────────────────────

    #[test]
    fn poll_once_applies_content() {
        let r = HotReloader::with_defaults();
        let w = PollingWatcher::new(
            ConfigSource::File(PathBuf::from("app.conf")),
            Duration::from_millis(50),
        );
        w.set_content("host=localhost\nport=8080");
        let diff = r.poll_once(&w).unwrap();
        assert_eq!(diff.added.len(), 2);
        assert_eq!(r.current_config()["host"], "localhost");
    }

    #[test]
    fn poll_once_error_on_empty() {
        let r = HotReloader::with_defaults();
        let w = PollingWatcher::new(
            ConfigSource::File(PathBuf::from("e.conf")),
            Duration::from_millis(50),
        );
        assert!(r.poll_once(&w).is_err());
    }

    // ── multi-callback ───────────────────────────────────────────

    #[test]
    fn multiple_callbacks_invoked() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = Arc::new(AtomicU32::new(0));
        let c1 = Arc::clone(&counter);
        let c2 = Arc::clone(&counter);

        let r = HotReloader::with_defaults();
        r.on_reload(Box::new(move |_| {
            c1.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));
        r.on_reload(Box::new(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));

        r.apply(HashMap::from([("k".into(), "v".into())])).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    // ── thread safety ────────────────────────────────────────────

    #[test]
    fn atomic_config_concurrent_access() {
        use std::thread;

        let ac = Arc::new(AtomicConfig::new(0u64));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ac = Arc::clone(&ac);
                thread::spawn(move || {
                    ac.store(i);
                    ac.load()
                })
            })
            .collect();
        for h in handles {
            let _ = h.join().unwrap();
        }
        // Just verify no panics; final value is non-deterministic.
        let _ = ac.load();
    }

    // ── edge cases ───────────────────────────────────────────────

    #[test]
    fn apply_successive_updates() {
        let r = HotReloader::with_defaults();
        r.apply(HashMap::from([("a".into(), "1".into())])).unwrap();
        r.apply(HashMap::from([("a".into(), "2".into()), ("b".into(), "3".into())])).unwrap();
        assert_eq!(r.current_config()["a"], "2");
        assert_eq!(r.current_config()["b"], "3");
        assert_eq!(r.version(), 2);
    }

    #[test]
    fn rollback_after_multiple_applies() {
        let r = HotReloader::with_defaults();
        r.apply(HashMap::from([("v".into(), "1".into())])).unwrap();
        r.apply(HashMap::from([("v".into(), "2".into())])).unwrap();
        r.apply(HashMap::from([("v".into(), "3".into())])).unwrap();
        r.force_rollback().unwrap();
        // Rollback restores to snapshot before last apply → v=2.
        assert_eq!(r.current_config()["v"], "2");
    }

    #[test]
    fn large_config_diff() {
        let old: HashMap<String, String> =
            (0..100).map(|i| (format!("k{i}"), format!("v{i}"))).collect();
        let mut new = old.clone();
        new.insert("k0".into(), "changed".into());
        new.remove("k99");
        new.insert("k100".into(), "added".into());

        let diff = ConfigDiff::compute(&old, &new);
        assert_eq!(diff.added, vec!["k100"]);
        assert_eq!(diff.removed, vec!["k99"]);
        assert_eq!(diff.changed, vec!["k0"]);
    }
}
