//! Runtime configuration management for GPU inference pipelines.
//!
//! Provides environment-variable parsing, TOML/JSON file loading, multi-source
//! merging, validation, hot-reload watching, feature flags, immutable snapshots,
//! and configuration diffing.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── RuntimeConfig ───────────────────────────────────────────────────────────

/// Top-level runtime configuration container.
///
/// Holds every tuneable knob for GPU inference: device selection, memory
/// limits, kernel parameters, logging, and feature flags.
///
/// **CPU reference**: this struct is backend-agnostic and works identically
/// for CPU-only builds where GPU fields are simply ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Preferred GPU device index (0-based).
    pub device_index: u32,
    /// Maximum GPU memory budget in bytes (0 = unlimited).
    pub max_memory_bytes: u64,
    /// Number of compute threads for CPU fallback.
    pub num_threads: u32,
    /// Batch size for inference requests.
    pub batch_size: u32,
    /// Log verbosity level (`"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`).
    pub log_level: String,
    /// Whether to enable profiling / tracing instrumentation.
    pub enable_profiling: bool,
    /// Optional model path override.
    pub model_path: Option<String>,
    /// Free-form key-value extensions.
    pub extra: BTreeMap<String, String>,
    /// Runtime feature flags.
    pub feature_flags: FeatureFlags,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            max_memory_bytes: 0,
            num_threads: 4,
            batch_size: 1,
            log_level: "info".into(),
            enable_profiling: false,
            model_path: None,
            extra: BTreeMap::new(),
            feature_flags: FeatureFlags::default(),
        }
    }
}

impl RuntimeConfig {
    /// Create a new configuration with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the log level string.
    pub fn log_level(&self) -> &str {
        &self.log_level
    }

    /// Return whether profiling is enabled.
    pub fn profiling_enabled(&self) -> bool {
        self.enable_profiling
    }

    /// Convenience: take an immutable snapshot of this configuration.
    pub fn snapshot(&self) -> ConfigSnapshot {
        ConfigSnapshot::capture(self)
    }
}

// ── ConfigError ─────────────────────────────────────────────────────────────

/// Errors produced during configuration parsing, validation, or I/O.
#[derive(Debug)]
pub enum ConfigError {
    /// An I/O error (file not found, permission denied, etc.).
    Io(std::io::Error),
    /// TOML parse / serialisation error.
    Toml(String),
    /// JSON parse / serialisation error.
    Json(String),
    /// A validation constraint was violated.
    Validation(String),
    /// An environment variable had an invalid value.
    EnvParse(String),
    /// A required configuration key was missing.
    MissingKey(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "config I/O error: {e}"),
            Self::Toml(s) => write!(f, "TOML error: {s}"),
            Self::Json(s) => write!(f, "JSON error: {s}"),
            Self::Validation(s) => write!(f, "validation error: {s}"),
            Self::EnvParse(s) => write!(f, "env parse error: {s}"),
            Self::MissingKey(s) => write!(f, "missing config key: {s}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(e: toml::de::Error) -> Self {
        Self::Toml(e.to_string())
    }
}

impl From<serde_json::Error> for ConfigError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}

// ── EnvParser ───────────────────────────────────────────────────────────────

/// Parses `BITNET_*` environment variables into a [`RuntimeConfig`].
///
/// Recognised variables:
/// - `BITNET_DEVICE_INDEX`
/// - `BITNET_MAX_MEMORY`
/// - `BITNET_NUM_THREADS`
/// - `BITNET_BATCH_SIZE`
/// - `BITNET_LOG_LEVEL`
/// - `BITNET_ENABLE_PROFILING` (`1`/`true`/`yes`)
/// - `BITNET_MODEL_PATH`
/// - `BITNET_FEATURE_*` — arbitrary feature flags
///
/// **CPU reference**: environment parsing is backend-agnostic.
#[derive(Debug, Clone)]
pub struct EnvParser {
    prefix: String,
}

impl Default for EnvParser {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvParser {
    /// Create a parser using the default `BITNET_` prefix.
    pub fn new() -> Self {
        Self {
            prefix: "BITNET_".into(),
        }
    }

    /// Create a parser with a custom prefix.
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    /// Return the configured prefix.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Parse environment variables from an explicit key-value map (testable
    /// without touching the real environment).
    pub fn parse_from_map(
        &self,
        vars: &HashMap<String, String>,
    ) -> Result<RuntimeConfig, ConfigError> {
        let mut cfg = RuntimeConfig::default();

        if let Some(v) = vars.get(&format!("{}DEVICE_INDEX", self.prefix)) {
            cfg.device_index = v
                .parse()
                .map_err(|_| ConfigError::EnvParse(format!("invalid DEVICE_INDEX: {v}")))?;
        }
        if let Some(v) = vars.get(&format!("{}MAX_MEMORY", self.prefix)) {
            cfg.max_memory_bytes = Self::parse_memory(v)?;
        }
        if let Some(v) = vars.get(&format!("{}NUM_THREADS", self.prefix)) {
            cfg.num_threads = v
                .parse()
                .map_err(|_| ConfigError::EnvParse(format!("invalid NUM_THREADS: {v}")))?;
        }
        if let Some(v) = vars.get(&format!("{}BATCH_SIZE", self.prefix)) {
            cfg.batch_size = v
                .parse()
                .map_err(|_| ConfigError::EnvParse(format!("invalid BATCH_SIZE: {v}")))?;
        }
        if let Some(v) = vars.get(&format!("{}LOG_LEVEL", self.prefix)) {
            cfg.log_level = v.to_lowercase();
        }
        if let Some(v) = vars.get(&format!("{}ENABLE_PROFILING", self.prefix)) {
            cfg.enable_profiling = matches!(v.to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Some(v) = vars.get(&format!("{}MODEL_PATH", self.prefix)) {
            cfg.model_path = Some(v.clone());
        }

        // Feature flags: BITNET_FEATURE_<name>=1|true|yes
        let feature_prefix = format!("{}FEATURE_", self.prefix);
        for (k, v) in vars {
            if let Some(name) = k.strip_prefix(&feature_prefix) {
                let enabled = matches!(v.to_lowercase().as_str(), "1" | "true" | "yes");
                cfg.feature_flags
                    .set(name.to_lowercase(), enabled);
            }
        }

        Ok(cfg)
    }

    /// Parse a memory string that may have a suffix (`K`, `M`, `G`, `T`).
    fn parse_memory(s: &str) -> Result<u64, ConfigError> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(0);
        }
        let (digits, multiplier) = match s.as_bytes().last() {
            Some(b'K' | b'k') => (&s[..s.len() - 1], 1024u64),
            Some(b'M' | b'm') => (&s[..s.len() - 1], 1024 * 1024),
            Some(b'G' | b'g') => (&s[..s.len() - 1], 1024 * 1024 * 1024),
            Some(b'T' | b't') => (&s[..s.len() - 1], 1024 * 1024 * 1024 * 1024),
            _ => (s, 1u64),
        };
        let base: u64 = digits
            .parse()
            .map_err(|_| ConfigError::EnvParse(format!("invalid memory value: {s}")))?;
        Ok(base * multiplier)
    }
}

// ── ConfigFile ──────────────────────────────────────────────────────────────

/// Reads and writes [`RuntimeConfig`] to/from TOML and JSON files.
///
/// **CPU reference**: file I/O is fully platform-agnostic.
#[derive(Debug, Clone)]
pub struct ConfigFile {
    path: PathBuf,
    format: ConfigFormat,
}

/// Supported configuration file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    Toml,
    Json,
}

impl ConfigFile {
    /// Create a handle for a config file, auto-detecting format from extension.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, ConfigError> {
        let path = path.into();
        let format = Self::detect_format(&path)?;
        Ok(Self { path, format })
    }

    /// Create a handle with an explicit format.
    pub fn with_format(path: impl Into<PathBuf>, format: ConfigFormat) -> Self {
        Self {
            path: path.into(),
            format,
        }
    }

    /// Return the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the detected / specified format.
    pub fn format(&self) -> ConfigFormat {
        self.format
    }

    /// Read and deserialise the file into a [`RuntimeConfig`].
    pub fn load(&self) -> Result<RuntimeConfig, ConfigError> {
        let text = std::fs::read_to_string(&self.path)?;
        self.parse_str(&text)
    }

    /// Serialise a [`RuntimeConfig`] and write it to the file.
    pub fn save(&self, cfg: &RuntimeConfig) -> Result<(), ConfigError> {
        let text = self.serialise(cfg)?;
        std::fs::write(&self.path, text)?;
        Ok(())
    }

    /// Parse a config from a string (using this file's format).
    pub fn parse_str(&self, text: &str) -> Result<RuntimeConfig, ConfigError> {
        match self.format {
            ConfigFormat::Toml => Ok(toml::from_str(text)?),
            ConfigFormat::Json => Ok(serde_json::from_str(text)?),
        }
    }

    /// Serialise a config to string (using this file's format).
    pub fn serialise(&self, cfg: &RuntimeConfig) -> Result<String, ConfigError> {
        match self.format {
            ConfigFormat::Toml => {
                toml::to_string_pretty(cfg).map_err(|e| ConfigError::Toml(e.to_string()))
            }
            ConfigFormat::Json => {
                serde_json::to_string_pretty(cfg).map_err(|e| ConfigError::Json(e.to_string()))
            }
        }
    }

    fn detect_format(path: &Path) -> Result<ConfigFormat, ConfigError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => Ok(ConfigFormat::Toml),
            Some("json") => Ok(ConfigFormat::Json),
            Some(ext) => Err(ConfigError::Validation(format!(
                "unsupported config extension: .{ext}"
            ))),
            None => Err(ConfigError::Validation(
                "config file has no extension".into(),
            )),
        }
    }
}

// ── ConfigMerger ────────────────────────────────────────────────────────────

/// Merges configurations from multiple sources.
///
/// Priority (highest → lowest): environment variables → file → defaults.
///
/// **CPU reference**: merging is pure data transformation with no
/// hardware dependency.
#[derive(Debug, Clone)]
pub struct ConfigMerger {
    sources: Vec<ConfigSource>,
}

/// A labelled configuration source, ordered by descending priority.
#[derive(Debug, Clone)]
pub struct ConfigSource {
    /// Human-readable name for diagnostics.
    pub name: String,
    /// The configuration from this source.
    pub config: RuntimeConfig,
    /// Priority (higher wins).
    pub priority: u32,
}

impl ConfigMerger {
    /// Create a new, empty merger.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Register a source. Higher priority overrides lower.
    pub fn add_source(&mut self, name: impl Into<String>, config: RuntimeConfig, priority: u32) {
        self.sources.push(ConfigSource {
            name: name.into(),
            config,
            priority,
        });
    }

    /// Return a reference to all registered sources.
    pub fn sources(&self) -> &[ConfigSource] {
        &self.sources
    }

    /// Merge all registered sources into a single [`RuntimeConfig`].
    ///
    /// Fields from higher-priority sources override defaults; `extra` maps
    /// and feature flags are merged (union, higher priority wins on conflicts).
    pub fn merge(&self) -> RuntimeConfig {
        let mut sorted: Vec<&ConfigSource> = self.sources.iter().collect();
        sorted.sort_by_key(|s| s.priority);

        let mut result = RuntimeConfig::default();
        for src in &sorted {
            Self::apply_override(&mut result, &src.config);
        }
        result
    }

    /// Overlay `over` onto `base`, skipping fields that are at their default.
    fn apply_override(base: &mut RuntimeConfig, over: &RuntimeConfig) {
        let defaults = RuntimeConfig::default();
        if over.device_index != defaults.device_index {
            base.device_index = over.device_index;
        }
        if over.max_memory_bytes != defaults.max_memory_bytes {
            base.max_memory_bytes = over.max_memory_bytes;
        }
        if over.num_threads != defaults.num_threads {
            base.num_threads = over.num_threads;
        }
        if over.batch_size != defaults.batch_size {
            base.batch_size = over.batch_size;
        }
        if over.log_level != defaults.log_level {
            base.log_level = over.log_level.clone();
        }
        if over.enable_profiling != defaults.enable_profiling {
            base.enable_profiling = over.enable_profiling;
        }
        if over.model_path.is_some() {
            base.model_path.clone_from(&over.model_path);
        }
        for (k, v) in &over.extra {
            base.extra.insert(k.clone(), v.clone());
        }
        for (flag, &enabled) in &over.feature_flags.flags {
            base.feature_flags.set(flag.clone(), enabled);
        }
    }
}

impl Default for ConfigMerger {
    fn default() -> Self {
        Self::new()
    }
}

// ── ConfigValidator ─────────────────────────────────────────────────────────

/// Validates a [`RuntimeConfig`] for consistency and constraint satisfaction.
///
/// **CPU reference**: validation logic is pure and backend-independent.
#[derive(Debug, Clone)]
pub struct ConfigValidator {
    max_device_index: u32,
    max_batch_size: u32,
    allowed_log_levels: HashSet<String>,
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigValidator {
    /// Create a validator with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_device_index: 15,
            max_batch_size: 4096,
            allowed_log_levels: ["error", "warn", "info", "debug", "trace"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }

    /// Set the maximum allowed device index.
    pub fn with_max_device_index(mut self, max: u32) -> Self {
        self.max_device_index = max;
        self
    }

    /// Set the maximum allowed batch size.
    pub fn with_max_batch_size(mut self, max: u32) -> Self {
        self.max_batch_size = max;
        self
    }

    /// Validate a configuration, returning all violations found.
    pub fn validate(&self, cfg: &RuntimeConfig) -> Vec<String> {
        let mut errors = Vec::new();

        if cfg.device_index > self.max_device_index {
            errors.push(format!(
                "device_index {} exceeds max {}",
                cfg.device_index, self.max_device_index
            ));
        }
        if cfg.num_threads == 0 {
            errors.push("num_threads must be > 0".into());
        }
        if cfg.batch_size == 0 {
            errors.push("batch_size must be > 0".into());
        }
        if cfg.batch_size > self.max_batch_size {
            errors.push(format!(
                "batch_size {} exceeds max {}",
                cfg.batch_size, self.max_batch_size
            ));
        }
        if !self.allowed_log_levels.contains(&cfg.log_level) {
            errors.push(format!("invalid log_level: {}", cfg.log_level));
        }

        errors
    }

    /// Convenience: validate and return a [`ConfigError`] if any violations.
    pub fn validate_strict(&self, cfg: &RuntimeConfig) -> Result<(), ConfigError> {
        let errors = self.validate(cfg);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors.join("; ")))
        }
    }
}

// ── ConfigWatcher ───────────────────────────────────────────────────────────

/// Watches a configuration file for changes and supports hot-reload.
///
/// Uses timestamp-based polling rather than OS-level filesystem events to
/// keep the implementation portable and dependency-free.
///
/// **CPU reference**: polling is OS-agnostic; no GPU-specific dependencies.
#[derive(Debug)]
pub struct ConfigWatcher {
    file: ConfigFile,
    poll_interval: Duration,
    last_modified: Option<SystemTime>,
    last_config: Option<RuntimeConfig>,
    change_count: AtomicU64,
}

impl ConfigWatcher {
    /// Create a watcher for `path` that polls at `poll_interval`.
    pub fn new(path: impl Into<PathBuf>, poll_interval: Duration) -> Result<Self, ConfigError> {
        let file = ConfigFile::new(path)?;
        Ok(Self {
            file,
            poll_interval,
            last_modified: None,
            last_config: None,
            change_count: AtomicU64::new(0),
        })
    }

    /// Return the polling interval.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Return the number of detected changes since creation.
    pub fn change_count(&self) -> u64 {
        self.change_count.load(Ordering::Relaxed)
    }

    /// Check whether the file has been modified since the last poll.
    /// Returns `Some(config)` on change, `None` if unchanged / file missing.
    pub fn poll(&mut self) -> Result<Option<RuntimeConfig>, ConfigError> {
        let mtime = match std::fs::metadata(self.file.path()) {
            Ok(m) => m.modified().unwrap_or(UNIX_EPOCH),
            Err(_) => return Ok(None),
        };

        if self.last_modified == Some(mtime) {
            return Ok(None);
        }

        let cfg = self.file.load()?;
        self.last_modified = Some(mtime);
        self.last_config = Some(cfg.clone());
        self.change_count.fetch_add(1, Ordering::Relaxed);
        Ok(Some(cfg))
    }

    /// Return the most recently loaded configuration, if any.
    pub fn last_config(&self) -> Option<&RuntimeConfig> {
        self.last_config.as_ref()
    }
}

// ── FeatureFlags ────────────────────────────────────────────────────────────

/// Runtime feature-flag store.
///
/// Flags are string-keyed booleans, supporting bulk query and toggle.
///
/// **CPU reference**: feature flags are data-only; no hardware coupling.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FeatureFlags {
    flags: BTreeMap<String, bool>,
}

impl FeatureFlags {
    /// Create an empty flag set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a flag to the given value.
    pub fn set(&mut self, name: impl Into<String>, enabled: bool) {
        self.flags.insert(name.into(), enabled);
    }

    /// Get the value of a flag (`None` if not defined).
    pub fn get(&self, name: &str) -> Option<bool> {
        self.flags.get(name).copied()
    }

    /// Check whether a flag is enabled (defaults to `false` if undefined).
    pub fn is_enabled(&self, name: &str) -> bool {
        self.flags.get(name).copied().unwrap_or(false)
    }

    /// Remove a flag entirely.
    pub fn remove(&mut self, name: &str) -> Option<bool> {
        self.flags.remove(name)
    }

    /// Return all flag names.
    pub fn names(&self) -> Vec<&str> {
        self.flags.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of defined flags.
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// Return whether the flag set is empty.
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    /// Merge another flag set into this one (other wins on conflicts).
    pub fn merge(&mut self, other: &FeatureFlags) {
        for (k, &v) in &other.flags {
            self.flags.insert(k.clone(), v);
        }
    }

    /// Return an iterator over `(name, enabled)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, bool)> {
        self.flags.iter().map(|(k, &v)| (k.as_str(), v))
    }
}

// ── ConfigSnapshot ──────────────────────────────────────────────────────────

/// Immutable, timestamped snapshot of a [`RuntimeConfig`].
///
/// Designed for safe sharing across threads (clone-on-capture).
///
/// **CPU reference**: snapshot is purely data; no hardware dependency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    config: RuntimeConfig,
    captured_at_ms: u64,
    version: u64,
}

/// Monotonically increasing global version counter for snapshots.
static SNAPSHOT_VERSION: AtomicU64 = AtomicU64::new(1);

impl ConfigSnapshot {
    /// Capture a snapshot of the given configuration.
    pub fn capture(cfg: &RuntimeConfig) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            config: cfg.clone(),
            captured_at_ms: now,
            version: SNAPSHOT_VERSION.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Return a reference to the captured configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Return the capture timestamp in epoch milliseconds.
    pub fn captured_at_ms(&self) -> u64 {
        self.captured_at_ms
    }

    /// Return the version number of this snapshot.
    pub fn version(&self) -> u64 {
        self.version
    }
}

// ── ConfigDiff ──────────────────────────────────────────────────────────────

/// Represents a single field change between two configurations.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    /// Dotted field path (e.g. `"device_index"`).
    pub field: String,
    /// Stringified old value.
    pub old_value: String,
    /// Stringified new value.
    pub new_value: String,
}

/// Computes differences between two [`RuntimeConfig`]s.
///
/// **CPU reference**: diff is a pure comparison with no side effects.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigDiff {
    changes: Vec<FieldChange>,
}

impl ConfigDiff {
    /// Compute the diff from `old` to `new`.
    pub fn compute(old: &RuntimeConfig, new: &RuntimeConfig) -> Self {
        let mut changes = Vec::new();

        macro_rules! cmp_field {
            ($field:ident) => {
                if old.$field != new.$field {
                    changes.push(FieldChange {
                        field: stringify!($field).into(),
                        old_value: format!("{:?}", old.$field),
                        new_value: format!("{:?}", new.$field),
                    });
                }
            };
        }

        cmp_field!(device_index);
        cmp_field!(max_memory_bytes);
        cmp_field!(num_threads);
        cmp_field!(batch_size);
        cmp_field!(log_level);
        cmp_field!(enable_profiling);
        cmp_field!(model_path);

        // Extra map changes
        let all_extra_keys: HashSet<_> = old.extra.keys().chain(new.extra.keys()).collect();
        for key in all_extra_keys {
            let ov = old.extra.get(key).map(|s| s.as_str()).unwrap_or("");
            let nv = new.extra.get(key).map(|s| s.as_str()).unwrap_or("");
            if ov != nv {
                changes.push(FieldChange {
                    field: format!("extra.{key}"),
                    old_value: ov.into(),
                    new_value: nv.into(),
                });
            }
        }

        // Feature flag changes
        let all_flag_keys: HashSet<_> = old
            .feature_flags
            .names()
            .into_iter()
            .chain(new.feature_flags.names())
            .collect();
        for key in all_flag_keys {
            let ov = old.feature_flags.is_enabled(key);
            let nv = new.feature_flags.is_enabled(key);
            if ov != nv {
                changes.push(FieldChange {
                    field: format!("feature_flags.{key}"),
                    old_value: ov.to_string(),
                    new_value: nv.to_string(),
                });
            }
        }

        Self { changes }
    }

    /// Return the list of changes.
    pub fn changes(&self) -> &[FieldChange] {
        &self.changes
    }

    /// Return `true` if there are no differences.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Return the number of changed fields.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Check whether a specific field was changed.
    pub fn has_change(&self, field: &str) -> bool {
        self.changes.iter().any(|c| c.field == field)
    }
}

impl fmt::Display for ConfigDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "(no changes)");
        }
        for (i, c) in self.changes.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}: {} -> {}", c.field, c.old_value, c.new_value)?;
        }
        Ok(())
    }
}

// ── RuntimeConfigEngine ─────────────────────────────────────────────────────

/// Unified configuration management engine.
///
/// Combines environment parsing, file loading, merging, validation, and
/// snapshot capture into a single entry-point.
///
/// **CPU reference**: the engine orchestrates pure-data transformations;
/// it works identically on CPU-only builds.
#[derive(Debug)]
pub struct RuntimeConfigEngine {
    env_parser: EnvParser,
    validator: ConfigValidator,
    file_path: Option<PathBuf>,
    current: RuntimeConfig,
    history: Vec<ConfigSnapshot>,
    max_history: usize,
}

impl RuntimeConfigEngine {
    /// Create an engine with default settings.
    pub fn new() -> Self {
        Self {
            env_parser: EnvParser::new(),
            validator: ConfigValidator::new(),
            file_path: None,
            current: RuntimeConfig::default(),
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Set the configuration file path.
    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Set the env parser.
    pub fn with_env_parser(mut self, parser: EnvParser) -> Self {
        self.env_parser = parser;
        self
    }

    /// Set the validator.
    pub fn with_validator(mut self, validator: ConfigValidator) -> Self {
        self.validator = validator;
        self
    }

    /// Set the maximum history length.
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Return the current configuration.
    pub fn current(&self) -> &RuntimeConfig {
        &self.current
    }

    /// Return the snapshot history.
    pub fn history(&self) -> &[ConfigSnapshot] {
        &self.history
    }

    /// Load configuration by merging defaults, file (if set), and the
    /// given environment variable map.
    pub fn load(
        &mut self,
        env_vars: &HashMap<String, String>,
    ) -> Result<&RuntimeConfig, ConfigError> {
        let mut merger = ConfigMerger::new();

        // Priority 0: defaults
        merger.add_source("defaults", RuntimeConfig::default(), 0);

        // Priority 1: file
        if let Some(ref path) = self.file_path {
            if path.exists() {
                let file = ConfigFile::new(path)?;
                let file_cfg = file.load()?;
                merger.add_source("file", file_cfg, 1);
            }
        }

        // Priority 2: environment
        let env_cfg = self.env_parser.parse_from_map(env_vars)?;
        merger.add_source("env", env_cfg, 2);

        let merged = merger.merge();
        self.validator.validate_strict(&merged)?;

        // Record snapshot before updating current.
        let snap = ConfigSnapshot::capture(&merged);
        self.history.push(snap);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        self.current = merged;
        Ok(&self.current)
    }

    /// Replace the current config and record a snapshot.
    pub fn update(&mut self, cfg: RuntimeConfig) -> Result<(), ConfigError> {
        self.validator.validate_strict(&cfg)?;
        let snap = ConfigSnapshot::capture(&cfg);
        self.history.push(snap);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        self.current = cfg;
        Ok(())
    }

    /// Compute the diff between the current config and a proposed one.
    pub fn diff_with(&self, other: &RuntimeConfig) -> ConfigDiff {
        ConfigDiff::compute(&self.current, other)
    }

    /// Take a snapshot of the current configuration.
    pub fn snapshot(&self) -> ConfigSnapshot {
        ConfigSnapshot::capture(&self.current)
    }
}

impl Default for RuntimeConfigEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn tmp_toml(cfg: &RuntimeConfig) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .unwrap();
        let text = toml::to_string_pretty(cfg).unwrap();
        f.write_all(text.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn tmp_json(cfg: &RuntimeConfig) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".json")
            .tempfile()
            .unwrap();
        let text = serde_json::to_string_pretty(cfg).unwrap();
        f.write_all(text.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    // ── RuntimeConfig tests ─────────────────────────────────────────────

    #[test]
    fn test_runtime_config_default() {
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.device_index, 0);
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.log_level, "info");
        assert!(!cfg.enable_profiling);
        assert!(cfg.model_path.is_none());
        assert!(cfg.extra.is_empty());
    }

    #[test]
    fn test_runtime_config_new_equals_default() {
        assert_eq!(RuntimeConfig::new(), RuntimeConfig::default());
    }

    #[test]
    fn test_runtime_config_accessors() {
        let mut cfg = RuntimeConfig::default();
        cfg.log_level = "debug".into();
        cfg.enable_profiling = true;
        assert_eq!(cfg.log_level(), "debug");
        assert!(cfg.profiling_enabled());
    }

    #[test]
    fn test_runtime_config_snapshot_convenience() {
        let cfg = RuntimeConfig::default();
        let snap = cfg.snapshot();
        assert_eq!(snap.config(), &cfg);
    }

    #[test]
    fn test_runtime_config_clone_eq() {
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 3;
        cfg.extra.insert("key".into(), "val".into());
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    #[test]
    fn test_runtime_config_serde_roundtrip_json() {
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 7;
        cfg.model_path = Some("/tmp/model.gguf".into());
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: RuntimeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn test_runtime_config_serde_roundtrip_toml() {
        let mut cfg = RuntimeConfig::default();
        cfg.batch_size = 16;
        cfg.extra.insert("custom".into(), "value".into());
        let toml_str = toml::to_string(&cfg).unwrap();
        let parsed: RuntimeConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(cfg, parsed);
    }

    // ── ConfigError tests ───────────────────────────────────────────────

    #[test]
    fn test_error_display_io() {
        let err = ConfigError::Io(std::io::Error::other("disk fail"));
        assert!(err.to_string().contains("disk fail"));
    }

    #[test]
    fn test_error_display_toml() {
        let err = ConfigError::Toml("bad toml".into());
        assert!(err.to_string().contains("bad toml"));
    }

    #[test]
    fn test_error_display_json() {
        let err = ConfigError::Json("bad json".into());
        assert!(err.to_string().contains("bad json"));
    }

    #[test]
    fn test_error_display_validation() {
        let err = ConfigError::Validation("too big".into());
        assert!(err.to_string().contains("too big"));
    }

    #[test]
    fn test_error_display_env_parse() {
        let err = ConfigError::EnvParse("NaN".into());
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn test_error_display_missing_key() {
        let err = ConfigError::MissingKey("foo".into());
        assert!(err.to_string().contains("foo"));
    }

    #[test]
    fn test_error_source_io() {
        let err = ConfigError::Io(std::io::Error::other("x"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_error_source_non_io() {
        let err = ConfigError::Toml("x".into());
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::other("io");
        let err: ConfigError = io_err.into();
        assert!(matches!(err, ConfigError::Io(_)));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_err = serde_json::from_str::<String>("!").unwrap_err();
        let err: ConfigError = json_err.into();
        assert!(matches!(err, ConfigError::Json(_)));
    }

    #[test]
    fn test_error_from_toml() {
        let toml_err = toml::from_str::<RuntimeConfig>("[[invalid").unwrap_err();
        let err: ConfigError = toml_err.into();
        assert!(matches!(err, ConfigError::Toml(_)));
    }

    // ── EnvParser tests ─────────────────────────────────────────────────

    #[test]
    fn test_env_parser_default_prefix() {
        let parser = EnvParser::new();
        assert_eq!(parser.prefix(), "BITNET_");
    }

    #[test]
    fn test_env_parser_custom_prefix() {
        let parser = EnvParser::with_prefix("MYAPP_");
        assert_eq!(parser.prefix(), "MYAPP_");
    }

    #[test]
    fn test_env_parser_empty_map() {
        let parser = EnvParser::new();
        let cfg = parser.parse_from_map(&HashMap::new()).unwrap();
        assert_eq!(cfg, RuntimeConfig::default());
    }

    #[test]
    fn test_env_parser_device_index() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_DEVICE_INDEX", "3")]))
            .unwrap();
        assert_eq!(cfg.device_index, 3);
    }

    #[test]
    fn test_env_parser_invalid_device_index() {
        let parser = EnvParser::new();
        let result = parser.parse_from_map(&env(&[("BITNET_DEVICE_INDEX", "abc")]));
        assert!(matches!(result, Err(ConfigError::EnvParse(_))));
    }

    #[test]
    fn test_env_parser_num_threads() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_NUM_THREADS", "16")]))
            .unwrap();
        assert_eq!(cfg.num_threads, 16);
    }

    #[test]
    fn test_env_parser_batch_size() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_BATCH_SIZE", "32")]))
            .unwrap();
        assert_eq!(cfg.batch_size, 32);
    }

    #[test]
    fn test_env_parser_log_level() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_LOG_LEVEL", "DEBUG")]))
            .unwrap();
        assert_eq!(cfg.log_level, "debug");
    }

    #[test]
    fn test_env_parser_profiling_true() {
        let parser = EnvParser::new();
        for val in &["1", "true", "yes"] {
            let cfg = parser
                .parse_from_map(&env(&[("BITNET_ENABLE_PROFILING", val)]))
                .unwrap();
            assert!(cfg.enable_profiling, "failed for {val}");
        }
    }

    #[test]
    fn test_env_parser_profiling_false() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_ENABLE_PROFILING", "0")]))
            .unwrap();
        assert!(!cfg.enable_profiling);
    }

    #[test]
    fn test_env_parser_model_path() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MODEL_PATH", "/models/x.gguf")]))
            .unwrap();
        assert_eq!(cfg.model_path.as_deref(), Some("/models/x.gguf"));
    }

    #[test]
    fn test_env_parser_memory_plain() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MAX_MEMORY", "1024")]))
            .unwrap();
        assert_eq!(cfg.max_memory_bytes, 1024);
    }

    #[test]
    fn test_env_parser_memory_kilobytes() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MAX_MEMORY", "4K")]))
            .unwrap();
        assert_eq!(cfg.max_memory_bytes, 4 * 1024);
    }

    #[test]
    fn test_env_parser_memory_megabytes() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MAX_MEMORY", "2M")]))
            .unwrap();
        assert_eq!(cfg.max_memory_bytes, 2 * 1024 * 1024);
    }

    #[test]
    fn test_env_parser_memory_gigabytes() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MAX_MEMORY", "8G")]))
            .unwrap();
        assert_eq!(cfg.max_memory_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_env_parser_memory_terabytes() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MAX_MEMORY", "1T")]))
            .unwrap();
        assert_eq!(cfg.max_memory_bytes, 1024 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_env_parser_memory_empty() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[("BITNET_MAX_MEMORY", "")]))
            .unwrap();
        assert_eq!(cfg.max_memory_bytes, 0);
    }

    #[test]
    fn test_env_parser_memory_invalid() {
        let parser = EnvParser::new();
        let result = parser.parse_from_map(&env(&[("BITNET_MAX_MEMORY", "abc")]));
        assert!(matches!(result, Err(ConfigError::EnvParse(_))));
    }

    #[test]
    fn test_env_parser_feature_flags() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[
                ("BITNET_FEATURE_CUDA", "1"),
                ("BITNET_FEATURE_PROFILING", "false"),
            ]))
            .unwrap();
        assert!(cfg.feature_flags.is_enabled("cuda"));
        assert!(!cfg.feature_flags.is_enabled("profiling"));
    }

    #[test]
    fn test_env_parser_custom_prefix_feature() {
        let parser = EnvParser::with_prefix("MY_");
        let cfg = parser
            .parse_from_map(&env(&[("MY_DEVICE_INDEX", "5"), ("MY_FEATURE_X", "yes")]))
            .unwrap();
        assert_eq!(cfg.device_index, 5);
        assert!(cfg.feature_flags.is_enabled("x"));
    }

    #[test]
    fn test_env_parser_all_fields() {
        let parser = EnvParser::new();
        let cfg = parser
            .parse_from_map(&env(&[
                ("BITNET_DEVICE_INDEX", "2"),
                ("BITNET_MAX_MEMORY", "4G"),
                ("BITNET_NUM_THREADS", "8"),
                ("BITNET_BATCH_SIZE", "64"),
                ("BITNET_LOG_LEVEL", "trace"),
                ("BITNET_ENABLE_PROFILING", "true"),
                ("BITNET_MODEL_PATH", "/m.gguf"),
            ]))
            .unwrap();
        assert_eq!(cfg.device_index, 2);
        assert_eq!(cfg.max_memory_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(cfg.num_threads, 8);
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.log_level, "trace");
        assert!(cfg.enable_profiling);
        assert_eq!(cfg.model_path.as_deref(), Some("/m.gguf"));
    }

    // ── ConfigFile tests ────────────────────────────────────────────────

    #[test]
    fn test_config_file_detect_toml() {
        let f = ConfigFile::new("/tmp/test.toml").unwrap();
        assert_eq!(f.format(), ConfigFormat::Toml);
    }

    #[test]
    fn test_config_file_detect_json() {
        let f = ConfigFile::new("/tmp/test.json").unwrap();
        assert_eq!(f.format(), ConfigFormat::Json);
    }

    #[test]
    fn test_config_file_unknown_extension() {
        let result = ConfigFile::new("/tmp/test.yaml");
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    #[test]
    fn test_config_file_no_extension() {
        let result = ConfigFile::new("/tmp/config");
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    #[test]
    fn test_config_file_with_format() {
        let f = ConfigFile::with_format("/tmp/x", ConfigFormat::Json);
        assert_eq!(f.format(), ConfigFormat::Json);
    }

    #[test]
    fn test_config_file_roundtrip_toml() {
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 5;
        cfg.batch_size = 8;
        let f = tmp_toml(&cfg);
        let loader = ConfigFile::new(f.path()).unwrap();
        let loaded = loader.load().unwrap();
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn test_config_file_roundtrip_json() {
        let mut cfg = RuntimeConfig::default();
        cfg.num_threads = 12;
        cfg.model_path = Some("model.gguf".into());
        let f = tmp_json(&cfg);
        let loader = ConfigFile::new(f.path()).unwrap();
        let loaded = loader.load().unwrap();
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn test_config_file_save_and_reload_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.toml");
        let file = ConfigFile::with_format(&path, ConfigFormat::Toml);
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 9;
        file.save(&cfg).unwrap();
        let loaded = file.load().unwrap();
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn test_config_file_save_and_reload_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.json");
        let file = ConfigFile::with_format(&path, ConfigFormat::Json);
        let cfg = RuntimeConfig::default();
        file.save(&cfg).unwrap();
        let loaded = file.load().unwrap();
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn test_config_file_parse_str_toml() {
        let cfg = RuntimeConfig::default();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let file = ConfigFile::with_format("x.toml", ConfigFormat::Toml);
        let parsed = file.parse_str(&text).unwrap();
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn test_config_file_parse_str_json() {
        let cfg = RuntimeConfig::default();
        let text = serde_json::to_string_pretty(&cfg).unwrap();
        let file = ConfigFile::with_format("x.json", ConfigFormat::Json);
        let parsed = file.parse_str(&text).unwrap();
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn test_config_file_path_accessor() {
        let f = ConfigFile::with_format("/a/b/c.toml", ConfigFormat::Toml);
        assert_eq!(f.path(), Path::new("/a/b/c.toml"));
    }

    #[test]
    fn test_config_file_serialise_toml() {
        let cfg = RuntimeConfig::default();
        let file = ConfigFile::with_format("x.toml", ConfigFormat::Toml);
        let text = file.serialise(&cfg).unwrap();
        assert!(text.contains("device_index"));
    }

    #[test]
    fn test_config_file_serialise_json() {
        let cfg = RuntimeConfig::default();
        let file = ConfigFile::with_format("x.json", ConfigFormat::Json);
        let text = file.serialise(&cfg).unwrap();
        assert!(text.contains("device_index"));
    }

    #[test]
    fn test_config_file_load_missing() {
        let file = ConfigFile::with_format("/nonexistent/cfg.toml", ConfigFormat::Toml);
        assert!(matches!(file.load(), Err(ConfigError::Io(_))));
    }

    // ── ConfigMerger tests ──────────────────────────────────────────────

    #[test]
    fn test_merger_empty() {
        let merger = ConfigMerger::new();
        let result = merger.merge();
        assert_eq!(result, RuntimeConfig::default());
    }

    #[test]
    fn test_merger_single_source() {
        let mut merger = ConfigMerger::new();
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 3;
        merger.add_source("env", cfg.clone(), 1);
        let result = merger.merge();
        assert_eq!(result.device_index, 3);
    }

    #[test]
    fn test_merger_priority_ordering() {
        let mut merger = ConfigMerger::new();
        let mut low = RuntimeConfig::default();
        low.device_index = 1;
        let mut high = RuntimeConfig::default();
        high.device_index = 9;
        merger.add_source("low", low, 0);
        merger.add_source("high", high, 10);
        let result = merger.merge();
        assert_eq!(result.device_index, 9);
    }

    #[test]
    fn test_merger_extra_maps_union() {
        let mut merger = ConfigMerger::new();
        let mut a = RuntimeConfig::default();
        a.extra.insert("key_a".into(), "a".into());
        let mut b = RuntimeConfig::default();
        b.extra.insert("key_b".into(), "b".into());
        merger.add_source("a", a, 0);
        merger.add_source("b", b, 1);
        let result = merger.merge();
        assert_eq!(result.extra.get("key_a").unwrap(), "a");
        assert_eq!(result.extra.get("key_b").unwrap(), "b");
    }

    #[test]
    fn test_merger_extra_maps_conflict() {
        let mut merger = ConfigMerger::new();
        let mut a = RuntimeConfig::default();
        a.extra.insert("k".into(), "low".into());
        let mut b = RuntimeConfig::default();
        b.extra.insert("k".into(), "high".into());
        merger.add_source("a", a, 0);
        merger.add_source("b", b, 1);
        let result = merger.merge();
        assert_eq!(result.extra.get("k").unwrap(), "high");
    }

    #[test]
    fn test_merger_feature_flags_merge() {
        let mut merger = ConfigMerger::new();
        let mut a = RuntimeConfig::default();
        a.feature_flags.set("cuda", true);
        let mut b = RuntimeConfig::default();
        b.feature_flags.set("vulkan", true);
        merger.add_source("a", a, 0);
        merger.add_source("b", b, 1);
        let result = merger.merge();
        assert!(result.feature_flags.is_enabled("cuda"));
        assert!(result.feature_flags.is_enabled("vulkan"));
    }

    #[test]
    fn test_merger_sources_accessor() {
        let mut merger = ConfigMerger::new();
        merger.add_source("a", RuntimeConfig::default(), 0);
        merger.add_source("b", RuntimeConfig::default(), 1);
        assert_eq!(merger.sources().len(), 2);
    }

    #[test]
    fn test_merger_defaults_preserved() {
        let mut merger = ConfigMerger::new();
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 5;
        // num_threads stays at default 4
        merger.add_source("partial", cfg, 1);
        let result = merger.merge();
        assert_eq!(result.device_index, 5);
        assert_eq!(result.num_threads, 4);
    }

    #[test]
    fn test_merger_model_path_override() {
        let mut merger = ConfigMerger::new();
        let mut cfg = RuntimeConfig::default();
        cfg.model_path = Some("/new/path.gguf".into());
        merger.add_source("env", cfg, 2);
        let result = merger.merge();
        assert_eq!(result.model_path.as_deref(), Some("/new/path.gguf"));
    }

    // ── ConfigValidator tests ───────────────────────────────────────────

    #[test]
    fn test_validator_default_passes() {
        let v = ConfigValidator::new();
        let cfg = RuntimeConfig::default();
        assert!(v.validate(&cfg).is_empty());
    }

    #[test]
    fn test_validator_device_index_too_high() {
        let v = ConfigValidator::new().with_max_device_index(3);
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 4;
        let errs = v.validate(&cfg);
        assert!(errs.iter().any(|e| e.contains("device_index")));
    }

    #[test]
    fn test_validator_zero_threads() {
        let v = ConfigValidator::new();
        let mut cfg = RuntimeConfig::default();
        cfg.num_threads = 0;
        let errs = v.validate(&cfg);
        assert!(errs.iter().any(|e| e.contains("num_threads")));
    }

    #[test]
    fn test_validator_zero_batch_size() {
        let v = ConfigValidator::new();
        let mut cfg = RuntimeConfig::default();
        cfg.batch_size = 0;
        let errs = v.validate(&cfg);
        assert!(errs.iter().any(|e| e.contains("batch_size")));
    }

    #[test]
    fn test_validator_batch_size_too_high() {
        let v = ConfigValidator::new().with_max_batch_size(64);
        let mut cfg = RuntimeConfig::default();
        cfg.batch_size = 128;
        let errs = v.validate(&cfg);
        assert!(errs.iter().any(|e| e.contains("batch_size")));
    }

    #[test]
    fn test_validator_invalid_log_level() {
        let v = ConfigValidator::new();
        let mut cfg = RuntimeConfig::default();
        cfg.log_level = "verbose".into();
        let errs = v.validate(&cfg);
        assert!(errs.iter().any(|e| e.contains("log_level")));
    }

    #[test]
    fn test_validator_multiple_errors() {
        let v = ConfigValidator::new();
        let mut cfg = RuntimeConfig::default();
        cfg.num_threads = 0;
        cfg.batch_size = 0;
        cfg.log_level = "xxx".into();
        let errs = v.validate(&cfg);
        assert!(errs.len() >= 3);
    }

    #[test]
    fn test_validator_strict_ok() {
        let v = ConfigValidator::new();
        let cfg = RuntimeConfig::default();
        assert!(v.validate_strict(&cfg).is_ok());
    }

    #[test]
    fn test_validator_strict_error() {
        let v = ConfigValidator::new();
        let mut cfg = RuntimeConfig::default();
        cfg.num_threads = 0;
        assert!(matches!(
            v.validate_strict(&cfg),
            Err(ConfigError::Validation(_))
        ));
    }

    #[test]
    fn test_validator_all_valid_log_levels() {
        let v = ConfigValidator::new();
        for level in &["error", "warn", "info", "debug", "trace"] {
            let mut cfg = RuntimeConfig::default();
            cfg.log_level = level.to_string();
            assert!(v.validate(&cfg).is_empty(), "failed for {level}");
        }
    }

    // ── ConfigWatcher tests ─────────────────────────────────────────────

    #[test]
    fn test_watcher_creation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("watch.toml");
        std::fs::write(&path, toml::to_string(&RuntimeConfig::default()).unwrap()).unwrap();
        let watcher = ConfigWatcher::new(&path, Duration::from_millis(100)).unwrap();
        assert_eq!(watcher.poll_interval(), Duration::from_millis(100));
        assert_eq!(watcher.change_count(), 0);
    }

    #[test]
    fn test_watcher_poll_detects_initial() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.toml");
        let cfg = RuntimeConfig::default();
        std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
        let mut watcher = ConfigWatcher::new(&path, Duration::from_secs(1)).unwrap();
        let result = watcher.poll().unwrap();
        assert!(result.is_some());
        assert_eq!(watcher.change_count(), 1);
    }

    #[test]
    fn test_watcher_poll_no_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.toml");
        std::fs::write(&path, toml::to_string(&RuntimeConfig::default()).unwrap()).unwrap();
        let mut watcher = ConfigWatcher::new(&path, Duration::from_secs(1)).unwrap();
        watcher.poll().unwrap(); // first
        let second = watcher.poll().unwrap();
        assert!(second.is_none());
        assert_eq!(watcher.change_count(), 1);
    }

    #[test]
    fn test_watcher_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.toml");
        let mut watcher = ConfigWatcher::new(&path, Duration::from_secs(1)).unwrap();
        let result = watcher.poll().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_watcher_last_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.toml");
        let mut cfg = RuntimeConfig::default();
        cfg.device_index = 7;
        std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
        let mut watcher = ConfigWatcher::new(&path, Duration::from_secs(1)).unwrap();
        assert!(watcher.last_config().is_none());
        watcher.poll().unwrap();
        assert_eq!(watcher.last_config().unwrap().device_index, 7);
    }

    // ── FeatureFlags tests ──────────────────────────────────────────────

    #[test]
    fn test_flags_empty() {
        let flags = FeatureFlags::new();
        assert!(flags.is_empty());
        assert_eq!(flags.len(), 0);
    }

    #[test]
    fn test_flags_set_get() {
        let mut flags = FeatureFlags::new();
        flags.set("cuda", true);
        assert_eq!(flags.get("cuda"), Some(true));
        assert!(flags.is_enabled("cuda"));
    }

    #[test]
    fn test_flags_undefined_default_false() {
        let flags = FeatureFlags::new();
        assert!(!flags.is_enabled("nonexistent"));
        assert_eq!(flags.get("nonexistent"), None);
    }

    #[test]
    fn test_flags_remove() {
        let mut flags = FeatureFlags::new();
        flags.set("x", true);
        assert_eq!(flags.remove("x"), Some(true));
        assert!(flags.is_empty());
    }

    #[test]
    fn test_flags_names() {
        let mut flags = FeatureFlags::new();
        flags.set("a", true);
        flags.set("b", false);
        let names = flags.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn test_flags_merge() {
        let mut a = FeatureFlags::new();
        a.set("x", true);
        a.set("y", false);
        let mut b = FeatureFlags::new();
        b.set("y", true);
        b.set("z", true);
        a.merge(&b);
        assert!(a.is_enabled("x"));
        assert!(a.is_enabled("y")); // overridden
        assert!(a.is_enabled("z"));
    }

    #[test]
    fn test_flags_iter() {
        let mut flags = FeatureFlags::new();
        flags.set("a", true);
        flags.set("b", false);
        let collected: Vec<_> = flags.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_flags_clone_eq() {
        let mut flags = FeatureFlags::new();
        flags.set("cuda", true);
        let cloned = flags.clone();
        assert_eq!(flags, cloned);
    }

    #[test]
    fn test_flags_serde_roundtrip() {
        let mut flags = FeatureFlags::new();
        flags.set("avx2", true);
        flags.set("neon", false);
        let json = serde_json::to_string(&flags).unwrap();
        let parsed: FeatureFlags = serde_json::from_str(&json).unwrap();
        assert_eq!(flags, parsed);
    }

    // ── ConfigSnapshot tests ────────────────────────────────────────────

    #[test]
    fn test_snapshot_capture() {
        let cfg = RuntimeConfig::default();
        let snap = ConfigSnapshot::capture(&cfg);
        assert_eq!(snap.config(), &cfg);
        assert!(snap.captured_at_ms() > 0);
        assert!(snap.version() > 0);
    }

    #[test]
    fn test_snapshot_versions_increase() {
        let cfg = RuntimeConfig::default();
        let s1 = ConfigSnapshot::capture(&cfg);
        let s2 = ConfigSnapshot::capture(&cfg);
        assert!(s2.version() > s1.version());
    }

    #[test]
    fn test_snapshot_clone_eq() {
        let cfg = RuntimeConfig::default();
        let snap = ConfigSnapshot::capture(&cfg);
        let cloned = snap.clone();
        assert_eq!(snap, cloned);
    }

    #[test]
    fn test_snapshot_serde_roundtrip() {
        let cfg = RuntimeConfig::default();
        let snap = ConfigSnapshot::capture(&cfg);
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: ConfigSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap, parsed);
    }

    // ── ConfigDiff tests ────────────────────────────────────────────────

    #[test]
    fn test_diff_identical() {
        let a = RuntimeConfig::default();
        let diff = ConfigDiff::compute(&a, &a);
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn test_diff_device_index() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.device_index = 7;
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("device_index"));
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn test_diff_multiple_fields() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.device_index = 3;
        b.num_threads = 16;
        b.log_level = "debug".into();
        let diff = ConfigDiff::compute(&a, &b);
        assert_eq!(diff.len(), 3);
        assert!(diff.has_change("device_index"));
        assert!(diff.has_change("num_threads"));
        assert!(diff.has_change("log_level"));
    }

    #[test]
    fn test_diff_extra_map() {
        let mut a = RuntimeConfig::default();
        a.extra.insert("k".into(), "old".into());
        let mut b = a.clone();
        b.extra.insert("k".into(), "new".into());
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("extra.k"));
    }

    #[test]
    fn test_diff_extra_added() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.extra.insert("newkey".into(), "val".into());
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("extra.newkey"));
    }

    #[test]
    fn test_diff_extra_removed() {
        let mut a = RuntimeConfig::default();
        a.extra.insert("gone".into(), "val".into());
        let b = RuntimeConfig::default();
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("extra.gone"));
    }

    #[test]
    fn test_diff_feature_flags() {
        let mut a = RuntimeConfig::default();
        a.feature_flags.set("cuda", false);
        let mut b = a.clone();
        b.feature_flags.set("cuda", true);
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("feature_flags.cuda"));
    }

    #[test]
    fn test_diff_display_empty() {
        let a = RuntimeConfig::default();
        let diff = ConfigDiff::compute(&a, &a);
        assert_eq!(diff.to_string(), "(no changes)");
    }

    #[test]
    fn test_diff_display_non_empty() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.device_index = 1;
        let diff = ConfigDiff::compute(&a, &b);
        let s = diff.to_string();
        assert!(s.contains("device_index"));
    }

    #[test]
    fn test_diff_changes_accessor() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.batch_size = 99;
        let diff = ConfigDiff::compute(&a, &b);
        assert_eq!(diff.changes().len(), 1);
        assert_eq!(diff.changes()[0].field, "batch_size");
    }

    #[test]
    fn test_diff_model_path_change() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.model_path = Some("new.gguf".into());
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("model_path"));
    }

    #[test]
    fn test_diff_profiling_change() {
        let a = RuntimeConfig::default();
        let mut b = a.clone();
        b.enable_profiling = true;
        let diff = ConfigDiff::compute(&a, &b);
        assert!(diff.has_change("enable_profiling"));
    }

    // ── RuntimeConfigEngine tests ───────────────────────────────────────

    #[test]
    fn test_engine_default() {
        let engine = RuntimeConfigEngine::new();
        assert_eq!(engine.current(), &RuntimeConfig::default());
        assert!(engine.history().is_empty());
    }

    #[test]
    fn test_engine_load_defaults_only() {
        let mut engine = RuntimeConfigEngine::new();
        engine.load(&HashMap::new()).unwrap();
        assert_eq!(engine.current(), &RuntimeConfig::default());
        assert_eq!(engine.history().len(), 1);
    }

    #[test]
    fn test_engine_load_with_env() {
        let mut engine = RuntimeConfigEngine::new();
        let vars = env(&[("BITNET_DEVICE_INDEX", "5"), ("BITNET_BATCH_SIZE", "8")]);
        engine.load(&vars).unwrap();
        assert_eq!(engine.current().device_index, 5);
        assert_eq!(engine.current().batch_size, 8);
    }

    #[test]
    fn test_engine_load_with_file() {
        let mut cfg = RuntimeConfig::default();
        cfg.num_threads = 32;
        let f = tmp_toml(&cfg);
        let mut engine = RuntimeConfigEngine::new().with_file(f.path());
        engine.load(&HashMap::new()).unwrap();
        assert_eq!(engine.current().num_threads, 32);
    }

    #[test]
    fn test_engine_env_overrides_file() {
        let mut file_cfg = RuntimeConfig::default();
        file_cfg.device_index = 1;
        let f = tmp_toml(&file_cfg);
        let mut engine = RuntimeConfigEngine::new().with_file(f.path());
        let vars = env(&[("BITNET_DEVICE_INDEX", "9")]);
        engine.load(&vars).unwrap();
        assert_eq!(engine.current().device_index, 9);
    }

    #[test]
    fn test_engine_update() {
        let mut engine = RuntimeConfigEngine::new();
        let mut cfg = RuntimeConfig::default();
        cfg.batch_size = 64;
        engine.update(cfg.clone()).unwrap();
        assert_eq!(engine.current().batch_size, 64);
        assert_eq!(engine.history().len(), 1);
    }

    #[test]
    fn test_engine_update_invalid() {
        let mut engine = RuntimeConfigEngine::new();
        let mut cfg = RuntimeConfig::default();
        cfg.num_threads = 0;
        assert!(engine.update(cfg).is_err());
    }

    #[test]
    fn test_engine_diff_with() {
        let mut engine = RuntimeConfigEngine::new();
        engine.load(&HashMap::new()).unwrap();
        let mut other = RuntimeConfig::default();
        other.device_index = 5;
        let diff = engine.diff_with(&other);
        assert!(diff.has_change("device_index"));
    }

    #[test]
    fn test_engine_snapshot() {
        let engine = RuntimeConfigEngine::new();
        let snap = engine.snapshot();
        assert_eq!(snap.config(), &RuntimeConfig::default());
    }

    #[test]
    fn test_engine_history_limit() {
        let mut engine = RuntimeConfigEngine::new().with_max_history(3);
        for i in 0..5 {
            let mut cfg = RuntimeConfig::default();
            cfg.device_index = i;
            engine.update(cfg).unwrap();
        }
        assert_eq!(engine.history().len(), 3);
    }

    #[test]
    fn test_engine_with_validator() {
        let v = ConfigValidator::new().with_max_device_index(2);
        let mut engine = RuntimeConfigEngine::new().with_validator(v);
        let vars = env(&[("BITNET_DEVICE_INDEX", "10")]);
        assert!(engine.load(&vars).is_err());
    }

    #[test]
    fn test_engine_with_env_parser() {
        let parser = EnvParser::with_prefix("MY_");
        let mut engine = RuntimeConfigEngine::new().with_env_parser(parser);
        let vars = env(&[("MY_BATCH_SIZE", "42")]);
        engine.load(&vars).unwrap();
        assert_eq!(engine.current().batch_size, 42);
    }

    #[test]
    fn test_engine_missing_file_ok() {
        let mut engine =
            RuntimeConfigEngine::new().with_file("/nonexistent/path/config.toml");
        // Missing file is not an error; it's just skipped.
        engine.load(&HashMap::new()).unwrap();
        assert_eq!(engine.current(), &RuntimeConfig::default());
    }

    #[test]
    fn test_engine_load_validation_blocks_bad_env() {
        let mut engine = RuntimeConfigEngine::new();
        let vars = env(&[("BITNET_LOG_LEVEL", "banana")]);
        assert!(engine.load(&vars).is_err());
    }
}
