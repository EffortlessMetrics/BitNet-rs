//! Module stub - implementation pending merge from feature branch
//! Configuration management with multi-source merging and live reload.
//!
//! Provides a complete configuration pipeline: discover sources, merge with
//! priority ordering, validate against a schema, watch for file changes,
//! resolve secret references, and manage named profiles.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// ConfigValue
// ---------------------------------------------------------------------------

/// A dynamically-typed configuration value (JSON-like).
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<Self>),
    Map(HashMap<String, Self>),
}

impl ConfigValue {
    /// Returns the type name of this value for diagnostics.
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::String(_) => "string",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Bool(_) => "bool",
            Self::Array(_) => "array",
            Self::Map(_) => "map",
        }
    }

    /// Try to interpret as a string.
    pub const fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret as an integer.
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to interpret as a float.
    pub const fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to interpret as a bool.
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl std::fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Self::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{k}\": {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigSource
// ---------------------------------------------------------------------------

/// Origin of configuration data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    /// Configuration loaded from a file at the given path.
    File(PathBuf),
    /// Configuration loaded from environment variables.
    Env,
    /// Hard-coded default values.
    Defaults,
    /// Values supplied on the command line.
    CommandLine,
    /// Configuration fetched from a remote URL.
    Remote(String),
}

impl std::fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(p) => write!(f, "file:{}", p.display()),
            Self::Env => write!(f, "env"),
            Self::Defaults => write!(f, "defaults"),
            Self::CommandLine => write!(f, "cli"),
            Self::Remote(url) => write!(f, "remote:{url}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigLayer – a bag of values from a single source
// ---------------------------------------------------------------------------

/// A single layer of configuration values from one source.
#[derive(Debug, Clone)]
pub struct ConfigLayer {
    /// Where this layer came from.
    pub source: ConfigSource,
    /// Priority (higher wins).
    pub priority: u32,
    /// The key-value pairs.
    pub values: HashMap<String, ConfigValue>,
}

// ---------------------------------------------------------------------------
// ConfigMerger
// ---------------------------------------------------------------------------

/// Merges configuration layers respecting priority ordering.
///
/// Higher-priority layers override lower-priority ones. Map values are
/// merged recursively; all other types are replaced wholesale.
#[derive(Debug, Default)]
pub struct ConfigMerger {
    layers: Vec<ConfigLayer>,
}

impl ConfigMerger {
    pub const fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer. Layers are sorted by priority before merging.
    pub fn add_layer(&mut self, layer: ConfigLayer) {
        self.layers.push(layer);
    }

    /// Returns the number of registered layers.
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Merge all layers into a single flat `HashMap`.
    pub fn merge(&self) -> HashMap<String, ConfigValue> {
        let mut sorted = self.layers.clone();
        sorted.sort_by_key(|l| l.priority);

        let mut result = HashMap::new();
        for layer in &sorted {
            for (k, v) in &layer.values {
                match (result.get(k), v) {
                    (Some(ConfigValue::Map(existing)), ConfigValue::Map(incoming)) => {
                        let mut merged = existing.clone();
                        for (mk, mv) in incoming {
                            merged.insert(mk.clone(), mv.clone());
                        }
                        result.insert(k.clone(), ConfigValue::Map(merged));
                    }
                    _ => {
                        result.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ConfigSchema & validation
// ---------------------------------------------------------------------------

/// Expected type for a schema entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedType {
    String,
    Int,
    Float,
    Bool,
    Array,
    Map,
}

impl std::fmt::Display for ExpectedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::String => "string",
            Self::Int => "int",
            Self::Float => "float",
            Self::Bool => "bool",
            Self::Array => "array",
            Self::Map => "map",
        };
        write!(f, "{s}")
    }
}

/// A single schema entry describing one config key.
#[derive(Debug, Clone)]
pub struct SchemaEntry {
    /// Expected value type.
    pub expected_type: ExpectedType,
    /// Whether the key is required.
    pub required: bool,
    /// Default value when absent and not required.
    pub default: Option<ConfigValue>,
    /// Human-readable description.
    pub description: String,
}

/// Schema describing expected keys, types, defaults, and validation rules.
#[derive(Debug, Clone, Default)]
pub struct ConfigSchema {
    entries: HashMap<String, SchemaEntry>,
}

impl ConfigSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a key definition.
    pub fn define(&mut self, key: impl Into<String>, entry: SchemaEntry) {
        self.entries.insert(key.into(), entry);
    }

    /// Number of defined keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entry for a key.
    pub fn get(&self, key: &str) -> Option<&SchemaEntry> {
        self.entries.get(key)
    }

    /// Iterate over all entries.
    pub fn entries(&self) -> impl Iterator<Item = (&String, &SchemaEntry)> {
        self.entries.iter()
    }
}

/// A single validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub key: String,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.key, self.message)
    }
}

// ---------------------------------------------------------------------------
// ConfigValidator
// ---------------------------------------------------------------------------

/// Validates a merged config map against a [`ConfigSchema`].
#[derive(Debug)]
pub struct ConfigValidator<'a> {
    schema: &'a ConfigSchema,
}

impl<'a> ConfigValidator<'a> {
    pub const fn new(schema: &'a ConfigSchema) -> Self {
        Self { schema }
    }

    /// Validate the config, returning all errors found.
    pub fn validate(&self, config: &HashMap<String, ConfigValue>) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for (key, entry) in self.schema.entries() {
            match config.get(key) {
                None => {
                    if entry.required {
                        errors.push(ValidationError {
                            key: key.clone(),
                            message: "required key is missing".into(),
                        });
                    }
                }
                Some(val) => {
                    if !type_matches(val, entry.expected_type) {
                        errors.push(ValidationError {
                            key: key.clone(),
                            message: format!(
                                "expected type {}, got {}",
                                entry.expected_type,
                                val.type_name()
                            ),
                        });
                    }
                }
            }
        }
        errors
    }

    /// Apply defaults from the schema to the config, filling in any
    /// missing optional keys that have a default value.
    pub fn apply_defaults(&self, config: &mut HashMap<String, ConfigValue>) {
        for (key, entry) in self.schema.entries() {
            if !config.contains_key(key)
                && let Some(ref default) = entry.default
            {
                config.insert(key.clone(), default.clone());
            }
        }
    }
}

/// Returns `true` if the value matches the expected type.
const fn type_matches(val: &ConfigValue, expected: ExpectedType) -> bool {
    matches!(
        (val, expected),
        (ConfigValue::String(_), ExpectedType::String)
            | (ConfigValue::Int(_), ExpectedType::Int)
            | (ConfigValue::Float(_), ExpectedType::Float)
            | (ConfigValue::Bool(_), ExpectedType::Bool)
            | (ConfigValue::Array(_), ExpectedType::Array)
            | (ConfigValue::Map(_), ExpectedType::Map)
    )
}

// ---------------------------------------------------------------------------
// ConfigSerializer
// ---------------------------------------------------------------------------

/// Serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    Json,
    Toml,
    Yaml,
}

/// Serializes a config map to various text formats.
#[derive(Debug)]
pub struct ConfigSerializer;

impl ConfigSerializer {
    /// Serialize a config map into the given format.
    pub fn serialize(config: &HashMap<String, ConfigValue>, format: SerializationFormat) -> String {
        match format {
            SerializationFormat::Json => Self::to_json(config),
            SerializationFormat::Toml => Self::to_toml(config),
            SerializationFormat::Yaml => Self::to_yaml(config),
        }
    }

    fn to_json(config: &HashMap<String, ConfigValue>) -> String {
        let mut lines = Vec::new();
        lines.push("{".to_string());
        let mut entries: Vec<_> = config.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        for (i, (k, v)) in entries.iter().enumerate() {
            let comma = if i + 1 < entries.len() { "," } else { "" };
            lines.push(format!("  \"{k}\": {}{comma}", json_value(v)));
        }
        lines.push("}".to_string());
        lines.join("\n")
    }

    fn to_toml(config: &HashMap<String, ConfigValue>) -> String {
        let mut lines = Vec::new();
        let mut entries: Vec<_> = config.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        for (k, v) in &entries {
            lines.push(format!("{k} = {}", toml_value(v)));
        }
        lines.join("\n")
    }

    fn to_yaml(config: &HashMap<String, ConfigValue>) -> String {
        let mut lines = Vec::new();
        let mut entries: Vec<_> = config.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        for (k, v) in &entries {
            lines.push(format!("{k}: {}", yaml_value(v)));
        }
        lines.join("\n")
    }
}

fn json_value(v: &ConfigValue) -> String {
    match v {
        ConfigValue::String(s) => format!("\"{s}\""),
        ConfigValue::Int(n) => format!("{n}"),
        ConfigValue::Float(f) => format!("{f}"),
        ConfigValue::Bool(b) => format!("{b}"),
        ConfigValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(json_value).collect();
            format!("[{}]", items.join(", "))
        }
        ConfigValue::Map(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let items: Vec<String> =
                entries.iter().map(|(k, v)| format!("\"{k}\": {}", json_value(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

fn toml_value(v: &ConfigValue) -> String {
    match v {
        ConfigValue::String(s) => format!("\"{s}\""),
        ConfigValue::Int(n) => format!("{n}"),
        ConfigValue::Float(f) => format!("{f}"),
        ConfigValue::Bool(b) => format!("{b}"),
        ConfigValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(toml_value).collect();
            format!("[{}]", items.join(", "))
        }
        ConfigValue::Map(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let items: Vec<String> =
                entries.iter().map(|(k, v)| format!("{k} = {}", toml_value(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

fn yaml_value(v: &ConfigValue) -> String {
    match v {
        ConfigValue::String(s) => format!("\"{s}\""),
        ConfigValue::Int(n) => format!("{n}"),
        ConfigValue::Float(f) => format!("{f}"),
        ConfigValue::Bool(b) => format!("{b}"),
        ConfigValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(yaml_value).collect();
            format!("[{}]", items.join(", "))
        }
        ConfigValue::Map(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let items: Vec<String> =
                entries.iter().map(|(k, v)| format!("{k}: {}", yaml_value(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigWatcher
// ---------------------------------------------------------------------------

/// Tracks modification times and signals when a config file has changed.
#[derive(Debug)]
pub struct ConfigWatcher {
    path: PathBuf,
    poll_interval: Duration,
    last_modified: Option<SystemTime>,
    last_check: Option<Instant>,
    change_count: u64,
    callbacks: Vec<String>,
}

impl ConfigWatcher {
    /// Create a new watcher for the given file path.
    pub fn new(path: impl Into<PathBuf>, poll_interval: Duration) -> Self {
        Self {
            path: path.into(),
            poll_interval,
            last_modified: None,
            last_check: None,
            change_count: 0,
            callbacks: Vec::new(),
        }
    }

    /// Path being watched.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Polling interval.
    pub const fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Number of changes detected so far.
    pub const fn change_count(&self) -> u64 {
        self.change_count
    }

    /// Register a named callback identifier.
    pub fn on_change(&mut self, callback_id: impl Into<String>) {
        self.callbacks.push(callback_id.into());
    }

    /// Number of registered callbacks.
    pub const fn callback_count(&self) -> usize {
        self.callbacks.len()
    }

    /// Check whether the file has been modified since last poll.
    ///
    /// Returns `true` if a change was detected, along with the list of
    /// callback identifiers that should be invoked.
    pub fn poll(&mut self) -> (bool, Vec<String>) {
        let now = Instant::now();
        if let Some(last) = self.last_check
            && now.duration_since(last) < self.poll_interval
        {
            return (false, Vec::new());
        }
        self.last_check = Some(now);

        let modified = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();

        let changed = match (self.last_modified, modified) {
            (Some(prev), Some(curr)) => curr > prev,
            (None, Some(_)) => true,
            _ => false,
        };

        if changed {
            self.last_modified = modified;
            self.change_count += 1;
            (true, self.callbacks.clone())
        } else {
            if self.last_modified.is_none() && modified.is_some() {
                self.last_modified = modified;
            }
            (false, Vec::new())
        }
    }

    /// Force-record a modification time so the next real change is detected.
    pub fn mark_seen(&mut self) {
        self.last_modified = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        self.last_check = Some(Instant::now());
    }
}

// ---------------------------------------------------------------------------
// SecretResolver
// ---------------------------------------------------------------------------

/// Resolves secret references embedded in configuration values.
///
/// Supported reference formats:
/// - `${env:VAR_NAME}` – resolved from environment variables
/// - `${vault:path/to/secret}` – placeholder for vault lookups
/// - `${file:/path/to/secret}` – reads the first line of a file
#[derive(Debug)]
pub struct SecretResolver {
    env_prefix: Option<String>,
    vault_values: HashMap<String, String>,
    resolved_count: u64,
}

impl SecretResolver {
    pub fn new() -> Self {
        Self { env_prefix: None, vault_values: HashMap::new(), resolved_count: 0 }
    }

    /// Set a prefix that is prepended to environment variable names.
    #[must_use]
    pub fn with_env_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefix = Some(prefix.into());
        self
    }

    /// Register a vault path → value mapping (for testing / offline use).
    pub fn register_vault_secret(&mut self, path: impl Into<String>, value: impl Into<String>) {
        self.vault_values.insert(path.into(), value.into());
    }

    /// Number of secrets resolved so far.
    pub const fn resolved_count(&self) -> u64 {
        self.resolved_count
    }

    /// Resolve all secret references inside a single string value.
    pub fn resolve_string(&mut self, input: &str) -> Result<String, String> {
        let mut result = input.to_string();

        // Resolve ${env:...}
        while let Some(start) = result.find("${env:") {
            let end = result[start..]
                .find('}')
                .ok_or_else(|| format!("unclosed secret ref at pos {start}"))?
                + start;
            let var_name = &result[start + 6..end];
            let full_name = self
                .env_prefix
                .as_ref()
                .map_or_else(|| var_name.to_string(), |pfx| format!("{pfx}{var_name}"));
            let value =
                std::env::var(&full_name).map_err(|_| format!("env var {full_name} not found"))?;
            result = format!("{}{value}{}", &result[..start], &result[end + 1..]);
            self.resolved_count += 1;
        }

        // Resolve ${vault:...}
        while let Some(start) = result.find("${vault:") {
            let end = result[start..]
                .find('}')
                .ok_or_else(|| format!("unclosed secret ref at pos {start}"))?
                + start;
            let path = &result[start + 8..end];
            let value = self
                .vault_values
                .get(path)
                .ok_or_else(|| format!("vault path not found: {path}"))?
                .clone();
            result = format!("{}{value}{}", &result[..start], &result[end + 1..]);
            self.resolved_count += 1;
        }

        // Resolve ${file:...}
        while let Some(start) = result.find("${file:") {
            let end = result[start..]
                .find('}')
                .ok_or_else(|| format!("unclosed secret ref at pos {start}"))?
                + start;
            let file_path = &result[start + 7..end];
            let contents = std::fs::read_to_string(file_path)
                .map_err(|e| format!("cannot read secret file {file_path}: {e}"))?;
            let first_line = contents.lines().next().unwrap_or("").trim().to_string();
            result = format!("{}{first_line}{}", &result[..start], &result[end + 1..]);
            self.resolved_count += 1;
        }

        Ok(result)
    }

    /// Resolve secret references in all string values of a config map.
    pub fn resolve_all(&mut self, config: &mut HashMap<String, ConfigValue>) -> Result<(), String> {
        let keys: Vec<String> = config.keys().cloned().collect();
        for key in keys {
            if let Some(ConfigValue::String(s)) = config.get(&key)
                && s.contains("${")
            {
                let resolved = self.resolve_string(s)?;
                config.insert(key, ConfigValue::String(resolved));
            }
        }
        Ok(())
    }
}

impl Default for SecretResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConfigProfile
// ---------------------------------------------------------------------------

/// A named configuration profile (e.g. dev, staging, production) that can
/// inherit from a parent profile.
#[derive(Debug, Clone)]
pub struct ConfigProfile {
    /// Profile name.
    pub name: String,
    /// Optional parent profile name for inheritance.
    pub parent: Option<String>,
    /// Values specific to this profile.
    pub values: HashMap<String, ConfigValue>,
    /// Description of this profile's purpose.
    pub description: String,
}

impl ConfigProfile {
    /// Create a profile with no parent.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            values: HashMap::new(),
            description: description.into(),
        }
    }

    /// Create a profile inheriting from another.
    pub fn with_parent(
        name: impl Into<String>,
        parent: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            parent: Some(parent.into()),
            values: HashMap::new(),
            description: description.into(),
        }
    }

    /// Set a value in this profile.
    pub fn set(&mut self, key: impl Into<String>, value: ConfigValue) {
        self.values.insert(key.into(), value);
    }
}

/// Registry of named profiles with inheritance resolution.
#[derive(Debug, Default)]
pub struct ProfileRegistry {
    profiles: HashMap<String, ConfigProfile>,
}

impl ProfileRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a profile.
    pub fn register(&mut self, profile: ConfigProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Number of registered profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Resolve a profile's values, walking the inheritance chain.
    /// Child values override parent values.
    pub fn resolve(&self, name: &str) -> Result<HashMap<String, ConfigValue>, String> {
        let mut chain = Vec::new();
        let mut current = Some(name.to_string());

        // Walk the chain to collect profiles from root to leaf.
        while let Some(ref profile_name) = current {
            let profile = self
                .profiles
                .get(profile_name)
                .ok_or_else(|| format!("profile not found: {profile_name}"))?;
            if chain.contains(&profile.name) {
                return Err(format!("circular inheritance: {}", profile.name));
            }
            chain.push(profile.name.clone());
            current = profile.parent.clone();
        }

        // Apply from root (last) to leaf (first).
        chain.reverse();
        let mut result = HashMap::new();
        for profile_name in &chain {
            if let Some(profile) = self.profiles.get(profile_name) {
                for (k, v) in &profile.values {
                    result.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(result)
    }

    /// List all profile names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.profiles.keys().cloned().collect();
        names.sort();
        names
    }
}

// ---------------------------------------------------------------------------
// ConfigManager
// ---------------------------------------------------------------------------

/// Snapshot of manager state for diagnostics.
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    pub values: HashMap<String, ConfigValue>,
    pub source_count: usize,
    pub active_profile: Option<String>,
    pub validation_errors: Vec<ValidationError>,
    pub created_at: Instant,
}

/// Orchestrates the full configuration pipeline: discover sources → merge →
/// validate → watch → resolve secrets.
#[derive(Debug)]
pub struct ConfigManager {
    merger: ConfigMerger,
    schema: ConfigSchema,
    profiles: ProfileRegistry,
    secret_resolver: SecretResolver,
    watcher: Option<ConfigWatcher>,
    active_profile: Option<String>,
    merged: HashMap<String, ConfigValue>,
    reload_count: u64,
    last_error: Option<String>,
    initialized: bool,
}

impl ConfigManager {
    /// Create a new manager with the given schema.
    pub fn new(schema: ConfigSchema) -> Self {
        Self {
            merger: ConfigMerger::new(),
            schema,
            profiles: ProfileRegistry::new(),
            secret_resolver: SecretResolver::new(),
            watcher: None,
            active_profile: None,
            merged: HashMap::new(),
            reload_count: 0,
            last_error: None,
            initialized: false,
        }
    }

    /// Add a configuration layer (source).
    pub fn add_source(&mut self, layer: ConfigLayer) {
        self.merger.add_layer(layer);
    }

    /// Set the secret resolver.
    pub fn set_secret_resolver(&mut self, resolver: SecretResolver) {
        self.secret_resolver = resolver;
    }

    /// Register a profile.
    pub fn register_profile(&mut self, profile: ConfigProfile) {
        self.profiles.register(profile);
    }

    /// Select the active profile by name.
    pub fn set_active_profile(&mut self, name: impl Into<String>) -> Result<(), String> {
        let name = name.into();
        // Validate the profile exists and can be resolved.
        let _ = self.profiles.resolve(&name)?;
        self.active_profile = Some(name);
        Ok(())
    }

    /// Attach a file watcher.
    pub fn watch_file(&mut self, path: impl Into<PathBuf>, interval: Duration) {
        self.watcher = Some(ConfigWatcher::new(path, interval));
    }

    /// Run the full pipeline: merge → apply profile → apply defaults →
    /// validate → resolve secrets.
    pub fn initialize(&mut self) -> Result<(), Vec<ValidationError>> {
        // 1. Merge all layers.
        self.merged = self.merger.merge();

        // 2. Apply active profile values (override merged).
        if let Some(ref profile_name) = self.active_profile
            && let Ok(profile_values) = self.profiles.resolve(profile_name)
        {
            for (k, v) in profile_values {
                self.merged.insert(k, v);
            }
        }

        // 3. Apply schema defaults.
        let validator = ConfigValidator::new(&self.schema);
        validator.apply_defaults(&mut self.merged);

        // 4. Validate.
        let errors = validator.validate(&self.merged);
        if !errors.is_empty() {
            self.last_error =
                Some(errors.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "));
            return Err(errors);
        }

        // 5. Resolve secrets.
        if let Err(e) = self.secret_resolver.resolve_all(&mut self.merged) {
            self.last_error = Some(e.clone());
            return Err(vec![ValidationError { key: "<secrets>".into(), message: e }]);
        }

        // 6. Mark watcher baseline.
        if let Some(ref mut w) = self.watcher {
            w.mark_seen();
        }

        self.initialized = true;
        self.reload_count += 1;
        self.last_error = None;
        Ok(())
    }

    /// Poll the watcher and reload if the file changed.
    /// Returns `true` if a reload occurred.
    pub fn check_reload(&mut self) -> Result<bool, Vec<ValidationError>> {
        let changed = self.watcher.as_mut().is_some_and(|w| w.poll().0);

        if changed {
            self.initialize()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.merged.get(key)
    }

    /// Get a value by key, returning the default from the schema if absent.
    pub fn get_or_default(&self, key: &str) -> Option<ConfigValue> {
        self.merged
            .get(key)
            .cloned()
            .or_else(|| self.schema.get(key).and_then(|e| e.default.clone()))
    }

    /// Return all merged config values.
    pub const fn values(&self) -> &HashMap<String, ConfigValue> {
        &self.merged
    }

    /// Whether the manager has been initialized.
    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Number of times the config has been loaded / reloaded.
    pub const fn reload_count(&self) -> u64 {
        self.reload_count
    }

    /// Last error message, if any.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Active profile name.
    pub fn active_profile(&self) -> Option<&str> {
        self.active_profile.as_deref()
    }

    /// Number of registered sources.
    pub const fn source_count(&self) -> usize {
        self.merger.layer_count()
    }

    /// Create a snapshot of current state.
    pub fn snapshot(&self) -> ConfigSnapshot {
        let validator = ConfigValidator::new(&self.schema);
        ConfigSnapshot {
            values: self.merged.clone(),
            source_count: self.merger.layer_count(),
            active_profile: self.active_profile.clone(),
            validation_errors: validator.validate(&self.merged),
            created_at: Instant::now(),
        }
    }

    /// Serialize the current config.
    pub fn serialize(&self, format: SerializationFormat) -> String {
        ConfigSerializer::serialize(&self.merged, format)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::literal_string_with_formatting_args)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ConfigValue tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_value_type_names() {
        assert_eq!(ConfigValue::String("a".into()).type_name(), "string");
        assert_eq!(ConfigValue::Int(1).type_name(), "int");
        assert_eq!(ConfigValue::Float(1.0).type_name(), "float");
        assert_eq!(ConfigValue::Bool(true).type_name(), "bool");
        assert_eq!(ConfigValue::Array(vec![]).type_name(), "array");
        assert_eq!(ConfigValue::Map(HashMap::new()).type_name(), "map");
    }

    #[test]
    fn config_value_accessors() {
        assert_eq!(ConfigValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(ConfigValue::Int(42).as_int(), Some(42));
        assert_eq!(ConfigValue::Float(2.5).as_float(), Some(2.5));
        assert_eq!(ConfigValue::Bool(false).as_bool(), Some(false));
    }

    #[test]
    fn config_value_wrong_accessor_returns_none() {
        assert!(ConfigValue::Int(1).as_str().is_none());
        assert!(ConfigValue::String("x".into()).as_int().is_none());
        assert!(ConfigValue::Bool(true).as_float().is_none());
        assert!(ConfigValue::Float(1.0).as_bool().is_none());
    }

    #[test]
    fn config_value_display_string() {
        let v = ConfigValue::String("hello".into());
        assert_eq!(format!("{v}"), "\"hello\"");
    }

    #[test]
    fn config_value_display_int() {
        assert_eq!(format!("{}", ConfigValue::Int(-7)), "-7");
    }

    #[test]
    fn config_value_display_float() {
        let s = format!("{}", ConfigValue::Float(2.5));
        assert!(s.contains("2.5"));
    }

    #[test]
    fn config_value_display_bool() {
        assert_eq!(format!("{}", ConfigValue::Bool(true)), "true");
    }

    #[test]
    fn config_value_display_array() {
        let arr = ConfigValue::Array(vec![ConfigValue::Int(1), ConfigValue::Int(2)]);
        assert_eq!(format!("{arr}"), "[1, 2]");
    }

    #[test]
    fn config_value_display_empty_array() {
        assert_eq!(format!("{}", ConfigValue::Array(vec![])), "[]");
    }

    #[test]
    fn config_value_display_map() {
        let mut m = HashMap::new();
        m.insert("a".into(), ConfigValue::Int(1));
        let s = format!("{}", ConfigValue::Map(m));
        assert!(s.contains("\"a\": 1"));
    }

    #[test]
    fn config_value_equality() {
        assert_eq!(ConfigValue::Int(5), ConfigValue::Int(5));
        assert_ne!(ConfigValue::Int(5), ConfigValue::Int(6));
        assert_ne!(ConfigValue::Int(1), ConfigValue::Bool(true));
    }

    #[test]
    fn config_value_clone() {
        let v = ConfigValue::String("test".into());
        let v2 = v.clone();
        assert_eq!(v, v2);
    }

    // -----------------------------------------------------------------------
    // ConfigSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_source_display_file() {
        let s = ConfigSource::File(PathBuf::from("/etc/config.toml"));
        let display = format!("{s}");
        assert!(display.contains("file:"));
        assert!(display.contains("config.toml"));
    }

    #[test]
    fn config_source_display_env() {
        assert_eq!(format!("{}", ConfigSource::Env), "env");
    }

    #[test]
    fn config_source_display_defaults() {
        assert_eq!(format!("{}", ConfigSource::Defaults), "defaults");
    }

    #[test]
    fn config_source_display_cli() {
        assert_eq!(format!("{}", ConfigSource::CommandLine), "cli");
    }

    #[test]
    fn config_source_display_remote() {
        let s = ConfigSource::Remote("https://cfg.example.com".into());
        assert!(format!("{s}").contains("remote:"));
    }

    #[test]
    fn config_source_equality() {
        assert_eq!(ConfigSource::Env, ConfigSource::Env);
        assert_ne!(ConfigSource::Env, ConfigSource::Defaults);
    }

    // -----------------------------------------------------------------------
    // ConfigMerger tests
    // -----------------------------------------------------------------------

    #[test]
    fn merger_empty() {
        let m = ConfigMerger::new();
        assert_eq!(m.layer_count(), 0);
        assert!(m.merge().is_empty());
    }

    #[test]
    fn merger_single_layer() {
        let mut m = ConfigMerger::new();
        let mut vals = HashMap::new();
        vals.insert("key".into(), ConfigValue::Int(42));
        m.add_layer(ConfigLayer { source: ConfigSource::Defaults, priority: 0, values: vals });
        let merged = m.merge();
        assert_eq!(merged.get("key"), Some(&ConfigValue::Int(42)));
    }

    #[test]
    fn merger_priority_override() {
        let mut m = ConfigMerger::new();
        let mut low = HashMap::new();
        low.insert("key".into(), ConfigValue::String("low".into()));
        let mut high = HashMap::new();
        high.insert("key".into(), ConfigValue::String("high".into()));

        m.add_layer(ConfigLayer { source: ConfigSource::Defaults, priority: 0, values: low });
        m.add_layer(ConfigLayer { source: ConfigSource::CommandLine, priority: 10, values: high });
        let merged = m.merge();
        assert_eq!(merged.get("key"), Some(&ConfigValue::String("high".into())));
    }

    #[test]
    fn merger_disjoint_keys() {
        let mut m = ConfigMerger::new();
        let mut a = HashMap::new();
        a.insert("a".into(), ConfigValue::Int(1));
        let mut b = HashMap::new();
        b.insert("b".into(), ConfigValue::Int(2));
        m.add_layer(ConfigLayer { source: ConfigSource::Defaults, priority: 0, values: a });
        m.add_layer(ConfigLayer { source: ConfigSource::Env, priority: 1, values: b });
        let merged = m.merge();
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merger_map_recursive_merge() {
        let mut m = ConfigMerger::new();
        let mut base_inner = HashMap::new();
        base_inner.insert("x".into(), ConfigValue::Int(1));
        base_inner.insert("y".into(), ConfigValue::Int(2));
        let mut base = HashMap::new();
        base.insert("nested".into(), ConfigValue::Map(base_inner));

        let mut over_inner = HashMap::new();
        over_inner.insert("y".into(), ConfigValue::Int(99));
        over_inner.insert("z".into(), ConfigValue::Int(3));
        let mut over = HashMap::new();
        over.insert("nested".into(), ConfigValue::Map(over_inner));

        m.add_layer(ConfigLayer { source: ConfigSource::Defaults, priority: 0, values: base });
        m.add_layer(ConfigLayer { source: ConfigSource::Env, priority: 1, values: over });

        let merged = m.merge();
        if let Some(ConfigValue::Map(inner)) = merged.get("nested") {
            assert_eq!(inner.get("x"), Some(&ConfigValue::Int(1)));
            assert_eq!(inner.get("y"), Some(&ConfigValue::Int(99)));
            assert_eq!(inner.get("z"), Some(&ConfigValue::Int(3)));
        } else {
            panic!("expected nested map");
        }
    }

    #[test]
    fn merger_three_layers() {
        let mut m = ConfigMerger::new();
        for (pri, val) in [(0, "a"), (5, "b"), (10, "c")] {
            let mut vals = HashMap::new();
            vals.insert("k".into(), ConfigValue::String(val.into()));
            m.add_layer(ConfigLayer {
                source: ConfigSource::Defaults,
                priority: pri,
                values: vals,
            });
        }
        assert_eq!(m.layer_count(), 3);
        let merged = m.merge();
        assert_eq!(merged.get("k"), Some(&ConfigValue::String("c".into())));
    }

    #[test]
    fn merger_layer_count() {
        let mut m = ConfigMerger::new();
        assert_eq!(m.layer_count(), 0);
        m.add_layer(ConfigLayer {
            source: ConfigSource::Defaults,
            priority: 0,
            values: HashMap::new(),
        });
        assert_eq!(m.layer_count(), 1);
    }

    // -----------------------------------------------------------------------
    // ConfigSchema tests
    // -----------------------------------------------------------------------

    #[test]
    fn schema_empty() {
        let s = ConfigSchema::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn schema_define_and_get() {
        let mut s = ConfigSchema::new();
        s.define(
            "port",
            SchemaEntry {
                expected_type: ExpectedType::Int,
                required: true,
                default: None,
                description: "server port".into(),
            },
        );
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
        let entry = s.get("port").unwrap();
        assert_eq!(entry.expected_type, ExpectedType::Int);
        assert!(entry.required);
    }

    #[test]
    fn schema_get_missing() {
        let s = ConfigSchema::new();
        assert!(s.get("missing").is_none());
    }

    #[test]
    fn schema_entries_iter() {
        let mut s = ConfigSchema::new();
        s.define(
            "a",
            SchemaEntry {
                expected_type: ExpectedType::String,
                required: false,
                default: None,
                description: String::new(),
            },
        );
        s.define(
            "b",
            SchemaEntry {
                expected_type: ExpectedType::Bool,
                required: false,
                default: None,
                description: String::new(),
            },
        );
        assert_eq!(s.entries().count(), 2);
    }

    // -----------------------------------------------------------------------
    // ConfigValidator tests
    // -----------------------------------------------------------------------

    fn make_schema() -> ConfigSchema {
        let mut s = ConfigSchema::new();
        s.define(
            "host",
            SchemaEntry {
                expected_type: ExpectedType::String,
                required: true,
                default: None,
                description: "hostname".into(),
            },
        );
        s.define(
            "port",
            SchemaEntry {
                expected_type: ExpectedType::Int,
                required: true,
                default: None,
                description: "port number".into(),
            },
        );
        s.define(
            "debug",
            SchemaEntry {
                expected_type: ExpectedType::Bool,
                required: false,
                default: Some(ConfigValue::Bool(false)),
                description: "debug flag".into(),
            },
        );
        s
    }

    #[test]
    fn validator_valid_config() {
        let schema = make_schema();
        let v = ConfigValidator::new(&schema);
        let mut cfg = HashMap::new();
        cfg.insert("host".into(), ConfigValue::String("localhost".into()));
        cfg.insert("port".into(), ConfigValue::Int(8080));
        assert!(v.validate(&cfg).is_empty());
    }

    #[test]
    fn validator_missing_required() {
        let schema = make_schema();
        let v = ConfigValidator::new(&schema);
        let cfg = HashMap::new();
        let errors = v.validate(&cfg);
        assert!(errors.len() >= 2);
        assert!(errors.iter().any(|e| e.key == "host"));
        assert!(errors.iter().any(|e| e.key == "port"));
    }

    #[test]
    fn validator_wrong_type() {
        let schema = make_schema();
        let v = ConfigValidator::new(&schema);
        let mut cfg = HashMap::new();
        cfg.insert("host".into(), ConfigValue::Int(123));
        cfg.insert("port".into(), ConfigValue::Int(8080));
        let errors = v.validate(&cfg);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("expected type string"));
    }

    #[test]
    fn validator_apply_defaults() {
        let schema = make_schema();
        let v = ConfigValidator::new(&schema);
        let mut cfg = HashMap::new();
        cfg.insert("host".into(), ConfigValue::String("h".into()));
        cfg.insert("port".into(), ConfigValue::Int(80));
        v.apply_defaults(&mut cfg);
        assert_eq!(cfg.get("debug"), Some(&ConfigValue::Bool(false)));
    }

    #[test]
    fn validator_does_not_overwrite_existing() {
        let schema = make_schema();
        let v = ConfigValidator::new(&schema);
        let mut cfg = HashMap::new();
        cfg.insert("host".into(), ConfigValue::String("h".into()));
        cfg.insert("port".into(), ConfigValue::Int(80));
        cfg.insert("debug".into(), ConfigValue::Bool(true));
        v.apply_defaults(&mut cfg);
        assert_eq!(cfg.get("debug"), Some(&ConfigValue::Bool(true)));
    }

    #[test]
    fn validator_missing_optional_no_error() {
        let schema = make_schema();
        let v = ConfigValidator::new(&schema);
        let mut cfg = HashMap::new();
        cfg.insert("host".into(), ConfigValue::String("h".into()));
        cfg.insert("port".into(), ConfigValue::Int(80));
        // "debug" is optional and absent — no error expected
        assert!(v.validate(&cfg).is_empty());
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError { key: "port".into(), message: "required key is missing".into() };
        assert_eq!(format!("{e}"), "port: required key is missing");
    }

    #[test]
    fn type_matches_all_variants() {
        assert!(type_matches(&ConfigValue::String(String::new()), ExpectedType::String));
        assert!(type_matches(&ConfigValue::Int(0), ExpectedType::Int));
        assert!(type_matches(&ConfigValue::Float(0.0), ExpectedType::Float));
        assert!(type_matches(&ConfigValue::Bool(false), ExpectedType::Bool));
        assert!(type_matches(&ConfigValue::Array(vec![]), ExpectedType::Array));
        assert!(type_matches(&ConfigValue::Map(HashMap::new()), ExpectedType::Map));
    }

    #[test]
    fn type_mismatch() {
        assert!(!type_matches(&ConfigValue::Int(0), ExpectedType::String));
        assert!(!type_matches(&ConfigValue::String(String::new()), ExpectedType::Int));
    }

    #[test]
    fn expected_type_display() {
        assert_eq!(format!("{}", ExpectedType::String), "string");
        assert_eq!(format!("{}", ExpectedType::Int), "int");
        assert_eq!(format!("{}", ExpectedType::Float), "float");
        assert_eq!(format!("{}", ExpectedType::Bool), "bool");
        assert_eq!(format!("{}", ExpectedType::Array), "array");
        assert_eq!(format!("{}", ExpectedType::Map), "map");
    }

    // -----------------------------------------------------------------------
    // ConfigSerializer tests
    // -----------------------------------------------------------------------

    fn sample_config() -> HashMap<String, ConfigValue> {
        let mut m = HashMap::new();
        m.insert("host".into(), ConfigValue::String("localhost".into()));
        m.insert("port".into(), ConfigValue::Int(8080));
        m.insert("debug".into(), ConfigValue::Bool(true));
        m
    }

    #[test]
    fn serialize_json() {
        let cfg = sample_config();
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Json);
        assert!(s.contains("\"host\""));
        assert!(s.contains("\"localhost\""));
        assert!(s.contains("8080"));
        assert!(s.contains("true"));
    }

    #[test]
    fn serialize_toml() {
        let cfg = sample_config();
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Toml);
        assert!(s.contains("host = \"localhost\""));
        assert!(s.contains("port = 8080"));
    }

    #[test]
    fn serialize_yaml() {
        let cfg = sample_config();
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Yaml);
        assert!(s.contains("host: \"localhost\""));
        assert!(s.contains("port: 8080"));
    }

    #[test]
    fn serialize_json_array_value() {
        let mut cfg = HashMap::new();
        cfg.insert(
            "tags".into(),
            ConfigValue::Array(vec![
                ConfigValue::String("a".into()),
                ConfigValue::String("b".into()),
            ]),
        );
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Json);
        assert!(s.contains("[\"a\", \"b\"]"));
    }

    #[test]
    fn serialize_empty_config_json() {
        let cfg = HashMap::new();
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Json);
        assert!(s.contains('{'));
        assert!(s.contains('}'));
    }

    #[test]
    fn serialize_toml_bool() {
        let mut cfg = HashMap::new();
        cfg.insert("flag".into(), ConfigValue::Bool(false));
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Toml);
        assert!(s.contains("flag = false"));
    }

    #[test]
    fn serialize_yaml_float() {
        let mut cfg = HashMap::new();
        cfg.insert("rate".into(), ConfigValue::Float(0.75));
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Yaml);
        assert!(s.contains("rate: 0.75"));
    }

    #[test]
    fn serialize_json_nested_map() {
        let mut inner = HashMap::new();
        inner.insert("x".into(), ConfigValue::Int(1));
        let mut cfg = HashMap::new();
        cfg.insert("nested".into(), ConfigValue::Map(inner));
        let s = ConfigSerializer::serialize(&cfg, SerializationFormat::Json);
        assert!(s.contains("\"x\": 1"));
    }

    // -----------------------------------------------------------------------
    // ConfigWatcher tests
    // -----------------------------------------------------------------------

    #[test]
    fn watcher_creation() {
        let w = ConfigWatcher::new("/tmp/test.toml", Duration::from_secs(5));
        assert_eq!(w.path(), Path::new("/tmp/test.toml"));
        assert_eq!(w.poll_interval(), Duration::from_secs(5));
        assert_eq!(w.change_count(), 0);
    }

    #[test]
    fn watcher_register_callbacks() {
        let mut w = ConfigWatcher::new("/tmp/x", Duration::from_millis(100));
        assert_eq!(w.callback_count(), 0);
        w.on_change("reload");
        w.on_change("notify_ui");
        assert_eq!(w.callback_count(), 2);
    }

    #[test]
    fn watcher_poll_nonexistent_file() {
        let mut w = ConfigWatcher::new("/nonexistent/path/config.toml", Duration::from_millis(0));
        let (changed, cbs) = w.poll();
        // File doesn't exist, no mtime — no change.
        assert!(!changed);
        assert!(cbs.is_empty());
    }

    #[test]
    fn watcher_poll_real_file_detects_change() {
        let dir = std::env::temp_dir().join("bitnet_cfg_watch_test");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("watch.toml");
        std::fs::write(&file, "v=1").unwrap();

        let mut w = ConfigWatcher::new(&file, Duration::from_millis(0));
        w.on_change("cb1");

        // First poll — baseline.
        w.mark_seen();

        // Overwrite file.
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&file, "v=2").unwrap();

        let (changed, cbs) = w.poll();
        assert!(changed);
        assert_eq!(cbs, vec!["cb1".to_string()]);
        assert_eq!(w.change_count(), 1);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn watcher_respects_poll_interval() {
        let mut w = ConfigWatcher::new("/tmp/x", Duration::from_secs(999));
        // First poll records the check time.
        let _ = w.poll();
        // Second poll within interval should be skipped.
        let (changed, _) = w.poll();
        assert!(!changed);
    }

    // -----------------------------------------------------------------------
    // SecretResolver tests
    // -----------------------------------------------------------------------

    #[test]
    fn secret_resolver_no_refs() {
        let mut r = SecretResolver::new();
        let resolved = r.resolve_string("plain value").unwrap();
        assert_eq!(resolved, "plain value");
        assert_eq!(r.resolved_count(), 0);
    }

    #[test]
    fn secret_resolver_env_var() {
        let mut r = SecretResolver::new();
        // SAFETY: test-only; no concurrent env access.
        unsafe { std::env::set_var("BITNET_TEST_CFG_SECRET", "s3cret") };
        let resolved = r.resolve_string("pw=${env:BITNET_TEST_CFG_SECRET}").unwrap();
        assert_eq!(resolved, "pw=s3cret");
        assert_eq!(r.resolved_count(), 1);
        unsafe { std::env::remove_var("BITNET_TEST_CFG_SECRET") };
    }

    #[test]
    fn secret_resolver_env_prefix() {
        let mut r = SecretResolver::new().with_env_prefix("MYAPP_");
        // SAFETY: test-only; no concurrent env access.
        unsafe { std::env::set_var("MYAPP_DB_PASS", "hunter2") };
        let resolved = r.resolve_string("${env:DB_PASS}").unwrap();
        assert_eq!(resolved, "hunter2");
        unsafe { std::env::remove_var("MYAPP_DB_PASS") };
    }

    #[test]
    fn secret_resolver_vault() {
        let mut r = SecretResolver::new();
        r.register_vault_secret("kv/data/api_key", "abc123");
        let resolved = r.resolve_string("key=${vault:kv/data/api_key}").unwrap();
        assert_eq!(resolved, "key=abc123");
    }

    #[test]
    fn secret_resolver_file() {
        let dir = std::env::temp_dir().join("bitnet_cfg_secret_test");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("secret.txt");
        std::fs::write(&file, "top_secret\nextra line\n").unwrap();

        let mut r = SecretResolver::new();
        let input = format!("val=${{file:{}}}", file.display());
        let resolved = r.resolve_string(&input).unwrap();
        assert_eq!(resolved, "val=top_secret");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn secret_resolver_missing_env_var() {
        let mut r = SecretResolver::new();
        let result = r.resolve_string("${env:DEFINITELY_NOT_SET_XYZ_12345}");
        assert!(result.is_err());
    }

    #[test]
    fn secret_resolver_missing_vault_path() {
        let mut r = SecretResolver::new();
        let result = r.resolve_string("${vault:missing/path}");
        assert!(result.is_err());
    }

    #[test]
    fn secret_resolver_unclosed_ref() {
        let mut r = SecretResolver::new();
        let result = r.resolve_string("${env:OOPS");
        assert!(result.is_err());
    }

    #[test]
    fn secret_resolver_resolve_all_in_map() {
        let mut r = SecretResolver::new();
        r.register_vault_secret("db/pass", "pw123");
        let mut cfg = HashMap::new();
        cfg.insert("db_password".into(), ConfigValue::String("${vault:db/pass}".into()));
        cfg.insert("port".into(), ConfigValue::Int(5432));
        r.resolve_all(&mut cfg).unwrap();
        assert_eq!(cfg.get("db_password"), Some(&ConfigValue::String("pw123".into())));
        // Non-string values untouched.
        assert_eq!(cfg.get("port"), Some(&ConfigValue::Int(5432)));
    }

    #[test]
    fn secret_resolver_multiple_refs_in_one_string() {
        let mut r = SecretResolver::new();
        // SAFETY: test-only; no concurrent env access.
        unsafe {
            std::env::set_var("BITNET_CFGTEST_HOST", "db.example.com");
            std::env::set_var("BITNET_CFGTEST_PORT", "5432");
        }
        let resolved =
            r.resolve_string("${env:BITNET_CFGTEST_HOST}:${env:BITNET_CFGTEST_PORT}").unwrap();
        assert_eq!(resolved, "db.example.com:5432");
        assert_eq!(r.resolved_count(), 2);
        unsafe {
            std::env::remove_var("BITNET_CFGTEST_HOST");
            std::env::remove_var("BITNET_CFGTEST_PORT");
        }
    }

    // -----------------------------------------------------------------------
    // ConfigProfile & ProfileRegistry tests
    // -----------------------------------------------------------------------

    #[test]
    fn profile_new() {
        let p = ConfigProfile::new("dev", "development profile");
        assert_eq!(p.name, "dev");
        assert!(p.parent.is_none());
        assert!(p.values.is_empty());
    }

    #[test]
    fn profile_with_parent() {
        let p = ConfigProfile::with_parent("staging", "base", "staging env");
        assert_eq!(p.parent.as_deref(), Some("base"));
    }

    #[test]
    fn profile_set_value() {
        let mut p = ConfigProfile::new("dev", "");
        p.set("debug", ConfigValue::Bool(true));
        assert_eq!(p.values.get("debug"), Some(&ConfigValue::Bool(true)));
    }

    #[test]
    fn registry_empty() {
        let r = ProfileRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn registry_register_and_resolve() {
        let mut r = ProfileRegistry::new();
        let mut p = ConfigProfile::new("dev", "");
        p.set("debug", ConfigValue::Bool(true));
        r.register(p);
        assert_eq!(r.len(), 1);
        let vals = r.resolve("dev").unwrap();
        assert_eq!(vals.get("debug"), Some(&ConfigValue::Bool(true)));
    }

    #[test]
    fn registry_inheritance() {
        let mut r = ProfileRegistry::new();
        let mut base = ConfigProfile::new("base", "");
        base.set("host", ConfigValue::String("localhost".into()));
        base.set("port", ConfigValue::Int(8080));
        r.register(base);

        let mut prod = ConfigProfile::with_parent("prod", "base", "production");
        prod.set("host", ConfigValue::String("prod.example.com".into()));
        r.register(prod);

        let vals = r.resolve("prod").unwrap();
        assert_eq!(vals.get("host"), Some(&ConfigValue::String("prod.example.com".into())));
        // Inherited from base.
        assert_eq!(vals.get("port"), Some(&ConfigValue::Int(8080)));
    }

    #[test]
    fn registry_multi_level_inheritance() {
        let mut r = ProfileRegistry::new();
        let mut root = ConfigProfile::new("root", "");
        root.set("a", ConfigValue::Int(1));
        root.set("b", ConfigValue::Int(2));
        r.register(root);

        let mut mid = ConfigProfile::with_parent("mid", "root", "");
        mid.set("b", ConfigValue::Int(20));
        mid.set("c", ConfigValue::Int(3));
        r.register(mid);

        let mut leaf = ConfigProfile::with_parent("leaf", "mid", "");
        leaf.set("c", ConfigValue::Int(30));
        r.register(leaf);

        let vals = r.resolve("leaf").unwrap();
        assert_eq!(vals.get("a"), Some(&ConfigValue::Int(1)));
        assert_eq!(vals.get("b"), Some(&ConfigValue::Int(20)));
        assert_eq!(vals.get("c"), Some(&ConfigValue::Int(30)));
    }

    #[test]
    fn registry_circular_inheritance() {
        let mut r = ProfileRegistry::new();
        let a = ConfigProfile::with_parent("a", "b", "");
        let b = ConfigProfile::with_parent("b", "a", "");
        r.register(a);
        r.register(b);
        assert!(r.resolve("a").is_err());
    }

    #[test]
    fn registry_missing_profile() {
        let r = ProfileRegistry::new();
        assert!(r.resolve("ghost").is_err());
    }

    #[test]
    fn registry_names_sorted() {
        let mut r = ProfileRegistry::new();
        r.register(ConfigProfile::new("staging", ""));
        r.register(ConfigProfile::new("dev", ""));
        r.register(ConfigProfile::new("prod", ""));
        assert_eq!(r.names(), vec!["dev", "prod", "staging"]);
    }

    // -----------------------------------------------------------------------
    // ConfigManager integration tests
    // -----------------------------------------------------------------------

    fn test_schema() -> ConfigSchema {
        let mut s = ConfigSchema::new();
        s.define(
            "host",
            SchemaEntry {
                expected_type: ExpectedType::String,
                required: true,
                default: None,
                description: "hostname".into(),
            },
        );
        s.define(
            "port",
            SchemaEntry {
                expected_type: ExpectedType::Int,
                required: true,
                default: None,
                description: "port".into(),
            },
        );
        s.define(
            "debug",
            SchemaEntry {
                expected_type: ExpectedType::Bool,
                required: false,
                default: Some(ConfigValue::Bool(false)),
                description: "debug mode".into(),
            },
        );
        s.define(
            "workers",
            SchemaEntry {
                expected_type: ExpectedType::Int,
                required: false,
                default: Some(ConfigValue::Int(4)),
                description: "worker count".into(),
            },
        );
        s
    }

    fn defaults_layer() -> ConfigLayer {
        let mut vals = HashMap::new();
        vals.insert("host".into(), ConfigValue::String("0.0.0.0".into()));
        vals.insert("port".into(), ConfigValue::Int(3000));
        ConfigLayer { source: ConfigSource::Defaults, priority: 0, values: vals }
    }

    #[test]
    fn manager_initialize_with_defaults() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        assert!(mgr.is_initialized());
        assert_eq!(mgr.reload_count(), 1);
        assert_eq!(mgr.get("host"), Some(&ConfigValue::String("0.0.0.0".into())));
        // Schema default applied.
        assert_eq!(mgr.get("debug"), Some(&ConfigValue::Bool(false)));
        assert_eq!(mgr.get("workers"), Some(&ConfigValue::Int(4)));
    }

    #[test]
    fn manager_validation_failure() {
        let mut mgr = ConfigManager::new(test_schema());
        // No sources → required keys missing.
        let result = mgr.initialize();
        assert!(result.is_err());
        assert!(mgr.last_error().is_some());
    }

    #[test]
    fn manager_source_count() {
        let mut mgr = ConfigManager::new(ConfigSchema::new());
        assert_eq!(mgr.source_count(), 0);
        mgr.add_source(defaults_layer());
        assert_eq!(mgr.source_count(), 1);
    }

    #[test]
    fn manager_active_profile() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());

        let mut dev = ConfigProfile::new("dev", "development");
        dev.set("debug", ConfigValue::Bool(true));
        mgr.register_profile(dev);

        mgr.set_active_profile("dev").unwrap();
        assert_eq!(mgr.active_profile(), Some("dev"));

        mgr.initialize().unwrap();
        assert_eq!(mgr.get("debug"), Some(&ConfigValue::Bool(true)));
    }

    #[test]
    fn manager_set_invalid_profile() {
        let mut mgr = ConfigManager::new(test_schema());
        assert!(mgr.set_active_profile("nonexistent").is_err());
    }

    #[test]
    fn manager_get_or_default() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        // Exists in merged.
        assert_eq!(mgr.get_or_default("host"), Some(ConfigValue::String("0.0.0.0".into())));
        // Falls back to schema default even though it was applied.
        assert_eq!(mgr.get_or_default("workers"), Some(ConfigValue::Int(4)));
    }

    #[test]
    fn manager_serialize_json() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        let json = mgr.serialize(SerializationFormat::Json);
        assert!(json.contains("\"host\""));
    }

    #[test]
    fn manager_snapshot() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        let snap = mgr.snapshot();
        assert!(snap.validation_errors.is_empty());
        assert_eq!(snap.source_count, 1);
        assert!(snap.active_profile.is_none());
    }

    #[test]
    fn manager_check_reload_no_watcher() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        // No watcher attached — reload returns false.
        assert!(!mgr.check_reload().unwrap());
    }

    #[test]
    fn manager_values_returns_all() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        let vals = mgr.values();
        assert!(vals.contains_key("host"));
        assert!(vals.contains_key("port"));
        assert!(vals.contains_key("debug"));
    }

    #[test]
    fn manager_reload_count_increments() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        assert_eq!(mgr.reload_count(), 1);
        mgr.initialize().unwrap();
        assert_eq!(mgr.reload_count(), 2);
    }

    #[test]
    fn manager_cli_overrides_env_overrides_defaults() {
        let mut mgr = ConfigManager::new(test_schema());

        mgr.add_source(defaults_layer()); // priority 0

        let mut env_vals = HashMap::new();
        env_vals.insert("port".into(), ConfigValue::Int(4000));
        mgr.add_source(ConfigLayer { source: ConfigSource::Env, priority: 5, values: env_vals });

        let mut cli_vals = HashMap::new();
        cli_vals.insert("port".into(), ConfigValue::Int(9999));
        mgr.add_source(ConfigLayer {
            source: ConfigSource::CommandLine,
            priority: 10,
            values: cli_vals,
        });

        mgr.initialize().unwrap();
        assert_eq!(mgr.get("port"), Some(&ConfigValue::Int(9999)));
    }

    #[test]
    fn manager_secret_resolution() {
        let mut mgr = ConfigManager::new(test_schema());

        let mut vals = HashMap::new();
        vals.insert("host".into(), ConfigValue::String("${vault:kv/host}".into()));
        vals.insert("port".into(), ConfigValue::Int(443));
        mgr.add_source(ConfigLayer { source: ConfigSource::Defaults, priority: 0, values: vals });

        let mut resolver = SecretResolver::new();
        resolver.register_vault_secret("kv/host", "secure.example.com");
        mgr.set_secret_resolver(resolver);

        mgr.initialize().unwrap();
        assert_eq!(mgr.get("host"), Some(&ConfigValue::String("secure.example.com".into())));
    }

    #[test]
    fn manager_with_file_watcher() {
        let dir = std::env::temp_dir().join("bitnet_cfg_mgr_watch");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("config.toml");
        std::fs::write(&file, "placeholder").unwrap();

        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());
        mgr.watch_file(&file, Duration::from_millis(10));
        mgr.initialize().unwrap();

        // No change yet.
        assert!(!mgr.check_reload().unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manager_profile_overrides_source() {
        let mut mgr = ConfigManager::new(test_schema());
        mgr.add_source(defaults_layer());

        let mut prod = ConfigProfile::new("prod", "production");
        prod.set("host", ConfigValue::String("prod.host".into()));
        prod.set("port", ConfigValue::Int(443));
        mgr.register_profile(prod);
        mgr.set_active_profile("prod").unwrap();

        mgr.initialize().unwrap();
        assert_eq!(mgr.get("host"), Some(&ConfigValue::String("prod.host".into())));
        assert_eq!(mgr.get("port"), Some(&ConfigValue::Int(443)));
    }

    #[test]
    fn manager_last_error_cleared_on_success() {
        let mut mgr = ConfigManager::new(test_schema());
        // First init fails (no required keys).
        assert!(mgr.initialize().is_err());
        assert!(mgr.last_error().is_some());
        // Add required keys and reinitialize.
        mgr.add_source(defaults_layer());
        mgr.initialize().unwrap();
        assert!(mgr.last_error().is_none());
    }
}
