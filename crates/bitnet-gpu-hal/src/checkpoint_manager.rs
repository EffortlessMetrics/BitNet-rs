//! Inference checkpoint manager for saving and resuming state.
//!
//! Provides [`CheckpointManager`] for creating, restoring, listing, pruning,
//! and verifying inference checkpoints. Supports incremental diffs, multiple
//! compression modes, and automatic checkpoint scheduling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── Configuration ────────────────────────────────────────────────────────────

/// Compression algorithm for checkpoint data.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionMode {
    /// No compression (fastest, largest files).
    #[default]
    None,
    /// Zstandard compression (good balance of speed and ratio).
    Zstd,
    /// LZ4 compression (very fast, moderate ratio).
    Lz4,
    /// Snappy compression (fast, lower ratio).
    Snappy,
}

impl fmt::Display for CompressionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Zstd => write!(f, "zstd"),
            Self::Lz4 => write!(f, "lz4"),
            Self::Snappy => write!(f, "snappy"),
        }
    }
}

/// Settings for checkpoint behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory where checkpoint files are stored.
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to retain (oldest pruned first).
    pub max_checkpoints: usize,
    /// Automatically checkpoint every N tokens (0 = disabled).
    pub auto_save_interval_tokens: u64,
    /// Compression mode for serialised data.
    pub compression: CompressionMode,
    /// Whether to keep incremental diffs between full checkpoints.
    pub enable_incremental: bool,
    /// Interval between full (non-incremental) checkpoints when incremental is
    /// enabled, measured in number of checkpoints.
    pub full_checkpoint_interval: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_checkpoints: 5,
            auto_save_interval_tokens: 0,
            compression: CompressionMode::default(),
            enable_incremental: false,
            full_checkpoint_interval: 5,
        }
    }
}

/// Errors produced by checkpoint operations.
#[derive(Debug)]
pub enum CheckpointError {
    /// An I/O error occurred.
    Io(io::Error),
    /// Serialisation / deserialisation failed.
    Serde(String),
    /// Requested checkpoint was not found.
    NotFound(String),
    /// Checkpoint data failed integrity verification.
    CorruptCheckpoint(String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// A base checkpoint required for diff application was missing.
    MissingBase(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Serde(e) => write!(f, "serialisation error: {e}"),
            Self::NotFound(id) => write!(f, "checkpoint not found: {id}"),
            Self::CorruptCheckpoint(msg) => write!(f, "corrupt checkpoint: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::MissingBase(id) => write!(f, "missing base checkpoint: {id}"),
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e.to_string())
    }
}

// ── Checkpoint metadata ──────────────────────────────────────────────────────

/// Metadata describing a single checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointMetadata {
    /// Unique identifier (e.g. UUID or monotonic counter).
    pub id: String,
    /// Creation timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
    /// Hash of the model weights used during this run.
    pub model_hash: String,
    /// Token position at the time the checkpoint was created.
    pub token_position: u64,
    /// Total size of the KV cache in bytes.
    pub kv_cache_size: u64,
    /// Snapshot of generation parameters at checkpoint time.
    pub generation_params: HashMap<String, String>,
    /// Whether this is a full checkpoint or an incremental diff.
    pub is_incremental: bool,
    /// ID of the base checkpoint if this is an incremental diff.
    pub base_checkpoint_id: Option<String>,
    /// Compression mode used for this checkpoint.
    pub compression: CompressionMode,
    /// CRC-32 checksum of the serialised state payload.
    pub checksum: u32,
}

// ── Inference state ──────────────────────────────────────────────────────────

/// A single KV cache layer entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KVCacheEntry {
    /// Transformer layer index.
    pub layer_idx: usize,
    /// Flattened key tensor data.
    pub key_data: Vec<f32>,
    /// Flattened value tensor data.
    pub value_data: Vec<f32>,
    /// Current sequence length for this layer.
    pub seq_len: usize,
}

/// Serialisable snapshot of the full inference state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceState {
    /// Token IDs generated so far.
    pub token_ids: Vec<u32>,
    /// KV cache entries, one per transformer layer.
    pub kv_cache_entries: Vec<KVCacheEntry>,
    /// RNG state for reproducibility (opaque bytes).
    pub rng_state: Vec<u8>,
    /// Sampling-specific state (e.g. repetition penalty history).
    pub sampling_state: HashMap<String, String>,
}

impl InferenceState {
    /// Total number of KV cache layers.
    #[must_use]
    pub const fn num_kv_layers(&self) -> usize {
        self.kv_cache_entries.len()
    }

    /// Total size of KV cache data in bytes (keys + values, `f32`).
    #[must_use]
    pub fn kv_cache_bytes(&self) -> u64 {
        self.kv_cache_entries
            .iter()
            .map(|e| ((e.key_data.len() + e.value_data.len()) * size_of::<f32>()) as u64)
            .sum()
    }
}

// ── Incremental diff ─────────────────────────────────────────────────────────

/// An incremental checkpoint storing only changed KV cache layers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointDiff {
    /// ID of the base (full) checkpoint this diff applies to.
    pub base_checkpoint_id: String,
    /// Updated token IDs (complete list, not a diff).
    pub token_ids: Vec<u32>,
    /// Only the KV cache layers that changed since the base.
    pub changed_kv_entries: Vec<KVCacheEntry>,
    /// Indices of layers that were changed.
    pub changed_layer_indices: Vec<usize>,
    /// Updated RNG state.
    pub rng_state: Vec<u8>,
    /// Updated sampling state.
    pub sampling_state: HashMap<String, String>,
}

impl CheckpointDiff {
    /// Apply this diff on top of `base` to reconstruct the full state.
    pub fn apply(&self, base: &InferenceState) -> Result<InferenceState, CheckpointError> {
        let mut state = base.clone();
        state.token_ids.clone_from(&self.token_ids);
        state.rng_state.clone_from(&self.rng_state);
        state.sampling_state.clone_from(&self.sampling_state);

        for entry in &self.changed_kv_entries {
            if entry.layer_idx >= state.kv_cache_entries.len() {
                return Err(CheckpointError::CorruptCheckpoint(format!(
                    "diff references layer {} but base has only {} layers",
                    entry.layer_idx,
                    state.kv_cache_entries.len()
                )));
            }
            state.kv_cache_entries[entry.layer_idx] = entry.clone();
        }

        Ok(state)
    }

    /// Create a diff between `base` and `current` by comparing KV layers.
    #[must_use]
    pub fn compute(base: &InferenceState, current: &InferenceState) -> Self {
        let mut changed_kv = Vec::new();
        let mut changed_idx = Vec::new();

        for (i, entry) in current.kv_cache_entries.iter().enumerate() {
            let differs = base
                .kv_cache_entries
                .get(i) != Some(entry);
            if differs {
                changed_kv.push(entry.clone());
                changed_idx.push(i);
            }
        }

        Self {
            base_checkpoint_id: String::new(), // caller fills in
            token_ids: current.token_ids.clone(),
            changed_kv_entries: changed_kv,
            changed_layer_indices: changed_idx,
            rng_state: current.rng_state.clone(),
            sampling_state: current.sampling_state.clone(),
        }
    }
}

// ── Storage abstraction ──────────────────────────────────────────────────────

/// Abstract storage backend for checkpoint data.
pub trait CheckpointStorage {
    /// Persist a full checkpoint.
    fn save(
        &mut self,
        metadata: &CheckpointMetadata,
        data: &[u8],
    ) -> Result<(), CheckpointError>;

    /// Load checkpoint data by ID.
    fn load(&self, id: &str) -> Result<Vec<u8>, CheckpointError>;

    /// List all available checkpoint metadata, newest first.
    fn list(&self) -> Result<Vec<CheckpointMetadata>, CheckpointError>;

    /// Delete a checkpoint by ID.
    fn delete(&mut self, id: &str) -> Result<(), CheckpointError>;

    /// Retrieve metadata for a single checkpoint.
    fn get_metadata(&self, id: &str) -> Result<CheckpointMetadata, CheckpointError>;
}

// ── File-based storage ───────────────────────────────────────────────────────

/// File-system-backed checkpoint storage.
///
/// Layout:
/// ```text
/// <checkpoint_dir>/
///   <id>.meta.json   – serialised [`CheckpointMetadata`]
///   <id>.data        – serialised state payload
/// ```
pub struct FileCheckpointStorage {
    dir: PathBuf,
}

impl FileCheckpointStorage {
    /// Create or open a file-based storage at `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, CheckpointError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.meta.json"))
    }

    fn data_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.data"))
    }
}

impl CheckpointStorage for FileCheckpointStorage {
    fn save(
        &mut self,
        metadata: &CheckpointMetadata,
        data: &[u8],
    ) -> Result<(), CheckpointError> {
        let meta_json = serde_json::to_vec_pretty(metadata)?;
        std::fs::write(self.meta_path(&metadata.id), meta_json)?;
        std::fs::write(self.data_path(&metadata.id), data)?;
        Ok(())
    }

    fn load(&self, id: &str) -> Result<Vec<u8>, CheckpointError> {
        let path = self.data_path(id);
        if !path.exists() {
            return Err(CheckpointError::NotFound(id.to_string()));
        }
        Ok(std::fs::read(path)?)
    }

    fn list(&self) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        let mut metas = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".meta.json") {
                let data = std::fs::read(entry.path())?;
                let meta: CheckpointMetadata = serde_json::from_slice(&data)?;
                metas.push(meta);
            }
        }
        metas.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(metas)
    }

    fn delete(&mut self, id: &str) -> Result<(), CheckpointError> {
        let meta = self.meta_path(id);
        let data = self.data_path(id);
        if !meta.exists() && !data.exists() {
            return Err(CheckpointError::NotFound(id.to_string()));
        }
        if meta.exists() {
            std::fs::remove_file(meta)?;
        }
        if data.exists() {
            std::fs::remove_file(data)?;
        }
        Ok(())
    }

    fn get_metadata(&self, id: &str) -> Result<CheckpointMetadata, CheckpointError> {
        let path = self.meta_path(id);
        if !path.exists() {
            return Err(CheckpointError::NotFound(id.to_string()));
        }
        let data = std::fs::read(path)?;
        let meta: CheckpointMetadata = serde_json::from_slice(&data)?;
        Ok(meta)
    }
}

// ── In-memory storage (testing) ──────────────────────────────────────────────

/// In-memory storage backend, useful for tests.
#[derive(Default)]
pub struct MemoryCheckpointStorage {
    metas: HashMap<String, CheckpointMetadata>,
    blobs: HashMap<String, Vec<u8>>,
}

impl CheckpointStorage for MemoryCheckpointStorage {
    fn save(
        &mut self,
        metadata: &CheckpointMetadata,
        data: &[u8],
    ) -> Result<(), CheckpointError> {
        self.metas
            .insert(metadata.id.clone(), metadata.clone());
        self.blobs.insert(metadata.id.clone(), data.to_vec());
        Ok(())
    }

    fn load(&self, id: &str) -> Result<Vec<u8>, CheckpointError> {
        self.blobs
            .get(id)
            .cloned()
            .ok_or_else(|| CheckpointError::NotFound(id.to_string()))
    }

    fn list(&self) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        let mut metas: Vec<_> = self.metas.values().cloned().collect();
        metas.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(metas)
    }

    fn delete(&mut self, id: &str) -> Result<(), CheckpointError> {
        if self.metas.remove(id).is_none() {
            return Err(CheckpointError::NotFound(id.to_string()));
        }
        self.blobs.remove(id);
        Ok(())
    }

    fn get_metadata(&self, id: &str) -> Result<CheckpointMetadata, CheckpointError> {
        self.metas
            .get(id)
            .cloned()
            .ok_or_else(|| CheckpointError::NotFound(id.to_string()))
    }
}

// ── Checkpoint scheduler ─────────────────────────────────────────────────────

/// Trigger reason returned by [`CheckpointScheduler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerReason {
    /// Token-count threshold reached.
    TokenCount,
    /// Time-based interval elapsed.
    TimeElapsed,
    /// Explicitly requested (e.g. user pause, risky operation).
    Explicit,
}

/// Decides when a checkpoint should be created.
pub struct CheckpointScheduler {
    token_interval: u64,
    time_interval: Option<Duration>,
    last_checkpoint_tokens: u64,
    last_checkpoint_time: Instant,
}

impl CheckpointScheduler {
    /// Create a new scheduler.
    ///
    /// - `token_interval`: checkpoint every N tokens (0 = disabled).
    /// - `time_interval`: checkpoint every duration (None = disabled).
    #[must_use]
    pub fn new(token_interval: u64, time_interval: Option<Duration>) -> Self {
        Self {
            token_interval,
            time_interval,
            last_checkpoint_tokens: 0,
            last_checkpoint_time: Instant::now(),
        }
    }

    /// Check whether a checkpoint should be triggered at `current_tokens`.
    #[must_use]
    pub fn should_checkpoint(&self, current_tokens: u64) -> Option<TriggerReason> {
        if self.token_interval > 0
            && current_tokens >= self.last_checkpoint_tokens + self.token_interval
        {
            return Some(TriggerReason::TokenCount);
        }
        if let Some(interval) = self.time_interval
            && self.last_checkpoint_time.elapsed() >= interval
        {
            return Some(TriggerReason::TimeElapsed);
        }
        None
    }

    /// Record that a checkpoint was just taken at `tokens`.
    pub fn record_checkpoint(&mut self, tokens: u64) {
        self.last_checkpoint_tokens = tokens;
        self.last_checkpoint_time = Instant::now();
    }
}

// ── Simple CRC-32 (IEEE / PKZIP) ────────────────────────────────────────────

/// Compute CRC-32 (IEEE) over `data`.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// ── Simulated compression helpers ────────────────────────────────────────────
//
// Real compression (zstd, lz4, snappy) would pull in C/Rust deps. For now we
// use a trivial tagged-envelope so the round-trip path is fully exercised.

/// Wrap `data` in a 1-byte tag indicating the compression mode.
fn compress(data: &[u8], mode: CompressionMode) -> Vec<u8> {
    let tag: u8 = match mode {
        CompressionMode::None => 0,
        CompressionMode::Zstd => 1,
        CompressionMode::Lz4 => 2,
        CompressionMode::Snappy => 3,
    };
    let mut out = Vec::with_capacity(1 + data.len());
    out.push(tag);
    out.extend_from_slice(data);
    out
}

/// Unwrap a tagged-envelope produced by [`compress`].
fn decompress(data: &[u8]) -> Result<Vec<u8>, CheckpointError> {
    if data.is_empty() {
        return Err(CheckpointError::CorruptCheckpoint(
            "empty compressed payload".into(),
        ));
    }
    // Tag byte is stripped; remainder is the original data.
    Ok(data[1..].to_vec())
}

// ── Checkpoint manager ───────────────────────────────────────────────────────

/// Main checkpoint manager.
///
/// Orchestrates creation, restoration, listing, pruning, and verification of
/// inference checkpoints.
pub struct CheckpointManager<S: CheckpointStorage> {
    config: CheckpointConfig,
    storage: S,
    counter: u64,
    scheduler: CheckpointScheduler,
    /// The most recent full checkpoint state (used as base for diffs).
    last_full_state: Option<InferenceState>,
    /// ID of the last full checkpoint.
    last_full_id: Option<String>,
    /// Number of incremental checkpoints since last full.
    incremental_since_full: usize,
}

impl<S: CheckpointStorage> CheckpointManager<S> {
    /// Create a new manager with the given config and storage backend.
    pub fn new(config: CheckpointConfig, storage: S) -> Result<Self, CheckpointError> {
        Self::validate_config(&config)?;
        let scheduler = CheckpointScheduler::new(
            config.auto_save_interval_tokens,
            None,
        );
        Ok(Self {
            config,
            storage,
            counter: 0,
            scheduler,
            last_full_state: None,
            last_full_id: None,
            incremental_since_full: 0,
        })
    }

    fn validate_config(config: &CheckpointConfig) -> Result<(), CheckpointError> {
        if config.max_checkpoints == 0 {
            return Err(CheckpointError::InvalidConfig(
                "max_checkpoints must be > 0".into(),
            ));
        }
        if config.enable_incremental && config.full_checkpoint_interval == 0 {
            return Err(CheckpointError::InvalidConfig(
                "full_checkpoint_interval must be > 0 when incremental is enabled".into(),
            ));
        }
        Ok(())
    }

    fn next_id(&mut self) -> String {
        self.counter += 1;
        format!("ckpt-{:06}", self.counter)
    }

    fn now_epoch() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Create a checkpoint of `state` with the given `model_hash`.
    pub fn create_checkpoint(
        &mut self,
        state: &InferenceState,
        model_hash: &str,
    ) -> Result<CheckpointMetadata, CheckpointError> {
        let use_incremental = self.config.enable_incremental
            && self.last_full_state.is_some()
            && self.incremental_since_full < self.config.full_checkpoint_interval;

        if use_incremental {
            self.create_incremental(state, model_hash)
        } else {
            self.create_full(state, model_hash)
        }
    }

    fn create_full(
        &mut self,
        state: &InferenceState,
        model_hash: &str,
    ) -> Result<CheckpointMetadata, CheckpointError> {
        let id = self.next_id();
        let json = serde_json::to_vec(state)?;
        let checksum = crc32(&json);
        let payload = compress(&json, self.config.compression);

        let meta = CheckpointMetadata {
            id: id.clone(),
            timestamp: Self::now_epoch(),
            model_hash: model_hash.to_string(),
            token_position: state.token_ids.len() as u64,
            kv_cache_size: state.kv_cache_bytes(),
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: self.config.compression,
            checksum,
        };

        self.storage.save(&meta, &payload)?;
        self.last_full_state = Some(state.clone());
        self.last_full_id = Some(id);
        self.incremental_since_full = 0;

        let token_pos = meta.token_position;
        self.scheduler.record_checkpoint(token_pos);
        self.prune()?;

        Ok(meta)
    }

    fn create_incremental(
        &mut self,
        state: &InferenceState,
        model_hash: &str,
    ) -> Result<CheckpointMetadata, CheckpointError> {
        let base = self.last_full_state.as_ref().expect("checked by caller");
        let base_id = self.last_full_id.as_ref().expect("checked by caller").clone();

        let mut diff = CheckpointDiff::compute(base, state);
        diff.base_checkpoint_id.clone_from(&base_id);

        let id = self.next_id();
        let json = serde_json::to_vec(&diff)?;
        let checksum = crc32(&json);
        let payload = compress(&json, self.config.compression);

        let meta = CheckpointMetadata {
            id,
            timestamp: Self::now_epoch(),
            model_hash: model_hash.to_string(),
            token_position: state.token_ids.len() as u64,
            kv_cache_size: state.kv_cache_bytes(),
            generation_params: HashMap::new(),
            is_incremental: true,
            base_checkpoint_id: Some(base_id),
            compression: self.config.compression,
            checksum,
        };

        self.storage.save(&meta, &payload)?;
        self.incremental_since_full += 1;

        let token_pos = meta.token_position;
        self.scheduler.record_checkpoint(token_pos);
        self.prune()?;

        Ok(meta)
    }

    /// Restore inference state from checkpoint `id`.
    pub fn restore_checkpoint(
        &self,
        id: &str,
    ) -> Result<InferenceState, CheckpointError> {
        let meta = self.storage.get_metadata(id)?;
        let raw = self.storage.load(id)?;
        let json = decompress(&raw)?;

        // Verify checksum.
        let actual_crc = crc32(&json);
        if actual_crc != meta.checksum {
            return Err(CheckpointError::CorruptCheckpoint(format!(
                "CRC mismatch: expected {:#010x}, got {:#010x}",
                meta.checksum, actual_crc
            )));
        }

        if meta.is_incremental {
            let diff: CheckpointDiff = serde_json::from_slice(&json)?;
            let base_id = meta.base_checkpoint_id.as_ref().ok_or_else(|| {
                CheckpointError::CorruptCheckpoint(
                    "incremental checkpoint missing base_checkpoint_id".into(),
                )
            })?;
            let base_state = self.restore_checkpoint(base_id)?;
            diff.apply(&base_state)
        } else {
            let state: InferenceState = serde_json::from_slice(&json)?;
            Ok(state)
        }
    }

    /// List all checkpoint metadata, newest first.
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
        self.storage.list()
    }

    /// Delete a checkpoint by ID.
    pub fn delete_checkpoint(&mut self, id: &str) -> Result<(), CheckpointError> {
        self.storage.delete(id)
    }

    /// Prune old checkpoints to honour `max_checkpoints`.
    pub fn prune(&mut self) -> Result<usize, CheckpointError> {
        let all = self.storage.list()?;
        let max = self.config.max_checkpoints;
        if all.len() <= max {
            return Ok(0);
        }
        let to_remove = all.len() - max;
        // Remove the oldest (they're sorted newest-first).
        let victims: Vec<_> = all.iter().rev().take(to_remove).cloned().collect();
        for m in &victims {
            // Best-effort: if deletion fails we still continue.
            let _ = self.storage.delete(&m.id);
        }
        Ok(victims.len())
    }

    /// Verify the integrity of a stored checkpoint.
    pub fn verify_checkpoint(&self, id: &str) -> Result<bool, CheckpointError> {
        let meta = self.storage.get_metadata(id)?;
        let raw = self.storage.load(id)?;
        let json = decompress(&raw)?;
        Ok(crc32(&json) == meta.checksum)
    }

    /// Return a reference to the inner scheduler.
    #[must_use]
    pub const fn scheduler(&self) -> &CheckpointScheduler {
        &self.scheduler
    }

    /// Return a mutable reference to the inner scheduler.
    pub const fn scheduler_mut(&mut self) -> &mut CheckpointScheduler {
        &mut self.scheduler
    }

    /// Return a reference to the config.
    #[must_use]
    pub const fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn sample_state(n_tokens: usize, n_layers: usize) -> InferenceState {
        InferenceState {
            token_ids: (0..n_tokens as u32).collect(),
            kv_cache_entries: (0..n_layers)
                .map(|i| KVCacheEntry {
                    layer_idx: i,
                    key_data: vec![i as f32; 4],
                    value_data: vec![(i as f32) * 0.5; 4],
                    seq_len: n_tokens,
                })
                .collect(),
            rng_state: vec![42, 43, 44],
            sampling_state: HashMap::from([("temperature".into(), "0.7".into())]),
        }
    }

    fn default_config_with_dir(dir: &Path) -> CheckpointConfig {
        CheckpointConfig {
            checkpoint_dir: dir.to_path_buf(),
            max_checkpoints: 5,
            auto_save_interval_tokens: 0,
            compression: CompressionMode::None,
            enable_incremental: false,
            full_checkpoint_interval: 5,
        }
    }

    fn memory_manager(
        config: CheckpointConfig,
    ) -> CheckpointManager<MemoryCheckpointStorage> {
        CheckpointManager::new(config, MemoryCheckpointStorage::default()).unwrap()
    }

    fn default_memory_manager() -> CheckpointManager<MemoryCheckpointStorage> {
        memory_manager(CheckpointConfig::default())
    }

    // ── CompressionMode tests ───────────────────────────────────────────

    #[test]
    fn test_compression_mode_default_is_none() {
        assert_eq!(CompressionMode::default(), CompressionMode::None);
    }

    #[test]
    fn test_compression_mode_display() {
        assert_eq!(CompressionMode::None.to_string(), "none");
        assert_eq!(CompressionMode::Zstd.to_string(), "zstd");
        assert_eq!(CompressionMode::Lz4.to_string(), "lz4");
        assert_eq!(CompressionMode::Snappy.to_string(), "snappy");
    }

    #[test]
    fn test_compression_mode_serde_roundtrip() {
        for mode in [
            CompressionMode::None,
            CompressionMode::Zstd,
            CompressionMode::Lz4,
            CompressionMode::Snappy,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: CompressionMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn test_compression_mode_equality() {
        assert_eq!(CompressionMode::Zstd, CompressionMode::Zstd);
        assert_ne!(CompressionMode::Zstd, CompressionMode::Lz4);
    }

    // ── CheckpointConfig tests ──────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = CheckpointConfig::default();
        assert_eq!(cfg.max_checkpoints, 5);
        assert_eq!(cfg.auto_save_interval_tokens, 0);
        assert_eq!(cfg.compression, CompressionMode::None);
        assert!(!cfg.enable_incremental);
        assert_eq!(cfg.full_checkpoint_interval, 5);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = CheckpointConfig {
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            max_checkpoints: 10,
            auto_save_interval_tokens: 256,
            compression: CompressionMode::Zstd,
            enable_incremental: true,
            full_checkpoint_interval: 3,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: CheckpointConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_checkpoints, 10);
        assert_eq!(back.compression, CompressionMode::Zstd);
        assert!(back.enable_incremental);
    }

    #[test]
    fn test_config_validation_max_checkpoints_zero() {
        let cfg = CheckpointConfig {
            max_checkpoints: 0,
            ..Default::default()
        };
        let result = CheckpointManager::new(cfg, MemoryCheckpointStorage::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_incremental_zero_interval() {
        let cfg = CheckpointConfig {
            enable_incremental: true,
            full_checkpoint_interval: 0,
            ..Default::default()
        };
        let result = CheckpointManager::new(cfg, MemoryCheckpointStorage::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_valid_incremental() {
        let cfg = CheckpointConfig {
            enable_incremental: true,
            full_checkpoint_interval: 3,
            ..Default::default()
        };
        assert!(CheckpointManager::new(cfg, MemoryCheckpointStorage::default()).is_ok());
    }

    // ── CheckpointMetadata tests ────────────────────────────────────────

    #[test]
    fn test_metadata_serde_roundtrip() {
        let meta = CheckpointMetadata {
            id: "ckpt-000001".into(),
            timestamp: 1_700_000_000,
            model_hash: "abc123".into(),
            token_position: 512,
            kv_cache_size: 4096,
            generation_params: HashMap::from([("temp".into(), "0.7".into())]),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::None,
            checksum: 0xDEAD_BEEF,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, back);
    }

    #[test]
    fn test_metadata_incremental_fields() {
        let meta = CheckpointMetadata {
            id: "ckpt-000002".into(),
            timestamp: 1_700_000_001,
            model_hash: "abc123".into(),
            token_position: 600,
            kv_cache_size: 8192,
            generation_params: HashMap::new(),
            is_incremental: true,
            base_checkpoint_id: Some("ckpt-000001".into()),
            compression: CompressionMode::Lz4,
            checksum: 0,
        };
        assert!(meta.is_incremental);
        assert_eq!(meta.base_checkpoint_id.as_deref(), Some("ckpt-000001"));
    }

    // ── KVCacheEntry tests ──────────────────────────────────────────────

    #[test]
    fn test_kv_cache_entry_serde() {
        let entry = KVCacheEntry {
            layer_idx: 3,
            key_data: vec![1.0, 2.0, 3.0],
            value_data: vec![4.0, 5.0, 6.0],
            seq_len: 10,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: KVCacheEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn test_kv_cache_entry_empty_data() {
        let entry = KVCacheEntry {
            layer_idx: 0,
            key_data: vec![],
            value_data: vec![],
            seq_len: 0,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: KVCacheEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    // ── InferenceState tests ────────────────────────────────────────────

    #[test]
    fn test_inference_state_serde_roundtrip() {
        let state = sample_state(10, 4);
        let json = serde_json::to_string(&state).unwrap();
        let back: InferenceState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, back);
    }

    #[test]
    fn test_inference_state_num_kv_layers() {
        assert_eq!(sample_state(5, 8).num_kv_layers(), 8);
        assert_eq!(sample_state(5, 0).num_kv_layers(), 0);
    }

    #[test]
    fn test_inference_state_kv_cache_bytes() {
        let state = sample_state(5, 2);
        // Each layer: 4 keys + 4 values = 8 f32s = 32 bytes. 2 layers = 64.
        assert_eq!(state.kv_cache_bytes(), 64);
    }

    #[test]
    fn test_inference_state_kv_cache_bytes_empty() {
        let state = InferenceState {
            token_ids: vec![],
            kv_cache_entries: vec![],
            rng_state: vec![],
            sampling_state: HashMap::new(),
        };
        assert_eq!(state.kv_cache_bytes(), 0);
    }

    #[test]
    fn test_inference_state_empty() {
        let state = InferenceState {
            token_ids: vec![],
            kv_cache_entries: vec![],
            rng_state: vec![],
            sampling_state: HashMap::new(),
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: InferenceState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, back);
    }

    #[test]
    fn test_inference_state_large_token_count() {
        let state = sample_state(100_000, 1);
        assert_eq!(state.token_ids.len(), 100_000);
        let json = serde_json::to_vec(&state).unwrap();
        let back: InferenceState = serde_json::from_slice(&json).unwrap();
        assert_eq!(state.token_ids.len(), back.token_ids.len());
    }

    // ── CheckpointDiff tests ────────────────────────────────────────────

    #[test]
    fn test_diff_compute_identical() {
        let state = sample_state(10, 4);
        let diff = CheckpointDiff::compute(&state, &state);
        assert!(diff.changed_kv_entries.is_empty());
        assert!(diff.changed_layer_indices.is_empty());
    }

    #[test]
    fn test_diff_compute_one_layer_changed() {
        let base = sample_state(10, 4);
        let mut current = base.clone();
        current.kv_cache_entries[2].key_data = vec![99.0; 4];
        let diff = CheckpointDiff::compute(&base, &current);
        assert_eq!(diff.changed_layer_indices, vec![2]);
        assert_eq!(diff.changed_kv_entries.len(), 1);
    }

    #[test]
    fn test_diff_compute_all_layers_changed() {
        let base = sample_state(10, 4);
        let mut current = base.clone();
        for e in &mut current.kv_cache_entries {
            e.key_data = vec![99.0; 4];
        }
        let diff = CheckpointDiff::compute(&base, &current);
        assert_eq!(diff.changed_layer_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_diff_apply_produces_current_state() {
        let base = sample_state(10, 4);
        let mut current = base.clone();
        current.token_ids.push(999);
        current.kv_cache_entries[1].key_data = vec![7.0; 4];
        current.rng_state = vec![1, 2, 3, 4, 5];

        let diff = CheckpointDiff::compute(&base, &current);
        let restored = diff.apply(&base).unwrap();
        assert_eq!(restored, current);
    }

    #[test]
    fn test_diff_apply_invalid_layer_index() {
        let base = sample_state(5, 2);
        let diff = CheckpointDiff {
            base_checkpoint_id: String::new(),
            token_ids: vec![],
            changed_kv_entries: vec![KVCacheEntry {
                layer_idx: 10, // out of range
                key_data: vec![],
                value_data: vec![],
                seq_len: 0,
            }],
            changed_layer_indices: vec![10],
            rng_state: vec![],
            sampling_state: HashMap::new(),
        };
        assert!(diff.apply(&base).is_err());
    }

    #[test]
    fn test_diff_apply_updates_sampling_state() {
        let base = sample_state(5, 2);
        let mut current = base.clone();
        current
            .sampling_state
            .insert("top_k".into(), "50".into());

        let diff = CheckpointDiff::compute(&base, &current);
        let restored = diff.apply(&base).unwrap();
        assert_eq!(
            restored.sampling_state.get("top_k").map(String::as_str),
            Some("50")
        );
    }

    // ── CRC-32 tests ────────────────────────────────────────────────────

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_value() {
        // "123456789" => 0xCBF43926 (IEEE CRC-32).
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = b"hello world";
        assert_eq!(crc32(data), crc32(data));
    }

    #[test]
    fn test_crc32_different_data_different_hash() {
        assert_ne!(crc32(b"aaa"), crc32(b"bbb"));
    }

    // ── Compress / decompress tests ─────────────────────────────────────

    #[test]
    fn test_compress_decompress_roundtrip_none() {
        let data = b"hello";
        let compressed = compress(data, CompressionMode::None);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compress_decompress_roundtrip_zstd() {
        let data = b"checkpoint data";
        let compressed = compress(data, CompressionMode::Zstd);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compress_decompress_roundtrip_lz4() {
        let data = b"lz4 test data";
        let compressed = compress(data, CompressionMode::Lz4);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compress_decompress_roundtrip_snappy() {
        let data = b"snappy test data";
        let compressed = compress(data, CompressionMode::Snappy);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_decompress_empty_payload() {
        assert!(decompress(b"").is_err());
    }

    // ── MemoryCheckpointStorage tests ───────────────────────────────────

    #[test]
    fn test_memory_storage_save_load() {
        let mut store = MemoryCheckpointStorage::default();
        let meta = CheckpointMetadata {
            id: "m1".into(),
            timestamp: 100,
            model_hash: "h".into(),
            token_position: 0,
            kv_cache_size: 0,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::None,
            checksum: 0,
        };
        store.save(&meta, b"data1").unwrap();
        assert_eq!(store.load("m1").unwrap(), b"data1");
    }

    #[test]
    fn test_memory_storage_load_missing() {
        let store = MemoryCheckpointStorage::default();
        assert!(store.load("nonexistent").is_err());
    }

    #[test]
    fn test_memory_storage_list_sorted() {
        let mut store = MemoryCheckpointStorage::default();
        for (id, ts) in [("a", 100u64), ("b", 300), ("c", 200)] {
            let meta = CheckpointMetadata {
                id: id.into(),
                timestamp: ts,
                model_hash: String::new(),
                token_position: 0,
                kv_cache_size: 0,
                generation_params: HashMap::new(),
                is_incremental: false,
                base_checkpoint_id: None,
                compression: CompressionMode::None,
                checksum: 0,
            };
            store.save(&meta, b"x").unwrap();
        }
        let list = store.list().unwrap();
        assert_eq!(list[0].id, "b"); // newest
        assert_eq!(list[1].id, "c");
        assert_eq!(list[2].id, "a"); // oldest
    }

    #[test]
    fn test_memory_storage_delete() {
        let mut store = MemoryCheckpointStorage::default();
        let meta = CheckpointMetadata {
            id: "d1".into(),
            timestamp: 1,
            model_hash: String::new(),
            token_position: 0,
            kv_cache_size: 0,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::None,
            checksum: 0,
        };
        store.save(&meta, b"x").unwrap();
        store.delete("d1").unwrap();
        assert!(store.load("d1").is_err());
    }

    #[test]
    fn test_memory_storage_delete_missing() {
        let mut store = MemoryCheckpointStorage::default();
        assert!(store.delete("nope").is_err());
    }

    #[test]
    fn test_memory_storage_get_metadata() {
        let mut store = MemoryCheckpointStorage::default();
        let meta = CheckpointMetadata {
            id: "gm".into(),
            timestamp: 42,
            model_hash: "hash".into(),
            token_position: 10,
            kv_cache_size: 20,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::Zstd,
            checksum: 999,
        };
        store.save(&meta, b"x").unwrap();
        let got = store.get_metadata("gm").unwrap();
        assert_eq!(got, meta);
    }

    #[test]
    fn test_memory_storage_get_metadata_missing() {
        let store = MemoryCheckpointStorage::default();
        assert!(store.get_metadata("nope").is_err());
    }

    // ── FileCheckpointStorage tests ─────────────────────────────────────

    #[test]
    fn test_file_storage_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FileCheckpointStorage::new(dir.path()).unwrap();
        let meta = CheckpointMetadata {
            id: "f1".into(),
            timestamp: 100,
            model_hash: "h".into(),
            token_position: 0,
            kv_cache_size: 0,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::None,
            checksum: 0,
        };
        store.save(&meta, b"file-data").unwrap();
        assert_eq!(store.load("f1").unwrap(), b"file-data");
    }

    #[test]
    fn test_file_storage_load_missing() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCheckpointStorage::new(dir.path()).unwrap();
        assert!(store.load("missing").is_err());
    }

    #[test]
    fn test_file_storage_list() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FileCheckpointStorage::new(dir.path()).unwrap();
        for (id, ts) in [("x", 10u64), ("y", 20)] {
            let meta = CheckpointMetadata {
                id: id.into(),
                timestamp: ts,
                model_hash: String::new(),
                token_position: 0,
                kv_cache_size: 0,
                generation_params: HashMap::new(),
                is_incremental: false,
                base_checkpoint_id: None,
                compression: CompressionMode::None,
                checksum: 0,
            };
            store.save(&meta, b"d").unwrap();
        }
        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "y"); // newest first
    }

    #[test]
    fn test_file_storage_delete() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FileCheckpointStorage::new(dir.path()).unwrap();
        let meta = CheckpointMetadata {
            id: "del".into(),
            timestamp: 1,
            model_hash: String::new(),
            token_position: 0,
            kv_cache_size: 0,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::None,
            checksum: 0,
        };
        store.save(&meta, b"x").unwrap();
        store.delete("del").unwrap();
        assert!(store.load("del").is_err());
    }

    #[test]
    fn test_file_storage_delete_missing() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FileCheckpointStorage::new(dir.path()).unwrap();
        assert!(store.delete("nope").is_err());
    }

    #[test]
    fn test_file_storage_get_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FileCheckpointStorage::new(dir.path()).unwrap();
        let meta = CheckpointMetadata {
            id: "gm".into(),
            timestamp: 55,
            model_hash: "mh".into(),
            token_position: 7,
            kv_cache_size: 14,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::Lz4,
            checksum: 12345,
        };
        store.save(&meta, b"payload").unwrap();
        let got = store.get_metadata("gm").unwrap();
        assert_eq!(got, meta);
    }

    #[test]
    fn test_file_storage_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("a").join("b").join("c");
        let _store = FileCheckpointStorage::new(&subdir).unwrap();
        assert!(subdir.is_dir());
    }

    // ── CheckpointScheduler tests ───────────────────────────────────────

    #[test]
    fn test_scheduler_disabled() {
        let sched = CheckpointScheduler::new(0, None);
        assert!(sched.should_checkpoint(1000).is_none());
    }

    #[test]
    fn test_scheduler_token_trigger() {
        let sched = CheckpointScheduler::new(100, None);
        assert!(sched.should_checkpoint(50).is_none());
        assert_eq!(
            sched.should_checkpoint(100),
            Some(TriggerReason::TokenCount)
        );
        assert_eq!(
            sched.should_checkpoint(200),
            Some(TriggerReason::TokenCount)
        );
    }

    #[test]
    fn test_scheduler_token_after_record() {
        let mut sched = CheckpointScheduler::new(100, None);
        sched.record_checkpoint(100);
        assert!(sched.should_checkpoint(150).is_none());
        assert_eq!(
            sched.should_checkpoint(200),
            Some(TriggerReason::TokenCount)
        );
    }

    #[test]
    fn test_scheduler_time_trigger() {
        let mut sched = CheckpointScheduler::new(0, Some(Duration::from_millis(1)));
        // Force the last-checkpoint time far enough in the past.
        sched.last_checkpoint_time = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        assert_eq!(
            sched.should_checkpoint(0),
            Some(TriggerReason::TimeElapsed)
        );
    }

    #[test]
    fn test_scheduler_token_takes_priority() {
        let mut sched = CheckpointScheduler::new(10, Some(Duration::from_millis(1)));
        sched.last_checkpoint_time = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        // Token trigger fires first in evaluation order.
        assert_eq!(
            sched.should_checkpoint(10),
            Some(TriggerReason::TokenCount)
        );
    }

    // ── CheckpointManager basic tests ───────────────────────────────────

    #[test]
    fn test_manager_create_and_list() {
        let mut mgr = default_memory_manager();
        let state = sample_state(10, 4);
        let meta = mgr.create_checkpoint(&state, "model1").unwrap();
        assert_eq!(meta.token_position, 10);
        assert!(!meta.is_incremental);

        let list = mgr.list_checkpoints().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, meta.id);
    }

    #[test]
    fn test_manager_create_sequential_ids() {
        let mut mgr = default_memory_manager();
        let state = sample_state(5, 1);
        let m1 = mgr.create_checkpoint(&state, "m").unwrap();
        let m2 = mgr.create_checkpoint(&state, "m").unwrap();
        assert_ne!(m1.id, m2.id);
        assert!(m2.id > m1.id); // lexicographic ordering of ckpt-NNNNNN
    }

    #[test]
    fn test_manager_restore_full() {
        let mut mgr = default_memory_manager();
        let state = sample_state(20, 6);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_manager_restore_missing() {
        let mgr = default_memory_manager();
        assert!(mgr.restore_checkpoint("nonexistent").is_err());
    }

    #[test]
    fn test_manager_delete() {
        let mut mgr = default_memory_manager();
        let state = sample_state(5, 1);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        mgr.delete_checkpoint(&meta.id).unwrap();
        assert!(mgr.list_checkpoints().unwrap().is_empty());
    }

    #[test]
    fn test_manager_verify_valid() {
        let mut mgr = default_memory_manager();
        let state = sample_state(5, 2);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        assert!(mgr.verify_checkpoint(&meta.id).unwrap());
    }

    #[test]
    fn test_manager_verify_corrupt() {
        let mut store = MemoryCheckpointStorage::default();
        let state = sample_state(5, 2);
        let json = serde_json::to_vec(&state).unwrap();
        let checksum = crc32(&json);

        // Store with wrong checksum.
        let meta = CheckpointMetadata {
            id: "corrupt".into(),
            timestamp: 1,
            model_hash: String::new(),
            token_position: 5,
            kv_cache_size: 0,
            generation_params: HashMap::new(),
            is_incremental: false,
            base_checkpoint_id: None,
            compression: CompressionMode::None,
            checksum: checksum.wrapping_add(1), // intentionally wrong
        };
        let payload = compress(&json, CompressionMode::None);
        store.save(&meta, &payload).unwrap();

        let mgr =
            CheckpointManager::new(CheckpointConfig::default(), store).unwrap();
        assert!(!mgr.verify_checkpoint("corrupt").unwrap());
    }

    // ── Pruning tests ───────────────────────────────────────────────────

    #[test]
    fn test_manager_prune_respects_max() {
        let cfg = CheckpointConfig {
            max_checkpoints: 3,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(5, 1);
        for _ in 0..6 {
            mgr.create_checkpoint(&state, "m").unwrap();
        }
        let list = mgr.list_checkpoints().unwrap();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_manager_prune_keeps_newest() {
        let cfg = CheckpointConfig {
            max_checkpoints: 2,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(5, 1);
        let _m1 = mgr.create_checkpoint(&state, "m").unwrap();
        let _m2 = mgr.create_checkpoint(&state, "m").unwrap();
        let m3 = mgr.create_checkpoint(&state, "m").unwrap();

        let list = mgr.list_checkpoints().unwrap();
        assert_eq!(list.len(), 2);
        // Newest should still be present.
        assert!(list.iter().any(|m| m.id == m3.id));
    }

    #[test]
    fn test_manager_prune_returns_count() {
        let cfg = CheckpointConfig {
            max_checkpoints: 2,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(5, 1);
        for _ in 0..5 {
            mgr.create_checkpoint(&state, "m").unwrap();
        }
        // After creation the prune already ran; manually running should remove 0.
        let pruned = mgr.prune().unwrap();
        assert_eq!(pruned, 0);
    }

    // ── Incremental checkpoint tests ────────────────────────────────────

    #[test]
    fn test_manager_incremental_creates_diff() {
        let cfg = CheckpointConfig {
            enable_incremental: true,
            full_checkpoint_interval: 5,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state1 = sample_state(10, 4);
        let m1 = mgr.create_checkpoint(&state1, "m").unwrap();
        assert!(!m1.is_incremental);

        let mut state2 = state1.clone();
        state2.token_ids.push(99);
        let m2 = mgr.create_checkpoint(&state2, "m").unwrap();
        assert!(m2.is_incremental);
        assert_eq!(m2.base_checkpoint_id.as_deref(), Some(&*m1.id));
    }

    #[test]
    fn test_manager_incremental_restore() {
        let cfg = CheckpointConfig {
            enable_incremental: true,
            full_checkpoint_interval: 5,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state1 = sample_state(10, 4);
        mgr.create_checkpoint(&state1, "m").unwrap();

        let mut state2 = state1.clone();
        state2.token_ids.push(123);
        state2.kv_cache_entries[0].key_data = vec![77.0; 4];
        let m2 = mgr.create_checkpoint(&state2, "m").unwrap();

        let restored = mgr.restore_checkpoint(&m2.id).unwrap();
        assert_eq!(restored, state2);
    }

    #[test]
    fn test_manager_incremental_full_after_interval() {
        let cfg = CheckpointConfig {
            enable_incremental: true,
            full_checkpoint_interval: 2,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(5, 2);

        let m1 = mgr.create_checkpoint(&state, "m").unwrap();
        assert!(!m1.is_incremental); // first is always full

        let m2 = mgr.create_checkpoint(&state, "m").unwrap();
        assert!(m2.is_incremental);

        let m3 = mgr.create_checkpoint(&state, "m").unwrap();
        assert!(m3.is_incremental);

        // After full_checkpoint_interval (2) incrementals, next should be full.
        let m4 = mgr.create_checkpoint(&state, "m").unwrap();
        assert!(!m4.is_incremental);
    }

    // ── Compression round-trip through manager ──────────────────────────

    #[test]
    fn test_manager_compression_zstd_roundtrip() {
        let cfg = CheckpointConfig {
            compression: CompressionMode::Zstd,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(15, 3);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        assert_eq!(meta.compression, CompressionMode::Zstd);
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_manager_compression_lz4_roundtrip() {
        let cfg = CheckpointConfig {
            compression: CompressionMode::Lz4,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(15, 3);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_manager_compression_snappy_roundtrip() {
        let cfg = CheckpointConfig {
            compression: CompressionMode::Snappy,
            ..Default::default()
        };
        let mut mgr = memory_manager(cfg);
        let state = sample_state(15, 3);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    // ── File storage round-trip through manager ─────────────────────────

    #[test]
    fn test_manager_file_storage_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = default_config_with_dir(dir.path());
        let storage = FileCheckpointStorage::new(dir.path()).unwrap();
        let mut mgr = CheckpointManager::new(cfg, storage).unwrap();

        let state = sample_state(12, 3);
        let meta = mgr.create_checkpoint(&state, "hash1").unwrap();
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_manager_file_storage_list_and_prune() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = CheckpointConfig {
            checkpoint_dir: dir.path().to_path_buf(),
            max_checkpoints: 2,
            ..Default::default()
        };
        let storage = FileCheckpointStorage::new(dir.path()).unwrap();
        let mut mgr = CheckpointManager::new(cfg, storage).unwrap();

        let state = sample_state(5, 1);
        for _ in 0..4 {
            mgr.create_checkpoint(&state, "h").unwrap();
        }
        assert_eq!(mgr.list_checkpoints().unwrap().len(), 2);
    }

    // ── Error handling tests ────────────────────────────────────────────

    #[test]
    fn test_error_display_io() {
        let err = CheckpointError::Io(io::Error::new(io::ErrorKind::NotFound, "gone"));
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_error_display_serde() {
        let err = CheckpointError::Serde("bad json".into());
        assert!(err.to_string().contains("serialisation error"));
    }

    #[test]
    fn test_error_display_not_found() {
        let err = CheckpointError::NotFound("ckpt-1".into());
        assert!(err.to_string().contains("ckpt-1"));
    }

    #[test]
    fn test_error_display_corrupt() {
        let err = CheckpointError::CorruptCheckpoint("bad crc".into());
        assert!(err.to_string().contains("corrupt"));
    }

    #[test]
    fn test_error_display_invalid_config() {
        let err = CheckpointError::InvalidConfig("zero".into());
        assert!(err.to_string().contains("invalid config"));
    }

    #[test]
    fn test_error_display_missing_base() {
        let err = CheckpointError::MissingBase("base-1".into());
        assert!(err.to_string().contains("missing base"));
    }

    #[test]
    fn test_error_source_io() {
        let inner = io::Error::other("oops");
        let err = CheckpointError::Io(inner);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_error_source_non_io() {
        let err = CheckpointError::Serde("x".into());
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "nope");
        let err: CheckpointError = io_err.into();
        assert!(matches!(err, CheckpointError::Io(_)));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_err = serde_json::from_str::<String>("not json").unwrap_err();
        let err: CheckpointError = json_err.into();
        assert!(matches!(err, CheckpointError::Serde(_)));
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_empty_state_roundtrip() {
        let mut mgr = default_memory_manager();
        let state = InferenceState {
            token_ids: vec![],
            kv_cache_entries: vec![],
            rng_state: vec![],
            sampling_state: HashMap::new(),
        };
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_zero_kv_cache_roundtrip() {
        let mut mgr = default_memory_manager();
        let state = sample_state(50, 0);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        let restored = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_huge_token_count_metadata() {
        let mut mgr = default_memory_manager();
        let mut state = sample_state(0, 1);
        state.token_ids = (0..1_000_000u32).collect();
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        assert_eq!(meta.token_position, 1_000_000);
    }

    #[test]
    fn test_multiple_restore_same_checkpoint() {
        let mut mgr = default_memory_manager();
        let state = sample_state(10, 4);
        let meta = mgr.create_checkpoint(&state, "m").unwrap();
        let r1 = mgr.restore_checkpoint(&meta.id).unwrap();
        let r2 = mgr.restore_checkpoint(&meta.id).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_manager_config_accessor() {
        let cfg = CheckpointConfig {
            max_checkpoints: 42,
            ..Default::default()
        };
        let mgr = memory_manager(cfg);
        assert_eq!(mgr.config().max_checkpoints, 42);
    }

    #[test]
    fn test_manager_scheduler_accessor() {
        let cfg = CheckpointConfig {
            auto_save_interval_tokens: 200,
            ..Default::default()
        };
        let mgr = memory_manager(cfg);
        assert_eq!(
            mgr.scheduler().should_checkpoint(200),
            Some(TriggerReason::TokenCount)
        );
    }
}
