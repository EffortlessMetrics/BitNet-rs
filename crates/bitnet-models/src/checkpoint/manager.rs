//! Thread-safe checkpoint inventory management.

use crate::checkpoint::error::CheckpointError;
use crate::checkpoint::format::CheckpointFormat;
use crate::checkpoint::metadata::{CheckpointMetadata, compute_sha256, extract_metadata};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Thread-safe manager for an inventory of model checkpoints.
///
/// All public methods acquire the inner lock for the minimum required
/// duration, so the manager is safe for concurrent access from multiple
/// threads.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    /// Keyed by the canonical string representation of the file path.
    inventory: Arc<RwLock<HashMap<String, CheckpointMetadata>>>,
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self { inventory: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Register a checkpoint file. Extracts metadata from the file system and
    /// stores it in the inventory. Returns an error if the path is already
    /// registered.
    pub fn add(&self, path: &Path) -> Result<CheckpointMetadata, CheckpointError> {
        let key = inventory_key(path);
        let meta = extract_metadata(path)?;

        let mut inv = self.inventory.write().expect("lock poisoned");
        if inv.contains_key(&key) {
            return Err(CheckpointError::Duplicate(key));
        }
        inv.insert(key, meta.clone());
        Ok(meta)
    }

    /// Remove a checkpoint from the inventory (does **not** delete the file).
    pub fn remove(&self, path: &Path) -> Result<CheckpointMetadata, CheckpointError> {
        let key = inventory_key(path);
        let mut inv = self.inventory.write().expect("lock poisoned");
        inv.remove(&key).ok_or(CheckpointError::NotFound(key))
    }

    /// Return the number of checkpoints in the inventory.
    pub fn len(&self) -> usize {
        self.inventory.read().expect("lock poisoned").len()
    }

    /// Return `true` when the inventory is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve metadata for a specific path.
    pub fn get(&self, path: &Path) -> Option<CheckpointMetadata> {
        let key = inventory_key(path);
        self.inventory.read().expect("lock poisoned").get(&key).cloned()
    }

    /// List all registered checkpoints.
    pub fn list(&self) -> Vec<CheckpointMetadata> {
        self.inventory.read().expect("lock poisoned").values().cloned().collect()
    }

    /// Search by model name (case-insensitive substring match).
    pub fn search_by_name(&self, query: &str) -> Vec<CheckpointMetadata> {
        let q = query.to_lowercase();
        self.inventory
            .read()
            .expect("lock poisoned")
            .values()
            .filter(|m| m.model_name.to_lowercase().contains(&q))
            .cloned()
            .collect()
    }

    /// Filter inventory by checkpoint format.
    pub fn filter_by_format(&self, format: CheckpointFormat) -> Vec<CheckpointMetadata> {
        self.inventory
            .read()
            .expect("lock poisoned")
            .values()
            .filter(|m| m.format == format)
            .cloned()
            .collect()
    }

    /// Verify the integrity of a registered checkpoint by recomputing its
    /// SHA-256 digest and comparing against the stored hash.
    pub fn verify(&self, path: &Path) -> Result<bool, CheckpointError> {
        let meta = self
            .get(path)
            .ok_or_else(|| CheckpointError::NotFound(path.to_string_lossy().to_string()))?;
        let actual = compute_sha256(path)?;
        if actual != meta.hash {
            return Err(CheckpointError::HashMismatch {
                path: path.to_string_lossy().to_string(),
                expected: meta.hash.clone(),
                actual,
            });
        }
        Ok(true)
    }

    /// Scan a directory for checkpoint files and register every one that
    /// matches a known extension. Returns the number of files added.
    pub fn scan_directory(&self, dir: &Path) -> Result<usize, CheckpointError> {
        let entries = std::fs::read_dir(dir)?;
        let mut count = 0usize;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() || !has_checkpoint_extension(&path) {
                continue;
            }

            // Silently skip duplicates during scanning.
            if self.add(&path).is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }
}

/// Build a stable inventory key for a path.
///
/// We prefer canonicalized paths to deduplicate relative/absolute aliases that
/// point to the same file, but fall back to an absolute best-effort key when
/// canonicalization is unavailable (e.g. path does not currently exist).
fn inventory_key(path: &Path) -> String {
    let normalized = std::fs::canonicalize(path).unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir().map(|cwd| cwd.join(path)).unwrap_or_else(|_| path.to_path_buf())
        }
    });
    normalized.to_string_lossy().to_string()
}

fn has_checkpoint_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_lowercase().as_str(), "gguf" | "safetensors" | "pt" | "pth" | "bin"))
        .unwrap_or(false)
}
