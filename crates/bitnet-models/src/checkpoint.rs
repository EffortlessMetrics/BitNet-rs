//! Model checkpoint management for BitNet inference.
//!
//! Provides checkpoint format detection, metadata extraction, integrity
//! verification (SHA-256), and a thread-safe inventory for managing model
//! checkpoint files across supported formats (GGUF, SafeTensors, PyTorch,
//! Custom).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// CheckpointFormat
// ---------------------------------------------------------------------------

/// Supported model checkpoint formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFormat {
    /// GGUF (llama.cpp / ggml ecosystem).
    Gguf,
    /// SafeTensors (HuggingFace standard).
    SafeTensors,
    /// PyTorch serialised checkpoint (`.pt` / `.bin` / `.pth`).
    PyTorch,
    /// User-defined / unrecognised format.
    Custom,
}

impl CheckpointFormat {
    /// Detect checkpoint format from a file path using extension heuristics
    /// followed by a header probe.
    pub fn detect(path: &Path) -> Self {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "gguf" => return Self::Gguf,
                "safetensors" => return Self::SafeTensors,
                "pt" | "pth" | "bin" => return Self::PyTorch,
                _ => {}
            }
        }
        // Fall back to header magic bytes when extension is absent or
        // unrecognised.
        Self::detect_from_header(path).unwrap_or(Self::Custom)
    }

    /// Inspect the first bytes of `path` for known magic values.
    fn detect_from_header(path: &Path) -> Option<Self> {
        let mut file = std::fs::File::open(path).ok()?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header).ok()?;

        // GGUF v3 magic: "GGUF" as LE u32 = 0x46475547
        if header[..4] == [0x47, 0x47, 0x55, 0x46] {
            return Some(Self::Gguf);
        }
        // SafeTensors files start with a little-endian u64 length followed by
        // JSON — the first byte is usually a small number (< 256) while bytes
        // 4..8 are zero for any header < 4 GiB, and the JSON typically starts
        // with '{'. We use a lightweight heuristic: if byte-8 would be '{'
        // (0x7b) we treat it as SafeTensors.
        if header[0] > 0 && header[4..8] == [0, 0, 0, 0] {
            // Read one more byte to check for JSON open brace.
            let mut json_byte = [0u8; 1];
            if file.read_exact(&mut json_byte).is_ok() && json_byte[0] == b'{' {
                return Some(Self::SafeTensors);
            }
        }
        // PyTorch ZIP-based checkpoints start with the PK magic.
        if header[..2] == [0x50, 0x4B] {
            return Some(Self::PyTorch);
        }
        None
    }

    /// Human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::PyTorch => "PyTorch",
            Self::Custom => "Custom",
        }
    }
}

impl std::fmt::Display for CheckpointFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// CheckpointMetadata
// ---------------------------------------------------------------------------

/// Metadata associated with a single model checkpoint file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Detected or overridden checkpoint format.
    pub format: CheckpointFormat,
    /// Model name (derived from the file stem by default).
    pub model_name: String,
    /// Optional version string.
    pub version: Option<String>,
    /// Timestamp when the metadata entry was created.
    pub created_at: SystemTime,
    /// File size in bytes.
    pub file_size: u64,
    /// SHA-256 hex digest of the file contents.
    pub hash: String,
    /// Canonical path to the checkpoint file.
    pub path: PathBuf,
    /// Last modification time reported by the filesystem.
    pub modified_at: Option<SystemTime>,
}

// ---------------------------------------------------------------------------
// CheckpointError
// ---------------------------------------------------------------------------

/// Errors specific to checkpoint management.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint not found: {0}")]
    NotFound(String),
    #[error("duplicate checkpoint: {0}")]
    Duplicate(String),
    #[error("hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch { path: String, expected: String, actual: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hex digest of a file using a streaming 1 MiB buffer.
pub fn compute_sha256(path: &Path) -> Result<String, CheckpointError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Derive a model name from a file path (uses the file stem).
fn model_name_from_path(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
}

/// Build [`CheckpointMetadata`] by inspecting a file on disk.
pub fn extract_metadata(path: &Path) -> Result<CheckpointMetadata, CheckpointError> {
    let meta = std::fs::metadata(path)?;
    let hash = compute_sha256(path)?;
    let format = CheckpointFormat::detect(path);
    let modified_at = meta.modified().ok();

    Ok(CheckpointMetadata {
        format,
        model_name: model_name_from_path(path),
        version: None,
        created_at: SystemTime::now(),
        file_size: meta.len(),
        hash,
        path: path.to_path_buf(),
        modified_at,
    })
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

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

    // -- mutating operations ------------------------------------------------

    /// Register a checkpoint file. Extracts metadata from the file system and
    /// stores it in the inventory. Returns an error if the path is already
    /// registered.
    pub fn add(&self, path: &Path) -> Result<CheckpointMetadata, CheckpointError> {
        let key = path.to_string_lossy().to_string();
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
        let key = path.to_string_lossy().to_string();
        let mut inv = self.inventory.write().expect("lock poisoned");
        inv.remove(&key).ok_or(CheckpointError::NotFound(key))
    }

    // -- read-only queries --------------------------------------------------

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
        let key = path.to_string_lossy().to_string();
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
            if !path.is_file() {
                continue;
            }
            let dominated = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    matches!(
                        e.to_lowercase().as_str(),
                        "gguf" | "safetensors" | "pt" | "pth" | "bin"
                    )
                })
                .unwrap_or(false);
            if dominated {
                // Silently skip duplicates during scanning.
                if self.add(&path).is_ok() {
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper: create a temp file with the given name and contents, return its
    /// path.
    fn temp_file(dir: &TempDir, name: &str, contents: &[u8]) -> PathBuf {
        let p = dir.path().join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(contents).unwrap();
        p
    }

    // -- CheckpointFormat detection from extension --------------------------

    #[test]
    fn format_detect_gguf_extension() {
        let p = Path::new("/tmp/model.gguf");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::Gguf);
    }

    #[test]
    fn format_detect_safetensors_extension() {
        let p = Path::new("/tmp/model.safetensors");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::SafeTensors);
    }

    #[test]
    fn format_detect_pytorch_pt_extension() {
        let p = Path::new("/tmp/model.pt");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::PyTorch);
    }

    #[test]
    fn format_detect_pytorch_pth_extension() {
        let p = Path::new("/tmp/weights.pth");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::PyTorch);
    }

    #[test]
    fn format_detect_pytorch_bin_extension() {
        let p = Path::new("/data/model.bin");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::PyTorch);
    }

    #[test]
    fn format_detect_unknown_extension_falls_back_to_custom() {
        let p = Path::new("/tmp/model.xyz");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::Custom);
    }

    #[test]
    fn format_detect_no_extension_falls_back_to_custom() {
        let p = Path::new("/tmp/model");
        assert_eq!(CheckpointFormat::detect(p), CheckpointFormat::Custom);
    }

    #[test]
    fn format_detect_case_insensitive() {
        assert_eq!(CheckpointFormat::detect(Path::new("m.GGUF")), CheckpointFormat::Gguf);
        assert_eq!(
            CheckpointFormat::detect(Path::new("m.SafeTensors")),
            CheckpointFormat::SafeTensors,
        );
    }

    #[test]
    fn format_detect_gguf_header_magic() {
        let dir = TempDir::new().unwrap();
        // GGUF magic: 0x47475546 LE followed by version 3
        let mut data = vec![0x47u8, 0x47, 0x55, 0x46]; // "GGUF"
        data.extend_from_slice(&[3, 0, 0, 0]); // version 3
        data.extend_from_slice(&[0u8; 64]); // padding
        let p = temp_file(&dir, "model.unknown", &data);
        assert_eq!(CheckpointFormat::detect(&p), CheckpointFormat::Gguf);
    }

    #[test]
    fn format_detect_pytorch_zip_header() {
        let dir = TempDir::new().unwrap();
        let mut data = vec![0x50u8, 0x4B, 0x03, 0x04]; // PK magic
        data.extend_from_slice(&[0u8; 64]);
        let p = temp_file(&dir, "model.unknown", &data);
        assert_eq!(CheckpointFormat::detect(&p), CheckpointFormat::PyTorch);
    }

    // -- CheckpointFormat Display / as_str ----------------------------------

    #[test]
    fn format_display_strings() {
        assert_eq!(CheckpointFormat::Gguf.as_str(), "GGUF");
        assert_eq!(CheckpointFormat::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(CheckpointFormat::PyTorch.to_string(), "PyTorch");
        assert_eq!(CheckpointFormat::Custom.to_string(), "Custom");
    }

    // -- CheckpointMetadata creation ----------------------------------------

    #[test]
    fn metadata_creation_and_serde_roundtrip() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "demo.gguf", b"fake gguf data for testing");
        let meta = extract_metadata(&p).unwrap();

        assert_eq!(meta.format, CheckpointFormat::Gguf);
        assert_eq!(meta.model_name, "demo");
        assert_eq!(meta.file_size, 26);
        assert!(!meta.hash.is_empty());

        // Roundtrip through JSON
        let json = serde_json::to_string(&meta).unwrap();
        let de: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(de.hash, meta.hash);
        assert_eq!(de.format, meta.format);
    }

    // -- SHA-256 hash -------------------------------------------------------

    #[test]
    fn hash_deterministic_for_same_content() {
        let dir = TempDir::new().unwrap();
        let p1 = temp_file(&dir, "a.bin", b"hello world");
        let p2 = temp_file(&dir, "b.bin", b"hello world");
        assert_eq!(compute_sha256(&p1).unwrap(), compute_sha256(&p2).unwrap());
    }

    #[test]
    fn hash_differs_for_different_content() {
        let dir = TempDir::new().unwrap();
        let p1 = temp_file(&dir, "a.bin", b"hello");
        let p2 = temp_file(&dir, "b.bin", b"world");
        assert_ne!(compute_sha256(&p1).unwrap(), compute_sha256(&p2).unwrap());
    }

    #[test]
    fn hash_known_value() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "known.bin", b"bitnet");
        let hash = compute_sha256(&p).unwrap();
        // Pre-computed: echo -n 'bitnet' | sha256sum
        assert_eq!(hash, "80656a6e019be5c15c71c5cba04b2324b286a1597de71429b3530e1a4c053422",);
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn hash_error_on_missing_file() {
        let result = compute_sha256(Path::new("/nonexistent/path/model.gguf"));
        assert!(result.is_err());
    }

    // -- Inventory operations -----------------------------------------------

    #[test]
    fn inventory_add_and_get() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "m.gguf", b"data");
        let mgr = CheckpointManager::new();
        let meta = mgr.add(&p).unwrap();
        assert_eq!(meta.model_name, "m");

        let fetched = mgr.get(&p).unwrap();
        assert_eq!(fetched.hash, meta.hash);
    }

    #[test]
    fn inventory_add_duplicate_errors() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "m.gguf", b"data");
        let mgr = CheckpointManager::new();
        mgr.add(&p).unwrap();
        assert!(matches!(mgr.add(&p), Err(CheckpointError::Duplicate(_))));
    }

    #[test]
    fn inventory_remove() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "m.gguf", b"data");
        let mgr = CheckpointManager::new();
        mgr.add(&p).unwrap();
        assert_eq!(mgr.len(), 1);

        mgr.remove(&p).unwrap();
        assert!(mgr.is_empty());
    }

    #[test]
    fn inventory_remove_missing_errors() {
        let mgr = CheckpointManager::new();
        assert!(matches!(mgr.remove(Path::new("/nope")), Err(CheckpointError::NotFound(_)),));
    }

    #[test]
    fn inventory_list_returns_all() {
        let dir = TempDir::new().unwrap();
        let mgr = CheckpointManager::new();
        for i in 0..3 {
            let p = temp_file(&dir, &format!("model{i}.gguf"), format!("d{i}").as_bytes());
            mgr.add(&p).unwrap();
        }
        assert_eq!(mgr.list().len(), 3);
    }

    // -- search / filter ----------------------------------------------------

    #[test]
    fn search_by_name_case_insensitive() {
        let dir = TempDir::new().unwrap();
        let mgr = CheckpointManager::new();
        let p = temp_file(&dir, "BitNet-Model.gguf", b"x");
        mgr.add(&p).unwrap();

        let results = mgr.search_by_name("bitnet");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model_name, "BitNet-Model");
    }

    #[test]
    fn search_by_name_no_match() {
        let dir = TempDir::new().unwrap();
        let mgr = CheckpointManager::new();
        let p = temp_file(&dir, "alpha.gguf", b"x");
        mgr.add(&p).unwrap();
        assert!(mgr.search_by_name("beta").is_empty());
    }

    #[test]
    fn filter_by_format() {
        let dir = TempDir::new().unwrap();
        let mgr = CheckpointManager::new();
        temp_file(&dir, "a.gguf", b"1");
        temp_file(&dir, "b.safetensors", b"2");
        temp_file(&dir, "c.pt", b"3");
        mgr.add(&dir.path().join("a.gguf")).unwrap();
        mgr.add(&dir.path().join("b.safetensors")).unwrap();
        mgr.add(&dir.path().join("c.pt")).unwrap();

        assert_eq!(mgr.filter_by_format(CheckpointFormat::Gguf).len(), 1);
        assert_eq!(mgr.filter_by_format(CheckpointFormat::SafeTensors).len(), 1);
        assert_eq!(mgr.filter_by_format(CheckpointFormat::PyTorch).len(), 1);
        assert_eq!(mgr.filter_by_format(CheckpointFormat::Custom).len(), 0);
    }

    // -- integrity verification ---------------------------------------------

    #[test]
    fn verify_passes_for_unmodified_file() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "ok.gguf", b"stable content");
        let mgr = CheckpointManager::new();
        mgr.add(&p).unwrap();
        assert!(mgr.verify(&p).unwrap());
    }

    #[test]
    fn verify_fails_after_modification() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "mut.gguf", b"original");
        let mgr = CheckpointManager::new();
        mgr.add(&p).unwrap();

        // Modify the file behind the manager's back.
        std::fs::write(&p, b"tampered").unwrap();
        assert!(matches!(mgr.verify(&p), Err(CheckpointError::HashMismatch { .. }),));
    }

    #[test]
    fn verify_errors_for_unregistered_path() {
        let mgr = CheckpointManager::new();
        assert!(matches!(mgr.verify(Path::new("/nope")), Err(CheckpointError::NotFound(_)),));
    }

    // -- directory scanning -------------------------------------------------

    #[test]
    fn scan_directory_finds_known_extensions() {
        let dir = TempDir::new().unwrap();
        temp_file(&dir, "a.gguf", b"1");
        temp_file(&dir, "b.safetensors", b"2");
        temp_file(&dir, "c.pt", b"3");
        temp_file(&dir, "readme.txt", b"skip me");

        let mgr = CheckpointManager::new();
        let added = mgr.scan_directory(dir.path()).unwrap();
        assert_eq!(added, 3);
        assert_eq!(mgr.len(), 3);
    }

    #[test]
    fn scan_empty_directory() {
        let dir = TempDir::new().unwrap();
        let mgr = CheckpointManager::new();
        assert_eq!(mgr.scan_directory(dir.path()).unwrap(), 0);
        assert!(mgr.is_empty());
    }

    // -- thread safety ------------------------------------------------------

    #[test]
    fn concurrent_adds_are_safe() {
        let dir = TempDir::new().unwrap();
        // Pre-create files
        for i in 0..8 {
            temp_file(&dir, &format!("t{i}.gguf"), format!("data{i}").as_bytes());
        }

        let mgr = CheckpointManager::new();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let mgr = mgr.clone();
                let path = dir.path().join(format!("t{i}.gguf"));
                std::thread::spawn(move || mgr.add(&path))
            })
            .collect();

        for h in handles {
            h.join().unwrap().unwrap();
        }
        assert_eq!(mgr.len(), 8);
    }

    #[test]
    fn concurrent_reads_while_writing() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "shared.gguf", b"shared");
        let mgr = CheckpointManager::new();
        mgr.add(&p).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let mgr = mgr.clone();
                let p = p.clone();
                std::thread::spawn(move || {
                    assert!(mgr.get(&p).is_some());
                    assert!(!mgr.list().is_empty());
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn add_missing_file_errors() {
        let mgr = CheckpointManager::new();
        assert!(mgr.add(Path::new("/does/not/exist.gguf")).is_err());
    }

    #[test]
    fn manager_default_is_empty() {
        let mgr = CheckpointManager::default();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn metadata_version_defaults_to_none() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "v.gguf", b"version_test");
        let meta = extract_metadata(&p).unwrap();
        assert!(meta.version.is_none());
    }

    #[test]
    fn metadata_modified_at_is_populated() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "ts.gguf", b"timestamp_test");
        let meta = extract_metadata(&p).unwrap();
        assert!(meta.modified_at.is_some());
    }
}
