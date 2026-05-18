//! Model checkpoint management for BitNet inference.
//!
//! Provides checkpoint format detection, metadata extraction, integrity
//! verification (SHA-256), and a thread-safe inventory for managing model
//! checkpoint files across supported formats (GGUF, SafeTensors, PyTorch,
//! Custom).
//!
//! The implementation is split by single responsibility:
//!
//! - `format` detects supported checkpoint encodings.
//! - `metadata` extracts filesystem metadata and integrity hashes.
//! - `manager` owns the thread-safe checkpoint inventory.
//! - `error` defines the shared error type.

mod error;
mod format;
mod manager;
mod metadata;

pub use error::CheckpointError;
pub use format::CheckpointFormat;
pub use manager::CheckpointManager;
pub use metadata::{CheckpointMetadata, compute_sha256, extract_metadata};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::{Path, PathBuf};
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
    fn inventory_rejects_duplicate_absolute_vs_relative_aliases() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "m.gguf", b"data");
        let relative = PathBuf::from("m.gguf");
        struct CwdGuard(PathBuf);
        impl Drop for CwdGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.0);
            }
        }

        let _guard = CwdGuard(std::env::current_dir().unwrap());
        std::env::set_current_dir(dir.path()).unwrap();

        let mgr = CheckpointManager::new();
        mgr.add(&relative).unwrap();
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

    #[test]
    fn metadata_path_is_canonicalized() {
        let dir = TempDir::new().unwrap();
        let p = temp_file(&dir, "canon.gguf", b"canonical_test");
        let meta = extract_metadata(&p).unwrap();
        assert_eq!(meta.path, p.canonicalize().unwrap());
    }
}
