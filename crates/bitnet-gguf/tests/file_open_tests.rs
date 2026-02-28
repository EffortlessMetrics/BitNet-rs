//! File-backed GGUF tests using the checked-in `mini.gguf` fixture.
//!
//! The fixture at `tests/models/mini.gguf` is a 224-byte synthetic GGUF v3
//! file containing only metadata (0 tensors) and is safe to commit to the
//! repository.  These tests exercise the real `open()` path without requiring
//! an actual model download.

use std::path::Path;

use bitnet_gguf::open;

/// Resolve the mini.gguf path relative to the workspace root (Cargo sets
/// `CARGO_MANIFEST_DIR` at compile time).
fn mini_gguf_path() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is the crate root.  We need to walk up two levels to
    // reach the workspace root, then descend into tests/models.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .join("..") // crates/ → workspace root
        .join("..") // (extra level since crate is at crates/bitnet-gguf)
        .join("tests")
        .join("models")
        .join("mini.gguf")
        .canonicalize()
        .expect("mini.gguf fixture must exist at tests/models/mini.gguf")
}

#[test]
fn open_mini_gguf_succeeds() {
    let path = mini_gguf_path();
    let info = open(&path).expect("open() should succeed on valid mini.gguf");
    // mini.gguf is v3
    assert_eq!(info.version, 3, "expected GGUF v3");
    // mini.gguf has 0 tensors (metadata-only synthetic fixture)
    assert_eq!(info.tensor_count, 0, "expected 0 tensors in mini.gguf");
    // mini.gguf has 4 metadata key-value entries
    assert_eq!(info.metadata_count, 4, "expected 4 metadata entries");
}

#[test]
fn open_mini_gguf_snapshot() {
    let path = mini_gguf_path();
    let info = open(&path).expect("open() should succeed");
    let summary = format!(
        "version={} tensors={} metadata={} alignment={}",
        info.version, info.tensor_count, info.metadata_count, info.alignment
    );
    insta::assert_snapshot!("mini_gguf_file_info_summary", summary);
}

#[test]
fn open_nonexistent_path_returns_error() {
    let result = open(Path::new("/nonexistent/path/that/does/not/exist.gguf"));
    assert!(result.is_err(), "expected error for nonexistent path");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("cannot open"), "unexpected error message: {msg}");
}

#[test]
fn open_file_with_wrong_magic_returns_error() {
    // Write a tiny temp file with wrong magic bytes.
    let dir = tempfile::tempdir().expect("tempdir should succeed");
    let bad_path = dir.path().join("bad.gguf");
    std::fs::write(&bad_path, b"NOPE\x00\x00\x00\x00").expect("write should succeed");

    let result = open(&bad_path);
    assert!(result.is_err(), "expected error for bad magic");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("magic") || msg.contains("GGUF"),
        "error should mention magic or GGUF: {msg}"
    );
}
