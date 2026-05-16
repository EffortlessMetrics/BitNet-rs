use std::{fs, path::Path};

/// Atomically writes bytes to `path` by writing to a temporary sibling file then renaming.
///
/// On Unix platforms this best-effort syncs both the temp file and parent directory to improve
/// durability across sudden power loss.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;

    #[cfg(unix)]
    {
        if let Ok(f) = fs::File::open(&tmp) {
            f.sync_all()?;
        }
    }

    fs::rename(&tmp, path)?;

    #[cfg(unix)]
    {
        if let Some(parent) = path.parent()
            && let Ok(dir) = fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::atomic_write;

    #[test]
    fn writes_and_replaces() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("data.json");

        atomic_write(&path, b"v1").expect("first write");
        assert_eq!(std::fs::read(&path).expect("read v1"), b"v1");

        atomic_write(&path, b"v2").expect("second write");
        assert_eq!(std::fs::read(&path).expect("read v2"), b"v2");
    }

    #[test]
    fn writes_empty_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty.bin");

        atomic_write(&path, b"").expect("write empty");
        let data = std::fs::read(&path).expect("read");
        assert!(data.is_empty());
    }

    #[test]
    fn writes_binary_payload_byte_for_byte() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("payload.bin");
        let payload: Vec<u8> = (0..=255_u8).collect();

        atomic_write(&path, &payload).expect("write");
        let read_back = std::fs::read(&path).expect("read");
        assert_eq!(read_back, payload);
    }

    #[test]
    fn rename_replaces_existing_target() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.txt");

        std::fs::write(&path, b"old").expect("seed");
        atomic_write(&path, b"new").expect("overwrite");
        assert_eq!(std::fs::read(&path).expect("read"), b"new");
    }

    #[test]
    fn does_not_leave_temp_sibling_on_success() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("output.json");

        atomic_write(&path, b"{}").expect("write");

        // The implementation writes to "<path>.tmp" then renames into place.
        let tmp = path.with_extension("tmp");
        assert!(!tmp.exists(), "atomic_write must not leave behind the tmp sibling");
        assert!(path.exists(), "final path must exist");
    }

    #[test]
    fn fails_when_parent_directory_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing").join("file.txt");

        let err = atomic_write(&path, b"data").expect_err("parent dir does not exist");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
