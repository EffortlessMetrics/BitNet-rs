//! Atomic file persistence helpers.
//! This crate keeps small file-write durability mechanics isolated from higher-level
//! domains like downloads and receipts.

use std::path::Path;

/// Atomically write bytes to `path` by writing a sibling temporary file then renaming.
///
/// On Unix, this fsyncs the temporary file and containing directory to reduce the risk
/// of data loss on sudden power failure.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;

    #[cfg(unix)]
    {
        if let Ok(file) = std::fs::File::open(&tmp) {
            file.sync_all()?;
        }
    }

    std::fs::rename(&tmp, path)?;

    #[cfg(unix)]
    {
        if let Some(parent) = path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
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
    fn atomic_write_persists_bytes() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("value.json");

        atomic_write(&path, br#"{"ok":true}"#).expect("write");
        let actual = std::fs::read_to_string(&path).expect("read");

        assert_eq!(actual, r#"{"ok":true}"#);
    }

    #[test]
    fn atomic_write_overwrites_existing_file() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("value.txt");
        std::fs::write(&path, "old").expect("seed file");

        atomic_write(&path, b"new").expect("write");

        assert_eq!(std::fs::read_to_string(&path).expect("read"), "new");
    }
}
