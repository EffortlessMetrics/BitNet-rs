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
}
