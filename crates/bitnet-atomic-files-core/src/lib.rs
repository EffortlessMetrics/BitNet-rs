use std::{fs, path::Path};

/// Atomically writes bytes to `path` via a temporary sibling file and rename.
///
/// This helper is intended for small metadata/config files where all bytes are
/// already available in-memory.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;

    #[cfg(unix)]
    {
        if let Ok(f) = std::fs::File::open(&tmp) {
            f.sync_all()?;
        }
    }

    fs::rename(&tmp, path)?;

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
    fn writes_and_overwrites_file_atomically() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("meta.json");

        atomic_write(&path, br#"{"v":1}"#).expect("initial write");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "{\"v\":1}");

        atomic_write(&path, br#"{"v":2}"#).expect("overwrite write");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "{\"v\":2}");
    }
}
