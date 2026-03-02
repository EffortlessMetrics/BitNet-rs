use std::{fs, path::Path};

/// Atomically writes bytes to `path` via a sibling temporary file.
///
/// On Unix, this also fsyncs the temporary file and its parent directory to
/// improve durability guarantees for metadata updates.
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
    use std::fs;

    #[test]
    fn writes_file_contents() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("meta.txt");

        atomic_write(&path, b"etag-value").expect("write succeeds");

        let got = fs::read(&path).expect("read written file");
        assert_eq!(got, b"etag-value");
    }

    #[test]
    fn overwrite_is_atomic_and_removes_tmp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("meta.txt");
        let tmp = dir.path().join("meta.tmp");

        atomic_write(&path, b"old").expect("initial write");
        atomic_write(&path, b"new").expect("overwrite write");

        let got = fs::read(&path).expect("read overwritten file");
        assert_eq!(got, b"new");
        assert!(!tmp.exists(), "temporary file should not remain after rename");
    }
}
