use serde::Serialize;
use std::{fs, path::Path};

/// Atomically replace a file by writing bytes to a sibling temporary path and renaming.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;

    #[cfg(unix)]
    {
        if let Ok(file) = std::fs::File::open(&tmp) {
            file.sync_all()?;
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

/// Serialize a value to pretty JSON and atomically replace the target file.

/// Atomically replace `destination` by renaming an existing temporary file.
pub fn atomic_rename(temp_path: &Path, destination: &Path) -> std::io::Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    {
        if let Ok(file) = std::fs::File::open(temp_path) {
            file.sync_all()?;
        }
    }

    fs::rename(temp_path, destination)?;

    #[cfg(unix)]
    {
        if let Some(parent) = destination.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

pub fn atomic_write_json_pretty<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), AtomicJsonError> {
    let json = serde_json::to_vec_pretty(value)?;
    atomic_write(path, &json)?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum AtomicJsonError {
    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    #[test]
    fn writes_bytes_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.bin");

        atomic_write(&path, b"first").expect("write first");
        assert_eq!(fs::read_to_string(&path).expect("read first"), "first");

        atomic_write(&path, b"second").expect("write second");
        assert_eq!(fs::read_to_string(&path).expect("read second"), "second");
    }

    #[test]
    fn renames_temp_file_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let temp = dir.path().join("out.tmp");
        let final_path = dir.path().join("out.bin");

        File::create(&temp).expect("create temp");
        fs::write(&temp, b"payload").expect("write temp");
        atomic_rename(&temp, &final_path).expect("rename");

        assert!(!temp.exists());
        assert_eq!(fs::read_to_string(&final_path).expect("read"), "payload");
    }

    #[test]
    fn writes_pretty_json_atomically() {
        #[derive(Serialize)]
        struct Payload {
            answer: u32,
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("payload.json");

        atomic_write_json_pretty(&path, &Payload { answer: 42 }).expect("write json");
        let written = fs::read_to_string(&path).expect("read json");
        assert!(written.contains("\"answer\": 42"));
    }
}
