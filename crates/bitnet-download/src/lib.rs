use std::{fs, path::Path};

pub use bitnet_download_core::{
    DownloadValidationError, exp_backoff_ms, parse_content_range_total, retry_after_secs,
    retry_after_secs_at, validate_downloaded_len,
};

/// Returns true when download logic should operate in offline mode.
#[must_use]
pub fn offline_enabled(cli_offline: bool) -> bool {
    cli_offline || std::env::var("BITNET_OFFLINE").as_deref() == Ok("1")
}

/// Atomic write helper for small metadata files (etag/last-modified).
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
    use super::*;
    use std::io::Read;

    #[test]
    fn offline_enabled_obeys_cli_and_env() {
        let key = "BITNET_OFFLINE";

        assert!(offline_enabled(true));

        unsafe {
            std::env::remove_var(key);
        }
        assert!(!offline_enabled(false));

        unsafe {
            std::env::set_var(key, "1");
        }
        assert!(offline_enabled(false));

        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn atomic_write_replaces_file_contents() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("meta.etag");

        atomic_write(&path, b"old").expect("initial write");
        atomic_write(&path, b"new").expect("replacement write");

        let mut file = std::fs::File::open(&path).expect("open written file");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("read file");

        assert_eq!(buf, b"new");
        assert!(!path.with_extension("tmp").exists());
    }
}
