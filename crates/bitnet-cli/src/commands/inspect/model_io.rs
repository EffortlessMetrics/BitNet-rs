//! File loading and integrity hashing for the inspect command.
//!
//! Single responsibility: open a model file, memory-map it, and compute its
//! SHA256 digest from the mmapped slice. Reusing the same mmap for both the
//! hash and the downstream GGUF reader avoids a second pass over the bytes.

use anyhow::{Context, Result};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::Path;

/// A model file that has been opened, memory-mapped, and hashed.
pub(crate) struct LoadedModel {
    pub(crate) mmap: Mmap,
    pub(crate) sha256: String,
}

/// Open `path`, memory-map it, and compute its SHA256 digest.
pub(crate) fn open_and_hash(path: &Path) -> Result<LoadedModel> {
    let file =
        File::open(path).with_context(|| format!("Failed to open model: {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file)? };

    let mut hasher = Sha256::new();
    hasher.update(&mmap);
    let hash = hasher.finalize();
    let sha256 = format!("{:x}", hash);

    Ok(LoadedModel { mmap, sha256 })
}
