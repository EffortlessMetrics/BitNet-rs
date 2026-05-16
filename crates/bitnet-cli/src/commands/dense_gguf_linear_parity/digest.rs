//! Stable SHA-256 helpers for dense GGUF diagnostics.
//!
//! This module owns all digest serialization choices used by the command
//! receipts, keeping hashing policy separate from fixture extraction and
//! receipt construction.

use anyhow::Result;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(super) fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn sha256_usize(values: &[usize]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update((*value as u64).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn sha256_u32(values: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn sha256_json(value: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(sha256_bytes(&bytes))
}
