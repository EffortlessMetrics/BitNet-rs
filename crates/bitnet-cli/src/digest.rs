//! Hashing helpers for stable receipt fields.

use anyhow::Result;
use sha2::{Digest, Sha256};

pub(crate) fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn sha256_token_ids(tokens: &[u32]) -> Result<String> {
    Ok(sha256_hex_bytes(&serde_json::to_vec(tokens)?))
}
