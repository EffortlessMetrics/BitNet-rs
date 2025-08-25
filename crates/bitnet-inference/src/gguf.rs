use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("bad magic: {0:02x?}")]
    BadMagic([u8; 4]),
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("short header: read {0} bytes, need 24")]
    ShortHeader(usize),
    #[error("malformed header")]
    Malformed,
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

pub const GGUF_HEADER_LEN: usize = 24;
const MAX_REASONABLE: u64 = 10_000_000;

/// Parse the first 24 bytes of a GGUF file.
pub fn parse_header(buf: &[u8]) -> Result<GgufHeader> {
    if buf.len() < GGUF_HEADER_LEN {
        return Err(GgufError::ShortHeader(buf.len()));
    }

    let magic = <[u8; 4]>::try_from(&buf[0..4]).unwrap();
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic(magic));
    }

    let version = u32::from_le_bytes(buf[4..8].try_into().map_err(|_| GgufError::Malformed)?);
    if !(1..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let n_tensors = u64::from_le_bytes(buf[8..16].try_into().map_err(|_| GgufError::Malformed)?);
    let n_kv = u64::from_le_bytes(buf[16..24].try_into().map_err(|_| GgufError::Malformed)?);

    if n_tensors > MAX_REASONABLE || n_kv > MAX_REASONABLE {
        return Err(GgufError::Malformed);
    }

    Ok(GgufHeader { version, n_tensors, n_kv })
}

/// Read GGUF header asynchronously using Tokio
#[cfg(feature = "rt-tokio")]
pub async fn read_header(path: impl AsRef<std::path::Path>) -> Result<GgufHeader> {
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path).await?;
    let mut buf = [0u8; GGUF_HEADER_LEN];
    let n = f.read(&mut buf).await?;
    if n < GGUF_HEADER_LEN {
        return Err(GgufError::ShortHeader(n));
    }
    parse_header(&buf)
}

/// Read GGUF header synchronously for CLI/offline use
pub fn read_header_blocking(path: impl AsRef<std::path::Path>) -> Result<GgufHeader> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; GGUF_HEADER_LEN];
    let n = f.read(&mut buf)?;
    if n < GGUF_HEADER_LEN {
        return Err(GgufError::ShortHeader(n));
    }
    parse_header(&buf)
}
