use anyhow::{Result, anyhow, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

/// Parse the first 24 bytes of a GGUF file.
pub fn parse_header(buf: &[u8]) -> Result<GgufHeader> {
    ensure!(buf.len() >= 24, "header too short: got {}", buf.len());

    let magic = &buf[0..4];
    ensure!(magic == b"GGUF", "bad magic: {:02x?}", magic);

    let version =
        u32::from_le_bytes(buf[4..8].try_into().map_err(|_| anyhow!("bad version bytes"))?);
    ensure!((1..=3).contains(&version), "unsupported GGUF version: {version}");
    let n_tensors =
        u64::from_le_bytes(buf[8..16].try_into().map_err(|_| anyhow!("bad n_tensors bytes"))?);
    let n_kv = u64::from_le_bytes(buf[16..24].try_into().map_err(|_| anyhow!("bad n_kv bytes"))?);

    Ok(GgufHeader { version, n_tensors, n_kv })
}

#[cfg(feature = "rt-tokio")]
pub async fn read_header(path: impl AsRef<std::path::Path>) -> Result<GgufHeader> {
    use tokio::io::AsyncReadExt;

    let mut f = tokio::fs::File::open(path).await?;
    let mut buf = [0u8; 24];
    f.read_exact(&mut buf).await?; // EOF → clean error
    parse_header(&buf)
}
