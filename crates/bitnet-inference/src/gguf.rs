use std::io;
use thiserror::Error;

/// Errors returned when reading/parsing a GGUF header.
#[non_exhaustive]
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
    #[error("invalid KV type: {0}")]
    InvalidKvType(u32),
    #[error("string too large: {0} bytes")]
    StringTooLarge(u64),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, GgufError>;

/// Minimal GGUF header (first 24 bytes).
/// Use `read_header_blocking` for CLIs or `read_header` for async contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

pub const GGUF_HEADER_LEN: usize = 24;

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

/// GGUF value types according to the spec
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// A key-value pair from the GGUF metadata
#[derive(Debug, Clone)]
pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

/// Read KV pairs from a GGUF file
pub fn read_kv_pairs(
    path: impl AsRef<std::path::Path>,
    limit: Option<usize>,
) -> Result<Vec<GgufKv>> {
    use std::io::{Read, Seek, SeekFrom};
    
    let mut f = std::fs::File::open(path)?;
    
    // Skip the header we already validated
    f.seek(SeekFrom::Start(GGUF_HEADER_LEN as u64))?;
    
    // Read n_kv from header first
    let mut header_buf = [0u8; GGUF_HEADER_LEN];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut header_buf)?;
    let header = parse_header(&header_buf)?;
    
    // Now position after header
    f.seek(SeekFrom::Start(GGUF_HEADER_LEN as u64))?;
    
    let mut kvs = Vec::new();
    let n_kv = limit.map(|l| l.min(header.n_kv as usize)).unwrap_or(header.n_kv as usize);
    
    for _ in 0..n_kv {
        // Read key length
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let key_len = u64::from_le_bytes(len_buf);
        
        if key_len > 1024 * 1024 {  // 1MB sanity limit for keys
            return Err(GgufError::StringTooLarge(key_len));
        }
        
        // Read key
        let mut key_buf = vec![0u8; key_len as usize];
        f.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf).map_err(|_| GgufError::Malformed)?;
        
        // Read value type
        let mut type_buf = [0u8; 4];
        f.read_exact(&mut type_buf)?;
        let value_type = u32::from_le_bytes(type_buf);
        
        // Read value based on type
        let value = match value_type {
            0 => {  // UINT8
                let mut buf = [0u8; 1];
                f.read_exact(&mut buf)?;
                GgufValue::U8(buf[0])
            }
            1 => {  // INT8
                let mut buf = [0u8; 1];
                f.read_exact(&mut buf)?;
                GgufValue::I8(buf[0] as i8)
            }
            2 => {  // UINT16
                let mut buf = [0u8; 2];
                f.read_exact(&mut buf)?;
                GgufValue::U16(u16::from_le_bytes(buf))
            }
            3 => {  // INT16
                let mut buf = [0u8; 2];
                f.read_exact(&mut buf)?;
                GgufValue::I16(i16::from_le_bytes(buf))
            }
            4 => {  // UINT32
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                GgufValue::U32(u32::from_le_bytes(buf))
            }
            5 => {  // INT32
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                GgufValue::I32(i32::from_le_bytes(buf))
            }
            6 => {  // FLOAT32
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                GgufValue::F32(f32::from_le_bytes(buf))
            }
            7 => {  // BOOL
                let mut buf = [0u8; 1];
                f.read_exact(&mut buf)?;
                GgufValue::Bool(buf[0] != 0)
            }
            8 => {  // STRING
                let mut len_buf = [0u8; 8];
                f.read_exact(&mut len_buf)?;
                let str_len = u64::from_le_bytes(len_buf);
                
                if str_len > 10 * 1024 * 1024 {  // 10MB sanity limit
                    return Err(GgufError::StringTooLarge(str_len));
                }
                
                let mut str_buf = vec![0u8; str_len as usize];
                f.read_exact(&mut str_buf)?;
                GgufValue::String(String::from_utf8(str_buf).map_err(|_| GgufError::Malformed)?)
            }
            9 => {  // ARRAY
                // For now, skip arrays as they're complex
                // Would need recursive parsing
                continue;
            }
            10 => {  // UINT64
                let mut buf = [0u8; 8];
                f.read_exact(&mut buf)?;
                GgufValue::U64(u64::from_le_bytes(buf))
            }
            11 => {  // INT64
                let mut buf = [0u8; 8];
                f.read_exact(&mut buf)?;
                GgufValue::I64(i64::from_le_bytes(buf))
            }
            12 => {  // FLOAT64
                let mut buf = [0u8; 8];
                f.read_exact(&mut buf)?;
                GgufValue::F64(f64::from_le_bytes(buf))
            }
            _ => return Err(GgufError::InvalidKvType(value_type)),
        };
        
        kvs.push(GgufKv { key, value });
        
        if kvs.len() >= limit.unwrap_or(usize::MAX) {
            break;
        }
    }
    
    Ok(kvs)
}
