use serde::{Deserialize, Serialize};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

// Helper constants and functions for KV reading
const MAX_KEY_LEN: u64 = 1024 * 1024; // 1 MiB
const MAX_STR_LEN: u64 = 10 * 1024 * 1024; // 10 MiB
/// Maximum number of array elements to return in samples.
/// For array-valued KVs, returns at most 256 elements (full payload is consumed
/// so subsequent KVs parse correctly).
const ARRAY_SAMPLE_LIMIT: usize = 256; // cap returned items per array

#[inline]
fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64_le(r)?;
    if len > MAX_STR_LEN {
        return Err(GgufError::StringTooLarge(len));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::Malformed)
}

#[inline]
fn value_from_type_scalar<R: Read>(r: &mut R, ty: u32) -> Result<GgufValue> {
    Ok(match ty {
        0 => {
            let mut b = [0];
            r.read_exact(&mut b)?;
            GgufValue::U8(b[0])
        }
        1 => {
            let mut b = [0];
            r.read_exact(&mut b)?;
            GgufValue::I8(b[0] as i8)
        }
        2 => {
            let mut b = [0; 2];
            r.read_exact(&mut b)?;
            GgufValue::U16(u16::from_le_bytes(b))
        }
        3 => {
            let mut b = [0; 2];
            r.read_exact(&mut b)?;
            GgufValue::I16(i16::from_le_bytes(b))
        }
        4 => {
            let mut b = [0; 4];
            r.read_exact(&mut b)?;
            GgufValue::U32(u32::from_le_bytes(b))
        }
        5 => {
            let mut b = [0; 4];
            r.read_exact(&mut b)?;
            GgufValue::I32(i32::from_le_bytes(b))
        }
        6 => {
            let mut b = [0; 4];
            r.read_exact(&mut b)?;
            GgufValue::F32(f32::from_le_bytes(b))
        }
        7 => {
            let mut b = [0];
            r.read_exact(&mut b)?;
            GgufValue::Bool(b[0] != 0)
        }
        10 => {
            let mut b = [0; 8];
            r.read_exact(&mut b)?;
            GgufValue::U64(u64::from_le_bytes(b))
        }
        11 => {
            let mut b = [0; 8];
            r.read_exact(&mut b)?;
            GgufValue::I64(i64::from_le_bytes(b))
        }
        12 => {
            let mut b = [0; 8];
            r.read_exact(&mut b)?;
            GgufValue::F64(f64::from_le_bytes(b))
        }
        _ => return Err(GgufError::InvalidKvType(ty)),
    })
}

#[inline]
fn scalar_size_bytes(ty: u32) -> Option<usize> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_array_value<R: Read + Seek>(r: &mut R) -> Result<Vec<GgufValue>> {
    // element type and length, then elements
    let elem_ty = read_u32_le(r)?;
    let len = read_u64_le(r)?;

    // We return at most ARRAY_SAMPLE_LIMIT items to bound memory.
    let keep = (len as usize).min(ARRAY_SAMPLE_LIMIT);
    let mut out = Vec::with_capacity(keep);

    if elem_ty == 8 {
        // string array: each item is [u64 len][bytes]
        for i in 0..len {
            let slen = read_u64_le(r)?;
            if slen > MAX_STR_LEN {
                return Err(GgufError::StringTooLarge(slen));
            }
            if (i as usize) < keep {
                let mut sbuf = vec![0u8; slen as usize];
                r.read_exact(&mut sbuf)?;
                let s = String::from_utf8(sbuf).map_err(|_| GgufError::Malformed)?;
                out.push(GgufValue::String(s));
            } else {
                // skip bytes for the remainder efficiently
                r.seek(SeekFrom::Current(slen as i64))?;
            }
        }
        return Ok(out);
    }

    // numeric/bool arrays: fixed-size elements
    if let Some(sz) = scalar_size_bytes(elem_ty) {
        for i in 0..len {
            if (i as usize) < keep {
                out.push(value_from_type_scalar(r, elem_ty)?);
            } else {
                // skip remaining elements in one shot
                let rem = len - i;
                let skip = (rem as u128) * (sz as u128); // avoid overflow
                if skip > i64::MAX as u128 {
                    // If skip is too large, do it in chunks
                    let chunk_size = 1_000_000_000; // 1GB chunks
                    let mut remaining = skip;
                    while remaining > 0 {
                        let to_skip = remaining.min(chunk_size as u128) as i64;
                        r.seek(SeekFrom::Current(to_skip))?;
                        remaining -= to_skip as u128;
                    }
                } else {
                    r.seek(SeekFrom::Current(skip as i64))?;
                }
                break;
            }
        }
        return Ok(out);
    }

    Err(GgufError::InvalidKvType(elem_ty))
}

/// Read KV pairs from a GGUF file
pub fn read_kv_pairs(
    path: impl AsRef<std::path::Path>,
    limit: Option<usize>,
) -> Result<Vec<GgufKv>> {
    let f = std::fs::File::open(path)?;
    let mut r = BufReader::new(f);

    // read header
    let mut header_buf = [0u8; GGUF_HEADER_LEN];
    r.read_exact(&mut header_buf)?;
    let header = parse_header(&header_buf)?;

    let mut kvs = Vec::new();
    #[allow(clippy::cast_possible_truncation)] // n_kv is a metadata count, won't exceed usize
    let n_kv = limit.map_or(header.n_kv as usize, |l| l.min(header.n_kv as usize));

    for _ in 0..n_kv {
        // key
        let key_len = read_u64_le(&mut r)?;
        if key_len > MAX_KEY_LEN {
            return Err(GgufError::StringTooLarge(key_len));
        }
        let mut key_buf = vec![0u8; key_len as usize];
        r.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf).map_err(|_| GgufError::Malformed)?;

        // type
        let value_type = read_u32_le(&mut r)?;

        // value
        let value = match value_type {
            8 => GgufValue::String(read_string(&mut r)?),
            9 => GgufValue::Array(read_array_value(&mut r)?),
            ty => value_from_type_scalar(&mut r, ty)?,
        };

        kvs.push(GgufKv { key, value });

        if kvs.len() >= limit.unwrap_or(usize::MAX) {
            break;
        }
    }

    Ok(kvs)
}
