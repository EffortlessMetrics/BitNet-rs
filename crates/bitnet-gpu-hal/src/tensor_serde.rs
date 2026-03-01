//! Module stub - implementation pending merge from feature branch
//! Tensor serialization and deserialization across multiple formats.
//!
//! Supports Binary, `SafeTensors`, `NumPy`, GGUF, and JSON formats with optional
//! compression (LZ4, Zstd, Snappy) and SHA-256 integrity checksums.

// Serialization code legitimately uses small identifiers, casts, and hex constants.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::format_collect,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_pass_by_value,
    clippy::redundant_clone,
    clippy::manual_div_ceil,
    clippy::use_self,
    clippy::unnecessary_wraps,
    clippy::float_cmp
)]

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Cursor, Read, Write};

// ── Format Enumeration ──────────────────────────────────────────────────────

/// Supported tensor serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SerializationFormat {
    /// Raw binary with a compact header.
    Binary,
    /// HuggingFace SafeTensors (JSON header + raw data).
    SafeTensors,
    /// GGUF quantized format.
    Gguf,
    /// NumPy `.npy` format (magic + version + header + data).
    NumPy,
    /// JSON (base64-encoded data, mainly for debugging).
    Json,
}

impl fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary => write!(f, "binary"),
            Self::SafeTensors => write!(f, "safetensors"),
            Self::Gguf => write!(f, "gguf"),
            Self::NumPy => write!(f, "numpy"),
            Self::Json => write!(f, "json"),
        }
    }
}

// ── Data Types ──────────────────────────────────────────────────────────────

/// Element data types for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    BF16,
}

impl DType {
    /// Size of a single element in bytes.
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// NumPy dtype descriptor string (little-endian).
    #[must_use]
    pub const fn numpy_descr(self) -> &'static str {
        match self {
            Self::F16 => "<f2",
            Self::F32 => "<f4",
            Self::F64 => "<f8",
            Self::I8 => "|i1",
            Self::I16 => "<i2",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::U8 => "|u1",
            Self::U16 => "<u2",
            Self::U32 => "<u4",
            Self::U64 => "<u8",
            Self::BF16 => "<V2", // bfloat16 stored as raw 2-byte
        }
    }

    /// Parse from a NumPy dtype descriptor.
    pub fn from_numpy_descr(s: &str) -> Option<Self> {
        match s {
            "<f2" => Some(Self::F16),
            "<f4" => Some(Self::F32),
            "<f8" => Some(Self::F64),
            "|i1" => Some(Self::I8),
            "<i2" => Some(Self::I16),
            "<i4" => Some(Self::I32),
            "<i8" => Some(Self::I64),
            "|u1" => Some(Self::U8),
            "<u2" => Some(Self::U16),
            "<u4" => Some(Self::U32),
            "<u8" => Some(Self::U64),
            "<V2" => Some(Self::BF16),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::BF16 => "bf16",
        };
        write!(f, "{s}")
    }
}

// ── Endianness ──────────────────────────────────────────────────────────────

/// Byte order for multi-byte values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Endianness {
    #[default]
    Little,
    Big,
}

// ── Tensor Header ───────────────────────────────────────────────────────────

/// Metadata describing a serialized tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeader {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub endianness: Endianness,
}

impl TensorHeader {
    /// Create a new header, computing `byte_size` from shape and dtype.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> Self {
        let num_elements: usize = shape.iter().product();
        let byte_size = (num_elements * dtype.element_size()) as u64;
        Self {
            name: name.into(),
            shape,
            dtype,
            byte_offset: 0,
            byte_size,
            endianness: Endianness::Little,
        }
    }

    /// Total number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Validate that the provided data length matches expectations.
    pub fn validate_data_len(&self, data_len: usize) -> Result<(), SerdeError> {
        let expected = self.byte_size as usize;
        if data_len != expected {
            return Err(SerdeError::SizeMismatch { expected, actual: data_len });
        }
        Ok(())
    }
}

// ── Error Type ──────────────────────────────────────────────────────────────

/// Errors from tensor serialization / deserialization.
#[derive(Debug)]
pub enum SerdeError {
    Io(io::Error),
    InvalidMagic { expected: &'static [u8], actual: Vec<u8> },
    SizeMismatch { expected: usize, actual: usize },
    InvalidHeader(String),
    UnsupportedFormat(SerializationFormat),
    ChecksumMismatch { expected: String, actual: String },
    UnsupportedCompression(CompressionCodec),
    TensorNotFound(String),
}

impl fmt::Display for SerdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidMagic { expected, actual } => {
                write!(f, "invalid magic: expected {expected:?}, got {actual:?}")
            }
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected} bytes, got {actual}")
            }
            Self::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            Self::UnsupportedFormat(fmt_val) => {
                write!(f, "unsupported format: {fmt_val}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, got {actual}")
            }
            Self::UnsupportedCompression(c) => {
                write!(f, "unsupported compression: {c:?}")
            }
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
        }
    }
}

impl std::error::Error for SerdeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for SerdeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Compression ─────────────────────────────────────────────────────────────

/// Compression codecs for tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompressionCodec {
    #[default]
    None,
    Lz4,
    Zstd,
    Snappy,
}

impl CompressionCodec {
    /// Compress data using this codec (only `None` is built-in).
    pub fn compress(self, data: &[u8]) -> Result<Vec<u8>, SerdeError> {
        match self {
            Self::None => Ok(data.to_vec()),
            other => Err(SerdeError::UnsupportedCompression(other)),
        }
    }

    /// Decompress data using this codec.
    pub fn decompress(self, data: &[u8]) -> Result<Vec<u8>, SerdeError> {
        match self {
            Self::None => Ok(data.to_vec()),
            other => Err(SerdeError::UnsupportedCompression(other)),
        }
    }

    /// Codec tag byte for binary headers.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Zstd => 2,
            Self::Snappy => 3,
        }
    }

    /// Parse from tag byte.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Zstd),
            3 => Some(Self::Snappy),
            _ => None,
        }
    }
}

// ── Checksum ────────────────────────────────────────────────────────────────

/// SHA-256 checksum for tensor data integrity verification.
///
/// Uses a simple hand-rolled SHA-256 to avoid external dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorChecksum {
    digest: [u8; 32],
}

impl TensorChecksum {
    /// Compute SHA-256 checksum over the given data.
    #[must_use]
    pub fn compute(data: &[u8]) -> Self {
        Self { digest: sha256(data) }
    }

    /// Return the digest as a hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Parse from a hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 64 {
            return None;
        }
        let mut digest = [0u8; 32];
        for (i, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self { digest })
    }

    /// Verify that the checksum matches the given data.
    pub fn verify(&self, data: &[u8]) -> Result<(), SerdeError> {
        let actual = Self::compute(data);
        if self.digest != actual.digest {
            return Err(SerdeError::ChecksumMismatch {
                expected: self.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(())
    }

    /// Raw digest bytes.
    #[must_use]
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

/// Minimal SHA-256 implementation (no external deps).
#[allow(clippy::unreadable_literal, clippy::min_ident_chars, clippy::many_single_char_names)]
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Pre-processing: pad message
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit block
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    out
}

// ── Binary Format ───────────────────────────────────────────────────────────

/// Magic bytes identifying our binary tensor format.
const BINARY_MAGIC: &[u8; 4] = b"BTSR";
/// Current binary format version.
const BINARY_VERSION: u8 = 1;

/// DType tag for binary header.
fn dtype_to_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F16 => 0,
        DType::F32 => 1,
        DType::F64 => 2,
        DType::I8 => 3,
        DType::I16 => 4,
        DType::I32 => 5,
        DType::I64 => 6,
        DType::U8 => 7,
        DType::U16 => 8,
        DType::U32 => 9,
        DType::U64 => 10,
        DType::BF16 => 11,
    }
}

fn dtype_from_tag(tag: u8) -> Option<DType> {
    match tag {
        0 => Some(DType::F16),
        1 => Some(DType::F32),
        2 => Some(DType::F64),
        3 => Some(DType::I8),
        4 => Some(DType::I16),
        5 => Some(DType::I32),
        6 => Some(DType::I64),
        7 => Some(DType::U8),
        8 => Some(DType::U16),
        9 => Some(DType::U32),
        10 => Some(DType::U64),
        11 => Some(DType::BF16),
        _ => None,
    }
}

/// Serializer for the raw binary format.
///
/// Layout: `BTSR` magic | version(1) | dtype(1) | ndim(u32 LE) | shape(ndim × u64 LE) |
///         compression(1) | name_len(u32 LE) | name | data_len(u64 LE) | data
pub struct BinarySerializer;

impl BinarySerializer {
    /// Serialize a tensor header + data into the writer.
    pub fn serialize<W: Write>(
        writer: &mut W,
        header: &TensorHeader,
        data: &[u8],
        compression: CompressionCodec,
    ) -> Result<(), SerdeError> {
        header.validate_data_len(data.len())?;
        let compressed = compression.compress(data)?;

        writer.write_all(BINARY_MAGIC)?;
        writer.write_all(&[BINARY_VERSION])?;
        writer.write_all(&[dtype_to_tag(header.dtype)])?;

        let ndim = header.shape.len() as u32;
        writer.write_all(&ndim.to_le_bytes())?;
        for &dim in &header.shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }

        writer.write_all(&[compression.tag()])?;

        let name_bytes = header.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;

        writer.write_all(&(compressed.len() as u64).to_le_bytes())?;
        writer.write_all(&compressed)?;

        Ok(())
    }
}

/// Deserializer for the raw binary format.
pub struct BinaryDeserializer;

impl BinaryDeserializer {
    /// Deserialize a tensor header + data from the reader.
    pub fn deserialize<R: Read>(reader: &mut R) -> Result<(TensorHeader, Vec<u8>), SerdeError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != BINARY_MAGIC {
            return Err(SerdeError::InvalidMagic {
                expected: BINARY_MAGIC,
                actual: magic.to_vec(),
            });
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != BINARY_VERSION {
            return Err(SerdeError::InvalidHeader(format!("unsupported version: {}", version[0])));
        }

        let mut dtype_tag = [0u8; 1];
        reader.read_exact(&mut dtype_tag)?;
        let dtype = dtype_from_tag(dtype_tag[0]).ok_or_else(|| {
            SerdeError::InvalidHeader(format!("unknown dtype tag: {}", dtype_tag[0]))
        })?;

        let mut ndim_buf = [0u8; 4];
        reader.read_exact(&mut ndim_buf)?;
        let ndim = u32::from_le_bytes(ndim_buf) as usize;

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            reader.read_exact(&mut dim_buf)?;
            shape.push(u64::from_le_bytes(dim_buf) as usize);
        }

        let mut comp_tag = [0u8; 1];
        reader.read_exact(&mut comp_tag)?;
        let compression = CompressionCodec::from_tag(comp_tag[0]).ok_or_else(|| {
            SerdeError::InvalidHeader(format!("unknown compression tag: {}", comp_tag[0]))
        })?;

        let mut name_len_buf = [0u8; 4];
        reader.read_exact(&mut name_len_buf)?;
        let name_len = u32::from_le_bytes(name_len_buf) as usize;
        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)
            .map_err(|e| SerdeError::InvalidHeader(format!("invalid tensor name: {e}")))?;

        let mut data_len_buf = [0u8; 8];
        reader.read_exact(&mut data_len_buf)?;
        let data_len = u64::from_le_bytes(data_len_buf) as usize;
        let mut compressed_data = vec![0u8; data_len];
        reader.read_exact(&mut compressed_data)?;

        let data = compression.decompress(&compressed_data)?;

        let header = TensorHeader {
            name,
            shape,
            dtype,
            byte_offset: 0,
            byte_size: data.len() as u64,
            endianness: Endianness::Little,
        };

        Ok((header, data))
    }
}

// ── SafeTensors Format ──────────────────────────────────────────────────────

/// Read/write SafeTensors-compatible format.
///
/// Layout: header_size(u64 LE) | JSON header | tensor data (contiguous).
pub struct SafeTensorsFormat;

impl SafeTensorsFormat {
    /// Serialize multiple tensors into SafeTensors format.
    pub fn serialize<W: Write>(
        writer: &mut W,
        tensors: &[(&TensorHeader, &[u8])],
    ) -> Result<(), SerdeError> {
        // Build JSON header and concatenate data
        let mut header_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut data_buf = Vec::new();

        for (hdr, raw) in tensors {
            hdr.validate_data_len(raw.len())?;
            let offset_start = data_buf.len();
            data_buf.extend_from_slice(raw);
            let offset_end = data_buf.len();

            let mut entry_map = BTreeMap::new();
            entry_map.insert("dtype".into(), serde_json::Value::from(hdr.dtype.to_string()));
            entry_map.insert("shape".into(), serde_json::Value::from(hdr.shape.clone()));
            entry_map.insert(
                "data_offsets".into(),
                serde_json::Value::Array(vec![
                    serde_json::Value::from(offset_start),
                    serde_json::Value::from(offset_end),
                ]),
            );
            let entry = serde_json::Value::Object(entry_map);
            header_map.insert(hdr.name.clone(), entry);
        }

        let header_json = serde_json::to_string(&header_map)
            .map_err(|e| SerdeError::InvalidHeader(format!("JSON serialization failed: {e}")))?;
        let header_bytes = header_json.as_bytes();

        writer.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(header_bytes)?;
        writer.write_all(&data_buf)?;

        Ok(())
    }

    /// Deserialize tensors from SafeTensors format.
    pub fn deserialize<R: Read>(
        reader: &mut R,
    ) -> Result<Vec<(TensorHeader, Vec<u8>)>, SerdeError> {
        let mut size_buf = [0u8; 8];
        reader.read_exact(&mut size_buf)?;
        let header_size = u64::from_le_bytes(size_buf) as usize;

        let mut header_bytes = vec![0u8; header_size];
        reader.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| SerdeError::InvalidHeader(format!("invalid UTF-8 in header: {e}")))?;

        let header_map: BTreeMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| SerdeError::InvalidHeader(format!("JSON parse failed: {e}")))?;

        let mut all_data = Vec::new();
        reader.read_to_end(&mut all_data)?;

        let mut results = Vec::new();
        for (name, entry) in &header_map {
            let dtype_str = entry["dtype"]
                .as_str()
                .ok_or_else(|| SerdeError::InvalidHeader("missing dtype".into()))?;
            let dtype = parse_dtype(dtype_str)?;

            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .ok_or_else(|| SerdeError::InvalidHeader("missing shape".into()))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| SerdeError::InvalidHeader("invalid shape dim".into()))
                        .map(|n| n as usize)
                })
                .collect::<Result<_, _>>()?;

            let offsets = entry["data_offsets"]
                .as_array()
                .ok_or_else(|| SerdeError::InvalidHeader("missing data_offsets".into()))?;
            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            if end > all_data.len() {
                return Err(SerdeError::SizeMismatch { expected: end, actual: all_data.len() });
            }

            let data = all_data[start..end].to_vec();
            let header = TensorHeader {
                name: name.clone(),
                shape,
                dtype,
                byte_offset: start as u64,
                byte_size: (end - start) as u64,
                endianness: Endianness::Little,
            };
            results.push((header, data));
        }

        Ok(results)
    }
}

fn parse_dtype(s: &str) -> Result<DType, SerdeError> {
    match s {
        "f16" | "F16" => Ok(DType::F16),
        "f32" | "F32" => Ok(DType::F32),
        "f64" | "F64" => Ok(DType::F64),
        "i8" | "I8" => Ok(DType::I8),
        "i16" | "I16" => Ok(DType::I16),
        "i32" | "I32" => Ok(DType::I32),
        "i64" | "I64" => Ok(DType::I64),
        "u8" | "U8" => Ok(DType::U8),
        "u16" | "U16" => Ok(DType::U16),
        "u32" | "U32" => Ok(DType::U32),
        "u64" | "U64" => Ok(DType::U64),
        "bf16" | "BF16" => Ok(DType::BF16),
        other => Err(SerdeError::InvalidHeader(format!("unknown dtype: {other}"))),
    }
}

// ── NumPy Format ────────────────────────────────────────────────────────────

/// Read/write NumPy `.npy` format (v1.0).
///
/// Layout: `\x93NUMPY` | major(1) | minor(1) | header_len(u16 LE) | header | data
pub struct NumpyFormat;

const NUMPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

impl NumpyFormat {
    /// Serialize a single tensor into `.npy` format.
    pub fn serialize<W: Write>(
        writer: &mut W,
        header: &TensorHeader,
        data: &[u8],
    ) -> Result<(), SerdeError> {
        header.validate_data_len(data.len())?;

        let fortran_order = "False";
        let shape_str = if header.shape.len() == 1 {
            format!("({},)", header.shape[0])
        } else {
            let dims: Vec<String> = header.shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        };
        let descr = header.dtype.numpy_descr();
        let header_dict = format!(
            "{{'descr': '{descr}', 'fortran_order': {fortran_order}, 'shape': {shape_str}, }}"
        );

        // Pad to 64-byte alignment
        let prefix_len = 6 + 1 + 1 + 2; // magic + major + minor + header_len
        let unpadded = prefix_len + header_dict.len() + 1; // +1 for newline
        let padding = (64 - (unpadded % 64)) % 64;
        let padded_header = format!("{header_dict}{}\n", " ".repeat(padding));

        writer.write_all(NUMPY_MAGIC)?;
        writer.write_all(&[1u8, 0u8])?; // version 1.0
        writer.write_all(&(padded_header.len() as u16).to_le_bytes())?;
        writer.write_all(padded_header.as_bytes())?;
        writer.write_all(data)?;

        Ok(())
    }

    /// Deserialize a tensor from `.npy` format.
    pub fn deserialize<R: Read>(reader: &mut R) -> Result<(TensorHeader, Vec<u8>), SerdeError> {
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != NUMPY_MAGIC {
            return Err(SerdeError::InvalidMagic { expected: NUMPY_MAGIC, actual: magic.to_vec() });
        }

        let mut version = [0u8; 2];
        reader.read_exact(&mut version)?;

        let header_len = if version[0] == 1 {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        } else if version[0] == 2 {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        } else {
            return Err(SerdeError::InvalidHeader(format!(
                "unsupported npy version: {}.{}",
                version[0], version[1]
            )));
        };

        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| SerdeError::InvalidHeader(format!("invalid header UTF-8: {e}")))?
            .trim();

        let dtype = Self::parse_descr(header_str)?;
        let shape = Self::parse_shape(header_str)?;

        let num_elements: usize = shape.iter().product();
        let byte_size = num_elements * dtype.element_size();
        let mut data = vec![0u8; byte_size];
        reader.read_exact(&mut data)?;

        let hdr = TensorHeader {
            name: String::new(),
            shape,
            dtype,
            byte_offset: 0,
            byte_size: byte_size as u64,
            endianness: Endianness::Little,
        };

        Ok((hdr, data))
    }

    fn parse_descr(header: &str) -> Result<DType, SerdeError> {
        let descr_start = header
            .find("'descr'")
            .ok_or_else(|| SerdeError::InvalidHeader("missing descr".into()))?;
        let after = &header[descr_start..];
        let q1 = after
            .find(": '")
            .ok_or_else(|| SerdeError::InvalidHeader("bad descr format".into()))?;
        let value_start = descr_start + q1 + 3;
        let value_end = header[value_start..]
            .find('\'')
            .ok_or_else(|| SerdeError::InvalidHeader("unterminated descr".into()))?;
        let descr_str = &header[value_start..value_start + value_end];
        DType::from_numpy_descr(descr_str)
            .ok_or_else(|| SerdeError::InvalidHeader(format!("unknown numpy descr: {descr_str}")))
    }

    fn parse_shape(header: &str) -> Result<Vec<usize>, SerdeError> {
        let shape_start = header
            .find("'shape'")
            .ok_or_else(|| SerdeError::InvalidHeader("missing shape".into()))?;
        let after = &header[shape_start..];
        let paren_start =
            after.find('(').ok_or_else(|| SerdeError::InvalidHeader("bad shape format".into()))?;
        let paren_end = after
            .find(')')
            .ok_or_else(|| SerdeError::InvalidHeader("unterminated shape".into()))?;
        let inner = &after[paren_start + 1..paren_end];
        if inner.trim().is_empty() {
            return Ok(vec![]);
        }
        inner
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|e| SerdeError::InvalidHeader(format!("bad shape dim: {e}")))
            })
            .collect()
    }
}

// ── Tensor Archive ──────────────────────────────────────────────────────────

/// An index entry for random access within an archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    pub header: TensorHeader,
    pub checksum: Option<TensorChecksum>,
    pub compression: CompressionCodec,
}

/// Collection of named tensors with an index for random access.
#[derive(Debug, Default)]
pub struct TensorArchive {
    entries: BTreeMap<String, (ArchiveEntry, Vec<u8>)>,
}

impl TensorArchive {
    /// Create an empty archive.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor to the archive.
    pub fn insert(
        &mut self,
        header: TensorHeader,
        data: Vec<u8>,
        compression: CompressionCodec,
        compute_checksum: bool,
    ) -> Result<(), SerdeError> {
        header.validate_data_len(data.len())?;
        let checksum = if compute_checksum { Some(TensorChecksum::compute(&data)) } else { None };
        let entry = ArchiveEntry { header: header.clone(), checksum, compression };
        self.entries.insert(header.name.clone(), (entry, data));
        Ok(())
    }

    /// Retrieve a tensor by name.
    pub fn get(&self, name: &str) -> Result<(&TensorHeader, &[u8]), SerdeError> {
        let (entry, data) =
            self.entries.get(name).ok_or_else(|| SerdeError::TensorNotFound(name.into()))?;
        Ok((&entry.header, data))
    }

    /// List all tensor names in the archive.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// Number of tensors in the archive.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a tensor by name.
    pub fn remove(&mut self, name: &str) -> Option<(ArchiveEntry, Vec<u8>)> {
        self.entries.remove(name)
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ArchiveEntry, &[u8])> {
        self.entries.iter().map(|(name, (entry, data))| (name.as_str(), entry, data.as_slice()))
    }

    /// Serialize the entire archive in binary format.
    pub fn serialize_binary<W: Write>(&self, writer: &mut W) -> Result<(), SerdeError> {
        // Archive header: magic + count
        writer.write_all(b"BTSA")?; // BitNet Tensor Serialized Archive
        writer.write_all(&(self.entries.len() as u32).to_le_bytes())?;

        for (entry, data) in self.entries.values() {
            BinarySerializer::serialize(writer, &entry.header, data, entry.compression)?;
            // Write checksum flag + optional checksum
            if let Some(ref cksum) = entry.checksum {
                writer.write_all(&[1u8])?;
                writer.write_all(cksum.digest())?;
            } else {
                writer.write_all(&[0u8])?;
            }
        }
        Ok(())
    }

    /// Deserialize an archive from binary format.
    pub fn deserialize_binary<R: Read>(reader: &mut R) -> Result<Self, SerdeError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"BTSA" {
            return Err(SerdeError::InvalidMagic { expected: b"BTSA", actual: magic.to_vec() });
        }

        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        let mut archive = Self::new();
        for _ in 0..count {
            let (header, data) = BinaryDeserializer::deserialize(reader)?;

            let mut cksum_flag = [0u8; 1];
            reader.read_exact(&mut cksum_flag)?;
            let checksum = if cksum_flag[0] == 1 {
                let mut digest = [0u8; 32];
                reader.read_exact(&mut digest)?;
                Some(TensorChecksum { digest })
            } else {
                None
            };

            let entry = ArchiveEntry {
                header: header.clone(),
                checksum,
                compression: CompressionCodec::None,
            };
            archive.entries.insert(header.name.clone(), (entry, data));
        }

        Ok(archive)
    }
}

// ── Tensor Serializer (Orchestrator) ────────────────────────────────────────

/// Orchestrates tensor serialization: format selection → header → compress →
/// write → checksum.
pub struct TensorSerializer {
    format: SerializationFormat,
    compression: CompressionCodec,
    compute_checksum: bool,
}

impl TensorSerializer {
    /// Create a new serializer with the given configuration.
    #[must_use]
    pub fn new(
        format: SerializationFormat,
        compression: CompressionCodec,
        compute_checksum: bool,
    ) -> Self {
        Self { format, compression, compute_checksum }
    }

    /// Serialize a single tensor, returning raw bytes and optional checksum.
    pub fn serialize(
        &self,
        header: &TensorHeader,
        data: &[u8],
    ) -> Result<(Vec<u8>, Option<TensorChecksum>), SerdeError> {
        header.validate_data_len(data.len())?;

        let checksum =
            if self.compute_checksum { Some(TensorChecksum::compute(data)) } else { None };

        let compressed = self.compression.compress(data)?;

        let mut output = Vec::new();
        match self.format {
            SerializationFormat::Binary => {
                BinarySerializer::serialize(&mut output, header, data, self.compression)?;
            }
            SerializationFormat::NumPy => {
                // NumPy doesn't support compression natively; write uncompressed
                NumpyFormat::serialize(&mut output, header, data)?;
            }
            SerializationFormat::SafeTensors => {
                SafeTensorsFormat::serialize(&mut output, &[(header, data)])?;
            }
            SerializationFormat::Json => {
                let mut json_map = BTreeMap::new();
                json_map.insert("name".into(), serde_json::Value::from(header.name.clone()));
                json_map.insert("dtype".into(), serde_json::Value::from(header.dtype.to_string()));
                json_map.insert("shape".into(), serde_json::Value::from(header.shape.clone()));
                json_map.insert(
                    "data_base64".into(),
                    serde_json::Value::from(base64_encode(&compressed)),
                );
                json_map.insert(
                    "compression".into(),
                    serde_json::Value::from(format!("{:?}", self.compression)),
                );
                let json = serde_json::Value::Object(json_map);
                let json_str = serde_json::to_string_pretty(&json).map_err(|e| {
                    SerdeError::InvalidHeader(format!("JSON serialization failed: {e}"))
                })?;
                output.extend_from_slice(json_str.as_bytes());
            }
            SerializationFormat::Gguf => {
                return Err(SerdeError::UnsupportedFormat(SerializationFormat::Gguf));
            }
        }

        Ok((output, checksum))
    }

    /// Deserialize a tensor from bytes in the configured format.
    pub fn deserialize(&self, bytes: &[u8]) -> Result<(TensorHeader, Vec<u8>), SerdeError> {
        let mut cursor = Cursor::new(bytes);
        match self.format {
            SerializationFormat::Binary => BinaryDeserializer::deserialize(&mut cursor),
            SerializationFormat::NumPy => NumpyFormat::deserialize(&mut cursor),
            SerializationFormat::SafeTensors => {
                let tensors = SafeTensorsFormat::deserialize(&mut cursor)?;
                tensors
                    .into_iter()
                    .next()
                    .ok_or_else(|| SerdeError::InvalidHeader("empty SafeTensors file".into()))
            }
            SerializationFormat::Json => {
                let json: serde_json::Value = serde_json::from_slice(bytes)
                    .map_err(|e| SerdeError::InvalidHeader(format!("JSON parse failed: {e}")))?;
                let name = json["name"].as_str().unwrap_or("").to_string();
                let dtype_str = json["dtype"]
                    .as_str()
                    .ok_or_else(|| SerdeError::InvalidHeader("missing dtype".into()))?;
                let dtype = parse_dtype(dtype_str)?;
                let shape: Vec<usize> = json["shape"]
                    .as_array()
                    .ok_or_else(|| SerdeError::InvalidHeader("missing shape".into()))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let data_b64 = json["data_base64"]
                    .as_str()
                    .ok_or_else(|| SerdeError::InvalidHeader("missing data_base64".into()))?;
                let compressed = base64_decode(data_b64)?;

                let comp_str = json["compression"].as_str().unwrap_or("None");
                let codec = match comp_str {
                    "None" => CompressionCodec::None,
                    "Lz4" => CompressionCodec::Lz4,
                    "Zstd" => CompressionCodec::Zstd,
                    "Snappy" => CompressionCodec::Snappy,
                    other => {
                        return Err(SerdeError::InvalidHeader(format!(
                            "unknown compression: {other}"
                        )));
                    }
                };
                let data = codec.decompress(&compressed)?;

                let header = TensorHeader::new(name, shape, dtype);
                Ok((header, data))
            }
            SerializationFormat::Gguf => {
                Err(SerdeError::UnsupportedFormat(SerializationFormat::Gguf))
            }
        }
    }
}

// ── Base64 Helpers (no external dep) ────────────────────────────────────────

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(input: &str) -> Result<Vec<u8>, SerdeError> {
    fn val(c: u8) -> Result<u32, SerdeError> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(SerdeError::InvalidHeader(format!("invalid base64 char: {c}"))),
        }
    }

    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);

    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 {
            break;
        }
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        result.push(((a << 2) | (b >> 4)) as u8);

        if chunk.len() > 2 && chunk[2] != b'=' {
            let c = val(chunk[2])?;
            result.push((((b & 0xF) << 4) | (c >> 2)) as u8);

            if chunk.len() > 3 && chunk[3] != b'=' {
                let d = val(chunk[3])?;
                result.push((((c & 0x3) << 6) | d) as u8);
            }
        }
    }

    Ok(result)
}

// We need serde_json for SafeTensors and JSON formats.
// This is a lightweight inline approach using the workspace serde_json.
mod serde_json {
    //! Minimal JSON serialization/deserialization using std only.
    //! Wraps a recursive `Value` type with parse/emit.

    use std::collections::BTreeMap;
    use std::fmt;

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        Number(f64),
        String(String),
        Array(Vec<Value>),
        Object(BTreeMap<String, Value>),
    }

    impl Value {
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Self::String(s) => Some(s),
                _ => None,
            }
        }

        pub fn as_u64(&self) -> Option<u64> {
            match self {
                Self::Number(n) => {
                    if *n >= 0.0 && *n <= u64::MAX as f64 {
                        Some(*n as u64)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        pub fn as_array(&self) -> Option<&Vec<Value>> {
            match self {
                Self::Array(a) => Some(a),
                _ => None,
            }
        }
    }

    impl std::ops::Index<&str> for Value {
        type Output = Value;
        fn index(&self, key: &str) -> &Value {
            match self {
                Self::Object(map) => map.get(key).unwrap_or(&Value::Null),
                _ => &Value::Null,
            }
        }
    }

    // ── Value conversions ──────────────────────────────────────────────

    impl From<&str> for Value {
        fn from(s: &str) -> Self {
            Self::String(s.to_string())
        }
    }

    impl From<String> for Value {
        fn from(s: String) -> Self {
            Self::String(s)
        }
    }

    impl From<usize> for Value {
        fn from(n: usize) -> Self {
            Self::Number(n as f64)
        }
    }

    impl From<u64> for Value {
        fn from(n: u64) -> Self {
            Self::Number(n as f64)
        }
    }

    impl From<Vec<Value>> for Value {
        fn from(v: Vec<Value>) -> Self {
            Self::Array(v)
        }
    }

    impl From<Vec<usize>> for Value {
        fn from(v: Vec<usize>) -> Self {
            Self::Array(v.into_iter().map(Value::from).collect())
        }
    }

    impl From<BTreeMap<String, Value>> for Value {
        fn from(m: BTreeMap<String, Value>) -> Self {
            Self::Object(m)
        }
    }

    // ── Serialization ───────────────────────────────────────────────────

    pub fn to_string(value: &BTreeMap<String, Value>) -> Result<String, String> {
        let v = Value::Object(value.clone());
        Ok(emit(&v))
    }

    pub fn to_string_pretty(value: &Value) -> Result<String, String> {
        Ok(emit_pretty(value, 0))
    }

    pub fn from_str(s: &str) -> Result<BTreeMap<String, Value>, String> {
        let value = parse(s)?;
        match value {
            Value::Object(m) => Ok(m),
            _ => Err("expected JSON object at top level".into()),
        }
    }

    pub fn from_slice(s: &[u8]) -> Result<Value, String> {
        let text = std::str::from_utf8(s).map_err(|e| e.to_string())?;
        parse(text)
    }

    fn emit(value: &Value) -> String {
        match value {
            Value::Null => "null".into(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => {
                if *n == (*n as u64) as f64 && *n >= 0.0 {
                    format!("{}", *n as u64)
                } else {
                    format!("{n}")
                }
            }
            Value::String(s) => format!("\"{}\"", escape_json(s)),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(emit).collect();
                format!("[{}]", items.join(","))
            }
            Value::Object(map) => {
                let pairs: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_json(k), emit(v)))
                    .collect();
                format!("{{{}}}", pairs.join(","))
            }
        }
    }

    fn emit_pretty(value: &Value, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        let pad_inner = "  ".repeat(indent + 1);
        match value {
            Value::Null => "null".into(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => {
                if *n == (*n as u64) as f64 && *n >= 0.0 {
                    format!("{}", *n as u64)
                } else {
                    format!("{n}")
                }
            }
            Value::String(s) => format!("\"{}\"", escape_json(s)),
            Value::Array(arr) => {
                if arr.is_empty() {
                    return "[]".into();
                }
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| format!("{pad_inner}{}", emit_pretty(v, indent + 1)))
                    .collect();
                format!("[\n{}\n{pad}]", items.join(",\n"))
            }
            Value::Object(map) => {
                if map.is_empty() {
                    return "{}".into();
                }
                let pairs: Vec<String> = map
                    .iter()
                    .map(|(k, v)| {
                        format!("{pad_inner}\"{}\": {}", escape_json(k), emit_pretty(v, indent + 1))
                    })
                    .collect();
                format!("{{\n{}\n{pad}}}", pairs.join(",\n"))
            }
        }
    }

    fn escape_json(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c => out.push(c),
            }
        }
        out
    }

    // ── Parsing ─────────────────────────────────────────────────────────

    struct Parser<'a> {
        input: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn new(input: &'a str) -> Self {
            Self { input: input.as_bytes(), pos: 0 }
        }

        fn skip_ws(&mut self) {
            while self.pos < self.input.len()
                && matches!(self.input[self.pos], b' ' | b'\t' | b'\n' | b'\r')
            {
                self.pos += 1;
            }
        }

        fn peek(&mut self) -> Option<u8> {
            self.skip_ws();
            self.input.get(self.pos).copied()
        }

        fn advance(&mut self) -> u8 {
            let b = self.input[self.pos];
            self.pos += 1;
            b
        }

        fn expect(&mut self, expected: u8) -> Result<(), String> {
            self.skip_ws();
            if self.pos >= self.input.len() {
                return Err(format!("unexpected EOF, expected '{}'", expected as char));
            }
            if self.input[self.pos] != expected {
                return Err(format!(
                    "expected '{}', got '{}'",
                    expected as char, self.input[self.pos] as char
                ));
            }
            self.pos += 1;
            Ok(())
        }

        fn parse_value(&mut self) -> Result<Value, String> {
            self.skip_ws();
            match self.peek() {
                Some(b'"') => self.parse_string().map(Value::String),
                Some(b'{') => self.parse_object().map(Value::Object),
                Some(b'[') => self.parse_array().map(Value::Array),
                Some(b't') => self.parse_literal("true", Value::Bool(true)),
                Some(b'f') => self.parse_literal("false", Value::Bool(false)),
                Some(b'n') => self.parse_literal("null", Value::Null),
                Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
                Some(c) => Err(format!("unexpected char: '{}'", c as char)),
                None => Err("unexpected EOF".into()),
            }
        }

        fn parse_string(&mut self) -> Result<String, String> {
            self.expect(b'"')?;
            let mut s = String::new();
            loop {
                if self.pos >= self.input.len() {
                    return Err("unterminated string".into());
                }
                let b = self.advance();
                match b {
                    b'"' => return Ok(s),
                    b'\\' => {
                        if self.pos >= self.input.len() {
                            return Err("unterminated escape".into());
                        }
                        let esc = self.advance();
                        match esc {
                            b'"' => s.push('"'),
                            b'\\' => s.push('\\'),
                            b'/' => s.push('/'),
                            b'n' => s.push('\n'),
                            b'r' => s.push('\r'),
                            b't' => s.push('\t'),
                            b'u' => {
                                // Simple 4-hex-digit unicode escape
                                if self.pos + 4 > self.input.len() {
                                    return Err("short unicode escape".into());
                                }
                                let hex = std::str::from_utf8(&self.input[self.pos..self.pos + 4])
                                    .map_err(|_| "invalid unicode escape")?;
                                let cp = u32::from_str_radix(hex, 16)
                                    .map_err(|_| "invalid hex in unicode")?;
                                if let Some(c) = char::from_u32(cp) {
                                    s.push(c);
                                }
                                self.pos += 4;
                            }
                            _ => {
                                s.push('\\');
                                s.push(esc as char);
                            }
                        }
                    }
                    _ => s.push(b as char),
                }
            }
        }

        fn parse_number(&mut self) -> Result<Value, String> {
            let start = self.pos;
            if self.pos < self.input.len() && self.input[self.pos] == b'-' {
                self.pos += 1;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos < self.input.len() && self.input[self.pos] == b'.' {
                self.pos += 1;
                while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
            if self.pos < self.input.len()
                && (self.input[self.pos] == b'e' || self.input[self.pos] == b'E')
            {
                self.pos += 1;
                if self.pos < self.input.len()
                    && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-')
                {
                    self.pos += 1;
                }
                while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
            let num_str =
                std::str::from_utf8(&self.input[start..self.pos]).map_err(|e| e.to_string())?;
            let n: f64 = num_str.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
            Ok(Value::Number(n))
        }

        fn parse_object(&mut self) -> Result<BTreeMap<String, Value>, String> {
            self.expect(b'{')?;
            let mut map = BTreeMap::new();
            if self.peek() == Some(b'}') {
                self.advance();
                return Ok(map);
            }
            loop {
                self.skip_ws();
                let key = self.parse_string()?;
                self.expect(b':')?;
                let value = self.parse_value()?;
                map.insert(key, value);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.advance();
                    }
                    Some(b'}') => {
                        self.advance();
                        return Ok(map);
                    }
                    _ => return Err("expected ',' or '}' in object".into()),
                }
            }
        }

        fn parse_array(&mut self) -> Result<Vec<Value>, String> {
            self.expect(b'[')?;
            let mut arr = Vec::new();
            if self.peek() == Some(b']') {
                self.advance();
                return Ok(arr);
            }
            loop {
                let value = self.parse_value()?;
                arr.push(value);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.advance();
                    }
                    Some(b']') => {
                        self.advance();
                        return Ok(arr);
                    }
                    _ => return Err("expected ',' or ']' in array".into()),
                }
            }
        }

        fn parse_literal(&mut self, literal: &str, value: Value) -> Result<Value, String> {
            for expected in literal.bytes() {
                if self.pos >= self.input.len() || self.input[self.pos] != expected {
                    return Err(format!("expected literal: {literal}"));
                }
                self.pos += 1;
            }
            Ok(value)
        }
    }

    pub fn parse(input: &str) -> Result<Value, String> {
        let mut parser = Parser::new(input);
        let value = parser.parse_value()?;
        parser.skip_ws();
        Ok(value)
    }

    impl fmt::Display for Value {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", emit(self))
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DType tests ─────────────────────────────────────────────────────

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(DType::U8.element_size(), 1);
        assert_eq!(DType::I8.element_size(), 1);
        assert_eq!(DType::F16.element_size(), 2);
        assert_eq!(DType::BF16.element_size(), 2);
        assert_eq!(DType::I16.element_size(), 2);
        assert_eq!(DType::U16.element_size(), 2);
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::I32.element_size(), 4);
        assert_eq!(DType::U32.element_size(), 4);
        assert_eq!(DType::F64.element_size(), 8);
        assert_eq!(DType::I64.element_size(), 8);
        assert_eq!(DType::U64.element_size(), 8);
    }

    #[test]
    fn dtype_display() {
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::U64.to_string(), "u64");
    }

    #[test]
    fn dtype_numpy_descr_roundtrip() {
        let dtypes = [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::BF16,
        ];
        for dt in dtypes {
            let descr = dt.numpy_descr();
            let parsed = DType::from_numpy_descr(descr);
            assert_eq!(parsed, Some(dt), "roundtrip failed for {dt}");
        }
    }

    #[test]
    fn dtype_from_numpy_descr_unknown() {
        assert_eq!(DType::from_numpy_descr("<f99"), None);
        assert_eq!(DType::from_numpy_descr(""), None);
    }

    // ── TensorHeader tests ──────────────────────────────────────────────

    #[test]
    fn header_new_computes_byte_size() {
        let h = TensorHeader::new("test", vec![2, 3, 4], DType::F32);
        assert_eq!(h.byte_size, 2 * 3 * 4 * 4);
        assert_eq!(h.num_elements(), 24);
    }

    #[test]
    fn header_scalar_tensor() {
        let h = TensorHeader::new("scalar", vec![], DType::F64);
        assert_eq!(h.num_elements(), 1);
        assert_eq!(h.byte_size, 8);
    }

    #[test]
    fn header_validate_data_len_ok() {
        let h = TensorHeader::new("x", vec![4], DType::U8);
        assert!(h.validate_data_len(4).is_ok());
    }

    #[test]
    fn header_validate_data_len_mismatch() {
        let h = TensorHeader::new("x", vec![4], DType::U8);
        let err = h.validate_data_len(5).unwrap_err();
        assert!(matches!(err, SerdeError::SizeMismatch { .. }));
    }

    #[test]
    fn header_default_endianness() {
        let h = TensorHeader::new("x", vec![1], DType::F32);
        assert_eq!(h.endianness, Endianness::Little);
    }

    // ── SerializationFormat tests ───────────────────────────────────────

    #[test]
    fn format_display() {
        assert_eq!(SerializationFormat::Binary.to_string(), "binary");
        assert_eq!(SerializationFormat::SafeTensors.to_string(), "safetensors");
        assert_eq!(SerializationFormat::Gguf.to_string(), "gguf");
        assert_eq!(SerializationFormat::NumPy.to_string(), "numpy");
        assert_eq!(SerializationFormat::Json.to_string(), "json");
    }

    #[test]
    fn format_equality() {
        assert_eq!(SerializationFormat::Binary, SerializationFormat::Binary);
        assert_ne!(SerializationFormat::Binary, SerializationFormat::Json);
    }

    // ── CompressionCodec tests ──────────────────────────────────────────

    #[test]
    fn compression_none_roundtrip() {
        let data = b"hello world";
        let compressed = CompressionCodec::None.compress(data).unwrap();
        assert_eq!(compressed, data);
        let decompressed = CompressionCodec::None.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compression_lz4_unsupported() {
        let err = CompressionCodec::Lz4.compress(b"data").unwrap_err();
        assert!(matches!(err, SerdeError::UnsupportedCompression(_)));
    }

    #[test]
    fn compression_zstd_unsupported() {
        let err = CompressionCodec::Zstd.decompress(b"data").unwrap_err();
        assert!(matches!(err, SerdeError::UnsupportedCompression(_)));
    }

    #[test]
    fn compression_snappy_unsupported() {
        let err = CompressionCodec::Snappy.compress(b"data").unwrap_err();
        assert!(matches!(err, SerdeError::UnsupportedCompression(_)));
    }

    #[test]
    fn compression_tag_roundtrip() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::Zstd,
            CompressionCodec::Snappy,
        ];
        for c in codecs {
            assert_eq!(CompressionCodec::from_tag(c.tag()), Some(c));
        }
    }

    #[test]
    fn compression_tag_unknown() {
        assert_eq!(CompressionCodec::from_tag(255), None);
    }

    #[test]
    fn compression_default_is_none() {
        assert_eq!(CompressionCodec::default(), CompressionCodec::None);
    }

    // ── Checksum tests ──────────────────────────────────────────────────

    #[test]
    fn checksum_empty_data() {
        let cksum = TensorChecksum::compute(b"");
        // SHA-256 of empty string is well-known
        assert_eq!(
            cksum.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn checksum_known_value() {
        let cksum = TensorChecksum::compute(b"hello");
        assert_eq!(
            cksum.to_hex(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn checksum_hex_roundtrip() {
        let cksum = TensorChecksum::compute(b"test data");
        let hex = cksum.to_hex();
        let parsed = TensorChecksum::from_hex(&hex).unwrap();
        assert_eq!(cksum, parsed);
    }

    #[test]
    fn checksum_from_hex_invalid_length() {
        assert!(TensorChecksum::from_hex("abc").is_none());
    }

    #[test]
    fn checksum_from_hex_invalid_chars() {
        let bad = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
        assert!(TensorChecksum::from_hex(bad).is_none());
    }

    #[test]
    fn checksum_verify_ok() {
        let data = b"verify me";
        let cksum = TensorChecksum::compute(data);
        assert!(cksum.verify(data).is_ok());
    }

    #[test]
    fn checksum_verify_mismatch() {
        let cksum = TensorChecksum::compute(b"original");
        let err = cksum.verify(b"modified").unwrap_err();
        assert!(matches!(err, SerdeError::ChecksumMismatch { .. }));
    }

    #[test]
    fn checksum_digest_bytes() {
        let cksum = TensorChecksum::compute(b"bytes");
        assert_eq!(cksum.digest().len(), 32);
    }

    // ── Binary format tests ─────────────────────────────────────────────

    #[test]
    fn binary_roundtrip_f32() {
        let header = TensorHeader::new("weights", vec![2, 3], DType::F32);
        let data: Vec<u8> = (0..24).collect();
        let mut buf = Vec::new();
        BinarySerializer::serialize(&mut buf, &header, &data, CompressionCodec::None).unwrap();
        let (h2, d2) = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.name, "weights");
        assert_eq!(h2.shape, vec![2, 3]);
        assert_eq!(h2.dtype, DType::F32);
        assert_eq!(d2, data);
    }

    #[test]
    fn binary_roundtrip_u8() {
        let header = TensorHeader::new("mask", vec![8], DType::U8);
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut buf = Vec::new();
        BinarySerializer::serialize(&mut buf, &header, &data, CompressionCodec::None).unwrap();
        let (h2, d2) = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.name, "mask");
        assert_eq!(d2, data);
    }

    #[test]
    fn binary_roundtrip_empty_name() {
        let header = TensorHeader::new("", vec![2], DType::U8);
        let data = vec![10, 20];
        let mut buf = Vec::new();
        BinarySerializer::serialize(&mut buf, &header, &data, CompressionCodec::None).unwrap();
        let (h2, d2) = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.name, "");
        assert_eq!(d2, data);
    }

    #[test]
    fn binary_roundtrip_scalar() {
        let header = TensorHeader::new("lr", vec![], DType::F64);
        let data = 0.001_f64.to_le_bytes().to_vec();
        let mut buf = Vec::new();
        BinarySerializer::serialize(&mut buf, &header, &data, CompressionCodec::None).unwrap();
        let (h2, d2) = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.shape, Vec::<usize>::new());
        assert_eq!(d2, data);
    }

    #[test]
    fn binary_roundtrip_large_shape() {
        let header = TensorHeader::new("big", vec![2, 3, 4, 5], DType::U8);
        let data = vec![42u8; 120];
        let mut buf = Vec::new();
        BinarySerializer::serialize(&mut buf, &header, &data, CompressionCodec::None).unwrap();
        let (h2, d2) = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.shape, vec![2, 3, 4, 5]);
        assert_eq!(d2, data);
    }

    #[test]
    fn binary_invalid_magic() {
        let bad = b"XXXX\x01\x01\x00\x00\x00\x00";
        let err = BinaryDeserializer::deserialize(&mut Cursor::new(bad.as_ref())).unwrap_err();
        assert!(matches!(err, SerdeError::InvalidMagic { .. }));
    }

    #[test]
    fn binary_invalid_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BTSR");
        buf.push(99); // bad version
        let err = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, SerdeError::InvalidHeader(_)));
    }

    #[test]
    fn binary_data_size_mismatch() {
        let header = TensorHeader::new("x", vec![4], DType::F32);
        let wrong_data = vec![0u8; 8]; // should be 16
        let err = BinarySerializer::serialize(
            &mut Vec::new(),
            &header,
            &wrong_data,
            CompressionCodec::None,
        )
        .unwrap_err();
        assert!(matches!(err, SerdeError::SizeMismatch { .. }));
    }

    #[test]
    fn binary_all_dtypes() {
        let dtypes = [
            (DType::F16, 2),
            (DType::F32, 4),
            (DType::F64, 8),
            (DType::I8, 1),
            (DType::I16, 2),
            (DType::I32, 4),
            (DType::I64, 8),
            (DType::U8, 1),
            (DType::U16, 2),
            (DType::U32, 4),
            (DType::U64, 8),
            (DType::BF16, 2),
        ];
        for (dt, size) in dtypes {
            let header = TensorHeader::new("t", vec![4], dt);
            let data = vec![0u8; 4 * size];
            let mut buf = Vec::new();
            BinarySerializer::serialize(&mut buf, &header, &data, CompressionCodec::None).unwrap();
            let (h2, d2) = BinaryDeserializer::deserialize(&mut Cursor::new(&buf)).unwrap();
            assert_eq!(h2.dtype, dt);
            assert_eq!(d2.len(), data.len());
        }
    }

    // ── SafeTensors format tests ────────────────────────────────────────

    #[test]
    fn safetensors_single_tensor_roundtrip() {
        let header = TensorHeader::new("layer.weight", vec![4, 3], DType::F32);
        let data = vec![0u8; 48];
        let mut buf = Vec::new();
        SafeTensorsFormat::serialize(&mut buf, &[(&header, &data)]).unwrap();
        let tensors = SafeTensorsFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].0.name, "layer.weight");
        assert_eq!(tensors[0].0.shape, vec![4, 3]);
        assert_eq!(tensors[0].1, data);
    }

    #[test]
    fn safetensors_multiple_tensors() {
        let h1 = TensorHeader::new("a", vec![2], DType::F32);
        let d1 = vec![0u8; 8];
        let h2 = TensorHeader::new("b", vec![3], DType::U8);
        let d2 = vec![1u8, 2, 3];

        let mut buf = Vec::new();
        SafeTensorsFormat::serialize(&mut buf, &[(&h1, d1.as_slice()), (&h2, d2.as_slice())])
            .unwrap();

        let tensors = SafeTensorsFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(tensors.len(), 2);
        let names: Vec<&str> = tensors.iter().map(|(h, _)| h.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn safetensors_empty_tensor_list() {
        let mut buf = Vec::new();
        SafeTensorsFormat::serialize(&mut buf, &[]).unwrap();
        let tensors = SafeTensorsFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert!(tensors.is_empty());
    }

    #[test]
    fn safetensors_data_mismatch() {
        let header = TensorHeader::new("x", vec![4], DType::F32);
        let wrong_data = vec![0u8; 4]; // should be 16
        let err =
            SafeTensorsFormat::serialize(&mut Vec::new(), &[(&header, wrong_data.as_slice())])
                .unwrap_err();
        assert!(matches!(err, SerdeError::SizeMismatch { .. }));
    }

    // ── NumPy format tests ──────────────────────────────────────────────

    #[test]
    fn numpy_roundtrip_f32() {
        let header = TensorHeader::new("", vec![3, 2], DType::F32);
        let data = vec![0u8; 24];
        let mut buf = Vec::new();
        NumpyFormat::serialize(&mut buf, &header, &data).unwrap();
        let (h2, d2) = NumpyFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.shape, vec![3, 2]);
        assert_eq!(h2.dtype, DType::F32);
        assert_eq!(d2, data);
    }

    #[test]
    fn numpy_roundtrip_u8() {
        let header = TensorHeader::new("", vec![5], DType::U8);
        let data = vec![10, 20, 30, 40, 50];
        let mut buf = Vec::new();
        NumpyFormat::serialize(&mut buf, &header, &data).unwrap();
        let (h2, d2) = NumpyFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.shape, vec![5]);
        assert_eq!(h2.dtype, DType::U8);
        assert_eq!(d2, data);
    }

    #[test]
    fn numpy_roundtrip_1d() {
        let header = TensorHeader::new("", vec![10], DType::I32);
        let data = vec![0u8; 40];
        let mut buf = Vec::new();
        NumpyFormat::serialize(&mut buf, &header, &data).unwrap();
        let (h2, d2) = NumpyFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.shape, vec![10]);
        assert_eq!(d2, data);
    }

    #[test]
    fn numpy_magic_bytes() {
        let header = TensorHeader::new("", vec![1], DType::U8);
        let data = vec![42u8];
        let mut buf = Vec::new();
        NumpyFormat::serialize(&mut buf, &header, &data).unwrap();
        assert_eq!(&buf[..6], b"\x93NUMPY");
    }

    #[test]
    fn numpy_invalid_magic() {
        let bad = b"BADMAGIC";
        let err = NumpyFormat::deserialize(&mut Cursor::new(bad.as_ref())).unwrap_err();
        assert!(matches!(err, SerdeError::InvalidMagic { .. }));
    }

    #[test]
    fn numpy_data_mismatch() {
        let header = TensorHeader::new("", vec![4], DType::F32);
        let wrong_data = vec![0u8; 8];
        let err = NumpyFormat::serialize(&mut Vec::new(), &header, &wrong_data).unwrap_err();
        assert!(matches!(err, SerdeError::SizeMismatch { .. }));
    }

    #[test]
    fn numpy_roundtrip_f64() {
        let header = TensorHeader::new("", vec![2], DType::F64);
        let data = vec![0u8; 16];
        let mut buf = Vec::new();
        NumpyFormat::serialize(&mut buf, &header, &data).unwrap();
        let (h2, d2) = NumpyFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h2.dtype, DType::F64);
        assert_eq!(d2, data);
    }

    // ── TensorArchive tests ─────────────────────────────────────────────

    #[test]
    fn archive_insert_and_get() {
        let mut archive = TensorArchive::new();
        let h = TensorHeader::new("w", vec![4], DType::U8);
        let data = vec![1, 2, 3, 4];
        archive.insert(h, data.clone(), CompressionCodec::None, false).unwrap();
        let (hdr, d) = archive.get("w").unwrap();
        assert_eq!(hdr.name, "w");
        assert_eq!(d, &data);
    }

    #[test]
    fn archive_not_found() {
        let archive = TensorArchive::new();
        let err = archive.get("missing").unwrap_err();
        assert!(matches!(err, SerdeError::TensorNotFound(_)));
    }

    #[test]
    fn archive_names() {
        let mut archive = TensorArchive::new();
        let h1 = TensorHeader::new("a", vec![1], DType::U8);
        let h2 = TensorHeader::new("b", vec![1], DType::U8);
        archive.insert(h1, vec![0], CompressionCodec::None, false).unwrap();
        archive.insert(h2, vec![0], CompressionCodec::None, false).unwrap();
        let names = archive.names();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn archive_is_empty() {
        let archive = TensorArchive::new();
        assert!(archive.is_empty());
    }

    #[test]
    fn archive_remove() {
        let mut archive = TensorArchive::new();
        let h = TensorHeader::new("x", vec![1], DType::U8);
        archive.insert(h, vec![42], CompressionCodec::None, false).unwrap();
        assert!(archive.remove("x").is_some());
        assert!(archive.is_empty());
    }

    #[test]
    fn archive_remove_nonexistent() {
        let mut archive = TensorArchive::new();
        assert!(archive.remove("nope").is_none());
    }

    #[test]
    fn archive_with_checksum() {
        let mut archive = TensorArchive::new();
        let h = TensorHeader::new("ck", vec![2], DType::U8);
        archive.insert(h, vec![1, 2], CompressionCodec::None, true).unwrap();
        for (_, entry, _) in archive.iter() {
            assert!(entry.checksum.is_some());
        }
    }

    #[test]
    fn archive_binary_roundtrip() {
        let mut archive = TensorArchive::new();
        let h1 = TensorHeader::new("a", vec![2], DType::F32);
        let d1 = vec![0u8; 8];
        let h2 = TensorHeader::new("b", vec![3], DType::U8);
        let d2 = vec![1, 2, 3];
        archive.insert(h1, d1.clone(), CompressionCodec::None, true).unwrap();
        archive.insert(h2, d2.clone(), CompressionCodec::None, false).unwrap();

        let mut buf = Vec::new();
        archive.serialize_binary(&mut buf).unwrap();

        let archive2 = TensorArchive::deserialize_binary(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(archive2.len(), 2);
        let (ha, da) = archive2.get("a").unwrap();
        assert_eq!(ha.dtype, DType::F32);
        assert_eq!(da, &d1);
        let (hb, db) = archive2.get("b").unwrap();
        assert_eq!(hb.shape, vec![3]);
        assert_eq!(db, &d2);
    }

    #[test]
    fn archive_binary_empty() {
        let archive = TensorArchive::new();
        let mut buf = Vec::new();
        archive.serialize_binary(&mut buf).unwrap();
        let archive2 = TensorArchive::deserialize_binary(&mut Cursor::new(&buf)).unwrap();
        assert!(archive2.is_empty());
    }

    #[test]
    fn archive_binary_invalid_magic() {
        let bad = b"XXXX\x00\x00\x00\x00";
        let err = TensorArchive::deserialize_binary(&mut Cursor::new(bad.as_ref())).unwrap_err();
        assert!(matches!(err, SerdeError::InvalidMagic { .. }));
    }

    #[test]
    fn archive_iter() {
        let mut archive = TensorArchive::new();
        let h = TensorHeader::new("t", vec![1], DType::U8);
        archive.insert(h, vec![0], CompressionCodec::None, false).unwrap();
        let items: Vec<_> = archive.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, "t");
    }

    #[test]
    fn archive_overwrite_tensor() {
        let mut archive = TensorArchive::new();
        let h1 = TensorHeader::new("x", vec![1], DType::U8);
        archive.insert(h1, vec![1], CompressionCodec::None, false).unwrap();
        let h2 = TensorHeader::new("x", vec![1], DType::U8);
        archive.insert(h2, vec![2], CompressionCodec::None, false).unwrap();
        let (_, d) = archive.get("x").unwrap();
        assert_eq!(d, &[2]);
        assert_eq!(archive.len(), 1);
    }

    // ── TensorSerializer orchestrator tests ─────────────────────────────

    #[test]
    fn serializer_binary_roundtrip() {
        let s = TensorSerializer::new(SerializationFormat::Binary, CompressionCodec::None, true);
        let header = TensorHeader::new("test", vec![4], DType::U8);
        let data = vec![1, 2, 3, 4];
        let (bytes, cksum) = s.serialize(&header, &data).unwrap();
        assert!(cksum.is_some());
        let (h2, d2) = s.deserialize(&bytes).unwrap();
        assert_eq!(h2.name, "test");
        assert_eq!(d2, data);
    }

    #[test]
    fn serializer_numpy_roundtrip() {
        let s = TensorSerializer::new(SerializationFormat::NumPy, CompressionCodec::None, false);
        let header = TensorHeader::new("", vec![3], DType::F32);
        let data = vec![0u8; 12];
        let (bytes, cksum) = s.serialize(&header, &data).unwrap();
        assert!(cksum.is_none());
        let (h2, d2) = s.deserialize(&bytes).unwrap();
        assert_eq!(h2.shape, vec![3]);
        assert_eq!(d2, data);
    }

    #[test]
    fn serializer_safetensors_roundtrip() {
        let s =
            TensorSerializer::new(SerializationFormat::SafeTensors, CompressionCodec::None, true);
        let header = TensorHeader::new("layer.0.weight", vec![2, 2], DType::F32);
        let data = vec![0u8; 16];
        let (bytes, cksum) = s.serialize(&header, &data).unwrap();
        assert!(cksum.is_some());
        let (h2, d2) = s.deserialize(&bytes).unwrap();
        assert_eq!(h2.name, "layer.0.weight");
        assert_eq!(d2, data);
    }

    #[test]
    fn serializer_json_roundtrip() {
        let s = TensorSerializer::new(SerializationFormat::Json, CompressionCodec::None, true);
        let header = TensorHeader::new("bias", vec![3], DType::U8);
        let data = vec![10, 20, 30];
        let (bytes, cksum) = s.serialize(&header, &data).unwrap();
        assert!(cksum.is_some());
        let (h2, d2) = s.deserialize(&bytes).unwrap();
        assert_eq!(h2.name, "bias");
        assert_eq!(d2, data);
    }

    #[test]
    fn serializer_gguf_unsupported() {
        let s = TensorSerializer::new(SerializationFormat::Gguf, CompressionCodec::None, false);
        let header = TensorHeader::new("x", vec![1], DType::U8);
        let err = s.serialize(&header, &[0]).unwrap_err();
        assert!(matches!(err, SerdeError::UnsupportedFormat(_)));
    }

    #[test]
    fn serializer_gguf_deserialize_unsupported() {
        let s = TensorSerializer::new(SerializationFormat::Gguf, CompressionCodec::None, false);
        let err = s.deserialize(b"anything").unwrap_err();
        assert!(matches!(err, SerdeError::UnsupportedFormat(_)));
    }

    #[test]
    fn serializer_no_checksum() {
        let s = TensorSerializer::new(SerializationFormat::Binary, CompressionCodec::None, false);
        let header = TensorHeader::new("x", vec![1], DType::U8);
        let (_, cksum) = s.serialize(&header, &[42]).unwrap();
        assert!(cksum.is_none());
    }

    #[test]
    fn serializer_data_mismatch() {
        let s = TensorSerializer::new(SerializationFormat::Binary, CompressionCodec::None, false);
        let header = TensorHeader::new("x", vec![4], DType::F32);
        let err = s.serialize(&header, &[0u8; 8]).unwrap_err();
        assert!(matches!(err, SerdeError::SizeMismatch { .. }));
    }

    // ── Base64 tests ────────────────────────────────────────────────────

    #[test]
    fn base64_encode_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn base64_roundtrip() {
        let data = b"Hello, World!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn base64_known_value() {
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"M"), "TQ==");
    }

    #[test]
    fn base64_roundtrip_binary() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn error_display_io() {
        let err = SerdeError::Io(io::Error::new(io::ErrorKind::NotFound, "nope"));
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn error_display_magic() {
        let err = SerdeError::InvalidMagic { expected: b"BTSR", actual: vec![0, 0, 0, 0] };
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn error_display_size() {
        let err = SerdeError::SizeMismatch { expected: 16, actual: 8 };
        assert!(err.to_string().contains("16"));
    }

    #[test]
    fn error_display_checksum() {
        let err = SerdeError::ChecksumMismatch { expected: "aaa".into(), actual: "bbb".into() };
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn error_display_tensor_not_found() {
        let err = SerdeError::TensorNotFound("missing".into());
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn error_source_io() {
        let inner = io::Error::other("inner");
        let err = SerdeError::Io(inner);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn error_source_non_io() {
        let err = SerdeError::InvalidHeader("bad".into());
        assert!(std::error::Error::source(&err).is_none());
    }

    // ── Internal JSON parser tests ──────────────────────────────────────

    #[test]
    fn json_parse_object() {
        let json = r#"{"key": "value", "num": 42}"#;
        let map = serde_json::from_str(json).unwrap();
        assert_eq!(map["key"].as_str(), Some("value"));
        assert_eq!(map["num"].as_u64(), Some(42));
    }

    #[test]
    fn json_parse_nested() {
        let json = r#"{"arr": [1, 2, 3], "nested": {"a": true}}"#;
        let map = serde_json::from_str(json).unwrap();
        assert_eq!(map["arr"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn json_emit_roundtrip() {
        let header = TensorHeader::new("test", vec![2, 3], DType::F32);
        let data = vec![0u8; 24];
        let s = TensorSerializer::new(SerializationFormat::Json, CompressionCodec::None, false);
        let (bytes, _) = s.serialize(&header, &data).unwrap();
        let (h2, d2) = s.deserialize(&bytes).unwrap();
        assert_eq!(h2.shape, vec![2, 3]);
        assert_eq!(d2, data);
    }

    #[test]
    fn json_parse_escaped_string() {
        let json = r#"{"msg": "hello \"world\""}"#;
        let map = serde_json::from_str(json).unwrap();
        assert_eq!(map["msg"].as_str(), Some("hello \"world\""));
    }

    // ── parse_dtype tests ───────────────────────────────────────────────

    #[test]
    fn parse_dtype_all_variants() {
        let cases = [
            ("f16", DType::F16),
            ("F16", DType::F16),
            ("f32", DType::F32),
            ("f64", DType::F64),
            ("i8", DType::I8),
            ("i16", DType::I16),
            ("i32", DType::I32),
            ("i64", DType::I64),
            ("u8", DType::U8),
            ("u16", DType::U16),
            ("u32", DType::U32),
            ("u64", DType::U64),
            ("bf16", DType::BF16),
            ("BF16", DType::BF16),
        ];
        for (s, expected) in cases {
            assert_eq!(parse_dtype(s).unwrap(), expected);
        }
    }

    #[test]
    fn parse_dtype_unknown() {
        let err = parse_dtype("complex128").unwrap_err();
        assert!(matches!(err, SerdeError::InvalidHeader(_)));
    }

    // ── Endianness tests ────────────────────────────────────────────────

    #[test]
    fn endianness_default_is_little() {
        assert_eq!(Endianness::default(), Endianness::Little);
    }

    // ── SHA-256 additional tests ────────────────────────────────────────

    #[test]
    fn sha256_abc() {
        let digest = sha256(b"abc");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sha256_long_message() {
        // 1000 zero bytes
        let data = vec![0u8; 1000];
        let cksum = TensorChecksum::compute(&data);
        assert_eq!(cksum.to_hex().len(), 64);
        // Deterministic
        let cksum2 = TensorChecksum::compute(&data);
        assert_eq!(cksum, cksum2);
    }

    // ── dtype_to_tag / dtype_from_tag roundtrip ─────────────────────────

    #[test]
    fn dtype_tag_roundtrip() {
        let dtypes = [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::BF16,
        ];
        for dt in dtypes {
            let tag = dtype_to_tag(dt);
            assert_eq!(dtype_from_tag(tag), Some(dt));
        }
    }

    #[test]
    fn dtype_from_tag_unknown() {
        assert_eq!(dtype_from_tag(255), None);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_dtype() -> impl Strategy<Value = DType> {
        prop_oneof![
            Just(DType::U8),
            Just(DType::U16),
            Just(DType::U32),
            Just(DType::U64),
            Just(DType::I8),
            Just(DType::I16),
            Just(DType::I32),
            Just(DType::I64),
            Just(DType::F16),
            Just(DType::F32),
            Just(DType::F64),
            Just(DType::BF16),
        ]
    }

    proptest! {
        #[test]
        fn binary_roundtrip_prop(
            name in "[a-z]{1,16}",
            dim1 in 1usize..8,
            dim2 in 1usize..8,
            dtype in arb_dtype(),
        ) {
            let shape = vec![dim1, dim2];
            let header = TensorHeader::new(name, shape, dtype);
            let data = vec![0xABu8; header.byte_size as usize];
            let mut buf = Vec::new();
            BinarySerializer::serialize(
                &mut buf,
                &header,
                &data,
                CompressionCodec::None,
            )
            .unwrap();
            let (h2, d2) =
                BinaryDeserializer::deserialize(&mut Cursor::new(&buf))
                    .unwrap();
            prop_assert_eq!(&h2.name, &header.name);
            prop_assert_eq!(&h2.shape, &header.shape);
            prop_assert_eq!(h2.dtype, header.dtype);
            prop_assert_eq!(d2, data);
        }

        #[test]
        fn checksum_roundtrip_prop(data in proptest::collection::vec(any::<u8>(), 0..256)) {
            let cksum = TensorChecksum::compute(&data);
            let hex = cksum.to_hex();
            let parsed = TensorChecksum::from_hex(&hex).unwrap();
            prop_assert_eq!(cksum, parsed.clone());
            prop_assert!(parsed.verify(&data).is_ok());
        }

        #[test]
        fn base64_roundtrip_prop(data in proptest::collection::vec(any::<u8>(), 0..128)) {
            let encoded = base64_encode(&data);
            let decoded = base64_decode(&encoded).unwrap();
            prop_assert_eq!(decoded, data);
        }

        #[test]
        fn numpy_roundtrip_prop(
            dim in 1usize..16,
            dtype in arb_dtype(),
        ) {
            let header = TensorHeader::new("", vec![dim], dtype);
            let data = vec![0u8; header.byte_size as usize];
            let mut buf = Vec::new();
            NumpyFormat::serialize(&mut buf, &header, &data).unwrap();
            let (h2, d2) =
                NumpyFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
            prop_assert_eq!(&h2.shape, &header.shape);
            prop_assert_eq!(h2.dtype, header.dtype);
            prop_assert_eq!(d2, data);
        }

        #[test]
        fn safetensors_roundtrip_prop(
            name in "[a-z.]{1,20}",
            dim in 1usize..8,
        ) {
            let header = TensorHeader::new(name, vec![dim], DType::F32);
            let data = vec![0u8; header.byte_size as usize];
            let mut buf = Vec::new();
            SafeTensorsFormat::serialize(&mut buf, &[(&header, &data)]).unwrap();
            let tensors =
                SafeTensorsFormat::deserialize(&mut Cursor::new(&buf)).unwrap();
            prop_assert_eq!(tensors.len(), 1);
            prop_assert_eq!(&tensors[0].0.name, &header.name);
            prop_assert_eq!(&tensors[0].1, &data);
        }
    }
}
