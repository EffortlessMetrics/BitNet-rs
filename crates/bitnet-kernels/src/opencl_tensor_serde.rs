//! Tensor serialization/deserialization for model loading and checkpointing.
//!
//! Provides a binary format for storing tensors with metadata (dtype, shape,
//! strides, byte order, compression) suitable for GPU weight loading. All
//! operations have CPU reference implementations so the module compiles and
//! tests without an actual OpenCL runtime.
//!
//! # Format layout
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │  magic (4 bytes): "BTSR"             │
//! │  version (u16)                       │
//! │  header_len (u32)                    │
//! │  [TensorHeader – variable length]    │
//! │  checksum_algo (u8)                  │
//! │  data_checksum (u64)                 │
//! │  [compressed/raw tensor data]        │
//! └──────────────────────────────────────┘
//! ```
//!
//! # Bundle layout (multi-tensor)
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │  magic (4 bytes): "BTSB"             │
//! │  version (u16)                       │
//! │  tensor_count (u32)                  │
//! │  [index: name_len, name, offset]*    │
//! │  [tensor data blocks]*               │
//! └──────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Magic bytes and version
// ---------------------------------------------------------------------------

const TENSOR_MAGIC: [u8; 4] = *b"BTSR";
const BUNDLE_MAGIC: [u8; 4] = *b"BTSB";
const FORMAT_VERSION: u16 = 1;

// ---------------------------------------------------------------------------
// DType — element data type
// ---------------------------------------------------------------------------

/// Element data type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    I2 = 4,
    U8 = 5,
    I32 = 6,
    I16 = 7,
    F64 = 8,
}

impl DType {
    /// Size of one element in bytes (I2 returns 1 for packed-byte storage).
    pub fn byte_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::I2 => 1,
            Self::F64 => 8,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::BF16),
            3 => Some(Self::I8),
            4 => Some(Self::I2),
            5 => Some(Self::U8),
            6 => Some(Self::I32),
            7 => Some(Self::I16),
            8 => Some(Self::F64),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2 => write!(f, "i2"),
            Self::U8 => write!(f, "u8"),
            Self::I32 => write!(f, "i32"),
            Self::I16 => write!(f, "i16"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// ByteOrder
// ---------------------------------------------------------------------------

/// Byte order for multi-byte elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ByteOrder {
    #[default]
    LittleEndian = 0,
    BigEndian = 1,
}

impl ByteOrder {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::LittleEndian),
            1 => Some(Self::BigEndian),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionCodec
// ---------------------------------------------------------------------------

/// Compression algorithm for tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionCodec {
    None,
    /// LZ4 block compression (CPU reference: simple RLE-like fallback).
    Lz4,
    /// Zstandard with a configurable level (1–22).
    Zstd {
        level: u8,
    },
}

impl CompressionCodec {
    fn tag(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Zstd { .. } => 2,
        }
    }

    fn level(self) -> u8 {
        match self {
            Self::Zstd { level } => level,
            _ => 0,
        }
    }

    fn from_tag_level(tag: u8, level: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Zstd { level }),
            _ => None,
        }
    }

    /// CPU reference: compress `data`.
    ///
    /// The CPU reference implementation uses a simple RLE scheme for LZ4/Zstd
    /// placeholders so the module works without external codec libraries.
    pub fn compress(self, data: &[u8]) -> Vec<u8> {
        match self {
            Self::None => data.to_vec(),
            Self::Lz4 => cpu_rle_compress(data),
            Self::Zstd { level } => {
                // Higher levels → extra pass (placeholder for real zstd).
                let mut out = cpu_rle_compress(data);
                if level > 10 {
                    out = cpu_rle_compress(&out);
                }
                out
            }
        }
    }

    /// CPU reference: decompress `data` previously compressed with this codec.
    pub fn decompress(self, data: &[u8]) -> Result<Vec<u8>, SerdeError> {
        match self {
            Self::None => Ok(data.to_vec()),
            Self::Lz4 => cpu_rle_decompress(data),
            Self::Zstd { level } => {
                let inner = if level > 10 { cpu_rle_decompress(data)? } else { data.to_vec() };
                cpu_rle_decompress(&inner)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference RLE codec (stand-in for LZ4/Zstd)
// ---------------------------------------------------------------------------

/// Simple run-length encoding: [marker=0xFF, byte, count_hi, count_lo] for
/// runs ≥ 3, literal bytes otherwise. Marker byte itself is escaped as
/// [0xFF, 0xFF, 0x00, 0x01].
fn cpu_rle_compress(data: &[u8]) -> Vec<u8> {
    const MARKER: u8 = 0xFF;
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run = 1usize;
        while i + run < data.len() && data[i + run] == byte && run < 65535 {
            run += 1;
        }
        if run >= 3 {
            out.push(MARKER);
            out.push(byte);
            out.push((run >> 8) as u8);
            out.push(run as u8);
            i += run;
        } else {
            if byte == MARKER {
                out.push(MARKER);
                out.push(MARKER);
                out.push(0x00);
                out.push(0x01);
            } else {
                out.push(byte);
            }
            i += 1;
            // Emit remaining non-run bytes one-by-one.
            for _ in 1..run {
                let b = data[i];
                if b == MARKER {
                    out.push(MARKER);
                    out.push(MARKER);
                    out.push(0x00);
                    out.push(0x01);
                } else {
                    out.push(b);
                }
                i += 1;
            }
        }
    }
    out
}

fn cpu_rle_decompress(data: &[u8]) -> Result<Vec<u8>, SerdeError> {
    const MARKER: u8 = 0xFF;
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == MARKER {
            if i + 3 >= data.len() {
                return Err(SerdeError::DecompressionFailed("truncated RLE marker".into()));
            }
            let byte = data[i + 1];
            let count = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            if byte == MARKER && count == 1 && data[i + 2] == 0 {
                // Escaped literal 0xFF.
                out.push(MARKER);
            } else {
                for _ in 0..count {
                    out.push(byte);
                }
            }
            i += 4;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ChecksumAlgorithm / ChecksumValidator
// ---------------------------------------------------------------------------

/// Checksum algorithm for data integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChecksumAlgorithm {
    None = 0,
    Crc32 = 1,
    XxHash64 = 2,
}

impl ChecksumAlgorithm {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Crc32),
            2 => Some(Self::XxHash64),
            _ => None,
        }
    }
}

/// CPU reference checksum computation and validation.
#[derive(Debug)]
pub struct ChecksumValidator;

impl ChecksumValidator {
    /// Compute a checksum over `data`.
    pub fn compute(algo: ChecksumAlgorithm, data: &[u8]) -> u64 {
        match algo {
            ChecksumAlgorithm::None => 0,
            ChecksumAlgorithm::Crc32 => Self::cpu_crc32(data) as u64,
            ChecksumAlgorithm::XxHash64 => Self::cpu_xxhash64(data),
        }
    }

    /// Validate `data` against an expected checksum.
    pub fn validate(algo: ChecksumAlgorithm, data: &[u8], expected: u64) -> Result<(), SerdeError> {
        let actual = Self::compute(algo, data);
        if algo == ChecksumAlgorithm::None || actual == expected {
            Ok(())
        } else {
            Err(SerdeError::ChecksumMismatch { expected, actual })
        }
    }

    /// CPU reference CRC-32 (ISO 3309 / ITU-T V.42 polynomial).
    fn cpu_crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    /// CPU reference xxHash64 (simplified — good enough for integrity checks).
    fn cpu_xxhash64(data: &[u8]) -> u64 {
        const PRIME1: u64 = 0x9E37_79B1_85EB_CA87;
        const PRIME2: u64 = 0xC2B2_AE3D_27D4_EB4F;
        const PRIME3: u64 = 0x1656_67B1_9E37_79F9;
        const PRIME5: u64 = 0x27D4_EB2F_1656_67C5;

        let len = data.len() as u64;
        let mut h: u64 = PRIME5.wrapping_add(len);

        let mut i = 0;
        while i + 8 <= data.len() {
            let k = u64::from_le_bytes([
                data[i],
                data[i + 1],
                data[i + 2],
                data[i + 3],
                data[i + 4],
                data[i + 5],
                data[i + 6],
                data[i + 7],
            ]);
            h ^= k.wrapping_mul(PRIME2).rotate_left(31).wrapping_mul(PRIME1);
            h = h.rotate_left(27).wrapping_mul(PRIME1).wrapping_add(PRIME3);
            i += 8;
        }
        while i + 4 <= data.len() {
            let k = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as u64;
            h ^= k.wrapping_mul(PRIME1);
            h = h.rotate_left(23).wrapping_mul(PRIME2).wrapping_add(PRIME3);
            i += 4;
        }
        while i < data.len() {
            h ^= (data[i] as u64).wrapping_mul(PRIME5);
            h = h.rotate_left(11).wrapping_mul(PRIME1);
            i += 1;
        }

        // Avalanche.
        h ^= h >> 33;
        h = h.wrapping_mul(PRIME2);
        h ^= h >> 29;
        h = h.wrapping_mul(PRIME3);
        h ^= h >> 32;
        h
    }
}

// ---------------------------------------------------------------------------
// SerdeError
// ---------------------------------------------------------------------------

/// Errors produced by tensor serde operations.
#[derive(Debug)]
pub enum SerdeError {
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u16),
    InvalidDType(u8),
    InvalidByteOrder(u8),
    InvalidCompression(u8),
    InvalidChecksum(u8),
    ChecksumMismatch { expected: u64, actual: u64 },
    DecompressionFailed(String),
    ShapeMismatch { expected: usize, actual: usize },
    Io(io::Error),
    TensorNotFound(String),
    InvalidHeaderLength,
    BufferTooSmall { needed: usize, got: usize },
    InvalidNameLength(u32),
}

impl fmt::Display for SerdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic(m) => {
                write!(f, "invalid magic: {:?}", m)
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported format version: {v}")
            }
            Self::InvalidDType(d) => write!(f, "invalid dtype tag: {d}"),
            Self::InvalidByteOrder(b) => {
                write!(f, "invalid byte order: {b}")
            }
            Self::InvalidCompression(c) => {
                write!(f, "invalid compression tag: {c}")
            }
            Self::InvalidChecksum(c) => {
                write!(f, "invalid checksum algorithm: {c}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#x}, got {actual:#x}")
            }
            Self::DecompressionFailed(msg) => {
                write!(f, "decompression failed: {msg}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected} bytes, got {actual}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::TensorNotFound(name) => {
                write!(f, "tensor not found: {name}")
            }
            Self::InvalidHeaderLength => write!(f, "invalid header length"),
            Self::BufferTooSmall { needed, got } => {
                write!(f, "buffer too small: need {needed}, got {got}")
            }
            Self::InvalidNameLength(n) => {
                write!(f, "invalid name length: {n}")
            }
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

// ---------------------------------------------------------------------------
// TensorHeader
// ---------------------------------------------------------------------------

/// Metadata describing a serialized tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeader {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub byte_order: ByteOrder,
    pub compression: CompressionCodec,
    pub checksum_algo: ChecksumAlgorithm,
    /// Raw (uncompressed) data size in bytes.
    pub raw_data_len: u64,
    /// Compressed data size in bytes (== raw_data_len when uncompressed).
    pub compressed_data_len: u64,
    /// Checksum of the *compressed* data blob.
    pub data_checksum: u64,
}

impl TensorHeader {
    /// Compute the expected raw byte count from shape and dtype.
    pub fn expected_raw_bytes(&self) -> usize {
        let elems: usize = self.shape.iter().product();
        elems * self.dtype.byte_size()
    }

    /// Compute default row-major (C-order) strides from shape and dtype.
    pub fn row_major_strides(shape: &[usize], dtype: DType) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![0usize; shape.len()];
        strides[shape.len() - 1] = dtype.byte_size();
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Encode header into a byte vector.
    fn encode(&self) -> Vec<u8> {
        let mut buf = vec![
            self.dtype as u8,
            self.byte_order as u8,
            self.compression.tag(),
            self.compression.level(),
            self.checksum_algo as u8,
            // ndim (u8)
            self.shape.len() as u8,
        ];

        // shape
        for &d in &self.shape {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        // strides
        for &s in &self.strides {
            buf.extend_from_slice(&(s as u64).to_le_bytes());
        }

        buf.extend_from_slice(&self.raw_data_len.to_le_bytes());
        buf.extend_from_slice(&self.compressed_data_len.to_le_bytes());
        buf.extend_from_slice(&self.data_checksum.to_le_bytes());
        buf
    }

    /// Decode header from a byte slice.  Returns `(header, bytes_consumed)`.
    fn decode(data: &[u8]) -> Result<(Self, usize), SerdeError> {
        if data.len() < 5 {
            return Err(SerdeError::InvalidHeaderLength);
        }
        let dtype = DType::from_u8(data[0]).ok_or(SerdeError::InvalidDType(data[0]))?;
        let byte_order =
            ByteOrder::from_u8(data[1]).ok_or(SerdeError::InvalidByteOrder(data[1]))?;
        let comp_tag = data[2];
        let comp_level = data[3];
        let compression = CompressionCodec::from_tag_level(comp_tag, comp_level)
            .ok_or(SerdeError::InvalidCompression(comp_tag))?;
        let checksum_algo =
            ChecksumAlgorithm::from_u8(data[4]).ok_or(SerdeError::InvalidChecksum(data[4]))?;

        if data.len() < 6 {
            return Err(SerdeError::InvalidHeaderLength);
        }
        let ndim = data[5] as usize;

        // We need 6 + ndim*8 (shape) + ndim*8 (strides) + 8+8+8 = 30 +
        // 16*ndim
        let needed = 6 + ndim * 16 + 24;
        if data.len() < needed {
            return Err(SerdeError::InvalidHeaderLength);
        }

        let mut off = 6;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let v = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
            shape.push(v);
            off += 8;
        }
        let mut strides = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let v = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
            strides.push(v);
            off += 8;
        }

        let raw_data_len = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        off += 8;
        let compressed_data_len = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        off += 8;
        let data_checksum = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        off += 8;

        Ok((
            Self {
                dtype,
                shape,
                strides,
                byte_order,
                compression,
                checksum_algo,
                raw_data_len,
                compressed_data_len,
                data_checksum,
            },
            off,
        ))
    }
}

// ---------------------------------------------------------------------------
// SerdeStats
// ---------------------------------------------------------------------------

/// Performance statistics for a read or write operation.
#[derive(Debug, Clone)]
pub struct SerdeStats {
    /// Wall-clock time for the operation.
    pub elapsed: Duration,
    /// Uncompressed data size in bytes.
    pub raw_bytes: u64,
    /// Compressed (on-wire/on-disk) size in bytes.
    pub compressed_bytes: u64,
}

impl SerdeStats {
    /// Compression ratio (>= 1.0 means compression helped).
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            return 1.0;
        }
        self.raw_bytes as f64 / self.compressed_bytes as f64
    }

    /// Throughput in bytes/second based on raw (uncompressed) size.
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 {
            return f64::INFINITY;
        }
        self.raw_bytes as f64 / secs
    }
}

impl fmt::Display for SerdeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "raw={} compressed={} ratio={:.2}x elapsed={:.3}ms",
            self.raw_bytes,
            self.compressed_bytes,
            self.compression_ratio(),
            self.elapsed.as_secs_f64() * 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// TensorWriter
// ---------------------------------------------------------------------------

/// Serializes a single tensor to a binary blob.
pub struct TensorWriter {
    pub compression: CompressionCodec,
    pub checksum_algo: ChecksumAlgorithm,
    pub byte_order: ByteOrder,
}

impl Default for TensorWriter {
    fn default() -> Self {
        Self {
            compression: CompressionCodec::None,
            checksum_algo: ChecksumAlgorithm::Crc32,
            byte_order: ByteOrder::LittleEndian,
        }
    }
}

impl TensorWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = codec;
        self
    }

    pub fn with_checksum(mut self, algo: ChecksumAlgorithm) -> Self {
        self.checksum_algo = algo;
        self
    }

    pub fn with_byte_order(mut self, order: ByteOrder) -> Self {
        self.byte_order = order;
        self
    }

    /// Serialize `data` with the given `dtype` and `shape` into `writer`.
    pub fn write<W: Write>(
        &self,
        writer: &mut W,
        dtype: DType,
        shape: &[usize],
        data: &[u8],
    ) -> Result<SerdeStats, SerdeError> {
        let start = Instant::now();
        let raw_len = data.len() as u64;

        let compressed = self.compression.compress(data);
        let compressed_len = compressed.len() as u64;
        let checksum = ChecksumValidator::compute(self.checksum_algo, &compressed);
        let strides = TensorHeader::row_major_strides(shape, dtype);

        let header = TensorHeader {
            dtype,
            shape: shape.to_vec(),
            strides,
            byte_order: self.byte_order,
            compression: self.compression,
            checksum_algo: self.checksum_algo,
            raw_data_len: raw_len,
            compressed_data_len: compressed_len,
            data_checksum: checksum,
        };

        let hdr_bytes = header.encode();

        writer.write_all(&TENSOR_MAGIC)?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&(hdr_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&hdr_bytes)?;
        writer.write_all(&compressed)?;

        Ok(SerdeStats {
            elapsed: start.elapsed(),
            raw_bytes: raw_len,
            compressed_bytes: compressed_len,
        })
    }

    /// Convenience: serialize to a `Vec<u8>`.
    pub fn write_to_vec(
        &self,
        dtype: DType,
        shape: &[usize],
        data: &[u8],
    ) -> Result<(Vec<u8>, SerdeStats), SerdeError> {
        let mut buf = Vec::new();
        let stats = self.write(&mut buf, dtype, shape, data)?;
        Ok((buf, stats))
    }
}

// ---------------------------------------------------------------------------
// TensorReader
// ---------------------------------------------------------------------------

/// Deserializes a single tensor from a binary blob.
pub struct TensorReader;

impl TensorReader {
    /// Read a tensor from `reader`, returning header, raw data, and stats.
    pub fn read<R: Read>(
        reader: &mut R,
    ) -> Result<(TensorHeader, Vec<u8>, SerdeStats), SerdeError> {
        let start = Instant::now();

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != TENSOR_MAGIC {
            return Err(SerdeError::InvalidMagic(magic));
        }

        let mut ver_buf = [0u8; 2];
        reader.read_exact(&mut ver_buf)?;
        let version = u16::from_le_bytes(ver_buf);
        if version != FORMAT_VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }

        let mut hdr_len_buf = [0u8; 4];
        reader.read_exact(&mut hdr_len_buf)?;
        let hdr_len = u32::from_le_bytes(hdr_len_buf) as usize;

        let mut hdr_buf = vec![0u8; hdr_len];
        reader.read_exact(&mut hdr_buf)?;
        let (header, _) = TensorHeader::decode(&hdr_buf)?;

        let mut compressed = vec![0u8; header.compressed_data_len as usize];
        reader.read_exact(&mut compressed)?;

        ChecksumValidator::validate(header.checksum_algo, &compressed, header.data_checksum)?;

        let raw = header.compression.decompress(&compressed)?;

        if raw.len() as u64 != header.raw_data_len {
            return Err(SerdeError::ShapeMismatch {
                expected: header.raw_data_len as usize,
                actual: raw.len(),
            });
        }

        let stats = SerdeStats {
            elapsed: start.elapsed(),
            raw_bytes: header.raw_data_len,
            compressed_bytes: header.compressed_data_len,
        };

        Ok((header, raw, stats))
    }

    /// Convenience: read from a byte slice.
    pub fn read_from_slice(data: &[u8]) -> Result<(TensorHeader, Vec<u8>, SerdeStats), SerdeError> {
        let mut cursor = io::Cursor::new(data);
        Self::read(&mut cursor)
    }
}

// ---------------------------------------------------------------------------
// MemoryMapReader — zero-copy tensor access (CPU simulation)
// ---------------------------------------------------------------------------

/// Simulated memory-mapped reader for zero-copy tensor access.
///
/// In production the backing buffer would be an OS mmap region; for CPU
/// reference testing we accept a pre-loaded `Vec<u8>`.
pub struct MemoryMapReader {
    buffer: Vec<u8>,
    header: TensorHeader,
    data_offset: usize,
}

impl MemoryMapReader {
    /// Create from a serialized tensor blob.
    pub fn from_bytes(blob: Vec<u8>) -> Result<Self, SerdeError> {
        if blob.len() < 10 {
            return Err(SerdeError::InvalidHeaderLength);
        }
        if blob[0..4] != TENSOR_MAGIC {
            return Err(SerdeError::InvalidMagic(blob[0..4].try_into().unwrap()));
        }
        let version = u16::from_le_bytes(blob[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }
        let hdr_len = u32::from_le_bytes(blob[6..10].try_into().unwrap()) as usize;
        let data_offset = 10 + hdr_len;
        if blob.len() < data_offset {
            return Err(SerdeError::InvalidHeaderLength);
        }
        let (header, _) = TensorHeader::decode(&blob[10..data_offset])?;

        Ok(Self { buffer: blob, header, data_offset })
    }

    /// Get the tensor header.
    pub fn header(&self) -> &TensorHeader {
        &self.header
    }

    /// Zero-copy access to the compressed data region.
    pub fn compressed_data(&self) -> &[u8] {
        &self.buffer[self.data_offset..]
    }

    /// Decompress and return the raw tensor data.
    pub fn decompress(&self) -> Result<Vec<u8>, SerdeError> {
        let compressed = self.compressed_data();
        ChecksumValidator::validate(
            self.header.checksum_algo,
            compressed,
            self.header.data_checksum,
        )?;
        self.header.compression.decompress(compressed)
    }

    /// Total size of the memory-mapped region.
    pub fn mapped_size(&self) -> usize {
        self.buffer.len()
    }
}

// ---------------------------------------------------------------------------
// StreamingReader — progressive tensor reading
// ---------------------------------------------------------------------------

/// Reads a tensor progressively in fixed-size chunks (for large models that
/// don't fit in RAM all at once).
pub struct StreamingReader<R: Read> {
    inner: R,
    header: TensorHeader,
    bytes_remaining: usize,
    chunk_size: usize,
}

impl<R: Read> StreamingReader<R> {
    /// Begin streaming from `reader` with the given chunk size.
    pub fn new(mut reader: R, chunk_size: usize) -> Result<Self, SerdeError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != TENSOR_MAGIC {
            return Err(SerdeError::InvalidMagic(magic));
        }
        let mut ver_buf = [0u8; 2];
        reader.read_exact(&mut ver_buf)?;
        let version = u16::from_le_bytes(ver_buf);
        if version != FORMAT_VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }
        let mut hdr_len_buf = [0u8; 4];
        reader.read_exact(&mut hdr_len_buf)?;
        let hdr_len = u32::from_le_bytes(hdr_len_buf) as usize;

        let mut hdr_buf = vec![0u8; hdr_len];
        reader.read_exact(&mut hdr_buf)?;
        let (header, _) = TensorHeader::decode(&hdr_buf)?;

        let bytes_remaining = header.compressed_data_len as usize;

        Ok(Self { inner: reader, header, bytes_remaining, chunk_size })
    }

    /// Header for the tensor being read.
    pub fn header(&self) -> &TensorHeader {
        &self.header
    }

    /// How many compressed bytes remain.
    pub fn remaining(&self) -> usize {
        self.bytes_remaining
    }

    /// Read the next chunk.  Returns `None` when the tensor is fully read.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, SerdeError> {
        if self.bytes_remaining == 0 {
            return Ok(None);
        }
        let to_read = self.chunk_size.min(self.bytes_remaining);
        let mut buf = vec![0u8; to_read];
        self.inner.read_exact(&mut buf)?;
        self.bytes_remaining -= to_read;
        Ok(Some(buf))
    }

    /// Read all remaining chunks into a single buffer.
    pub fn read_all(&mut self) -> Result<Vec<u8>, SerdeError> {
        let mut all = Vec::with_capacity(self.bytes_remaining);
        while let Some(chunk) = self.next_chunk()? {
            all.extend_from_slice(&chunk);
        }
        Ok(all)
    }
}

// ---------------------------------------------------------------------------
// TensorBundle — multi-tensor container
// ---------------------------------------------------------------------------

/// A multi-tensor container (similar to the safetensors format).
///
/// Stores an ordered collection of named tensors in a single blob.
pub struct TensorBundle {
    tensors: Vec<(String, DType, Vec<usize>, Vec<u8>)>,
    compression: CompressionCodec,
    checksum_algo: ChecksumAlgorithm,
}

impl TensorBundle {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            compression: CompressionCodec::None,
            checksum_algo: ChecksumAlgorithm::Crc32,
        }
    }

    pub fn with_compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = codec;
        self
    }

    pub fn with_checksum(mut self, algo: ChecksumAlgorithm) -> Self {
        self.checksum_algo = algo;
        self
    }

    /// Add a named tensor.
    pub fn add(&mut self, name: impl Into<String>, dtype: DType, shape: Vec<usize>, data: Vec<u8>) {
        self.tensors.push((name.into(), dtype, shape, data));
    }

    /// Number of tensors in the bundle.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Serialize the entire bundle.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<SerdeStats, SerdeError> {
        let start = Instant::now();
        let mut total_raw: u64 = 0;
        let mut total_compressed: u64 = 0;

        writer.write_all(&BUNDLE_MAGIC)?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&(self.tensors.len() as u32).to_le_bytes())?;

        // Serialize each tensor into temp buffers to build the index.
        let tw = TensorWriter::new()
            .with_compression(self.compression)
            .with_checksum(self.checksum_algo);

        let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(self.tensors.len());
        for (_, dtype, shape, data) in &self.tensors {
            let (blob, st) = tw.write_to_vec(*dtype, shape, data)?;
            total_raw += st.raw_bytes;
            total_compressed += st.compressed_bytes;
            blobs.push(blob);
        }

        // Write index: for each tensor [name_len(u32), name, blob_len(u64)].
        for (i, (name, ..)) in self.tensors.iter().enumerate() {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            writer.write_all(&(blobs[i].len() as u64).to_le_bytes())?;
        }

        // Write tensor blobs.
        for blob in &blobs {
            writer.write_all(blob)?;
        }

        Ok(SerdeStats {
            elapsed: start.elapsed(),
            raw_bytes: total_raw,
            compressed_bytes: total_compressed,
        })
    }

    /// Serialize to a `Vec<u8>`.
    pub fn write_to_vec(&self) -> Result<(Vec<u8>, SerdeStats), SerdeError> {
        let mut buf = Vec::new();
        let stats = self.write(&mut buf)?;
        Ok((buf, stats))
    }

    /// Deserialize a bundle from `reader`.
    pub fn read<R: Read>(reader: &mut R) -> Result<(BundleIndex, Vec<u8>), SerdeError> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Self::read_from_slice(&buf)
    }

    /// Deserialize a bundle from a byte slice.
    pub fn read_from_slice(data: &[u8]) -> Result<(BundleIndex, Vec<u8>), SerdeError> {
        if data.len() < 10 {
            return Err(SerdeError::InvalidHeaderLength);
        }
        let magic: [u8; 4] = data[0..4].try_into().unwrap();
        if magic != BUNDLE_MAGIC {
            return Err(SerdeError::InvalidMagic(magic));
        }
        let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }
        let tensor_count = u32::from_le_bytes(data[6..10].try_into().unwrap()) as usize;

        let mut off = 10;
        let mut entries: Vec<(String, u64)> = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            if off + 4 > data.len() {
                return Err(SerdeError::InvalidHeaderLength);
            }
            let name_len = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            off += 4;
            if name_len > 10_000 {
                return Err(SerdeError::InvalidNameLength(name_len));
            }
            let name_end = off + name_len as usize;
            if name_end + 8 > data.len() {
                return Err(SerdeError::InvalidHeaderLength);
            }
            let name = String::from_utf8_lossy(&data[off..name_end]).to_string();
            off = name_end;
            let blob_len = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            off += 8;
            entries.push((name, blob_len));
        }

        // Build index with byte offsets into the data blob region.
        let blob_base = off;
        let mut index = BundleIndex::new();
        let mut blob_off = blob_base;
        for (name, blob_len) in &entries {
            index.insert(name.clone(), blob_off, *blob_len as usize);
            blob_off += *blob_len as usize;
        }

        Ok((index, data.to_vec()))
    }

    /// Read a single named tensor from an already-parsed bundle.
    pub fn read_tensor(
        name: &str,
        index: &BundleIndex,
        data: &[u8],
    ) -> Result<(TensorHeader, Vec<u8>, SerdeStats), SerdeError> {
        let entry = index.get(name).ok_or_else(|| SerdeError::TensorNotFound(name.into()))?;
        let blob = &data[entry.offset..entry.offset + entry.length];
        TensorReader::read_from_slice(blob)
    }
}

impl Default for TensorBundle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BundleIndex
// ---------------------------------------------------------------------------

/// Index entry pointing into a bundle's raw data.
#[derive(Debug, Clone)]
pub struct BundleIndexEntry {
    pub offset: usize,
    pub length: usize,
}

/// Name → location mapping for tensors in a bundle.
#[derive(Debug, Clone, Default)]
pub struct BundleIndex {
    entries: HashMap<String, BundleIndexEntry>,
    order: Vec<String>,
}

impl BundleIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, name: String, offset: usize, length: usize) {
        self.entries.insert(name.clone(), BundleIndexEntry { offset, length });
        self.order.push(name);
    }

    pub fn get(&self, name: &str) -> Option<&BundleIndexEntry> {
        self.entries.get(name)
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helper -----------------------------------------------------------

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn sample_f32_data(n: usize) -> Vec<u8> {
        let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        f32_to_bytes(&vals)
    }

    // =====================================================================
    // DType tests
    // =====================================================================

    #[test]
    fn test_dtype_byte_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::I2.byte_size(), 1);
        assert_eq!(DType::U8.byte_size(), 1);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::I16.byte_size(), 2);
        assert_eq!(DType::F64.byte_size(), 8);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::I2.to_string(), "i2");
    }

    #[test]
    fn test_dtype_roundtrip() {
        for tag in 0..=8u8 {
            let dt = DType::from_u8(tag).unwrap();
            assert_eq!(dt as u8, tag);
        }
        assert!(DType::from_u8(99).is_none());
    }

    // =====================================================================
    // ByteOrder tests
    // =====================================================================

    #[test]
    fn test_byte_order_default() {
        assert_eq!(ByteOrder::default(), ByteOrder::LittleEndian);
    }

    #[test]
    fn test_byte_order_roundtrip() {
        assert_eq!(ByteOrder::from_u8(0), Some(ByteOrder::LittleEndian));
        assert_eq!(ByteOrder::from_u8(1), Some(ByteOrder::BigEndian));
        assert!(ByteOrder::from_u8(2).is_none());
    }

    // =====================================================================
    // CompressionCodec tests
    // =====================================================================

    #[test]
    fn test_compression_none_roundtrip() {
        let data = b"hello world";
        let c = CompressionCodec::None.compress(data);
        let d = CompressionCodec::None.decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compression_lz4_roundtrip() {
        let data = vec![0xABu8; 1000];
        let c = CompressionCodec::Lz4.compress(&data);
        assert!(c.len() < data.len(), "LZ4 should compress repeated data");
        let d = CompressionCodec::Lz4.decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compression_zstd_low_level_roundtrip() {
        let data = vec![0x42u8; 500];
        let codec = CompressionCodec::Zstd { level: 3 };
        let c = codec.compress(&data);
        let d = codec.decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compression_zstd_high_level_roundtrip() {
        let data = vec![0x42u8; 500];
        let codec = CompressionCodec::Zstd { level: 15 };
        let c = codec.compress(&data);
        let d = codec.decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compression_codec_tag_roundtrip() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::Zstd { level: 5 },
            CompressionCodec::Zstd { level: 15 },
        ];
        for c in codecs {
            let decoded = CompressionCodec::from_tag_level(c.tag(), c.level()).unwrap();
            assert_eq!(decoded.tag(), c.tag());
            assert_eq!(decoded.level(), c.level());
        }
    }

    #[test]
    fn test_compression_invalid_tag() {
        assert!(CompressionCodec::from_tag_level(99, 0).is_none());
    }

    #[test]
    fn test_rle_marker_escape() {
        // Data containing the marker byte 0xFF.
        let data = vec![0xFF, 0xFF, 0x00, 0x01, 0x42];
        let c = cpu_rle_compress(&data);
        let d = cpu_rle_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_rle_empty() {
        let c = cpu_rle_compress(&[]);
        assert!(c.is_empty());
        let d = cpu_rle_decompress(&c).unwrap();
        assert!(d.is_empty());
    }

    #[test]
    fn test_rle_all_marker_bytes() {
        let data = vec![0xFFu8; 10];
        let c = cpu_rle_compress(&data);
        let d = cpu_rle_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_rle_short_runs() {
        // Runs of length 1 and 2 should be emitted literally.
        let data = vec![0x01, 0x02, 0x02, 0x03];
        let c = cpu_rle_compress(&data);
        let d = cpu_rle_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_rle_mixed_data() {
        let mut data = Vec::new();
        data.extend(vec![0x00; 100]);
        data.extend(vec![0x01; 2]);
        data.extend(vec![0xFF; 50]);
        data.extend(vec![0x42; 300]);
        data.push(0x99);
        let c = cpu_rle_compress(&data);
        let d = cpu_rle_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_rle_decompress_truncated() {
        // Truncated marker sequence.
        let bad = vec![0xFF, 0x42];
        assert!(cpu_rle_decompress(&bad).is_err());
    }

    // =====================================================================
    // Checksum tests
    // =====================================================================

    #[test]
    fn test_checksum_none() {
        assert_eq!(ChecksumValidator::compute(ChecksumAlgorithm::None, b"data"), 0);
    }

    #[test]
    fn test_checksum_crc32_deterministic() {
        let data = b"hello world";
        let a = ChecksumValidator::compute(ChecksumAlgorithm::Crc32, data);
        let b = ChecksumValidator::compute(ChecksumAlgorithm::Crc32, data);
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    #[test]
    fn test_checksum_xxhash64_deterministic() {
        let data = b"hello world";
        let a = ChecksumValidator::compute(ChecksumAlgorithm::XxHash64, data);
        let b = ChecksumValidator::compute(ChecksumAlgorithm::XxHash64, data);
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    #[test]
    fn test_checksum_crc32_differs_for_different_data() {
        let a = ChecksumValidator::compute(ChecksumAlgorithm::Crc32, b"aaa");
        let b = ChecksumValidator::compute(ChecksumAlgorithm::Crc32, b"bbb");
        assert_ne!(a, b);
    }

    #[test]
    fn test_checksum_xxhash64_differs_for_different_data() {
        let a = ChecksumValidator::compute(ChecksumAlgorithm::XxHash64, b"aaa");
        let b = ChecksumValidator::compute(ChecksumAlgorithm::XxHash64, b"bbb");
        assert_ne!(a, b);
    }

    #[test]
    fn test_checksum_validate_ok() {
        let data = b"tensor data";
        let cksum = ChecksumValidator::compute(ChecksumAlgorithm::Crc32, data);
        ChecksumValidator::validate(ChecksumAlgorithm::Crc32, data, cksum).unwrap();
    }

    #[test]
    fn test_checksum_validate_corrupted() {
        let data = b"tensor data";
        let cksum = ChecksumValidator::compute(ChecksumAlgorithm::Crc32, data);
        let result = ChecksumValidator::validate(ChecksumAlgorithm::Crc32, b"corrupted!X", cksum);
        assert!(result.is_err());
    }

    #[test]
    fn test_checksum_validate_none_always_ok() {
        ChecksumValidator::validate(ChecksumAlgorithm::None, b"x", 999).unwrap();
    }

    #[test]
    fn test_checksum_algo_roundtrip() {
        for tag in 0..=2u8 {
            let a = ChecksumAlgorithm::from_u8(tag).unwrap();
            assert_eq!(a as u8, tag);
        }
        assert!(ChecksumAlgorithm::from_u8(99).is_none());
    }

    // =====================================================================
    // TensorHeader tests
    // =====================================================================

    #[test]
    fn test_header_expected_raw_bytes() {
        let h = TensorHeader {
            dtype: DType::F32,
            shape: vec![2, 3],
            strides: vec![12, 4],
            byte_order: ByteOrder::LittleEndian,
            compression: CompressionCodec::None,
            checksum_algo: ChecksumAlgorithm::None,
            raw_data_len: 24,
            compressed_data_len: 24,
            data_checksum: 0,
        };
        assert_eq!(h.expected_raw_bytes(), 24); // 2*3*4
    }

    #[test]
    fn test_header_row_major_strides() {
        let s = TensorHeader::row_major_strides(&[2, 3, 4], DType::F32);
        assert_eq!(s, vec![48, 16, 4]);
    }

    #[test]
    fn test_header_row_major_strides_scalar() {
        let s = TensorHeader::row_major_strides(&[], DType::F32);
        assert!(s.is_empty());
    }

    #[test]
    fn test_header_row_major_strides_1d() {
        let s = TensorHeader::row_major_strides(&[10], DType::I8);
        assert_eq!(s, vec![1]);
    }

    #[test]
    fn test_header_encode_decode_roundtrip() {
        let h = TensorHeader {
            dtype: DType::BF16,
            shape: vec![4, 8],
            strides: vec![16, 2],
            byte_order: ByteOrder::BigEndian,
            compression: CompressionCodec::Zstd { level: 7 },
            checksum_algo: ChecksumAlgorithm::XxHash64,
            raw_data_len: 64,
            compressed_data_len: 50,
            data_checksum: 0xDEADBEEF,
        };
        let buf = h.encode();
        let (decoded, consumed) = TensorHeader::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded, h);
    }

    #[test]
    fn test_header_decode_too_short() {
        assert!(TensorHeader::decode(&[0, 1]).is_err());
    }

    #[test]
    fn test_header_decode_invalid_dtype() {
        let mut buf = vec![0u8; 64];
        buf[0] = 99; // invalid dtype
        assert!(matches!(TensorHeader::decode(&buf), Err(SerdeError::InvalidDType(99))));
    }

    #[test]
    fn test_header_decode_invalid_byte_order() {
        let mut buf = vec![0u8; 64];
        buf[0] = 0; // F32
        buf[1] = 99; // invalid byte order
        assert!(matches!(TensorHeader::decode(&buf), Err(SerdeError::InvalidByteOrder(99))));
    }

    #[test]
    fn test_header_decode_invalid_compression() {
        let mut buf = vec![0u8; 64];
        buf[0] = 0; // F32
        buf[1] = 0; // LE
        buf[2] = 99; // invalid compression
        assert!(matches!(TensorHeader::decode(&buf), Err(SerdeError::InvalidCompression(99))));
    }

    #[test]
    fn test_header_decode_invalid_checksum() {
        let mut buf = vec![0u8; 64];
        buf[0] = 0; // F32
        buf[1] = 0; // LE
        buf[2] = 0; // None compression
        buf[3] = 0;
        buf[4] = 99; // invalid checksum
        assert!(matches!(TensorHeader::decode(&buf), Err(SerdeError::InvalidChecksum(99))));
    }

    // =====================================================================
    // TensorWriter / TensorReader roundtrip tests
    // =====================================================================

    #[test]
    fn test_write_read_f32_uncompressed() {
        let data = sample_f32_data(12);
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::None)
            .with_checksum(ChecksumAlgorithm::Crc32);
        let (blob, wstats) = tw.write_to_vec(DType::F32, &[3, 4], &data).unwrap();
        assert_eq!(wstats.raw_bytes, data.len() as u64);

        let (hdr, raw, _rstats) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::F32);
        assert_eq!(hdr.shape, vec![3, 4]);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_f32_lz4() {
        let data = vec![0u8; 1024]; // highly compressible
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::Lz4)
            .with_checksum(ChecksumAlgorithm::Crc32);
        let (blob, _) = tw.write_to_vec(DType::F32, &[256], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::F32);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_f32_zstd() {
        let data = vec![0x42u8; 512];
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::Zstd { level: 3 })
            .with_checksum(ChecksumAlgorithm::XxHash64);
        let (blob, _) = tw.write_to_vec(DType::F32, &[128], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.compression, CompressionCodec::Zstd { level: 3 });
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_i8() {
        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::I8, &[10, 10], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::I8);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_u8() {
        let data = vec![255u8; 64];
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::U8, &[8, 8], &data).unwrap();
        let (_, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_i32() {
        let values: Vec<i32> = (-10..10).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::I32, &[20], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::I32);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_f16() {
        let data = vec![0u8; 32]; // 16 f16 elements
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::F16, &[4, 4], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::F16);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_bf16() {
        let data = vec![0xABu8; 20];
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::BF16, &[10], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::BF16);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_i16() {
        let values: Vec<i16> = (0..8).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::I16, &[8], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::I16);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_f64() {
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::F64, &[4], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::F64);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_i2() {
        let data = vec![0b10_01_00_11u8; 8]; // packed i2
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::I2, &[32], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.dtype, DType::I2);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_big_endian() {
        let data = sample_f32_data(4);
        let tw = TensorWriter::new().with_byte_order(ByteOrder::BigEndian);
        let (blob, _) = tw.write_to_vec(DType::F32, &[4], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.byte_order, ByteOrder::BigEndian);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_no_checksum() {
        let data = sample_f32_data(4);
        let tw = TensorWriter::new().with_checksum(ChecksumAlgorithm::None);
        let (blob, _) = tw.write_to_vec(DType::F32, &[4], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.checksum_algo, ChecksumAlgorithm::None);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_write_read_xxhash64_checksum() {
        let data = sample_f32_data(8);
        let tw = TensorWriter::new().with_checksum(ChecksumAlgorithm::XxHash64);
        let (blob, _) = tw.write_to_vec(DType::F32, &[8], &data).unwrap();
        let (_, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(raw, data);
    }

    // =====================================================================
    // Corrupted data detection
    // =====================================================================

    #[test]
    fn test_read_invalid_magic() {
        let bad = b"XXXX\x01\x00\x00\x00\x00\x00";
        let result = TensorReader::read_from_slice(bad);
        assert!(matches!(result, Err(SerdeError::InvalidMagic(_))));
    }

    #[test]
    fn test_read_unsupported_version() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&TENSOR_MAGIC);
        blob.extend_from_slice(&99u16.to_le_bytes());
        blob.extend_from_slice(&[0; 4]); // header len
        let result = TensorReader::read_from_slice(&blob);
        assert!(matches!(result, Err(SerdeError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_read_corrupted_data_checksum_fails() {
        let data = sample_f32_data(16);
        let tw = TensorWriter::new().with_checksum(ChecksumAlgorithm::Crc32);
        let (mut blob, _) = tw.write_to_vec(DType::F32, &[4, 4], &data).unwrap();
        // Corrupt the last byte of the data region.
        if let Some(last) = blob.last_mut() {
            *last ^= 0xFF;
        }
        let result = TensorReader::read_from_slice(&blob);
        assert!(matches!(result, Err(SerdeError::ChecksumMismatch { .. })));
    }

    // =====================================================================
    // MemoryMapReader tests
    // =====================================================================

    #[test]
    fn test_mmap_reader_basic() {
        let data = sample_f32_data(16);
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::F32, &[4, 4], &data).unwrap();
        let mmr = MemoryMapReader::from_bytes(blob.clone()).unwrap();
        assert_eq!(mmr.header().dtype, DType::F32);
        assert_eq!(mmr.header().shape, vec![4, 4]);
        assert_eq!(mmr.mapped_size(), blob.len());
        let raw = mmr.decompress().unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_mmap_reader_compressed() {
        let data = vec![0x00u8; 512];
        let tw = TensorWriter::new().with_compression(CompressionCodec::Lz4);
        let (blob, _) = tw.write_to_vec(DType::U8, &[512], &data).unwrap();
        let mmr = MemoryMapReader::from_bytes(blob).unwrap();
        let raw = mmr.decompress().unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_mmap_reader_invalid_magic() {
        let result = MemoryMapReader::from_bytes(b"XXXX1234567890".to_vec());
        assert!(matches!(result, Err(SerdeError::InvalidMagic(_))));
    }

    #[test]
    fn test_mmap_reader_too_short() {
        let result = MemoryMapReader::from_bytes(vec![0; 5]);
        assert!(matches!(result, Err(SerdeError::InvalidHeaderLength)));
    }

    #[test]
    fn test_mmap_compressed_data_slice() {
        let data = sample_f32_data(4);
        let tw = TensorWriter::new().with_compression(CompressionCodec::None);
        let (blob, _) = tw.write_to_vec(DType::F32, &[4], &data).unwrap();
        let mmr = MemoryMapReader::from_bytes(blob).unwrap();
        // With no compression, compressed_data should equal raw data.
        assert_eq!(mmr.compressed_data(), &data[..]);
    }

    // =====================================================================
    // StreamingReader tests
    // =====================================================================

    #[test]
    fn test_streaming_reader_basic() {
        let data = sample_f32_data(32);
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::None)
            .with_checksum(ChecksumAlgorithm::None);
        let (blob, _) = tw.write_to_vec(DType::F32, &[8, 4], &data).unwrap();

        let cursor = io::Cursor::new(blob);
        let mut sr = StreamingReader::new(cursor, 16).unwrap();
        assert_eq!(sr.header().dtype, DType::F32);

        let mut collected = Vec::new();
        while let Some(chunk) = sr.next_chunk().unwrap() {
            assert!(chunk.len() <= 16);
            collected.extend_from_slice(&chunk);
        }
        assert_eq!(collected, data);
        assert_eq!(sr.remaining(), 0);
    }

    #[test]
    fn test_streaming_reader_read_all() {
        let data = sample_f32_data(20);
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::None)
            .with_checksum(ChecksumAlgorithm::None);
        let (blob, _) = tw.write_to_vec(DType::F32, &[20], &data).unwrap();

        let cursor = io::Cursor::new(blob);
        let mut sr = StreamingReader::new(cursor, 32).unwrap();
        let all = sr.read_all().unwrap();
        assert_eq!(all, data);
    }

    #[test]
    fn test_streaming_reader_single_byte_chunks() {
        let data = vec![1u8, 2, 3, 4];
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::None)
            .with_checksum(ChecksumAlgorithm::None);
        let (blob, _) = tw.write_to_vec(DType::U8, &[4], &data).unwrap();

        let cursor = io::Cursor::new(blob);
        let mut sr = StreamingReader::new(cursor, 1).unwrap();
        let mut collected = Vec::new();
        let mut chunk_count = 0;
        while let Some(chunk) = sr.next_chunk().unwrap() {
            assert_eq!(chunk.len(), 1);
            collected.extend_from_slice(&chunk);
            chunk_count += 1;
        }
        assert_eq!(chunk_count, 4);
        assert_eq!(collected, data);
    }

    #[test]
    fn test_streaming_reader_returns_none_after_done() {
        let data = vec![0u8; 4];
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::None)
            .with_checksum(ChecksumAlgorithm::None);
        let (blob, _) = tw.write_to_vec(DType::U8, &[4], &data).unwrap();

        let cursor = io::Cursor::new(blob);
        let mut sr = StreamingReader::new(cursor, 1024).unwrap();
        assert!(sr.next_chunk().unwrap().is_some());
        assert!(sr.next_chunk().unwrap().is_none());
        assert!(sr.next_chunk().unwrap().is_none());
    }

    // =====================================================================
    // TensorBundle tests
    // =====================================================================

    #[test]
    fn test_bundle_empty() {
        let bundle = TensorBundle::new();
        assert!(bundle.is_empty());
        assert_eq!(bundle.len(), 0);
    }

    #[test]
    fn test_bundle_single_tensor() {
        let data = sample_f32_data(4);
        let mut bundle = TensorBundle::new();
        bundle.add("weight", DType::F32, vec![4], data.clone());
        assert_eq!(bundle.len(), 1);

        let (blob, stats) = bundle.write_to_vec().unwrap();
        assert!(stats.raw_bytes > 0);

        let (index, raw_blob) = TensorBundle::read_from_slice(&blob).unwrap();
        assert_eq!(index.len(), 1);
        assert!(index.contains("weight"));

        let (hdr, raw, _) = TensorBundle::read_tensor("weight", &index, &raw_blob).unwrap();
        assert_eq!(hdr.dtype, DType::F32);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_bundle_multiple_tensors() {
        let w1 = sample_f32_data(8);
        let w2 = vec![1u8; 16];
        let w3 = vec![0u8; 32];

        let mut bundle = TensorBundle::new();
        bundle.add("layer.0.weight", DType::F32, vec![2, 4], w1.clone());
        bundle.add("layer.0.bias", DType::I8, vec![16], w2.clone());
        bundle.add("layer.1.weight", DType::U8, vec![4, 8], w3.clone());
        assert_eq!(bundle.len(), 3);

        let (blob, _) = bundle.write_to_vec().unwrap();
        let (index, raw_blob) = TensorBundle::read_from_slice(&blob).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(index.names(), &["layer.0.weight", "layer.0.bias", "layer.1.weight"]);

        let (h1, d1, _) = TensorBundle::read_tensor("layer.0.weight", &index, &raw_blob).unwrap();
        assert_eq!(h1.dtype, DType::F32);
        assert_eq!(h1.shape, vec![2, 4]);
        assert_eq!(d1, w1);

        let (_, d2, _) = TensorBundle::read_tensor("layer.0.bias", &index, &raw_blob).unwrap();
        assert_eq!(d2, w2);

        let (_, d3, _) = TensorBundle::read_tensor("layer.1.weight", &index, &raw_blob).unwrap();
        assert_eq!(d3, w3);
    }

    #[test]
    fn test_bundle_compressed() {
        let data = vec![0u8; 1024];
        let mut bundle = TensorBundle::new().with_compression(CompressionCodec::Lz4);
        bundle.add("t", DType::U8, vec![1024], data.clone());

        let (blob, stats) = bundle.write_to_vec().unwrap();
        assert!(stats.compressed_bytes < stats.raw_bytes, "LZ4 should compress zeros");

        let (index, raw_blob) = TensorBundle::read_from_slice(&blob).unwrap();
        let (_, raw, _) = TensorBundle::read_tensor("t", &index, &raw_blob).unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_bundle_not_found() {
        let mut bundle = TensorBundle::new();
        bundle.add("a", DType::F32, vec![1], vec![0; 4]);
        let (blob, _) = bundle.write_to_vec().unwrap();
        let (index, raw_blob) = TensorBundle::read_from_slice(&blob).unwrap();
        let result = TensorBundle::read_tensor("nonexistent", &index, &raw_blob);
        assert!(matches!(result, Err(SerdeError::TensorNotFound(_))));
    }

    #[test]
    fn test_bundle_invalid_magic() {
        let result = TensorBundle::read_from_slice(b"XXXX0123456789");
        assert!(matches!(result, Err(SerdeError::InvalidMagic(_))));
    }

    #[test]
    fn test_bundle_xxhash_checksum() {
        let data = sample_f32_data(8);
        let mut bundle = TensorBundle::new().with_checksum(ChecksumAlgorithm::XxHash64);
        bundle.add("w", DType::F32, vec![8], data.clone());
        let (blob, _) = bundle.write_to_vec().unwrap();
        let (index, raw_blob) = TensorBundle::read_from_slice(&blob).unwrap();
        let (_, raw, _) = TensorBundle::read_tensor("w", &index, &raw_blob).unwrap();
        assert_eq!(raw, data);
    }

    // =====================================================================
    // SerdeStats tests
    // =====================================================================

    #[test]
    fn test_serde_stats_compression_ratio() {
        let s = SerdeStats {
            elapsed: Duration::from_millis(10),
            raw_bytes: 1000,
            compressed_bytes: 500,
        };
        assert!((s.compression_ratio() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serde_stats_compression_ratio_zero_compressed() {
        let s =
            SerdeStats { elapsed: Duration::from_millis(1), raw_bytes: 100, compressed_bytes: 0 };
        assert!((s.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serde_stats_throughput() {
        let s = SerdeStats {
            elapsed: Duration::from_secs(1),
            raw_bytes: 1_000_000,
            compressed_bytes: 500_000,
        };
        assert!((s.throughput_bytes_per_sec() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_serde_stats_display() {
        let s = SerdeStats {
            elapsed: Duration::from_millis(50),
            raw_bytes: 1024,
            compressed_bytes: 512,
        };
        let text = s.to_string();
        assert!(text.contains("raw=1024"));
        assert!(text.contains("compressed=512"));
        assert!(text.contains("ratio=2.00x"));
    }

    // =====================================================================
    // Large tensor tests
    // =====================================================================

    #[test]
    fn test_large_tensor_roundtrip() {
        let n = 10_000;
        let data = sample_f32_data(n);
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::Lz4)
            .with_checksum(ChecksumAlgorithm::Crc32);
        let (blob, _) = tw.write_to_vec(DType::F32, &[100, 100], &data).unwrap();
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.shape, vec![100, 100]);
        assert_eq!(raw, data);
    }

    #[test]
    fn test_large_tensor_zeros_compresses_well() {
        let data = vec![0u8; 100_000];
        let tw = TensorWriter::new().with_compression(CompressionCodec::Lz4);
        let (_, stats) = tw.write_to_vec(DType::U8, &[100_000], &data).unwrap();
        assert!(
            stats.compression_ratio() > 10.0,
            "100k zeros should compress >10x, got {:.1}x",
            stats.compression_ratio()
        );
    }

    // =====================================================================
    // SerdeError display tests
    // =====================================================================

    #[test]
    fn test_error_display() {
        let err = SerdeError::InvalidMagic(*b"XXXX");
        assert!(err.to_string().contains("invalid magic"));

        let err = SerdeError::ChecksumMismatch { expected: 1, actual: 2 };
        assert!(err.to_string().contains("checksum mismatch"));

        let err = SerdeError::TensorNotFound("w".into());
        assert!(err.to_string().contains("tensor not found"));

        let err = SerdeError::BufferTooSmall { needed: 10, got: 5 };
        assert!(err.to_string().contains("buffer too small"));
    }

    // =====================================================================
    // Property-like tests (serialize ∘ deserialize = identity)
    // =====================================================================

    #[test]
    fn test_identity_all_dtypes_no_compression() {
        let dtypes = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I8,
            DType::I2,
            DType::U8,
            DType::I32,
            DType::I16,
            DType::F64,
        ];
        for dtype in dtypes {
            let size = 16 * dtype.byte_size();
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let tw = TensorWriter::new()
                .with_compression(CompressionCodec::None)
                .with_checksum(ChecksumAlgorithm::Crc32);
            let (blob, _) = tw.write_to_vec(dtype, &[16], &data).unwrap();
            let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
            assert_eq!(hdr.dtype, dtype, "dtype roundtrip failed for {dtype}");
            assert_eq!(raw, data, "data roundtrip failed for {dtype}");
        }
    }

    #[test]
    fn test_identity_all_compressions() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::Zstd { level: 3 },
            CompressionCodec::Zstd { level: 15 },
        ];
        let data = sample_f32_data(32);
        for codec in codecs {
            let tw = TensorWriter::new().with_compression(codec);
            let (blob, _) = tw.write_to_vec(DType::F32, &[8, 4], &data).unwrap();
            let (_, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
            assert_eq!(raw, data, "data roundtrip failed for codec {codec:?}");
        }
    }

    #[test]
    fn test_identity_all_checksums() {
        let algos =
            [ChecksumAlgorithm::None, ChecksumAlgorithm::Crc32, ChecksumAlgorithm::XxHash64];
        let data = sample_f32_data(8);
        for algo in algos {
            let tw = TensorWriter::new().with_checksum(algo);
            let (blob, _) = tw.write_to_vec(DType::F32, &[8], &data).unwrap();
            let (_, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
            assert_eq!(raw, data, "data roundtrip failed for {algo:?}");
        }
    }

    #[test]
    fn test_identity_various_shapes() {
        let shapes: Vec<Vec<usize>> =
            vec![vec![1], vec![100], vec![2, 3], vec![4, 5, 6], vec![2, 2, 2, 2]];
        for shape in &shapes {
            let elems: usize = shape.iter().product();
            let data: Vec<u8> = (0..elems).map(|i| (i % 256) as u8).collect();
            let tw = TensorWriter::new();
            let (blob, _) = tw.write_to_vec(DType::U8, shape, &data).unwrap();
            let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
            assert_eq!(hdr.shape, *shape);
            assert_eq!(raw, data);
        }
    }

    #[test]
    fn test_identity_f32_values_preserved() {
        let values = vec![
            0.0f32,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            1.23456789,
            -9.87654321,
        ];
        let data = f32_to_bytes(&values);
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::F32, &[values.len()], &data).unwrap();
        let (_, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        let recovered = bytes_to_f32(&raw);
        for (a, b) in values.iter().zip(recovered.iter()) {
            assert!(a.to_bits() == b.to_bits(), "bit-exact mismatch: {a} vs {b}");
        }
    }

    // =====================================================================
    // BundleIndex tests
    // =====================================================================

    #[test]
    fn test_bundle_index_operations() {
        let mut idx = BundleIndex::new();
        assert!(idx.is_empty());
        idx.insert("a".into(), 0, 10);
        idx.insert("b".into(), 10, 20);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
        assert!(idx.contains("a"));
        assert!(idx.contains("b"));
        assert!(!idx.contains("c"));
        assert_eq!(idx.get("a").unwrap().offset, 0);
        assert_eq!(idx.get("b").unwrap().length, 20);
        assert_eq!(idx.names(), &["a", "b"]);
    }

    // =====================================================================
    // Writer builder tests
    // =====================================================================

    #[test]
    fn test_writer_builder() {
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::Lz4)
            .with_checksum(ChecksumAlgorithm::XxHash64)
            .with_byte_order(ByteOrder::BigEndian);
        assert_eq!(tw.compression, CompressionCodec::Lz4);
        assert_eq!(tw.checksum_algo, ChecksumAlgorithm::XxHash64);
        assert_eq!(tw.byte_order, ByteOrder::BigEndian);
    }

    #[test]
    fn test_bundle_builder() {
        let bundle = TensorBundle::new()
            .with_compression(CompressionCodec::Zstd { level: 5 })
            .with_checksum(ChecksumAlgorithm::XxHash64);
        assert_eq!(bundle.compression, CompressionCodec::Zstd { level: 5 });
        assert_eq!(bundle.checksum_algo, ChecksumAlgorithm::XxHash64);
    }

    // =====================================================================
    // Edge case: empty tensor
    // =====================================================================

    #[test]
    fn test_write_read_empty_tensor() {
        let tw = TensorWriter::new();
        let (blob, stats) = tw.write_to_vec(DType::F32, &[0], &[]).unwrap();
        assert_eq!(stats.raw_bytes, 0);
        let (hdr, raw, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.shape, vec![0]);
        assert!(raw.is_empty());
    }

    #[test]
    fn test_bundle_with_empty_tensor() {
        let mut bundle = TensorBundle::new();
        bundle.add("empty", DType::F32, vec![0], vec![]);
        bundle.add("notempty", DType::U8, vec![2], vec![1, 2]);
        let (blob, _) = bundle.write_to_vec().unwrap();
        let (index, raw_blob) = TensorBundle::read_from_slice(&blob).unwrap();
        let (_, d1, _) = TensorBundle::read_tensor("empty", &index, &raw_blob).unwrap();
        assert!(d1.is_empty());
        let (_, d2, _) = TensorBundle::read_tensor("notempty", &index, &raw_blob).unwrap();
        assert_eq!(d2, vec![1, 2]);
    }

    // =====================================================================
    // Streaming + compression
    // =====================================================================

    #[test]
    fn test_streaming_compressed_read_all() {
        let data = vec![0xAAu8; 256];
        let tw = TensorWriter::new()
            .with_compression(CompressionCodec::Lz4)
            .with_checksum(ChecksumAlgorithm::None);
        let (blob, _) = tw.write_to_vec(DType::U8, &[256], &data).unwrap();
        let cursor = io::Cursor::new(blob);
        let mut sr = StreamingReader::new(cursor, 8).unwrap();
        let compressed = sr.read_all().unwrap();
        let raw = sr.header().compression.decompress(&compressed).unwrap();
        assert_eq!(raw, data);
    }

    // =====================================================================
    // Multi-dimensional strides
    // =====================================================================

    #[test]
    fn test_strides_written_correctly() {
        let data = sample_f32_data(24);
        let tw = TensorWriter::new();
        let (blob, _) = tw.write_to_vec(DType::F32, &[2, 3, 4], &data).unwrap();
        let (hdr, _, _) = TensorReader::read_from_slice(&blob).unwrap();
        assert_eq!(hdr.strides, vec![48, 16, 4]); // C-order for f32
    }

    // =====================================================================
    // Writer via Write trait (non-vec)
    // =====================================================================

    #[test]
    fn test_write_to_write_trait() {
        let data = sample_f32_data(4);
        let tw = TensorWriter::new();
        let mut buf: Vec<u8> = Vec::new();
        let stats = tw.write(&mut buf, DType::F32, &[4], &data).unwrap();
        assert!(stats.raw_bytes > 0);
        let (_, raw, _) = TensorReader::read_from_slice(&buf).unwrap();
        assert_eq!(raw, data);
    }

    // =====================================================================
    // Bundle with Read trait
    // =====================================================================

    #[test]
    fn test_bundle_read_from_reader() {
        let data = sample_f32_data(4);
        let mut bundle = TensorBundle::new();
        bundle.add("t", DType::F32, vec![4], data.clone());
        let (blob, _) = bundle.write_to_vec().unwrap();

        let mut cursor = io::Cursor::new(blob);
        let (index, raw_blob) = TensorBundle::read(&mut cursor).unwrap();
        let (_, raw, _) = TensorBundle::read_tensor("t", &index, &raw_blob).unwrap();
        assert_eq!(raw, data);
    }
}
