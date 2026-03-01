//! Module stub - implementation pending merge from feature branch
//! GGUF file writer for exporting quantized models.
//!
//! Provides a complete pipeline for writing GGUF v2/v3 files:
//! header → metadata → tensor info → tensor data, with alignment
//! padding, validation, and round-trip verification.

use std::collections::HashMap;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic bytes: ASCII "GGUF".
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Default data alignment in bytes (must be power of 2).
pub const DEFAULT_ALIGNMENT: u32 = 32;

// ---------------------------------------------------------------------------
// Metadata value‐type tags (§ GGUF spec)
// ---------------------------------------------------------------------------

/// GGUF metadata value‐type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// ---------------------------------------------------------------------------
// Tensor data‐type tags
// ---------------------------------------------------------------------------

/// Subset of GGUF tensor data types relevant for `BitNet` export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFTensorDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    I2S = 36,
}

impl GGUFTensorDType {
    /// Bytes per element (approximate; quantised types use block size).
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 | Self::Q8_0 | Self::I2S => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Endianness
// ---------------------------------------------------------------------------

/// Byte order for the GGUF file (little-endian is canonical).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Endianness {
    #[default]
    Little,
    Big,
}

// ---------------------------------------------------------------------------
// GGUFWriterConfig
// ---------------------------------------------------------------------------

/// Configuration for the GGUF writer.
#[derive(Debug, Clone)]
pub struct GGUFWriterConfig {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Data alignment in bytes (must be power of 2).
    pub alignment: u32,
    /// Enable zlib-style compression placeholder (reserved, not yet used).
    pub compression: bool,
    /// Byte order.
    pub endianness: Endianness,
}

impl Default for GGUFWriterConfig {
    fn default() -> Self {
        Self {
            version: 3,
            alignment: DEFAULT_ALIGNMENT,
            compression: false,
            endianness: Endianness::Little,
        }
    }
}

impl GGUFWriterConfig {
    /// Create config for GGUF v2.
    pub fn v2() -> Self {
        Self { version: 2, ..Default::default() }
    }

    /// Create config for GGUF v3.
    pub fn v3() -> Self {
        Self::default()
    }

    /// Validate configuration.
    pub const fn validate(&self) -> Result<(), GGUFWriteError> {
        if self.version != 2 && self.version != 3 {
            return Err(GGUFWriteError::InvalidVersion(self.version));
        }
        if !self.alignment.is_power_of_two() || self.alignment == 0 {
            return Err(GGUFWriteError::InvalidAlignment(self.alignment));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during GGUF writing or validation.
#[derive(Debug)]
pub enum GGUFWriteError {
    Io(io::Error),
    InvalidVersion(u32),
    InvalidAlignment(u32),
    ValidationFailed(String),
}

impl From<io::Error> for GGUFWriteError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for GGUFWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::InvalidVersion(v) => write!(f, "invalid GGUF version: {v}"),
            Self::InvalidAlignment(a) => write!(f, "invalid alignment: {a}"),
            Self::ValidationFailed(msg) => write!(f, "validation failed: {msg}"),
        }
    }
}

impl std::error::Error for GGUFWriteError {}

// ---------------------------------------------------------------------------
// MetadataValue / MetadataEntry
// ---------------------------------------------------------------------------

/// A typed metadata value matching the GGUF spec.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GGUFValueType, Vec<Self>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Returns the GGUF type tag for this value.
    pub const fn value_type(&self) -> GGUFValueType {
        match self {
            Self::Uint8(_) => GGUFValueType::Uint8,
            Self::Int8(_) => GGUFValueType::Int8,
            Self::Uint16(_) => GGUFValueType::Uint16,
            Self::Int16(_) => GGUFValueType::Int16,
            Self::Uint32(_) => GGUFValueType::Uint32,
            Self::Int32(_) => GGUFValueType::Int32,
            Self::Float32(_) => GGUFValueType::Float32,
            Self::Bool(_) => GGUFValueType::Bool,
            Self::String(_) => GGUFValueType::String,
            Self::Array(_, _) => GGUFValueType::Array,
            Self::Uint64(_) => GGUFValueType::Uint64,
            Self::Int64(_) => GGUFValueType::Int64,
            Self::Float64(_) => GGUFValueType::Float64,
        }
    }
}

/// A key-value metadata entry with its GGUF type tag.
#[derive(Debug, Clone)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

impl MetadataEntry {
    pub fn new(key: impl Into<String>, value: MetadataValue) -> Self {
        Self { key: key.into(), value }
    }
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

/// Descriptor for a tensor to be written into the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GGUFTensorDType,
    pub data: Vec<u8>,
}

impl TensorInfo {
    pub fn new(
        name: impl Into<String>,
        shape: Vec<u64>,
        dtype: GGUFTensorDType,
        data: Vec<u8>,
    ) -> Self {
        Self { name: name.into(), shape, dtype, data }
    }
}

// ---------------------------------------------------------------------------
// AlignmentPadder
// ---------------------------------------------------------------------------

/// Adds zero-padding to maintain the configured alignment.
pub struct AlignmentPadder {
    alignment: u32,
}

impl AlignmentPadder {
    pub const fn new(alignment: u32) -> Self {
        Self { alignment }
    }

    /// Number of padding bytes needed to align `offset`.
    pub fn padding_for(&self, offset: u64) -> u64 {
        let a = u64::from(self.alignment);
        let rem = offset % a;
        if rem == 0 { 0 } else { a - rem }
    }

    /// Write padding zeros to reach the next aligned offset.
    pub fn write_padding<W: Write + Seek>(&self, w: &mut W) -> io::Result<()> {
        let pos = w.stream_position()?;
        let pad = self.padding_for(pos);
        if pad > 0 {
            #[allow(clippy::cast_possible_truncation)]
            let zeros = vec![0u8; pad as usize];
            w.write_all(&zeros)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Low-level write helpers (always little-endian)
// ---------------------------------------------------------------------------

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn write_i8<W: Write>(w: &mut W, v: i8) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_i16<W: Write>(w: &mut W, v: i16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_i64<W: Write>(w: &mut W, v: i64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Write a length-prefixed UTF-8 string (u64 length + bytes).
fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    write_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())
}

// ---------------------------------------------------------------------------
// Low-level read helpers (for validation / round-trip)
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> io::Result<i8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(i8::from_le_bytes(buf))
}

fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> io::Result<String> {
    #[allow(clippy::cast_possible_truncation)]
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_bytes<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// GGUFHeaderWriter
// ---------------------------------------------------------------------------

/// Writes the fixed GGUF header: magic, version, tensor count, metadata count.
pub struct GGUFHeaderWriter;

impl GGUFHeaderWriter {
    /// Write the GGUF header. Returns the number of bytes written.
    pub fn write<W: Write>(
        w: &mut W,
        version: u32,
        tensor_count: u64,
        metadata_count: u64,
    ) -> Result<usize, GGUFWriteError> {
        w.write_all(&GGUF_MAGIC)?;
        write_u32(w, version)?;
        write_u64(w, tensor_count)?;
        write_u64(w, metadata_count)?;
        // 4 (magic) + 4 (version) + 8 (tensor_count) + 8 (metadata_count) = 24
        Ok(24)
    }
}

// ---------------------------------------------------------------------------
// MetadataWriter
// ---------------------------------------------------------------------------

/// Serialises metadata entries in GGUF binary format.
pub struct MetadataWriter;

impl MetadataWriter {
    /// Write a single metadata entry.
    pub fn write_entry<W: Write>(w: &mut W, entry: &MetadataEntry) -> Result<(), GGUFWriteError> {
        write_gguf_string(w, &entry.key)?;
        Self::write_value(w, &entry.value)?;
        Ok(())
    }

    /// Write all metadata entries.
    pub fn write_all<W: Write>(w: &mut W, entries: &[MetadataEntry]) -> Result<(), GGUFWriteError> {
        for entry in entries {
            Self::write_entry(w, entry)?;
        }
        Ok(())
    }

    fn write_value<W: Write>(w: &mut W, v: &MetadataValue) -> Result<(), GGUFWriteError> {
        write_u32(w, v.value_type() as u32)?;
        match v {
            MetadataValue::Uint8(x) => write_u8(w, *x)?,
            MetadataValue::Int8(x) => write_i8(w, *x)?,
            MetadataValue::Uint16(x) => write_u16(w, *x)?,
            MetadataValue::Int16(x) => write_i16(w, *x)?,
            MetadataValue::Uint32(x) => write_u32(w, *x)?,
            MetadataValue::Int32(x) => write_i32(w, *x)?,
            MetadataValue::Float32(x) => write_f32(w, *x)?,
            MetadataValue::Bool(x) => write_u8(w, u8::from(*x))?,
            MetadataValue::String(x) => write_gguf_string(w, x)?,
            MetadataValue::Array(elem_type, elems) => {
                write_u32(w, *elem_type as u32)?;
                write_u64(w, elems.len() as u64)?;
                for elem in elems {
                    Self::write_value_body(w, elem)?;
                }
            }
            MetadataValue::Uint64(x) => write_u64(w, *x)?,
            MetadataValue::Int64(x) => write_i64(w, *x)?,
            MetadataValue::Float64(x) => write_f64(w, *x)?,
        }
        Ok(())
    }

    /// Write just the value body (no type tag) — used inside arrays.
    fn write_value_body<W: Write>(w: &mut W, v: &MetadataValue) -> Result<(), GGUFWriteError> {
        match v {
            MetadataValue::Uint8(x) => write_u8(w, *x)?,
            MetadataValue::Int8(x) => write_i8(w, *x)?,
            MetadataValue::Uint16(x) => write_u16(w, *x)?,
            MetadataValue::Int16(x) => write_i16(w, *x)?,
            MetadataValue::Uint32(x) => write_u32(w, *x)?,
            MetadataValue::Int32(x) => write_i32(w, *x)?,
            MetadataValue::Float32(x) => write_f32(w, *x)?,
            MetadataValue::Bool(x) => write_u8(w, u8::from(*x))?,
            MetadataValue::String(x) => write_gguf_string(w, x)?,
            MetadataValue::Array(elem_type, elems) => {
                write_u32(w, *elem_type as u32)?;
                write_u64(w, elems.len() as u64)?;
                for elem in elems {
                    Self::write_value_body(w, elem)?;
                }
            }
            MetadataValue::Uint64(x) => write_u64(w, *x)?,
            MetadataValue::Int64(x) => write_i64(w, *x)?,
            MetadataValue::Float64(x) => write_f64(w, *x)?,
        }
        Ok(())
    }

    /// Read back a single metadata entry for verification.
    pub fn read_entry<R: Read>(r: &mut R) -> Result<MetadataEntry, GGUFWriteError> {
        let key = read_gguf_string(r)?;
        let value = Self::read_value(r)?;
        Ok(MetadataEntry { key, value })
    }

    fn read_value<R: Read>(r: &mut R) -> Result<MetadataValue, GGUFWriteError> {
        let type_tag = read_u32(r)?;
        Self::read_value_body(r, type_tag)
    }

    fn read_value_body<R: Read>(r: &mut R, type_tag: u32) -> Result<MetadataValue, GGUFWriteError> {
        match type_tag {
            0 => Ok(MetadataValue::Uint8(read_u8(r)?)),
            1 => Ok(MetadataValue::Int8(read_i8(r)?)),
            2 => Ok(MetadataValue::Uint16(read_u16(r)?)),
            3 => Ok(MetadataValue::Int16(read_i16(r)?)),
            4 => Ok(MetadataValue::Uint32(read_u32(r)?)),
            5 => Ok(MetadataValue::Int32(read_i32(r)?)),
            6 => Ok(MetadataValue::Float32(read_f32(r)?)),
            7 => {
                let b = read_u8(r)?;
                Ok(MetadataValue::Bool(b != 0))
            }
            8 => Ok(MetadataValue::String(read_gguf_string(r)?)),
            9 => {
                let elem_type_tag = read_u32(r)?;
                #[allow(clippy::cast_possible_truncation)]
                let count = read_u64(r)? as usize;
                let mut elems = Vec::with_capacity(count);
                for _ in 0..count {
                    elems.push(Self::read_value_body(r, elem_type_tag)?);
                }
                let elem_type = tag_to_value_type(elem_type_tag)?;
                Ok(MetadataValue::Array(elem_type, elems))
            }
            10 => Ok(MetadataValue::Uint64(read_u64(r)?)),
            11 => Ok(MetadataValue::Int64(read_i64(r)?)),
            12 => Ok(MetadataValue::Float64(read_f64(r)?)),
            other => {
                Err(GGUFWriteError::ValidationFailed(format!("unknown metadata type tag: {other}")))
            }
        }
    }
}

fn tag_to_value_type(tag: u32) -> Result<GGUFValueType, GGUFWriteError> {
    match tag {
        0 => Ok(GGUFValueType::Uint8),
        1 => Ok(GGUFValueType::Int8),
        2 => Ok(GGUFValueType::Uint16),
        3 => Ok(GGUFValueType::Int16),
        4 => Ok(GGUFValueType::Uint32),
        5 => Ok(GGUFValueType::Int32),
        6 => Ok(GGUFValueType::Float32),
        7 => Ok(GGUFValueType::Bool),
        8 => Ok(GGUFValueType::String),
        9 => Ok(GGUFValueType::Array),
        10 => Ok(GGUFValueType::Uint64),
        11 => Ok(GGUFValueType::Int64),
        12 => Ok(GGUFValueType::Float64),
        other => Err(GGUFWriteError::ValidationFailed(format!("unknown value type tag: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// TensorDataWriter
// ---------------------------------------------------------------------------

/// Writes tensor descriptors and raw data with alignment padding.
pub struct TensorDataWriter;

impl TensorDataWriter {
    /// Write the tensor info table (name, ndims, shape, dtype, offset).
    /// Returns the offsets assigned to each tensor's data (relative to data start).
    pub fn write_info_table<W: Write>(
        w: &mut W,
        tensors: &[TensorInfo],
        alignment: u32,
    ) -> Result<Vec<u64>, GGUFWriteError> {
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut current_offset: u64 = 0;

        for tensor in tensors {
            write_gguf_string(w, &tensor.name)?;
            #[allow(clippy::cast_possible_truncation)]
            let ndims = tensor.shape.len() as u32;
            write_u32(w, ndims)?;
            for &dim in &tensor.shape {
                write_u64(w, dim)?;
            }
            write_u32(w, tensor.dtype as u32)?;
            write_u64(w, current_offset)?;

            offsets.push(current_offset);

            // Advance offset: data size + alignment padding
            let data_len = tensor.data.len() as u64;
            current_offset += data_len;
            let a = u64::from(alignment);
            let rem = current_offset % a;
            if rem != 0 {
                current_offset += a - rem;
            }
        }

        Ok(offsets)
    }

    /// Write raw tensor data with alignment padding between tensors.
    pub fn write_data<W: Write + Seek>(
        w: &mut W,
        tensors: &[TensorInfo],
        padder: &AlignmentPadder,
    ) -> Result<(), GGUFWriteError> {
        for (i, tensor) in tensors.iter().enumerate() {
            w.write_all(&tensor.data)?;
            // Pad after each tensor except possibly the last
            if i + 1 < tensors.len() {
                padder.write_padding(w)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GGUFValidator
// ---------------------------------------------------------------------------

/// Validates a written GGUF file for structural correctness.
pub struct GGUFValidator;

impl GGUFValidator {
    /// Validate the GGUF file in the given buffer.
    pub fn validate(data: &[u8]) -> Result<ValidationReport, GGUFWriteError> {
        let mut r = Cursor::new(data);
        let mut report = ValidationReport::default();

        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(GGUFWriteError::ValidationFailed(format!(
                "bad magic: {magic:?}, expected {GGUF_MAGIC:?}"
            )));
        }
        report.magic_ok = true;

        // Version
        let version = read_u32(&mut r)?;
        if version != 2 && version != 3 {
            return Err(GGUFWriteError::ValidationFailed(format!(
                "unsupported version: {version}"
            )));
        }
        report.version = version;
        report.version_ok = true;

        // Counts
        let tensor_count = read_u64(&mut r)?;
        let metadata_count = read_u64(&mut r)?;
        report.tensor_count = tensor_count;
        report.metadata_count = metadata_count;

        // Read metadata entries to verify they parse
        for _ in 0..metadata_count {
            MetadataWriter::read_entry(&mut r)?;
        }
        report.metadata_ok = true;

        // Read tensor info entries
        let mut tensor_data_sizes: Vec<(String, u64, u64)> = Vec::new();
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut r)?;
            let ndims = read_u32(&mut r)?;
            for _ in 0..ndims {
                let _ = read_u64(&mut r)?;
            }
            let _dtype = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensor_data_sizes.push((name, offset, 0));
        }
        report.tensor_info_ok = true;

        // Verify offsets are non-decreasing
        for window in tensor_data_sizes.windows(2) {
            if window[1].1 < window[0].1 {
                return Err(GGUFWriteError::ValidationFailed(format!(
                    "tensor offsets not monotonic: '{}' offset {} < '{}' offset {}",
                    window[1].0, window[1].1, window[0].0, window[0].1
                )));
            }
        }
        report.offsets_ok = true;

        Ok(report)
    }
}

/// Report from GGUF validation.
#[derive(Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct ValidationReport {
    pub magic_ok: bool,
    pub version_ok: bool,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub metadata_ok: bool,
    pub tensor_info_ok: bool,
    pub offsets_ok: bool,
}

impl ValidationReport {
    pub const fn all_ok(&self) -> bool {
        self.magic_ok
            && self.version_ok
            && self.metadata_ok
            && self.tensor_info_ok
            && self.offsets_ok
    }
}

// ---------------------------------------------------------------------------
// GGUFRoundTripper
// ---------------------------------------------------------------------------

/// Writes a GGUF file then reads it back to verify correctness.
pub struct GGUFRoundTripper;

/// Parsed-back GGUF content for round-trip comparison.
#[derive(Debug)]
pub struct RoundTripResult {
    pub version: u32,
    pub metadata: Vec<MetadataEntry>,
    pub tensor_names: Vec<String>,
    pub tensor_data: HashMap<String, Vec<u8>>,
}

impl GGUFRoundTripper {
    /// Write then read back, returning parsed content.
    pub fn round_trip(
        config: &GGUFWriterConfig,
        metadata: &[MetadataEntry],
        tensors: &[TensorInfo],
    ) -> Result<RoundTripResult, GGUFWriteError> {
        let buf = GGUFWriter::write_to_vec(config, metadata, tensors)?;
        Self::read_back(&buf)
    }

    /// Read a GGUF buffer back into structured data.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn read_back(data: &[u8]) -> Result<RoundTripResult, GGUFWriteError> {
        let mut r = Cursor::new(data);

        // Header
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(GGUFWriteError::ValidationFailed("bad magic".into()));
        }
        let version = read_u32(&mut r)?;
        let tensor_count = read_u64(&mut r)? as usize;
        let metadata_count = read_u64(&mut r)? as usize;

        // Metadata
        let mut meta = Vec::with_capacity(metadata_count);
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..metadata_count {
            let entry = MetadataWriter::read_entry(&mut r)?;
            if entry.key == "general.alignment"
                && let MetadataValue::Uint32(a) = &entry.value
            {
                alignment = *a;
            }
            meta.push(entry);
        }

        // Tensor info
        let mut names = Vec::with_capacity(tensor_count);
        let mut offsets = Vec::with_capacity(tensor_count);
        let mut shapes: Vec<Vec<u64>> = Vec::with_capacity(tensor_count);
        let mut dtypes = Vec::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut r)?;
            let ndims = read_u32(&mut r)?;
            let mut shape = Vec::with_capacity(ndims as usize);
            for _ in 0..ndims {
                shape.push(read_u64(&mut r)?);
            }
            let dtype = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            names.push(name);
            offsets.push(offset);
            shapes.push(shape);
            dtypes.push(dtype);
        }

        // Compute data lengths from offsets
        let mut data_lens: Vec<u64> = vec![0; tensor_count];
        for i in 0..tensor_count {
            if i + 1 < tensor_count {
                data_lens[i] = offsets[i + 1] - offsets[i];
            }
        }

        // Align to data start
        let pos = r.position();
        let a = u64::from(alignment);
        let rem = pos % a;
        if rem != 0 {
            r.seek(SeekFrom::Current((a - rem) as i64))?;
        }

        let data_start = r.position();

        // Read tensor data
        let mut tensor_names = Vec::with_capacity(tensor_count);
        let mut tensor_data = HashMap::new();
        for i in 0..tensor_count {
            tensor_names.push(names[i].clone());
            r.seek(SeekFrom::Start(data_start + offsets[i]))?;

            // Determine actual data size
            let elem_size: u64 = match dtypes[i] {
                0 => 4, // F32
                1 => 2, // F16
                _ => 1,
            };
            let total_elems: u64 =
                if shapes[i].is_empty() { 0 } else { shapes[i].iter().product() };
            let actual_len = total_elems * elem_size;

            let read_len = if i + 1 < tensor_count {
                actual_len.min(data_lens[i])
            } else {
                actual_len.min((data.len() as u64).saturating_sub(r.position()))
            };

            let bytes = read_bytes(&mut r, read_len as usize)?;
            tensor_data.insert(names[i].clone(), bytes);
        }

        Ok(RoundTripResult { version, metadata: meta, tensor_names, tensor_data })
    }
}

// ---------------------------------------------------------------------------
// GGUFWriter — top-level orchestrator
// ---------------------------------------------------------------------------

/// Orchestrates the full GGUF write pipeline:
/// header → metadata → tensor info → alignment padding → tensor data → validate.
pub struct GGUFWriter;

impl GGUFWriter {
    /// Write a complete GGUF file to a byte vector.
    pub fn write_to_vec(
        config: &GGUFWriterConfig,
        metadata: &[MetadataEntry],
        tensors: &[TensorInfo],
    ) -> Result<Vec<u8>, GGUFWriteError> {
        config.validate()?;

        let mut buf = Cursor::new(Vec::new());
        Self::write_to(&mut buf, config, metadata, tensors)?;
        Ok(buf.into_inner())
    }

    /// Write a complete GGUF file to any `Write + Seek` destination.
    pub fn write_to<W: Write + Seek>(
        w: &mut W,
        config: &GGUFWriterConfig,
        metadata: &[MetadataEntry],
        tensors: &[TensorInfo],
    ) -> Result<(), GGUFWriteError> {
        config.validate()?;

        let padder = AlignmentPadder::new(config.alignment);

        // Inject alignment metadata for v3 if non-default
        let mut all_metadata: Vec<MetadataEntry> = metadata.to_vec();
        if config.version == 3
            && config.alignment != DEFAULT_ALIGNMENT
            && !metadata.iter().any(|e| e.key == "general.alignment")
        {
            all_metadata.push(MetadataEntry::new(
                "general.alignment",
                MetadataValue::Uint32(config.alignment),
            ));
        }

        // 1. Header
        GGUFHeaderWriter::write(
            w,
            config.version,
            tensors.len() as u64,
            all_metadata.len() as u64,
        )?;

        // 2. Metadata
        MetadataWriter::write_all(w, &all_metadata)?;

        // 3. Tensor info table
        TensorDataWriter::write_info_table(w, tensors, config.alignment)?;

        // 4. Alignment padding before data section
        padder.write_padding(w)?;

        // 5. Tensor data
        TensorDataWriter::write_data(w, tensors, &padder)?;

        Ok(())
    }

    /// Write and then validate.
    pub fn write_and_validate(
        config: &GGUFWriterConfig,
        metadata: &[MetadataEntry],
        tensors: &[TensorInfo],
    ) -> Result<(Vec<u8>, ValidationReport), GGUFWriteError> {
        let buf = Self::write_to_vec(config, metadata, tensors)?;
        let report = GGUFValidator::validate(&buf)?;
        if !report.all_ok() {
            return Err(GGUFWriteError::ValidationFailed("validation report has failures".into()));
        }
        Ok((buf, report))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn default_config() -> GGUFWriterConfig {
        GGUFWriterConfig::default()
    }

    fn make_f32_tensor(name: &str, values: &[f32]) -> TensorInfo {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorInfo::new(name, vec![values.len() as u64], GGUFTensorDType::F32, data)
    }

    fn make_f16_tensor(name: &str, n: usize) -> TensorInfo {
        let data = vec![0u8; n * 2];
        TensorInfo::new(name, vec![n as u64], GGUFTensorDType::F16, data)
    }

    // -----------------------------------------------------------------------
    // Header tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_header_magic_bytes() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 3, 0, 0).unwrap();
        assert_eq!(&buf[0..4], b"GGUF");
    }

    #[test]
    fn test_header_magic_byte_values() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 3, 0, 0).unwrap();
        assert_eq!(buf[0], 0x47);
        assert_eq!(buf[1], 0x47);
        assert_eq!(buf[2], 0x55);
        assert_eq!(buf[3], 0x46);
    }

    #[test]
    fn test_header_version_2() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 2, 0, 0).unwrap();
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, 2);
    }

    #[test]
    fn test_header_version_3() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 3, 0, 0).unwrap();
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, 3);
    }

    #[test]
    fn test_header_tensor_count() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 3, 42, 0).unwrap();
        let tc = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        assert_eq!(tc, 42);
    }

    #[test]
    fn test_header_metadata_count() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 3, 0, 7).unwrap();
        let mc = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        assert_eq!(mc, 7);
    }

    #[test]
    fn test_header_size_is_24() {
        let mut buf = Vec::new();
        let n = GGUFHeaderWriter::write(&mut buf, 3, 0, 0).unwrap();
        assert_eq!(n, 24);
        assert_eq!(buf.len(), 24);
    }

    #[test]
    fn test_header_large_tensor_count() {
        let mut buf = Vec::new();
        GGUFHeaderWriter::write(&mut buf, 3, u64::MAX, 0).unwrap();
        let tc = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        assert_eq!(tc, u64::MAX);
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = GGUFWriterConfig::default();
        assert_eq!(cfg.version, 3);
        assert_eq!(cfg.alignment, 32);
        assert!(!cfg.compression);
        assert_eq!(cfg.endianness, Endianness::Little);
    }

    #[test]
    fn test_config_v2() {
        let cfg = GGUFWriterConfig::v2();
        assert_eq!(cfg.version, 2);
        assert_eq!(cfg.alignment, 32);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_invalid_version() {
        let cfg = GGUFWriterConfig { version: 4, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_alignment_zero() {
        let cfg = GGUFWriterConfig { alignment: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_alignment_not_power_of_two() {
        let cfg = GGUFWriterConfig { alignment: 13, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_alignment_powers_of_two() {
        for a in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let cfg = GGUFWriterConfig { alignment: a, ..Default::default() };
            assert!(cfg.validate().is_ok(), "alignment {a} should be valid");
        }
    }

    // -----------------------------------------------------------------------
    // Metadata scalar type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_uint8() {
        let entry = MetadataEntry::new("k", MetadataValue::Uint8(255));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.key, "k");
        assert_eq!(parsed.value, MetadataValue::Uint8(255));
    }

    #[test]
    fn test_metadata_int8() {
        let entry = MetadataEntry::new("k", MetadataValue::Int8(-128));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Int8(-128));
    }

    #[test]
    fn test_metadata_uint16() {
        let entry = MetadataEntry::new("k", MetadataValue::Uint16(65535));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Uint16(65535));
    }

    #[test]
    fn test_metadata_int16() {
        let entry = MetadataEntry::new("k", MetadataValue::Int16(-32768));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Int16(-32768));
    }

    #[test]
    fn test_metadata_uint32() {
        let entry = MetadataEntry::new("k", MetadataValue::Uint32(0xDEAD_BEEF));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Uint32(0xDEAD_BEEF));
    }

    #[test]
    fn test_metadata_int32() {
        let entry = MetadataEntry::new("k", MetadataValue::Int32(i32::MIN));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Int32(i32::MIN));
    }

    #[test]
    fn test_metadata_float32() {
        let entry = MetadataEntry::new("k", MetadataValue::Float32(3.14));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        if let MetadataValue::Float32(v) = parsed.value {
            assert!((v - 3.14).abs() < 1e-6);
        } else {
            panic!("wrong type");
        }
    }

    #[test]
    fn test_metadata_bool_true() {
        let entry = MetadataEntry::new("k", MetadataValue::Bool(true));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Bool(true));
    }

    #[test]
    fn test_metadata_bool_false() {
        let entry = MetadataEntry::new("k", MetadataValue::Bool(false));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Bool(false));
    }

    #[test]
    fn test_metadata_uint64() {
        let entry = MetadataEntry::new("k", MetadataValue::Uint64(u64::MAX));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Uint64(u64::MAX));
    }

    #[test]
    fn test_metadata_int64() {
        let entry = MetadataEntry::new("k", MetadataValue::Int64(i64::MIN));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Int64(i64::MIN));
    }

    #[test]
    fn test_metadata_float64() {
        let entry = MetadataEntry::new("k", MetadataValue::Float64(std::f64::consts::PI));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        if let MetadataValue::Float64(v) = parsed.value {
            assert!((v - std::f64::consts::PI).abs() < 1e-15);
        } else {
            panic!("wrong type");
        }
    }

    // -----------------------------------------------------------------------
    // Metadata string tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_string_basic() {
        let entry = MetadataEntry::new("model.name", MetadataValue::String("bitnet".into()));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.key, "model.name");
        assert_eq!(parsed.value, MetadataValue::String("bitnet".into()));
    }

    #[test]
    fn test_metadata_string_empty() {
        let entry = MetadataEntry::new("k", MetadataValue::String(String::new()));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::String(String::new()));
    }

    #[test]
    fn test_metadata_string_utf8() {
        let entry = MetadataEntry::new("k", MetadataValue::String("日本語テスト🦀".into()));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::String("日本語テスト🦀".into()));
    }

    #[test]
    fn test_metadata_string_length_prefix() {
        let s = "hello";
        let mut buf = Vec::new();
        write_gguf_string(&mut buf, s).unwrap();
        let len = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        assert_eq!(len, 5);
        assert_eq!(&buf[8..13], b"hello");
    }

    #[test]
    fn test_metadata_string_long() {
        let long_str = "a".repeat(10_000);
        let entry = MetadataEntry::new("k", MetadataValue::String(long_str.clone()));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::String(long_str));
    }

    // -----------------------------------------------------------------------
    // Metadata array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_array_uint32() {
        let arr = MetadataValue::Array(
            GGUFValueType::Uint32,
            vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2), MetadataValue::Uint32(3)],
        );
        let entry = MetadataEntry::new("dims", arr.clone());
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, arr);
    }

    #[test]
    fn test_metadata_array_string() {
        let arr = MetadataValue::Array(
            GGUFValueType::String,
            vec![MetadataValue::String("alpha".into()), MetadataValue::String("beta".into())],
        );
        let entry = MetadataEntry::new("tokens", arr.clone());
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, arr);
    }

    #[test]
    fn test_metadata_array_empty() {
        let arr = MetadataValue::Array(GGUFValueType::Uint32, vec![]);
        let entry = MetadataEntry::new("empty", arr.clone());
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, arr);
    }

    #[test]
    fn test_metadata_array_float32() {
        let arr = MetadataValue::Array(
            GGUFValueType::Float32,
            vec![MetadataValue::Float32(1.0), MetadataValue::Float32(-0.5)],
        );
        let entry = MetadataEntry::new("scales", arr.clone());
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, arr);
    }

    #[test]
    fn test_metadata_array_int64() {
        let arr = MetadataValue::Array(
            GGUFValueType::Int64,
            vec![MetadataValue::Int64(i64::MAX), MetadataValue::Int64(i64::MIN)],
        );
        let entry = MetadataEntry::new("big", arr.clone());
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, arr);
    }

    #[test]
    fn test_metadata_array_encoding_layout() {
        // Array encoding: type_tag(u32) + elem_type(u32) + count(u64) + elems
        let arr = MetadataValue::Array(
            GGUFValueType::Uint8,
            vec![MetadataValue::Uint8(10), MetadataValue::Uint8(20)],
        );
        let entry = MetadataEntry::new("a", arr);
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();

        let mut r = Cursor::new(&buf);
        let _key = read_gguf_string(&mut r).unwrap(); // "a"
        let type_tag = read_u32(&mut r).unwrap();
        assert_eq!(type_tag, GGUFValueType::Array as u32);
        let elem_type = read_u32(&mut r).unwrap();
        assert_eq!(elem_type, GGUFValueType::Uint8 as u32);
        let count = read_u64(&mut r).unwrap();
        assert_eq!(count, 2);
        assert_eq!(read_u8(&mut r).unwrap(), 10);
        assert_eq!(read_u8(&mut r).unwrap(), 20);
    }

    // -----------------------------------------------------------------------
    // Metadata value type tag tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_type_discriminants() {
        assert_eq!(GGUFValueType::Uint8 as u32, 0);
        assert_eq!(GGUFValueType::Int8 as u32, 1);
        assert_eq!(GGUFValueType::Uint16 as u32, 2);
        assert_eq!(GGUFValueType::Int16 as u32, 3);
        assert_eq!(GGUFValueType::Uint32 as u32, 4);
        assert_eq!(GGUFValueType::Int32 as u32, 5);
        assert_eq!(GGUFValueType::Float32 as u32, 6);
        assert_eq!(GGUFValueType::Bool as u32, 7);
        assert_eq!(GGUFValueType::String as u32, 8);
        assert_eq!(GGUFValueType::Array as u32, 9);
        assert_eq!(GGUFValueType::Uint64 as u32, 10);
        assert_eq!(GGUFValueType::Int64 as u32, 11);
        assert_eq!(GGUFValueType::Float64 as u32, 12);
    }

    #[test]
    fn test_metadata_value_type_returns_correct_tag() {
        assert_eq!(MetadataValue::Uint8(0).value_type(), GGUFValueType::Uint8);
        assert_eq!(MetadataValue::Int8(0).value_type(), GGUFValueType::Int8);
        assert_eq!(MetadataValue::Uint16(0).value_type(), GGUFValueType::Uint16);
        assert_eq!(MetadataValue::Int16(0).value_type(), GGUFValueType::Int16);
        assert_eq!(MetadataValue::Uint32(0).value_type(), GGUFValueType::Uint32);
        assert_eq!(MetadataValue::Int32(0).value_type(), GGUFValueType::Int32);
        assert_eq!(MetadataValue::Float32(0.0).value_type(), GGUFValueType::Float32);
        assert_eq!(MetadataValue::Bool(false).value_type(), GGUFValueType::Bool);
        assert_eq!(MetadataValue::String(String::new()).value_type(), GGUFValueType::String);
        assert_eq!(
            MetadataValue::Array(GGUFValueType::Uint8, vec![]).value_type(),
            GGUFValueType::Array
        );
        assert_eq!(MetadataValue::Uint64(0).value_type(), GGUFValueType::Uint64);
        assert_eq!(MetadataValue::Int64(0).value_type(), GGUFValueType::Int64);
        assert_eq!(MetadataValue::Float64(0.0).value_type(), GGUFValueType::Float64);
    }

    // -----------------------------------------------------------------------
    // AlignmentPadder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_padder_no_padding_needed() {
        let padder = AlignmentPadder::new(32);
        assert_eq!(padder.padding_for(0), 0);
        assert_eq!(padder.padding_for(32), 0);
        assert_eq!(padder.padding_for(64), 0);
    }

    #[test]
    fn test_padder_padding_needed() {
        let padder = AlignmentPadder::new(32);
        assert_eq!(padder.padding_for(1), 31);
        assert_eq!(padder.padding_for(24), 8);
        assert_eq!(padder.padding_for(33), 31);
    }

    #[test]
    fn test_padder_alignment_1() {
        let padder = AlignmentPadder::new(1);
        assert_eq!(padder.padding_for(0), 0);
        assert_eq!(padder.padding_for(1), 0);
        assert_eq!(padder.padding_for(999), 0);
    }

    #[test]
    fn test_padder_writes_zeros() {
        let padder = AlignmentPadder::new(16);
        let mut buf = Cursor::new(Vec::new());
        buf.write_all(&[1, 2, 3]).unwrap(); // 3 bytes in
        padder.write_padding(&mut buf).unwrap();
        let data = buf.into_inner();
        assert_eq!(data.len(), 16);
        // Padding bytes are all zero
        assert!(data[3..16].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_padder_no_write_when_aligned() {
        let padder = AlignmentPadder::new(4);
        let mut buf = Cursor::new(Vec::new());
        buf.write_all(&[1, 2, 3, 4]).unwrap();
        padder.write_padding(&mut buf).unwrap();
        assert_eq!(buf.into_inner().len(), 4); // no extra bytes
    }

    // -----------------------------------------------------------------------
    // TensorInfo tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_info_construction() {
        let t = make_f32_tensor("weight", &[1.0, 2.0, 3.0]);
        assert_eq!(t.name, "weight");
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.dtype, GGUFTensorDType::F32);
        assert_eq!(t.data.len(), 12);
    }

    #[test]
    fn test_tensor_dtype_element_size() {
        assert_eq!(GGUFTensorDType::F32.element_size(), 4);
        assert_eq!(GGUFTensorDType::F16.element_size(), 2);
        assert_eq!(GGUFTensorDType::Q4_0.element_size(), 1);
        assert_eq!(GGUFTensorDType::Q8_0.element_size(), 1);
        assert_eq!(GGUFTensorDType::I2S.element_size(), 1);
    }

    #[test]
    fn test_tensor_dtype_discriminants() {
        assert_eq!(GGUFTensorDType::F32 as u32, 0);
        assert_eq!(GGUFTensorDType::F16 as u32, 1);
        assert_eq!(GGUFTensorDType::Q4_0 as u32, 2);
        assert_eq!(GGUFTensorDType::Q8_0 as u32, 8);
        assert_eq!(GGUFTensorDType::I2S as u32, 36);
    }

    // -----------------------------------------------------------------------
    // TensorDataWriter info table tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_info_table_single() {
        let t = make_f32_tensor("w", &[1.0, 2.0]);
        let mut buf = Vec::new();
        let offsets =
            TensorDataWriter::write_info_table(&mut buf, &[t], DEFAULT_ALIGNMENT).unwrap();
        assert_eq!(offsets, vec![0]);
    }

    #[test]
    fn test_tensor_info_table_offsets_aligned() {
        let t1 = make_f32_tensor("a", &[1.0]); // 4 bytes
        let t2 = make_f32_tensor("b", &[2.0]); // 4 bytes
        let mut buf = Vec::new();
        let offsets = TensorDataWriter::write_info_table(&mut buf, &[t1, t2], 32).unwrap();
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 32); // 4 bytes data + 28 padding = 32
    }

    #[test]
    fn test_tensor_info_table_multiple_offsets() {
        let t1 = TensorInfo::new("a", vec![64], GGUFTensorDType::F32, vec![0u8; 256]); // 256 bytes
        let t2 = TensorInfo::new("b", vec![8], GGUFTensorDType::F32, vec![0u8; 32]); // 32 bytes
        let t3 = TensorInfo::new("c", vec![4], GGUFTensorDType::F32, vec![0u8; 16]); // 16 bytes
        let mut buf = Vec::new();
        let offsets = TensorDataWriter::write_info_table(&mut buf, &[t1, t2, t3], 32).unwrap();
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 256); // 256 is already aligned to 32
        assert_eq!(offsets[2], 288); // 256 + 32 = 288
    }

    // -----------------------------------------------------------------------
    // Full writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_empty_model() {
        let buf = GGUFWriter::write_to_vec(&default_config(), &[], &[]).unwrap();
        assert!(buf.len() >= 24);
        assert_eq!(&buf[0..4], b"GGUF");
    }

    #[test]
    fn test_write_empty_model_validate() {
        let (_, report) = GGUFWriter::write_and_validate(&default_config(), &[], &[]).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.tensor_count, 0);
        assert_eq!(report.metadata_count, 0);
    }

    #[test]
    fn test_write_metadata_only() {
        let meta = vec![
            MetadataEntry::new("arch", MetadataValue::String("bitnet".into())),
            MetadataEntry::new("layers", MetadataValue::Uint32(24)),
        ];
        let (buf, report) = GGUFWriter::write_and_validate(&default_config(), &meta, &[]).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.metadata_count, 2);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_single_tensor() {
        let t = make_f32_tensor("w.0", &[1.0, 2.0, 3.0, 4.0]);
        let (_, report) = GGUFWriter::write_and_validate(&default_config(), &[], &[t]).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.tensor_count, 1);
    }

    #[test]
    fn test_write_multiple_tensors() {
        let tensors = vec![
            make_f32_tensor("layer.0.weight", &[1.0; 16]),
            make_f32_tensor("layer.0.bias", &[0.0; 4]),
            make_f16_tensor("layer.1.weight", 32),
        ];
        let (_, report) = GGUFWriter::write_and_validate(&default_config(), &[], &tensors).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.tensor_count, 3);
    }

    #[test]
    fn test_write_v2() {
        let cfg = GGUFWriterConfig::v2();
        let (_, report) =
            GGUFWriter::write_and_validate(&cfg, &[], &[make_f32_tensor("w", &[1.0])]).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.version, 2);
    }

    #[test]
    fn test_write_v3() {
        let (_, report) =
            GGUFWriter::write_and_validate(&default_config(), &[], &[make_f32_tensor("w", &[1.0])])
                .unwrap();
        assert!(report.all_ok());
        assert_eq!(report.version, 3);
    }

    #[test]
    fn test_write_custom_alignment() {
        let cfg = GGUFWriterConfig { alignment: 64, ..Default::default() };
        let meta = vec![MetadataEntry::new("x", MetadataValue::Uint32(1))];
        let (buf, report) =
            GGUFWriter::write_and_validate(&cfg, &meta, &[make_f32_tensor("w", &[1.0; 8])])
                .unwrap();
        assert!(report.all_ok());
        // Verify alignment metadata was injected
        let rt = GGUFRoundTripper::read_back(&buf).unwrap();
        let has_alignment = rt
            .metadata
            .iter()
            .any(|e| e.key == "general.alignment" && e.value == MetadataValue::Uint32(64));
        assert!(has_alignment);
    }

    #[test]
    fn test_write_no_alignment_injection_for_default() {
        let cfg = default_config(); // alignment = 32 = default
        let (buf, _) = GGUFWriter::write_and_validate(&cfg, &[], &[]).unwrap();
        let rt = GGUFRoundTripper::read_back(&buf).unwrap();
        let has_alignment = rt.metadata.iter().any(|e| e.key == "general.alignment");
        assert!(!has_alignment);
    }

    #[test]
    fn test_write_invalid_config_rejected() {
        let cfg = GGUFWriterConfig { version: 99, ..Default::default() };
        assert!(GGUFWriter::write_to_vec(&cfg, &[], &[]).is_err());
    }

    // -----------------------------------------------------------------------
    // Tensor alignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_data_starts_at_aligned_offset() {
        let t = make_f32_tensor("w", &[1.0, 2.0]);
        let buf = GGUFWriter::write_to_vec(&default_config(), &[], &[t]).unwrap();
        // After header (24) + tensor info, there should be alignment padding
        // Find the tensor data (the f32 values 1.0, 2.0)
        let needle_1 = 1.0f32.to_le_bytes();
        let pos = buf.windows(4).position(|w| w == needle_1).expect("tensor data not found");
        assert_eq!(pos % 32, 0, "tensor data at offset {pos} not aligned to 32");
    }

    #[test]
    fn test_tensor_data_alignment_64() {
        let cfg = GGUFWriterConfig { alignment: 64, ..Default::default() };
        let t = make_f32_tensor("w", &[42.0]);
        let buf = GGUFWriter::write_to_vec(&cfg, &[], &[t]).unwrap();
        let needle = 42.0f32.to_le_bytes();
        let pos = buf.windows(4).position(|w| w == needle).expect("data not found");
        assert_eq!(pos % 64, 0, "tensor data at offset {pos} not aligned to 64");
    }

    // -----------------------------------------------------------------------
    // Validator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validator_good_file() {
        let buf = GGUFWriter::write_to_vec(
            &default_config(),
            &[MetadataEntry::new("k", MetadataValue::Uint32(1))],
            &[make_f32_tensor("w", &[1.0])],
        )
        .unwrap();
        let report = GGUFValidator::validate(&buf).unwrap();
        assert!(report.all_ok());
    }

    #[test]
    fn test_validator_bad_magic() {
        let mut buf = GGUFWriter::write_to_vec(&default_config(), &[], &[]).unwrap();
        buf[0] = 0x00; // corrupt magic
        assert!(GGUFValidator::validate(&buf).is_err());
    }

    #[test]
    fn test_validator_bad_version() {
        let mut buf = GGUFWriter::write_to_vec(&default_config(), &[], &[]).unwrap();
        buf[4..8].copy_from_slice(&99u32.to_le_bytes()); // corrupt version
        assert!(GGUFValidator::validate(&buf).is_err());
    }

    #[test]
    fn test_validator_reports_counts() {
        let meta = vec![
            MetadataEntry::new("a", MetadataValue::Uint32(1)),
            MetadataEntry::new("b", MetadataValue::Uint32(2)),
        ];
        let tensors = vec![make_f32_tensor("t1", &[1.0]), make_f32_tensor("t2", &[2.0])];
        let buf = GGUFWriter::write_to_vec(&default_config(), &meta, &tensors).unwrap();
        let report = GGUFValidator::validate(&buf).unwrap();
        assert_eq!(report.metadata_count, 2);
        assert_eq!(report.tensor_count, 2);
    }

    // -----------------------------------------------------------------------
    // Round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_trip_empty() {
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[]).unwrap();
        assert_eq!(rt.version, 3);
        assert!(rt.metadata.is_empty());
        assert!(rt.tensor_names.is_empty());
    }

    #[test]
    fn test_round_trip_metadata() {
        let meta = vec![
            MetadataEntry::new("arch", MetadataValue::String("bitnet".into())),
            MetadataEntry::new("layers", MetadataValue::Uint32(24)),
            MetadataEntry::new("hidden", MetadataValue::Float32(2048.0)),
        ];
        let rt = GGUFRoundTripper::round_trip(&default_config(), &meta, &[]).unwrap();
        assert_eq!(rt.metadata.len(), 3);
        assert_eq!(rt.metadata[0].key, "arch");
        assert_eq!(rt.metadata[0].value, MetadataValue::String("bitnet".into()));
        assert_eq!(rt.metadata[1].value, MetadataValue::Uint32(24));
    }

    #[test]
    fn test_round_trip_single_tensor() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = make_f32_tensor("weight", &values);
        let expected_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t]).unwrap();
        assert_eq!(rt.tensor_names, vec!["weight"]);
        assert_eq!(rt.tensor_data["weight"], expected_data);
    }

    #[test]
    fn test_round_trip_multiple_tensors() {
        let t1 = make_f32_tensor("a", &[1.0, 2.0]);
        let t2 = make_f32_tensor("b", &[3.0, 4.0, 5.0]);
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t1, t2]).unwrap();
        assert_eq!(rt.tensor_names, vec!["a", "b"]);
        assert_eq!(rt.tensor_data["a"].len(), 8);
        assert_eq!(rt.tensor_data["b"].len(), 12);
    }

    #[test]
    fn test_round_trip_mixed_dtypes() {
        let t_f32 = make_f32_tensor("f32_t", &[1.0]);
        let t_f16 = make_f16_tensor("f16_t", 4);
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t_f32, t_f16]).unwrap();
        assert_eq!(rt.tensor_names.len(), 2);
    }

    #[test]
    fn test_round_trip_v2() {
        let cfg = GGUFWriterConfig::v2();
        let rt = GGUFRoundTripper::round_trip(
            &cfg,
            &[MetadataEntry::new("k", MetadataValue::Uint32(42))],
            &[make_f32_tensor("w", &[1.0])],
        )
        .unwrap();
        assert_eq!(rt.version, 2);
        assert_eq!(rt.metadata[0].value, MetadataValue::Uint32(42));
    }

    #[test]
    fn test_round_trip_preserves_tensor_data_bytes() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        let t = TensorInfo::new("raw", vec![8], GGUFTensorDType::Q8_0, data.clone());
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t]).unwrap();
        assert_eq!(rt.tensor_data["raw"], data);
    }

    #[test]
    fn test_round_trip_metadata_and_tensors() {
        let meta = vec![
            MetadataEntry::new("general.architecture", MetadataValue::String("bitnet".into())),
            MetadataEntry::new("general.name", MetadataValue::String("test-model".into())),
            MetadataEntry::new(
                "tokenizer.ggml.tokens",
                MetadataValue::Array(
                    GGUFValueType::String,
                    vec![
                        MetadataValue::String("<pad>".into()),
                        MetadataValue::String("<unk>".into()),
                    ],
                ),
            ),
        ];
        let tensors = vec![
            make_f32_tensor("token_embd.weight", &[0.1; 32]),
            make_f32_tensor("blk.0.attn_q.weight", &[0.2; 16]),
        ];
        let rt = GGUFRoundTripper::round_trip(&default_config(), &meta, &tensors).unwrap();
        assert_eq!(rt.metadata.len(), 3);
        assert_eq!(rt.tensor_names.len(), 2);
        assert_eq!(rt.tensor_names[0], "token_embd.weight");
    }

    // -----------------------------------------------------------------------
    // Edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_very_long_tensor_name() {
        let name = "layer.".to_string() + &"x".repeat(1000) + ".weight";
        let t = make_f32_tensor(&name, &[1.0]);
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t]).unwrap();
        assert_eq!(rt.tensor_names[0], name);
    }

    #[test]
    fn test_empty_tensor_name() {
        let t = make_f32_tensor("", &[1.0]);
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t]).unwrap();
        assert_eq!(rt.tensor_names[0], "");
    }

    #[test]
    fn test_max_uint32_metadata() {
        let entry = MetadataEntry::new("max", MetadataValue::Uint32(u32::MAX));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, MetadataValue::Uint32(u32::MAX));
    }

    #[test]
    fn test_zero_values_metadata() {
        for val in [
            MetadataValue::Uint8(0),
            MetadataValue::Int8(0),
            MetadataValue::Uint16(0),
            MetadataValue::Int16(0),
            MetadataValue::Uint32(0),
            MetadataValue::Int32(0),
            MetadataValue::Float32(0.0),
            MetadataValue::Uint64(0),
            MetadataValue::Int64(0),
            MetadataValue::Float64(0.0),
        ] {
            let entry = MetadataEntry::new("z", val.clone());
            let mut buf = Vec::new();
            MetadataWriter::write_entry(&mut buf, &entry).unwrap();
            let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
            assert_eq!(parsed.value, val);
        }
    }

    #[test]
    fn test_min_values_signed() {
        let vals = vec![
            MetadataValue::Int8(i8::MIN),
            MetadataValue::Int16(i16::MIN),
            MetadataValue::Int32(i32::MIN),
            MetadataValue::Int64(i64::MIN),
        ];
        for val in vals {
            let entry = MetadataEntry::new("min", val.clone());
            let mut buf = Vec::new();
            MetadataWriter::write_entry(&mut buf, &entry).unwrap();
            let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
            assert_eq!(parsed.value, val);
        }
    }

    #[test]
    fn test_max_values_unsigned() {
        let vals = vec![
            MetadataValue::Uint8(u8::MAX),
            MetadataValue::Uint16(u16::MAX),
            MetadataValue::Uint32(u32::MAX),
            MetadataValue::Uint64(u64::MAX),
        ];
        for val in vals {
            let entry = MetadataEntry::new("max", val.clone());
            let mut buf = Vec::new();
            MetadataWriter::write_entry(&mut buf, &entry).unwrap();
            let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
            assert_eq!(parsed.value, val);
        }
    }

    #[test]
    fn test_float_special_values() {
        for v in [f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX] {
            let entry = MetadataEntry::new("f", MetadataValue::Float32(v));
            let mut buf = Vec::new();
            MetadataWriter::write_entry(&mut buf, &entry).unwrap();
            let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
            assert_eq!(parsed.value, MetadataValue::Float32(v));
        }
    }

    #[test]
    fn test_float32_nan() {
        let entry = MetadataEntry::new("f", MetadataValue::Float32(f32::NAN));
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        if let MetadataValue::Float32(v) = parsed.value {
            assert!(v.is_nan());
        } else {
            panic!("wrong type");
        }
    }

    #[test]
    fn test_float64_special_values() {
        for v in [f64::INFINITY, f64::NEG_INFINITY, f64::MIN, f64::MAX] {
            let entry = MetadataEntry::new("d", MetadataValue::Float64(v));
            let mut buf = Vec::new();
            MetadataWriter::write_entry(&mut buf, &entry).unwrap();
            let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
            assert_eq!(parsed.value, MetadataValue::Float64(v));
        }
    }

    #[test]
    fn test_multiple_metadata_entries_sequential() {
        let entries = vec![
            MetadataEntry::new("a", MetadataValue::Uint32(1)),
            MetadataEntry::new("b", MetadataValue::String("hello".into())),
            MetadataEntry::new("c", MetadataValue::Bool(true)),
            MetadataEntry::new("d", MetadataValue::Float64(2.718)),
        ];
        let mut buf = Vec::new();
        MetadataWriter::write_all(&mut buf, &entries).unwrap();
        let mut cursor = Cursor::new(&buf);
        for expected in &entries {
            let parsed = MetadataWriter::read_entry(&mut cursor).unwrap();
            assert_eq!(parsed.key, expected.key);
        }
    }

    #[test]
    fn test_tag_to_value_type_all() {
        for tag in 0..=12u32 {
            assert!(tag_to_value_type(tag).is_ok());
        }
    }

    #[test]
    fn test_tag_to_value_type_invalid() {
        assert!(tag_to_value_type(13).is_err());
        assert!(tag_to_value_type(255).is_err());
    }

    #[test]
    fn test_endianness_default_is_little() {
        assert_eq!(Endianness::default(), Endianness::Little);
    }

    #[test]
    fn test_error_display() {
        let e = GGUFWriteError::InvalidVersion(99);
        assert!(format!("{e}").contains("99"));
        let e = GGUFWriteError::InvalidAlignment(7);
        assert!(format!("{e}").contains("7"));
        let e = GGUFWriteError::ValidationFailed("oops".into());
        assert!(format!("{e}").contains("oops"));
    }

    #[test]
    fn test_write_to_generic_writer() {
        let mut buf = Cursor::new(Vec::new());
        GGUFWriter::write_to(&mut buf, &default_config(), &[], &[]).unwrap();
        let data = buf.into_inner();
        assert_eq!(&data[0..4], b"GGUF");
    }

    #[test]
    fn test_write_and_validate_returns_both() {
        let (buf, report) = GGUFWriter::write_and_validate(
            &default_config(),
            &[MetadataEntry::new("k", MetadataValue::Uint32(1))],
            &[make_f32_tensor("w", &[1.0])],
        )
        .unwrap();
        assert!(!buf.is_empty());
        assert!(report.all_ok());
    }

    #[test]
    fn test_validation_report_default_all_false() {
        let report = ValidationReport::default();
        assert!(!report.all_ok());
    }

    #[test]
    fn test_many_tensors() {
        let tensors: Vec<TensorInfo> =
            (0..50).map(|i| make_f32_tensor(&format!("t.{i}"), &[i as f32; 4])).collect();
        let (_, report) = GGUFWriter::write_and_validate(&default_config(), &[], &tensors).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.tensor_count, 50);
    }

    #[test]
    fn test_many_metadata_entries() {
        let meta: Vec<MetadataEntry> = (0..100)
            .map(|i| MetadataEntry::new(format!("key.{i}"), MetadataValue::Uint32(i)))
            .collect();
        let (_, report) = GGUFWriter::write_and_validate(&default_config(), &meta, &[]).unwrap();
        assert!(report.all_ok());
        assert_eq!(report.metadata_count, 100);
    }

    #[test]
    fn test_nested_array_metadata() {
        let inner = MetadataValue::Array(
            GGUFValueType::Uint32,
            vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2)],
        );
        let outer = MetadataValue::Array(GGUFValueType::Array, vec![inner.clone()]);
        let entry = MetadataEntry::new("nested", outer.clone());
        let mut buf = Vec::new();
        MetadataWriter::write_entry(&mut buf, &entry).unwrap();
        let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed.value, outer);
    }

    #[test]
    fn test_i2s_tensor_type() {
        let data = vec![0xAA; 64];
        let t = TensorInfo::new("quant", vec![256], GGUFTensorDType::I2S, data.clone());
        let (_, report) = GGUFWriter::write_and_validate(&default_config(), &[], &[t]).unwrap();
        assert!(report.all_ok());
    }

    #[test]
    fn test_round_trip_large_tensor() {
        let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let t = make_f32_tensor("big", &values);
        let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t]).unwrap();
        assert_eq!(rt.tensor_data["big"].len(), 4096);
    }

    #[test]
    fn test_alignment_padding_between_tensors() {
        // Two small tensors with alignment 32 — second should be offset
        let t1 = TensorInfo::new("a", vec![1], GGUFTensorDType::F32, vec![0u8; 4]);
        let t2 = TensorInfo::new("b", vec![1], GGUFTensorDType::F32, vec![0u8; 4]);
        let mut buf = Vec::new();
        let offsets = TensorDataWriter::write_info_table(&mut buf, &[t1, t2], 32).unwrap();
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1] % 32, 0);
        assert!(offsets[1] >= 4); // at least past first tensor
    }

    // -----------------------------------------------------------------------
    // proptest
    // -----------------------------------------------------------------------

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_metadata_value() -> impl Strategy<Value = MetadataValue> {
            prop_oneof![
                any::<u8>().prop_map(MetadataValue::Uint8),
                any::<i8>().prop_map(MetadataValue::Int8),
                any::<u16>().prop_map(MetadataValue::Uint16),
                any::<i16>().prop_map(MetadataValue::Int16),
                any::<u32>().prop_map(MetadataValue::Uint32),
                any::<i32>().prop_map(MetadataValue::Int32),
                // Use finite floats to avoid NaN comparison issues
                (-1e30f32..1e30f32).prop_map(MetadataValue::Float32),
                any::<bool>().prop_map(MetadataValue::Bool),
                "[a-zA-Z0-9_]{0,100}".prop_map(|s| MetadataValue::String(s)),
                any::<u64>().prop_map(MetadataValue::Uint64),
                any::<i64>().prop_map(MetadataValue::Int64),
                (-1e100f64..1e100f64).prop_map(MetadataValue::Float64),
            ]
        }

        proptest! {
            #[test]
            fn prop_metadata_round_trip(
                key in "[a-z][a-z0-9.]{0,50}",
                val in arb_metadata_value(),
            ) {
                let entry = MetadataEntry::new(&key, val);
                let mut buf = Vec::new();
                MetadataWriter::write_entry(&mut buf, &entry).unwrap();
                let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
                prop_assert_eq!(parsed.key, key);
                prop_assert_eq!(parsed.value, entry.value);
            }

            #[test]
            fn prop_tensor_data_preserved(
                data in proptest::collection::vec(any::<u8>(), 1..256),
                name in "[a-z]{1,20}",
            ) {
                let n = data.len();
                let t = TensorInfo::new(&name, vec![n as u64], GGUFTensorDType::Q8_0, data.clone());
                let rt = GGUFRoundTripper::round_trip(&default_config(), &[], &[t]).unwrap();
                prop_assert_eq!(&rt.tensor_data[name.as_str()], &data);
            }

            #[test]
            fn prop_alignment_valid(alignment in (0u32..10).prop_map(|e| 1u32 << e)) {
                let cfg = GGUFWriterConfig { alignment, ..Default::default() };
                let t = make_f32_tensor("w", &[1.0]);
                let buf = GGUFWriter::write_to_vec(&cfg, &[], &[t]).unwrap();
                let report = GGUFValidator::validate(&buf).unwrap();
                prop_assert!(report.all_ok());
            }

            #[test]
            fn prop_multiple_tensors_validate(count in 1usize..20) {
                let tensors: Vec<TensorInfo> = (0..count)
                    .map(|i| make_f32_tensor(&format!("t{i}"), &[i as f32; 4]))
                    .collect();
                let (_, report) = GGUFWriter::write_and_validate(
                    &default_config(), &[], &tensors,
                ).unwrap();
                prop_assert!(report.all_ok());
                prop_assert_eq!(report.tensor_count, count as u64);
            }

            #[test]
            fn prop_array_metadata_round_trip(
                elems in proptest::collection::vec(any::<u32>(), 0..50),
            ) {
                let arr = MetadataValue::Array(
                    GGUFValueType::Uint32,
                    elems.iter().map(|&v| MetadataValue::Uint32(v)).collect(),
                );
                let entry = MetadataEntry::new("arr", arr.clone());
                let mut buf = Vec::new();
                MetadataWriter::write_entry(&mut buf, &entry).unwrap();
                let parsed = MetadataWriter::read_entry(&mut Cursor::new(&buf)).unwrap();
                prop_assert_eq!(parsed.value, arr);
            }

            #[test]
            fn prop_full_round_trip(
                n_meta in 0usize..10,
                n_tensors in 0usize..10,
            ) {
                let meta: Vec<MetadataEntry> = (0..n_meta)
                    .map(|i| MetadataEntry::new(format!("k{i}"), MetadataValue::Uint32(i as u32)))
                    .collect();
                let tensors: Vec<TensorInfo> = (0..n_tensors)
                    .map(|i| make_f32_tensor(&format!("t{i}"), &[i as f32]))
                    .collect();
                let rt = GGUFRoundTripper::round_trip(&default_config(), &meta, &tensors).unwrap();
                prop_assert_eq!(rt.metadata.len(), n_meta);
                prop_assert_eq!(rt.tensor_names.len(), n_tensors);
            }
        }
    }
}
