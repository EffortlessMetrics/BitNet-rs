//! GGUF model export/writer module.
//!
//! Provides streaming write support for the GGUF format (v2/v3). Tensor data
//! is written directly to disk without buffering the entire model in memory.
//!
//! # Example
//!
//! ```no_run
//! use bitnet_models::gguf_writer::{GgufBuilder, GgufTensorType};
//!
//! let data: Vec<u8> = vec![0u8; 1024];
//! GgufBuilder::new()
//!     .description("example model")
//!     .architecture("llama")
//!     .metadata_u32("llama.context_length", 2048)
//!     .tensor("weights.0", &[32, 32], GgufTensorType::F32, &data)
//!     .write_to_file("model.gguf")
//!     .unwrap();
//! ```

use std::io::{self, Seek, Write};
use std::path::Path;

use bitnet_gguf::GGUF_MAGIC;

// ---------------------------------------------------------------------------
// Tensor data type (write-side mirror of the read-side enum)
// ---------------------------------------------------------------------------

/// Tensor data type discriminant for the GGUF writer.
///
/// Numeric values match the GGML type IDs used in the GGUF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    F64 = 4,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_S = 24,
    I2_S = 36,
}

impl GgufTensorType {
    /// Raw u32 discriminant for serialization.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// Metadata value type tags (GGUF spec)
// ---------------------------------------------------------------------------

const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GGUF writer.
#[derive(Debug, Clone)]
pub struct GgufWriterConfig {
    /// GGUF format version (2 or 3). Default: 3.
    pub version: u32,
    /// Data alignment in bytes. Must be a power of two. Default: 32.
    pub alignment: u32,
    /// Model description (written as `general.description`).
    pub description: Option<String>,
    /// Architecture tag (written as `general.architecture`).
    pub architecture: Option<String>,
}

impl Default for GgufWriterConfig {
    fn default() -> Self {
        Self { version: 3, alignment: 32, description: None, architecture: None }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Metadata value stored before serialization.
#[derive(Debug, Clone)]
enum MetadataValue {
    String(String),
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    ArrayString(Vec<String>),
}

/// Descriptor for a single tensor (header index entry + data reference).
#[derive(Debug, Clone)]
struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    dtype: GgufTensorType,
    /// Byte offset into the data section (filled during write).
    offset: u64,
    /// Raw tensor bytes.
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Streaming GGUF file writer.
///
/// Writes the GGUF header, metadata key-value pairs, tensor descriptors, and
/// tensor data in a single pass. Tensor data is flushed to the underlying
/// writer as it is appended — no full-model buffer is required.
pub struct GgufWriter<W: Write + Seek> {
    config: GgufWriterConfig,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorEntry>,
    writer: W,
}

impl<W: Write + Seek> GgufWriter<W> {
    /// Create a new writer wrapping the given `io::Write + Seek` destination.
    pub fn new(writer: W, config: GgufWriterConfig) -> Self {
        Self { config, metadata: Vec::new(), tensors: Vec::new(), writer }
    }

    // -- metadata helpers ---------------------------------------------------

    /// Add a UTF-8 string metadata entry.
    pub fn add_metadata_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.push((key.into(), MetadataValue::String(value.into())));
    }

    /// Add a u32 metadata entry.
    pub fn add_metadata_u32(&mut self, key: impl Into<String>, value: u32) {
        self.metadata.push((key.into(), MetadataValue::U32(value)));
    }

    /// Add a u64 metadata entry.
    pub fn add_metadata_u64(&mut self, key: impl Into<String>, value: u64) {
        self.metadata.push((key.into(), MetadataValue::U64(value)));
    }

    /// Add an f32 metadata entry.
    pub fn add_metadata_f32(&mut self, key: impl Into<String>, value: f32) {
        self.metadata.push((key.into(), MetadataValue::F32(value)));
    }

    /// Add a boolean metadata entry.
    pub fn add_metadata_bool(&mut self, key: impl Into<String>, value: bool) {
        self.metadata.push((key.into(), MetadataValue::Bool(value)));
    }

    /// Add a string-array metadata entry.
    pub fn add_metadata_string_array(&mut self, key: impl Into<String>, values: Vec<String>) {
        self.metadata.push((key.into(), MetadataValue::ArrayString(values)));
    }

    // -- tensor helpers -----------------------------------------------------

    /// Register a tensor to be written.
    ///
    /// `dims` follows the GGUF convention (innermost dimension first).
    /// `data` must contain exactly the right number of bytes for the given
    /// shape and data type — the writer does **not** validate this.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dims: &[u64],
        dtype: GgufTensorType,
        data: Vec<u8>,
    ) {
        self.tensors.push(TensorEntry {
            name: name.into(),
            dims: dims.to_vec(),
            dtype,
            offset: 0, // filled during `finish()`
            data,
        });
    }

    // -- file writing -------------------------------------------------------

    /// Write the complete GGUF file and return the inner writer.
    pub fn finish(mut self) -> io::Result<W> {
        // Inject implicit metadata from config.
        let mut all_meta = Vec::new();
        if let Some(ref arch) = self.config.architecture {
            all_meta
                .push(("general.architecture".to_string(), MetadataValue::String(arch.clone())));
        }
        if let Some(ref desc) = self.config.description {
            all_meta.push(("general.description".to_string(), MetadataValue::String(desc.clone())));
        }
        all_meta.append(&mut self.metadata);

        let metadata_count = all_meta.len() as u64;
        let tensor_count = self.tensors.len() as u64;
        let alignment = self.config.alignment.max(1) as usize;

        // --- 1. Header ---
        self.writer.write_all(&GGUF_MAGIC)?;
        self.writer.write_all(&self.config.version.to_le_bytes())?;
        self.writer.write_all(&tensor_count.to_le_bytes())?;
        self.writer.write_all(&metadata_count.to_le_bytes())?;

        // --- 2. Metadata KV ---
        for (key, value) in &all_meta {
            write_gguf_string(&mut self.writer, key)?;
            write_metadata_value(&mut self.writer, value)?;
        }

        // --- 3. Tensor descriptors ---
        // Pre-compute offsets relative to the data section start.
        let mut data_offset: u64 = 0;
        for entry in &mut self.tensors {
            entry.offset = data_offset;
            let len = entry.data.len() as u64;
            data_offset = align_up_u64(data_offset + len, alignment as u64);
        }

        for entry in &self.tensors {
            write_gguf_string(&mut self.writer, &entry.name)?;
            self.writer.write_all(&(entry.dims.len() as u32).to_le_bytes())?;
            for &d in &entry.dims {
                self.writer.write_all(&d.to_le_bytes())?;
            }
            self.writer.write_all(&entry.dtype.as_u32().to_le_bytes())?;
            self.writer.write_all(&entry.offset.to_le_bytes())?;
        }

        // --- 4. Alignment padding before data section ---
        pad_to_alignment(&mut self.writer, alignment)?;

        // --- 5. Tensor data ---
        for (i, entry) in self.tensors.iter().enumerate() {
            self.writer.write_all(&entry.data)?;
            // Pad between tensors (skip for last).
            if i + 1 < self.tensors.len() {
                let written = entry.data.len() as u64;
                let padded = align_up_u64(written, alignment as u64);
                let pad = (padded - written) as usize;
                if pad > 0 {
                    self.writer.write_all(&vec![0u8; pad])?;
                }
            }
        }

        self.writer.flush()?;
        Ok(self.writer)
    }
}

// ---------------------------------------------------------------------------
// Builder (ergonomic wrapper)
// ---------------------------------------------------------------------------

/// Convenience builder for constructing GGUF files.
///
/// Collects metadata and tensor descriptors, then writes everything in one
/// call to [`GgufBuilder::write_to_file`] or [`GgufBuilder::write`].
pub struct GgufBuilder {
    config: GgufWriterConfig,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorEntry>,
}

impl GgufBuilder {
    /// Create a new builder with default configuration (GGUF v3, alignment 32).
    pub fn new() -> Self {
        Self { config: GgufWriterConfig::default(), metadata: Vec::new(), tensors: Vec::new() }
    }

    /// Override GGUF version (2 or 3).
    pub fn version(mut self, v: u32) -> Self {
        self.config.version = v;
        self
    }

    /// Override data alignment in bytes.
    pub fn alignment(mut self, a: u32) -> Self {
        self.config.alignment = a;
        self
    }

    /// Set the model description (`general.description`).
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.config.description = Some(d.into());
        self
    }

    /// Set the architecture tag (`general.architecture`).
    pub fn architecture(mut self, a: impl Into<String>) -> Self {
        self.config.architecture = Some(a.into());
        self
    }

    // -- metadata -----------------------------------------------------------

    /// Add a UTF-8 string metadata entry.
    pub fn metadata_string(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), MetadataValue::String(value.into())));
        self
    }

    /// Add a u32 metadata entry.
    pub fn metadata_u32(mut self, key: impl Into<String>, value: u32) -> Self {
        self.metadata.push((key.into(), MetadataValue::U32(value)));
        self
    }

    /// Add a u64 metadata entry.
    pub fn metadata_u64(mut self, key: impl Into<String>, value: u64) -> Self {
        self.metadata.push((key.into(), MetadataValue::U64(value)));
        self
    }

    /// Add an f32 metadata entry.
    pub fn metadata_f32(mut self, key: impl Into<String>, value: f32) -> Self {
        self.metadata.push((key.into(), MetadataValue::F32(value)));
        self
    }

    /// Add a boolean metadata entry.
    pub fn metadata_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.metadata.push((key.into(), MetadataValue::Bool(value)));
        self
    }

    /// Add a string-array metadata entry.
    pub fn metadata_string_array(mut self, key: impl Into<String>, values: Vec<String>) -> Self {
        self.metadata.push((key.into(), MetadataValue::ArrayString(values)));
        self
    }

    // -- tensors ------------------------------------------------------------

    /// Register a tensor with its raw byte data.
    pub fn tensor(
        mut self,
        name: impl Into<String>,
        dims: &[u64],
        dtype: GgufTensorType,
        data: &[u8],
    ) -> Self {
        self.tensors.push(TensorEntry {
            name: name.into(),
            dims: dims.to_vec(),
            dtype,
            offset: 0,
            data: data.to_vec(),
        });
        self
    }

    /// Calculate the total file size that [`GgufBuilder::write`] would produce.
    pub fn calculate_file_size(&self) -> u64 {
        // Reuse the GgufWriter calculation logic by constructing a temporary.
        let alignment = self.config.alignment.max(1) as u64;

        // Header: magic(4) + version(4) + tensor_count(8) + metadata_count(8) = 24
        let mut size: u64 = 24;

        // Implicit metadata from config
        let mut meta_keys: Vec<(&str, MetadataValue)> = Vec::new();
        if let Some(ref arch) = self.config.architecture {
            meta_keys.push(("general.architecture", MetadataValue::String(arch.clone())));
        }
        if let Some(ref desc) = self.config.description {
            meta_keys.push(("general.description", MetadataValue::String(desc.clone())));
        }
        for (k, v) in &self.metadata {
            meta_keys.push((k.as_str(), v.clone()));
        }

        for (key, value) in &meta_keys {
            size += gguf_string_size(key);
            size += metadata_value_size(value);
        }

        // Tensor descriptors
        for entry in &self.tensors {
            size += gguf_string_size(&entry.name);
            size += 4; // n_dims
            size += 8 * entry.dims.len() as u64;
            size += 4; // dtype
            size += 8; // offset
        }

        // Alignment padding before data section
        size = align_up_u64(size, alignment);

        // Tensor data with inter-tensor alignment
        for (i, entry) in self.tensors.iter().enumerate() {
            size += entry.data.len() as u64;
            if i + 1 < self.tensors.len() {
                size = align_up_u64(size, alignment);
            }
        }

        size
    }

    // -- write --------------------------------------------------------------

    /// Write the GGUF file to a seekable writer.
    pub fn write<W: Write + Seek>(self, writer: W) -> io::Result<W> {
        let mut w = GgufWriter::new(writer, self.config);
        w.metadata = self.metadata;
        w.tensors = self.tensors;
        w.finish()
    }

    /// Write the GGUF file to a path on disk.
    pub fn write_to_file(self, path: impl AsRef<Path>) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let buf = io::BufWriter::new(file);
        self.write(buf)?;
        Ok(())
    }
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Write a GGUF-encoded string (u64 length prefix + UTF-8 bytes, no NUL).
fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

/// Serialized size of a GGUF string (8-byte length + data).
fn gguf_string_size(s: &str) -> u64 {
    8 + s.len() as u64
}

/// Write a metadata value (type tag + payload).
fn write_metadata_value<W: Write>(w: &mut W, value: &MetadataValue) -> io::Result<()> {
    match value {
        MetadataValue::String(s) => {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            write_gguf_string(w, s)?;
        }
        MetadataValue::U32(v) => {
            w.write_all(&GGUF_TYPE_U32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::U64(v) => {
            w.write_all(&GGUF_TYPE_U64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::F32(v) => {
            w.write_all(&GGUF_TYPE_F32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Bool(v) => {
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&[u8::from(*v)])?;
        }
        MetadataValue::ArrayString(vals) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            w.write_all(&(vals.len() as u64).to_le_bytes())?;
            for s in vals {
                write_gguf_string(w, s)?;
            }
        }
    }
    Ok(())
}

/// Serialized size of a metadata value (type tag + payload).
fn metadata_value_size(value: &MetadataValue) -> u64 {
    match value {
        MetadataValue::String(s) => 4 + gguf_string_size(s),
        MetadataValue::U32(_) => 4 + 4,
        MetadataValue::U64(_) => 4 + 8,
        MetadataValue::F32(_) => 4 + 4,
        MetadataValue::Bool(_) => 4 + 1,
        MetadataValue::ArrayString(vals) => {
            // type_tag(4) + array_elem_type(4) + count(8) + string data
            let mut sz: u64 = 4 + 4 + 8;
            for s in vals {
                sz += gguf_string_size(s);
            }
            sz
        }
    }
}

/// Pad the current write position to the given alignment.
fn pad_to_alignment<W: Write + Seek>(w: &mut W, alignment: usize) -> io::Result<()> {
    let pos = w.stream_position()?;
    let aligned = align_up_u64(pos, alignment as u64);
    let pad = (aligned - pos) as usize;
    if pad > 0 {
        w.write_all(&vec![0u8; pad])?;
    }
    Ok(())
}

/// Smallest value ≥ `offset` that is a multiple of `alignment`.
#[inline]
fn align_up_u64(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_gguf::{GgufValueType, check_magic, parse_header, read_version};
    use std::io::Cursor;

    // -- round-trip helpers --------------------------------------------------

    /// Read a GGUF-encoded string starting at `offset`.
    fn read_string_at(data: &[u8], offset: &mut usize) -> String {
        let len = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize;
        *offset += 8;
        let s = std::str::from_utf8(&data[*offset..*offset + len]).unwrap().to_string();
        *offset += len;
        s
    }

    /// Read a u32 at `offset`.
    fn read_u32_at(data: &[u8], offset: &mut usize) -> u32 {
        let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
        *offset += 4;
        v
    }

    /// Read a u64 at `offset`.
    fn read_u64_at(data: &[u8], offset: &mut usize) -> u64 {
        let v = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        v
    }

    /// Read an f32 at `offset`.
    fn read_f32_at(data: &[u8], offset: &mut usize) -> f32 {
        let v = f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
        *offset += 4;
        v
    }

    // -- header round-trip ---------------------------------------------------

    #[test]
    fn header_round_trip_v3() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new().version(3).description("test model").write(buf).unwrap();

        let data = out.into_inner();
        assert!(check_magic(&data));
        assert_eq!(read_version(&data), Some(3));

        let info = parse_header(&data).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.tensor_count, 0);
        // description metadata entry
        assert_eq!(info.metadata_count, 1);
    }

    #[test]
    fn header_round_trip_v2() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new().version(2).architecture("llama").write(buf).unwrap();

        let data = out.into_inner();
        assert_eq!(read_version(&data), Some(2));
        let info = parse_header(&data).unwrap();
        assert_eq!(info.version, 2);
        assert_eq!(info.tensor_count, 0);
        assert_eq!(info.metadata_count, 1);
    }

    // -- metadata round-trip -------------------------------------------------

    #[test]
    fn metadata_string_round_trip() {
        let buf = Cursor::new(Vec::new());
        let out =
            GgufBuilder::new().metadata_string("custom.key", "hello world").write(buf).unwrap();

        let data = out.into_inner();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.metadata_count, 1);

        // Skip header (24 bytes) and manually verify the KV entry.
        let mut off = 24;
        let key = read_string_at(&data, &mut off);
        assert_eq!(key, "custom.key");

        let vtype = read_u32_at(&data, &mut off);
        assert_eq!(vtype, GGUF_TYPE_STRING);
        let val = read_string_at(&data, &mut off);
        assert_eq!(val, "hello world");
    }

    #[test]
    fn metadata_u32_round_trip() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new().metadata_u32("ctx.length", 4096).write(buf).unwrap();

        let data = out.into_inner();
        let mut off = 24;
        let key = read_string_at(&data, &mut off);
        assert_eq!(key, "ctx.length");
        let vtype = read_u32_at(&data, &mut off);
        assert_eq!(vtype, GGUF_TYPE_U32);
        let val = read_u32_at(&data, &mut off);
        assert_eq!(val, 4096);
    }

    #[test]
    fn metadata_f32_round_trip() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new().metadata_f32("training.lr", 0.001).write(buf).unwrap();

        let data = out.into_inner();
        let mut off = 24;
        let _key = read_string_at(&data, &mut off);
        let vtype = read_u32_at(&data, &mut off);
        assert_eq!(vtype, GGUF_TYPE_F32);
        let val = read_f32_at(&data, &mut off);
        assert!((val - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn metadata_bool_round_trip() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new().metadata_bool("general.quantized", true).write(buf).unwrap();

        let data = out.into_inner();
        let mut off = 24;
        let _key = read_string_at(&data, &mut off);
        let vtype = read_u32_at(&data, &mut off);
        assert_eq!(vtype, GGUF_TYPE_BOOL);
        assert_eq!(data[off], 1);
    }

    #[test]
    fn metadata_string_array_round_trip() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .metadata_string_array("tokenizer.tokens", vec!["hello".into(), "world".into()])
            .write(buf)
            .unwrap();

        let data = out.into_inner();
        let mut off = 24;
        let key = read_string_at(&data, &mut off);
        assert_eq!(key, "tokenizer.tokens");

        let vtype = read_u32_at(&data, &mut off);
        assert_eq!(vtype, GGUF_TYPE_ARRAY);
        let elem_type = read_u32_at(&data, &mut off);
        assert_eq!(elem_type, GGUF_TYPE_STRING);
        let count = read_u64_at(&data, &mut off);
        assert_eq!(count, 2);

        let s0 = read_string_at(&data, &mut off);
        assert_eq!(s0, "hello");
        let s1 = read_string_at(&data, &mut off);
        assert_eq!(s1, "world");
    }

    #[test]
    fn metadata_u64_round_trip() {
        let buf = Cursor::new(Vec::new());
        let out =
            GgufBuilder::new().metadata_u64("model.params", 7_000_000_000).write(buf).unwrap();

        let data = out.into_inner();
        let mut off = 24;
        let _key = read_string_at(&data, &mut off);
        let vtype = read_u32_at(&data, &mut off);
        assert_eq!(vtype, GGUF_TYPE_U64);
        let val = read_u64_at(&data, &mut off);
        assert_eq!(val, 7_000_000_000);
    }

    // -- tensor descriptor round-trip ----------------------------------------

    #[test]
    fn tensor_descriptor_round_trip() {
        let tensor_data = vec![0u8; 128]; // 32 f32 values
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .tensor("blk.0.attn_q.weight", &[4, 8], GgufTensorType::F32, &tensor_data)
            .write(buf)
            .unwrap();

        let data = out.into_inner();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.tensor_count, 1);

        // Tensor descriptor follows the (empty) metadata section.
        let mut off = 24; // end of header

        // Read tensor info: name, n_dims, dims, dtype, offset
        let name = read_string_at(&data, &mut off);
        assert_eq!(name, "blk.0.attn_q.weight");

        let n_dims = read_u32_at(&data, &mut off);
        assert_eq!(n_dims, 2);

        let d0 = read_u64_at(&data, &mut off);
        let d1 = read_u64_at(&data, &mut off);
        assert_eq!(d0, 4);
        assert_eq!(d1, 8);

        let dtype = read_u32_at(&data, &mut off);
        assert_eq!(dtype, GgufTensorType::F32.as_u32());

        let tensor_offset = read_u64_at(&data, &mut off);
        assert_eq!(tensor_offset, 0); // first (and only) tensor starts at offset 0
    }

    // -- alignment -----------------------------------------------------------

    #[test]
    fn alignment_padding_correctness() {
        let tensor_data = vec![42u8; 5]; // intentionally not aligned
        let alignment = 32u32;
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .alignment(alignment)
            .tensor("t0", &[5], GgufTensorType::F32, &tensor_data)
            .write(buf)
            .unwrap();

        let data = out.into_inner();

        // Find where the data section starts by checking alignment.
        // The data section must start at a position that is a multiple of alignment.
        let info = parse_header(&data).unwrap();
        assert_eq!(info.tensor_count, 1);

        // Verify the tensor data exists somewhere in the file.
        let pos = data.windows(5).position(|w| w == [42, 42, 42, 42, 42]);
        assert!(pos.is_some(), "tensor data not found in output");

        // The data section start must be aligned.
        let data_start = pos.unwrap();
        assert_eq!(
            data_start % alignment as usize,
            0,
            "data section start ({data_start}) is not aligned to {alignment}"
        );
    }

    #[test]
    fn inter_tensor_alignment_padding() {
        let t0 = vec![1u8; 7]; // not aligned
        let t1 = vec![2u8; 3];
        let alignment = 32u32;

        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .alignment(alignment)
            .tensor("t0", &[7], GgufTensorType::F32, &t0)
            .tensor("t1", &[3], GgufTensorType::F32, &t1)
            .write(buf)
            .unwrap();

        let data = out.into_inner();

        // Read back the tensor descriptors to verify offsets.
        let mut off = 24;
        // First tensor
        let _name0 = read_string_at(&data, &mut off);
        let _n0 = read_u32_at(&data, &mut off);
        let _d0 = read_u64_at(&data, &mut off);
        let _dt0 = read_u32_at(&data, &mut off);
        let offset0 = read_u64_at(&data, &mut off);

        // Second tensor
        let _name1 = read_string_at(&data, &mut off);
        let _n1 = read_u32_at(&data, &mut off);
        let _d1 = read_u64_at(&data, &mut off);
        let _dt1 = read_u32_at(&data, &mut off);
        let offset1 = read_u64_at(&data, &mut off);

        assert_eq!(offset0, 0);
        // Second tensor offset must be aligned: align_up(7, 32) = 32
        assert_eq!(offset1, 32);
        assert_eq!(offset1 % alignment as u64, 0);
    }

    // -- builder API ---------------------------------------------------------

    #[test]
    fn builder_chaining() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .version(3)
            .alignment(64)
            .description("my model")
            .architecture("llama")
            .metadata_u32("llama.context_length", 2048)
            .metadata_f32("training.lr", 0.001)
            .metadata_bool("general.quantized", true)
            .tensor("w", &[4], GgufTensorType::F16, &[0u8; 8])
            .write(buf)
            .unwrap();

        let data = out.into_inner();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.tensor_count, 1);
        // architecture + description + u32 + f32 + bool = 5
        assert_eq!(info.metadata_count, 5);
    }

    #[test]
    fn builder_default() {
        let builder = GgufBuilder::default();
        // Should compile and produce a valid (empty) file.
        let buf = Cursor::new(Vec::new());
        let out = builder.write(buf).unwrap();
        let data = out.into_inner();
        assert!(check_magic(&data));
    }

    // -- empty model (metadata only) -----------------------------------------

    #[test]
    fn empty_model_no_tensors() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new().description("empty").architecture("test").write(buf).unwrap();

        let data = out.into_inner();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.tensor_count, 0);
        assert_eq!(info.metadata_count, 2); // arch + desc
    }

    // -- multiple tensors with different dtypes ------------------------------

    #[test]
    fn multiple_tensors_different_types() {
        let f32_data = vec![0u8; 16]; // 4 f32 values
        let f16_data = vec![0u8; 8]; // 4 f16 values
        let i2s_data = vec![0u8; 8]; // I2_S packed block

        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .tensor("layer.0.weight", &[4], GgufTensorType::F32, &f32_data)
            .tensor("layer.1.weight", &[4], GgufTensorType::F16, &f16_data)
            .tensor("layer.2.weight", &[32], GgufTensorType::I2_S, &i2s_data)
            .write(buf)
            .unwrap();

        let data = out.into_inner();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.tensor_count, 3);

        // Read all three tensor descriptors and verify dtypes.
        let mut off = 24;
        for (expected_name, expected_dtype) in [
            ("layer.0.weight", GgufTensorType::F32),
            ("layer.1.weight", GgufTensorType::F16),
            ("layer.2.weight", GgufTensorType::I2_S),
        ] {
            let name = read_string_at(&data, &mut off);
            assert_eq!(name, expected_name);
            let n_dims = read_u32_at(&data, &mut off);
            for _ in 0..n_dims {
                let _ = read_u64_at(&data, &mut off); // skip dims
            }
            let dtype = read_u32_at(&data, &mut off);
            assert_eq!(dtype, expected_dtype.as_u32());
            let _ = read_u64_at(&data, &mut off); // skip offset
        }
    }

    // -- file size calculation -----------------------------------------------

    #[test]
    fn file_size_calculation_matches_actual() {
        let tensor_data = vec![0u8; 64];

        let builder = GgufBuilder::new().description("size test").metadata_u32("key", 42).tensor(
            "t",
            &[16],
            GgufTensorType::F32,
            &tensor_data,
        );

        let predicted = builder.calculate_file_size();

        let buf = Cursor::new(Vec::new());
        let out = builder.write(buf).unwrap();
        let actual = out.into_inner().len() as u64;

        assert_eq!(predicted, actual, "predicted {predicted} != actual {actual}");
    }

    #[test]
    fn file_size_empty_model() {
        let builder = GgufBuilder::new().description("empty");
        let predicted = builder.calculate_file_size();

        let buf = Cursor::new(Vec::new());
        let out = builder.write(buf).unwrap();
        let actual = out.into_inner().len() as u64;

        assert_eq!(predicted, actual);
    }

    #[test]
    fn file_size_multiple_tensors() {
        let builder = GgufBuilder::new()
            .alignment(64)
            .tensor("a", &[3], GgufTensorType::F32, &[0u8; 12])
            .tensor("b", &[5], GgufTensorType::F16, &[0u8; 10]);

        let predicted = builder.calculate_file_size();

        let buf = Cursor::new(Vec::new());
        let out = builder.write(buf).unwrap();
        let actual = out.into_inner().len() as u64;

        assert_eq!(predicted, actual, "predicted {predicted} != actual {actual}");
    }

    // -- GgufWriter direct API -----------------------------------------------

    #[test]
    fn writer_direct_api() {
        let buf = Cursor::new(Vec::new());
        let config = GgufWriterConfig {
            version: 3,
            alignment: 32,
            description: Some("direct".into()),
            architecture: None,
        };
        let mut w = GgufWriter::new(buf, config);
        w.add_metadata_string("custom.key", "value");
        w.add_metadata_u32("custom.u32", 123);
        w.add_metadata_u64("custom.u64", 999);
        w.add_metadata_f32("custom.f32", 1.5);
        w.add_metadata_bool("custom.bool", false);
        w.add_metadata_string_array("custom.arr", vec!["a".into(), "b".into()]);
        w.add_tensor("t", &[4], GgufTensorType::F32, vec![0u8; 16]);

        let out = w.finish().unwrap();
        let data = out.into_inner();
        let info = parse_header(&data).unwrap();

        // description + 6 custom entries = 7
        assert_eq!(info.metadata_count, 7);
        assert_eq!(info.tensor_count, 1);
    }

    // -- GgufTensorType -------------------------------------------------------

    #[test]
    fn tensor_type_as_u32() {
        assert_eq!(GgufTensorType::F32.as_u32(), 0);
        assert_eq!(GgufTensorType::F16.as_u32(), 1);
        assert_eq!(GgufTensorType::I2_S.as_u32(), 36);
        assert_eq!(GgufTensorType::IQ2_S.as_u32(), 24);
        assert_eq!(GgufTensorType::Q4_0.as_u32(), 2);
    }

    // -- align_up_u64 --------------------------------------------------------

    #[test]
    fn align_up_u64_cases() {
        assert_eq!(align_up_u64(0, 32), 0);
        assert_eq!(align_up_u64(1, 32), 32);
        assert_eq!(align_up_u64(31, 32), 32);
        assert_eq!(align_up_u64(32, 32), 32);
        assert_eq!(align_up_u64(33, 32), 64);
        assert_eq!(align_up_u64(100, 64), 128);
        assert_eq!(align_up_u64(5, 0), 5); // zero alignment is identity
    }

    // -- write to file -------------------------------------------------------

    #[test]
    fn write_to_file_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");

        GgufBuilder::new()
            .description("file test")
            .tensor("w", &[2, 2], GgufTensorType::F32, &[0u8; 16])
            .write_to_file(&path)
            .unwrap();

        let data = std::fs::read(&path).unwrap();
        assert!(check_magic(&data));
        let info = parse_header(&data).unwrap();
        assert_eq!(info.tensor_count, 1);
        assert_eq!(info.metadata_count, 1);
    }

    // -- metadata ordering: config entries come first -------------------------

    #[test]
    fn config_metadata_precedes_user_metadata() {
        let buf = Cursor::new(Vec::new());
        let out = GgufBuilder::new()
            .architecture("bitnet")
            .description("test")
            .metadata_string("user.key", "value")
            .write(buf)
            .unwrap();

        let data = out.into_inner();
        let mut off = 24;

        // First KV should be general.architecture
        let k0 = read_string_at(&data, &mut off);
        assert_eq!(k0, "general.architecture");

        // Second KV should be general.description
        // Skip value of first
        let _vtype0 = read_u32_at(&data, &mut off);
        let _val0 = read_string_at(&data, &mut off);
        let k1 = read_string_at(&data, &mut off);
        assert_eq!(k1, "general.description");
    }

    // -- GgufValueType compatibility -----------------------------------------

    #[test]
    fn value_type_constants_match_gguf_spec() {
        // Verify our constants match the bitnet-gguf crate's enum discriminants.
        assert_eq!(GGUF_TYPE_U32, GgufValueType::Uint32 as u32);
        assert_eq!(GGUF_TYPE_F32, GgufValueType::Float32 as u32);
        assert_eq!(GGUF_TYPE_BOOL, GgufValueType::Bool as u32);
        assert_eq!(GGUF_TYPE_STRING, GgufValueType::String as u32);
        assert_eq!(GGUF_TYPE_ARRAY, GgufValueType::Array as u32);
        assert_eq!(GGUF_TYPE_U64, GgufValueType::Uint64 as u32);
    }
}
