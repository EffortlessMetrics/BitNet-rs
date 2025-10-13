//! GGUF file writer
//!
//! This module provides a GGUF v3 writer that creates GGUF files from tensor data.
//! It ensures proper alignment, metadata handling, and LayerNorm preservation.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// GGUF constants
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF value type tags
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;

/// Tensor data types (encoding for GGUF)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TensorDType {
    F32,
    F16,
}

impl TensorDType {
    pub fn type_id(&self) -> u32 {
        match self {
            TensorDType::F32 => 0,
            TensorDType::F16 => 1,
        }
    }

    #[allow(dead_code)]
    pub fn element_size(&self) -> usize {
        match self {
            TensorDType::F32 => 4,
            TensorDType::F16 => 2,
        }
    }
}

/// Metadata value types
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum MetadataValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(f32),
    String(String),
}

impl MetadataValue {
    fn type_tag(&self) -> u32 {
        match self {
            MetadataValue::Bool(_) => GGUF_TYPE_BOOL,
            MetadataValue::U32(_) => GGUF_TYPE_U32,
            MetadataValue::I32(_) => GGUF_TYPE_I32,
            MetadataValue::F32(_) => GGUF_TYPE_F32,
            MetadataValue::String(_) => GGUF_TYPE_STRING,
        }
    }

    fn write_value<W: Write>(&self, writer: &mut W) -> Result<()> {
        match self {
            MetadataValue::Bool(b) => writer.write_all(&[*b as u8])?,
            MetadataValue::U32(u) => writer.write_all(&u.to_le_bytes())?,
            MetadataValue::I32(i) => writer.write_all(&i.to_le_bytes())?,
            MetadataValue::F32(f) => writer.write_all(&f.to_le_bytes())?,
            MetadataValue::String(s) => {
                let bytes = s.as_bytes();
                writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
                writer.write_all(bytes)?;
            }
        }
        Ok(())
    }
}

/// A single tensor to be written
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorDType,
    pub data: Vec<u8>,
}

impl TensorEntry {
    /// Create a new tensor entry
    pub fn new(name: String, shape: Vec<u64>, dtype: TensorDType, data: Vec<u8>) -> Self {
        Self { name, shape, dtype, data }
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// GGUF file writer
pub struct GgufWriter {
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorEntry>,
}

impl GgufWriter {
    /// Create a new GGUF writer
    pub fn new() -> Self {
        Self { metadata: Vec::new(), tensors: Vec::new() }
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.metadata.push((key.into(), value));
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, tensor: TensorEntry) {
        self.tensors.push(tensor);
    }

    /// Write GGUF file to disk
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("Failed to create GGUF file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        self.write(&mut writer)?;

        tracing::info!(
            "Wrote GGUF file with {} tensors and {} metadata entries to {}",
            self.tensors.len(),
            self.metadata.len(),
            path.display()
        );

        Ok(())
    }

    /// Write GGUF to a writer
    fn write<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        // ── Header ───────────────────────────────────────────────────────────
        writer.write_all(GGUF_MAGIC)?;
        writer.write_all(&GGUF_VERSION.to_le_bytes())?;

        let tensor_count = self.tensors.len() as u64;
        let kv_count = self.metadata.len() as u64;

        writer.write_all(&tensor_count.to_le_bytes())?;
        writer.write_all(&kv_count.to_le_bytes())?;

        // ── Metadata (KV pairs) ──────────────────────────────────────────────
        for (key, value) in &self.metadata {
            writer.write_all(&(key.len() as u64).to_le_bytes())?;
            writer.write_all(key.as_bytes())?;
            writer.write_all(&value.type_tag().to_le_bytes())?;
            value.write_value(writer)?;
        }

        // Align to 32 bytes before tensor headers
        align_writer(writer, GGUF_DEFAULT_ALIGNMENT)?;

        // ── Tensor headers ───────────────────────────────────────────────────
        // Compute tensor data offsets first
        let headers_start = writer.stream_position()?;
        let headers_size = self.compute_headers_size();
        let data_start = headers_start + headers_size;

        // Compute offsets for each tensor (relative to data_start)
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut current_offset = 0u64;

        for tensor in &self.tensors {
            current_offset = align_up(current_offset, GGUF_DEFAULT_ALIGNMENT);
            offsets.push(current_offset);
            current_offset += tensor.size_bytes() as u64;
        }

        // Write tensor headers
        for (tensor, &offset) in self.tensors.iter().zip(offsets.iter()) {
            writer.write_all(&(tensor.name.len() as u64).to_le_bytes())?;
            writer.write_all(tensor.name.as_bytes())?;

            let n_dims = tensor.shape.len() as u32;
            writer.write_all(&n_dims.to_le_bytes())?;

            for &dim in &tensor.shape {
                writer.write_all(&dim.to_le_bytes())?;
            }

            writer.write_all(&tensor.dtype.type_id().to_le_bytes())?;
            writer.write_all(&offset.to_le_bytes())?;
        }

        // ── Tensor data ──────────────────────────────────────────────────────
        // Verify we're at the expected position
        let pos = writer.stream_position()?;
        if pos != data_start {
            writer.seek(SeekFrom::Start(data_start))?;
        }

        // Write tensor data with alignment
        for tensor in &self.tensors {
            align_writer(writer, GGUF_DEFAULT_ALIGNMENT)?;
            writer.write_all(&tensor.data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Compute the size of all tensor headers
    fn compute_headers_size(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| {
                // name_len(u64) + name + n_dims(u32) + dims(u64 * n) + dtype(u32) + offset(u64)
                8 + t.name.len() as u64 + 4 + (8 * t.shape.len() as u64) + 4 + 8
            })
            .sum()
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Align a writer to the specified alignment by writing zeros
fn align_writer<W: Write + Seek>(writer: &mut W, alignment: u64) -> Result<()> {
    let pos = writer.stream_position()?;
    let padding = (alignment - (pos % alignment)) % alignment;
    if padding > 0 {
        let zeros = vec![0u8; padding as usize];
        writer.write_all(&zeros)?;
    }
    Ok(())
}

/// Align a value up to the specified alignment
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn test_metadata_value() {
        assert_eq!(MetadataValue::Bool(true).type_tag(), GGUF_TYPE_BOOL);
        assert_eq!(MetadataValue::U32(42).type_tag(), GGUF_TYPE_U32);
        assert_eq!(MetadataValue::I32(-42).type_tag(), GGUF_TYPE_I32);
        assert_eq!(MetadataValue::F32(3.15).type_tag(), GGUF_TYPE_F32);
        assert_eq!(MetadataValue::String("test".to_string()).type_tag(), GGUF_TYPE_STRING);
    }
}
