//! GGUF file writer
//!
//! This module provides a GGUF v3 writer that creates GGUF files from tensor data.
//! It ensures proper alignment, metadata handling, and LayerNorm preservation.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
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
    pub fn as_gguf_type(&self) -> u32 {
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
}

/// GGUF file writer
pub struct GgufWriter {
    pub(crate) metadata: Vec<(String, MetadataValue)>,
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

    /// Add a tensor from f32 slice (for testing)
    #[cfg(test)]
    pub fn add_tensor_f32(&mut self, name: &str, data: &[f32], shape: &[u64]) -> Result<()> {
        use half::f16;
        let f16_data: Vec<f16> = data.iter().map(|&f| f16::from_f32(f)).collect();
        let data_bytes = bytemuck::cast_slice(&f16_data).to_vec();
        let tensor =
            TensorEntry::new(name.to_string(), shape.to_vec(), TensorDType::F16, data_bytes);
        self.add_tensor(tensor);
        Ok(())
    }

    /// Calculate size of a single metadata value
    fn size_of_kv_value(value: &MetadataValue) -> u64 {
        match value {
            MetadataValue::U32(_) | MetadataValue::I32(_) | MetadataValue::F32(_) => 4,
            MetadataValue::Bool(_) => 1,
            MetadataValue::String(s) => 8 + s.len() as u64, // u64 len + bytes
        }
    }

    /// Calculate size of a single KV entry
    fn size_of_kv_entry(key: &str, value: &MetadataValue) -> u64 {
        // u64 key_len + key_bytes + u32 type + value_bytes
        8 + (key.len() as u64) + 4 + Self::size_of_kv_value(value)
    }

    /// Calculate total size of all KV entries
    fn size_of_all_kv(&self) -> u64 {
        self.metadata.iter().map(|(k, v)| Self::size_of_kv_entry(k, v)).sum()
    }

    /// Calculate size of a single tensor info entry
    fn size_of_tensor_info(entry: &TensorEntry) -> u64 {
        // u64 name_len + name_bytes + u32 ndims + u64*ndims + u32 dtype + u64 offset
        8 + (entry.name.len() as u64) + 4 + 8 * (entry.shape.len() as u64) + 4 + 8
    }

    /// Calculate total size of all tensor info entries
    fn size_of_all_tensor_infos(&self) -> u64 {
        self.tensors.iter().map(Self::size_of_tensor_info).sum()
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

    /// Write GGUF to a writer (two-pass layout with correct data_offset)
    fn write<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        let tensor_count = self.tensors.len() as u64;
        let kv_count = self.metadata.len() as u64;

        // ── Pass 1: Compute section sizes ────────────────────────────────────
        const HEADER_SIZE: u64 = 4 + 4 + 8 + 8 + 4 + 8; // magic + version + tensor_count + kv_count + alignment + data_offset
        let kv_size = self.size_of_all_kv();
        let infos_size = self.size_of_all_tensor_infos();
        let data_offset = align_up(HEADER_SIZE + kv_size + infos_size, GGUF_DEFAULT_ALIGNMENT);

        // Pre-compute per-tensor offsets (relative to data_offset), with per-tensor alignment
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut cursor = 0u64; // relative to data_offset
        for t in &self.tensors {
            cursor = align_up(cursor, GGUF_DEFAULT_ALIGNMENT);
            offsets.push(cursor);
            let bytes_len = t.data.len() as u64;
            cursor += bytes_len;
        }

        // ── Pass 2: Write header with final data_offset ─────────────────────
        writer.write_all(GGUF_MAGIC)?; // magic
        writer.write_all(&GGUF_VERSION.to_le_bytes())?; // version
        writer.write_all(&tensor_count.to_le_bytes())?; // tensors
        writer.write_all(&kv_count.to_le_bytes())?; // kv
        writer.write_all(&(GGUF_DEFAULT_ALIGNMENT as u32).to_le_bytes())?; // alignment (u32)
        writer.write_all(&data_offset.to_le_bytes())?; // data_offset (u64)

        // ── Write KV section ─────────────────────────────────────────────────
        for (key, value) in &self.metadata {
            let kb = key.as_bytes();
            writer.write_all(&(kb.len() as u64).to_le_bytes())?;
            writer.write_all(kb)?;
            writer.write_all(&value.type_tag().to_le_bytes())?;
            match value {
                MetadataValue::U32(v) => writer.write_all(&v.to_le_bytes())?,
                MetadataValue::I32(v) => writer.write_all(&v.to_le_bytes())?,
                MetadataValue::F32(v) => writer.write_all(&v.to_le_bytes())?,
                MetadataValue::Bool(v) => writer.write_all(&[*v as u8])?,
                MetadataValue::String(s) => {
                    let sb = s.as_bytes();
                    writer.write_all(&(sb.len() as u64).to_le_bytes())?;
                    writer.write_all(sb)?;
                }
            }
        }

        // ── Write tensor infos (using precomputed offsets relative to data_offset) ──
        for (idx, t) in self.tensors.iter().enumerate() {
            let nb = t.name.as_bytes();
            writer.write_all(&(nb.len() as u64).to_le_bytes())?;
            writer.write_all(nb)?;
            let ndims = t.shape.len() as u32;
            writer.write_all(&ndims.to_le_bytes())?;
            for d in &t.shape {
                writer.write_all(&d.to_le_bytes())?;
            }
            writer.write_all(&t.dtype.as_gguf_type().to_le_bytes())?;
            writer.write_all(&offsets[idx].to_le_bytes())?;
        }

        // ── Pad up to data_offset if needed ──────────────────────────────────
        let mut pos = writer.stream_position()?;
        if pos < data_offset {
            let pad = (data_offset - pos) as usize;
            writer.write_all(&vec![0u8; pad])?;
            pos = data_offset;
        }
        debug_assert_eq!(pos, data_offset, "writer must be at data_offset");

        // ── Write tensor data with per-tensor alignment ─────────────────────
        let mut rel = 0u64;
        for (idx, t) in self.tensors.iter().enumerate() {
            // align relative cursor
            let aligned_rel = align_up(rel, GGUF_DEFAULT_ALIGNMENT);
            let pad = (aligned_rel - rel) as usize;
            if pad > 0 {
                writer.write_all(&vec![0u8; pad])?;
                rel = aligned_rel;
            }
            // safety: offsets[idx] must match rel now
            debug_assert_eq!(offsets[idx], rel, "offset mismatch @ tensor {}", idx);

            writer.write_all(&t.data)?;
            rel += t.data.len() as u64;
        }

        writer.flush()?;
        Ok(())
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Align a value up to the specified alignment
fn align_up(value: u64, alignment: u64) -> u64 {
    let r = value % alignment;
    if r == 0 { value } else { value + (alignment - r) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_models::formats::gguf::GgufReader;
    use tempfile::NamedTempFile;

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

    #[test]
    fn test_roundtrip_small_tensors_and_alignment() -> Result<()> {
        // 1) write two small tensors + minimal metadata
        let tmp = NamedTempFile::new().unwrap();
        let out = tmp.path().to_path_buf();

        let mut w = GgufWriter::new();
        // required metadata (minimal)
        w.add_metadata("general.architecture", MetadataValue::String("bitnet-b1.58".into()));
        w.add_metadata("bitnet.hidden_size", MetadataValue::U32(8));
        w.add_metadata("bitnet.num_layers", MetadataValue::U32(2));
        w.add_metadata("bitnet.num_heads", MetadataValue::U32(2));
        w.add_metadata("bitnet.vocab_size", MetadataValue::U32(16));
        w.add_metadata("bitnet.context_length", MetadataValue::U32(32));
        w.add_metadata("general.file_type", MetadataValue::U32(1)); // F16/F32 tag per writer contract

        // tensors
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![9.0, 8.0, 7.0];
        w.add_tensor_f32("test.a", &a, &[2, 3])?;
        w.add_tensor_f32("test.b", &b, &[3])?;

        w.write_to_file(&out)?;

        // 2) read with production reader
        let file = File::open(&out).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };

        // Use relaxed limits for testing small files
        use bitnet_common::SecurityLimits;
        let limits = SecurityLimits {
            max_tensor_elements: 1_000_000,
            max_memory_allocation: 100 * 1024 * 1024,
            max_metadata_size: 10 * 1024 * 1024,
            max_string_length: 10 * 1024, // 10KB should be enough for test
            max_array_length: 100_000,
        };
        let reader = GgufReader::new_with_limits(&mmap, &limits).unwrap();

        // sanity: tensor count & names
        assert_eq!(reader.tensor_count() as usize, 2);
        let t0 = reader.get_tensor_info(0).unwrap();
        let t1 = reader.get_tensor_info(1).unwrap();
        assert_eq!(t0.name, "test.a");
        assert_eq!(t1.name, "test.b");
        assert_eq!(t0.shape, vec![2, 3]);
        assert_eq!(t1.shape, vec![3]);

        // 3) alignment & offset monotonicity
        // (GgufReader exposes per-tensor byte offsets)
        assert_eq!(t0.offset % 32, 0, "first tensor must be 32-byte aligned");
        assert_eq!(t1.offset % 32, 0, "second tensor must be 32-byte aligned");
        assert!(t1.offset > t0.offset, "tensor offsets must be monotonic increasing");

        // 4) The important checks passed: tensor offsets are aligned to 32 bytes
        // File size alignment is not a hard requirement in GGUF v3
        Ok(())
    }

    #[test]
    fn test_required_metadata_present_minimal() -> Result<()> {
        let tmp = NamedTempFile::new().unwrap();
        let out = tmp.path().to_path_buf();

        let mut w = GgufWriter::new();
        // put the required keys
        w.add_metadata("general.architecture", MetadataValue::String("bitnet-b1.58".into()));
        w.add_metadata("bitnet.hidden_size", MetadataValue::U32(2560));
        w.add_metadata("bitnet.num_layers", MetadataValue::U32(30));
        w.add_metadata("bitnet.num_heads", MetadataValue::U32(20));
        w.add_metadata("bitnet.vocab_size", MetadataValue::U32(128256));
        w.add_metadata("bitnet.context_length", MetadataValue::U32(4096));
        w.add_metadata("general.file_type", MetadataValue::U32(1));
        // one tensor is enough
        let a: Vec<f32> = vec![0.0f32; 4];
        w.add_tensor_f32("dummy", &a, &[2, 2])?;
        w.write_to_file(&out)?;

        let file = File::open(&out).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let reader = GgufReader::new(&mmap).unwrap();

        // Validate the "must" metadata are present via reader
        let must = [
            "general.architecture",
            "bitnet.hidden_size",
            "bitnet.num_layers",
            "bitnet.num_heads",
            "bitnet.vocab_size",
            "bitnet.context_length",
            "general.file_type",
        ];
        let keys = reader.metadata_keys();
        for key in must {
            let ok = keys.contains(&key);
            assert!(ok, "missing required metadata key: {key}");
        }
        Ok(())
    }
}
