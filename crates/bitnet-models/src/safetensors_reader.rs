//! SafeTensors reader for loading HuggingFace model files.
//!
//! Provides low-level access to SafeTensors data for single-file and
//! sharded (multi-file) models. Supports tensor metadata queries and
//! data loading with automatic BF16/F16 → F32 conversion.
//!
//! # Examples
//!
//! ```no_run
//! use bitnet_models::safetensors_reader::SafeTensorsReader;
//!
//! let reader = SafeTensorsReader::from_file("model.safetensors").unwrap();
//! for name in reader.tensor_names() {
//!     let (shape, dtype) = reader.tensor_info(&name).unwrap();
//!     println!("{name}: {shape:?} ({dtype:?})");
//! }
//! let weights = reader.load_tensor("model.embed_tokens.weight").unwrap();
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
use thiserror::Error;
use tracing::debug;

/// Errors specific to SafeTensors reading operations.
#[derive(Debug, Error)]
pub enum SafeTensorsReaderError {
    /// The requested tensor was not found in the model.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    /// The tensor uses an unsupported data type for F32 conversion.
    #[error("unsupported dtype for F32 conversion: {0:?}")]
    UnsupportedDtype(SafeDtype),

    /// An I/O error occurred while reading the file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse the SafeTensors binary data.
    #[error("failed to parse SafeTensors data: {0}")]
    Parse(String),

    /// A shard file referenced by the index was not found.
    #[error("shard file not found: {0}")]
    ShardNotFound(String),

    /// The shard index JSON file is invalid.
    #[error("invalid index file: {0}")]
    InvalidIndex(String),
}

type Result<T> = std::result::Result<T, SafeTensorsReaderError>;

/// Metadata about a single tensor.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions (e.g. `[4096, 4096]`).
    pub shape: Vec<usize>,
    /// Data type as stored on disk.
    pub dtype: SafeDtype,
    /// Which shard file contains this tensor.
    pub shard: String,
}

/// A reader for SafeTensors files that supports both single-file and
/// sharded (multi-file) HuggingFace models.
///
/// The reader memory-maps all shard files and provides zero-copy access
/// to tensor metadata. Tensor data is deserialized on demand when
/// [`load_tensor`](Self::load_tensor) is called.
pub struct SafeTensorsReader {
    tensor_metadata: HashMap<String, TensorMeta>,
    shards: HashMap<String, Mmap>,
}

impl std::fmt::Debug for SafeTensorsReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensorsReader")
            .field("tensor_count", &self.tensor_metadata.len())
            .field("shard_count", &self.shards.len())
            .finish()
    }
}

impl SafeTensorsReader {
    /// Open a single SafeTensors file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        let shard_name =
            path.file_name().and_then(|n| n.to_str()).unwrap_or("model.safetensors").to_string();

        let tensor_metadata = Self::extract_metadata_from_bytes(&mmap, &shard_name)?;

        let mut shards = HashMap::new();
        shards.insert(shard_name, mmap);

        Ok(Self { tensor_metadata, shards })
    }

    /// Open a sharded SafeTensors model from its index file.
    ///
    /// The `index_path` should point to `model.safetensors.index.json`.
    /// Shard files are resolved relative to `dir`.
    pub fn from_sharded(dir: impl AsRef<Path>, index_path: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let index_data =
            std::fs::read_to_string(index_path.as_ref()).map_err(SafeTensorsReaderError::Io)?;

        let index: ShardIndex = serde_json::from_str(&index_data)
            .map_err(|e| SafeTensorsReaderError::InvalidIndex(e.to_string()))?;

        // Collect unique shard filenames
        let unique_shards: HashSet<&str> = index.weight_map.values().map(|s| s.as_str()).collect();

        debug!(
            "Loading sharded model: {} tensors across {} shards",
            index.weight_map.len(),
            unique_shards.len()
        );

        // Memory-map each shard
        let mut shards = HashMap::new();
        for shard_name in &unique_shards {
            let shard_path = dir.join(shard_name);
            if !shard_path.exists() {
                return Err(SafeTensorsReaderError::ShardNotFound(
                    shard_path.display().to_string(),
                ));
            }
            let file = File::open(&shard_path)?;
            let mmap = unsafe { Mmap::map(&file) }?;
            shards.insert(shard_name.to_string(), mmap);
        }

        // Extract tensor metadata from all shards
        let mut tensor_metadata = HashMap::new();
        for (tensor_name, shard_name) in &index.weight_map {
            let mmap = shards
                .get(shard_name.as_str())
                .ok_or_else(|| SafeTensorsReaderError::ShardNotFound(shard_name.clone()))?;
            let st = SafeTensors::deserialize(mmap)
                .map_err(|e| SafeTensorsReaderError::Parse(e.to_string()))?;
            let view = st.tensor(tensor_name).map_err(|e| {
                SafeTensorsReaderError::Parse(format!(
                    "tensor '{}' in shard '{}': {}",
                    tensor_name, shard_name, e
                ))
            })?;
            tensor_metadata.insert(
                tensor_name.clone(),
                TensorMeta {
                    name: tensor_name.clone(),
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    shard: shard_name.clone(),
                },
            );
        }

        Ok(Self { tensor_metadata, shards })
    }

    /// List all tensor names in the model, sorted alphabetically.
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tensor_metadata.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get metadata (shape and dtype) for a named tensor.
    pub fn tensor_info(&self, name: &str) -> Result<(Vec<usize>, SafeDtype)> {
        let meta = self
            .tensor_metadata
            .get(name)
            .ok_or_else(|| SafeTensorsReaderError::TensorNotFound(name.to_string()))?;
        Ok((meta.shape.clone(), meta.dtype))
    }

    /// Load a tensor's data, converting to F32.
    ///
    /// Supported source dtypes:
    /// - **F32** — passthrough (no conversion)
    /// - **F16** — IEEE 754 half-precision → F32
    /// - **BF16** — bfloat16 → F32
    ///
    /// Returns [`SafeTensorsReaderError::UnsupportedDtype`] for other types.
    pub fn load_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let meta = self
            .tensor_metadata
            .get(name)
            .ok_or_else(|| SafeTensorsReaderError::TensorNotFound(name.to_string()))?;

        let mmap = self
            .shards
            .get(&meta.shard)
            .ok_or_else(|| SafeTensorsReaderError::ShardNotFound(meta.shard.clone()))?;

        let st = SafeTensors::deserialize(mmap)
            .map_err(|e| SafeTensorsReaderError::Parse(e.to_string()))?;

        let view = st.tensor(name).map_err(|e| SafeTensorsReaderError::Parse(e.to_string()))?;

        let data = view.data();

        match view.dtype() {
            SafeDtype::F32 => Ok(read_f32_le(data)),
            SafeDtype::F16 => {
                let u16s = read_u16_le(data);
                Ok(u16s.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect())
            }
            SafeDtype::BF16 => {
                let u16s = read_u16_le(data);
                Ok(u16s.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect())
            }
            other => Err(SafeTensorsReaderError::UnsupportedDtype(other)),
        }
    }

    /// Get the number of tensors in the model.
    pub fn tensor_count(&self) -> usize {
        self.tensor_metadata.len()
    }

    /// Get the number of shards backing this model.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Extract tensor metadata from raw SafeTensors bytes.
    fn extract_metadata_from_bytes(
        data: &[u8],
        shard_name: &str,
    ) -> Result<HashMap<String, TensorMeta>> {
        let st = SafeTensors::deserialize(data)
            .map_err(|e| SafeTensorsReaderError::Parse(e.to_string()))?;

        let mut metadata = HashMap::new();
        for name in st.names() {
            let view = st.tensor(name).map_err(|e| SafeTensorsReaderError::Parse(e.to_string()))?;
            metadata.insert(
                name.to_string(),
                TensorMeta {
                    name: name.to_string(),
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    shard: shard_name.to_string(),
                },
            );
        }
        Ok(metadata)
    }
}

/// Read little-endian f32 values from a byte slice, handling potential
/// alignment issues with memory-mapped data.
fn read_f32_le(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Read little-endian u16 values from a byte slice, handling potential
/// alignment issues with memory-mapped data.
fn read_u16_le(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2).map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]])).collect()
}

/// Deserialized shard index (`model.safetensors.index.json`).
#[derive(Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

#[cfg(test)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    /// Helper: serialize tensors to a safetensors byte vector.
    fn make_safetensors(tensors: Vec<(&str, SafeDtype, Vec<usize>, &[u8])>) -> Vec<u8> {
        let views: Vec<(String, TensorView)> = tensors
            .into_iter()
            .map(|(name, dtype, shape, data)| {
                let view = TensorView::new(dtype, shape, data).unwrap();
                (name.to_string(), view)
            })
            .collect();
        safetensors::serialize(views, None).unwrap()
    }

    #[test]
    fn round_trip_f32() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        let data = make_safetensors(vec![("weight", SafeDtype::F32, vec![2, 3], &bytes)]);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&data).unwrap();
        file.flush().unwrap();

        let reader = SafeTensorsReader::from_file(file.path()).unwrap();

        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.shard_count(), 1);
        assert_eq!(reader.tensor_names(), vec!["weight"]);

        let (shape, dtype) = reader.tensor_info("weight").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(dtype, SafeDtype::F32);

        let loaded = reader.load_tensor("weight").unwrap();
        assert_eq!(loaded, values);
    }

    #[test]
    fn bf16_to_f32_conversion() {
        let f32_values: Vec<f32> = vec![1.0, -2.5, 0.0, 42.0];
        let bf16_bytes: Vec<u8> =
            f32_values.iter().flat_map(|&f| half::bf16::from_f32(f).to_le_bytes()).collect();

        let data = make_safetensors(vec![("bias", SafeDtype::BF16, vec![4], &bf16_bytes)]);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&data).unwrap();
        file.flush().unwrap();

        let reader = SafeTensorsReader::from_file(file.path()).unwrap();
        let loaded = reader.load_tensor("bias").unwrap();

        // BF16 has limited precision, check within tolerance
        for (got, expected) in loaded.iter().zip(&f32_values) {
            assert!(
                (got - expected).abs() < 0.1,
                "BF16 conversion mismatch: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn f16_to_f32_conversion() {
        let f32_values: Vec<f32> = vec![0.5, -1.0, 3.14, 0.0];
        let f16_bytes: Vec<u8> =
            f32_values.iter().flat_map(|&f| half::f16::from_f32(f).to_le_bytes()).collect();

        let data = make_safetensors(vec![("proj", SafeDtype::F16, vec![2, 2], &f16_bytes)]);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&data).unwrap();
        file.flush().unwrap();

        let reader = SafeTensorsReader::from_file(file.path()).unwrap();
        let loaded = reader.load_tensor("proj").unwrap();

        for (got, expected) in loaded.iter().zip(&f32_values) {
            assert!(
                (got - expected).abs() < 0.01,
                "F16 conversion mismatch: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn sharded_model_loading() {
        let dir = TempDir::new().unwrap();

        // Shard 1: embedding
        let embed_vals: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let embed_bytes: Vec<u8> = embed_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shard1 = make_safetensors(vec![("embed", SafeDtype::F32, vec![2, 2], &embed_bytes)]);

        // Shard 2: projection
        let proj_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let proj_bytes: Vec<u8> = proj_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shard2 = make_safetensors(vec![("proj", SafeDtype::F32, vec![3, 2], &proj_bytes)]);

        std::fs::write(dir.path().join("shard-00001.safetensors"), &shard1).unwrap();
        std::fs::write(dir.path().join("shard-00002.safetensors"), &shard2).unwrap();

        // Index file
        let index = serde_json::json!({
            "metadata": { "total_size": 40 },
            "weight_map": {
                "embed": "shard-00001.safetensors",
                "proj": "shard-00002.safetensors"
            }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

        let reader = SafeTensorsReader::from_sharded(dir.path(), &index_path).unwrap();

        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.shard_count(), 2);

        let mut names = reader.tensor_names();
        names.sort();
        assert_eq!(names, vec!["embed", "proj"]);

        assert_eq!(reader.load_tensor("embed").unwrap(), embed_vals);
        assert_eq!(reader.load_tensor("proj").unwrap(), proj_vals);
    }

    #[test]
    fn error_missing_tensor() {
        let data =
            make_safetensors(vec![("existing", SafeDtype::F32, vec![1], &1.0f32.to_le_bytes())]);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&data).unwrap();
        file.flush().unwrap();

        let reader = SafeTensorsReader::from_file(file.path()).unwrap();

        let err = reader.tensor_info("nonexistent").unwrap_err();
        assert!(
            matches!(err, SafeTensorsReaderError::TensorNotFound(ref name) if name == "nonexistent"),
            "expected TensorNotFound, got: {err}"
        );

        let err = reader.load_tensor("nonexistent").unwrap_err();
        assert!(matches!(err, SafeTensorsReaderError::TensorNotFound(_)));
    }

    #[test]
    fn error_unsupported_dtype() {
        // I64 is not supported for F32 conversion
        let i64_bytes: Vec<u8> = vec![1i64, 2i64].iter().flat_map(|i| i.to_le_bytes()).collect();
        let data = make_safetensors(vec![("ids", SafeDtype::I64, vec![2], &i64_bytes)]);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&data).unwrap();
        file.flush().unwrap();

        let reader = SafeTensorsReader::from_file(file.path()).unwrap();
        let err = reader.load_tensor("ids").unwrap_err();
        assert!(
            matches!(err, SafeTensorsReaderError::UnsupportedDtype(SafeDtype::I64)),
            "expected UnsupportedDtype(I64), got: {err}"
        );
    }

    #[test]
    fn error_missing_shard() {
        let dir = TempDir::new().unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "tensor_a": "missing_shard.safetensors"
            }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).unwrap()).unwrap();

        let err = SafeTensorsReader::from_sharded(dir.path(), &index_path).unwrap_err();
        assert!(matches!(err, SafeTensorsReaderError::ShardNotFound(_)));
    }

    #[test]
    fn error_invalid_index_json() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, "not valid json").unwrap();

        let err = SafeTensorsReader::from_sharded(dir.path(), &index_path).unwrap_err();
        assert!(matches!(err, SafeTensorsReaderError::InvalidIndex(_)));
    }

    #[test]
    fn multiple_tensors_single_file() {
        let w1: Vec<u8> = vec![1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let w2: Vec<u8> = vec![3.0f32, 4.0, 5.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        let data = make_safetensors(vec![
            ("layer.0.weight", SafeDtype::F32, vec![2], &w1),
            ("layer.1.weight", SafeDtype::F32, vec![3], &w2),
        ]);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&data).unwrap();
        file.flush().unwrap();

        let reader = SafeTensorsReader::from_file(file.path()).unwrap();
        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.load_tensor("layer.0.weight").unwrap(), vec![1.0, 2.0]);
        assert_eq!(reader.load_tensor("layer.1.weight").unwrap(), vec![3.0, 4.0, 5.0]);
    }
}
