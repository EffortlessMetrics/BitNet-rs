//! GGUF file reader

use super::types::*;
use bitnet_common::{BitNetError, ModelError, QuantizationType, Result};

/// GGUF file reader.
///
/// Supports **GGUF v2 and v3**:
/// - v2: header = `magic`, `version`, `n_tensors (u64)`, `n_kv (u64)`. Alignment defaults to 32.
/// - v3: header additionally contains `alignment (u32)` and `data_offset (u64)`.
///
/// For v3, the reader **prefers** `data_offset` when it is valid (>= end of KV section,
/// <= file size, and aligned). Otherwise it falls back to `align_up(kv_end, alignment)`.
/// Alignment is sanitized to a power-of-two (defaults to 32 if the file is malformed).
pub struct GgufReader<'a> {
    data: &'a [u8],
    header: GgufHeader,
    metadata: Vec<GgufMetadata>,
    tensor_infos: Vec<TensorInfo>,
    data_start: usize,
}

impl<'a> GgufReader<'a> {
    /// Compute the tensor-data start offset, preferring v3 `data_offset` when valid.
    /// Falls back to `align_up(kv_end_offset, alignment)` if `data_offset` is invalid.
    #[inline]
    fn compute_data_start(header: &GgufHeader, kv_end_offset: usize, file_size: usize) -> usize {
        let a = (header.alignment.max(1)) as usize;

        if header.version >= 3 {
            let doff = header.data_offset as usize;

            // sanity: doff must be >= kv_end, <= file_size, and aligned
            if doff >= kv_end_offset && doff <= file_size && doff % a == 0 {
                return doff;
            }

            tracing::warn!(
                "GGUF v{}: invalid data_offset={} (kv_end={}, align={}); falling back to align_up",
                header.version,
                doff,
                kv_end_offset,
                a
            );
        }

        align_up(kv_end_offset, a)
    }

    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "File too small to be a valid GGUF file".to_string(),
            }));
        }

        let mut offset = 0;

        // Read header
        let header = GgufHeader::read(data, &mut offset)?;

        // Validate version
        if header.version < 2 || header.version > 3 {
            return Err(BitNetError::Model(ModelError::UnsupportedVersion {
                version: format!("GGUF version {}", header.version),
            }));
        }

        // Read metadata
        let mut metadata = Vec::new();
        for _ in 0..header.metadata_kv_count {
            metadata.push(GgufMetadata::read(data, &mut offset)?);
            // KVs are tightly packed; no per-KV alignment.
        }

        // --- tensor infos: offsets are authoritative ---
        let mut tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            tensor_infos.push(TensorInfo::read(data, &mut offset)?);
        }

        // Compute data start (uses data_offset for v3 if valid)
        let data_start = Self::compute_data_start(&header, offset, data.len());
        let file_size = data.len();

        // Recompute sizes from successive offsets (and file end for the last tensor)
        for i in 0..tensor_infos.len() {
            let cur_off = tensor_infos[i].offset as usize;
            let abs_start = data_start.checked_add(cur_off).ok_or_else(|| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Tensor '{}' offset overflow", tensor_infos[i].name),
                })
            })?;

            let abs_end = if i + 1 < tensor_infos.len() {
                let next_off = tensor_infos[i + 1].offset as usize;
                data_start.checked_add(next_off).ok_or_else(|| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Tensor '{}' next offset overflow", tensor_infos[i].name),
                    })
                })?
            } else {
                file_size
            };

            if abs_end < abs_start {
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!(
                        "Tensor '{}' has decreasing offsets (start: {}, end: {})",
                        tensor_infos[i].name, abs_start, abs_end
                    ),
                }));
            }
            let sz = abs_end - abs_start;
            tensor_infos[i].size = sz as u64;
        }

        // Final bound check (defensive)
        for (i, info) in tensor_infos.iter().enumerate() {
            let start = data_start + info.offset as usize;
            let end = start + info.size as usize;
            if end > file_size {
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!(
                        "Tensor {} '{}' extends beyond file (start {}, end {}, file {})",
                        i, info.name, start, end, file_size
                    ),
                }));
            }
        }

        Ok(Self { data, header, metadata, tensor_infos, data_start })
    }

    /// Get the GGUF version
    pub fn version(&self) -> u32 {
        self.header.version
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> u64 {
        self.header.tensor_count
    }

    /// Get the number of metadata entries
    pub fn metadata_count(&self) -> usize {
        self.header.metadata_kv_count as usize
    }

    /// Get the number of metadata KV pairs (same as metadata_count)
    pub fn metadata_kv_count(&self) -> u64 {
        self.header.metadata_kv_count
    }

    /// Get tensor information by index
    pub fn get_tensor_info(&self, index: usize) -> Result<&TensorInfo> {
        self.tensor_infos
            .get(index)
            .ok_or_else(|| BitNetError::Validation(format!("Tensor index {} out of bounds", index)))
    }

    /// Get tensor information by name
    pub fn get_tensor_info_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.iter().find(|info| info.name == name)
    }

    /// Get tensor data by index
    pub fn get_tensor_data(&self, index: usize) -> Result<&[u8]> {
        let info = self.get_tensor_info(index)?;
        self.get_tensor_data_by_info(info)
    }

    /// Get tensor data by tensor info
    pub fn get_tensor_data_by_info(&self, info: &TensorInfo) -> Result<&[u8]> {
        // Tensor offsets in GGUF are relative to data_start
        let start = self.data_start.checked_add(info.offset as usize).ok_or_else(|| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tensor '{}' offset overflow", info.name),
            })
        })?;

        let end = start.checked_add(info.size as usize).ok_or_else(|| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tensor '{}' size overflow", info.name),
            })
        })?;

        if end > self.data.len() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!(
                    "Tensor '{}' data extends beyond file bounds (start: {}, end: {}, file size: {})",
                    info.name,
                    start,
                    end,
                    self.data.len()
                ),
            }));
        }

        Ok(&self.data[start..end])
    }

    /// Get tensor data by name
    pub fn get_tensor_data_by_name(&self, name: &str) -> Result<&[u8]> {
        let info = self.get_tensor_info_by_name(name).ok_or_else(|| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tensor '{}' not found", name),
            })
        })?;
        self.get_tensor_data_by_info(info)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_infos.iter().map(|info| info.name.as_str()).collect()
    }

    /// Get string metadata by key
    pub fn get_string_metadata(&self, key: &str) -> Option<String> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// Get U32 metadata by key
    pub fn get_u32_metadata(&self, key: &str) -> Option<u32> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::U32(v) => Some(*v),
            _ => None,
        })
    }

    /// Get I32 metadata by key
    pub fn get_i32_metadata(&self, key: &str) -> Option<i32> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::I32(v) => Some(*v),
            _ => None,
        })
    }

    /// Get F32 metadata by key
    pub fn get_f32_metadata(&self, key: &str) -> Option<f32> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        })
    }

    /// Get Bool metadata by key
    pub fn get_bool_metadata(&self, key: &str) -> Option<bool> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        })
    }

    /// Get an array of strings metadata by key
    pub fn get_string_array_metadata(&self, key: &str) -> Option<Vec<String>> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    if let GgufValue::String(s) = v {
                        out.push(s.clone());
                    } else {
                        return None;
                    }
                }
                Some(out)
            }
            _ => None,
        })
    }

    /// Get Array metadata by key (for tokenizer.ggml.model bytes)
    pub fn get_array_metadata(&self, key: &str) -> Option<Vec<u8>> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::Array(arr) => {
                // Convert array of U8 values to byte vector
                let bytes: Vec<u8> = arr
                    .iter()
                    .filter_map(|v| match v {
                        GgufValue::U8(b) => Some(*b),
                        _ => None,
                    })
                    .collect();
                if bytes.len() == arr.len() { Some(bytes) } else { None }
            }
            _ => None,
        })
    }

    /// Get binary metadata by key
    pub fn get_bin_metadata(&self, key: &str) -> Option<Vec<u8>> {
        self.metadata.iter().find(|m| m.key == key).and_then(|m| match &m.value {
            GgufValue::Array(arr) => {
                // Binary data could be stored as U8 array
                let bytes: Vec<u8> = arr
                    .iter()
                    .filter_map(|v| match v {
                        GgufValue::U8(b) => Some(*b),
                        _ => None,
                    })
                    .collect();
                if bytes.len() == arr.len() { Some(bytes) } else { None }
            }
            _ => None,
        })
    }

    /// Get binary or array metadata - tries both formats
    pub fn get_bin_or_u8_array(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(v) = self.get_array_metadata(key) {
            return Some(v);
        }
        if let Some(v) = self.get_bin_metadata(key) {
            return Some(v);
        }
        None
    }

    /// Get all metadata keys
    pub fn metadata_keys(&self) -> Vec<&str> {
        self.metadata.iter().map(|m| m.key.as_str()).collect()
    }

    /// Get alignment value from header
    #[inline]
    pub fn alignment(&self) -> u32 {
        // NOTE: GgufHeader::read() already clamps invalid values (0, non-POT) to 32.
        // So this is always >=1 and a power of two.
        self.header.alignment
    }

    /// Get data offset value from header (v3 only, 0 for v2)
    #[inline]
    pub fn data_offset(&self) -> u64 {
        self.header.data_offset
    }

    /// Infer quantization type from tensor types
    pub fn get_quantization_type(&self) -> Option<QuantizationType> {
        // Count different quantization types
        let mut has_i2s = false;
        let mut has_tl1 = false;
        let mut has_tl2 = false;

        for tensor_info in &self.tensor_infos {
            match tensor_info.tensor_type {
                GgufTensorType::Q4_0 | GgufTensorType::Q4_1 => has_i2s = true,
                GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => has_tl1 = true,
                GgufTensorType::Q8_0 | GgufTensorType::Q8_1 => has_tl2 = true,
                GgufTensorType::Q2_K
                | GgufTensorType::Q3_K
                | GgufTensorType::Q4_K
                | GgufTensorType::Q5_K
                | GgufTensorType::Q6_K
                | GgufTensorType::Q8_K => has_tl2 = true,
                _ => continue,
            }
        }

        // Return the most advanced quantization type found
        if has_tl2 {
            Some(QuantizationType::TL2)
        } else if has_tl1 {
            Some(QuantizationType::TL1)
        } else if has_i2s {
            Some(QuantizationType::I2S)
        } else {
            None
        }
    }

    /// Validate the GGUF file structure
    pub fn validate(&self) -> Result<()> {
        // Check for required metadata
        let required_keys = ["general.architecture", "general.name"];

        for key in &required_keys {
            if self.get_string_metadata(key).is_none() {
                return Err(BitNetError::Validation(format!(
                    "Missing required metadata key: {}",
                    key
                )));
            }
        }

        // Validate tensor names are unique
        let mut tensor_names = std::collections::HashSet::new();
        for tensor_info in &self.tensor_infos {
            if !tensor_names.insert(&tensor_info.name) {
                return Err(BitNetError::Validation(format!(
                    "Duplicate tensor name: {}",
                    tensor_info.name
                )));
            }
        }

        // Validate tensor shapes
        for tensor_info in &self.tensor_infos {
            if tensor_info.shape.is_empty() {
                return Err(BitNetError::Validation(format!(
                    "Tensor '{}' has empty shape",
                    tensor_info.name
                )));
            }

            if tensor_info.shape.contains(&0) {
                return Err(BitNetError::Validation(format!(
                    "Tensor '{}' has zero dimension",
                    tensor_info.name
                )));
            }
        }

        Ok(())
    }

    /// Get file size
    pub fn file_size(&self) -> usize {
        self.data.len()
    }

    /// Calculate total tensor data size
    pub fn total_tensor_size(&self) -> u64 {
        self.tensor_infos.iter().map(|info| info.size).sum()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> GgufMemoryStats {
        let header_size = 24 + self.metadata.len() * 64 + self.tensor_infos.len() * 64; // Approximate
        let tensor_data_size = self.total_tensor_size() as usize;

        GgufMemoryStats {
            total_size: self.data.len(),
            header_size,
            tensor_data_size,
            metadata_count: self.metadata.len(),
            tensor_count: self.tensor_infos.len(),
        }
    }
}

/// Memory usage statistics for GGUF files
#[derive(Debug, Clone)]
pub struct GgufMemoryStats {
    pub total_size: usize,
    pub header_size: usize,
    pub tensor_data_size: usize,
    pub metadata_count: usize,
    pub tensor_count: usize,
}

impl GgufMemoryStats {
    pub fn overhead_percentage(&self) -> f64 {
        if self.total_size == 0 {
            0.0
        } else {
            (self.header_size as f64 / self.total_size as f64) * 100.0
        }
    }
}
