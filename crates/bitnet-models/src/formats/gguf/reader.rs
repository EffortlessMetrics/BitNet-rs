//! GGUF file reader

use super::types::*;
use bitnet_common::{BitNetError, ModelError, QuantizationType, Result};

/// GGUF file reader
pub struct GgufReader<'a> {
    data: &'a [u8],
    header: GgufHeader,
    metadata: Vec<GgufMetadata>,
    tensor_infos: Vec<TensorInfo>,
}

impl<'a> GgufReader<'a> {
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
        }

        // Read tensor infos
        let mut tensor_infos = Vec::new();
        for _ in 0..header.tensor_count {
            tensor_infos.push(TensorInfo::read(data, &mut offset)?);
        }

        // Validate tensor offsets
        for (i, tensor_info) in tensor_infos.iter().enumerate() {
            if tensor_info.offset as usize + tensor_info.size as usize > data.len() {
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Tensor {} data extends beyond file bounds", i),
                }));
            }
        }

        Ok(Self { data, header, metadata, tensor_infos })
    }

    /// Get the GGUF version
    pub fn version(&self) -> u32 {
        self.header.version
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> usize {
        self.header.tensor_count as usize
    }

    /// Get the number of metadata entries
    pub fn metadata_count(&self) -> usize {
        self.header.metadata_kv_count as usize
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
        let start = info.offset as usize;
        let end = start + info.size as usize;

        if end > self.data.len() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tensor '{}' data extends beyond file bounds", info.name),
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

    /// Get all metadata keys
    pub fn metadata_keys(&self) -> Vec<&str> {
        self.metadata.iter().map(|m| m.key.as_str()).collect()
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
