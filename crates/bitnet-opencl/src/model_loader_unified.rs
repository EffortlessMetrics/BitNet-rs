//! Unified model loading trait and format-specific implementations.
//!
//! Provides the [`UnifiedModelLoader`] trait and a [`ModelLoaderRegistry`]
//! for extensible, format-agnostic model loading.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::ModelFormatError;
use crate::model_format::{ModelFormat, ModelMetadata, QuantizationHint};

/// Opaque device identifier for GPU memory targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

/// Placeholder for loaded model weight data.
#[derive(Debug)]
pub struct ModelWeights {
    /// The format the weights were loaded from.
    pub source_format: ModelFormat,
    /// Total size in bytes of the loaded weight data.
    pub total_bytes: u64,
}

/// Trait for format-specific model loaders.
pub trait UnifiedModelLoader: Send + Sync {
    /// The format this loader handles.
    fn format(&self) -> ModelFormat;

    /// Extract metadata from a model file header.
    fn load_metadata(&self, path: &Path) -> Result<ModelMetadata, ModelFormatError>;

    /// Load model weights, optionally targeting a specific device.
    fn load_weights(&self, path: &Path, device: DeviceId)
    -> Result<ModelWeights, ModelFormatError>;

    /// Estimate GPU memory required for the given model metadata.
    fn estimate_memory(&self, metadata: &ModelMetadata) -> u64;
}

// -----------------------------------------------------------------------
// GGUF loader
// -----------------------------------------------------------------------

/// Loader for GGUF model files.
pub struct GgufLoader;

impl GgufLoader {
    /// Bytes per parameter for different quantization types.
    const fn bytes_per_param(hint: QuantizationHint) -> f64 {
        match hint {
            QuantizationHint::None => 4.0,          // FP32
            QuantizationHint::Ternary1Bit => 0.125, // 1-bit
            QuantizationHint::TwoBit => 0.25,       // 2-bit
            QuantizationHint::FourBit => 0.5,       // 4-bit
            QuantizationHint::EightBit => 1.0,      // 8-bit
            QuantizationHint::Unknown => 2.0,       // assume FP16
        }
    }
}

impl UnifiedModelLoader for GgufLoader {
    fn format(&self) -> ModelFormat {
        ModelFormat::Gguf
    }

    fn load_metadata(&self, path: &Path) -> Result<ModelMetadata, ModelFormatError> {
        if !path.exists() {
            return Err(ModelFormatError::FileNotFound(path.display().to_string()));
        }

        let mut file = File::open(path).map_err(|e| {
            ModelFormatError::IoError(format!("cannot open {}: {e}", path.display()))
        })?;

        let mut buf = [0u8; 24];
        let n = file.read(&mut buf).map_err(|e| {
            ModelFormatError::IoError(format!("cannot read {}: {e}", path.display()))
        })?;

        if n < 24 {
            return Err(ModelFormatError::CorruptHeader {
                path: path.display().to_string(),
                position: n,
                detail: format!("expected at least 24 bytes, got {n}"),
            });
        }

        if buf[..4] != *b"GGUF" {
            return Err(ModelFormatError::CorruptHeader {
                path: path.display().to_string(),
                position: 0,
                detail: "invalid GGUF magic bytes".to_string(),
            });
        }

        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if !(2..=3).contains(&version) {
            return Err(ModelFormatError::CorruptHeader {
                path: path.display().to_string(),
                position: 4,
                detail: format!("unsupported GGUF version {version}"),
            });
        }

        let tensor_count = u64::from_le_bytes(buf[8..16].try_into().unwrap());

        // We cannot fully parse metadata KVs here without a
        // full GGUF parser, so we return partial metadata.
        let mut meta = ModelMetadata::with_format(ModelFormat::Gguf);
        // Estimate num_layers from tensor count (rough heuristic:
        // each transformer layer ~7 tensors + embeddings).
        if tensor_count > 7 {
            #[allow(clippy::cast_possible_truncation)]
            let layers = ((tensor_count - 2) / 7) as u32;
            meta.num_layers = Some(layers);
        }
        Ok(meta)
    }

    fn load_weights(
        &self,
        path: &Path,
        device: DeviceId,
    ) -> Result<ModelWeights, ModelFormatError> {
        let _meta = self.load_metadata(path)?;
        let file_size = std::fs::metadata(path)
            .map_err(|e| ModelFormatError::IoError(format!("cannot stat {}: {e}", path.display())))?
            .len();
        tracing::debug!(device = device.0, file_size, "loading GGUF weights");
        Ok(ModelWeights { source_format: ModelFormat::Gguf, total_bytes: file_size })
    }

    fn estimate_memory(&self, metadata: &ModelMetadata) -> u64 {
        estimate_memory_generic(metadata, Self::bytes_per_param)
    }
}

// -----------------------------------------------------------------------
// SafeTensors loader
// -----------------------------------------------------------------------

/// Loader for `SafeTensors` model files.
pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    const fn bytes_per_param(hint: QuantizationHint) -> f64 {
        match hint {
            QuantizationHint::None => 4.0,
            QuantizationHint::Ternary1Bit => 0.125,
            QuantizationHint::TwoBit => 0.25,
            QuantizationHint::FourBit => 0.5,
            QuantizationHint::EightBit => 1.0,
            QuantizationHint::Unknown => 2.0,
        }
    }
}

impl UnifiedModelLoader for SafeTensorsLoader {
    fn format(&self) -> ModelFormat {
        ModelFormat::SafeTensors
    }

    fn load_metadata(&self, path: &Path) -> Result<ModelMetadata, ModelFormatError> {
        if !path.exists() {
            return Err(ModelFormatError::FileNotFound(path.display().to_string()));
        }

        let mut file = File::open(path).map_err(|e| {
            ModelFormatError::IoError(format!("cannot open {}: {e}", path.display()))
        })?;

        let mut len_buf = [0u8; 8];
        let n = file.read(&mut len_buf).map_err(|e| {
            ModelFormatError::IoError(format!("cannot read {}: {e}", path.display()))
        })?;

        if n < 8 {
            return Err(ModelFormatError::CorruptHeader {
                path: path.display().to_string(),
                position: n,
                detail: format!(
                    "expected at least 8 bytes for header \
                     length, got {n}"
                ),
            });
        }

        let header_len_u64 = u64::from_le_bytes(len_buf);
        #[allow(clippy::cast_possible_truncation)]
        let header_len = header_len_u64 as usize;
        if header_len == 0 || header_len > 100_000_000 {
            return Err(ModelFormatError::CorruptHeader {
                path: path.display().to_string(),
                position: 0,
                detail: format!(
                    "invalid SafeTensors header length: \
                     {header_len}"
                ),
            });
        }

        // Read enough of the header to verify JSON structure.
        let peek_len = header_len.min(1024);
        let mut header_buf = vec![0u8; peek_len];
        let read_n = file.read(&mut header_buf).map_err(|e| {
            ModelFormatError::IoError(format!("cannot read header from {}: {e}", path.display()))
        })?;

        if read_n == 0 || header_buf[0] != b'{' {
            return Err(ModelFormatError::CorruptHeader {
                path: path.display().to_string(),
                position: 8,
                detail: "SafeTensors header is not valid JSON".to_string(),
            });
        }

        Ok(ModelMetadata::with_format(ModelFormat::SafeTensors))
    }

    fn load_weights(
        &self,
        path: &Path,
        device: DeviceId,
    ) -> Result<ModelWeights, ModelFormatError> {
        let _meta = self.load_metadata(path)?;
        let file_size = std::fs::metadata(path)
            .map_err(|e| ModelFormatError::IoError(format!("cannot stat {}: {e}", path.display())))?
            .len();
        tracing::debug!(device = device.0, file_size, "loading SafeTensors weights");
        Ok(ModelWeights { source_format: ModelFormat::SafeTensors, total_bytes: file_size })
    }

    fn estimate_memory(&self, metadata: &ModelMetadata) -> u64 {
        estimate_memory_generic(metadata, Self::bytes_per_param)
    }
}

// -----------------------------------------------------------------------
// Generic memory estimation
// -----------------------------------------------------------------------

/// Estimate GPU memory for a model based on its architecture metadata.
///
/// Formula: `num_layers × hidden_size² × bytes_per_param × 4`
/// (4 matrices per layer: Q, K, V, O projections).
/// Plus embedding table: `vocab_size × hidden_size × bytes_per_param`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
fn estimate_memory_generic(
    metadata: &ModelMetadata,
    bytes_per_param: impl Fn(QuantizationHint) -> f64,
) -> u64 {
    let layers = u64::from(metadata.num_layers.unwrap_or(1));
    let hidden = u64::from(metadata.hidden_size.unwrap_or(2048));
    let vocab = u64::from(metadata.vocab_size.unwrap_or(32000));
    let bpp = bytes_per_param(metadata.quantization_type);

    // 4 projection matrices per layer (Q, K, V, O).
    let layer_params = hidden * hidden * 4;
    let total_layer_bytes = (layers as f64 * layer_params as f64 * bpp) as u64;

    // Embedding + output head.
    let embed_bytes = (vocab as f64 * hidden as f64 * bpp) as u64;

    // KV cache overhead estimate (~10% of weight memory).
    let kv_overhead = (total_layer_bytes + embed_bytes) / 10;

    total_layer_bytes + embed_bytes + kv_overhead
}

// -----------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------

/// Registry of format-specific model loaders.
///
/// Register loaders for each supported format, then use
/// [`get_loader`](Self::get_loader) to retrieve the appropriate
/// loader for a detected format.
pub struct ModelLoaderRegistry {
    loaders: HashMap<ModelFormat, Box<dyn UnifiedModelLoader>>,
}

impl ModelLoaderRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { loaders: HashMap::new() }
    }

    /// Create a registry pre-populated with built-in loaders.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(GgufLoader));
        reg.register(Box::new(SafeTensorsLoader));
        reg
    }

    /// Register a loader for its declared format.
    pub fn register(&mut self, loader: Box<dyn UnifiedModelLoader>) {
        self.loaders.insert(loader.format(), loader);
    }

    /// Retrieve the loader for a given format.
    pub fn get_loader(
        &self,
        format: ModelFormat,
    ) -> Result<&dyn UnifiedModelLoader, ModelFormatError> {
        self.loaders
            .get(&format)
            .map(AsRef::as_ref)
            .ok_or_else(|| ModelFormatError::NoLoaderRegistered(format.to_string()))
    }

    /// List all registered formats.
    #[must_use]
    pub fn registered_formats(&self) -> Vec<ModelFormat> {
        self.loaders.keys().copied().collect()
    }
}

impl Default for ModelLoaderRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_loader_format() {
        assert_eq!(GgufLoader.format(), ModelFormat::Gguf);
    }

    #[test]
    fn safetensors_loader_format() {
        assert_eq!(SafeTensorsLoader.format(), ModelFormat::SafeTensors);
    }

    #[test]
    fn memory_estimate_scales_with_layers() {
        let small = ModelMetadata {
            format: ModelFormat::Gguf,
            num_layers: Some(12),
            hidden_size: Some(768),
            vocab_size: Some(32000),
            quantization_type: QuantizationHint::TwoBit,
        };
        let large = ModelMetadata {
            format: ModelFormat::Gguf,
            num_layers: Some(32),
            hidden_size: Some(4096),
            vocab_size: Some(32000),
            quantization_type: QuantizationHint::TwoBit,
        };
        let small_mem = GgufLoader.estimate_memory(&small);
        let large_mem = GgufLoader.estimate_memory(&large);
        assert!(
            large_mem > small_mem,
            "32-layer model ({large_mem}) should need more \
             memory than 12-layer ({small_mem})"
        );
    }

    #[test]
    fn memory_estimate_quantization_reduces_size() {
        let fp32 = ModelMetadata {
            format: ModelFormat::Gguf,
            num_layers: Some(24),
            hidden_size: Some(2048),
            vocab_size: Some(32000),
            quantization_type: QuantizationHint::None,
        };
        let q2 = ModelMetadata {
            format: ModelFormat::Gguf,
            num_layers: Some(24),
            hidden_size: Some(2048),
            vocab_size: Some(32000),
            quantization_type: QuantizationHint::TwoBit,
        };
        let fp32_mem = GgufLoader.estimate_memory(&fp32);
        let q2_mem = GgufLoader.estimate_memory(&q2);
        assert!(fp32_mem > q2_mem * 10, "FP32 ({fp32_mem}) should be >10× 2-bit ({q2_mem})");
    }

    #[test]
    fn memory_estimate_nonzero_for_defaults() {
        let meta = ModelMetadata::with_format(ModelFormat::Gguf);
        let mem = GgufLoader.estimate_memory(&meta);
        assert!(mem > 0, "default metadata should yield nonzero memory");
    }

    #[test]
    fn registry_with_defaults_has_gguf() {
        let reg = ModelLoaderRegistry::with_defaults();
        assert!(reg.get_loader(ModelFormat::Gguf).is_ok());
    }

    #[test]
    fn registry_with_defaults_has_safetensors() {
        let reg = ModelLoaderRegistry::with_defaults();
        assert!(reg.get_loader(ModelFormat::SafeTensors).is_ok());
    }

    #[test]
    fn registry_no_onnx_by_default() {
        let reg = ModelLoaderRegistry::with_defaults();
        let err = reg.get_loader(ModelFormat::Onnx);
        assert!(err.is_err());
    }

    #[test]
    fn registry_custom_loader() {
        struct CustomLoader;
        impl UnifiedModelLoader for CustomLoader {
            fn format(&self) -> ModelFormat {
                ModelFormat::Custom
            }
            fn load_metadata(&self, _path: &Path) -> Result<ModelMetadata, ModelFormatError> {
                Ok(ModelMetadata::with_format(ModelFormat::Custom))
            }
            fn load_weights(
                &self,
                _path: &Path,
                _device: DeviceId,
            ) -> Result<ModelWeights, ModelFormatError> {
                Ok(ModelWeights { source_format: ModelFormat::Custom, total_bytes: 0 })
            }
            fn estimate_memory(&self, _metadata: &ModelMetadata) -> u64 {
                42
            }
        }

        let mut reg = ModelLoaderRegistry::new();
        reg.register(Box::new(CustomLoader));
        assert!(reg.get_loader(ModelFormat::Custom).is_ok());
    }

    #[test]
    fn registry_registered_formats() {
        let reg = ModelLoaderRegistry::with_defaults();
        let fmts = reg.registered_formats();
        assert!(fmts.contains(&ModelFormat::Gguf));
        assert!(fmts.contains(&ModelFormat::SafeTensors));
    }
}
