//! HuggingFace format loader

use crate::loader::{FormatLoader, LoadConfig};
use crate::{BitNetModel, Model};
use bitnet_common::{BitNetConfig, Device, ModelMetadata, Result};
use std::path::Path;
use tracing::{debug, info};

/// HuggingFace format loader
pub struct HuggingFaceLoader;

impl FormatLoader for HuggingFaceLoader {
    fn name(&self) -> &'static str {
        "HuggingFace"
    }

    fn can_load(&self, path: &Path) -> bool {
        path.is_dir() && path.join("config.json").exists()
    }

    fn detect_format(&self, path: &Path) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }

        Ok(self.can_load(path))
    }

    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        debug!("Extracting HuggingFace metadata from: {}", path.display());

        // TODO: Parse config.json to extract metadata
        let metadata = ModelMetadata {
            name: path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            version: "unknown".to_string(),
            architecture: "bitnet".to_string(),
            vocab_size: 32000,
            context_length: 2048,
            quantization: None,
        };

        debug!("Extracted HuggingFace metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, _config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading HuggingFace model from: {}", path.display());

        // TODO: Implement HuggingFace loading
        let config = BitNetConfig::default();
        let model = BitNetModel::new(config, device.clone());

        Ok(Box::new(model))
    }
}
