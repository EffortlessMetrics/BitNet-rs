//! SafeTensors format loader

use crate::loader::{FormatLoader, LoadConfig};
use crate::{Model, BitNetModel};
use bitnet_common::{BitNetConfig, ModelMetadata, Result};
use candle_core::Device;
use std::path::Path;
use tracing::{debug, info};

/// SafeTensors format loader
pub struct SafeTensorsLoader;

impl FormatLoader for SafeTensorsLoader {
    fn name(&self) -> &'static str {
        "SafeTensors"
    }
    
    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase() == "safetensors")
            .unwrap_or(false)
    }
    
    fn detect_format(&self, path: &Path) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }
        
        // Check file extension first
        if self.can_load(path) {
            return Ok(true);
        }
        
        // TODO: Check SafeTensors magic bytes/header
        Ok(false)
    }
    
    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        debug!("Extracting SafeTensors metadata from: {}", path.display());
        
        // TODO: Implement SafeTensors metadata extraction
        let metadata = ModelMetadata {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            version: "unknown".to_string(),
            architecture: "bitnet".to_string(),
            vocab_size: 32000,
            context_length: 2048,
            quantization: None,
        };
        
        debug!("Extracted SafeTensors metadata: {:?}", metadata);
        Ok(metadata)
    }
    
    fn load(
        &self,
        path: &Path,
        device: &Device,
        _config: &LoadConfig,
    ) -> Result<Box<dyn Model<Config = BitNetConfig>>> {
        info!("Loading SafeTensors model from: {}", path.display());
        
        // TODO: Implement SafeTensors loading
        let config = BitNetConfig::default();
        let model = BitNetModel::new(config, device.clone());
        
        Ok(Box::new(model))
    }
}