//! Model loading utilities

use bitnet_common::{BitNetConfig, ModelFormat, Result};
use crate::Model;
use candle_core::Device;
use std::path::Path;

/// Model loader trait
pub trait ModelLoader: Send + Sync {
    fn can_load(&self, path: &Path) -> bool;
    fn load(&self, path: &Path, device: &Device) -> Result<Box<dyn Model<Config = BitNetConfig>>>;
}

/// Main model loader with format detection
pub struct BitNetModelLoader {
    device: Device,
}

impl BitNetModelLoader {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn Model<Config = BitNetConfig>>> {
        let path = path.as_ref();
        let format = self.detect_format(path)?;
        
        match format {
            ModelFormat::Gguf => self.load_gguf(path),
            ModelFormat::SafeTensors => self.load_safetensors(path),
            ModelFormat::HuggingFace => self.load_huggingface(path),
        }
    }
    
    fn detect_format(&self, path: &Path) -> Result<ModelFormat> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("gguf") => Ok(ModelFormat::Gguf),
            Some("safetensors") => Ok(ModelFormat::SafeTensors),
            _ => {
                // Try to detect based on directory structure
                if path.join("config.json").exists() {
                    Ok(ModelFormat::HuggingFace)
                } else {
                    Ok(ModelFormat::Gguf) // Default fallback
                }
            }
        }
    }
    
    fn load_gguf(&self, _path: &Path) -> Result<Box<dyn Model<Config = BitNetConfig>>> {
        // Placeholder implementation
        let config = BitNetConfig::default();
        Ok(Box::new(crate::BitNetModel::new(config, self.device.clone())))
    }
    
    fn load_safetensors(&self, _path: &Path) -> Result<Box<dyn Model<Config = BitNetConfig>>> {
        // Placeholder implementation
        let config = BitNetConfig::default();
        Ok(Box::new(crate::BitNetModel::new(config, self.device.clone())))
    }
    
    fn load_huggingface(&self, _path: &Path) -> Result<Box<dyn Model<Config = BitNetConfig>>> {
        // Placeholder implementation
        let config = BitNetConfig::default();
        Ok(Box::new(crate::BitNetModel::new(config, self.device.clone())))
    }
}