//! Model loading utilities

use crate::Model;
use bitnet_common::{BitNetError, Device, ModelMetadata, Result};
use memmap2::Mmap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

#[cfg(test)]
mod tests;

/// Progress callback for model loading operations
pub type ProgressCallback = Arc<dyn Fn(f32, &str) + Send + Sync>;

/// Model loading configuration
#[derive(Clone)]
pub struct LoadConfig {
    pub use_mmap: bool,
    pub validate_checksums: bool,
    pub progress_callback: Option<ProgressCallback>,
}

impl std::fmt::Debug for LoadConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadConfig")
            .field("use_mmap", &self.use_mmap)
            .field("validate_checksums", &self.validate_checksums)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self { use_mmap: true, validate_checksums: true, progress_callback: None }
    }
}

/// Format-specific loader trait
pub trait FormatLoader: Send + Sync {
    fn name(&self) -> &'static str;
    fn can_load(&self, path: &Path) -> bool;
    fn detect_format(&self, path: &Path) -> Result<bool>;
    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata>;
    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>>;
}

/// Main model loader with automatic format detection
pub struct ModelLoader {
    device: Device,
    loaders: Vec<Box<dyn FormatLoader>>,
}

impl ModelLoader {
    pub fn new(device: Device) -> Self {
        // Register format loaders
        let loaders: Vec<Box<dyn FormatLoader>> = vec![
            Box::new(crate::formats::gguf::GgufLoader),
            Box::new(crate::formats::safetensors::SafeTensorsLoader),
            Box::new(crate::formats::huggingface::HuggingFaceLoader),
        ];

        Self { device, loaders }
    }

    /// Load a model with automatic format detection
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn Model>> {
        self.load_with_config(path, &LoadConfig::default())
    }

    /// Load a model with custom configuration
    pub fn load_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        config: &LoadConfig,
    ) -> Result<Box<dyn Model>> {
        let path = path.as_ref();

        info!("Loading model from: {}", path.display());

        // Report progress
        if let Some(callback) = &config.progress_callback {
            callback(0.0, "Detecting model format...");
        }

        // Detect format using multiple strategies
        let loader = self.detect_format_loader(path)?;

        info!("Detected format: {}", loader.name());

        // Extract metadata first
        if let Some(callback) = &config.progress_callback {
            callback(0.1, "Extracting model metadata...");
        }

        let metadata = loader.extract_metadata(path)?;
        info!("Model metadata: {:?}", metadata);

        // Validate model compatibility
        self.validate_model_compatibility(&metadata)?;

        // Load the model
        if let Some(callback) = &config.progress_callback {
            callback(0.2, "Loading model weights...");
        }

        let model = loader.load(path, &self.device, config)?;

        if let Some(callback) = &config.progress_callback {
            callback(1.0, "Model loaded successfully");
        }

        info!("Model loaded successfully");
        Ok(model)
    }

    /// Detect the appropriate format loader for a given path
    fn detect_format_loader(&self, path: &Path) -> Result<&dyn FormatLoader> {
        // First try extension-based detection
        if let Some(loader) = self.detect_by_extension(path)
            && loader.detect_format(path)?
        {
            return Ok(loader);
        }

        // Try magic byte detection
        if let Some(loader) = self.detect_by_magic_bytes(path)? {
            return Ok(loader);
        }

        // Try directory structure detection
        if let Some(loader) = self.detect_by_structure(path)
            && loader.detect_format(path)?
        {
            return Ok(loader);
        }

        Err(BitNetError::Model(bitnet_common::ModelError::InvalidFormat {
            format: format!("Unable to detect format for: {}", path.display()),
        }))
    }

    /// Detect format by file extension
    fn detect_by_extension(&self, path: &Path) -> Option<&dyn FormatLoader> {
        let extension = path.extension()?.to_str()?;

        match extension.to_lowercase().as_str() {
            "gguf" => self.find_loader("GGUF"),
            "safetensors" => self.find_loader("SafeTensors"),
            _ => None,
        }
    }

    /// Detect format by magic bytes
    fn detect_by_magic_bytes(&self, path: &Path) -> Result<Option<&dyn FormatLoader>> {
        let mut file = File::open(path).map_err(BitNetError::Io)?;

        let mut magic = [0u8; 8];
        let n = file.read(&mut magic).map_err(BitNetError::Io)?;
        if n < 4 {
            return Ok(None);
        }

        // GGUF magic: "GGUF" (0x47475546)
        if &magic[0..4] == b"GGUF" {
            return Ok(self.find_loader("GGUF"));
        }

        // SafeTensors magic: starts with JSON header length or '{'
        if magic[0] == b'{' || (n >= 8 && u64::from_le_bytes(magic) < 1024 * 1024) {
            return Ok(self.find_loader("SafeTensors"));
        }

        Ok(None)
    }

    /// Detect format by directory structure
    fn detect_by_structure(&self, path: &Path) -> Option<&dyn FormatLoader> {
        if path.is_dir() {
            // Check for HuggingFace structure
            if path.join("config.json").exists() {
                return self.find_loader("HuggingFace");
            }
        }

        None
    }

    /// Find a loader by name
    fn find_loader(&self, name: &str) -> Option<&dyn FormatLoader> {
        self.loaders.iter().find(|loader| loader.name() == name).map(|loader| loader.as_ref())
    }

    /// Validate model compatibility
    fn validate_model_compatibility(&self, metadata: &ModelMetadata) -> Result<()> {
        // Check if the model architecture is supported
        if !self.is_supported_architecture(&metadata.architecture) {
            return Err(BitNetError::Model(bitnet_common::ModelError::UnsupportedVersion {
                version: metadata.architecture.clone(),
            }));
        }

        // Check vocabulary size limits
        if metadata.vocab_size == 0 || metadata.vocab_size > 1_000_000 {
            return Err(BitNetError::Validation(format!(
                "Invalid vocabulary size: {}",
                metadata.vocab_size
            )));
        }

        // Check context length limits
        if metadata.context_length == 0 || metadata.context_length > 1_000_000 {
            return Err(BitNetError::Validation(format!(
                "Invalid context length: {}",
                metadata.context_length
            )));
        }

        Ok(())
    }

    /// Check if the architecture is supported
    fn is_supported_architecture(&self, architecture: &str) -> bool {
        matches!(
            architecture.to_lowercase().as_str(),
            "bitnet"
                | "bitnet-b1.58"
                | "llama"
                | "mistral"
                | "qwen"
                | "phi"
                | "gemma"
                | "gemma2"
        )
    }

    /// Get available format loaders
    pub fn available_formats(&self) -> Vec<&str> {
        self.loaders.iter().map(|loader| loader.name()).collect()
    }

    /// Extract metadata without loading the full model
    pub fn extract_metadata<P: AsRef<Path>>(&self, path: P) -> Result<ModelMetadata> {
        let path = path.as_ref();
        let loader = self.detect_format_loader(path)?;
        loader.extract_metadata(path)
    }
}

/// Memory-mapped file wrapper for zero-copy operations
pub struct MmapFile {
    _file: File,
    mmap: Mmap,
}

impl MmapFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(BitNetError::Io)?;

        let mmap = unsafe { Mmap::map(&file) }.map_err(BitNetError::Io)?;

        Ok(Self { _file: file, mmap })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

/// Utility functions for model loading
pub mod utils {
    use super::*;

    /// Create a progress callback that logs to tracing
    pub fn create_logging_progress_callback() -> ProgressCallback {
        Arc::new(|progress, message| {
            debug!("Loading progress: {:.1}% - {}", progress * 100.0, message);
        })
    }

    /// Create a progress callback that prints to stdout
    pub fn create_stdout_progress_callback() -> ProgressCallback {
        Arc::new(|progress, message| {
            println!("Loading progress: {:.1}% - {}", progress * 100.0, message);
        })
    }

    /// Validate file exists and is readable
    pub fn validate_file_access(path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(BitNetError::Model(bitnet_common::ModelError::NotFound {
                path: path.display().to_string(),
            }));
        }

        if path.is_dir() {
            // For directories, check if they contain expected files
            return Ok(());
        }

        // Check if file is readable
        File::open(path).map_err(BitNetError::Io)?;

        Ok(())
    }

    /// Get file size in bytes
    pub fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = std::fs::metadata(path).map_err(BitNetError::Io)?;
        Ok(metadata.len())
    }
}
