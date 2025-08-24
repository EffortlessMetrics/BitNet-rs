//! Format-specific model loaders and unified loading interface

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::debug;

pub mod gguf;
pub mod huggingface;
pub mod safetensors;

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// SafeTensors format (HuggingFace standard)
    SafeTensors,
    /// GGUF format (llama.cpp/ggml ecosystem)
    Gguf,
}

impl ModelFormat {
    /// Detect format from file extension
    pub fn detect_from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("No file extension found"))?;

        match extension.to_lowercase().as_str() {
            "gguf" => Ok(Self::Gguf),
            "safetensors" => Ok(Self::SafeTensors),
            "bin" | "pt" => Ok(Self::SafeTensors), // PyTorch formats
            _ => {
                // Try to detect by reading the file header
                Self::detect_from_header(path)
            }
        }
    }

    /// Detect format by reading file header
    pub fn detect_from_header(path: &Path) -> Result<Self> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let file =
            File::open(path).with_context(|| format!("Failed to open file: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        // Read first 16 bytes for magic number detection
        let mut header = [0u8; 16];
        if reader.read_exact(&mut header).is_err() {
            return Err(anyhow!("File too small to determine format"));
        }

        // Check for GGUF magic: "GGUF" (0x46554747 little-endian)
        if &header[0..4] == b"GGUF" {
            return Ok(Self::Gguf);
        }

        // Check for SafeTensors header (JSON metadata)
        // SafeTensors files start with an 8-byte little-endian size followed by JSON
        if header[0] == b'{' || header[8] == b'{' {
            return Ok(Self::SafeTensors);
        }

        // Default to SafeTensors for unknown formats
        debug!("Unknown format, defaulting to SafeTensors");
        Ok(Self::SafeTensors)
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::SafeTensors => "SafeTensors",
            Self::Gguf => "GGUF",
        }
    }

    /// Get typical file extension
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
        }
    }
}
