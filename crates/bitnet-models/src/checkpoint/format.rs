//! Checkpoint format detection from file extensions and header magic bytes.

use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];
const PYTORCH_ZIP_MAGIC: [u8; 2] = [0x50, 0x4B];

/// Supported model checkpoint formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFormat {
    /// GGUF (llama.cpp / ggml ecosystem).
    Gguf,
    /// SafeTensors (HuggingFace standard).
    SafeTensors,
    /// PyTorch serialised checkpoint (`.pt` / `.bin` / `.pth`).
    PyTorch,
    /// User-defined / unrecognised format.
    Custom,
}

impl CheckpointFormat {
    /// Detect checkpoint format from a file path using extension heuristics
    /// followed by a header probe.
    pub fn detect(path: &Path) -> Self {
        if let Some(format) =
            path.extension().and_then(|e| e.to_str()).and_then(Self::from_extension)
        {
            return format;
        }

        // Fall back to header magic bytes when extension is absent or
        // unrecognised.
        Self::detect_from_header(path).unwrap_or(Self::Custom)
    }

    /// Human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::PyTorch => "PyTorch",
            Self::Custom => "Custom",
        }
    }

    fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_lowercase().as_str() {
            "gguf" => Some(Self::Gguf),
            "safetensors" => Some(Self::SafeTensors),
            "pt" | "pth" | "bin" => Some(Self::PyTorch),
            _ => None,
        }
    }

    /// Inspect the first bytes of `path` for known magic values.
    fn detect_from_header(path: &Path) -> Option<Self> {
        let mut file = std::fs::File::open(path).ok()?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header).ok()?;

        // GGUF v3 magic: "GGUF" as LE u32 = 0x46475547.
        if header[..4] == GGUF_MAGIC {
            return Some(Self::Gguf);
        }
        if Self::looks_like_safetensors_header(&header, &mut file) {
            return Some(Self::SafeTensors);
        }
        // PyTorch ZIP-based checkpoints start with the PK magic.
        if header[..2] == PYTORCH_ZIP_MAGIC {
            return Some(Self::PyTorch);
        }
        None
    }

    fn looks_like_safetensors_header(header: &[u8; 8], file: &mut std::fs::File) -> bool {
        // SafeTensors files start with a little-endian u64 length followed by
        // JSON — bytes 4..8 are zero for any header < 4 GiB, and the JSON
        // typically starts with `{`.
        if header[0] == 0 || header[4..8] != [0, 0, 0, 0] {
            return false;
        }

        let mut json_byte = [0u8; 1];
        file.read_exact(&mut json_byte).is_ok() && json_byte[0] == b'{'
    }
}

impl std::fmt::Display for CheckpointFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
