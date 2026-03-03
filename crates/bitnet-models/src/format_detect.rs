//! Model format detection from file magic bytes.
//!
//! Auto-detect GGUF, SafeTensors, ONNX, and PyTorch formats
//! by inspecting file headers and extensions.

use std::path::Path;

/// Known model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// GGUF format (llama.cpp compatible).
    Gguf,
    /// HuggingFace SafeTensors.
    SafeTensors,
    /// ONNX model.
    Onnx,
    /// PyTorch checkpoint (.pt/.pth).
    PyTorch,
    /// NumPy array (.npy/.npz).
    NumPy,
    /// Unknown format.
    Unknown,
}

impl ModelFormat {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::Onnx => "ONNX",
            Self::PyTorch => "PyTorch",
            Self::NumPy => "NumPy",
            Self::Unknown => "Unknown",
        }
    }

    /// Typical file extensions for this format.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Gguf => &["gguf"],
            Self::SafeTensors => &["safetensors"],
            Self::Onnx => &["onnx"],
            Self::PyTorch => &["pt", "pth", "bin"],
            Self::NumPy => &["npy", "npz"],
            Self::Unknown => &[],
        }
    }

    /// Whether this format supports sharded loading.
    pub fn supports_sharding(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::PyTorch)
    }
}

/// GGUF magic bytes: "GGUF" in little-endian.
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// SafeTensors files start with a JSON header length (u64 LE).
/// The JSON typically starts with '{'.
const SAFETENSORS_MAX_HEADER: u64 = 100_000_000;

/// ONNX magic: protobuf with field tag 0x08.
const ONNX_MAGIC: [u8; 2] = [0x08, 0x00];

/// PyTorch/Pickle magic: \x80\x02 (pickle protocol 2).
const PICKLE_MAGIC: [u8; 2] = [0x80, 0x02];

/// NumPy magic: \x93NUMPY.
const NUMPY_MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

/// PK zip magic (PyTorch .pt files are often zip archives).
const ZIP_MAGIC: [u8; 4] = [0x50, 0x4B, 0x03, 0x04];

/// Detect format from magic bytes.
pub fn detect_from_bytes(header: &[u8]) -> ModelFormat {
    if header.len() < 4 {
        return ModelFormat::Unknown;
    }

    // GGUF
    if header[..4] == GGUF_MAGIC {
        return ModelFormat::Gguf;
    }

    // NumPy (check before SafeTensors since it has distinct magic)
    if header.len() >= 6 && header[..6] == NUMPY_MAGIC {
        return ModelFormat::NumPy;
    }

    // ZIP (PyTorch)
    if header[..4] == ZIP_MAGIC {
        return ModelFormat::PyTorch;
    }

    // Pickle (PyTorch)
    if header[..2] == PICKLE_MAGIC {
        return ModelFormat::PyTorch;
    }

    // SafeTensors: first 8 bytes are u64 LE header length
    if header.len() >= 16 {
        let header_len = u64::from_le_bytes(header[..8].try_into().unwrap_or([0; 8]));
        if header_len > 0 && header_len < SAFETENSORS_MAX_HEADER && header[8] == b'{' {
            return ModelFormat::SafeTensors;
        }
    }

    // ONNX (rough heuristic)
    if header.len() >= 2 && header[0] == ONNX_MAGIC[0] {
        // Protobuf-style: field 1, varint
        if header[1] < 0x80 {
            return ModelFormat::Onnx;
        }
    }

    ModelFormat::Unknown
}

/// Detect format from file extension.
pub fn detect_from_extension(path: &Path) -> ModelFormat {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    match ext.as_str() {
        "gguf" => ModelFormat::Gguf,
        "safetensors" => ModelFormat::SafeTensors,
        "onnx" => ModelFormat::Onnx,
        "pt" | "pth" | "bin" => ModelFormat::PyTorch,
        "npy" | "npz" => ModelFormat::NumPy,
        _ => ModelFormat::Unknown,
    }
}

/// Combined detection: try magic bytes first, fall back to extension.
pub fn detect_format(path: &Path, header: &[u8]) -> ModelFormat {
    let from_bytes = detect_from_bytes(header);
    if from_bytes != ModelFormat::Unknown {
        return from_bytes;
    }
    detect_from_extension(path)
}

/// Detection result with confidence.
#[derive(Debug)]
pub struct DetectionResult {
    pub format: ModelFormat,
    pub confidence: DetectionConfidence,
    pub source: &'static str,
}

/// Confidence level of detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionConfidence {
    /// Magic bytes matched exactly.
    High,
    /// Extension matched but no magic confirmation.
    Medium,
    /// Heuristic guess.
    Low,
    /// Could not determine.
    None,
}

/// Full detection with confidence.
pub fn detect_with_confidence(path: &Path, header: &[u8]) -> DetectionResult {
    let from_bytes = detect_from_bytes(header);
    if from_bytes != ModelFormat::Unknown {
        return DetectionResult {
            format: from_bytes,
            confidence: DetectionConfidence::High,
            source: "magic_bytes",
        };
    }
    let from_ext = detect_from_extension(path);
    if from_ext != ModelFormat::Unknown {
        return DetectionResult {
            format: from_ext,
            confidence: DetectionConfidence::Medium,
            source: "file_extension",
        };
    }
    DetectionResult {
        format: ModelFormat::Unknown,
        confidence: DetectionConfidence::None,
        source: "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gguf() {
        let header = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(detect_from_bytes(header), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_safetensors() {
        let mut header = vec![0u8; 16];
        // header_len = 42 as u64 LE
        header[..8].copy_from_slice(&42u64.to_le_bytes());
        header[8] = b'{';
        assert_eq!(detect_from_bytes(&header), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_pytorch_zip() {
        let header = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_from_bytes(&header), ModelFormat::PyTorch);
    }

    #[test]
    fn test_detect_pytorch_pickle() {
        let header = [0x80, 0x02, 0x00, 0x00];
        assert_eq!(detect_from_bytes(&header), ModelFormat::PyTorch);
    }

    #[test]
    fn test_detect_numpy() {
        let header = [0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00];
        assert_eq!(detect_from_bytes(&header), ModelFormat::NumPy);
    }

    #[test]
    fn test_detect_unknown() {
        let header = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_from_bytes(&header), ModelFormat::Unknown);
    }

    #[test]
    fn test_detect_from_extension() {
        assert_eq!(detect_from_extension(Path::new("model.gguf")), ModelFormat::Gguf);
        assert_eq!(detect_from_extension(Path::new("model.safetensors")), ModelFormat::SafeTensors);
        assert_eq!(detect_from_extension(Path::new("model.onnx")), ModelFormat::Onnx);
    }

    #[test]
    fn test_combined_detection() {
        let path = Path::new("model.gguf");
        let header = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(detect_format(path, header), ModelFormat::Gguf);
    }

    #[test]
    fn test_extension_fallback() {
        let path = Path::new("model.safetensors");
        let header = [0x00; 4]; // no magic match
        assert_eq!(detect_format(path, &header), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_confidence_high() {
        let path = Path::new("model.bin");
        let header = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = detect_with_confidence(path, header);
        assert_eq!(result.confidence, DetectionConfidence::High);
        assert_eq!(result.format, ModelFormat::Gguf);
    }

    #[test]
    fn test_format_name() {
        assert_eq!(ModelFormat::Gguf.name(), "GGUF");
        assert_eq!(ModelFormat::SafeTensors.name(), "SafeTensors");
    }

    #[test]
    fn test_sharding_support() {
        assert!(ModelFormat::SafeTensors.supports_sharding());
        assert!(!ModelFormat::Gguf.supports_sharding());
    }
}
