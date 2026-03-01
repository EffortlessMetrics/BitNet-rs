//! Model format detection and metadata extraction.
//!
//! Supports auto-detection of GGUF, `SafeTensors`, and ONNX formats
//! from file extensions and magic bytes.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::ModelFormatError;

// Magic bytes for supported formats.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const ONNX_MAGIC: [u8; 4] = [0x08, 0x00, 0x00, 0x00];

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelFormat {
    /// GGUF format (llama.cpp ecosystem).
    Gguf,
    /// `SafeTensors` format (Hugging Face ecosystem).
    SafeTensors,
    /// ONNX format (Open Neural Network Exchange).
    Onnx,
    /// Custom / unknown format with user-provided identifier.
    Custom,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::Onnx => write!(f, "ONNX"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Quantization type hint extracted from model metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantizationHint {
    /// No quantization (full precision).
    None,
    /// 1-bit ternary quantization.
    Ternary1Bit,
    /// 2-bit quantization (e.g., `I2_S`).
    TwoBit,
    /// 4-bit quantization (e.g., `Q4_0`).
    FourBit,
    /// 8-bit quantization (e.g., `Q8_0`).
    EightBit,
    /// Unknown or unrecognized quantization.
    Unknown,
}

/// Metadata extracted from a model file header.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Detected format.
    pub format: ModelFormat,
    /// Number of transformer layers (if available).
    pub num_layers: Option<u32>,
    /// Hidden dimension size (if available).
    pub hidden_size: Option<u32>,
    /// Vocabulary size (if available).
    pub vocab_size: Option<u32>,
    /// Quantization type hint.
    pub quantization_type: QuantizationHint,
}

impl ModelMetadata {
    /// Create metadata with only the format known.
    #[must_use]
    pub const fn with_format(format: ModelFormat) -> Self {
        Self {
            format,
            num_layers: None,
            hidden_size: None,
            vocab_size: None,
            quantization_type: QuantizationHint::Unknown,
        }
    }
}

/// Auto-detects the model format from file extension and magic bytes.
pub struct ModelFormatDetector;

impl ModelFormatDetector {
    /// Detect format from file extension alone.
    #[must_use]
    pub fn from_extension(path: &Path) -> Option<ModelFormat> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "gguf" => Some(ModelFormat::Gguf),
            "safetensors" => Some(ModelFormat::SafeTensors),
            "onnx" => Some(ModelFormat::Onnx),
            _ => None,
        }
    }

    /// Detect format from magic bytes in a buffer.
    #[must_use]
    pub fn from_magic_bytes(data: &[u8]) -> Option<ModelFormat> {
        if data.len() < 4 {
            return None;
        }

        // GGUF: starts with b"GGUF"
        if data[..4] == GGUF_MAGIC {
            return Some(ModelFormat::Gguf);
        }

        // SafeTensors: starts with a little-endian u64 length followed
        // by a JSON object (the header). We check for a '{' after 8 bytes.
        if data.len() >= 9 {
            let header_len = u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]));
            if header_len > 0 && header_len < 100_000_000 && data[8] == b'{' {
                return Some(ModelFormat::SafeTensors);
            }
        }

        // ONNX: protobuf-encoded, starts with field tag 0x08 (varint
        // field 1) followed by the IR version. We look for the common
        // pattern [0x08, ir_version, ...] where ir_version ≤ 20.
        if data[0] == ONNX_MAGIC[0] && data[1] > 0 && data[1] <= 20 {
            return Some(ModelFormat::Onnx);
        }

        None
    }
}

/// Detect the model format for a file on disk.
///
/// Tries extension-based detection first, then falls back to magic bytes.
pub fn detect_format(path: &Path) -> Result<ModelFormat, ModelFormatError> {
    // Fast path: extension check.
    if let Some(fmt) = ModelFormatDetector::from_extension(path) {
        return Ok(fmt);
    }

    // Slow path: read first bytes and check magic.
    if !path.exists() {
        return Err(ModelFormatError::FileNotFound(path.display().to_string()));
    }

    let mut buf = [0u8; 64];
    let mut file = File::open(path)
        .map_err(|e| ModelFormatError::IoError(format!("cannot open {}: {e}", path.display())))?;
    let n = file
        .read(&mut buf)
        .map_err(|e| ModelFormatError::IoError(format!("cannot read {}: {e}", path.display())))?;

    ModelFormatDetector::from_magic_bytes(&buf[..n]).ok_or_else(|| {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("<none>");
        ModelFormatError::UnknownFormat {
            path: path.display().to_string(),
            extension: ext.to_string(),
            suggestion: "Supported formats: .gguf, .safetensors, .onnx".to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn extension_detection_gguf() {
        let p = PathBuf::from("model.gguf");
        assert_eq!(ModelFormatDetector::from_extension(&p), Some(ModelFormat::Gguf));
    }

    #[test]
    fn extension_detection_safetensors() {
        let p = PathBuf::from("weights.safetensors");
        assert_eq!(ModelFormatDetector::from_extension(&p), Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn extension_detection_onnx() {
        let p = PathBuf::from("model.onnx");
        assert_eq!(ModelFormatDetector::from_extension(&p), Some(ModelFormat::Onnx));
    }

    #[test]
    fn extension_detection_unknown() {
        let p = PathBuf::from("model.bin");
        assert_eq!(ModelFormatDetector::from_extension(&p), None);
    }

    #[test]
    fn extension_detection_case_insensitive() {
        let p = PathBuf::from("model.GGUF");
        assert_eq!(ModelFormatDetector::from_extension(&p), Some(ModelFormat::Gguf));
    }

    #[test]
    fn magic_bytes_gguf() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 32]);
        assert_eq!(ModelFormatDetector::from_magic_bytes(&data), Some(ModelFormat::Gguf));
    }

    #[test]
    fn magic_bytes_safetensors() {
        // SafeTensors: 8-byte LE header length + JSON object.
        let header = b"{\"weight\": {}}";
        let mut data = Vec::new();
        data.extend_from_slice(&(header.len() as u64).to_le_bytes());
        data.extend_from_slice(header);
        assert_eq!(ModelFormatDetector::from_magic_bytes(&data), Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn magic_bytes_onnx() {
        // ONNX protobuf: field tag 0x08, IR version 7.
        let data = [0x08, 0x07, 0x12, 0x04];
        assert_eq!(ModelFormatDetector::from_magic_bytes(&data), Some(ModelFormat::Onnx));
    }

    #[test]
    fn magic_bytes_too_short() {
        assert_eq!(ModelFormatDetector::from_magic_bytes(&[0, 1, 2]), None);
    }

    #[test]
    fn magic_bytes_unknown() {
        let data = [0xFF; 64];
        assert_eq!(ModelFormatDetector::from_magic_bytes(&data), None);
    }

    #[test]
    fn model_format_display() {
        assert_eq!(format!("{}", ModelFormat::Gguf), "GGUF");
        assert_eq!(format!("{}", ModelFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", ModelFormat::Onnx), "ONNX");
        assert_eq!(format!("{}", ModelFormat::Custom), "Custom");
    }

    #[test]
    fn metadata_with_format_defaults() {
        let meta = ModelMetadata::with_format(ModelFormat::Gguf);
        assert_eq!(meta.format, ModelFormat::Gguf);
        assert!(meta.num_layers.is_none());
        assert!(meta.hidden_size.is_none());
        assert!(meta.vocab_size.is_none());
        assert_eq!(meta.quantization_type, QuantizationHint::Unknown);
    }

    #[test]
    fn detect_format_file_not_found() {
        let result = detect_format(Path::new("/nonexistent/model.bin"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ModelFormatError::FileNotFound(_)),
            "expected FileNotFound, got: {err:?}"
        );
    }

    #[test]
    fn detect_format_from_gguf_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"GGUF\x02\x00\x00\x00").unwrap();
        assert_eq!(detect_format(&path).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn detect_format_unknown_with_suggestion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.xyz");
        std::fs::write(&path, &[0xFFu8; 64]).unwrap();
        let err = detect_format(&path).unwrap_err();
        match err {
            ModelFormatError::UnknownFormat { suggestion, .. } => {
                assert!(
                    suggestion.contains("Supported formats"),
                    "suggestion should mention supported formats"
                );
            }
            other => panic!("expected UnknownFormat, got: {other:?}"),
        }
    }
}
