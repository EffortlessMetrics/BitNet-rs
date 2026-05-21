//! Model format detection from file headers and extensions.

use std::path::Path;

/// Supported model formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    SafeTensorsIndex,
    PyTorchBin,
    OnnxModel,
    Unknown,
}

impl ModelFormat {
    /// Detect format from file extension.
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => Self::Gguf,
            Some("safetensors") => Self::SafeTensors,
            Some("bin") => Self::PyTorchBin,
            Some("onnx") => Self::OnnxModel,
            Some("json") => {
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("model.safetensors.index"))
                {
                    Self::SafeTensorsIndex
                } else {
                    Self::Unknown
                }
            }
            _ => Self::Unknown,
        }
    }

    /// Detect format from file magic bytes.
    pub fn from_magic(bytes: &[u8]) -> Self {
        if bytes.len() >= 4 {
            // GGUF magic: "GGUF" (0x46475547 LE)
            if bytes[0..4] == [0x47, 0x47, 0x55, 0x46] {
                return Self::Gguf;
            }
            // SafeTensors starts with a u64 header length (JSON)
            if bytes.len() >= 8 {
                let header_len = u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]);
                // Reasonable header size for SafeTensors (< 100MB)
                if header_len > 0 && header_len < 100_000_000 && bytes.len() > 8 && bytes[8] == b'{'
                {
                    return Self::SafeTensors;
                }
            }
            // ONNX protobuf (typically starts with \x08)
            if bytes[0] == 0x08 {
                return Self::OnnxModel;
            }
        }
        Self::Unknown
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::SafeTensorsIndex => "SafeTensors Index",
            Self::PyTorchBin => "PyTorch Binary",
            Self::OnnxModel => "ONNX",
            Self::Unknown => "Unknown",
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Gguf | Self::SafeTensors | Self::SafeTensorsIndex)
    }

    pub fn needs_conversion(&self) -> bool {
        matches!(self, Self::PyTorchBin | Self::OnnxModel)
    }
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Represents a detected model file with metadata.
#[derive(Debug, Clone)]
pub struct DetectedModel {
    pub path: std::path::PathBuf,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub shard_index: Option<usize>,
    pub total_shards: Option<usize>,
}

impl DetectedModel {
    pub fn new(path: std::path::PathBuf, format: ModelFormat, size_bytes: u64) -> Self {
        Self { path, format, size_bytes, shard_index: None, total_shards: None }
    }

    pub fn with_shard_info(mut self, index: usize, total: usize) -> Self {
        self.shard_index = Some(index);
        self.total_shards = Some(total);
        self
    }

    pub fn is_sharded(&self) -> bool {
        self.total_shards.is_some_and(|t| t > 1)
    }

    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Parse shard info from a filename like
/// "model-00001-of-00006.safetensors".
pub fn parse_shard_info(filename: &str) -> Option<(usize, usize)> {
    // Pattern: *-NNNNN-of-NNNNN.*
    for (prefix, suffix) in filename.rsplit_once("-of-").into_iter() {
        let idx_str = prefix.rsplit('-').next()?;
        let total_str = suffix.split('.').next()?;
        let (Ok(idx), Ok(total)) = (idx_str.parse::<usize>(), total_str.parse::<usize>()) else {
            return None;
        };

        if idx > 0 && total > 0 && idx <= total {
            return Some((idx, total));
        }
    }
    None
}

/// Conversion capability between formats.
#[derive(Debug, Clone)]
pub struct ConversionCapability {
    pub from: ModelFormat,
    pub to: ModelFormat,
    pub available: bool,
    pub description: String,
}

/// Get available conversion paths.
pub fn available_conversions() -> Vec<ConversionCapability> {
    vec![
        ConversionCapability {
            from: ModelFormat::SafeTensors,
            to: ModelFormat::Gguf,
            available: true,
            description: "SafeTensors to GGUF via bitnet-st2gguf".to_string(),
        },
        ConversionCapability {
            from: ModelFormat::PyTorchBin,
            to: ModelFormat::SafeTensors,
            available: false,
            description: "PyTorch to SafeTensors (requires Python)".to_string(),
        },
        ConversionCapability {
            from: ModelFormat::OnnxModel,
            to: ModelFormat::Gguf,
            available: false,
            description: "ONNX to GGUF (not yet implemented)".to_string(),
        },
    ]
}

/// Find available conversion path.
pub fn find_conversion(from: ModelFormat, to: ModelFormat) -> Option<ConversionCapability> {
    available_conversions().into_iter().find(|c| c.from == from && c.to == to && c.available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_format_from_extension_gguf() {
        assert_eq!(ModelFormat::from_extension(Path::new("model.gguf")), ModelFormat::Gguf);
    }

    #[test]
    fn test_format_from_extension_safetensors() {
        assert_eq!(
            ModelFormat::from_extension(Path::new("model.safetensors")),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_format_from_extension_bin() {
        assert_eq!(ModelFormat::from_extension(Path::new("model.bin")), ModelFormat::PyTorchBin);
    }

    #[test]
    fn test_format_from_extension_onnx() {
        assert_eq!(ModelFormat::from_extension(Path::new("model.onnx")), ModelFormat::OnnxModel);
    }

    #[test]
    fn test_format_from_extension_unknown() {
        assert_eq!(ModelFormat::from_extension(Path::new("model.xyz")), ModelFormat::Unknown);
    }

    #[test]
    fn test_format_from_magic_gguf() {
        let magic = b"GGUF\x03\x00\x00\x00";
        assert_eq!(ModelFormat::from_magic(magic), ModelFormat::Gguf);
    }

    #[test]
    fn test_format_from_magic_safetensors() {
        let mut bytes = vec![0u8; 16];
        // Header length = 100 (little endian u64)
        bytes[0..8].copy_from_slice(&100u64.to_le_bytes());
        bytes[8] = b'{';
        assert_eq!(ModelFormat::from_magic(&bytes), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_format_from_magic_unknown() {
        assert_eq!(ModelFormat::from_magic(&[0xFF, 0xFF, 0xFF, 0xFF]), ModelFormat::Unknown);
    }

    #[test]
    fn test_format_from_magic_short() {
        assert_eq!(ModelFormat::from_magic(&[0x47]), ModelFormat::Unknown);
    }

    #[test]
    fn test_display_name() {
        assert_eq!(ModelFormat::Gguf.display_name(), "GGUF");
        assert_eq!(ModelFormat::SafeTensors.display_name(), "SafeTensors");
    }

    #[test]
    fn test_is_supported() {
        assert!(ModelFormat::Gguf.is_supported());
        assert!(ModelFormat::SafeTensors.is_supported());
        assert!(!ModelFormat::PyTorchBin.is_supported());
        assert!(!ModelFormat::Unknown.is_supported());
    }

    #[test]
    fn test_needs_conversion() {
        assert!(!ModelFormat::Gguf.needs_conversion());
        assert!(ModelFormat::PyTorchBin.needs_conversion());
        assert!(ModelFormat::OnnxModel.needs_conversion());
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", ModelFormat::Gguf), "GGUF");
    }

    #[test]
    fn test_detected_model_new() {
        let m = DetectedModel::new(PathBuf::from("test.gguf"), ModelFormat::Gguf, 1000);
        assert_eq!(m.format, ModelFormat::Gguf);
        assert!(!m.is_sharded());
    }

    #[test]
    fn test_detected_model_sharded() {
        let m = DetectedModel::new(
            PathBuf::from("test.safetensors"),
            ModelFormat::SafeTensors,
            5_000_000_000,
        )
        .with_shard_info(1, 6);
        assert!(m.is_sharded());
        assert_eq!(m.shard_index, Some(1));
    }

    #[test]
    fn test_size_conversions() {
        let m = DetectedModel::new(PathBuf::from("test.gguf"), ModelFormat::Gguf, 1_073_741_824);
        assert!((m.size_mb() - 1024.0).abs() < 0.01);
        assert!((m.size_gb() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_shard_info() {
        assert_eq!(parse_shard_info("model-00001-of-00006.safetensors"), Some((1, 6)));
    }

    #[test]
    fn test_parse_shard_info_no_shard() {
        assert_eq!(parse_shard_info("model.safetensors"), None);
    }

    #[test]
    fn test_parse_shard_info_varied() {
        assert_eq!(parse_shard_info("weights-00003-of-00012.safetensors"), Some((3, 12)));
    }

    #[test]
    fn test_parse_shard_info_rejects_zero_and_out_of_range() {
        assert_eq!(parse_shard_info("model-00000-of-00006.safetensors"), None);
        assert_eq!(parse_shard_info("model-00007-of-00006.safetensors"), None);
        assert_eq!(parse_shard_info("model-00001-of-00000.safetensors"), None);
    }

    #[test]
    fn test_available_conversions() {
        let convs = available_conversions();
        assert!(convs.len() >= 3);
    }

    #[test]
    fn test_find_conversion_exists() {
        let conv = find_conversion(ModelFormat::SafeTensors, ModelFormat::Gguf);
        assert!(conv.is_some());
        assert!(conv.unwrap().available);
    }

    #[test]
    fn test_find_conversion_missing() {
        let conv = find_conversion(ModelFormat::Unknown, ModelFormat::Gguf);
        assert!(conv.is_none());
    }

    #[test]
    fn test_format_equality() {
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
    }
}
