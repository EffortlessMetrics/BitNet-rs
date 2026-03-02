//! Weight format detection.
//!
//! Auto-detects tensor weight formats from binary data patterns.

/// Weight storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    F32,
    F16,
    BF16,
    I2S,
    QK256,
    Int8,
    Int4,
    Unknown,
}

impl WeightFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I2S => "i2s",
            Self::QK256 => "qk256",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Unknown => "unknown",
        }
    }

    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
            Self::I2S => 0.25,
            Self::QK256 => 0.28, // 256-element blocks with scale
            Self::Unknown => 0.0,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::I2S | Self::QK256 | Self::Int8 | Self::Int4)
    }

    pub fn is_floating(&self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

/// Detection result.
#[derive(Debug, Clone)]
pub struct FormatDetection {
    pub format: WeightFormat,
    pub confidence: f64,
    pub evidence: String,
}

/// Detect format from tensor byte size and element count.
pub fn detect_from_size(byte_size: usize, element_count: usize) -> FormatDetection {
    if element_count == 0 {
        return FormatDetection {
            format: WeightFormat::Unknown,
            confidence: 0.0,
            evidence: "zero elements".into(),
        };
    }

    let ratio = byte_size as f64 / element_count as f64;

    if (ratio - 4.0).abs() < 0.01 {
        FormatDetection {
            format: WeightFormat::F32,
            confidence: 0.99,
            evidence: format!("byte ratio {ratio:.3} matches f32"),
        }
    } else if (ratio - 2.0).abs() < 0.01 {
        // Could be F16 or BF16, default to F16
        FormatDetection {
            format: WeightFormat::F16,
            confidence: 0.8,
            evidence: format!("byte ratio {ratio:.3} matches f16/bf16"),
        }
    } else if (ratio - 1.0).abs() < 0.01 {
        FormatDetection {
            format: WeightFormat::Int8,
            confidence: 0.9,
            evidence: format!("byte ratio {ratio:.3} matches int8"),
        }
    } else if (ratio - 0.5).abs() < 0.05 {
        FormatDetection {
            format: WeightFormat::Int4,
            confidence: 0.85,
            evidence: format!("byte ratio {ratio:.3} matches int4"),
        }
    } else if (ratio - 0.25).abs() < 0.05 {
        FormatDetection {
            format: WeightFormat::I2S,
            confidence: 0.85,
            evidence: format!("byte ratio {ratio:.3} matches i2s"),
        }
    } else if ratio > 0.25 && ratio < 0.35 {
        FormatDetection {
            format: WeightFormat::QK256,
            confidence: 0.7,
            evidence: format!("byte ratio {ratio:.3} in qk256 range"),
        }
    } else {
        FormatDetection {
            format: WeightFormat::Unknown,
            confidence: 0.0,
            evidence: format!("byte ratio {ratio:.3} matches no known format"),
        }
    }
}

/// Detect format from GGUF type ID.
pub fn detect_from_gguf_type(type_id: u32) -> FormatDetection {
    match type_id {
        0 => FormatDetection {
            format: WeightFormat::F32,
            confidence: 1.0,
            evidence: "GGUF type 0 = F32".into(),
        },
        1 => FormatDetection {
            format: WeightFormat::F16,
            confidence: 1.0,
            evidence: "GGUF type 1 = F16".into(),
        },
        30 => FormatDetection {
            format: WeightFormat::BF16,
            confidence: 1.0,
            evidence: "GGUF type 30 = BF16".into(),
        },
        10 => FormatDetection {
            format: WeightFormat::I2S,
            confidence: 1.0,
            evidence: "GGUF type 10 = I2_S".into(),
        },
        _ => FormatDetection {
            format: WeightFormat::Unknown,
            confidence: 0.5,
            evidence: format!("GGUF type {type_id} not mapped"),
        },
    }
}

/// Estimate memory for a tensor.
pub fn estimate_memory(element_count: usize, format: WeightFormat) -> usize {
    (element_count as f64 * format.bytes_per_element()) as usize
}

/// Model-level format summary.
#[derive(Debug, Clone)]
pub struct FormatSummary {
    pub primary_format: WeightFormat,
    pub formats_seen: Vec<(WeightFormat, usize)>, // format, tensor count
    pub total_tensors: usize,
    pub total_bytes: usize,
}

impl FormatSummary {
    pub fn new() -> Self {
        Self {
            primary_format: WeightFormat::Unknown,
            formats_seen: Vec::new(),
            total_tensors: 0,
            total_bytes: 0,
        }
    }

    pub fn add(&mut self, format: WeightFormat, byte_size: usize) {
        self.total_tensors += 1;
        self.total_bytes += byte_size;
        if let Some(entry) = self.formats_seen.iter_mut().find(|(f, _)| *f == format) {
            entry.1 += 1;
        } else {
            self.formats_seen.push((format, 1));
        }
        // Primary = most common
        if let Some((fmt, _)) = self.formats_seen.iter().max_by_key(|(_, c)| *c) {
            self.primary_format = *fmt;
        }
    }

    pub fn is_mixed(&self) -> bool {
        self.formats_seen.len() > 1
    }
}

impl Default for FormatSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_f32() {
        let d = detect_from_size(4096, 1024);
        assert_eq!(d.format, WeightFormat::F32);
        assert!(d.confidence > 0.9);
    }

    #[test]
    fn test_detect_f16() {
        let d = detect_from_size(2048, 1024);
        assert_eq!(d.format, WeightFormat::F16);
    }

    #[test]
    fn test_detect_int8() {
        let d = detect_from_size(1024, 1024);
        assert_eq!(d.format, WeightFormat::Int8);
    }

    #[test]
    fn test_detect_int4() {
        let d = detect_from_size(512, 1024);
        assert_eq!(d.format, WeightFormat::Int4);
    }

    #[test]
    fn test_detect_i2s() {
        let d = detect_from_size(256, 1024);
        assert_eq!(d.format, WeightFormat::I2S);
    }

    #[test]
    fn test_detect_zero() {
        let d = detect_from_size(0, 0);
        assert_eq!(d.format, WeightFormat::Unknown);
    }

    #[test]
    fn test_gguf_f32() {
        let d = detect_from_gguf_type(0);
        assert_eq!(d.format, WeightFormat::F32);
        assert_eq!(d.confidence, 1.0);
    }

    #[test]
    fn test_gguf_i2s() {
        let d = detect_from_gguf_type(10);
        assert_eq!(d.format, WeightFormat::I2S);
    }

    #[test]
    fn test_gguf_bf16() {
        let d = detect_from_gguf_type(30);
        assert_eq!(d.format, WeightFormat::BF16);
    }

    #[test]
    fn test_is_quantized() {
        assert!(WeightFormat::I2S.is_quantized());
        assert!(WeightFormat::Int4.is_quantized());
        assert!(!WeightFormat::F32.is_quantized());
    }

    #[test]
    fn test_is_floating() {
        assert!(WeightFormat::F32.is_floating());
        assert!(WeightFormat::BF16.is_floating());
        assert!(!WeightFormat::Int8.is_floating());
    }

    #[test]
    fn test_estimate_memory() {
        assert_eq!(estimate_memory(1024, WeightFormat::F32), 4096);
        assert_eq!(estimate_memory(1024, WeightFormat::F16), 2048);
    }

    #[test]
    fn test_format_summary() {
        let mut s = FormatSummary::new();
        s.add(WeightFormat::F16, 2048);
        s.add(WeightFormat::F16, 4096);
        s.add(WeightFormat::F32, 1024);
        assert_eq!(s.primary_format, WeightFormat::F16);
        assert!(s.is_mixed());
        assert_eq!(s.total_tensors, 3);
    }

    #[test]
    fn test_format_as_str() {
        assert_eq!(WeightFormat::QK256.as_str(), "qk256");
        assert_eq!(WeightFormat::BF16.as_str(), "bf16");
    }

    #[test]
    fn test_bytes_per_element() {
        assert_eq!(WeightFormat::F32.bytes_per_element(), 4.0);
        assert_eq!(WeightFormat::Int4.bytes_per_element(), 0.5);
    }
}
