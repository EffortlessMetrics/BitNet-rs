//! Quantization format registry.
//!
//! Registry of supported quantization formats with their capabilities,
//! bit widths, and compression characteristics.

use std::collections::HashMap;

/// Quantization format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// BitNet I2_S (1.58-bit ternary).
    I2S,
    /// Table Lookup 1 (TL1).
    Tl1,
    /// Table Lookup 2 (TL2).
    Tl2,
    /// GGML QK256 (256-element blocks).
    Qk256,
    /// Standard INT4.
    Int4,
    /// Standard INT8.
    Int8,
    /// FP16 (baseline, no quantization).
    Fp16,
    /// BF16.
    Bf16,
    /// FP32.
    Fp32,
}

impl QuantFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I2S => "i2s",
            Self::Tl1 => "tl1",
            Self::Tl2 => "tl2",
            Self::Qk256 => "qk256",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
        }
    }

    pub fn bits_per_element(&self) -> f64 {
        match self {
            Self::I2S => 1.58,
            Self::Tl1 | Self::Tl2 => 2.0,
            Self::Qk256 => 2.0625, // 2 bits + scale overhead
            Self::Int4 => 4.0,
            Self::Int8 => 8.0,
            Self::Fp16 | Self::Bf16 => 16.0,
            Self::Fp32 => 32.0,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::Fp16 | Self::Bf16 | Self::Fp32)
    }

    pub fn compression_vs_fp16(&self) -> f64 {
        16.0 / self.bits_per_element()
    }
}

/// Capabilities of a quantization format.
#[derive(Debug, Clone)]
pub struct FormatCapabilities {
    pub format: QuantFormat,
    pub name: String,
    pub description: String,
    pub has_simd_kernel: bool,
    pub has_gpu_kernel: bool,
    pub production_ready: bool,
}

/// Registry of all supported quantization formats.
#[derive(Debug)]
pub struct FormatRegistry {
    formats: HashMap<QuantFormat, FormatCapabilities>,
}

impl FormatRegistry {
    pub fn new() -> Self {
        let mut reg = Self { formats: HashMap::new() };
        reg.register_defaults();
        reg
    }

    fn register_defaults(&mut self) {
        self.register(FormatCapabilities {
            format: QuantFormat::I2S,
            name: "BitNet I2_S".into(),
            description: "1.58-bit ternary quantization (-1, 0, +1)".into(),
            has_simd_kernel: true,
            has_gpu_kernel: false,
            production_ready: true,
        });
        self.register(FormatCapabilities {
            format: QuantFormat::Tl1,
            name: "Table Lookup 1".into(),
            description: "2-bit table lookup quantization".into(),
            has_simd_kernel: false,
            has_gpu_kernel: false,
            production_ready: false,
        });
        self.register(FormatCapabilities {
            format: QuantFormat::Tl2,
            name: "Table Lookup 2".into(),
            description: "2-bit table lookup quantization (v2)".into(),
            has_simd_kernel: false,
            has_gpu_kernel: false,
            production_ready: false,
        });
        self.register(FormatCapabilities {
            format: QuantFormat::Qk256,
            name: "GGML QK256".into(),
            description: "256-element block quantization with scales".into(),
            has_simd_kernel: true,
            has_gpu_kernel: false,
            production_ready: false,
        });
        self.register(FormatCapabilities {
            format: QuantFormat::Int4,
            name: "INT4".into(),
            description: "4-bit integer quantization".into(),
            has_simd_kernel: false,
            has_gpu_kernel: true,
            production_ready: false,
        });
        self.register(FormatCapabilities {
            format: QuantFormat::Int8,
            name: "INT8".into(),
            description: "8-bit integer quantization".into(),
            has_simd_kernel: false,
            has_gpu_kernel: true,
            production_ready: false,
        });
        self.register(FormatCapabilities {
            format: QuantFormat::Fp16,
            name: "FP16".into(),
            description: "16-bit floating point (baseline)".into(),
            has_simd_kernel: true,
            has_gpu_kernel: true,
            production_ready: true,
        });
    }

    pub fn register(&mut self, caps: FormatCapabilities) {
        self.formats.insert(caps.format, caps);
    }

    pub fn get(&self, format: QuantFormat) -> Option<&FormatCapabilities> {
        self.formats.get(&format)
    }

    pub fn list(&self) -> Vec<&FormatCapabilities> {
        self.formats.values().collect()
    }

    pub fn production_formats(&self) -> Vec<&FormatCapabilities> {
        self.formats.values().filter(|f| f.production_ready).collect()
    }

    pub fn formats_with_simd(&self) -> Vec<&FormatCapabilities> {
        self.formats.values().filter(|f| f.has_simd_kernel).collect()
    }

    pub fn formats_with_gpu(&self) -> Vec<&FormatCapabilities> {
        self.formats.values().filter(|f| f.has_gpu_kernel).collect()
    }

    pub fn count(&self) -> usize {
        self.formats.len()
    }

    pub fn is_supported(&self, format: QuantFormat) -> bool {
        self.formats.contains_key(&format)
    }
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bits() {
        assert!((QuantFormat::I2S.bits_per_element() - 1.58).abs() < 0.01);
        assert!((QuantFormat::Int4.bits_per_element() - 4.0).abs() < 0.01);
        assert!((QuantFormat::Fp16.bits_per_element() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_is_quantized() {
        assert!(QuantFormat::I2S.is_quantized());
        assert!(QuantFormat::Int8.is_quantized());
        assert!(!QuantFormat::Fp16.is_quantized());
        assert!(!QuantFormat::Fp32.is_quantized());
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = QuantFormat::I2S.compression_vs_fp16();
        assert!(ratio > 10.0); // 16/1.58 ≈ 10.1x
    }

    #[test]
    fn test_registry_defaults() {
        let reg = FormatRegistry::new();
        assert!(reg.count() >= 7);
        assert!(reg.is_supported(QuantFormat::I2S));
        assert!(reg.is_supported(QuantFormat::Fp16));
    }

    #[test]
    fn test_get_format() {
        let reg = FormatRegistry::new();
        let caps = reg.get(QuantFormat::I2S).unwrap();
        assert_eq!(caps.name, "BitNet I2_S");
        assert!(caps.production_ready);
    }

    #[test]
    fn test_production_formats() {
        let reg = FormatRegistry::new();
        let prod = reg.production_formats();
        assert!(prod.iter().any(|f| f.format == QuantFormat::I2S));
        assert!(prod.iter().any(|f| f.format == QuantFormat::Fp16));
    }

    #[test]
    fn test_simd_formats() {
        let reg = FormatRegistry::new();
        let simd = reg.formats_with_simd();
        assert!(simd.iter().any(|f| f.format == QuantFormat::I2S));
    }

    #[test]
    fn test_gpu_formats() {
        let reg = FormatRegistry::new();
        let gpu = reg.formats_with_gpu();
        assert!(gpu.iter().any(|f| f.format == QuantFormat::Fp16));
    }

    #[test]
    fn test_format_as_str() {
        assert_eq!(QuantFormat::I2S.as_str(), "i2s");
        assert_eq!(QuantFormat::Qk256.as_str(), "qk256");
        assert_eq!(QuantFormat::Fp16.as_str(), "fp16");
    }

    #[test]
    fn test_custom_registration() {
        let mut reg = FormatRegistry::new();
        reg.register(FormatCapabilities {
            format: QuantFormat::Bf16,
            name: "BF16".into(),
            description: "Brain float 16".into(),
            has_simd_kernel: false,
            has_gpu_kernel: true,
            production_ready: true,
        });
        assert!(reg.is_supported(QuantFormat::Bf16));
    }

    #[test]
    fn test_list_all() {
        let reg = FormatRegistry::new();
        let all = reg.list();
        assert_eq!(all.len(), reg.count());
    }

    #[test]
    fn test_unsupported() {
        let reg = FormatRegistry::new();
        assert!(!reg.is_supported(QuantFormat::Bf16)); // not registered by default
    }
}
