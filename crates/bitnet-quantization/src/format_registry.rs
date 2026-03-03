//! Quantization format registry.
//!
//! Registry of all supported quantization formats with metadata.

/// Quantization format descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantFormat {
    pub name: String,
    pub bits: u8,
    pub block_size: usize,
    pub has_scale: bool,
    pub has_zero_point: bool,
    pub symmetric: bool,
    pub description: String,
}

impl QuantFormat {
    pub fn bytes_per_block(&self) -> usize {
        let data_bits = self.bits as usize * self.block_size;
        let scale_bits = if self.has_scale { 16 } else { 0 }; // f16 scale
        let zp_bits = if self.has_zero_point { 8 } else { 0 };
        (data_bits + scale_bits + zp_bits).div_ceil(8)
    }

    pub fn compression_ratio_vs_f32(&self) -> f64 {
        32.0 / self.bits as f64
    }

    pub fn is_sub_byte(&self) -> bool {
        self.bits < 8
    }
}

/// Registry of quantization formats.
#[derive(Debug, Clone)]
pub struct FormatRegistry {
    formats: Vec<QuantFormat>,
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatRegistry {
    pub fn new() -> Self {
        Self { formats: Vec::new() }
    }

    /// Build with all known formats.
    pub fn builtin() -> Self {
        let mut reg = Self::new();

        reg.add(QuantFormat {
            name: "I2_S".into(),
            bits: 2,
            block_size: 32,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "BitNet 1.58-bit ternary (I2_S encoding)".into(),
        });

        reg.add(QuantFormat {
            name: "QK256".into(),
            bits: 2,
            block_size: 256,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "GGML QK256 256-element ternary blocks".into(),
        });

        reg.add(QuantFormat {
            name: "TL1".into(),
            bits: 2,
            block_size: 32,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "Table lookup v1 quantization".into(),
        });

        reg.add(QuantFormat {
            name: "TL2".into(),
            bits: 2,
            block_size: 32,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "Table lookup v2 quantization".into(),
        });

        reg.add(QuantFormat {
            name: "Q4_0".into(),
            bits: 4,
            block_size: 32,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "GGML 4-bit symmetric quantization".into(),
        });

        reg.add(QuantFormat {
            name: "Q4_1".into(),
            bits: 4,
            block_size: 32,
            has_scale: true,
            has_zero_point: true,
            symmetric: false,
            description: "GGML 4-bit asymmetric quantization".into(),
        });

        reg.add(QuantFormat {
            name: "Q8_0".into(),
            bits: 8,
            block_size: 32,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "GGML 8-bit symmetric quantization".into(),
        });

        reg.add(QuantFormat {
            name: "F16".into(),
            bits: 16,
            block_size: 1,
            has_scale: false,
            has_zero_point: false,
            symmetric: true,
            description: "IEEE half-precision float".into(),
        });

        reg.add(QuantFormat {
            name: "BF16".into(),
            bits: 16,
            block_size: 1,
            has_scale: false,
            has_zero_point: false,
            symmetric: true,
            description: "Brain float 16-bit".into(),
        });

        reg
    }

    pub fn add(&mut self, format: QuantFormat) {
        self.formats.push(format);
    }

    pub fn get(&self, name: &str) -> Option<&QuantFormat> {
        self.formats.iter().find(|f| f.name == name)
    }

    pub fn all(&self) -> &[QuantFormat] {
        &self.formats
    }

    pub fn count(&self) -> usize {
        self.formats.len()
    }

    pub fn by_bits(&self, bits: u8) -> Vec<&QuantFormat> {
        self.formats.iter().filter(|f| f.bits == bits).collect()
    }

    pub fn sub_byte_formats(&self) -> Vec<&QuantFormat> {
        self.formats.iter().filter(|f| f.is_sub_byte()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.formats.iter().map(|f| f.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_count() {
        let reg = FormatRegistry::builtin();
        assert!(reg.count() >= 9);
    }

    #[test]
    fn test_get_i2s() {
        let reg = FormatRegistry::builtin();
        let fmt = reg.get("I2_S").unwrap();
        assert_eq!(fmt.bits, 2);
        assert!(fmt.symmetric);
    }

    #[test]
    fn test_get_missing() {
        let reg = FormatRegistry::builtin();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_by_bits() {
        let reg = FormatRegistry::builtin();
        let two_bit = reg.by_bits(2);
        assert!(two_bit.len() >= 3); // I2_S, QK256, TL1, TL2
    }

    #[test]
    fn test_sub_byte() {
        let reg = FormatRegistry::builtin();
        let sub = reg.sub_byte_formats();
        assert!(sub.len() >= 5); // 2-bit and 4-bit formats
    }

    #[test]
    fn test_compression_ratio() {
        let reg = FormatRegistry::builtin();
        let i2 = reg.get("I2_S").unwrap();
        assert!((i2.compression_ratio_vs_f32() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_bytes_per_block() {
        let reg = FormatRegistry::builtin();
        let q8 = reg.get("Q8_0").unwrap();
        // 8 bits * 32 + 16 (scale) = 272 bits = 34 bytes
        assert_eq!(q8.bytes_per_block(), 34);
    }

    #[test]
    fn test_f16_not_sub_byte() {
        let reg = FormatRegistry::builtin();
        let f16 = reg.get("F16").unwrap();
        assert!(!f16.is_sub_byte());
    }

    #[test]
    fn test_names() {
        let reg = FormatRegistry::builtin();
        let names = reg.names();
        assert!(names.contains(&"I2_S"));
        assert!(names.contains(&"Q4_0"));
    }

    #[test]
    fn test_add_custom() {
        let mut reg = FormatRegistry::new();
        reg.add(QuantFormat {
            name: "custom".into(),
            bits: 3,
            block_size: 16,
            has_scale: true,
            has_zero_point: false,
            symmetric: true,
            description: "test".into(),
        });
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_q4_asymmetric() {
        let reg = FormatRegistry::builtin();
        let q4_1 = reg.get("Q4_1").unwrap();
        assert!(!q4_1.symmetric);
        assert!(q4_1.has_zero_point);
    }

    #[test]
    fn test_bf16_format() {
        let reg = FormatRegistry::builtin();
        let bf16 = reg.get("BF16").unwrap();
        assert_eq!(bf16.block_size, 1);
        assert!(!bf16.has_scale);
    }
}
