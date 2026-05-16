use bitnet_common::{BitNetError, Result};

/// Model type detection utilities for neural-network tokenizers.
pub struct ModelTypeDetector;

impl ModelTypeDetector {
    /// Detect model type from vocabulary size.
    #[must_use]
    pub fn detect_from_vocab_size(vocab_size: usize) -> String {
        match vocab_size {
            32000 => "llama2".to_string(),
            128256 => "llama3".to_string(),
            32016 => "codellama".to_string(),
            50257 => "gpt2".to_string(),
            100352 => "phi4".to_string(),
            51200 => "phi2".to_string(),
            32064 => "phi3".to_string(),
            152064 => "qwen2.5".to_string(),
            151936 => "qwen2".to_string(),
            256000 => "gemma".to_string(),
            32768 => "mistral-v03".to_string(),
            131072 => "mistral-nemo".to_string(),
            49152 => "smollm".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Check if vocabulary size indicates a model that usually benefits from GPU acceleration.
    #[must_use]
    pub fn requires_gpu_acceleration(vocab_size: usize) -> bool {
        vocab_size > 65536
    }

    /// Validate vocabulary size for tokenizer usage.
    pub fn validate_vocab_size(vocab_size: usize) -> Result<()> {
        if vocab_size == 0 {
            return Err(BitNetError::Config("Vocabulary size cannot be zero".to_string()));
        }

        if vocab_size > 2_000_000 {
            return Err(BitNetError::Config(format!(
                "Vocabulary size {} exceeds reasonable limit (2M)",
                vocab_size
            )));
        }

        Ok(())
    }

    /// Expected vocabulary sizes for common model families.
    #[must_use]
    pub fn expected_vocab_size(model_type: &str) -> Option<usize> {
        match model_type {
            "llama2" => Some(32000),
            "llama3" => Some(128256),
            "codellama" => Some(32016),
            "gpt2" => Some(50257),
            "phi4" => Some(100352),
            "phi3" => Some(32064),
            "phi2" => Some(51200),
            "qwen2.5" => Some(152064),
            "qwen2" => Some(151936),
            "gemma" | "gemma2" => Some(256000),
            "mistral" => Some(32000),
            "mistral-v03" => Some(32768),
            "mistral-nemo" => Some(131072),
            "smollm" | "smollm2" => Some(49152),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelTypeDetector;

    #[test]
    fn detects_known_vocab_sizes() {
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(32000), "llama2");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(128256), "llama3");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(32016), "codellama");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(50257), "gpt2");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(100352), "phi4");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(51200), "phi2");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(32064), "phi3");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(152064), "qwen2.5");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(151936), "qwen2");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(256000), "gemma");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(32768), "mistral-v03");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(131072), "mistral-nemo");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(49152), "smollm");
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(42), "unknown");
    }

    #[test]
    fn expected_vocab_for_slm_families() {
        assert_eq!(ModelTypeDetector::expected_vocab_size("phi4"), Some(100352));
        assert_eq!(ModelTypeDetector::expected_vocab_size("qwen2.5"), Some(152064));
        assert_eq!(ModelTypeDetector::expected_vocab_size("gemma"), Some(256000));
        assert_eq!(ModelTypeDetector::expected_vocab_size("gemma2"), Some(256000));
        assert_eq!(ModelTypeDetector::expected_vocab_size("mistral-v03"), Some(32768));
        assert_eq!(ModelTypeDetector::expected_vocab_size("mistral-nemo"), Some(131072));
        assert_eq!(ModelTypeDetector::expected_vocab_size("smollm"), Some(49152));
        assert_eq!(ModelTypeDetector::expected_vocab_size("smollm2"), Some(49152));
        assert_eq!(ModelTypeDetector::expected_vocab_size("nonexistent"), None);
    }

    #[test]
    fn validates_vocab_bounds() {
        assert!(ModelTypeDetector::validate_vocab_size(1).is_ok());
        assert!(ModelTypeDetector::validate_vocab_size(2_000_000).is_ok());
        assert!(ModelTypeDetector::validate_vocab_size(0).is_err());
        assert!(ModelTypeDetector::validate_vocab_size(2_000_001).is_err());
    }

    #[test]
    fn requires_gpu_acceleration_threshold_is_inclusive_below() {
        // 65536 is the boundary documented as "> 65536"; anything at or below must not trigger.
        assert!(!ModelTypeDetector::requires_gpu_acceleration(0));
        assert!(!ModelTypeDetector::requires_gpu_acceleration(32000));
        assert!(!ModelTypeDetector::requires_gpu_acceleration(65536));
    }

    #[test]
    fn requires_gpu_acceleration_triggers_above_threshold() {
        assert!(ModelTypeDetector::requires_gpu_acceleration(65537));
        assert!(ModelTypeDetector::requires_gpu_acceleration(128256));
        assert!(ModelTypeDetector::requires_gpu_acceleration(256000));
    }

    #[test]
    fn expected_vocab_size_unknown_family() {
        assert_eq!(ModelTypeDetector::expected_vocab_size(""), None);
        assert_eq!(ModelTypeDetector::expected_vocab_size("LLAMA2"), None);
        assert_eq!(ModelTypeDetector::expected_vocab_size("phi"), None);
    }

    #[test]
    fn expected_vocab_size_handles_mistral_aliases() {
        // "mistral" without a version maps to 32000 (the original llama-derived vocab).
        assert_eq!(ModelTypeDetector::expected_vocab_size("mistral"), Some(32000));
    }

    #[test]
    fn validate_vocab_error_messages_carry_size_context() {
        let too_large = ModelTypeDetector::validate_vocab_size(5_000_000).unwrap_err();
        let rendered = format!("{too_large}");
        assert!(
            rendered.contains("5000000"),
            "error should mention the offending size: {rendered}"
        );
    }

    #[test]
    fn detection_and_expected_size_are_round_trip_for_known_families() {
        for family in [
            "llama2",
            "llama3",
            "codellama",
            "gpt2",
            "phi4",
            "phi3",
            "phi2",
            "qwen2.5",
            "qwen2",
            "gemma",
            "mistral-v03",
            "mistral-nemo",
            "smollm",
        ] {
            let vocab = ModelTypeDetector::expected_vocab_size(family)
                .expect("known family should have an expected vocab size");
            assert_eq!(
                ModelTypeDetector::detect_from_vocab_size(vocab),
                family,
                "detection of {family} should round-trip through its expected vocab size",
            );
        }
    }
}
