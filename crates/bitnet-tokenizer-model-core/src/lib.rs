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
        assert_eq!(ModelTypeDetector::detect_from_vocab_size(42), "unknown");
    }

    #[test]
    fn validates_vocab_bounds() {
        assert!(ModelTypeDetector::validate_vocab_size(1).is_ok());
        assert!(ModelTypeDetector::validate_vocab_size(2_000_000).is_ok());
        assert!(ModelTypeDetector::validate_vocab_size(0).is_err());
        assert!(ModelTypeDetector::validate_vocab_size(2_000_001).is_err());
    }
}
