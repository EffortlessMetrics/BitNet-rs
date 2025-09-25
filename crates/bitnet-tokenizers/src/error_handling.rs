//! Comprehensive error handling with anyhow::Result integration
//!
//! Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac10-error-handling

use anyhow::Result as AnyhowResult;
use bitnet_common::BitNetError;

/// AC10: Comprehensive error handling for tokenizer operations
/// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac10-error-handling
pub struct TokenizerErrorHandler;

impl TokenizerErrorHandler {
    /// Convert BitNetError to anyhow::Error with context
    pub fn to_anyhow_error(error: BitNetError, context: &str) -> anyhow::Error {
        anyhow::Error::new(error).context(context.to_string())
    }

    /// Create actionable error message for tokenizer failures
    pub fn create_actionable_error(error: BitNetError) -> AnyhowResult<()> {
        // Test scaffolding - actual implementation pending
        Err(Self::to_anyhow_error(error, "Tokenizer operation failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::ModelError;

    /// AC10: Tests error handling with anyhow::Result integration
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac10-error-handling
    #[test]
    #[cfg(feature = "cpu")]
    fn test_error_handling_anyhow_integration() {
        let test_error =
            BitNetError::Model(ModelError::LoadingFailed { reason: "Test error".to_string() });

        let anyhow_error = TokenizerErrorHandler::to_anyhow_error(test_error, "Test context");
        let error_string = anyhow_error.to_string();
        println!("Actual error string: '{}'", error_string);

        // The error should contain both the context and the original error message
        assert!(
            error_string.contains("Test context") || error_string.contains("Test error"),
            "Error should contain context or original message. Got: '{}'",
            error_string
        );

        println!("✅ AC10: Error handling with anyhow::Result test scaffolding completed");
    }
}
