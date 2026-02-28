//! GPU inference pipeline integration tests.
//!
//! Validates the full GPU inference pipeline end-to-end using mock backends
//! for CI compatibility. Tests cover pipeline construction, backend selection,
//! memory management, error recovery, and cross-backend consistency.

mod backend_selection_tests;
mod error_recovery_tests;
mod memory_management_tests;
mod multi_backend_tests;
mod pipeline_tests;

use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, InferenceError};
use bitnet_models::Model;

// ---------------------------------------------------------------------------
// Shared mock model used across all sub-modules
// ---------------------------------------------------------------------------

/// Mock model for integration tests — returns deterministic tensors.
struct MockModel {
    config: BitNetConfig,
    should_fail: bool,
}

impl MockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default(), should_fail: false }
    }

    fn with_config(config: BitNetConfig) -> Self {
        Self { config, should_fail: false }
    }

    fn with_failure() -> Self {
        Self { config: BitNetConfig::default(), should_fail: true }
    }
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        if self.should_fail {
            return Err(BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Mock model failure".into(),
            }));
        }
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }

    fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        if self.should_fail {
            return Err(BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Mock embed failure".into(),
            }));
        }
        let seq_len = tokens.len();
        Ok(ConcreteTensor::mock(vec![seq_len, self.config.model.hidden_size]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        if self.should_fail {
            return Err(BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Mock logits failure".into(),
            }));
        }
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }
}
