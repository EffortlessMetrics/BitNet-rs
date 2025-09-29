//! Strict mode enforcement for BitNet.rs
//!
//! This module provides strict mode functionality to prevent mock fallbacks
//! and ensure real quantized computation is used throughout the inference pipeline.

use crate::{BitNetError, Result};
use std::env;
use std::sync::OnceLock;

/// Global strict mode configuration
static STRICT_MODE_CONFIG: OnceLock<StrictModeConfig> = OnceLock::new();

/// Strict mode configuration
#[derive(Debug, Clone, PartialEq)]
pub struct StrictModeConfig {
    pub enabled: bool,
    pub fail_on_mock: bool,
    pub require_quantization: bool,
    pub validate_performance: bool,
    pub ci_enhanced_mode: bool,
    pub log_all_validations: bool,
    pub fail_fast_on_any_mock: bool,
}

impl StrictModeConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let enabled = env::var("BITNET_STRICT_MODE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Self {
            enabled,
            fail_on_mock: enabled,
            require_quantization: enabled,
            validate_performance: enabled,
            ci_enhanced_mode: false,
            log_all_validations: false,
            fail_fast_on_any_mock: false,
        }
    }

    /// Create detailed configuration from environment variables
    pub fn from_env_detailed() -> Self {
        let base_enabled = env::var("BITNET_STRICT_MODE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Self {
            enabled: base_enabled,
            fail_on_mock: env::var("BITNET_STRICT_FAIL_ON_MOCK")
                .map(|v| v == "1")
                .unwrap_or(base_enabled),
            require_quantization: env::var("BITNET_STRICT_REQUIRE_QUANTIZATION")
                .map(|v| v == "1")
                .unwrap_or(base_enabled),
            validate_performance: env::var("BITNET_STRICT_VALIDATE_PERFORMANCE")
                .map(|v| v == "1")
                .unwrap_or(base_enabled),
            ci_enhanced_mode: false,
            log_all_validations: false,
            fail_fast_on_any_mock: false,
        }
    }

    /// Create configuration with CI enhancements
    pub fn from_env_with_ci_enhancements() -> Self {
        let mut config = Self::from_env_detailed();

        if env::var("CI").is_ok()
            && env::var("BITNET_CI_ENHANCED_STRICT").unwrap_or_default() == "1"
        {
            config.ci_enhanced_mode = true;
            config.log_all_validations = true;
            config.fail_fast_on_any_mock = true;
        }

        config
    }

    /// Validate inference path for mock usage
    pub fn validate_inference_path(&self, path: &MockInferencePath) -> Result<()> {
        if self.enabled && self.fail_on_mock && path.uses_mock_computation {
            return Err(BitNetError::StrictMode(format!(
                "Strict mode: Mock computation detected in inference path: {}",
                path.description
            )));
        }
        Ok(())
    }

    /// Validate kernel availability
    pub fn validate_kernel_availability(&self, scenario: &MissingKernelScenario) -> Result<()> {
        if self.enabled && self.require_quantization && scenario.fallback_available {
            return Err(BitNetError::StrictMode(format!(
                "Strict mode: Required quantization kernel not available: {:?} on {:?}",
                scenario.quantization_type, scenario.device
            )));
        }
        Ok(())
    }

    /// Validate performance metrics for suspicious values
    pub fn validate_performance_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        if self.enabled && self.validate_performance {
            if metrics.computation_type == ComputationType::Mock {
                return Err(BitNetError::StrictMode(
                    "Strict mode: Mock computation detected in performance metrics".to_string(),
                ));
            }

            if metrics.tokens_per_second > 150.0 {
                return Err(BitNetError::StrictMode(format!(
                    "Strict mode: Suspicious performance detected: {:.2} tok/s",
                    metrics.tokens_per_second
                )));
            }
        }
        Ok(())
    }
}

/// Strict mode enforcer for cross-crate consistency
#[derive(Debug)]
pub struct StrictModeEnforcer {
    config: StrictModeConfig,
}

impl StrictModeEnforcer {
    /// Create a new strict mode enforcer
    pub fn new() -> Self {
        Self::with_config(None)
    }

    /// Create enforcer with detailed configuration
    pub fn new_detailed() -> Self {
        let config = STRICT_MODE_CONFIG.get_or_init(StrictModeConfig::from_env_detailed).clone();
        Self { config }
    }

    /// Create enforcer with optional custom configuration (for testing)
    pub fn with_config(config: Option<StrictModeConfig>) -> Self {
        let config = config
            .unwrap_or_else(|| STRICT_MODE_CONFIG.get_or_init(StrictModeConfig::from_env).clone());
        Self { config }
    }

    /// Create enforcer with fresh environment reading (bypasses OnceLock for testing)
    #[cfg(test)]
    pub fn new_fresh() -> Self {
        let config = StrictModeConfig::from_env();
        Self { config }
    }

    /// Check if strict mode is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the strict mode configuration
    pub fn get_config(&self) -> &StrictModeConfig {
        &self.config
    }

    /// Validate an inference path
    pub fn validate_inference_path(&self, path: &MockInferencePath) -> Result<()> {
        self.config.validate_inference_path(path)
    }

    /// Validate kernel availability
    pub fn validate_kernel_availability(&self, scenario: &MissingKernelScenario) -> Result<()> {
        self.config.validate_kernel_availability(scenario)
    }

    /// Validate performance metrics
    pub fn validate_performance_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        self.config.validate_performance_metrics(metrics)
    }
}

impl Default for StrictModeEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock inference path for validation
#[derive(Debug, Clone)]
pub struct MockInferencePath {
    pub description: String,
    pub uses_mock_computation: bool,
    pub fallback_reason: String,
}

/// Missing kernel scenario for validation
#[derive(Debug, Clone)]
pub struct MissingKernelScenario {
    pub quantization_type: crate::QuantizationType,
    pub device: crate::Device,
    pub fallback_available: bool,
}

/// Performance metrics for validation
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PerformanceMetrics {
    #[serde(default)]
    pub tokens_per_second: f64,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub memory_usage_mb: f64,
    #[serde(default)]
    pub computation_type: ComputationType,
    #[serde(default)]
    pub gpu_utilization: Option<f64>,
}

/// Computation type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputationType {
    #[default]
    Real,
    Mock,
}
