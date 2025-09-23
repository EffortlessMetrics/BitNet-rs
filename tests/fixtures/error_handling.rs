//! Error handling test fixtures and failure scenarios
//!
//! Provides comprehensive error testing infrastructure including failure simulation,
//! recovery validation, and error message quality assessment for BitNet.rs components.

use bitnet_common::{Device, Result, BitNetError, ModelError, InferenceError, KernelError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Comprehensive failure scenarios for error testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureScenarios {
    pub model_loading_failures: Vec<ModelLoadingFailure>,
    pub inference_failures: Vec<InferenceFailure>,
    pub quantization_failures: Vec<QuantizationFailure>,
    pub device_failures: Vec<DeviceFailure>,
    pub resource_failures: Vec<ResourceFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadingFailure {
    pub name: String,
    pub failure_type: String,
    pub trigger_condition: String,
    pub expected_error: String,
    pub recovery_suggestion: String,
    pub test_file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceFailure {
    pub name: String,
    pub failure_type: String,
    pub input_condition: String,
    pub expected_error: String,
    pub recovery_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationFailure {
    pub name: String,
    pub quantization_type: String,
    pub failure_condition: String,
    pub expected_error: String,
    pub fallback_behavior: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFailure {
    pub name: String,
    pub device_type: String,
    pub failure_condition: String,
    pub expected_error: String,
    pub fallback_device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFailure {
    pub name: String,
    pub resource_type: String,
    pub failure_condition: String,
    pub expected_error: String,
    pub mitigation_strategy: String,
}

/// Error testing configuration
#[derive(Debug, Clone)]
pub struct ErrorTestConfig {
    pub simulate_network_failures: bool,
    pub simulate_memory_pressure: bool,
    pub simulate_device_unavailability: bool,
    pub error_injection_rate: f32,
    pub recovery_timeout: std::time::Duration,
}

/// Error handling fixtures manager
pub struct ErrorHandlingFixtures {
    pub failure_scenarios: FailureScenarios,
    pub test_files: HashMap<String, PathBuf>,
    pub config: ErrorTestConfig,
}

impl ErrorHandlingFixtures {
    pub fn new() -> Self {
        Self {
            failure_scenarios: Self::create_failure_scenarios(),
            test_files: HashMap::new(),
            config: ErrorTestConfig {
                simulate_network_failures: true,
                simulate_memory_pressure: false, // Disabled by default for CI
                simulate_device_unavailability: true,
                error_injection_rate: 0.1, // 10% injection rate
                recovery_timeout: std::time::Duration::from_secs(5),
            },
        }
    }

    /// Create comprehensive failure scenarios
    fn create_failure_scenarios() -> FailureScenarios {
        let model_loading_failures = vec![
            ModelLoadingFailure {
                name: "missing_model_file".to_string(),
                failure_type: "FileNotFound".to_string(),
                trigger_condition: "Model file does not exist".to_string(),
                expected_error: "Model not found".to_string(),
                recovery_suggestion: "Check file path or download model using xtask".to_string(),
                test_file_path: Some("/nonexistent/model.gguf".to_string()),
            },
            ModelLoadingFailure {
                name: "corrupted_gguf_header".to_string(),
                failure_type: "GGUFFormatError".to_string(),
                trigger_condition: "Invalid GGUF magic number".to_string(),
                expected_error: "Invalid GGUF format".to_string(),
                recovery_suggestion: "Re-download model or check file integrity".to_string(),
                test_file_path: None, // Will be generated
            },
            ModelLoadingFailure {
                name: "unsupported_quantization".to_string(),
                failure_type: "UnsupportedQuantization".to_string(),
                trigger_condition: "Model uses unsupported quantization type".to_string(),
                expected_error: "Unsupported quantization type".to_string(),
                recovery_suggestion: "Use supported quantization format or enable FFI".to_string(),
                test_file_path: None,
            },
            ModelLoadingFailure {
                name: "insufficient_memory".to_string(),
                failure_type: "InsufficientMemory".to_string(),
                trigger_condition: "Model size exceeds available memory".to_string(),
                expected_error: "Insufficient memory to load model".to_string(),
                recovery_suggestion: "Free memory or use memory-mapped loading".to_string(),
                test_file_path: None,
            },
            ModelLoadingFailure {
                name: "tensor_alignment_error".to_string(),
                failure_type: "TensorAlignmentError".to_string(),
                trigger_condition: "Tensor data not aligned to 32-byte boundary".to_string(),
                expected_error: "Tensor alignment validation failed".to_string(),
                recovery_suggestion: "Re-export model with proper alignment".to_string(),
                test_file_path: None,
            },
        ];

        let inference_failures = vec![
            InferenceFailure {
                name: "context_length_exceeded".to_string(),
                failure_type: "ContextLengthExceeded".to_string(),
                input_condition: "Input tokens exceed model context length".to_string(),
                expected_error: "Context length exceeded".to_string(),
                recovery_strategy: "Truncate input or use sliding window".to_string(),
            },
            InferenceFailure {
                name: "invalid_token_ids".to_string(),
                failure_type: "InvalidInput".to_string(),
                input_condition: "Token IDs outside vocabulary range".to_string(),
                expected_error: "Invalid token IDs in input".to_string(),
                recovery_strategy: "Validate tokenization output".to_string(),
            },
            InferenceFailure {
                name: "tokenization_failure".to_string(),
                failure_type: "TokenizationFailed".to_string(),
                input_condition: "Tokenizer fails on malformed input".to_string(),
                expected_error: "Tokenization failed".to_string(),
                recovery_strategy: "Preprocess input text or use fallback tokenizer".to_string(),
            },
            InferenceFailure {
                name: "generation_timeout".to_string(),
                failure_type: "GenerationTimeout".to_string(),
                input_condition: "Generation exceeds maximum time limit".to_string(),
                expected_error: "Generation timed out".to_string(),
                recovery_strategy: "Reduce max tokens or increase timeout".to_string(),
            },
        ];

        let quantization_failures = vec![
            QuantizationFailure {
                name: "invalid_block_size".to_string(),
                quantization_type: "I2_S".to_string(),
                failure_condition: "Block size not multiple of required alignment".to_string(),
                expected_error: "Invalid block size".to_string(),
                fallback_behavior: "Use default block size".to_string(),
            },
            QuantizationFailure {
                name: "quantization_overflow".to_string(),
                quantization_type: "TL1".to_string(),
                failure_condition: "Values exceed quantization range".to_string(),
                expected_error: "Quantization overflow".to_string(),
                fallback_behavior: "Clamp values to valid range".to_string(),
            },
            QuantizationFailure {
                name: "lookup_table_corruption".to_string(),
                quantization_type: "TL2".to_string(),
                failure_condition: "Lookup table data is corrupted".to_string(),
                expected_error: "Lookup table validation failed".to_string(),
                fallback_behavior: "Regenerate lookup table".to_string(),
            },
        ];

        let device_failures = vec![
            DeviceFailure {
                name: "cuda_initialization_failure".to_string(),
                device_type: "GPU".to_string(),
                failure_condition: "CUDA driver not available or outdated".to_string(),
                expected_error: "CUDA initialization failed".to_string(),
                fallback_device: "CPU".to_string(),
            },
            DeviceFailure {
                name: "gpu_memory_exhausted".to_string(),
                device_type: "GPU".to_string(),
                failure_condition: "Insufficient GPU memory".to_string(),
                expected_error: "GPU memory allocation failed".to_string(),
                fallback_device: "CPU".to_string(),
            },
            DeviceFailure {
                name: "kernel_launch_failure".to_string(),
                device_type: "GPU".to_string(),
                failure_condition: "GPU kernel fails to launch".to_string(),
                expected_error: "Kernel execution failed".to_string(),
                fallback_device: "CPU".to_string(),
            },
        ];

        let resource_failures = vec![
            ResourceFailure {
                name: "disk_space_exhausted".to_string(),
                resource_type: "Storage".to_string(),
                failure_condition: "Insufficient disk space for model cache".to_string(),
                expected_error: "Disk space exhausted".to_string(),
                mitigation_strategy: "Clear cache or use streaming".to_string(),
            },
            ResourceFailure {
                name: "network_timeout".to_string(),
                resource_type: "Network".to_string(),
                failure_condition: "Model download timeout".to_string(),
                expected_error: "Network operation timed out".to_string(),
                mitigation_strategy: "Retry with exponential backoff".to_string(),
            },
            ResourceFailure {
                name: "file_permission_denied".to_string(),
                resource_type: "FileSystem".to_string(),
                failure_condition: "Insufficient permissions to read model file".to_string(),
                expected_error: "Permission denied".to_string(),
                mitigation_strategy: "Check file permissions or run with appropriate privileges".to_string(),
            },
        ];

        FailureScenarios {
            model_loading_failures,
            inference_failures,
            quantization_failures,
            device_failures,
            resource_failures,
        }
    }

    /// Initialize error handling fixtures
    pub async fn initialize(&mut self) -> Result<()> {
        // Create test files for error scenarios
        self.create_test_files().await?;

        // Validate error scenarios
        self.validate_error_scenarios().await?;

        Ok(())
    }

    /// Create test files for error simulation
    async fn create_test_files(&mut self) -> Result<()> {
        // Create corrupted GGUF file
        let corrupted_gguf = self.create_corrupted_gguf_file()?;
        self.test_files.insert("corrupted_gguf".to_string(), corrupted_gguf);

        // Create misaligned tensor file
        let misaligned_tensor = self.create_misaligned_tensor_file()?;
        self.test_files.insert("misaligned_tensor".to_string(), misaligned_tensor);

        // Create oversized model file (for memory tests)
        let oversized_model = self.create_oversized_model_file()?;
        self.test_files.insert("oversized_model".to_string(), oversized_model);

        // Create permission-denied file (Unix only)
        #[cfg(unix)]
        {
            let restricted_file = self.create_restricted_file()?;
            self.test_files.insert("restricted_file".to_string(), restricted_file);
        }

        Ok(())
    }

    /// Create corrupted GGUF file for testing
    fn create_corrupted_gguf_file(&self) -> Result<PathBuf> {
        let temp_file = NamedTempFile::new().map_err(BitNetError::Io)?;

        // Write invalid GGUF magic number
        std::fs::write(&temp_file, b"BADF00D").map_err(BitNetError::Io)?;

        // Convert to permanent path
        let path = temp_file.into_temp_path();
        Ok(path.keep().map_err(BitNetError::Io)?)
    }

    /// Create GGUF file with misaligned tensors
    fn create_misaligned_tensor_file(&self) -> Result<PathBuf> {
        let temp_file = NamedTempFile::new().map_err(BitNetError::Io)?;

        let mut buffer = Vec::new();

        // Valid GGUF header
        buffer.extend_from_slice(b"GGUF");        // Magic
        buffer.extend_from_slice(&3u32.to_le_bytes()); // Version
        buffer.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
        buffer.extend_from_slice(&0u64.to_le_bytes()); // KV count

        // Tensor info with misaligned offset
        let tensor_name = b"test.weight";
        buffer.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buffer.extend_from_slice(tensor_name);

        buffer.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buffer.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
        buffer.extend_from_slice(&4u64.to_le_bytes()); // dim[1]
        buffer.extend_from_slice(&0u32.to_le_bytes()); // type (F32)

        // Misaligned offset (not multiple of 32)
        let misaligned_offset = buffer.len() as u64 + 17; // Intentionally misaligned
        buffer.extend_from_slice(&misaligned_offset.to_le_bytes());

        // Padding
        buffer.resize(buffer.len() + 50, 0);

        std::fs::write(&temp_file, buffer).map_err(BitNetError::Io)?;

        let path = temp_file.into_temp_path();
        Ok(path.keep().map_err(BitNetError::Io)?)
    }

    /// Create oversized model file for memory testing
    fn create_oversized_model_file(&self) -> Result<PathBuf> {
        let temp_file = NamedTempFile::new().map_err(BitNetError::Io)?;

        // Create sparse file that appears large but doesn't consume disk space
        let file = std::fs::File::create(&temp_file).map_err(BitNetError::Io)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            // Write a single byte at a very large offset to create sparse file
            file.write_at(b"X", 1024 * 1024 * 1024 * 8) // 8GB offset
                .map_err(BitNetError::Io)?;
        }

        #[cfg(not(unix))]
        {
            // On non-Unix, just create a regular large file (may be slow)
            use std::io::Write;
            let mut writer = std::io::BufWriter::new(file);
            let chunk = vec![0u8; 1024 * 1024]; // 1MB chunk
            for _ in 0..10 { // Write 10MB (smaller for Windows)
                writer.write_all(&chunk).map_err(BitNetError::Io)?;
            }
            writer.flush().map_err(BitNetError::Io)?;
        }

        let path = temp_file.into_temp_path();
        Ok(path.keep().map_err(BitNetError::Io)?)
    }

    /// Create restricted file (Unix only)
    #[cfg(unix)]
    fn create_restricted_file(&self) -> Result<PathBuf> {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;

        let temp_file = NamedTempFile::new().map_err(BitNetError::Io)?;
        std::fs::write(&temp_file, b"restricted content").map_err(BitNetError::Io)?;

        // Remove all permissions
        let perms = Permissions::from_mode(0o000);
        std::fs::set_permissions(&temp_file, perms).map_err(BitNetError::Io)?;

        let path = temp_file.into_temp_path();
        Ok(path.keep().map_err(BitNetError::Io)?)
    }

    /// Validate error scenarios configuration
    async fn validate_error_scenarios(&self) -> Result<()> {
        let total_scenarios = self.failure_scenarios.model_loading_failures.len()
            + self.failure_scenarios.inference_failures.len()
            + self.failure_scenarios.quantization_failures.len()
            + self.failure_scenarios.device_failures.len()
            + self.failure_scenarios.resource_failures.len();

        if total_scenarios == 0 {
            return Err(BitNetError::Validation(
                "No error scenarios defined".to_string()
            ));
        }

        // Validate that each scenario has required fields
        for scenario in &self.failure_scenarios.model_loading_failures {
            if scenario.expected_error.is_empty() {
                return Err(BitNetError::Validation(
                    format!("Missing expected error for scenario: {}", scenario.name)
                ));
            }
        }

        println!("Validated {} error scenarios", total_scenarios);
        Ok(())
    }

    /// Simulate model loading failure
    pub fn simulate_model_loading_failure(&self, failure_name: &str) -> Result<BitNetError> {
        let failure = self.failure_scenarios.model_loading_failures
            .iter()
            .find(|f| f.name == failure_name)
            .ok_or_else(|| BitNetError::Validation(
                format!("Unknown failure scenario: {}", failure_name)
            ))?;

        let error = match failure.failure_type.as_str() {
            "FileNotFound" => BitNetError::Model(ModelError::NotFound {
                path: failure.test_file_path.clone().unwrap_or_default()
            }),
            "GGUFFormatError" => BitNetError::Model(ModelError::InvalidFormat {
                format: "Corrupted GGUF file".to_string()
            }),
            "UnsupportedQuantization" => BitNetError::Model(ModelError::UnsupportedVersion {
                version: "Unsupported quantization type".to_string()
            }),
            "InsufficientMemory" => BitNetError::Model(ModelError::LoadingFailed {
                reason: "Insufficient memory".to_string()
            }),
            "TensorAlignmentError" => BitNetError::Model(ModelError::InvalidFormat {
                format: "Tensor alignment validation failed".to_string()
            }),
            _ => BitNetError::Validation(
                format!("Unknown failure type: {}", failure.failure_type)
            ),
        };

        Ok(error)
    }

    /// Simulate inference failure
    pub fn simulate_inference_failure(&self, failure_name: &str) -> Result<BitNetError> {
        let failure = self.failure_scenarios.inference_failures
            .iter()
            .find(|f| f.name == failure_name)
            .ok_or_else(|| BitNetError::Validation(
                format!("Unknown inference failure: {}", failure_name)
            ))?;

        let error = match failure.failure_type.as_str() {
            "ContextLengthExceeded" => BitNetError::Inference(InferenceError::ContextLengthExceeded {
                length: 4096
            }),
            "InvalidInput" => BitNetError::Inference(InferenceError::InvalidInput {
                reason: failure.input_condition.clone()
            }),
            "TokenizationFailed" => BitNetError::Inference(InferenceError::TokenizationFailed {
                reason: failure.input_condition.clone()
            }),
            "GenerationTimeout" => BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Generation timed out".to_string()
            }),
            _ => BitNetError::Validation(
                format!("Unknown inference failure type: {}", failure.failure_type)
            ),
        };

        Ok(error)
    }

    /// Simulate device failure
    pub fn simulate_device_failure(&self, failure_name: &str) -> Result<BitNetError> {
        let failure = self.failure_scenarios.device_failures
            .iter()
            .find(|f| f.name == failure_name)
            .ok_or_else(|| BitNetError::Validation(
                format!("Unknown device failure: {}", failure_name)
            ))?;

        let error = BitNetError::Kernel(KernelError::GpuError {
            reason: failure.expected_error.clone()
        });

        Ok(error)
    }

    /// Test error message quality
    pub fn test_error_message_quality(&self, error: &BitNetError) -> ErrorMessageQuality {
        let error_str = format!("{}", error);
        let error_debug = format!("{:?}", error);

        ErrorMessageQuality {
            has_user_friendly_message: !error_str.contains("Error(") && error_str.len() > 10,
            has_technical_details: error_debug.contains("reason") || error_debug.contains("path"),
            includes_recovery_suggestion: error_str.to_lowercase().contains("try") ||
                                         error_str.to_lowercase().contains("check") ||
                                         error_str.to_lowercase().contains("ensure"),
            message_length: error_str.len(),
            clarity_score: self.calculate_clarity_score(&error_str),
        }
    }

    /// Calculate error message clarity score
    fn calculate_clarity_score(&self, message: &str) -> f32 {
        let mut score = 0.5; // Base score

        // Positive indicators
        if message.contains("not found") || message.contains("missing") {
            score += 0.2;
        }
        if message.contains("check") || message.contains("verify") {
            score += 0.1;
        }
        if message.len() > 20 && message.len() < 200 {
            score += 0.1;
        }

        // Negative indicators
        if message.contains("unknown error") || message.contains("unexpected") {
            score -= 0.2;
        }
        if message.len() < 10 {
            score -= 0.3;
        }

        score.min(1.0).max(0.0)
    }

    /// Get test file path for error scenario
    pub fn get_test_file(&self, file_key: &str) -> Option<&PathBuf> {
        self.test_files.get(file_key)
    }

    /// Cleanup error handling fixtures
    pub async fn cleanup(&mut self) -> Result<()> {
        // Remove all test files
        for (_, path) in &self.test_files {
            if path.exists() {
                let _ = tokio::fs::remove_file(path).await;
            }
        }
        self.test_files.clear();

        Ok(())
    }
}

/// Error message quality assessment
#[derive(Debug)]
pub struct ErrorMessageQuality {
    pub has_user_friendly_message: bool,
    pub has_technical_details: bool,
    pub includes_recovery_suggestion: bool,
    pub message_length: usize,
    pub clarity_score: f32,
}

/// Error recovery test result
#[derive(Debug)]
pub struct ErrorRecoveryResult {
    pub scenario_name: String,
    pub error_triggered: bool,
    pub recovery_attempted: bool,
    pub recovery_successful: bool,
    pub recovery_time_ms: u64,
    pub error_message_quality: ErrorMessageQuality,
}