# GGUF Compatibility Requirements and Tensor Validation Approach

## Overview

This specification defines comprehensive GGUF format compatibility requirements for real BitNet model integration, including enhanced tensor validation, alignment verification, and cross-platform consistency for production-ready neural network inference.

## GGUF Format Compatibility Requirements

### 1. Core GGUF Format Support

**Supported GGUF Versions**:
- GGUF v3: Primary support for BitNet models
- GGUF v2: Legacy compatibility for existing models
- Future compatibility: Design for GGUF v4+ extensibility

**BitNet-Specific Extensions**:
```rust
// Enhanced GGUF metadata for BitNet models
pub struct BitNetGGUFMetadata {
    pub general: GeneralMetadata,
    pub bitnet: BitNetSpecificMetadata,
    pub quantization: QuantizationMetadata,
    pub tokenizer: TokenizerMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitNetSpecificMetadata {
    pub version: String,                    // BitNet implementation version
    pub group_size: u32,                    // Quantization group size
    pub activation_dtype: String,           // Activation data type
    pub weight_dtype: String,               // Weight data type
    pub architecture_variant: String,       // BitNet-1.58, BitNet-b1.58, etc.
    pub training_precision: String,         // Original training precision
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    pub format: String,                     // "i2_s", "tl1", "tl2"
    pub block_size: u32,                    // Quantization block size
    pub scale_dtype: String,                // Scale factor data type
    pub offset_dtype: Option<String>,       // Zero-point data type
    pub lookup_table_size: Option<u32>,     // For TL1/TL2 quantization
}
```

### 2. Tensor Layout and Alignment Requirements

**Mandatory Alignment**:
- **Header Alignment**: 32-byte boundary for GGUF header
- **Tensor Data Alignment**: 32-byte boundary for all tensor data
- **Metadata Alignment**: 8-byte boundary for metadata entries
- **Cross-Platform Consistency**: Same alignment across x86_64, ARM64, WebAssembly

**Tensor Organization**:
```rust
// Enhanced tensor descriptor with validation
#[derive(Debug, Clone)]
pub struct ValidatedTensorDescriptor {
    pub name: String,
    pub dims: Vec<u64>,
    pub quantization_type: QuantizationType,
    pub offset: u64,
    pub size: u64,
    pub alignment: u32,
    pub validation_state: TensorValidationState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorValidationState {
    NotValidated,
    Valid,
    InvalidAlignment { offset: u64, required: u32 },
    InvalidSize { declared: u64, calculated: u64 },
    InvalidDimensions { declared: u32, actual: usize },
    InaccessibleData { offset: u64, size: u64, file_size: u64 },
    UnsupportedQuantization(QuantizationType),
}

impl ValidatedTensorDescriptor {
    pub fn validate(&mut self, file_size: u64) -> Result<(), ValidationError> {
        // 1. Alignment validation
        if self.offset % self.alignment as u64 != 0 {
            self.validation_state = TensorValidationState::InvalidAlignment {
                offset: self.offset,
                required: self.alignment,
            };
            return Err(ValidationError::TensorAlignment(self.name.clone()));
        }

        // 2. Size validation
        let calculated_size = self.calculate_tensor_size()?;
        if self.size != calculated_size {
            self.validation_state = TensorValidationState::InvalidSize {
                declared: self.size,
                calculated: calculated_size,
            };
            return Err(ValidationError::SizeMismatch(self.name.clone()));
        }

        // 3. Accessibility validation
        if self.offset + self.size > file_size {
            self.validation_state = TensorValidationState::InaccessibleData {
                offset: self.offset,
                size: self.size,
                file_size,
            };
            return Err(ValidationError::TensorOutOfBounds(self.name.clone()));
        }

        // 4. Quantization format validation
        self.validate_quantization_format()?;

        self.validation_state = TensorValidationState::Valid;
        Ok(())
    }

    fn calculate_tensor_size(&self) -> Result<u64, ValidationError> {
        let element_count: u64 = self.dims.iter().product();
        let element_size = self.quantization_type.element_size_bytes();

        // Account for quantization block overhead
        let size = match self.quantization_type {
            QuantizationType::I2S => {
                // 2 bits per element + scale factor per 32 elements
                let packed_elements = (element_count + 3) / 4; // 4 elements per byte
                let scale_factors = (element_count + 31) / 32; // 1 scale per 32 elements
                packed_elements + scale_factors * 4 // 4 bytes per f32 scale
            }
            QuantizationType::TL1 | QuantizationType::TL2 => {
                // Table lookup + indices
                let table_size = self.quantization_type.table_size();
                let index_bits = self.quantization_type.index_bits();
                let packed_indices = (element_count * index_bits + 7) / 8;
                table_size + packed_indices
            }
            _ => element_count * element_size as u64,
        };

        Ok(size)
    }
}
```

### 3. Enhanced GGUF Parser with Validation

**Comprehensive Parser Implementation**:
```rust
// Enhanced GGUF parser with real-time validation
pub struct ValidatingGGUFParser {
    strict_mode: bool,
    alignment_requirements: AlignmentRequirements,
    supported_quantizations: HashSet<QuantizationType>,
    validation_config: ValidationConfig,
}

impl ValidatingGGUFParser {
    pub fn parse_and_validate<R: Read + Seek>(&self, reader: &mut R) -> Result<ValidatedGGUFModel, ParseError> {
        let mut model = ValidatedGGUFModel::new();

        // 1. Parse and validate header
        let header = self.parse_header(reader)?;
        self.validate_header(&header)?;
        model.header = header;

        // 2. Parse and validate metadata
        let metadata = self.parse_metadata(reader)?;
        self.validate_metadata(&metadata)?;
        model.metadata = metadata;

        // 3. Parse and validate tensor descriptors
        let tensor_descriptors = self.parse_tensor_descriptors(reader)?;
        self.validate_tensor_descriptors(&tensor_descriptors, reader)?;
        model.tensors = tensor_descriptors;

        // 4. Cross-reference validation
        self.validate_cross_references(&model)?;

        // 5. BitNet-specific validation
        self.validate_bitnet_requirements(&model)?;

        Ok(model)
    }

    fn validate_tensor_descriptors<R: Read + Seek>(&self, tensors: &[ValidatedTensorDescriptor], reader: &mut R) -> Result<(), ValidationError> {
        let file_size = reader.seek(SeekFrom::End(0))?;

        for tensor in tensors {
            // Basic validation
            tensor.validate(file_size)?;

            // Content validation (sample-based for large tensors)
            if self.validation_config.validate_tensor_content {
                self.validate_tensor_content(tensor, reader)?;
            }

            // Quantization-specific validation
            self.validate_quantization_integrity(tensor, reader)?;
        }

        // Cross-tensor validation
        self.validate_tensor_relationships(tensors)?;

        Ok(())
    }

    fn validate_quantization_integrity<R: Read + Seek>(&self, tensor: &ValidatedTensorDescriptor, reader: &mut R) -> Result<(), ValidationError> {
        match tensor.quantization_type {
            QuantizationType::I2S => self.validate_i2s_integrity(tensor, reader),
            QuantizationType::TL1 => self.validate_tl1_integrity(tensor, reader),
            QuantizationType::TL2 => self.validate_tl2_integrity(tensor, reader),
            _ => Ok(()), // Other formats use default validation
        }
    }

    fn validate_i2s_integrity<R: Read + Seek>(&self, tensor: &ValidatedTensorDescriptor, reader: &mut R) -> Result<(), ValidationError> {
        reader.seek(SeekFrom::Start(tensor.offset))?;

        // Sample validation for I2S format
        let sample_size = std::cmp::min(1024, tensor.size); // Sample first 1KB
        let mut buffer = vec![0u8; sample_size as usize];
        reader.read_exact(&mut buffer)?;

        // Validate I2S block structure
        let block_size = 32; // I2S uses 32-element blocks
        let elements_per_byte = 4; // 2 bits per element

        for chunk in buffer.chunks(block_size / elements_per_byte + 4) { // +4 for scale factor
            if chunk.len() < block_size / elements_per_byte + 4 {
                break; // Last incomplete block
            }

            // Extract scale factor (last 4 bytes of block)
            let scale_bytes = &chunk[chunk.len() - 4..];
            let scale = f32::from_le_bytes([
                scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]
            ]);

            // Validate scale factor is reasonable
            if !scale.is_finite() || scale.abs() > 100.0 {
                return Err(ValidationError::InvalidScaleFactor {
                    tensor_name: tensor.name.clone(),
                    scale,
                });
            }
        }

        Ok(())
    }

    fn validate_bitnet_requirements(&self, model: &ValidatedGGUFModel) -> Result<(), ValidationError> {
        // 1. Required metadata presence
        let required_keys = [
            "general.architecture",
            "general.name",
            "bitnet.version",
            "bitnet.group_size",
        ];

        for key in required_keys {
            if !model.metadata.contains_key(key) {
                return Err(ValidationError::MissingRequiredMetadata(key.to_string()));
            }
        }

        // 2. Architecture validation
        let architecture = model.metadata.get("general.architecture")
            .ok_or_else(|| ValidationError::MissingRequiredMetadata("general.architecture".to_string()))?;

        if !architecture.contains("bitnet") && !architecture.contains("BitNet") {
            return Err(ValidationError::UnsupportedArchitecture(architecture.clone()));
        }

        // 3. Quantization consistency
        let quantization_version = model.metadata.get("general.quantization_version");
        if let Some(quant_ver) = quantization_version {
            match quant_ver.as_str() {
                "i2_s" | "tl1" | "tl2" => {
                    // Validate that all tensors use compatible quantization
                    self.validate_quantization_consistency(model, quant_ver)?;
                }
                _ => return Err(ValidationError::UnsupportedQuantization(quant_ver.clone())),
            }
        }

        // 4. Tokenizer compatibility
        self.validate_tokenizer_metadata(model)?;

        Ok(())
    }
}
```

### 4. Tensor Validation Framework

**Multi-Level Validation Approach**:
```rust
// Comprehensive tensor validation system
pub struct TensorValidator {
    validation_level: ValidationLevel,
    error_tolerance: ErrorTolerance,
    device_config: DeviceConfig,
}

#[derive(Debug, Clone)]
pub enum ValidationLevel {
    Basic,          // Header and alignment only
    Standard,       // + content sampling and format validation
    Comprehensive,  // + full content validation and cross-references
    Paranoid,       // + cryptographic checksums and deep analysis
}

impl TensorValidator {
    pub fn validate_model(&self, model: &ValidatedGGUFModel) -> Result<ValidationReport, ValidationError> {
        let mut report = ValidationReport::new();

        // Stage 1: Structural validation
        report.structural = self.validate_structure(model)?;

        // Stage 2: Content validation
        if self.validation_level >= ValidationLevel::Standard {
            report.content = self.validate_content(model)?;
        }

        // Stage 3: Semantic validation
        if self.validation_level >= ValidationLevel::Comprehensive {
            report.semantic = self.validate_semantics(model)?;
        }

        // Stage 4: Cryptographic validation
        if self.validation_level >= ValidationLevel::Paranoid {
            report.cryptographic = self.validate_cryptographic(model)?;
        }

        Ok(report)
    }

    fn validate_content(&self, model: &ValidatedGGUFModel) -> Result<ContentValidationResult, ValidationError> {
        let mut result = ContentValidationResult::new();

        for tensor in &model.tensors {
            let tensor_validation = self.validate_tensor_content(tensor, model)?;
            result.tensor_results.insert(tensor.name.clone(), tensor_validation);
        }

        // Cross-tensor content validation
        result.cross_tensor = self.validate_cross_tensor_content(model)?;

        Ok(result)
    }

    fn validate_tensor_content(&self, tensor: &ValidatedTensorDescriptor, model: &ValidatedGGUFModel) -> Result<TensorContentValidation, ValidationError> {
        let mut validation = TensorContentValidation::new();

        // 1. Data integrity checks
        validation.data_integrity = self.check_data_integrity(tensor, model)?;

        // 2. Quantization format compliance
        validation.quantization_compliance = self.check_quantization_compliance(tensor, model)?;

        // 3. Statistical validation
        validation.statistical = self.validate_tensor_statistics(tensor, model)?;

        // 4. Range validation
        validation.range = self.validate_tensor_ranges(tensor, model)?;

        Ok(validation)
    }

    fn validate_tensor_statistics(&self, tensor: &ValidatedTensorDescriptor, model: &ValidatedGGUFModel) -> Result<StatisticalValidation, ValidationError> {
        let sample_data = self.sample_tensor_data(tensor, model, 1000)?; // Sample 1000 elements

        let stats = TensorStatistics::calculate(&sample_data);

        // Validate statistics are reasonable for neural network weights
        let validation = StatisticalValidation {
            mean_reasonable: stats.mean.abs() < 10.0,
            std_reasonable: stats.std_dev > 0.0 && stats.std_dev < 100.0,
            no_infinite_values: stats.infinite_count == 0,
            no_nan_values: stats.nan_count == 0,
            reasonable_range: stats.max - stats.min < 1000.0,
        };

        if !validation.is_valid() {
            return Err(ValidationError::UnreasonableStatistics {
                tensor_name: tensor.name.clone(),
                stats,
            });
        }

        Ok(validation)
    }
}
```

### 5. Cross-Platform Compatibility

**Platform-Specific Validation**:
```rust
// Cross-platform GGUF compatibility layer
pub struct CrossPlatformValidator {
    target_platforms: Vec<Platform>,
    endianness_handling: EndiannessPolicy,
    alignment_policies: HashMap<Platform, AlignmentPolicy>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Platform {
    X86_64Linux,
    X86_64Windows,
    X86_64MacOS,
    Aarch64Linux,
    Aarch64MacOS,
    Wasm32,
}

impl CrossPlatformValidator {
    pub fn validate_cross_platform_compatibility(&self, model: &ValidatedGGUFModel) -> Result<CrossPlatformReport, ValidationError> {
        let mut report = CrossPlatformReport::new();

        for platform in &self.target_platforms {
            let platform_result = self.validate_for_platform(model, platform)?;
            report.platform_results.insert(platform.clone(), platform_result);
        }

        // Check for platform-specific incompatibilities
        report.incompatibilities = self.detect_incompatibilities(&report.platform_results);

        Ok(report)
    }

    fn validate_for_platform(&self, model: &ValidatedGGUFModel, platform: &Platform) -> Result<PlatformValidationResult, ValidationError> {
        let mut result = PlatformValidationResult::new();

        // 1. Alignment validation for platform
        let alignment_policy = self.alignment_policies.get(platform)
            .unwrap_or(&AlignmentPolicy::default());

        result.alignment = self.validate_platform_alignment(model, alignment_policy)?;

        // 2. Endianness validation
        result.endianness = self.validate_endianness(model, platform)?;

        // 3. Platform-specific size limits
        result.size_limits = self.validate_size_limits(model, platform)?;

        // 4. Memory mapping compatibility
        result.memory_mapping = self.validate_memory_mapping(model, platform)?;

        Ok(result)
    }

    fn validate_memory_mapping(&self, model: &ValidatedGGUFModel, platform: &Platform) -> Result<MemoryMappingValidation, ValidationError> {
        let validation = MemoryMappingValidation {
            file_size_supported: self.check_file_size_limits(model, platform)?,
            alignment_compatible: self.check_memory_alignment(model, platform)?,
            address_space_sufficient: self.check_address_space(model, platform)?,
        };

        Ok(validation)
    }
}
```

### 6. Error Recovery and Diagnostics

**Enhanced Error Reporting**:
```rust
// Comprehensive error reporting for GGUF validation
#[derive(Debug, thiserror::Error)]
pub enum GGUFValidationError {
    #[error("Invalid GGUF header: {reason}")]
    InvalidHeader { reason: String },

    #[error("Tensor '{name}' alignment error: offset {offset} not aligned to {required} bytes")]
    TensorAlignment { name: String, offset: u64, required: u32 },

    #[error("Tensor '{name}' size mismatch: declared {declared}, calculated {calculated}")]
    SizeMismatch { name: String, declared: u64, calculated: u64 },

    #[error("Tensor '{name}' out of bounds: offset {offset} + size {size} > file size {file_size}")]
    TensorOutOfBounds { name: String, offset: u64, size: u64, file_size: u64 },

    #[error("Unsupported quantization format: {format}")]
    UnsupportedQuantization { format: String },

    #[error("Invalid scale factor in tensor '{name}': {scale}")]
    InvalidScaleFactor { name: String, scale: f32 },

    #[error("Missing required metadata: {key}")]
    MissingRequiredMetadata { key: String },

    #[error("Corrupt tensor data in '{name}': {reason}")]
    CorruptTensorData { name: String, reason: String },
}

impl GGUFValidationError {
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::TensorAlignment { .. } => vec![
                "Re-export the model with proper alignment".to_string(),
                "Use bitnet-compat export-fixed to fix alignment".to_string(),
            ],
            Self::SizeMismatch { .. } => vec![
                "Verify model export was completed successfully".to_string(),
                "Check for file corruption during transfer".to_string(),
            ],
            Self::TensorOutOfBounds { .. } => vec![
                "File may be truncated during download".to_string(),
                "Re-download the model file".to_string(),
            ],
            Self::UnsupportedQuantization { .. } => vec![
                "Update BitNet.rs to support this quantization format".to_string(),
                "Convert model to supported quantization format".to_string(),
            ],
            _ => vec!["Contact support with validation report".to_string()],
        }
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(self,
            Self::TensorAlignment { .. } |
            Self::UnsupportedQuantization { .. }
        )
    }
}
```

### 7. Performance-Optimized Validation

**Lazy Validation Strategy**:
```rust
// Performance-optimized validation for large models
pub struct LazyValidator {
    cache: ValidationCache,
    async_validation: bool,
    parallel_validation: bool,
}

impl LazyValidator {
    pub async fn validate_incrementally(&mut self, model: &ValidatedGGUFModel) -> Result<ValidationStream, ValidationError> {
        let validation_stream = ValidationStream::new();

        // Start validation in background
        if self.async_validation {
            self.start_async_validation(model, validation_stream.sender()).await?;
        } else {
            self.validate_synchronously(model, validation_stream.sender()).await?;
        }

        Ok(validation_stream)
    }

    async fn start_async_validation(&self, model: &ValidatedGGUFModel, sender: ValidationSender) -> Result<(), ValidationError> {
        let tensors = model.tensors.clone();

        tokio::spawn(async move {
            for tensor in tensors {
                let validation_result = validate_tensor_async(&tensor).await;
                if sender.send(validation_result).await.is_err() {
                    break; // Receiver dropped
                }
            }
        });

        Ok(())
    }
}

// Validation result streaming
pub struct ValidationStream {
    receiver: tokio::sync::mpsc::Receiver<TensorValidationResult>,
}

impl ValidationStream {
    pub async fn next(&mut self) -> Option<TensorValidationResult> {
        self.receiver.recv().await
    }

    pub fn try_next(&mut self) -> Option<TensorValidationResult> {
        self.receiver.try_recv().ok()
    }
}
```

## Integration with BitNet.rs Ecosystem

### 1. bitnet-models Integration

```rust
// Integration with model loading
impl BitNetModel {
    pub fn load_with_validation(path: &Path, validation_config: ValidationConfig) -> Result<Self, ModelLoadError> {
        let parser = ValidatingGGUFParser::new(validation_config);
        let validated_model = parser.parse_and_validate_file(path)?;

        // Create BitNet model from validated GGUF
        Self::from_validated_gguf(validated_model)
    }

    pub fn validate_compatibility(&self) -> Result<CompatibilityReport, ValidationError> {
        let validator = TensorValidator::new(ValidationLevel::Comprehensive);
        validator.validate_model(&self.gguf_model)
    }
}
```

### 2. bitnet-cli Integration

```rust
// CLI commands for GGUF validation
#[derive(Subcommand)]
pub enum GGUFCommands {
    /// Validate GGUF model compatibility
    Validate {
        #[arg(help = "Path to GGUF model file")]
        model_path: PathBuf,

        #[arg(long, default_value = "standard")]
        level: ValidationLevel,

        #[arg(long)]
        json_output: bool,

        #[arg(long)]
        fix_issues: bool,
    },

    /// Inspect GGUF model metadata and structure
    Inspect {
        #[arg(help = "Path to GGUF model file")]
        model_path: PathBuf,

        #[arg(long)]
        show_tensors: bool,

        #[arg(long)]
        show_metadata: bool,
    },

    /// Fix common GGUF compatibility issues
    Fix {
        #[arg(help = "Input GGUF model file")]
        input: PathBuf,

        #[arg(help = "Output fixed GGUF model file")]
        output: PathBuf,

        #[arg(long)]
        fix_alignment: bool,

        #[arg(long)]
        verify_output: bool,
    },
}
```

## Testing and Validation Framework

### Comprehensive Test Suite

```rust
// Integration tests for GGUF validation
#[cfg(feature = "integration-tests")]
mod gguf_validation_tests {
    use super::*;

    #[tokio::test]
    async fn test_real_bitnet_model_validation() {
        let test_models = discover_test_models().await;

        for model_path in test_models {
            let parser = ValidatingGGUFParser::new(ValidationConfig::strict());
            let result = parser.parse_and_validate_file(&model_path);

            match result {
                Ok(model) => {
                    println!("✅ {} validated successfully", model_path.display());
                    validate_model_inference_capability(&model).await.unwrap();
                }
                Err(e) => {
                    panic!("❌ {} validation failed: {}", model_path.display(), e);
                }
            }
        }
    }

    #[test]
    fn test_corrupted_model_detection() {
        let corrupted_models = create_corrupted_test_models();

        for (corruption_type, model_path) in corrupted_models {
            let parser = ValidatingGGUFParser::new(ValidationConfig::strict());
            let result = parser.parse_and_validate_file(&model_path);

            assert!(result.is_err(), "Should detect corruption: {:?}", corruption_type);

            if let Err(e) = result {
                assert!(e.is_recoverable() || corruption_type.is_fatal());
            }
        }
    }

    #[test]
    fn test_cross_platform_compatibility() {
        let model_path = get_reference_model_path();
        let validator = CrossPlatformValidator::new(vec![
            Platform::X86_64Linux,
            Platform::Aarch64Linux,
            Platform::Wasm32,
        ]);

        let model = ValidatingGGUFParser::default().parse_and_validate_file(&model_path).unwrap();
        let report = validator.validate_cross_platform_compatibility(&model).unwrap();

        for (platform, result) in report.platform_results {
            assert!(result.is_compatible(), "Platform {:?} compatibility failed", platform);
        }
    }
}
```

This comprehensive GGUF compatibility specification ensures robust real BitNet model integration with production-grade validation, cross-platform support, and seamless integration with the BitNet.rs neural network ecosystem.
