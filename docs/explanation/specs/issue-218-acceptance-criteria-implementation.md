# Acceptance Criteria Implementation Approach: Issue #218

## AC1: Real BitNet Models Download and Load Successfully

### Technical Approach

**Enhanced xtask Model Management**:
```rust
// xtask/src/model_management.rs
pub struct ModelManager {
    cache_dir: PathBuf,
    hf_client: HuggingFaceClient,
    validation: ModelValidator,
}

impl ModelManager {
    pub async fn download_real_model(&self, id: &str, file: &str) -> Result<ModelArtifacts, DownloadError> {
        // 1. Check cache first
        if let Some(cached) = self.check_cache(id, file)? {
            return Ok(cached);
        }

        // 2. Download with progress tracking
        let artifacts = self.hf_client.download_with_tokenizer(id, file).await?;

        // 3. Validate integrity
        self.validation.validate_artifacts(&artifacts)?;

        // 4. Cache for future use
        self.cache_artifacts(&artifacts)?;

        Ok(artifacts)
    }
}
```

**Enhanced GGUF Loading in bitnet-models**:
```rust
// crates/bitnet-models/src/real_model_loader.rs
pub struct RealModelLoader {
    validation_level: ValidationLevel,
    memory_mapping: bool,
    device_preference: DevicePreference,
}

impl RealModelLoader {
    pub fn load_production_model(&self, path: &Path) -> Result<BitNetModel, ModelLoadError> {
        // 1. Memory-map the GGUF file
        let mmap = self.create_memory_map(path)?;

        // 2. Parse GGUF header with enhanced validation
        let header = GGUFHeader::parse_with_validation(&mmap)?;

        // 3. Validate tensor alignment and quantization formats
        self.validate_tensor_layout(&header)?;

        // 4. Extract tokenizer metadata
        let tokenizer_config = self.extract_tokenizer_config(&header)?;

        // 5. Create production model instance
        Ok(BitNetModel::new_production(header, tokenizer_config, mmap))
    }

    fn validate_tensor_layout(&self, header: &GGUFHeader) -> Result<(), ValidationError> {
        for tensor in &header.tensors {
            // Check 32-byte alignment
            if tensor.offset % 32 != 0 {
                return Err(ValidationError::TensorAlignmentError {
                    tensor_name: tensor.name.clone(),
                    offset: tensor.offset,
                    expected_alignment: 32,
                });
            }

            // Validate quantization format
            match tensor.quantization_type {
                QuantizationType::I2S | QuantizationType::TL1 | QuantizationType::TL2 => {
                    // BitNet-specific validation
                    self.validate_bitnet_tensor(tensor)?;
                }
                _ => return Err(ValidationError::UnsupportedQuantization(tensor.quantization_type)),
            }
        }
        Ok(())
    }
}
```

**Testing Strategy**:
- **Unit Tests**: Mock download scenarios, GGUF parsing edge cases
- **Integration Tests**: Real model download from Hugging Face
- **Error Handling**: Network failures, corrupted downloads, invalid GGUF files

### Success Metrics
- [ ] Downloads 2B BitNet model without errors
- [ ] GGUF parsing succeeds with tensor validation
- [ ] Memory mapping works for large files (>2GB)
- [ ] Cache mechanism reduces download time by 90%

## AC2: Examples and CLI Tools Support Real and Mock Models

### Technical Approach

**Feature-Gated Model Selection**:
```rust
// crates/bitnet-cli/src/model_provider.rs
pub trait ModelProvider {
    fn load_model(&self, path: Option<&Path>) -> Result<Box<dyn Model>, ModelError>;
    fn provider_type(&self) -> ModelProviderType;
}

pub struct HybridModelProvider {
    real_provider: RealModelProvider,
    mock_provider: MockModelProvider,
    config: ModelConfig,
}

impl HybridModelProvider {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            real_provider: RealModelProvider::new(config.real_config),
            mock_provider: MockModelProvider::new(config.mock_config),
            config,
        }
    }
}

impl ModelProvider for HybridModelProvider {
    fn load_model(&self, path: Option<&Path>) -> Result<Box<dyn Model>, ModelError> {
        match (path, self.config.prefer_real, cfg!(feature = "inference")) {
            // Real model path provided and inference feature enabled
            (Some(path), _, true) => {
                match self.real_provider.load_model(Some(path)) {
                    Ok(model) => Ok(model),
                    Err(e) if self.config.fallback_to_mock => {
                        eprintln!("⚠️  Real model failed, falling back to mock: {}", e);
                        self.mock_provider.load_model(None)
                    }
                    Err(e) => Err(e),
                }
            }
            // Development mode or no inference feature
            _ => self.mock_provider.load_model(None),
        }
    }
}
```

**Enhanced CLI Interface**:
```rust
// crates/bitnet-cli/src/main.rs
#[derive(Parser)]
#[command(name = "bitnet")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Force real model usage (requires --features inference)
    #[arg(long, global = true)]
    pub real_model: bool,

    /// Allow mock model fallback
    #[arg(long, global = true)]
    pub allow_mock: bool,

    /// Model path (auto-detect if not provided)
    #[arg(long, global = true)]
    pub model: Option<PathBuf>,

    /// Tokenizer path (extract from model if not provided)
    #[arg(long, global = true)]
    pub tokenizer: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run inference with model auto-detection
    Infer {
        #[arg(long)]
        prompt: String,

        #[arg(long, default_value = "32")]
        max_tokens: usize,

        #[arg(long)]
        deterministic: bool,

        #[arg(long)]
        gpu: bool,
    },
    /// Validate model compatibility
    Validate {
        #[arg(long)]
        strict: bool,

        #[arg(long)]
        format: OutputFormat,
    },
    /// Run performance benchmarks
    Benchmark {
        #[arg(long, default_value = "128")]
        tokens: usize,

        #[arg(long)]
        warmup: bool,

        #[arg(long)]
        json_output: Option<PathBuf>,
    },
}
```

**Example Integration**:
```rust
// examples/real_model_inference.rs
#[cfg(feature = "inference")]
use bitnet_inference::ProductionEngine;

#[cfg(not(feature = "inference"))]
use bitnet_inference::MockEngine as ProductionEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect model or use mock
    let model_provider = HybridModelProvider::new(ModelConfig {
        prefer_real: cfg!(feature = "inference"),
        fallback_to_mock: true,
        auto_download: true,
    });

    let model = model_provider.load_model(None)?;
    let mut engine = ProductionEngine::new(model).await?;

    let result = engine.infer("The capital of France is").await?;

    match engine.provider_type() {
        ModelProviderType::Real => println!("✅ Real model output: {}", result.text),
        ModelProviderType::Mock => println!("🔄 Mock model output: {}", result.text),
    }

    Ok(())
}
```

### Success Metrics
- [ ] CLI automatically detects real vs mock models
- [ ] Feature flags correctly gate real model usage
- [ ] Examples work without modification in both modes
- [ ] Graceful fallback when real models unavailable

## AC3: Complete Inference Pipeline with Performance Metrics

### Technical Approach

**Enhanced Inference Engine**:
```rust
// crates/bitnet-inference/src/production_engine.rs
pub struct ProductionEngine {
    model: BitNetModel,
    tokenizer: UniversalTokenizer,
    quantization: QuantizationEngine,
    device_manager: DeviceManager,
    metrics_collector: MetricsCollector,
}

impl ProductionEngine {
    pub async fn infer_with_full_metrics(&mut self, prompt: &str) -> Result<InferenceResult, InferenceError> {
        let start_time = Instant::now();

        // Stage 1: Tokenization
        let tokenization_start = Instant::now();
        let tokens = self.tokenizer.encode(prompt)?;
        let tokenization_time = tokenization_start.elapsed();

        // Stage 2: Model Loading (if not cached)
        let loading_start = Instant::now();
        self.ensure_model_loaded().await?;
        let loading_time = loading_start.elapsed();

        // Stage 3: Quantization
        let quantization_start = Instant::now();
        let quantized_weights = self.quantization.prepare_for_inference(&self.model)?;
        let quantization_time = quantization_start.elapsed();

        // Stage 4: Kernel Execution
        let kernel_start = Instant::now();
        let logits = self.device_manager.execute_inference(&quantized_weights, &tokens).await?;
        let kernel_time = kernel_start.elapsed();

        // Stage 5: Output Generation
        let output_start = Instant::now();
        let output_tokens = self.generate_tokens(logits, tokens.len())?;
        let output_text = self.tokenizer.decode(&output_tokens)?;
        let output_time = output_start.elapsed();

        let total_time = start_time.elapsed();

        // Collect comprehensive metrics
        let metrics = InferenceMetrics {
            timing: TimingMetrics {
                tokenization: tokenization_time,
                loading: loading_time,
                quantization: quantization_time,
                kernel_execution: kernel_time,
                output_generation: output_time,
                total: total_time,
            },
            throughput: ThroughputMetrics {
                input_tokens_per_sec: tokens.len() as f64 / tokenization_time.as_secs_f64(),
                output_tokens_per_sec: output_tokens.len() as f64 / output_time.as_secs_f64(),
                total_tokens_per_sec: (tokens.len() + output_tokens.len()) as f64 / total_time.as_secs_f64(),
            },
            device: self.device_manager.get_performance_metrics(),
            quantization: self.quantization.get_accuracy_metrics(),
        };

        Ok(InferenceResult {
            text: output_text,
            tokens: output_tokens,
            metrics,
            model_info: self.model.get_info(),
        })
    }
}
```

**Device-Aware Execution**:
```rust
// crates/bitnet-kernels/src/device_manager.rs
pub struct DeviceManager {
    gpu_available: bool,
    cpu_fallback: bool,
    current_device: Device,
    performance_monitor: PerformanceMonitor,
}

impl DeviceManager {
    pub async fn execute_inference(&mut self, weights: &QuantizedWeights, tokens: &[u32]) -> Result<Tensor, ExecutionError> {
        match self.current_device {
            Device::GPU(ref gpu) => {
                match self.execute_gpu_inference(gpu, weights, tokens).await {
                    Ok(result) => {
                        self.performance_monitor.record_gpu_success();
                        Ok(result)
                    }
                    Err(e) if self.cpu_fallback => {
                        eprintln!("⚠️  GPU inference failed, falling back to CPU: {}", e);
                        self.execute_cpu_inference(weights, tokens).await
                    }
                    Err(e) => Err(e),
                }
            }
            Device::CPU => self.execute_cpu_inference(weights, tokens).await,
        }
    }

    async fn execute_gpu_inference(&mut self, gpu: &GPUDevice, weights: &QuantizedWeights, tokens: &[u32]) -> Result<Tensor, ExecutionError> {
        // GPU-specific inference with CUDA kernels
        let memory_usage_start = gpu.get_memory_usage()?;

        let result = match weights.format {
            QuantizationFormat::I2S => gpu.execute_i2s_inference(weights, tokens).await?,
            QuantizationFormat::TL1 => gpu.execute_tl1_inference(weights, tokens).await?,
            QuantizationFormat::TL2 => gpu.execute_tl2_inference(weights, tokens).await?,
            _ => return Err(ExecutionError::UnsupportedQuantization(weights.format)),
        };

        let memory_usage_end = gpu.get_memory_usage()?;
        self.performance_monitor.record_memory_usage(memory_usage_start, memory_usage_end);

        Ok(result)
    }
}
```

### Success Metrics
- [ ] End-to-end pipeline completes without errors
- [ ] Performance metrics collected for all stages
- [ ] GPU acceleration working with CPU fallback
- [ ] Memory usage tracking and leak detection

## AC4: Text Generation with Real Models

### Technical Approach

**Quality Validation Framework**:
```rust
// crates/bitnet-inference/src/text_quality.rs
pub struct TextQualityValidator {
    coherence_threshold: f32,
    context_similarity: ContextSimilarity,
    language_model: LanguageModelChecker,
}

impl TextQualityValidator {
    pub fn validate_generation(&self, prompt: &str, generated: &str) -> QualityResult {
        let coherence_score = self.measure_coherence(prompt, generated);
        let context_relevance = self.context_similarity.measure(prompt, generated);
        let grammar_score = self.language_model.check_grammar(generated);
        let repetition_penalty = self.detect_repetition(generated);

        QualityResult {
            coherence: coherence_score,
            context_relevance,
            grammar: grammar_score,
            repetition: repetition_penalty,
            overall: self.calculate_overall_score(coherence_score, context_relevance, grammar_score, repetition_penalty),
        }
    }

    fn measure_coherence(&self, prompt: &str, generated: &str) -> f32 {
        // Measure semantic coherence between prompt and generated text
        // Using embedding similarity or other NLP metrics
        todo!("Implement coherence measurement")
    }

    fn detect_repetition(&self, text: &str) -> f32 {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut repetition_count = 0;
        let window_size = 4;

        for i in 0..tokens.len().saturating_sub(window_size * 2) {
            let window1 = &tokens[i..i + window_size];
            for j in (i + window_size)..tokens.len().saturating_sub(window_size) {
                let window2 = &tokens[j..j + window_size];
                if window1 == window2 {
                    repetition_count += 1;
                    break;
                }
            }
        }

        repetition_count as f32 / tokens.len().max(1) as f32
    }
}
```

**Generation Testing Suite**:
```rust
// crates/bitnet-inference/tests/text_generation_tests.rs
#[cfg(feature = "integration-tests")]
mod real_model_tests {
    use super::*;

    #[tokio::test]
    async fn test_coherent_text_generation() {
        let engine = create_real_model_engine().await.expect("Failed to create engine");
        let validator = TextQualityValidator::new();

        let test_cases = vec![
            ("The capital of France is", "Paris"),
            ("Explain quantum computing in simple terms:", "computing"),
            ("Write a short poem about", "poem"),
        ];

        for (prompt, expected_keyword) in test_cases {
            let result = engine.infer(prompt).await.expect("Inference failed");

            // Check that generation contains expected content
            assert!(result.text.to_lowercase().contains(expected_keyword));

            // Validate text quality
            let quality = validator.validate_generation(prompt, &result.text);
            assert!(quality.coherence > 0.7, "Low coherence: {}", quality.coherence);
            assert!(quality.context_relevance > 0.6, "Low context relevance: {}", quality.context_relevance);
            assert!(quality.repetition < 0.3, "High repetition: {}", quality.repetition);
        }
    }

    #[tokio::test]
    async fn test_deterministic_generation() {
        let mut engine = create_real_model_engine().await.expect("Failed to create engine");
        engine.set_deterministic(true, 42);

        let prompt = "Once upon a time";
        let result1 = engine.infer(prompt).await.expect("First inference failed");
        let result2 = engine.infer(prompt).await.expect("Second inference failed");

        assert_eq!(result1.text, result2.text, "Non-deterministic generation");
        assert_eq!(result1.tokens, result2.tokens, "Non-deterministic tokens");
    }
}
```

### Success Metrics
- [ ] Generated text is contextually relevant to prompts
- [ ] Coherence scores above 0.7 for standard prompts
- [ ] Deterministic generation with fixed seeds
- [ ] No excessive repetition or degenerate outputs

## AC5: Tokenization Pipeline with Real Model Vocabulary

### Technical Approach

**Enhanced Universal Tokenizer**:
```rust
// crates/bitnet-tokenizers/src/universal_tokenizer.rs
pub struct UniversalTokenizer {
    backend: TokenizerBackend,
    config: TokenizerConfig,
    vocab_size: usize,
    special_tokens: SpecialTokens,
}

impl UniversalTokenizer {
    pub fn from_gguf_model(model_path: &Path) -> Result<Self, TokenizerError> {
        let model = GGUFModel::load(model_path)?;
        let metadata = model.extract_tokenizer_metadata()?;

        match metadata.tokenizer_type.as_str() {
            "sentencepiece" => Self::create_sentencepiece_tokenizer(metadata),
            "bpe" | "gpt2" => Self::create_bpe_tokenizer(metadata),
            "llama" => Self::create_llama_tokenizer(metadata),
            _ => Err(TokenizerError::UnsupportedType(metadata.tokenizer_type)),
        }
    }

    fn create_bpe_tokenizer(metadata: TokenizerMetadata) -> Result<Self, TokenizerError> {
        let vocab = metadata.extract_vocabulary()?;
        let merges = metadata.extract_merges()?;

        let backend = BPEBackend::new(vocab, merges)?;

        Ok(Self {
            backend: TokenizerBackend::BPE(backend),
            config: metadata.config,
            vocab_size: metadata.vocab_size,
            special_tokens: metadata.special_tokens,
        })
    }

    pub fn validate_compatibility(&self, model: &BitNetModel) -> Result<CompatibilityReport, ValidationError> {
        let model_vocab_size = model.get_vocab_size();
        let embedding_dim = model.get_embedding_dim();

        let compatibility = CompatibilityReport {
            vocab_size_match: self.vocab_size == model_vocab_size,
            special_tokens_present: self.validate_special_tokens(model)?,
            encoding_consistency: self.test_encoding_consistency()?,
            model_expectations: ModelExpectations {
                expected_vocab_size: model_vocab_size,
                actual_vocab_size: self.vocab_size,
                embedding_dimension: embedding_dim,
            },
        };

        if !compatibility.vocab_size_match {
            return Err(ValidationError::VocabSizeMismatch {
                expected: model_vocab_size,
                actual: self.vocab_size,
            });
        }

        Ok(compatibility)
    }

    fn validate_special_tokens(&self, model: &BitNetModel) -> Result<bool, ValidationError> {
        let required_tokens = ["<bos>", "<eos>", "<pad>", "<unk>"];

        for token in required_tokens {
            if self.special_tokens.get(token).is_none() {
                return Err(ValidationError::MissingSpecialToken(token.to_string()));
            }
        }

        Ok(true)
    }
}
```

**Tokenization Testing Framework**:
```rust
// crates/bitnet-tokenizers/tests/real_model_tokenization.rs
#[cfg(feature = "integration-tests")]
mod tests {
    use super::*;

    #[test]
    fn test_real_model_tokenizer_extraction() {
        let model_path = get_test_model_path();
        let tokenizer = UniversalTokenizer::from_gguf_model(&model_path)
            .expect("Failed to extract tokenizer from real model");

        assert!(tokenizer.vocab_size > 50000, "Vocab size too small: {}", tokenizer.vocab_size);
        assert!(tokenizer.supports_special_tokens(), "Missing special token support");
    }

    #[test]
    fn test_tokenization_round_trip() {
        let tokenizer = create_real_tokenizer();
        let test_texts = vec![
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "🌟 Unicode test with emojis 🚀",
            "Code example: let x = 42;",
        ];

        for text in test_texts {
            let tokens = tokenizer.encode(text).expect("Encoding failed");
            let decoded = tokenizer.decode(&tokens).expect("Decoding failed");

            // Allow minor whitespace differences
            let normalized_original = normalize_whitespace(text);
            let normalized_decoded = normalize_whitespace(&decoded);

            assert_eq!(normalized_original, normalized_decoded,
                      "Round-trip failed for: '{}'", text);
        }
    }

    #[test]
    fn test_tokenizer_model_compatibility() {
        let model = load_real_bitnet_model();
        let tokenizer = UniversalTokenizer::from_gguf_model(model.path())
            .expect("Failed to create tokenizer");

        let compatibility = tokenizer.validate_compatibility(&model)
            .expect("Compatibility validation failed");

        assert!(compatibility.vocab_size_match, "Vocab size mismatch");
        assert!(compatibility.special_tokens_present, "Missing special tokens");
        assert!(compatibility.encoding_consistency, "Encoding inconsistency");
    }
}
```

### Success Metrics
- [ ] Tokenizer extracted from real GGUF model metadata
- [ ] Vocabulary size matches model expectations
- [ ] Round-trip encoding/decoding preserves text
- [ ] Special tokens correctly identified and handled

## AC6: GGUF Compatibility Validation

### Technical Approach

**Enhanced GGUF Validator**:
```rust
// crates/bitnet-compat/src/gguf_validator.rs
pub struct GGUFValidator {
    strict_mode: bool,
    supported_versions: Vec<u32>,
    alignment_requirements: AlignmentRequirements,
}

impl GGUFValidator {
    pub fn validate_bitnet_model(&self, path: &Path) -> Result<ValidationReport, ValidationError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let mut report = ValidationReport::new();

        // 1. Header validation
        self.validate_header(&mmap, &mut report)?;

        // 2. Metadata validation
        self.validate_metadata(&mmap, &mut report)?;

        // 3. Tensor validation
        self.validate_tensors(&mmap, &mut report)?;

        // 4. BitNet-specific validation
        self.validate_bitnet_specifics(&mmap, &mut report)?;

        // 5. Cross-reference validation
        self.validate_cross_references(&mmap, &mut report)?;

        Ok(report)
    }

    fn validate_tensors(&self, mmap: &Mmap, report: &mut ValidationReport) -> Result<(), ValidationError> {
        let header = GGUFHeader::parse(mmap)?;

        for (i, tensor) in header.tensors.iter().enumerate() {
            // Check tensor alignment
            if tensor.offset % self.alignment_requirements.tensor_alignment != 0 {
                report.errors.push(ValidationError::TensorAlignment {
                    tensor_index: i,
                    tensor_name: tensor.name.clone(),
                    offset: tensor.offset,
                    required_alignment: self.alignment_requirements.tensor_alignment,
                });
            }

            // Validate tensor dimensions
            if tensor.n_dims != tensor.dims.len() as u32 {
                report.errors.push(ValidationError::DimensionMismatch {
                    tensor_name: tensor.name.clone(),
                    declared_dims: tensor.n_dims,
                    actual_dims: tensor.dims.len(),
                });
            }

            // Check tensor accessibility
            let tensor_size = self.calculate_tensor_size(tensor)?;
            if tensor.offset + tensor_size > mmap.len() as u64 {
                report.errors.push(ValidationError::TensorOutOfBounds {
                    tensor_name: tensor.name.clone(),
                    offset: tensor.offset,
                    size: tensor_size,
                    file_size: mmap.len(),
                });
            }

            // BitNet quantization validation
            self.validate_quantization_format(tensor, report)?;
        }

        Ok(())
    }

    fn validate_bitnet_specifics(&self, mmap: &Mmap, report: &mut ValidationReport) -> Result<(), ValidationError> {
        let metadata = self.extract_metadata(mmap)?;

        // Check for required BitNet metadata
        let required_keys = [
            "general.architecture",
            "general.quantization_version",
            "bitnet.version",
            "bitnet.group_size",
        ];

        for key in required_keys {
            if !metadata.contains_key(key) {
                report.warnings.push(ValidationWarning::MissingMetadata(key.to_string()));
            }
        }

        // Validate quantization configuration
        if let Some(quant_type) = metadata.get("general.quantization_version") {
            match quant_type.as_str() {
                "i2_s" | "tl1" | "tl2" => {
                    // Valid BitNet quantization
                    report.bitnet_info.quantization_type = Some(quant_type.to_string());
                }
                _ => {
                    report.warnings.push(ValidationWarning::UnsupportedQuantization(quant_type.to_string()));
                }
            }
        }

        Ok(())
    }
}
```

**Comprehensive Testing Suite**:
```rust
// crates/bitnet-compat/tests/gguf_validation_tests.rs
#[cfg(feature = "integration-tests")]
mod tests {
    use super::*;

    #[test]
    fn test_real_bitnet_model_validation() {
        let model_paths = discover_test_models();

        for model_path in model_paths {
            let validator = GGUFValidator::new(ValidationConfig::strict());
            let report = validator.validate_bitnet_model(&model_path)
                .expect("Validation should not fail catastrophically");

            // Check critical validations
            assert!(report.errors.is_empty(), "Validation errors found: {:?}", report.errors);
            assert!(report.header_valid, "Invalid GGUF header");
            assert!(report.tensors_accessible, "Tensors not accessible");

            // BitNet-specific checks
            assert!(report.bitnet_info.quantization_type.is_some(), "Missing quantization type");
            assert!(report.bitnet_info.group_size.is_some(), "Missing group size");

            println!("✅ Model {} validated successfully", model_path.display());
        }
    }

    #[test]
    fn test_tensor_alignment_validation() {
        let model_path = get_test_model_path();
        let validator = GGUFValidator::new(ValidationConfig::strict());

        let report = validator.validate_bitnet_model(&model_path).unwrap();

        // Check that all tensors are properly aligned
        for tensor_info in &report.tensor_info {
            assert_eq!(tensor_info.offset % 32, 0,
                      "Tensor {} not aligned: offset {}",
                      tensor_info.name, tensor_info.offset);
        }
    }

    #[test]
    fn test_corrupted_model_detection() {
        let corrupted_models = create_corrupted_test_models();
        let validator = GGUFValidator::new(ValidationConfig::strict());

        for (corruption_type, model_path) in corrupted_models {
            let result = validator.validate_bitnet_model(&model_path);

            match corruption_type {
                CorruptionType::InvalidHeader => {
                    assert!(result.is_err(), "Should detect invalid header");
                }
                CorruptionType::MisalignedTensor => {
                    let report = result.unwrap();
                    assert!(!report.errors.is_empty(), "Should detect misaligned tensors");
                }
                CorruptionType::TruncatedFile => {
                    assert!(result.is_err(), "Should detect truncated file");
                }
            }
        }
    }
}
```

### Success Metrics
- [ ] All tensor alignments verified (32-byte alignment)
- [ ] Metadata consistency validated
- [ ] BitNet-specific format requirements checked
- [ ] Corrupted model detection working

## AC7: Cross-Validation Framework

### Technical Approach

**Enhanced Cross-Validation System**:
```rust
// crossval/src/inference_comparison.rs
pub struct CrossValidationFramework {
    rust_engine: BitNetEngine,
    cpp_reference: CppReference,
    tolerance_config: ToleranceConfig,
    test_suite: ValidationTestSuite,
}

impl CrossValidationFramework {
    pub async fn run_comprehensive_validation(&mut self) -> Result<ValidationReport, ValidationError> {
        let mut report = ValidationReport::new();

        // 1. Token-level validation
        self.validate_tokenization(&mut report).await?;

        // 2. Inference output validation
        self.validate_inference_outputs(&mut report).await?;

        // 3. Perplexity validation
        self.validate_perplexity(&mut report).await?;

        // 4. Performance comparison
        self.compare_performance(&mut report).await?;

        Ok(report)
    }

    async fn validate_inference_outputs(&mut self, report: &mut ValidationReport) -> Result<(), ValidationError> {
        let test_prompts = self.test_suite.get_inference_prompts();

        for prompt in test_prompts {
            // Run both implementations
            let rust_result = self.rust_engine.infer(&prompt.text).await?;
            let cpp_result = self.cpp_reference.infer(&prompt.text).await?;

            // Compare outputs
            let comparison = self.compare_inference_results(&rust_result, &cpp_result)?;

            if comparison.token_match_rate < self.tolerance_config.min_token_match_rate {
                report.failures.push(ValidationFailure::TokenMismatch {
                    prompt: prompt.text.clone(),
                    rust_tokens: rust_result.tokens,
                    cpp_tokens: cpp_result.tokens,
                    match_rate: comparison.token_match_rate,
                });
            }

            if comparison.logit_correlation < self.tolerance_config.min_logit_correlation {
                report.failures.push(ValidationFailure::LogitMismatch {
                    prompt: prompt.text.clone(),
                    correlation: comparison.logit_correlation,
                    expected: self.tolerance_config.min_logit_correlation,
                });
            }

            report.test_results.push(InferenceTestResult {
                prompt: prompt.text,
                rust_output: rust_result.text,
                cpp_output: cpp_result.text,
                comparison,
            });
        }

        Ok(())
    }

    fn compare_inference_results(&self, rust_result: &InferenceResult, cpp_result: &CppInferenceResult) -> Result<ComparisonResult, ValidationError> {
        // Token-level comparison
        let token_matches = rust_result.tokens.iter()
            .zip(&cpp_result.tokens)
            .filter(|(r, c)| r == c)
            .count();
        let token_match_rate = token_matches as f64 / rust_result.tokens.len().max(cpp_result.tokens.len()) as f64;

        // Logit correlation (if available)
        let logit_correlation = if let (Some(rust_logits), Some(cpp_logits)) = (&rust_result.logits, &cpp_result.logits) {
            self.calculate_pearson_correlation(rust_logits, cpp_logits)?
        } else {
            1.0 // Assume perfect if logits not available
        };

        // Perplexity comparison
        let perplexity_diff = if let (Some(rust_ppl), Some(cpp_ppl)) = (rust_result.perplexity, cpp_result.perplexity) {
            (rust_ppl - cpp_ppl).abs() / cpp_ppl
        } else {
            0.0
        };

        Ok(ComparisonResult {
            token_match_rate,
            logit_correlation,
            perplexity_relative_diff: perplexity_diff,
            performance_ratio: rust_result.inference_time.as_secs_f64() / cpp_result.inference_time.as_secs_f64(),
        })
    }
}
```

**Configurable Tolerance System**:
```rust
// crossval/src/tolerance_config.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    /// Minimum token match rate for deterministic generation
    pub min_token_match_rate: f64,

    /// Minimum logit correlation (Pearson correlation coefficient)
    pub min_logit_correlation: f64,

    /// Maximum relative perplexity difference
    pub max_perplexity_diff: f64,

    /// Maximum relative performance difference
    pub max_performance_diff: f64,

    /// Quantization-specific tolerances
    pub quantization_tolerance: QuantizationTolerance,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            min_token_match_rate: 0.95,
            min_logit_correlation: 0.98,
            max_perplexity_diff: 0.02,
            max_performance_diff: 0.50,
            quantization_tolerance: QuantizationTolerance::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTolerance {
    pub i2s_weight_tolerance: f32,
    pub tl1_table_tolerance: f32,
    pub tl2_table_tolerance: f32,
    pub activation_tolerance: f32,
}

impl Default for QuantizationTolerance {
    fn default() -> Self {
        Self {
            i2s_weight_tolerance: 1e-5,
            tl1_table_tolerance: 1e-4,
            tl2_table_tolerance: 1e-4,
            activation_tolerance: 1e-3,
        }
    }
}
```

### Success Metrics
- [ ] Token match rate ≥95% for deterministic generation
- [ ] Logit correlation ≥0.98 between implementations
- [ ] Perplexity difference ≤2% relative
- [ ] Cross-validation passes for all supported models

## AC8: Perplexity Calculations and Quantization Accuracy

### Technical Approach

**Perplexity Validation Framework**:
```rust
// crates/bitnet-inference/src/perplexity.rs
pub struct PerplexityCalculator {
    model: Arc<BitNetModel>,
    tokenizer: Arc<UniversalTokenizer>,
    device: Device,
    batch_size: usize,
}

impl PerplexityCalculator {
    pub async fn calculate_corpus_perplexity(&self, corpus_path: &Path) -> Result<PerplexityResult, PerplexityError> {
        let corpus = self.load_corpus(corpus_path)?;
        let mut total_log_likelihood = 0.0;
        let mut total_tokens = 0;
        let mut batch_results = Vec::new();

        for batch in corpus.chunks(self.batch_size) {
            let batch_result = self.process_batch(batch).await?;
            total_log_likelihood += batch_result.log_likelihood;
            total_tokens += batch_result.token_count;
            batch_results.push(batch_result);
        }

        let perplexity = (-total_log_likelihood / total_tokens as f64).exp();

        Ok(PerplexityResult {
            perplexity,
            log_likelihood: total_log_likelihood,
            token_count: total_tokens,
            batch_results,
            model_info: self.model.get_info(),
        })
    }

    async fn process_batch(&self, texts: &[String]) -> Result<BatchPerplexityResult, PerplexityError> {
        let mut batch_log_likelihood = 0.0;
        let mut batch_token_count = 0;

        for text in texts {
            let tokens = self.tokenizer.encode(text)?;
            let text_log_likelihood = self.calculate_text_log_likelihood(&tokens).await?;

            batch_log_likelihood += text_log_likelihood;
            batch_token_count += tokens.len();
        }

        Ok(BatchPerplexityResult {
            log_likelihood: batch_log_likelihood,
            token_count: batch_token_count,
            perplexity: (-batch_log_likelihood / batch_token_count as f64).exp(),
        })
    }

    async fn calculate_text_log_likelihood(&self, tokens: &[u32]) -> Result<f64, PerplexityError> {
        let mut log_likelihood = 0.0;

        for i in 1..tokens.len() {
            let context = &tokens[..i];
            let target = tokens[i];

            // Run inference to get logits
            let logits = self.model.forward(context).await?;

            // Calculate log probability of target token
            let log_probs = self.softmax_log(&logits);
            log_likelihood += log_probs[target as usize];
        }

        Ok(log_likelihood)
    }
}
```

**Quantization Accuracy Validation**:
```rust
// crates/bitnet-quantization/src/accuracy_validation.rs
pub struct QuantizationAccuracyValidator {
    reference_weights: Tensor,
    quantized_weights: QuantizedTensor,
    tolerance: f32,
}

impl QuantizationAccuracyValidator {
    pub fn validate_quantization_accuracy(&self) -> Result<AccuracyReport, ValidationError> {
        let mut report = AccuracyReport::new();

        // 1. Weight-level accuracy
        report.weight_accuracy = self.validate_weight_accuracy()?;

        // 2. Layer-level accuracy
        report.layer_accuracy = self.validate_layer_accuracy()?;

        // 3. Model-level accuracy
        report.model_accuracy = self.validate_model_accuracy().await?;

        // 4. Perplexity preservation
        report.perplexity_preservation = self.validate_perplexity_preservation().await?;

        Ok(report)
    }

    fn validate_weight_accuracy(&self) -> Result<WeightAccuracyResult, ValidationError> {
        // Dequantize weights and compare with reference
        let dequantized = self.quantized_weights.dequantize()?;

        let mse = self.calculate_mse(&self.reference_weights, &dequantized);
        let max_error = self.calculate_max_error(&self.reference_weights, &dequantized);
        let relative_error = self.calculate_relative_error(&self.reference_weights, &dequantized);

        Ok(WeightAccuracyResult {
            mse,
            max_error,
            relative_error,
            passes_tolerance: relative_error < self.tolerance,
        })
    }

    async fn validate_perplexity_preservation(&self) -> Result<PerplexityPreservationResult, ValidationError> {
        let reference_perplexity = self.calculate_reference_perplexity().await?;
        let quantized_perplexity = self.calculate_quantized_perplexity().await?;

        let relative_diff = (quantized_perplexity - reference_perplexity).abs() / reference_perplexity;

        Ok(PerplexityPreservationResult {
            reference_perplexity,
            quantized_perplexity,
            relative_difference: relative_diff,
            passes_tolerance: relative_diff < 0.02, // 2% tolerance
        })
    }
}
```

### Success Metrics
- [ ] Perplexity calculations match reference within ±0.1%
- [ ] Quantization preserves accuracy within tolerance
- [ ] Weight-level validation passes for all formats
- [ ] Model-level accuracy maintained after quantization

## AC9: CI Integration with Real Model Support

### Technical Approach

**Three-Tier CI Strategy**:
```yaml
# .github/workflows/real-model-validation.yml
name: Real Model Validation

on:
  pull_request:
    paths:
      - 'crates/bitnet-models/**'
      - 'crates/bitnet-inference/**'
      - 'crates/bitnet-quantization/**'
  push:
    branches: [main]

jobs:
  # Tier 1: Fast validation with mock models
  fast-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: 1.90.0
      - name: Run fast tests with mock models
        run: |
          cargo test --workspace --no-default-features --features cpu

  # Tier 2: Real model validation (cached)
  real-model-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    needs: fast-validation
    steps:
      - uses: actions/checkout@v4
      - name: Cache BitNet models
        uses: actions/cache@v4
        with:
          path: ~/.cache/bitnet-models
          key: bitnet-models-${{ hashFiles('**/model-requirements.json') }}
      - name: Download real models if not cached
        run: |
          if [ ! -f ~/.cache/bitnet-models/bitnet-2b/model.gguf ]; then
            cargo run -p xtask -- download-model \
              --id microsoft/bitnet-b1.58-2B-4T-gguf \
              --file ggml-model-i2_s.gguf \
              --cache-dir ~/.cache/bitnet-models
          fi
      - name: Run real model tests
        env:
          BITNET_MODEL_PATH: ~/.cache/bitnet-models/bitnet-2b/model.gguf
          BITNET_STRICT_TOKENIZERS: 1
        run: |
          cargo test --workspace --no-default-features --features "cpu,inference" \
            --features integration-tests

  # Tier 3: Full validation with cross-validation
  full-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    needs: real-model-validation
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Setup C++ environment
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake
      - name: Cache models and C++ build
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/bitnet-models
            ~/.cache/bitnet-cpp
          key: full-validation-${{ hashFiles('**/Cargo.lock') }}
      - name: Run full cross-validation
        env:
          BITNET_MODEL_PATH: ~/.cache/bitnet-models/bitnet-2b/model.gguf
          BITNET_CPP_DIR: ~/.cache/bitnet-cpp
        run: |
          cargo run -p xtask -- full-crossval
```

**Smart Model Caching**:
```rust
// xtask/src/model_cache.rs
pub struct ModelCache {
    cache_dir: PathBuf,
    manifest: CacheManifest,
}

impl ModelCache {
    pub fn new(cache_dir: PathBuf) -> Result<Self, CacheError> {
        let manifest_path = cache_dir.join("manifest.json");
        let manifest = if manifest_path.exists() {
            CacheManifest::load(&manifest_path)?
        } else {
            CacheManifest::new()
        };

        Ok(Self { cache_dir, manifest })
    }

    pub async fn get_or_download(&mut self, model_id: &str, file: &str) -> Result<PathBuf, CacheError> {
        let cache_key = format!("{}:{}", model_id, file);

        if let Some(cached_path) = self.manifest.get(&cache_key) {
            if cached_path.exists() && self.validate_integrity(&cached_path)? {
                return Ok(cached_path);
            }
        }

        // Download and cache
        let downloaded_path = self.download_model(model_id, file).await?;
        self.manifest.insert(cache_key, downloaded_path.clone());
        self.manifest.save(&self.cache_dir.join("manifest.json"))?;

        Ok(downloaded_path)
    }

    pub fn cleanup_old_models(&mut self, max_age_days: u32) -> Result<(), CacheError> {
        let cutoff = SystemTime::now() - Duration::from_secs(max_age_days as u64 * 24 * 3600);

        let mut to_remove = Vec::new();

        for (key, path) in &self.manifest.entries {
            if let Ok(metadata) = path.metadata() {
                if let Ok(modified) = metadata.modified() {
                    if modified < cutoff {
                        to_remove.push(key.clone());
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }

        for key in to_remove {
            self.manifest.entries.remove(&key);
        }

        Ok(())
    }
}
```

**Feature-Gated Testing**:
```rust
// Integration test configuration
#[cfg(feature = "integration-tests")]
mod real_model_tests {
    #[test]
    fn test_real_model_loading() {
        if std::env::var("BITNET_MODEL_PATH").is_err() {
            eprintln!("⚠️  BITNET_MODEL_PATH not set, skipping real model test");
            return;
        }

        // Test with real model
        test_real_model_implementation();
    }

    #[test]
    fn test_with_mock_fallback() {
        // Always runs, uses mock if real model unavailable
        test_with_hybrid_provider();
    }
}
```

### Success Metrics
- [ ] Fast lane completes in <5 minutes
- [ ] Real model tests complete in <15 minutes
- [ ] Model caching reduces download time by 90%
- [ ] CI passes with both real and mock models

## AC10: Performance Benchmarks with Real Models

### Technical Approach

**Comprehensive Benchmarking Framework**:
```rust
// crates/bitnet-inference/benches/real_model_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_real_model_inference(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Setup real model (skip if not available)
    let model_path = std::env::var("BITNET_MODEL_PATH")
        .unwrap_or_else(|_| "mock".to_string());

    if model_path == "mock" {
        eprintln!("⚠️  No real model available, skipping performance benchmarks");
        return;
    }

    let engine = rt.block_on(async {
        ProductionEngine::load_from_path(&model_path).await
            .expect("Failed to load real model")
    });

    let mut group = c.benchmark_group("real_model_inference");

    // Test different prompt lengths
    for prompt_length in [10, 50, 100, 200] {
        let prompt = generate_test_prompt(prompt_length);

        group.bench_with_input(
            BenchmarkId::new("cpu_inference", prompt_length),
            &prompt,
            |b, prompt| {
                b.to_async(&rt).iter(|| async {
                    let mut engine = engine.clone();
                    engine.set_device(Device::CPU);
                    engine.infer(prompt).await.unwrap()
                });
            },
        );

        #[cfg(feature = "gpu")]
        group.bench_with_input(
            BenchmarkId::new("gpu_inference", prompt_length),
            &prompt,
            |b, prompt| {
                b.to_async(&rt).iter(|| async {
                    let mut engine = engine.clone();
                    if engine.set_device(Device::GPU).is_ok() {
                        engine.infer(prompt).await.unwrap()
                    } else {
                        // Fallback to CPU if GPU unavailable
                        engine.set_device(Device::CPU);
                        engine.infer(prompt).await.unwrap()
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_quantization_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_performance");

    // Test different quantization formats with real tensors
    for format in [QuantizationFormat::I2S, QuantizationFormat::TL1, QuantizationFormat::TL2] {
        let test_tensor = load_real_tensor_sample(&format);

        group.bench_with_input(
            BenchmarkId::new("cpu_quantization", format!("{:?}", format)),
            &test_tensor,
            |b, tensor| {
                b.iter(|| {
                    let quantizer = CPUQuantizer::new(format);
                    quantizer.quantize(tensor).unwrap()
                });
            },
        );

        #[cfg(feature = "gpu")]
        group.bench_with_input(
            BenchmarkId::new("gpu_quantization", format!("{:?}", format)),
            &test_tensor,
            |b, tensor| {
                b.iter(|| {
                    if let Ok(quantizer) = GPUQuantizer::new(format) {
                        quantizer.quantize(tensor).unwrap()
                    } else {
                        // GPU unavailable, skip
                        criterion::black_box(tensor.clone())
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_real_model_inference, benchmark_quantization_performance);
criterion_main!(benches);
```

**Performance Regression Detection**:
```rust
// xtask/src/performance_tracking.rs
pub struct PerformanceTracker {
    baseline_path: PathBuf,
    current_results: BenchmarkResults,
    regression_threshold: f64,
}

impl PerformanceTracker {
    pub fn detect_regressions(&self) -> Result<RegressionReport, TrackingError> {
        let baseline = self.load_baseline()?;
        let mut regressions = Vec::new();

        for (benchmark_name, current_result) in &self.current_results.benchmarks {
            if let Some(baseline_result) = baseline.benchmarks.get(benchmark_name) {
                let performance_ratio = current_result.mean / baseline_result.mean;

                if performance_ratio > 1.0 + self.regression_threshold {
                    regressions.push(PerformanceRegression {
                        benchmark: benchmark_name.clone(),
                        baseline_time: baseline_result.mean,
                        current_time: current_result.mean,
                        regression_percent: (performance_ratio - 1.0) * 100.0,
                    });
                }
            }
        }

        Ok(RegressionReport {
            regressions,
            total_benchmarks: self.current_results.benchmarks.len(),
            baseline_date: baseline.timestamp,
            current_date: self.current_results.timestamp,
        })
    }

    pub fn update_baseline(&self) -> Result<(), TrackingError> {
        // Only update baseline if no regressions detected
        let regression_report = self.detect_regressions()?;

        if regression_report.regressions.is_empty() {
            self.current_results.save_as_baseline(&self.baseline_path)?;
            println!("✅ Performance baseline updated");
        } else {
            println!("⚠️  Regressions detected, baseline not updated");
            for regression in &regression_report.regressions {
                println!("  - {}: {:.1}% slower", regression.benchmark, regression.regression_percent);
            }
        }

        Ok(())
    }
}
```

**Automated Performance Monitoring**:
```bash
#!/bin/bash
# scripts/performance-monitoring.sh

set -e

MODEL_PATH="${BITNET_MODEL_PATH:-mock}"
BASELINE_DIR="${BITNET_BASELINE_DIR:-baselines}"

echo "🚀 Running performance benchmarks..."

if [ "$MODEL_PATH" = "mock" ]; then
    echo "⚠️  No real model available, running with mock data"
    cargo bench --no-default-features --features cpu -- --output-format json > current_results.json
else
    echo "✅ Running with real model: $MODEL_PATH"

    # CPU benchmarks
    BITNET_MODEL_PATH="$MODEL_PATH" cargo bench --no-default-features --features cpu \
        -- --output-format json > cpu_results.json

    # GPU benchmarks (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected, running GPU benchmarks"
        BITNET_MODEL_PATH="$MODEL_PATH" cargo bench --no-default-features --features gpu \
            -- --output-format json > gpu_results.json
    fi
fi

# Detect regressions
echo "📊 Analyzing performance results..."
cargo run -p xtask -- analyze-performance \
    --results current_results.json \
    --baseline "$BASELINE_DIR/baseline.json" \
    --threshold 0.05

echo "✅ Performance analysis complete"
```

### Success Metrics
- [ ] GPU inference ≥100 tokens/sec (2B model)
- [ ] CPU inference ≥15 tokens/sec (2B model)
- [ ] Memory usage ≤4GB GPU, ≤8GB system
- [ ] Performance regression detection working
- [ ] Baseline tracking and updates automated

This comprehensive implementation approach provides detailed technical strategies for each Acceptance Criteria, ensuring production-ready real BitNet model integration with BitNet.rs neural network standards.