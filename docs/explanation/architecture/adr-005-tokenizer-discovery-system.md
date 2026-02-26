# ADR-005: Tokenizer Discovery and Automatic Download System

## Status
**Proposed** - Architectural Decision Record for Issue #249

## Context

BitNet-rs neural network inference currently requires manual tokenizer specification, creating friction for users and limiting production-ready model usage. The existing `auto.rs` provides basic co-location discovery, but lacks:

- **Neural Network Scale**: Support for large vocabulary models (LLaMA-3: 128K tokens, LLaMA-2: 32K tokens)
- **GGUF Integration**: Comprehensive metadata parsing for tokenizer discovery
- **Smart Downloads**: Automatic tokenizer acquisition from HuggingFace Hub
- **Quantization Compatibility**: Seamless integration with I2S/TL1/TL2 quantization pipelines
- **Production Reliability**: Robust error handling and fallback strategies

## Decision

We will implement a comprehensive **TokenizerDiscovery System** with the following architecture:

### 1. TokenizerDiscovery Core Engine
```rust
pub struct TokenizerDiscovery {
    gguf_reader: Arc<GgufReader<'static>>,
    model_path: PathBuf,
    vocab_size: usize,
    model_type: String,
    quantization_type: Option<QuantizationType>,
}
```

**Responsibilities**:
- Parse GGUF model metadata to extract tokenizer information
- Determine vocab size, model architecture, and quantization requirements
- Infer compatible tokenizer download sources based on neural network model patterns
- Validate tokenizer compatibility with BitNet-rs quantization formats

### 2. SmartTokenizerDownload System
```rust
pub struct SmartTokenizerDownload {
    cache_dir: PathBuf,
    client: reqwest::Client,
    neural_network_compatibility: ModelCompatibilityMatrix,
}
```

**Responsibilities**:
- Download missing tokenizers from HuggingFace Hub with resume capability
- Implement intelligent caching strategy for large tokenizer files
- Support neural network model-specific tokenizer variants (LLaMA-2/3, GPT-2)
- Validate downloaded tokenizers against model requirements

### 3. TokenizerStrategy Resolution System
```rust
pub enum TokenizerStrategy {
    Exact(PathBuf),                          // User-specified path
    Discovered(PathBuf),                     // Auto-discovered colocated
    NeedsDownload(TokenizerDownloadInfo),    // Smart download required
    EmbeddedGguf(Arc<dyn Tokenizer>),       // GGUF-embedded tokenizer
    Mock,                                    // Testing fallback
}
```

**Responsibilities**:
- Provide unified strategy pattern for tokenizer resolution
- Support deterministic behavior with `BITNET_DETERMINISTIC=1`
- Enable strict mode validation with `BITNET_STRICT_TOKENIZERS=1`
- Integrate seamlessly with neural network model wrappers

## Architecture Decisions

### Decision 1: GGUF-First Discovery Strategy
**Choice**: Parse GGUF metadata as primary tokenizer discovery mechanism
**Rationale**:
- GGUF files contain authoritative model architecture information
- Enables zero-configuration inference for supported models
- Maintains compatibility with existing BitNet-rs GGUF infrastructure
- Supports neural network-specific metadata (vocab size, special tokens)

**Alternatives Considered**:
- File extension-based inference: Limited accuracy for neural network models
- Model name parsing: Unreliable across different model distributions
- User configuration files: Adds complexity without clear benefit

### Decision 2: Neural Network Model Compatibility Matrix
**Choice**: Implement hardcoded compatibility matrix for major neural network architectures
**Rationale**:
- Provides reliable tokenizer resolution for LLaMA-2/3, GPT-2, BitNet models
- Enables smart downloading from known-good tokenizer sources
- Supports quantization-aware tokenizer selection (I2S for large vocab, TL1/TL2 for smaller)
- Allows for rapid expansion to new model architectures

**Architecture**:
```rust
pub struct ModelCompatibilityMatrix {
    llama3_128k: TokenizerDownloadInfo,  // LLaMA-3 with 128K vocab
    llama2_32k: TokenizerDownloadInfo,   // LLaMA-2 with 32K vocab
    gpt2_50k: TokenizerDownloadInfo,     // GPT-2 with 50K vocab
    bitnet_custom: TokenizerDownloadInfo, // BitNet-specific tokenizers
}
```

### Decision 3: Async-First Download System
**Choice**: Implement async tokenizer downloading with progress reporting
**Rationale**:
- Large tokenizer files (10MB+) require non-blocking downloads
- Enables progress reporting in `cargo xtask infer` for better UX
- Supports concurrent downloads for multi-file tokenizer packages
- Compatible with BitNet-rs async inference pipeline

### Decision 4: Robust Fallback Chain
**Choice**: Implement comprehensive fallback strategy with actionable error messages
**Rationale**:
- Production reliability requires graceful degradation
- Clear error messages reduce user support overhead
- Maintains backward compatibility with existing `--tokenizer` flag
- Supports both development (mock) and production (strict) modes

**Fallback Order**:
1. GGUF embedded tokenizer (if available)
2. Co-located tokenizer files (tokenizer.json, tokenizer.model)
3. Standard cache directories (~/.cache/bitnet/tokenizers)
4. Smart download from HuggingFace Hub
5. Mock tokenizer (non-strict mode only)

## Performance Implications

### Quantization Pipeline Integration
- **I2S Quantization**: Optimized for large vocabularies with GPU acceleration
- **TL1/TL2 Quantization**: Efficient for smaller vocabularies with lookup tables
- **Memory Efficiency**: Memory-mapped tokenizer files to reduce overhead
- **Token Validation**: Range checking to ensure compatibility with quantization formats

### Neural Network Scale Optimization
- **Large Vocabulary Support**: O(1) token lookup for 128K+ vocabularies
- **Device-Aware Tokenization**: Automatic GPU/CPU selection based on model size
- **Caching Strategy**: Persistent tokenizer caching to minimize repeated downloads
- **Batch Processing**: Vectorized tokenization for improved throughput

## Implementation Risk Mitigation

### Risk 1: Network Dependency
**Mitigation**: Comprehensive offline caching, graceful degradation to cached tokenizers
**Validation**: `BITNET_OFFLINE=1 cargo test` simulates network-free environment

### Risk 2: GGUF Metadata Inconsistency
**Mitigation**: Robust metadata parsing with fallback to architecture-specific defaults
**Validation**: `cargo run -p bitnet-cli -- compat-check <model.gguf>` validates metadata

### Risk 3: Large Vocabulary Performance
**Mitigation**: GPU acceleration for large vocabularies, memory-mapped files
**Validation**: Performance benchmarks with `cargo run -p xtask -- benchmark --vocab-stress-test`

## Integration Points

### xtask CLI Enhancement
```bash
# Zero-configuration inference with automatic discovery
cargo run -p xtask -- infer --model model.gguf --prompt "Test" --auto-download

# Strict mode for production environments
BITNET_STRICT_TOKENIZERS=1 cargo run -p xtask -- infer --model model.gguf --prompt "Test"
```

### Cross-Validation Framework
- Maintain 100% compatibility with existing `UniversalTokenizer`
- Comprehensive test suite against C++ reference implementation
- Quantization accuracy validation for I2S/TL1/TL2 formats

### Feature Flag Integration
- `cpu`: Basic tokenizer discovery with HuggingFace JSON support
- `gpu`: GPU-accelerated tokenization for large vocabularies
- `spm`: SentencePiece tokenizer support
- `ffi`: C++ bridge compatibility for cross-validation

## Success Metrics

1. **Zero-Configuration Success Rate**: >95% of supported models work without manual tokenizer specification
2. **Performance Overhead**: Tokenizer discovery adds <5% overhead to inference pipeline
3. **Network Efficiency**: Smart caching reduces duplicate downloads by >90%
4. **Error Reduction**: Actionable error messages reduce support requests by >80%
5. **Cross-Validation Accuracy**: 100% compatibility with existing UniversalTokenizer

## Conclusion

This tokenizer discovery system provides a production-grade solution for automatic tokenizer resolution in BitNet-rs neural network inference. The architecture balances user experience (zero-configuration), performance (neural network scale optimization), and reliability (comprehensive fallback strategies).

The system maintains backward compatibility while enabling advanced features like smart downloading, quantization-aware selection, and robust error handling. This foundation supports BitNet-rs evolution toward production-ready neural network inference with minimal user configuration overhead.
