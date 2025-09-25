# Issue #249: Complete Tokenizer Integration and Automatic Discovery

## Context

BitNet.rs currently has incomplete tokenizer integration that blocks real neural network model inference. The existing system requires manual tokenizer specification and lacks automatic discovery mechanisms, forcing users to rely on mock tokenizers for testing. This limitation prevents running production-ready neural network models and impacts the entire inference pipeline from Model Loading → Quantization → Kernels → Inference → Output.

The current `bitnet-tokenizers` crate provides basic infrastructure with `UniversalTokenizer`, `HfTokenizer`, `SpmTokenizer` (feature-gated), and `MockTokenizer` implementations. However, it lacks:
- Automatic tokenizer discovery from GGUF model metadata
- Smart downloading capabilities for missing tokenizer files
- Robust fallback strategies beyond mock implementations
- Seamless integration with the cargo xtask workflow
- Production-ready tokenizer strategy implementations

This affects BitNet.rs neural network inference performance as tokenizers are critical for proper text preprocessing in large language models (LLaMA-2: 32000 vocab, LLaMA-3: 128256 vocab, GPT-2: 50257 vocab).

## User Story

As a BitNet.rs developer, I want automatic tokenizer discovery and smart downloading so that I can run real neural network model inference without manual tokenizer configuration or relying on mock implementations.

## Acceptance Criteria

AC1: Implement `TokenizerDiscovery` that automatically detects tokenizer type and paths from GGUF model metadata without manual configuration

AC2: Create `SmartTokenizerDownload` that automatically downloads missing tokenizer files (tokenizer.json, tokenizer.model) when referenced in model metadata

AC3: Develop production-ready `TokenizerStrategy` implementations that handle LLaMA-2 (32000 vocab), LLaMA-3 (128256 vocab), and GPT-2 (50257 vocab) with proper special token handling

AC4: Integrate automatic tokenizer discovery with `cargo xtask infer` command to enable seamless model inference without manual tokenizer specification

AC5: Add fallback strategy system that tries GGUF metadata → local files → download → mock (only in non-strict mode) with proper error reporting

AC6: Implement cross-validation tests that verify tokenizer compatibility against existing universal tokenizer architecture using `cargo test --no-default-features --features cpu`

AC7: Create integration tests with real model files that validate end-to-end tokenizer discovery and inference pipeline using `cargo run -p xtask -- verify --model <path>`

AC8: Add documentation and examples showing automatic tokenizer usage patterns with different neural network model types

AC9: Ensure deterministic tokenizer behavior across runs when `BITNET_DETERMINISTIC=1` environment variable is set for reproducible inference

AC10: Implement proper error handling with `anyhow::Result<T>` patterns that provide actionable error messages when tokenizer discovery fails

## Technical Implementation Notes

- **Affected crates**: bitnet-tokenizers (primary), bitnet-models (GGUF integration), bitnet-inference (tokenizer usage), xtask (CLI integration)
- **Pipeline stages**: Model Loading (tokenizer discovery from GGUF metadata), Inference (seamless tokenizer usage)
- **Performance considerations**: O(1) token lookup for large vocabulary models, memory-efficient tokenizer loading, GPU/CPU device-aware tokenization
- **Quantization requirements**: Compatible with I2_S, TL1, TL2 quantization types and cross-validation via `cargo run -p xtask -- crossval`
- **Cross-validation**: Maintain compatibility with existing universal tokenizer architecture and C++ reference implementation
- **Feature flags**: Ensure proper CPU/GPU feature flag handling (`--no-default-features --features cpu|gpu`) with SentencePiece support (`--features spm`)
- **GGUF compatibility**: Parse tokenizer metadata from GGUF files, validate tensor alignment, support model loading via `cargo run -p xtask -- verify --model <path>`
- **Testing strategy**: TDD with `// AC:ID` tags, CPU/GPU smoke testing, integration tests with real models, benchmark baseline establishment
- **Environment variables**: Support `BITNET_STRICT_TOKENIZERS=1` (no mock fallbacks), `BITNET_DETERMINISTIC=1` (reproducible runs), `SPM_MODEL` (test fixture paths)
- **Neural Network Scale**: Efficient handling of large vocabulary models (128K+ tokens), support for mixed precision tokenization, device-aware optimization
- **Universal Tokenizer Integration**: Extend existing `UniversalTokenizer::from_gguf()` and `TokenizerBuilder::from_pretrained()` APIs
- **Download Strategy**: Implement smart downloading with HuggingFace Hub integration, local caching, and resume capability
- **Fallback Chain**: GGUF metadata → local file discovery → smart download → mock fallback (non-strict mode only)
- **Model Compatibility**: Support for BitNet models, LLaMA variants, GPT-2/3 architectures with proper vocab size validation