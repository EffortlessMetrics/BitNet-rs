# Issue #218: MVP Requirement: Real BitNet Model Integration and Validation

## Context
BitNet.rs currently relies on mock models and placeholder data for testing and examples, creating a gap between testing infrastructure and production-ready neural network inference. The MVP requires end-to-end validation with actual BitNet model artifacts to ensure the 1-bit neural network inference pipeline works correctly with real-world models. This affects the complete inference pipeline: Model Loading → Quantization → Kernels → Inference → Output, and impacts model compatibility validation, numerical accuracy verification, and cross-validation with the C++ reference implementation.

The existing infrastructure includes `cargo xtask download-model` and `cargo xtask full-crossval`, but lacks integration with examples and comprehensive end-to-end testing. This creates risks for production deployment where inference accuracy and model compatibility are critical for enterprise-scale neural network applications.

## User Story
As a BitNet.rs developer, I want to validate the complete inference pipeline with real BitNet models so that I can ensure production-ready neural network inference with verified accuracy and compatibility.

## Acceptance Criteria

AC1: Real BitNet models download and load successfully through existing xtask infrastructure without requiring mock fallbacks

AC2: Examples and CLI tools support both real and mock models with feature-gated selection for development and production workflows

AC3: Complete inference pipeline processes real models through all stages (Model Loading → Quantization → Kernels → Inference → Output) with measurable performance metrics

AC4: Text generation with real models produces coherent, contextually appropriate outputs validated against expected neural network behavior

AC5: Tokenization pipeline correctly processes real model vocabulary and produces token sequences compatible with model expectations

AC6: GGUF compatibility validation passes for real BitNet models with proper tensor alignment and metadata verification

AC7: Cross-validation framework compares BitNet.rs outputs with C++ reference implementation within configurable numerical tolerance

AC8: Perplexity calculations on real models match expected values and validate quantization accuracy preservation

AC9: CI integration supports both mock model testing (fast) and real model validation (gated) with proper feature flags

AC10: Performance benchmarks demonstrate acceptable inference latency and throughput with real models on both CPU and GPU backends

## Technical Implementation Notes

- **Affected crates**: bitnet-models (GGUF loading), bitnet-inference (engine), bitnet-quantization (I2S/TL1/TL2), bitnet-tokenizers (universal tokenizer), bitnet-cli (examples), xtask (model management)
- **Pipeline stages**: All stages affected - model loading requires real GGUF parsing, quantization needs actual weight tensors, kernels process real data, inference generates authentic outputs
- **Performance considerations**: Real models significantly larger than mocks, GPU acceleration essential for acceptable performance, memory optimization critical for large model handling
- **Quantization requirements**: I2S, TL1, TL2 accuracy validation with real weights, cross-validation against C++ reference for numerical correctness
- **Cross-validation**: C++ BitNet compatibility for inference output parity, numerical tolerance configuration, perplexity metric validation
- **Model compatibility**: GGUF format validation, tensor alignment verification, metadata consistency checks, error handling for malformed models
- **Feature gating**: `--features inference` for real inference, mock fallbacks for development, CI optimization with model download caching
- **Testing strategy**: Smoke tests with mock models (fast CI), integration tests with real models (gated), performance benchmarks (local development)
- **Infrastructure integration**: Leverage existing `cargo xtask download-model` and `cargo xtask full-crossval`, extend with example integration and CI workflows
