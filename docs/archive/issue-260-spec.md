# Issue #260: Mock Inference Performance Reporting

## Context
BitNet-rs currently reports misleading inference performance of 200.0 tokens/sec through a mock inference path rather than actual quantized neural network computation. The real quantized inference pipeline is blocked by 21+ compilation errors, preventing accurate performance measurement and validation of the 1-bit neural network architecture. This creates false evidence for production readiness and blocks proper benchmarking against the C++ reference implementation.

The issue affects the core BitNet-rs inference pipeline stages:
- **Model Loading**: GGUF integration working but needs QLinear layer replacement
- **Quantization**: I2S (2-bit signed), TL1/TL2 (table lookup) kernels not integrated
- **Kernels**: SIMD/CUDA compute kernels bypassed by mock path
- **Inference**: Autoregressive generation using dummy computation
- **Output**: Performance metrics based on mock rather than real computation

## User Story
As a neural network researcher evaluating BitNet-rs for production deployment, I want accurate performance reporting from real quantized inference computation so that I can make informed decisions about model deployment, compare against baseline implementations, and validate the 1-bit quantization accuracy claims.

## Acceptance Criteria
AC1: Fix all compilation errors blocking real quantized inference execution with proper error context and anyhow::Result patterns
AC2: Implement strict mode environment variable (BITNET_STRICT_MODE=1) that prevents mock fallbacks and fails fast on missing quantization kernels
AC3: Integrate I2S quantization kernels for 2-bit signed weights with device-aware selection (CPU SIMD/GPU CUDA)
AC4: Integrate TL1/TL2 table lookup quantization kernels with memory-efficient lookup tables
AC5: Replace QLinear mock layers with real quantized matrix multiplication using integrated kernels
AC6: Update CI pipeline to reject performance evidence from mock inference paths
AC7: Establish realistic CPU performance baselines (10-20 tokens/sec for I2S quantization)
AC8: Establish realistic GPU performance baselines (50-100 tokens/sec with mixed precision FP16/BF16)
AC9: Cross-validate performance against C++ reference implementation within 5% accuracy tolerance
AC10: Update performance documentation to reflect real quantized compute capabilities

## Technical Implementation Notes
- **Affected crates**: bitnet-quantization (kernel integration), bitnet-inference (mock removal), bitnet-kernels (I2S/TL1/TL2), bitnet-models (QLinear replacement), crossval (baseline validation)
- **Pipeline stages**: All stages affected - model loading needs QLinear integration, quantization needs kernel activation, inference needs mock removal
- **Performance considerations**: Device-aware quantization selection, memory efficiency for large models, deterministic inference with proper seeding, mixed precision GPU acceleration
- **Quantization requirements**: I2S (2-bit signed) for production accuracy, TL1/TL2 table lookup for memory efficiency, cross-validation via `cargo run -p xtask -- crossval`
- **Cross-validation**: C++ reference compatibility for performance and accuracy baseline establishment
- **Feature flags**: CPU/GPU feature compatibility with `--no-default-features --features cpu|gpu` and graceful fallback mechanisms
- **GGUF compatibility**: Full tensor loading working, needs quantized layer replacement for QLinear components
- **Testing strategy**: TDD with `// AC:ID` tags, strict mode validation, CPU/GPU smoke testing, performance regression prevention, benchmark baseline establishment
- **Error handling**: Proper compilation error resolution, strict mode fail-fast behavior, device fallback mechanisms
- **Environment variables**: BITNET_STRICT_MODE=1 for mock prevention, BITNET_DETERMINISTIC=1 for reproducible benchmarks
- **CI integration**: Performance gate updates to reject mock evidence, baseline regression detection
