# Advanced Examples

This directory contains complex examples demonstrating advanced BitNet.rs features and integration patterns.

## Examples

- **`ffi_threading_demo.rs`** - FFI bridge threading and safety demonstration
- **`enhanced_error_demo.rs`** - Comprehensive error handling and recovery
- **`monitoring_demo.rs`** - Performance monitoring and observability
- **`prefill_performance_demo.rs`** - Prefill optimization and benchmarking

## Running Examples

```bash
# FFI threading demo (requires C++ bridge)
cargo run --example ffi_threading_demo --no-default-features --features cpu,ffi

# Error handling demonstration
cargo run --example enhanced_error_demo --no-default-features --features cpu

# Monitoring and metrics
cargo run --example monitoring_demo --no-default-features --features cpu
```

## Prerequisites

- All basic example prerequisites
- For FFI examples: C++ compiler and linker setup
- For monitoring examples: Additional dependencies may be required
- Understanding of concurrent programming for threading examples