# BitNet.rs Examples

This directory contains comprehensive examples demonstrating how to integrate BitNet.rs with various Rust ecosystem components and deploy it in different environments.

## Directory Structure

- **integrations/** - Integration examples with popular Rust frameworks and libraries
  - `candle_interop.rs` - Tensor interoperability with Candle
  - `tokenizers_integration.rs` - HuggingFace tokenizers integration
  - `axum_server.rs` - Web service with Axum framework
  - `warp_server.rs` - Web service with Warp framework
  - `actix_server.rs` - Web service with Actix-web framework
  - `tracing_observability.rs` - Observability with tracing and metrics

- **deployment/** - Cloud deployment examples and configurations
  - `aws/` - AWS deployment examples
  - `gcp/` - Google Cloud Platform deployment examples
  - `azure/` - Microsoft Azure deployment examples
  - `docker/` - Docker containerization examples

- **basic/** - Basic usage examples
  - `cpu_inference.rs` - Simple CPU inference example
  - `gpu_inference.rs` - GPU-accelerated inference example
  - `streaming.rs` - Streaming generation example

## Running Examples

Each example includes detailed comments and can be run with:

```bash
cargo run --example <example_name>
```

For examples requiring specific features:

```bash
cargo run --example <example_name> --features="gpu,server"
```

## Prerequisites

Some examples require additional dependencies or services:
- GPU examples require CUDA toolkit
- Server examples may require additional runtime dependencies
- Cloud deployment examples require appropriate cloud credentials