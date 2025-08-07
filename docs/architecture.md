# Architecture

This document provides a high-level overview of the BitNet.rs architecture. It is intended for developers and contributors who want to understand the structure of the project and how the different components interact.

The design emphasizes modularity, performance, and a clean separation of concerns, allowing for independent development and testing of each part of the system.

## Modular System Architecture

The following diagram illustrates the high-level components of BitNet.rs and their relationships:

```mermaid
graph TB
    subgraph "User Interfaces"
        CLI[bitnet-cli]
        CAPI[bitnet-ffi]
        PyAPI[bitnet-py]
        WASM[bitnet-wasm]
        Server[bitnet-server]
    end

    subgraph "Core Inference Layer"
        IE[bitnet-inference]
        TOK[bitnet-tokenizers]
    end

    subgraph "Model Layer"
        MOD[bitnet-models]
        QUANT[bitnet-quantization]
    end

    subgraph "Compute Layer"
        KERN[bitnet-kernels]
        CPU[CPU Kernels: AVX2/NEON]
        GPU[GPU Kernels: CUDA/cudarc]
        FALL[Fallback Kernels]
    end

    subgraph "Foundation"
        COMMON[bitnet-common]
    end

    subgraph "External Formats"
        GGUF[GGUF Files]
        ST[SafeTensors]
        HF[HuggingFace]
    end

    CLI --> IE
    CAPI --> IE
    PyAPI --> IE
    WASM --> IE

    IE --> MOD
    IE --> QUANT
    MOD --> KERN
    QUANT --> KERN

    KERN --> CPU
    KERN --> GPU
    KERN --> FALL

    MOD --> GGUF
    MOD --> ST
    MOD --> HF

    CPU --> COMMON
    GPU --> COMMON
    FALL --> COMMON
    Server --> IE

    IE --> MOD
    IE --> TOK
    IE --> KERN

    MOD --> QUANT
    MOD --> COMMON

    QUANT --> KERN
    QUANT --> COMMON

    KERN --> COMMON

    TOK --> COMMON

    MOD --> GGUF
    MOD --> ST
    MOD --> HF
```

## Modular Workspace Structure

The repository is organized as a Rust workspace with multiple specialized crates. This structure promotes code reuse, clear ownership, and efficient builds.

```
bitnet-rs/
├── Cargo.toml                 # Workspace root
├── README.md                  # Main project README
├── .github/                   # CI/CD workflows
├── crates/
│   ├── bitnet-common/        # Shared types and utilities
│   ├── bitnet-models/        # Model definitions and loading
│   ├── bitnet-quantization/  # Quantization algorithms
│   ├── bitnet-kernels/       # High-performance compute kernels
│   ├── bitnet-inference/     # Inference engines
│   ├── bitnet-tokenizers/    # Tokenization support
│   ├── bitnet-server/        # HTTP server implementation
│   ├── bitnet-cli/           # Command-line interface
│   ├── bitnet-ffi/           # C API bindings
│   ├── bitnet-py/            # Python bindings
│   └── bitnet-wasm/          # WebAssembly bindings
├── examples/                 # Usage examples
├── tests/                    # Integration tests
├── benches/                  # Comprehensive benchmarks
└── docs/                     # Project documentation
```
