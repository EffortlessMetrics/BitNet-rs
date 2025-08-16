# BitNet.rs Overview

This document provides a high-level overview of the BitNet.rs project, its architecture, and its core concepts.

## Philosophy

BitNet.rs is a production-ready implementation of the BitNet 1-bit Large Language Model inference architecture. The project is built on the following principles:

- **Performance**: To be the fastest and most memory-efficient BitNet implementation available.
- **Safety**: To leverage the safety guarantees of the Rust programming language to provide a robust and reliable inference solution.
- **Compatibility**: To provide drop-in compatibility with existing BitNet implementations in Python and C++.
- **Extensibility**: To provide a modular and extensible architecture that can be easily adapted to new hardware and use cases.

## Architecture

The BitNet.rs project is organized into a collection of crates, each responsible for a specific part of the inference stack.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │    │   Bindings      │    │   Interfaces    │
│                 │    │                 │    │                 │
│ • CLI Tool      │    │ • Python (PyO3) │    │ • C API         │
│ • Web Services  │    │ • JavaScript    │    │ • WebAssembly   │
│ • Desktop Apps  │    │ • Rust Native   │    │ • REST API      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                        BitNet Rust Core                           │
├─────────────────────────────────┼─────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│ │ Inference       │  │ Models          │  │ Quantization    │   │
│ │ • CPU Engine    │  │ • BitNet        │  │ • I2S           │   │
│ │ • GPU Engine    │  │ • Transformers  │  │ • TL1/TL2       │   │
│ │ • Streaming     │  │ • Model Loading │  │ • Dynamic       │   │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│ │ Kernels         │  │ Memory          │  │ Device          │   │
│ │ • SIMD (CPU)    │  │ • Allocators    │  │ • CPU           │   │
│ │ • CUDA (GPU)    │  │ • KV Cache      │  │ • CUDA          │   │
│ │ • Metal (macOS) │  │ • Memory Pool   │  │ • Metal         │   │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

The core components of the architecture are:

- **`bitnet-common`**: Common types and error handling for the entire workspace.
- **`bitnet-models`**: Model definitions and loading for BitNet inference.
- **`bitnet-quantization`**: Quantization algorithms for BitNet models.
- **`bitnet-inference`**: The inference engine, which combines the model, tokenizer, and backend to run inference.
- **`bitnet-tokenizers`**: Tokenization support for BitNet models.
- **`bitnet-kernels`**: Low-level compute kernels for CPU and GPU.
- **Bindings**: C, Python, and WebAssembly bindings for the core library.
- **Applications**: A CLI tool and example web services that use the core library.

## Core Concepts

### Models

BitNet.rs supports multiple model formats and architectures, including GGUF, SafeTensors, and direct loading from the HuggingFace Hub. The `Model` trait provides a common interface for all model implementations.

### Quantization

Quantization is the process of reducing the precision of the model's weights to reduce memory usage and improve performance. BitNet.rs supports several quantization schemes, including 1.58-bit, 2-bit, and 4-bit quantization.

### Inference Engine

The inference engine is responsible for running the model and generating text. It supports both CPU and GPU backends, and provides a high-level API for text generation, including streaming support.

### Memory Management

BitNet.rs includes several features for optimizing memory usage, including a KV cache for storing intermediate results, and support for memory-mapped files to reduce memory usage when loading large models.
