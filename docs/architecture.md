# BitNet.rs Architecture

This document provides an explanation of the `BitNet.rs` architecture. It is intended for developers who want to understand the overall structure of the project, how its components are organized, and how they interact.

## Guiding Principles

The architecture of `BitNet.rs` is designed around the following principles:

- **Modularity:** The project is broken down into small, specialized crates, each with a single responsibility. This makes the codebase easier to understand, maintain, and test.
- **Performance:** Key components are designed for high performance, with a focus on zero-cost abstractions, efficient memory management, and the use of optimized kernels.
- **Extensibility:** The modular design allows for new features, backends, or language bindings to be added without disrupting the core functionality.
- **Safety:** The architecture leverages Rust's safety guarantees to minimize the risk of memory-related bugs and concurrency issues.

## Workspace and Crate Structure

`BitNet.rs` is organized as a Rust workspace containing multiple crates. This structure allows for shared dependencies and a unified build process while maintaining clear separation between components.

The crates can be grouped into four main categories:

1.  **Core Library Crates:** The fundamental building blocks of the inference engine.
2.  **Application Crates:** User-facing applications that consume the core libraries.
3.  **Language Binding Crates:** Wrappers that expose the core functionality to other programming languages.
4.  **Testing and Validation Crates:** Crates dedicated to testing, benchmarking, and validation.

### 1. Core Library Crates

These crates form the heart of the `BitNet.rs` engine. They are designed to be used as libraries by other crates in the workspace or by external projects.

| Crate                 | Description                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `bitnet-common`       | Contains shared data structures, types (like `Tensor`), traits, and error-handling utilities used across the entire workspace. It establishes the foundational vocabulary of the project. |
| `bitnet-models`       | Responsible for loading, parsing, and representing the BitNet models. It handles various model formats (GGUF, SafeTensors) and defines the model's structure (e.g., layers, weights). |
| `bitnet-quantization` | Implements the 1-bit quantization algorithms (e.g., I2_S, TL1, TL2). This crate contains the logic for converting and preparing model weights for efficient computation. |
| `bitnet-kernels`      | Provides optimized computation kernels for different hardware architectures. It contains low-level, performance-critical code for operations like matrix multiplication, with backends for CPU (including SIMD) and GPU (CUDA). |
| `bitnet-inference`    | The high-level inference engine. It orchestrates the other core crates to run the model and generate text. It manages the KV cache, implements sampling strategies, and provides both synchronous and asynchronous (streaming) APIs. |
| `bitnet-tokenizers`   | Handles text tokenization and detokenization. It integrates with existing tokenizer formats to convert between raw text and the token IDs that the model understands. |

### 2. Application Crates

These crates provide ready-to-use tools for end-users.

| Crate            | Description                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `bitnet-cli`     | A command-line interface for interacting with the `BitNet.rs` engine. It allows users to run inference, convert models, and perform benchmarks directly from the terminal. |
| `bitnet-server`  | A standalone HTTP server for exposing the inference engine as a web service. It provides an OpenAI-compatible API endpoint for easy integration with existing applications. |

### 3. Language Binding Crates

These crates make it possible to use `BitNet.rs` from other programming languages.

| Crate          | Description                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| `bitnet-ffi`   | A C-compatible Foreign Function Interface (FFI). This allows `BitNet.rs` to be used by any language that can call C functions (e.g., C++, C#, Go). |
| `bitnet-py`    | High-level Python bindings created using PyO3. This crate provides a mature, production-ready Python package that offers a drop-in replacement for previous Python implementations. |
| `bitnet-wasm`  | WebAssembly bindings that allow `BitNet.rs` to run in a web browser or other JavaScript environments. |

### 4. Testing and Validation Crates

These crates are essential for ensuring the correctness and performance of the project.

| Crate          | Description                                                                                                                              |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `bitnet-sys`   | Low-level FFI bindings *to* the original C++ BitNet implementation. This is used exclusively for cross-validation testing. |
| `crossval`     | A comprehensive cross-validation framework that uses `bitnet-sys` to compare the output of `BitNet.rs` against the original C++ implementation, ensuring numerical accuracy and performance parity. |

## High-Level Data Flow

A typical inference request flows through the system as follows:

1.  **Entry Point:** A request is initiated by an **Application** (e.g., `bitnet-server`) or through a **Language Binding** (e.g., `bitnet-py`).
2.  **Inference Engine:** The request is sent to the `bitnet-inference` engine.
3.  **Model and Tokenizer:** The engine uses `bitnet-tokenizers` to convert the input prompt into tokens and `bitnet-models` to access the model's structure and weights.
4.  **Computation:** For each step of the generation process, the inference engine dispatches computation tasks (like matrix multiplication) to `bitnet-kernels`.
5.  **Kernel Execution:** `bitnet-kernels` selects the optimal kernel based on the available hardware (e.g., AVX2 on CPU, or a CUDA kernel on GPU) and executes the computation.
6.  **Sampling and Output:** The result of the computation (logits) is returned to the `bitnet-inference` engine, which uses a sampling strategy to select the next token. The token is then converted back to text by `bitnet-tokenizers`.
7.  **Streaming:** If the request is for streaming, the generated text is yielded back to the caller as soon as it's available. Otherwise, the process repeats until the full response is generated.

## Configuration System

The project uses a unified configuration system, defined in `bitnet-common`. This allows for hierarchical configuration of the model, inference, quantization, and performance settings from a single source (e.g., a TOML file), with support for environment variable overrides. This ensures consistent behavior across all crates.

## Feature Flags

The architecture makes extensive use of Cargo feature flags to manage optional components and optimizations. This allows users to compile a version of the library that is tailored to their specific needs, for example:
- `cpu` vs. `gpu` backends.
- Specific SIMD optimizations (`avx2`, `neon`).
- A `full` feature set for maximum functionality.

This approach reduces binary size and compilation time for users who do not need all features.
