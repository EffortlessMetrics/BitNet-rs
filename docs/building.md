# How to Build BitNet.rs from Source

This guide provides step-by-step instructions for building the `BitNet.rs` project from source. Following these instructions, you can compile the libraries, applications, and run tests on your local machine.

## 1. Prerequisites

Before you begin, ensure you have the following tools installed on your system:

- **Rust:** Version 1.70.0 or later. You can install it from [rust-lang.org](https://www.rust-lang.org/).
- **Cargo:** The Rust package manager, which is included with the Rust installation.
- **Git:** For cloning the repository.
- **C++ Compiler:** A C++ compiler (like GCC, Clang, or MSVC) is required for the cross-validation tests against the original C++ implementation.
- **CUDA Toolkit (Optional):** Version 11.0 or later, if you want to build with GPU support.
- **Python (Optional):** Version 3.8 or later, if you plan to work with the Python bindings.

## 2. Clone the Repository

First, clone the `BitNet.rs` repository from GitHub:

```bash
git clone https://github.com/microsoft/BitNet.git
cd BitNet
```

## 3. Standard Build (CPU Only)

A standard build includes the core libraries and applications with CPU-only support. This is the quickest way to get started.

To create a release build (optimized for performance), run:

```bash
cargo build --release
```

The compiled binaries will be located in the `target/release/` directory. For example, the CLI application will be at `target/release/bitnet-cli`.

## 4. Building with Features

`BitNet.rs` uses feature flags to enable optional functionality. You can customize your build by enabling different features.

### GPU Support (CUDA)

To build with support for NVIDIA GPUs, enable the `gpu` feature. You must have the CUDA Toolkit installed.

```bash
cargo build --release --features gpu
```

### SIMD Optimizations

For better performance on modern CPUs, you can enable SIMD (Single Instruction, Multiple Data) features. The library will auto-detect CPU capabilities at runtime, but you can include the optimized code in your build.

- **For x86-64 CPUs with AVX2 support:**
  ```bash
  cargo build --release --features avx2
  ```
- **For x86-64 CPUs with AVX-512 support:**
  ```bash
  cargo build --release --features avx512
  ```
- **For ARM64 CPUs with NEON support (e.g., Apple Silicon):**
  ```bash
  cargo build --release --features neon
  ```

### Building with All Features

To build with all available features, including GPU and all SIMD optimizations, use the `full` feature flag:

```bash
cargo build --release --features full
```

## 5. Target-Specific Builds for Maximum Performance

For the absolute best performance, you can instruct the Rust compiler to optimize the build for your specific CPU architecture. This can yield significant performance gains but may make the binary non-portable to older CPUs.

Use the `RUSTFLAGS` environment variable to set the `target-cpu`.

- **For your native CPU:**
  ```bash
  RUSTFLAGS="-C target-cpu=native" cargo build --release --features full
  ```
- **For a specific x86-64 architecture (e.g., Haswell for AVX2):**
  ```bash
  RUSTFLAGS="-C target-cpu=haswell" cargo build --release --features "cpu,avx2"
  ```
- **For an x86-64 architecture with AVX-512:**
  ```bash
  RUSTFLAGS="-C target-cpu=skylake-avx512" cargo build --release --features "cpu,avx512"
  ```

## 6. Running Tests

The project includes a comprehensive test suite.

- **Run standard tests (fast, Rust-only):**
  This command runs all tests in the workspace that do not require optional dependencies.
  ```bash
  cargo test --workspace
  ```

- **Run cross-validation tests:**
  This runs tests that compare `BitNet.rs` against the original C++ implementation for numerical accuracy. This requires a C++ compiler.
  ```bash
  cargo test --workspace --features crossval
  ```

## 7. Running Benchmarks

To measure the performance of the inference engine, you can run the built-in benchmarks:

```bash
cargo bench --workspace
```

The benchmark results will be saved in the `target/criterion/` directory.

## 8. Development Tools (`xtask`)

The project includes a set of developer scripts using the `xtask` pattern. These scripts help with common development tasks like downloading models and running complex test workflows. For more information, see the `xtask` section in the main `README.md` or the `CONTRIBUTING.md` guide.

## 9. Troubleshooting Common Build Issues

- **CUDA Not Found:** If you encounter errors related to CUDA when building with the `gpu` feature, ensure:
  1. The CUDA Toolkit is correctly installed and its `bin` directory is in your system's `PATH`.
  2. You have a compatible NVIDIA GPU and the latest drivers.
  3. If issues persist, try building with CPU-only features as a fallback.

- **Compilation Errors:** If you encounter other compilation errors:
  1. Ensure you are using the minimum supported Rust version (1.70.0 or later).
  2. Run `cargo clean` to remove any old build artifacts and try building again.
  3. Check that all prerequisites are installed correctly.
