# CI Helper Scripts

This directory contains scripts for build automation and external dependency management.

## Scripts

- `fetch_bitnet_cpp.sh` - Downloads and builds Microsoft's BitNet.cpp for cross-validation
- `apply_patches.sh` - Applies minimal patches if needed for FFI compatibility
- `bump_bitnet_tag.sh` - Updates pinned version of external dependencies

## Usage

These scripts are primarily used by CI/CD workflows but can be run locally for development and testing.

## Compiler Matrix Testing

The CI system supports cross-compiler compatibility testing with matrix builds:

- **GCC builds**: Use `CC=gcc` and `CXX=g++` environment variables
- **Clang builds**: Use `CC=clang` and `CXX=clang++` environment variables
- **FFI Smoke Build**: Tests both compiler toolchains in parallel to ensure compatibility

The matrix testing ensures that FFI components build successfully with both GCC and Clang compilers.

## External Dependencies

External dependencies are cached in `$HOME/.cache/bitnet_cpp/` to avoid recompilation.
