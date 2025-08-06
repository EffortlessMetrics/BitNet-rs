# CI Helper Scripts

This directory contains scripts for build automation and external dependency management.

## Scripts

- `fetch_bitnet_cpp.sh` - Downloads and builds Microsoft's BitNet.cpp for cross-validation
- `apply_patches.sh` - Applies minimal patches if needed for FFI compatibility
- `bump_bitnet_tag.sh` - Updates pinned version of external dependencies

## Usage

These scripts are primarily used by CI/CD workflows but can be run locally for development and testing.

## External Dependencies

External dependencies are cached in `$HOME/.cache/bitnet_cpp/` to avoid recompilation.