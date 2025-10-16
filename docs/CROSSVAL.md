# Cross-Validation Guide

This document describes how to use BitNet.rs cross-validation tools to ensure compatibility and correctness against reference implementations.

## Fetching test models (no repo binaries)

BitNet.rs uses a lockfile-based approach to fetch models deterministically without committing large binary files to the repository.

### Quick Start

```bash
cargo run -p xtask -- fetch-models --lock crossval-models.lock.json
```

### Output

The command prints JSON with model information:

```json
{
  "id": "microsoft/bitnet-b1.58-2B-4T-gguf@ggml-model-i2_s.gguf",
  "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
  "local": "/home/user/.cache/bitnet/models/<sha>/model.gguf",
  "status": "downloaded"
}
```

### Cache Location

Models are cached in `~/.cache/bitnet/models/<sha256>/model.gguf` for deterministic retrieval. The SHA256 hash ensures integrity and deduplication.

### Lockfile Format

The lockfile (`crossval-models.lock.json`) contains:

```json
[
  {
    "id": "model-identifier",
    "sha256": "expected-sha256-hash",
    "bytes": 12345,
    "urls": ["https://..."],
    "license": "license-name"
  }
]
```

### Using in Cross-Validation

Set environment variables to use cached models:

```bash
# Fetch model
cargo run -p xtask -- fetch-models --lock crossval-models.lock.json

# Use in parity/bench tests
export CROSSVAL_GGUF="/home/user/.cache/bitnet/models/<sha>/model.gguf"
cargo test -p crossval --features crossval-bitnetcpp
```

## Cross-Validation Workflow

1. **Fetch models**: `cargo run -p xtask -- fetch-models --lock crossval-models.lock.json`
2. **Build C++ reference**: `cargo run -p xtask -- fetch-cpp`
3. **Run cross-validation**: `cargo run -p xtask -- crossval`
4. **Review results**: Check reports in `crossval/reports/`

## See Also

- [Validation Framework](development/validation-framework.md)
- [Test Suite](development/test-suite.md)
