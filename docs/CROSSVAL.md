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

## Parity Harness (Rust ↔ bitnet.cpp)

The parity harness validates that the Rust inference engine produces identical outputs to Microsoft's BitNet C++ implementation for deterministic inference. This ensures correctness and helps catch regressions.

### Prerequisites

1. **Model**: Fetch a test model using the lockfile approach
2. **BitNet C++ (optional)**: For full parity validation, set `BITNET_CPP_DIR`

### Running Parity Tests

**Rust-only validation** (no C++ comparison):
```bash
# Fetch model
cargo run -p xtask -- fetch-models --lock crossval-models.lock.json | tee /tmp/fetch.json
export CROSSVAL_GGUF=$(jq -r '.local // .[0].local' /tmp/fetch.json)

# Run parity test
cargo test -p bitnet-crossval --features crossval,integration-tests -- parity_bitnetcpp
```

**Full parity validation** (with C++ comparison):
```bash
# Set up environment
export CROSSVAL_GGUF=$(jq -r '.local // .[0].local' /tmp/fetch.json)
export BITNET_CPP_DIR=/path/to/bitnetcpp/build

# Run parity test
cargo test -p bitnet-crossval --features crossval,integration-tests,cpu -- parity_bitnetcpp
```

### Custom Prompts

Override the default test prompt:
```bash
export CROSSVAL_PROMPT="Explain photosynthesis in one sentence."
cargo test -p bitnet-crossval --features crossval,integration-tests -- parity_bitnetcpp
```

Or use predefined prompts from `crossval/prompts.yaml`:
```bash
export CROSSVAL_PROMPT="$(yq '.parity_prompts.math.text' crossval/prompts.yaml)"
cargo test -p bitnet-crossval --features crossval,integration-tests -- parity_bitnetcpp
```

### Parity Receipts

The parity test writes a receipt to `docs/baselines/<date>/parity-bitnetcpp.json` containing:
- **Timestamp** and **commit hash**
- **Rust engine outputs**: tokens, logits, greedy decode results
- **Parity metrics** (when C++ is available): cosine similarity, exact match rate
- **Template detection**: which prompt template was used
- **Validation status**: production vs mock, deterministic flags

Example receipt:
```json
{
  "timestamp": "2025-01-16T10:30:00Z",
  "commit": "a606a0d2",
  "model_path": "/home/user/.cache/bitnet/models/.../model.gguf",
  "template": "instruct",
  "prompt": "Q: 2+2? A:",
  "rust": {
    "token_count": 5,
    "vocab_size": 50257,
    "decoded_tokens": [657, 604],
    "n_steps": 8
  },
  "parity": {
    "cpp_available": false,
    "status": "rust_only"
  },
  "validation": {
    "rust_engine": "production",
    "deterministic": true
  }
}
```

### Deterministic Inference

The parity harness enforces deterministic inference:
- **Single-threaded**: `RAYON_NUM_THREADS=1`
- **Greedy sampling**: temperature = 0.0
- **Fixed seed**: seed = 0
- **Template-aware BOS**: Respects template's `should_add_bos()` policy
- **Token-level EOT**: Uses `<|eot_id|>` for LLaMA-3 models

### Troubleshooting

**"CROSSVAL_GGUF not set"**
- Run `cargo run -p xtask -- fetch-models` first
- Or set `CROSSVAL_GGUF` manually to a model path

**"BITNET_CPP_DIR not set"**
- Rust-only validation will run (no C++ comparison)
- To enable C++ parity, build bitnet.cpp and set the env var

**"C++ library available but FFI not yet integrated"**
- The C++ shim exists but build.rs needs updating
- This is expected in the current implementation
- Track issue for build.rs integration

## See Also

- [Validation Framework](development/validation-framework.md)
- [Test Suite](development/test-suite.md)
- [Parity Prompts](../crossval/prompts.yaml)
