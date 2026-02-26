# run_crossval_sweep.sh - Cross-Validation Sweep Script

## Overview

`run_crossval_sweep.sh` orchestrates comprehensive cross-validation between BitNet-rs (Rust implementation) and bitnet.cpp (C++ reference) across multiple deterministic test scenarios.

The script runs 3 predefined scenarios with increasing token counts, captures detailed execution traces, compares outputs with the C++ reference (if available), and generates actionable divergence reports.

## Features

### Deterministic Execution
- Sets `BITNET_DETERMINISTIC=1` for reproducible results
- Fixed seed: `BITNET_SEED=42`
- Thread count: `RAYON_NUM_THREADS=4`
- Greedy decoding with `temperature=0.0`

### Comprehensive Tracing
- Enables `BITNET_TRACE_DIR` for each scenario
- Captures 90+ trace files per scenario (tensor activations, Blake3 hashes, RMS statistics)
- Trace files written as JSON for programmatic analysis

### Parity Comparison
- Compares Rust vs C++ token outputs
- Computes cosine similarity of logits
- Detects first divergence position
- Identifies exact match rates

### Graceful Degradation
- Works in Rust-only mode if C++ reference not available
- Provides useful diagnostics even without C++ comparison
- Timeout protection (configurable, default: 180s per scenario)

### Reporting
- Per-scenario reports with detailed metrics
- Summary markdown with actionable recommendations
- Organized output directory structure

## Usage

### Basic Usage

```bash
./scripts/run_crossval_sweep.sh <model.gguf> <tokenizer.json> [output_dir]
```

### Example

```bash
./scripts/run_crossval_sweep.sh \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  /tmp/crossval-sweep
```

### With C++ Reference

```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval
```

### Custom Timeout

```bash
CROSSVAL_TIMEOUT_SECS=300 ./scripts/run_crossval_sweep.sh model.gguf tokenizer.json
```

## Test Scenarios

The script runs 3 predefined scenarios:

| Scenario | Prompt | Max Tokens | Description |
|----------|--------|------------|-------------|
| scenario1 | `2+2=` | 1 | Single token prefill (minimal test) |
| scenario2 | `Hello` | 2 | Two token generation |
| scenario3 | `Count: 1,2,3,` | 4 | Four token generation |

All scenarios use:
- Temperature: 0.0 (greedy decoding)
- Seed: 42 (deterministic)
- No sampling (greedy only)

## Output Structure

```
crossval-results/
├── scenario1/
│   ├── rs-traces/              # Rust trace files (90+ JSON files)
│   │   ├── blk0_attn_norm.trace
│   │   ├── blk0_ffn_norm.trace
│   │   └── ...
│   ├── rs-output.txt           # Rust inference output
│   ├── cpp-output.txt          # C++ inference output (if available)
│   ├── logits-comparison.json  # Parity receipt (if C++ available)
│   └── report.txt              # Scenario-specific report
├── scenario2/
│   └── ...
├── scenario3/
│   └── ...
└── summary.md                  # Final divergence report
```

## Generated Reports

### Scenario Reports

Each scenario generates a `report.txt` with:
- Test configuration (prompt, tokens, model, tokenizer)
- Execution summary (exit codes, trace count)
- Output comparison (token match status)
- Parity metrics (cosine similarity, exact match rate)
- Trace analysis (first diverging trace)
- File locations (all artifacts)
- Actionable recommendations

### Summary Report

The `summary.md` provides:
- Consolidated results table (all scenarios)
- Success criteria checklist
- Divergence detection analysis
- Actionable recommendations based on parity metrics
- Directory structure overview
- Environment configuration details

## Interpreting Results

### Parity Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Cosine Similarity** | ≥ 0.99 | ✅ Excellent parity (green) |
|                       | 0.95-0.99 | ⚠️ Minor divergence (yellow) |
|                       | < 0.95 | ❌ Significant divergence (red) |
| **Exact Match Rate** | 1.0 | ✅ Perfect token match |
|                      | < 1.0 | ❌ Token mismatch - investigate |
| **Token Match** | `ok` | ✅ Full parity with C++ |
|                 | `rust_only` | ℹ️ C++ not available |
|                 | Other | ❌ Parity failure |

### Trace Analysis

Trace files contain:
- **Blake3 hash**: Cryptographic fingerprint of tensor data
- **RMS value**: Root mean square (numerical stability indicator)
- **Shape**: Tensor dimensions
- **Dtype**: Data type (before F32 conversion)
- **Element count**: Total tensor size

Use `jq` to analyze traces:

```bash
# Extract Blake3 hashes
jq -r '.blake3' crossval-results/scenario1/rs-traces/*.trace | sort

# Compare RMS values across scenarios
jq -r '"\(.name): \(.rms)"' crossval-results/scenario*/rs-traces/*.trace | sort

# Find traces with unusual RMS (outside [0.1, 10])
jq 'select(.rms < 0.1 or .rms > 10)' crossval-results/scenario1/rs-traces/*.trace
```

## Dependencies

### Required
- `cargo` - Rust build system
- `bash` - Shell (version 4.0+)

### Optional
- `jq` - JSON parsing (recommended for analysis)
- `timeout` - Command timeout (GNU coreutils)
- C++ reference (`BITNET_CPP_DIR`) - For full parity validation

### Rust Crates
- `bitnet-cli` - Built with `--features cpu,full-cli`
- `bitnet-crossval` - Built with `--features integration-tests,crossval,ffi` (if C++ available)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BITNET_CPP_DIR` | `$HOME/.cache/bitnet_cpp` | Path to C++ reference |
| `CROSSVAL_TIMEOUT_SECS` | `180` | Timeout per scenario (seconds) |
| `BITNET_DETERMINISTIC` | `1` (set by script) | Enable deterministic mode |
| `BITNET_SEED` | `42` (set by script) | Random seed |
| `RAYON_NUM_THREADS` | `4` (set by script) | Thread count |
| `BITNET_TRACE_DIR` | (set by script) | Trace output directory |

## Troubleshooting

### Script Exits with "Model not found"
**Solution:** Verify model path is correct and file exists:
```bash
ls -lh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

### "No trace files generated"
**Possible causes:**
- Inference failed before tracing could start
- `BITNET_TRACE_DIR` not properly set (script should handle this)
- Model loading error

**Solution:** Check `rs-output.txt` for error messages

### "C++ parity test failed"
**Possible causes:**
- C++ reference not built correctly
- Library path issues (LD_LIBRARY_PATH)
- FFI feature not compiled

**Solution:**
1. Verify C++ build: `ls $BITNET_CPP_DIR/build/3rdparty/llama.cpp/src/libllama.so`
2. Check library paths: `ldd target/debug/deps/parity_bitnetcpp-*`
3. Rebuild with FFI: `cargo build --features integration-tests,crossval,ffi -p bitnet-crossval`

### Timeout on QK256 Models
**Cause:** QK256 MVP uses scalar kernels (~0.1 tok/s for 2B models)

**Solution:**
- Increase timeout: `CROSSVAL_TIMEOUT_SECS=600 ./scripts/run_crossval_sweep.sh ...`
- Or use I2_S BitNet32-F16 models for 10-20× faster inference

### Permission Denied
**Solution:** Make script executable:
```bash
chmod +x scripts/run_crossval_sweep.sh
```

## Integration with CI/CD

### Local Validation

```bash
# Run before committing inference changes
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval

# Check summary for failures
cat /tmp/crossval/summary.md

# Fail if any scenario failed
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval || exit 1
```

### CI Pipeline Example

```yaml
- name: Cross-validation sweep
  run: |
    export BITNET_CPP_DIR=/opt/bitnet_cpp
    ./scripts/run_crossval_sweep.sh \
      models/model.gguf \
      models/tokenizer.json \
      ${{ runner.temp }}/crossval
  timeout-minutes: 15

- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: crossval-results
    path: ${{ runner.temp }}/crossval
```

## Related Scripts

- `scripts/parity_smoke.sh` - Quick one-command parity check
- `scripts/crossval.sh` - Traditional cross-validation test runner
- `scripts/validate_gguf.sh` - Model validation (LayerNorm, projection, inference)

## See Also

- [Validation Framework](../docs/development/validation-framework.md)
- [Cross-Validation Guide](../docs/development/test-suite.md)
- [Trace Format Specification](../crates/bitnet-trace/README.md)
- [CLAUDE.md](../CLAUDE.md) - Project guide

## License

MIT OR Apache-2.0 (same as BitNet-rs project)
