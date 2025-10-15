## Environment Variables Reference

### Inference Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `BITNET_DETERMINISTIC` | Enable deterministic inference | `0` | `BITNET_DETERMINISTIC=1` |
| `BITNET_SEED` | Random seed for deterministic mode | `42` | `BITNET_SEED=42` |
| `RAYON_NUM_THREADS` | Thread count (use `1` for determinism) | auto | `RAYON_NUM_THREADS=1` |
| `BITNET_GGUF` | Model path override | auto-discover `models/` | `BITNET_GGUF=/path/to/model.gguf` |

### Validation Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `BITNET_STRICT_MODE` | Fail on validation warnings (exit code 8) | `0` | `BITNET_STRICT_MODE=1` |
| `BITNET_VALIDATION_GATE` | Validation mode: `none`, `auto`, `policy` | `auto` | `BITNET_VALIDATION_GATE=policy` |
| `BITNET_VALIDATION_POLICY` | Path to validation policy YAML | none | `BITNET_VALIDATION_POLICY=policy.yml` |
| `BITNET_VALIDATION_POLICY_KEY` | Policy key for rules lookup | none | `BITNET_VALIDATION_POLICY_KEY=arch:variant` |

### Testing Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `BITNET_GPU_FAKE` | Override GPU detection: `cuda`, `none` | auto-detect | `BITNET_GPU_FAKE=none` |

### Usage Examples

**Deterministic CPU Inference:**
```bash
export BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 BITNET_SEED=42
cargo run -p xtask -- benchmark --model models/model.gguf --tokens 128
```

**Strict Receipt Verification:**
```bash
BITNET_STRICT_MODE=1 cargo run -p xtask -- verify-receipt ci/inference.json
```

**Custom Model Path:**
```bash
export BITNET_GGUF=/path/to/custom/model.gguf
cargo run -p xtask -- infer --prompt "Hello"
```
