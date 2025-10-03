# How-To: Deterministic Inference Setup

This guide shows you how to configure BitNet.rs for reproducible, deterministic neural network inference. Deterministic inference is essential for testing, validation, cross-validation with reference implementations, and ensuring consistent behavior across different hardware.

## Prerequisites

- BitNet.rs installed with CPU features: `--no-default-features --features cpu`
- A valid GGUF model file
- Basic familiarity with environment variables

## Quick Start

### 1. Enable Deterministic Mode

Set the following environment variables before running inference:

```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

### 2. Run Inference

```bash
cargo run -p bitnet-cli -- infer \
  --model path/to/model.gguf \
  --prompt "The future of AI is" \
  --max-tokens 50 \
  --seed 42
```

### 3. Verify Determinism

Run the same command twice and verify identical outputs:

```bash
# Run 1
OUTPUT1=$(cargo run -p bitnet-cli -- infer \
  --model path/to/model.gguf \
  --prompt "Test" \
  --max-tokens 20 \
  --seed 42)

# Run 2
OUTPUT2=$(cargo run -p bitnet-cli -- infer \
  --model path/to/model.gguf \
  --prompt "Test" \
  --max-tokens 20 \
  --seed 42)

# Compare (should be identical)
diff <(echo "$OUTPUT1") <(echo "$OUTPUT2")
```

## Environment Variables

### Required Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `BITNET_DETERMINISTIC` | `1` | Enable deterministic mode (disables non-deterministic optimizations) |
| `BITNET_SEED` | `42` (or any u64) | Seed for random number generator (affects sampling) |
| `RAYON_NUM_THREADS` | `1` | Single-threaded execution (prevents thread-order non-determinism) |

### Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BITNET_BATCH_SIZE` | `1` | Batch size (set to 1 for determinism) |
| `BITNET_CACHE_SIZE` | Auto | KV-cache size (fixed size improves determinism) |

## Programmatic Configuration

### Rust API

```rust
use bitnet::{BitNetModel, GenerationConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set environment variables programmatically
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    std::env::set_var("BITNET_SEED", "42");
    std::env::set_var("RAYON_NUM_THREADS", "1");

    // Load model
    let model = BitNetModel::from_file("model.gguf").await?;

    // Configure deterministic generation
    let config = GenerationConfig {
        seed: Some(42),
        max_new_tokens: 50,
        temperature: 0.8,
        do_sample: true,
        ..Default::default()
    };

    // Generate with determinism
    let result1 = model.generate("Test prompt", &config).await?;
    let result2 = model.generate("Test prompt", &config).await?;

    // Verify identical outputs
    assert_eq!(result1.token_ids, result2.token_ids);
    println!("Deterministic generation validated!");

    Ok(())
}
```

### Python Bindings (if available)

```python
import os
import bitnet

# Configure determinism
os.environ['BITNET_DETERMINISTIC'] = '1'
os.environ['BITNET_SEED'] = '42'
os.environ['RAYON_NUM_THREADS'] = '1'

# Load model
model = bitnet.BitNetModel.from_file("model.gguf")

# Generate with determinism
config = bitnet.GenerationConfig(
    seed=42,
    max_new_tokens=50,
    temperature=0.8
)

result1 = model.generate("Test prompt", config)
result2 = model.generate("Test prompt", config)

assert result1.token_ids == result2.token_ids
print("Deterministic generation validated!")
```

## Testing Determinism

### Integration Test

Create a test to validate deterministic behavior:

```rust
#[cfg(test)]
mod determinism_tests {
    use super::*;

    // AC:6 - Determinism integration test
    #[tokio::test]
    #[serial_test::serial]
    async fn test_deterministic_inference_identical_runs() -> Result<()> {
        // Configure deterministic environment
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("BITNET_SEED", "42");
        std::env::set_var("RAYON_NUM_THREADS", "1");

        let model = load_test_model()?;
        let config = GenerationConfig {
            seed: Some(42),
            max_new_tokens: 50,
            temperature: 1.0,
            do_sample: true,
            ..Default::default()
        };

        let input_ids = vec![1, 2, 3, 4, 5];

        // Run 1
        let tokens1 = model.generate_tokens(&input_ids, &config).await?;

        // Run 2
        let tokens2 = model.generate_tokens(&input_ids, &config).await?;

        // AC6: Identical sequences
        assert_eq!(tokens1, tokens2, "Deterministic inference produced different tokens");
        assert!(tokens1.len() > 10, "Generation too short to validate determinism");

        Ok(())
    }
}
```

### Validation Script

Create a shell script to validate determinism:

```bash
#!/bin/bash
# scripts/validate-determinism.sh

set -euo pipefail

# Configure determinism
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

MODEL_PATH="${1:-models/test-model.gguf}"
PROMPT="${2:-The future of AI is}"
MAX_TOKENS="${3:-50}"

echo "Validating deterministic inference..."
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"

# Run inference twice
OUTPUT1=$(cargo run -p bitnet-cli -- infer \
  --model "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --seed 42)

OUTPUT2=$(cargo run -p bitnet-cli -- infer \
  --model "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --seed 42)

# Compare outputs
if [ "$OUTPUT1" == "$OUTPUT2" ]; then
    echo "✓ Deterministic inference validated"
    exit 0
else
    echo "✗ Outputs differ - determinism failed"
    echo "Output 1:"
    echo "$OUTPUT1"
    echo ""
    echo "Output 2:"
    echo "$OUTPUT2"
    exit 1
fi
```

## Common Issues

### Issue: Outputs Still Differ

**Symptoms**: Two runs with same seed produce different results

**Solutions**:

1. **Verify environment variables are set**:
   ```bash
   echo $BITNET_DETERMINISTIC  # Should be "1"
   echo $BITNET_SEED           # Should be "42" (or your seed)
   echo $RAYON_NUM_THREADS     # Should be "1"
   ```

2. **Check for parallel operations**:
   - Ensure `RAYON_NUM_THREADS=1`
   - Avoid concurrent inference requests

3. **Validate model loading**:
   ```bash
   cargo run -p bitnet-cli -- compat-check model.gguf
   ```

4. **Use greedy sampling for debugging**:
   ```rust
   let config = GenerationConfig {
       temperature: 0.0,  // Greedy (deterministic by default)
       seed: Some(42),
       ..Default::default()
   };
   ```

### Issue: Performance Degradation

**Symptoms**: Deterministic mode is slower than normal inference

**Expected**: Deterministic mode trades performance for reproducibility

**Mitigations**:

1. **Use determinism only for testing/validation**:
   - Enable for CI/CD validation
   - Disable for production inference

2. **Profile to identify bottlenecks**:
   ```bash
   cargo run --release -p bitnet-cli -- infer \
     --model model.gguf \
     --prompt "Test" \
     --profile
   ```

3. **Use receipt artifacts to verify non-deterministic inference**:
   ```bash
   # Normal inference with receipts
   cargo run -p bitnet-cli -- infer --model model.gguf --prompt "Test"

   # Verify real inference path
   cat ci/inference.json | jq '.compute_path'  # Should be "real"
   ```

### Issue: Seed Not Applied

**Symptoms**: Different seeds produce identical outputs

**Solutions**:

1. **Verify sampling is enabled**:
   ```rust
   let config = GenerationConfig {
       do_sample: true,  // Required for seed to affect output
       seed: Some(42),
       temperature: 1.0,  // Non-zero for sampling
       ..Default::default()
   };
   ```

2. **Check temperature**:
   - Temperature 0.0 = greedy (seed ignored)
   - Temperature > 0.0 = sampling (seed applied)

## Cross-Validation

Use deterministic inference for cross-validation with C++ reference:

```bash
# Configure determinism
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
export BITNET_GGUF="path/to/model.gguf"

# Run cross-validation
cargo run -p xtask -- crossval

# Check tolerance
# I2S: 1e-5 MSE vs FP32
# TL1/TL2: 1e-4 MSE vs FP32
```

## Receipt Validation

Verify deterministic inference produces valid receipts:

```bash
# Run inference
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo test --workspace --no-default-features --features cpu

# Check receipt
cat ci/inference.json | jq '{
  compute_path,
  deterministic,
  environment: .environment | {
    BITNET_DETERMINISTIC,
    BITNET_SEED,
    RAYON_NUM_THREADS
  }
}'

# Should output:
# {
#   "compute_path": "real",
#   "deterministic": true,
#   "environment": {
#     "BITNET_DETERMINISTIC": "1",
#     "BITNET_SEED": "42",
#     "RAYON_NUM_THREADS": "1"
#   }
# }
```

## Best Practices

1. **Always set all three environment variables together**:
   ```bash
   export BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1
   ```

2. **Use consistent seeds across test runs**:
   - Seed 42 is conventional for reproducibility

3. **Document determinism requirements in tests**:
   ```rust
   /// This test requires deterministic execution:
   /// BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1
   #[test]
   fn test_deterministic_behavior() { /* ... */ }
   ```

4. **Validate determinism in CI/CD**:
   ```yaml
   # .github/workflows/determinism-validation.yml
   - name: Validate Deterministic Inference
     run: |
       export BITNET_DETERMINISTIC=1
       export BITNET_SEED=42
       export RAYON_NUM_THREADS=1
       cargo test --workspace --no-default-features --features cpu test_deterministic_inference_identical_runs
   ```

5. **Use receipt artifacts for non-deterministic inference**:
   - Determinism is for testing/validation
   - Production inference can use parallel execution
   - Receipts provide evidence of real inference paths

## Related Documentation

- [Issue #254 Specification](../explanation/issue-254-real-inference-spec.md) - Real inference requirements
- [Performance Benchmarking](../performance-benchmarking.md) - Receipt-backed performance claims
- [API Reference](../reference/api-reference.md#deterministic-inference) - Programmatic configuration
- [Test Suite Guide](../development/test-suite.md) - Determinism testing patterns

## Summary

Deterministic inference in BitNet.rs requires:

1. **Environment Configuration**: `BITNET_DETERMINISTIC=1`, `BITNET_SEED=42`, `RAYON_NUM_THREADS=1`
2. **Generation Config**: `seed: Some(42)`, `do_sample: true`
3. **Validation**: Run twice, compare outputs
4. **Receipt Artifacts**: Verify `deterministic: true` in `ci/inference.json`

This ensures reproducible results across runs, essential for testing, cross-validation, and ensuring correctness of real neural network inference.
