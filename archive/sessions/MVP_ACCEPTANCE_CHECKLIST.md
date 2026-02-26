# CPU MVP Acceptance Checklist

This checklist provides the acceptance criteria for the BitNet-rs CPU inference Minimum Viable Product (MVP). Use this to validate that the inference pipeline meets production quality standards.

## Pre-Flight Checks

- [ ] Model file exists and is accessible
- [ ] Tokenizer file exists (or model has embedded tokenizer)
- [ ] Build completes with `--no-default-features --features cpu`
- [ ] All workspace tests pass: `cargo test --workspace --no-default-features --features cpu`

## 1. Model Loading and Validation

### 1.1 GGUF Parsing
- [ ] GGUF header is read successfully
- [ ] Version is in supported range (1-3)
- [ ] Tensor count is reasonable (<10M)
- [ ] KV pair count is reasonable (<10M)
- [ ] Model architecture is detected correctly

**Test Command**:
```bash
cargo run -p bitnet-cli -- compat-check model.gguf
```

**Success Criteria**:
- Exit code 0
- Output shows "✓ Valid GGUF"
- Version, tensor count, and KV count are displayed

### 1.2 LayerNorm Validation (Strict Mode)
- [ ] Strict mode detects suspicious LayerNorm gamma weights
- [ ] Warning or error is issued for quantized LN weights
- [ ] RMS statistics are computed correctly
- [ ] Suspicious weights are identified (RMS outside [0.5, 2.0])

**Test Command**:
```bash
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --model model.gguf
```

**Success Criteria**:
- If LN weights are bad: warning/error issued
- If LN weights are good: inspection passes without warnings
- RMS statistics are logged (if `--ln-stats` flag exists)

### 1.3 Non-Strict Mode
- [ ] Non-strict mode issues warnings but continues
- [ ] Model loads successfully despite suspicious weights
- [ ] Warnings are logged to stderr
- [ ] Inference can proceed

**Test Command**:
```bash
cargo run -p bitnet-cli -- inspect --model model.gguf
```

**Success Criteria**:
- Exit code 0
- Warnings are displayed if weights are suspicious
- Inspection completes successfully

## 2. Numerical Stability

### 2.1 Attention Path
- [ ] No NaN in attention scores
- [ ] No Inf in attention scores
- [ ] Scale factor computed correctly (1/sqrt(head_dim))
- [ ] Max-subtraction applied before softmax
- [ ] Attention weights sum to 1.0

**Test Command**:
```bash
BITNET_DEBUG_ATTN_SCALE=1 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "Test" --max-new-tokens 5 2>&1 | grep -i "nan\|inf"
```

**Success Criteria**:
- No matches for "nan" or "inf"
- Debug logs show scale factor ≈ 0.0883883 (for head_dim=128)
- Scores range is reasonable (e.g., [-50, 0] after mask)

### 2.2 RMSNorm Path
- [ ] No NaN in RMSNorm inputs
- [ ] No Inf in RMSNorm outputs
- [ ] RMS computed correctly: sqrt(mean(x²))
- [ ] Output L2 norm is reasonable
- [ ] Epsilon prevents division by zero

**Test Command**:
```bash
BITNET_DEBUG_RMSNORM=1 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "Test" --max-new-tokens 5 2>&1 | grep -E "RMSNorm|nan|inf"
```

**Success Criteria**:
- No matches for "nan" or "inf"
- mean(x²) is positive and finite
- RMS is positive and finite
- Output L2 norm is in reasonable range (0.1-10.0)

### 2.3 Logits Path
- [ ] No NaN in logits
- [ ] No Inf in logits
- [ ] Logits distribution is reasonable
- [ ] Tied embeddings work correctly
- [ ] Top logits are distinct

**Test Command**:
```bash
BITNET_DEBUG_LOGITS=1 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "Test" --max-new-tokens 5 2>&1 | grep -E "logits|nan|inf"
```

**Success Criteria**:
- No matches for "nan" or "inf"
- Logits span a reasonable range (e.g., [-100, 100])
- Top-k logits are distinct
- Argmax token is valid

## 3. Deterministic Inference

### 3.1 Single-Threaded Determinism
- [ ] Two runs with same seed produce identical tokens
- [ ] Token IDs match exactly
- [ ] Generated text matches exactly
- [ ] Logits match exactly (if dumped)

**Test Command**:
```bash
# Run 1
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Why is the sky blue?" --max-new-tokens 32 \
  --temperature 0.0 --json-out run1.json

# Run 2
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Why is the sky blue?" --max-new-tokens 32 \
  --temperature 0.0 --json-out run2.json

# Compare
diff <(jq -c '.tokens.ids' run1.json) <(jq -c '.tokens.ids' run2.json)
```

**Success Criteria**:
- Exit code 0 (no differences)
- Token arrays are identical
- Text output is identical

### 3.2 Greedy Decoding
- [ ] Temperature 0.0 forces greedy sampling
- [ ] Argmax token is always chosen
- [ ] No randomness in token selection
- [ ] Multiple runs produce same output

**Test Command**:
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Count to five:" --max-new-tokens 20 \
  --temperature 0.0 --greedy --json-out greedy.json
```

**Success Criteria**:
- Greedy flag enforces argmax selection
- No sampling randomness
- Deterministic across runs

### 3.3 Seed Reproducibility
- [ ] Same seed produces same output
- [ ] Different seeds produce different output
- [ ] Seed is respected in all sampling modes

**Test Command**:
```bash
# Seed A
BITNET_SEED=42 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "Test" --max-new-tokens 10 \
  --temperature 0.8 --seed 42 --json-out seed42.json

# Seed B
BITNET_SEED=99 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "Test" --max-new-tokens 10 \
  --temperature 0.8 --seed 99 --json-out seed99.json

# Compare (should differ)
diff <(jq -c '.tokens.ids' seed42.json) <(jq -c '.tokens.ids' seed99.json)
```

**Success Criteria**:
- Seed 42 runs are identical
- Seed 99 runs are identical
- Seed 42 ≠ Seed 99 (with high probability for temp > 0)

## 4. Output Quality

### 4.1 Factual Question
- [ ] Prompt: "Why is the sky blue?"
- [ ] Output mentions relevant concepts
- [ ] Keywords present: rayleigh, scatter, light, atmosphere, wavelength
- [ ] Text is coherent and grammatical
- [ ] No repetitive tokens

**Test Command**:
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Why is the sky blue?" --max-new-tokens 32 \
  --temperature 0.0 --json-out sky.json
```

**Success Criteria**:
- At least one keyword present (case-insensitive)
- Text is readable and on-topic
- No gibberish or token repetition

### 4.2 Counting Test
- [ ] Prompt: "Count to five:"
- [ ] Output contains numbers 1, 2, 3, 4, 5
- [ ] Numbers appear in order
- [ ] Format is reasonable (e.g., "1, 2, 3, 4, 5" or "1\n2\n3\n4\n5")

**Test Command**:
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Count to five:" --max-new-tokens 32 \
  --temperature 0.0 --json-out count.json
```

**Success Criteria**:
- Numbers 1-5 present in output
- Sequential order maintained
- No missing or extra numbers

### 4.3 Translation Test
- [ ] Prompt: "Translate 'bonjour' to English:"
- [ ] Output contains "hello" or "Hello"
- [ ] Translation is correct
- [ ] No extraneous text

**Test Command**:
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Translate 'bonjour' to English:" --max-new-tokens 32 \
  --temperature 0.0 --json-out translate.json
```

**Success Criteria**:
- Output contains "hello" (case-insensitive)
- Translation is accurate
- Response is concise

### 4.4 No Gibberish
- [ ] No repetitive token loops (e.g., "the the the the...")
- [ ] No nonsense words
- [ ] Grammar is generally correct
- [ ] Punctuation is appropriate

**Manual Review**:
Review all generated outputs for:
- Token repetition patterns
- Grammatical errors
- Semantic coherence
- Appropriate vocabulary

## 5. Receipt Validation

### 5.1 JSON Structure
- [ ] Receipt is valid JSON
- [ ] All required fields present
- [ ] Field types are correct
- [ ] No extra unexpected fields

**Test Command**:
```bash
jq empty inference_output.json
```

**Success Criteria**:
- Exit code 0 (valid JSON)
- No parse errors

### 5.2 Compute Path
- [ ] `compute_path: "real"` (not "mock")
- [ ] No mock inference fallback
- [ ] Real tensors used
- [ ] Real quantization applied

**Test Command**:
```bash
jq -r '.compute_path' inference_output.json
```

**Success Criteria**:
- Output is exactly "real"

### 5.3 Backend
- [ ] `backend: "cpu"` for CPU inference
- [ ] Backend matches compilation flags
- [ ] No GPU backend for CPU-only builds

**Test Command**:
```bash
jq -r '.backend' inference_output.json
```

**Success Criteria**:
- Output is exactly "cpu"

### 5.4 Kernels
- [ ] `kernels` array is non-empty
- [ ] Kernel names are valid
- [ ] CPU kernels only (no GPU kernel IDs)
- [ ] Expected kernels present (e.g., i2s_cpu, avx2_matmul)

**Test Command**:
```bash
jq '.kernels | length' inference_output.json
```

**Success Criteria**:
- Count > 0
- All kernel names match CPU naming convention

### 5.5 Corrections (if present)
- [ ] Each correction has `layer`, `rms_before`, `rms_after`, `factor`
- [ ] `rms_before` is the original RMS
- [ ] `rms_after` is in range [0.5, 2.0]
- [ ] `factor` is in range [0.1, 10.0] (clamped)
- [ ] Corrections are applied only when policy is set

**Test Command**:
```bash
jq '.corrections[]' inference_output.json
```

**Success Criteria**:
- All corrections have required fields
- RMS and factor values are within bounds
- Corrections improve stability

## 6. Performance Metrics

### 6.1 Latency
- [ ] `latency.total_ms` is reasonable (>0, <1 hour)
- [ ] `latency.cmd_to_first_ms` is reasonable
- [ ] First token latency < total latency

**Test Command**:
```bash
jq '.latency' inference_output.json
```

**Success Criteria**:
- All latency values are positive
- First token < total time
- Values are in expected range (seconds to minutes)

### 6.2 Throughput
- [ ] `throughput.tokens_per_second` is reasonable for CPU
- [ ] CPU baseline: 10-20 tok/s for I2_S
- [ ] No suspiciously high values (>150 tok/s suggests mock)

**Test Command**:
```bash
jq '.throughput.tokens_per_second' inference_output.json
```

**Success Criteria**:
- Value is in range [5, 50] for CPU
- Matches expected CPU performance

### 6.3 Token Counts
- [ ] `tokens.prompt` matches input prompt length
- [ ] `tokens.generated` matches requested max_new_tokens (or EOS)
- [ ] `tokens.total` = prompt + generated

**Test Command**:
```bash
jq '.tokens' inference_output.json
```

**Success Criteria**:
- Counts are consistent
- Total = prompt + generated

## Automated Test Script

The full acceptance test suite is available as a script:

```bash
./scripts/accept_mvp_cpu.sh
```

This script automatically runs all tests and produces a summary report.

## Exit Criteria

All of the following must be true for MVP acceptance:

1. ✅ All pre-flight checks pass
2. ✅ Model loading and validation succeeds
3. ✅ Zero NaN/Inf in inference logs
4. ✅ Deterministic inference produces identical outputs
5. ✅ Output quality meets keyword baselines
6. ✅ Receipt validation passes
7. ✅ Performance metrics are in expected ranges
8. ✅ Automated test script exits with code 0

## Failure Modes and Resolutions

### NaN/Inf Detected
**Resolution**:
1. Check LayerNorm statistics
2. Enable debug logging
3. Review attention and normalization code
4. Consider correction policy if weights are bad

### Non-Deterministic Outputs
**Resolution**:
1. Ensure `BITNET_DETERMINISTIC=1`
2. Set `RAYON_NUM_THREADS=1`
3. Use `--temperature 0.0`
4. Check for GPU interference

### Poor Output Quality
**Resolution**:
1. Validate model with `compat-check --strict`
2. Check for quantized LayerNorm weights
3. Enable comprehensive diagnostics
4. Review model provenance

### Receipt Validation Failure
**Resolution**:
1. Check JSON format
2. Verify compute_path and backend
3. Inspect kernel names
4. Review correction parameters

## Related Documentation

- [INFERENCE_MVP.md](INFERENCE_MVP.md): Full MVP documentation
- [INFERENCE_FIXES.md](INFERENCE_FIXES.md): Surgical fixes reference
- [CLAUDE.md](CLAUDE.md): Essential project guidance
- [docs/development/test-suite.md](docs/development/test-suite.md): Test suite guide

## Sign-Off

Date: __________

Tester: __________

Results:
- [ ] All tests passed
- [ ] Known issues documented
- [ ] MVP accepted for release

Notes:
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
