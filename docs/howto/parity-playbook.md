# Parity Playbook: Diagnosing Cross-Validation Issues

**Purpose**: Step-by-step guide for diagnosing inference parity issues between Rust and C++ reference implementations.

**When to Use**: When you encounter divergence in cross-validation tests or suspect numerical differences between implementations.

---

## Overview: The Parity Ladder

The parity ladder is a systematic approach to isolating divergence:

```
Step 0: Preflight       → C++ backend available?
Step 1: Token Parity    → Tokenization matches exactly?
Step 2: Shapes & Masks  → Attention masks & KV shapes match?
Step 3: First Logits    → Position 0 logits within tolerance?
Step 4: Per-Position    → All prompt positions within tolerance?
Step 5: Greedy Decode   → Generated tokens match exactly?
```

Each step builds on the previous. **Stop at the first failure** to isolate the root cause.

---

## Threshold Reference

| Ladder Stage | MSE Threshold | KL Divergence | Top-K Overlap | Max Abs Diff | Notes |
|--------------|---------------|---------------|---------------|--------------|-------|
| **first-logit** | ≤ 1e-6 | N/A | N/A | ≤ 5e-4 | Strictest - single forward pass |
| **positions** | ≤ 2e-6 | ≤ 5e-4 | ≥ 95% (top-5) | ≤ 1e-3 | Cumulative error allowed |
| **decode** | N/A | N/A | 100% (exact) | N/A | Greedy token must match |

**Tolerances Rationale**:
- **first-logit**: No cumulative error - detects issues in embedding, rope, or single layer
- **positions**: Allows minor floating-point drift across positions
- **decode**: Greedy argmax must produce identical tokens (binary pass/fail)

---

## Step 0: Preflight Check

**Goal**: Verify C++ backend is available and loadable.

### Command

```bash
cargo run -p xtask --features crossval-all -- preflight --verbose
```

### Expected Output (Success)

```
Backend Library Status:

  ✓ bitnet.cpp: AVAILABLE
    Libraries: libbitnet*

  ✓ llama.cpp: AVAILABLE
    Libraries: libllama*, libggml*

Both backends available. Dual-backend cross-validation supported.
```

### Troubleshooting

**Problem**: `bitnet.cpp: NOT AVAILABLE`

```
  ✗ bitnet.cpp: NOT AVAILABLE
    Error: Libraries not found in /path/to/bitnet.cpp
```

**Solutions**:

1. **Auto-bootstrap** (recommended):
   ```bash
   # Bash/Zsh
   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

   # Fish
   cargo run -p xtask -- setup-cpp-auto --emit=fish | source

   # PowerShell
   cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
   ```

2. **Manual setup**:
   ```bash
   cargo run -p xtask -- fetch-cpp
   export BITNET_CPP_DIR=/path/to/bitnet.cpp
   export LD_LIBRARY_PATH=$BITNET_CPP_DIR:$LD_LIBRARY_PATH  # Linux
   export DYLD_LIBRARY_PATH=$BITNET_CPP_DIR:$DYLD_LIBRARY_PATH  # macOS
   ```

3. **Verify environment**:
   ```bash
   echo $BITNET_CPP_DIR
   ls $BITNET_CPP_DIR/libbitnet*
   ```

**Problem**: `llama.cpp: NOT AVAILABLE`

**Solutions**:
1. Install llama.cpp via system package manager
2. Build from source: https://github.com/ggerganov/llama.cpp
3. Ensure `libllama.so` and `libggml.so` are in library path

---

## Step 1: Token Parity

**Goal**: Verify tokenization produces identical token IDs.

### Why This Matters

Token mismatch causes **total divergence** - the C++ reference processes different inputs. All subsequent comparisons are invalid if tokens don't match.

### Command

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose
```

### Expected Output (Success)

```
[Rust token IDs]: [128000, 3923, 374, 220, 17, 10, 17, 30, ...]
[C++ token IDs]:  [128000, 3923, 374, 220, 17, 10, 17, 30, ...]

Token parity: OK (8 tokens match exactly)
```

### Troubleshooting

**Problem**: Token ID mismatch

```
[Rust token IDs]: [128000, 3923, 374, 220, 17, 10, 17, 30]
[C++ token IDs]:  [1, 3923, 374, 220, 17, 10, 17, 30]
                   ^^^^^^ BOS token differs!
```

**Root Causes**:

1. **BOS Token Mismatch**
   - Symptom: First token differs (e.g., `128000` vs `1`)
   - Cause: Tokenizer adds BOS automatically, C++ reference doesn't (or vice versa)
   - Solution: Check `tokenizer.json` for `"add_bos_token": true`
   - Workaround: Adjust template or use `--prompt-template raw`

2. **Template Formatting Difference**
   - Symptom: Extra tokens at start/end (e.g., `<|begin_of_text|>`, `<|start_header_id|>`)
   - Cause: Prompt template applied in Rust but not C++
   - Solution: Use `--prompt-template raw` for both implementations

   ```bash
   # Try raw template to bypass formatting
   cargo run -p xtask --features crossval-all -- crossval-per-token \
     --model models/model.gguf \
     --tokenizer models/tokenizer.json \
     --prompt-template raw \
     --prompt "2+2=" \
     --max-tokens 1 \
     --dump-ids --dump-cpp-ids
   ```

3. **Tokenizer Version Mismatch**
   - Symptom: Random token differences throughout sequence
   - Cause: Different tokenizer vocabulary or merges file
   - Solution: Use same `tokenizer.json` for both Rust and C++
   - Verify: `ls -lh models/tokenizer.json` (same file for both)

4. **Special Token Handling**
   - Symptom: EOS/EOT tokens appear unexpectedly
   - Cause: Template adds `<|eot_id|>` or similar tokens
   - Solution: Check `--stop-id` and template defaults

**Diagnostic Steps**:

```bash
# 1. Check tokenizer metadata
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check \
  models/model.gguf --show-kv | grep -i bos

# 2. Test with raw template (no formatting)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template raw \
  --prompt "Test" \
  --max-tokens 1 \
  --dump-ids --dump-cpp-ids

# 3. Test with different template
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template instruct \
  --prompt "Test" \
  --max-tokens 1 \
  --dump-ids --dump-cpp-ids
```

---

## Step 2: Shapes & Masks

**Goal**: Verify attention mask and KV cache shapes match between implementations.

### Why This Matters

Shape mismatches cause silent numerical divergence. Even if tokens match, incorrect shapes lead to:
- Incorrect attention patterns (using wrong context)
- Buffer overflows/underflows
- Silent NaN propagation

### Command

```bash
# Current implementation: use crossval-per-token with verbose mode
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --verbose
```

**Note**: `--ladder masks` is planned for future implementation (see Step 5 for decode ladder).

### Expected Output (Success)

```
[Rust] Prompt tokens: 8, max_tokens: 4, context_size: 512
[C++]  Prompt tokens: 8, max_tokens: 4, context_size: 512

Shapes match: OK
```

### Troubleshooting

**Problem**: Context size mismatch

```
[Rust] context_size: 512
[C++]  context_size: 2048
```

**Root Cause**: Model configured with different `n_ctx` parameter

**Solution**:
1. Check GGUF metadata: `cargo run -p bitnet-cli -- compat-check model.gguf --show-kv | grep ctx`
2. Ensure C++ reference uses same context size
3. Verify `llama_context_params` in C++ FFI bridge

**Problem**: Prompt token count mismatch

```
[Rust] Prompt tokens: 8
[C++]  Prompt tokens: 10
```

**Root Cause**: Template formatting differs (see Step 1 - Token Parity)

**Solution**: Ensure token parity first before checking shapes

---

## Step 3: First Logits (Position 0)

**Goal**: Verify the very first logits (position 0) match within tight tolerance.

### Why This Matters

The first forward pass has **no cumulative error** - it's a pure test of:
- Embedding layer
- RoPE (rotary position encoding)
- LayerNorm
- Attention computation
- FFN (feed-forward network)

If position 0 fails, the issue is in a single layer's computation, not cumulative drift.

### Command

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --cos-tol 0.9999 \
  --format json
```

**Note**: `--ladder first-logit` is planned for future implementation with explicit MSE/max_abs metrics.

### Expected Output (Success)

```json
{
  "status": "ok",
  "backend": "bitnet",
  "divergence_token": -1,
  "metrics": {
    "min_cosine_similarity": 0.99995,
    "max_l2_distance": 0.00042,
    "mean_abs_difference": 0.00018,
    "token_count": 1
  }
}
```

**Thresholds**:
- Cosine similarity: ≥ 0.9999 (current), target MSE ≤ 1e-6
- L2 distance: ≤ 0.001 (current), target max_abs ≤ 5e-4

### Troubleshooting

**Problem**: First logits diverge (cosine similarity < 0.999)

```json
{
  "status": "diverged",
  "divergence_token": 0,
  "metrics": {
    "min_cosine_similarity": 0.8234,
    "max_l2_distance": 2.4512
  }
}
```

**Root Causes**:

1. **Embedding Layer Issue**
   - Symptom: Large L2 distance (> 1.0), low cosine similarity (< 0.95)
   - Cause: Embedding weights not loaded correctly
   - Diagnostic: Check embedding tensor in GGUF
     ```bash
     cargo run -p bitnet-cli -- compat-check model.gguf --show-kv | grep embed
     ```
   - Solution: Verify `token_embd.weight` tensor loads correctly

2. **LayerNorm Issue**
   - Symptom: All logits shifted by constant factor
   - Cause: LayerNorm gamma/beta quantized or corrupted
   - Diagnostic:
     ```bash
     cargo run -p bitnet-cli --features cpu,full-cli -- inspect \
       --ln-stats --gate auto models/model.gguf
     ```
   - Solution: Re-export GGUF with FP16 LayerNorm (see `docs/howto/export-clean-gguf.md`)

3. **RoPE (Rotary Position Encoding) Issue**
   - Symptom: Divergence increases with position
   - Cause: RoPE frequency calculation differs
   - Diagnostic: Compare position 0 vs position 7 divergence
     ```bash
     # Position 0 (no cumulative error)
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "Test" --max-tokens 1 --format json

     # Position 7 (cumulative error)
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "What is the capital of France?" --max-tokens 1 --format json
     ```
   - Solution: Check `rope_freq_base` and `rope_freq_scale` in GGUF metadata

4. **Quantization Dequantization Issue**
   - Symptom: Random noise pattern, not systematic shift
   - Cause: Dequantization produces different FP values
   - Diagnostic: Test with F16/F32 model (no quantization)
   - Solution: Check quantization kernel (I2_S, TL1, etc.) implementation

**Diagnostic Steps**:

```bash
# 1. Validate LayerNorm weights
cargo run -p bitnet-cli --features cpu,full-cli -- inspect \
  --ln-stats --gate auto models/model.gguf

# 2. Check GGUF metadata for RoPE parameters
cargo run -p bitnet-cli -- compat-check models/model.gguf --show-kv | grep rope

# 3. Test with shorter prompt (single token)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 1 \
  --format json

# 4. Compare with known-good model
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/reference-model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 1 \
  --format json
```

---

## Step 4: Per-Position (First N Positions)

**Goal**: Verify all prompt positions produce logits within tolerance.

### Why This Matters

This tests cumulative error across the full prompt:
- Attention mask application (causal masking)
- KV cache updates per position
- Floating-point drift across layers

If position 0 passes but position 7 fails, the issue is in **sequential processing**, not single-layer computation.

### Command

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 1 \
  --cos-tol 0.999 \
  --format text
```

**Note**: `--ladder positions --positions 8` is planned for future implementation with explicit per-position metrics.

### Expected Output (Success)

```
Position 0: OK (cos_sim: 0.9999, l2_dist: 0.0042)
Position 1: OK (cos_sim: 0.9998, l2_dist: 0.0051)
Position 2: OK (cos_sim: 0.9997, l2_dist: 0.0058)
Position 3: OK (cos_sim: 0.9996, l2_dist: 0.0064)
Position 4: OK (cos_sim: 0.9995, l2_dist: 0.0072)
Position 5: OK (cos_sim: 0.9994, l2_dist: 0.0081)
Position 6: OK (cos_sim: 0.9993, l2_dist: 0.0089)
Position 7: OK (cos_sim: 0.9992, l2_dist: 0.0095)

Summary: All positions parity OK
Minimum cosine similarity: 0.99920
Maximum L2 distance: 0.00950
```

**Thresholds** (planned):
- MSE: ≤ 2e-6 (allows cumulative drift)
- KL divergence: ≤ 5e-4
- Top-5 overlap: ≥ 95%
- Max abs diff: ≤ 1e-3

### Troubleshooting

**Problem**: Divergence increases linearly with position

```
Position 0: OK (cos_sim: 0.9999)
Position 1: OK (cos_sim: 0.9995)
Position 2: WARN (cos_sim: 0.9980)  ← Drift accelerating
Position 3: FAIL (cos_sim: 0.9850)
```

**Root Causes**:

1. **Attention Mask Issue**
   - Symptom: Divergence increases with context length
   - Cause: Causal mask not applied correctly (model "sees" future tokens)
   - Diagnostic: Test with single-token prompt vs multi-token prompt
     ```bash
     # Single token (no mask needed)
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "Test" --max-tokens 1 --format json

     # Multi-token (requires causal mask)
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "What is the capital" --max-tokens 1 --format json
     ```
   - Solution: Verify mask generation in `apply_causal_mask()` (see Issue #254)

2. **KV Cache Stride/Indexing Issue**
   - Symptom: Sudden jump in divergence at specific position (e.g., position 4)
   - Cause: KV cache indexing off-by-one or incorrect stride
   - Diagnostic: Check KV cache shapes in verbose mode
   - Solution: Verify `kv_cache.update()` logic

3. **RoPE Frequency Drift**
   - Symptom: Divergence proportional to position index
   - Cause: RoPE `freq_base` or `freq_scale` differs between implementations
   - Diagnostic: Compare position 0 vs position 15+ divergence rate
   - Solution: Check `rope_freq_base` in GGUF metadata

4. **Numerical Precision Loss**
   - Symptom: Gradual drift, no sudden jumps
   - Cause: FP16 vs FP32 accumulation, SIMD rounding differences
   - Diagnostic: Test with deterministic flags
     ```bash
     export BITNET_DETERMINISTIC=1
     export RAYON_NUM_THREADS=1
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "Test prompt" --max-tokens 1 --format json
     ```
   - Solution: Acceptable if within tolerance (MSE ≤ 2e-6)

**Diagnostic Steps**:

```bash
# 1. Test with increasing prompt lengths
for prompt in "Test" "What is" "What is the capital"; do
  echo "=== Prompt: $prompt ==="
  cargo run -p xtask --features crossval-all -- crossval-per-token \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --prompt "$prompt" \
    --max-tokens 1 \
    --format text | grep "Summary"
done

# 2. Capture position-by-position metrics (verbose)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France and Germany?" \
  --max-tokens 1 \
  --format text \
  --verbose

# 3. Test deterministic inference
export BITNET_DETERMINISTIC=1
export RAYON_NUM_THREADS=1
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 1 \
  --format json
```

---

## Step 5: Greedy Decode (First M Tokens)

**Goal**: Verify greedy decoding produces identical next tokens for M generation steps.

### Why This Matters

Greedy decode is the **ultimate integration test**:
- Tests KV cache updates (not just reads)
- Tests autoregressive loop (each step depends on previous)
- Tests sampling (argmax must produce same token ID)

If position N logits match but generated token differs, the issue is in **sampling** or **KV cache state management**.

### Command

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 8 \
  --cos-tol 0.999 \
  --format text \
  --dump-ids --dump-cpp-ids
```

**Note**: `--ladder decode --positions 8` is planned for future implementation with explicit per-step token comparison.

### Expected Output (Success)

```
[Rust generated IDs]: [220, 19, 374, 220, 19, 13]
[C++ generated IDs]:  [220, 19, 374, 220, 19, 13]

Position 0: OK (cos_sim: 0.9999) → token 220 (both)
Position 1: OK (cos_sim: 0.9998) → token 19 (both)
Position 2: OK (cos_sim: 0.9997) → token 374 (both)
Position 3: OK (cos_sim: 0.9996) → token 220 (both)
Position 4: OK (cos_sim: 0.9995) → token 19 (both)
Position 5: OK (cos_sim: 0.9994) → token 13 (both)

Summary: All generated tokens match (6/6 exact match)
```

**Thresholds**:
- Token match rate: 100% (exact match required for greedy decoding)
- If tokens match, logits comparison is informational only

### Troubleshooting

**Problem**: Position 0 matches, but position 1 diverges

```
Position 0: OK (cos_sim: 0.9999) → token 220 (both)
Position 1: FAIL (cos_sim: 0.7234) → token 19 (Rust) vs token 42 (C++)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      First generation step fails!
```

**Root Cause**: KV cache update issue

**Why**:
- Position 0 uses only prompt tokens (read-only KV cache)
- Position 1 requires updating KV cache with generated token
- Mismatch indicates KV cache state management bug

**Diagnostic Steps**:

```bash
# 1. Verify position 0 (no KV update needed)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --prompt "Test" --max-tokens 1 --dump-ids --dump-cpp-ids

# 2. Test position 1 (first KV update)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --prompt "Test" --max-tokens 2 --dump-ids --dump-cpp-ids --verbose

# 3. Check KV cache shapes
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --prompt "Test" --max-tokens 4 --verbose | grep -i "kv_cache"
```

**Solution**:
1. Verify `kv_cache.update()` logic in `bitnet-inference/src/engine.rs`
2. Check sequence position tracking (off-by-one errors)
3. Ensure cache slicing uses correct offsets

**Problem**: Random token mismatch at unpredictable positions

```
Position 0: OK → token 220 (both)
Position 1: OK → token 19 (both)
Position 2: OK → token 374 (both)
Position 3: FAIL → token 42 (Rust) vs token 99 (C++)  ← Random divergence
Position 4: FAIL → token 123 (Rust) vs token 456 (C++)
```

**Root Causes**:

1. **Non-Deterministic Sampling**
   - Symptom: Different tokens even with greedy decoding
   - Cause: Sampling uses random seed or top-p/top-k instead of argmax
   - Solution: Verify greedy sampling (temperature = 0.0, no nucleus/top-k)
     ```bash
     # Ensure greedy decoding
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "Test" --max-tokens 4 --dump-ids --dump-cpp-ids
     # Note: crossval-per-token uses greedy by default (no temperature/top-p flags)
     ```

2. **Floating-Point Argmax Tie**
   - Symptom: Divergence on specific tokens with similar logits
   - Cause: Multiple tokens have identical logits (tie), argmax picks arbitrarily
   - Diagnostic: Check logits for top-5 tokens
   - Solution: Acceptable if logits are truly identical (< 1e-6 difference)

3. **Uninitialized Memory**
   - Symptom: Non-reproducible divergence (changes each run)
   - Cause: KV cache or logits buffer not zeroed
   - Diagnostic: Run twice with same seed
     ```bash
     export BITNET_DETERMINISTIC=1
     export BITNET_SEED=42
     # Run 1
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "Test" --max-tokens 4 --dump-ids > run1.txt
     # Run 2
     cargo run -p xtask --features crossval-all -- crossval-per-token \
       --prompt "Test" --max-tokens 4 --dump-ids > run2.txt
     diff run1.txt run2.txt  # Should be identical
     ```
   - Solution: Verify buffer initialization in KV cache and logits tensors

**Problem**: All tokens shifted by constant offset

```
Position 0: FAIL → token 220 (Rust) vs token 128220 (C++)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   Offset by 128000 (vocab size?)
```

**Root Cause**: Token ID offset issue

**Diagnostic**:
- Check vocab size: `cargo run -p bitnet-cli -- compat-check model.gguf --show-kv | grep vocab`
- Verify BOS token ID: `grep -i bos models/tokenizer.json`

**Solution**:
1. Check if C++ adds/subtracts offset for special tokens
2. Verify Rust tokenizer uses same special token IDs
3. See Step 1 (Token Parity) for BOS token troubleshooting

---

## Common Failure Patterns (Quick Reference)

| Symptom | Likely Cause | Diagnostic Step | Fix |
|---------|--------------|-----------------|-----|
| **Token ID mismatch at position 0** | BOS token or template formatting | Step 1: `--dump-ids --dump-cpp-ids` | Use `--prompt-template raw` |
| **All token IDs shifted by constant** | Vocab offset or special token handling | Check vocab size in GGUF metadata | Verify tokenizer special tokens |
| **Logits all scaled by constant** | LayerNorm gamma issue | `--ln-stats --gate auto` | Re-export GGUF with FP16 LN |
| **Divergence increases with position** | RoPE or attention mask issue | Compare pos 0 vs pos 7 divergence | Check `rope_freq_base` in GGUF |
| **Sudden divergence at position N** | KV cache indexing off-by-one | Test with prompts of length N-1, N, N+1 | Fix KV cache stride/offset |
| **Position 0 OK, position 1 fails** | KV cache update bug | Step 5: `--max-tokens 2` | Verify `kv_cache.update()` logic |
| **Random token divergence (greedy)** | Non-deterministic sampling or uninitialized memory | Run twice with same seed, compare | Enable deterministic flags |
| **Top-5 overlap < 95%** | Quantization dequant differs | Test with F16 model | Check I2_S/TL1 kernel implementation |

---

## Workflow Summary

**Quick Diagnostic Workflow**:

```bash
# 1. Preflight
cargo run -p xtask --features crossval-all -- preflight --verbose

# 2. Token parity (must pass before continuing)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --dump-ids --dump-cpp-ids

# 3. First logits (position 0 only)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 1 \
  --cos-tol 0.9999 \
  --format json

# 4. Per-position (full prompt)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 1 \
  --cos-tol 0.999 \
  --format text

# 5. Greedy decode (8 generation steps)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 8 \
  --dump-ids --dump-cpp-ids
```

**Stop at first failure** and use troubleshooting section for that step.

---

## Advanced Diagnostics

### Trace Comparison (Deep Dive)

If cross-validation fails and basic diagnostics don't reveal the issue, capture full traces:

```bash
# 1. Run cross-validation sweep (captures 90+ traces per scenario)
./scripts/run_crossval_sweep.sh \
  models/model.gguf \
  models/tokenizer.json \
  /tmp/crossval-traces

# 2. Compare traces layer-by-layer
cargo run -p xtask -- trace-diff \
  /tmp/crossval-traces/scenario_1/rs_traces \
  /tmp/crossval-traces/scenario_1/cpp_traces

# Output shows first diverging layer and tensor statistics
```

### Receipt Verification (Kernel Validation)

Verify that GPU kernels are actually used (not CPU fallback):

```bash
# 1. Run benchmark with receipt generation
cargo run -p xtask -- benchmark \
  --model models/model.gguf \
  --tokens 128

# 2. Verify GPU kernel IDs in receipt
cargo run -p xtask -- verify-receipt --require-gpu-kernels

# Expected kernel IDs for GPU:
# - gemm_cuda_*
# - i2s_gpu_*
# - rope_cuda_*
```

### Deterministic Inference Validation

Ensure bit-exact reproducibility:

```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Run twice, compare outputs
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 4 \
  --dump-ids > run1.txt

cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 4 \
  --dump-ids > run2.txt

diff run1.txt run2.txt  # Should be identical (no output)
```

---

## Related Documentation

- **C++ Setup**: `docs/howto/cpp-setup.md` - Setting up bitnet.cpp and llama.cpp references
- **Cross-Validation**: `docs/development/validation-framework.md` - AC9 cross-validation framework
- **Trace Analysis**: `docs/development/trace-comparison.md` - Layer-by-layer divergence analysis
- **Model Validation**: `docs/howto/validate-models.md` - LayerNorm and projection validation

---

## Future Enhancements

Planned improvements to parity tooling:

1. **Explicit Ladder Subcommands**:
   - `--ladder masks` - Shape and mask comparison
   - `--ladder first-logit` - Position 0 with MSE/max_abs metrics
   - `--ladder positions --positions N` - Per-position metrics with KL divergence
   - `--ladder decode --positions M` - Greedy decode validation

2. **Enhanced Metrics**:
   - Mean Squared Error (MSE) alongside cosine similarity
   - KL divergence for probability distribution comparison
   - Top-K overlap percentage (top-1, top-5, top-10)
   - Max absolute difference (per-element worst-case)

3. **Automated Bisection**:
   - Automatic prompt length bisection to find divergence boundary
   - Binary search for first diverging position
   - Automatic trace capture on divergence

4. **Visual Diff Output**:
   - HTML report with side-by-side logits comparison
   - Heatmap of per-position divergence
   - Token-by-token probability distribution plots

See GitHub issues for tracking: #254, #260, #469
