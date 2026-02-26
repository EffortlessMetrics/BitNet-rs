# Troubleshooting Inference Intelligibility

**Target audience:** Users experiencing garbled or non-sensical output from BitNet-rs inference

**Related docs:**
- [CLAUDE.md](../../CLAUDE.md) - Project overview and known issues
- [Inference Usage](../CLAUDE.md#inference-usage) - Basic inference examples
- [Prompt Templates](../reference/prompt-templates.md) - Template reference
- [Investigation Report](../tdd/receipts/inference_quality_investigation.md) - Technical analysis

---

## Quick Diagnosis

If you're seeing garbled output like `"jjjj kkkk llll mmmm..."` or nonsensical text instead of coherent responses, work through this checklist:

1. ✅ **Is your model instruction-tuned?** Base models produce poor Q&A output
2. ✅ **Are you using the right template?** Use `--prompt-template instruct` for Q&A
3. ✅ **Are you using a release build?** Debug builds are 10-100× slower
4. ✅ **Is your model actually a BitNet model?** Check with `bitnet compat-check model.gguf`
5. ✅ **Have you tried different sampling parameters?** Adjust temperature, top-p, repetition penalty

---

## Common Causes (In Order of Likelihood)

### 1. Model Quality Issues ⚠️ **MOST COMMON**

**Symptom:** Even with correct configuration, output is garbled or off-topic

**Root cause:** The model weights themselves have quality issues, not the inference engine

**Known affected models:**
- `microsoft-bitnet-b1.58-2B-4T-gguf` (documented in CLAUDE.md)

**How to verify:**
1. Test with a simple, controlled prompt:
   ```bash
   cargo run -p bitnet-cli --release --features cpu,full-cli -- run \
     --model model.gguf \
     --tokenizer tokenizer.json \
     --prompt "2+2=" \
     --max-tokens 1 \
     --temperature 0.0 --greedy
   ```

2. If output is not "4" or close to it, suspect model quality issue

**Fix:**
- ✅ **Try alternative BitNet models** (different checkpoints, different providers)
- ✅ **Lower expectations for base models** (they complete, not answer)
- ✅ **Use instruction-tuned models** (look for "instruct", "chat", or "Q&A" in model name)
- ✅ **Test with synthetic inputs** (simple patterns, not open-ended questions)

**NOT a fix:**
- ❌ Changing sampler parameters (won't fix bad weights)
- ❌ Changing templates (helps, but can't fix fundamental model issues)
- ❌ Updating BitNet-rs code (engine is mathematically correct)

---

### 2. Template Mismatch 🔧 **EASY FIX**

**Symptom:** Output is verbose, doesn't stop, or doesn't follow Q&A format

**Root cause:** Model expects a specific prompt format (instruct/chat) but you're using `raw`

**How to verify:**
```bash
# Try with instruct template
cargo run -p bitnet-cli --release --features cpu,full-cli -- run \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt-template instruct \
  --prompt "What is the capital of France?" \
  --max-tokens 32
```

**Fix:**

**For Q&A prompts:**
```bash
--prompt-template instruct
```

**For conversational models (LLaMA-3 compatible):**
```bash
--prompt-template llama3-chat \
--system-prompt "You are a helpful assistant"
```

**For completion-style tasks:**
```bash
--prompt-template raw
```

**Auto-detection fallback (v0.9.x):**
- Previous default: `raw` (poor for Q&A)
- **New default: `instruct`** (better for Q&A)
- Override with `--prompt-template` if needed

---

### 3. Sampling Configuration 🎛️ **TUNING REQUIRED**

**Symptom:** Repetitive text, degenerate loops, or overly random output

**Root cause:** Sampling parameters not suited to the model's characteristics

**Default sampler:**
```rust
temperature: 0.7
top_k: 50
top_p: 0.9
repetition_penalty: 1.0  // Disabled by default
```

**Recommended adjustments:**

**For repetitive output:**
```bash
--repetition-penalty 1.1  # Or 1.15, 1.2 (higher = stronger penalty)
```

**For too random/incoherent output:**
```bash
--temperature 0.5  # Lower temperature (more focused)
--top-p 0.85       # Narrower nucleus (more conservative)
```

**For deterministic testing:**
```bash
--temperature 0.0 --greedy  # Greedy decoding (always picks top token)
```

**For creative generation:**
```bash
--temperature 0.9  # Higher temperature (more variety)
--top-p 0.95       # Wider nucleus
```

**Example:**
```bash
cargo run -p bitnet-cli --release --features cpu,full-cli -- run \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt-template instruct \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --repetition-penalty 1.1  # Add this to reduce repetition
```

---

### 4. Performance Issues 🐌 **BUILD OPTIMIZATION**

**Symptom:** Inference takes >10 seconds per token, or process hangs

**Root cause:** Using unoptimized debug builds or wrong quantization format

**How to verify:**
```bash
# Check build profile
cargo run -p bitnet-cli -- --version
# If it doesn't say "release", you're using debug

# Check quantization format
cargo run -p bitnet-cli --release --features cpu -- inspect model.gguf
```

**Fix:**

**Use release builds:**
```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# Then run with:
target/release/bitnet run --model model.gguf ...
```

**Avoid QK256 for long generations:**
```bash
# QK256 is slow (~0.1 tok/s for 2B models)
# Stick to short --max-tokens (4-16) for testing
--max-tokens 16
```

**Use I2_S/TL1/TL2 models (faster):**
```bash
# Look for models with "i2_s", "tl1", or "tl2" in the name
# These are typically 10-100× faster than QK256
```

---

### 5. Tokenizer Issues 🔤 **RARE**

**Symptom:** Tokens are wrong, special tokens not handled, BOS/EOS missing

**Root cause:** Tokenizer mismatch between model training and BitNet-rs tokenizer

**How to verify:**
```bash
# Test round-trip encoding
cargo test -p bitnet-tokenizers --test tokenizer_parity --release --features cpu -- --ignored
```

**Fix:**

**Ensure tokenizer file matches model:**
```bash
# Use the tokenizer.json from the same model directory
--tokenizer models/your-model/tokenizer.json
```

**Check special tokens:**
```bash
# Inspect model metadata
cargo run -p bitnet-cli --features cpu -- compat-check model.gguf --show-kv | grep token
```

**Note:** As of v0.9.x, `token_to_id()` is **already implemented** in both HfTokenizer and RustGgufTokenizer. This is NOT a bug.

---

## Step-by-Step Troubleshooting Guide

### Step 1: Verify Model Quality (Baseline Test)

Run the simplest possible test:

```bash
# Build release
cargo build --release --no-default-features --features cpu,full-cli

# Test with simple math (greedy, no sampling noise)
RUST_LOG=warn target/release/bitnet run \
  --model models/your-model.gguf \
  --tokenizer models/your-model/tokenizer.json \
  --prompt "2+2=" \
  --max-tokens 1 \
  --temperature 0.0 --greedy
```

**Expected:** Token should be "4" or close to it

**If not:** Model quality issue (see Section 1)

---

### Step 2: Test Template Impact

Try all three templates:

```bash
# Raw (completion-style)
RUST_LOG=warn target/release/bitnet run \
  --model model.gguf --tokenizer tokenizer.json \
  --prompt-template raw \
  --prompt "The capital of France is" \
  --max-tokens 8

# Instruct (Q&A-style)
RUST_LOG=warn target/release/bitnet run \
  --model model.gguf --tokenizer tokenizer.json \
  --prompt-template instruct \
  --prompt "What is the capital of France?" \
  --max-tokens 16

# LLaMA-3 chat (conversational)
RUST_LOG=warn target/release/bitnet run \
  --model model.gguf --tokenizer tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "What is the capital of France?" \
  --max-tokens 32
```

**Expected:** `instruct` or `llama3-chat` should produce better Q&A output than `raw`

**If not:** Model may not be instruction-tuned (see Section 1)

---

### Step 3: Tune Sampler

Add repetition penalty and adjust temperature:

```bash
RUST_LOG=warn target/release/bitnet run \
  --model model.gguf --tokenizer tokenizer.json \
  --prompt-template instruct \
  --prompt "What is photosynthesis?" \
  --max-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --repetition-penalty 1.1  # KEY: reduces repetitive text
```

**Expected:** Output should have less repetition, more variety

---

### Step 4: Verify Performance

Check if release build is being used:

```bash
# Should show "release" in the output
target/release/bitnet --version

# If using cargo run, ensure --release flag:
cargo run --release -p bitnet-cli --features cpu,full-cli -- run ...
```

Check quantization format:

```bash
cargo run --release -p bitnet-cli --features cpu -- inspect model.gguf
# Look for quantization type: I2_S (fast), TL1/TL2 (fast), QK256 (slow)
```

---

### Step 5: Check Tokenizer Parity

Run tokenizer round-trip tests:

```bash
cargo test -p bitnet-tokenizers --release --test tokenizer_parity --features cpu -- --ignored
```

**Expected:** All tests pass

**If not:** Tokenizer mismatch (see Section 5)

---

## When to Report a Bug

**Report a bug if:**
- ✅ You've tried all troubleshooting steps above
- ✅ You're using a known-good model (not microsoft-bitnet-b1.58)
- ✅ You're using release builds with proper RUSTFLAGS
- ✅ Greedy decoding (`temperature=0 --greedy`) is non-deterministic (same prompt → different outputs)
- ✅ Tokenizer round-trip tests fail (`decode(encode(text)) != text`)

**Don't report if:**
- ❌ You're using microsoft-bitnet-b1.58 (known model quality issue)
- ❌ You're using debug builds (expected to be slow)
- ❌ You're using `raw` template for Q&A (expected to be poor)
- ❌ You haven't tried adjusting sampling parameters

---

## Quick Reference: Common Fixes

| Symptom | Quick Fix |
|---------|-----------|
| **Garbled output** | Try different model, use `--prompt-template instruct` |
| **Doesn't stop** | Add `--stop-id <eos_id>` or `--stop "</s>"` |
| **Too repetitive** | Add `--repetition-penalty 1.1` (or higher) |
| **Too random** | Lower `--temperature 0.5` |
| **Too boring** | Raise `--temperature 0.9` |
| **Very slow** | Use `--release` builds, `RUSTFLAGS="-C target-cpu=native"` |
| **Process hangs** | Limit `--max-tokens 16` (especially for QK256 models) |

---

## Additional Resources

**Documentation:**
- [CLAUDE.md](../../CLAUDE.md) - Known issues and limitations
- [Inference Usage](../../CLAUDE.md#inference-usage) - Complete inference guide
- [Prompt Templates](../reference/prompt-templates.md) - Template reference
- [Sampling Controls](../../CLAUDE.md#sampling-controls) - Sampler reference

**Technical Investigation:**
- [Inference Quality Investigation](../tdd/receipts/inference_quality_investigation.md) - Detailed technical analysis

**Test Suites:**
- `crates/bitnet-tokenizers/tests/tokenizer_parity.rs` - Tokenizer correctness tests
- `crates/bitnet-inference/tests/greedy_decode_parity.rs` - Determinism tests
- `crates/bitnet-cli/tests/intelligibility_smoke.rs` - Quality smoke tests

**GitHub Issues:**
- Issue #254: Shape mismatch in layer-norm (blocks some real inference tests)
- Issue #260: Mock elimination not complete
- Issue #469: Tokenizer parity and FFI build hygiene (blocks cross-validation tests, NOT inference)

---

**Last updated:** 2025-10-22
**Version:** v0.9.x (post-MVP, template fallback changed to `instruct`)
