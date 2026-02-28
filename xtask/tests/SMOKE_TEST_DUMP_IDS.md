# Smoke Test Guide: --dump-ids and --dump-cpp-ids Flags

## Purpose

Manual verification guide for token ID dumping flags in `crossval-per-token` command.

## Prerequisites

1. Model file available (e.g., `models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`)
2. Tokenizer file available (e.g., `models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json`)
3. C++ FFI setup (for `--dump-cpp-ids`):

   ```bash
   # One-command setup
   eval "$(cargo run -p xtask --features crossval-all -- setup-cpp-auto --emit=sh)"

   ```

## Test Commands

### Test 1: Rust Token Dumping (--dump-ids)

**Command:**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --dump-ids

```

**Expected Output (stderr):**

```

🦀 Rust tokens (N total):
  [token1, token2, token3, ...]

```

**Verification:**

- [ ] Output appears on stderr (not stdout)
- [ ] Shows emoji: 🦀
- [ ] Shows token count: `N total`
- [ ] Shows array format with brackets: `[...]`
- [ ] Token IDs are integers

---

### Test 2: C++ Token Dumping (--dump-cpp-ids)

**Command:**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --dump-cpp-ids

```

**Expected Output (stderr):**

```

🔧 C++ tokens (N total, backend: bitnet|llama):
  [token1, token2, token3, ...]

```

**Verification:**

- [ ] Output appears on stderr (not stdout)
- [ ] Shows emoji: 🔧
- [ ] Shows token count: `N total`
- [ ] Shows backend name: `backend: bitnet` or `backend: llama`
- [ ] Shows array format with brackets: `[...]`
- [ ] Token IDs are integers

---

### Test 3: Both Flags Combined

**Command:**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --dump-ids \
  --dump-cpp-ids

```

**Expected Output (stderr):**

```

🦀 Rust tokens (N total):
  [token1, token2, ...]

🔧 C++ tokens (N total, backend: bitnet|llama):
  [token1, token2, ...]

```

**Verification:**

- [ ] Both outputs appear on stderr
- [ ] Rust tokens shown first
- [ ] C++ tokens shown second
- [ ] Token sequences can be compared side-by-side

---

### Test 4: With JSON Output (Verify stderr separation)

**Command:**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --format json \
  --dump-ids \
  --dump-cpp-ids 2>/dev/null

```

**Expected Output (stdout only):**

```json
{
  "status": "ok",
  "backend": "bitnet",
  "divergence_token": -1,
  "metrics": {
    ...
  }
}

```

**Verification:**

- [ ] stdout contains only valid JSON (no emoji or debug output)
- [ ] JSON can be parsed with `jq`
- [ ] Token dumps went to stderr (not visible with `2>/dev/null`)

**To verify stderr separation:**

```bash

# Redirect stderr to see token dumps

cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --format json \
  --dump-ids \
  --dump-cpp-ids 2>&1 | tee /tmp/output.txt

# stdout should be valid JSON

cat /tmp/output.txt | grep -v "🦀" | grep -v "🔧" | jq .

```

---

### Test 5: Backend Auto-Detection

**Command (BitNet model):**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "test" \
  --max-tokens 1 \
  --dump-cpp-ids \
  --verbose

```

**Expected:**

- [ ] C++ tokens show `backend: bitnet` (auto-detected from path)
- [ ] Verbose output shows backend selection

**Command (LLaMA model):**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model /path/to/llama-model.gguf \
  --tokenizer /path/to/tokenizer.json \
  --prompt "test" \
  --max-tokens 1 \
  --dump-cpp-ids \
  --verbose

```

**Expected:**

- [ ] C++ tokens show `backend: llama` (auto-detected or fallback)

---

### Test 6: Template Integration

**Command (raw template):**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "2+2=" \
  --prompt-template raw \
  --max-tokens 1 \
  --dump-ids \
  --dump-cpp-ids

```

**Verification:**

- [ ] Rust and C++ token sequences match (raw template)
- [ ] Token counts are identical

**Command (instruct template):**

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --prompt-template instruct \
  --max-tokens 1 \
  --dump-ids \
  --dump-cpp-ids

```

**Verification:**

- [ ] Rust tokens include instruct formatting (e.g., "Q: ... A:")
- [ ] C++ tokens may differ if template not matched
- [ ] Can compare sequences to debug template issues

---

## Common Use Cases

### Debugging Token Mismatch

When `crossval-per-token` exits with code 2 (token mismatch):

```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model <model.gguf> \
  --tokenizer <tokenizer.json> \
  --prompt "<your prompt>" \
  --dump-ids \
  --dump-cpp-ids \
  --verbose 2>&1 | tee /tmp/debug.log

```

**Look for:**

- Different token counts between Rust and C++
- BOS token duplication (e.g., `[128000, 128000, ...]` vs `[128000, ...]`)
- Template formatting differences
- Special token handling differences

### Verifying BOS Token Handling

```bash

# With BOS (default)

cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model <model.gguf> \
  --tokenizer <tokenizer.json> \
  --prompt "test" \
  --dump-ids

# Without BOS (if flag exists)

cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model <model.gguf> \
  --tokenizer <tokenizer.json> \
  --prompt "test" \
  --no-bos \
  --dump-ids

```

Compare first token in output to verify BOS handling.

---

## Expected Behavior Summary

| Flag | Output Stream | Format | Purpose |
|------|---------------|--------|---------|
| `--dump-ids` | stderr | `🦀 Rust tokens (N total):\n  [...]` | Show Rust tokenization result |
| `--dump-cpp-ids` | stderr | `🔧 C++ tokens (N total, backend: X):\n  [...]` | Show C++ tokenization result |
| Both | stderr | Both outputs in sequence | Side-by-side comparison |

**Key Points:**

- Always output to stderr (preserves JSON stdout)
- Token IDs are shown in debug format: `[id1, id2, id3]`
- C++ output includes backend name for diagnostics
- Can be used with `--format json` without polluting output
- Useful for debugging tokenization parity issues

---

## Troubleshooting

### "C++ FFI not available"

**Solution:** Set up C++ reference:

```bash
eval "$(cargo run -p xtask --features crossval-all -- setup-cpp-auto --emit=sh)"

```

### "error while loading shared libraries"

**Solution:** Ensure library paths are set:

```bash
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH"

```

### No output on stderr

**Check:**

- Are you redirecting stderr to `/dev/null`?
- Run without redirection to see all output
- Verify flags are spelled correctly: `--dump-ids` not `--dump_ids`

### Token sequences don't match

**This is expected** when:

- Different templates are used (Rust vs C++)
- BOS token handling differs
- Special token parsing differs

**Use this output to diagnose the root cause!**

---

## Success Criteria

- [x] CLI parsing tests pass (4/4)
- [ ] Manual smoke test with real model confirms output format
- [ ] Both flags work independently
- [ ] Both flags work together
- [ ] Output goes to stderr (not stdout)
- [ ] Compatible with `--format json`
- [ ] Documentation comments added to source code
