# PR #468: BitNet.cpp Cross-Validation Parity Harness

## Summary

Implements real Rust↔BitNet.cpp parity validation to eliminate mocks and establish deterministic cross-validation infrastructure. This PR adds a production-ready harness that compares tokenization, logits, and greedy decoding between Rust and Microsoft's C++ implementation.

## What Changed

### Core Infrastructure

1. **C++ Shim (`bitnet_c_shim.cc`)**
   - Implements C API bridge to llama.cpp
   - Wraps `llama_model`, `llama_context`, and inference operations
   - Provides: `bitnet_tokenize`, `bitnet_eval`, `bitnet_decode_greedy`
   - **Fixed:** Parameter mapping for modern `llama_tokenize` API
     - BitNet `add_bos` → llama.cpp `add_special` (BOS insertion)
     - BitNet `parse_special` → llama.cpp `parse_special` (special token parsing)
   - **Fixed:** Greedy decode positions using `llama_get_kv_cache_token_count` for correct KV cache alignment

2. **Rust Wrappers (`bitnet-sys/src/wrapper.rs`)**
   - Safe wrappers around C shim: `BitnetModel`, `BitnetContext`
   - Helper functions: `bitnet_tokenize_text`, `bitnet_eval_tokens`, `bitnet_decode_greedy`
   - Memory safety via RAII (Drop implementations)

3. **Parity Test (`crossval/tests/parity_bitnetcpp.rs`)**
   - Tokenization parity: exact token ID comparison
   - Prefill logits parity: cosine similarity ≥ 0.99
   - N-step greedy decode parity: exact match rate = 1.0
   - Template-aware BOS/parse_special handling (LLaMA-3 chat support)
   - SHA256 model fingerprinting
   - Atomic receipt writing to workspace-anchored `docs/baselines/YYYY-MM-DD/`
   - **Fixed:** Receipt path uses `CARGO_MANIFEST_DIR` for workspace root (no chrono dependency)

### Template/BOS/EOT Contract

✅ **Verified correct behavior:**

- **Template Detection:** Auto-detects from GGUF metadata (`tokenizer.name`, `chat_template`)
- **BOS Handling:** Template-aware (LLaMA-3 chat: `add_bos=false`, Instruct: `add_bos=true`)
- **Special Token Parsing:** Template-aware (LLaMA-3 chat: `parse_special=true`, Instruct: `parse_special=false`)
- **Formatted Prompt:** Both Rust and C++ tokenize the **same** template-formatted string (no raw prompt leakage)
- **EOT Resolution:** For LLaMA-3 chat, encodes `"<|eot_id|>"` with `parse_special=true` to get token-level stop ID

### Quality Gates

- ✅ `cargo fmt --all` (clean)
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` (clean)
- ✅ Template/BOS/EOT contract verified
- ✅ llama.cpp API signatures synced

## How to Test

See [`CROSSVAL_TESTING.md`](./CROSSVAL_TESTING.md) for comprehensive testing guide.

**Quick Start:**

```bash
# 1. Set up environment
export BITNET_CPP_DIR=/path/to/bitnet.cpp/build
export CROSSVAL_GGUF=/path/to/model.gguf
export RAYON_NUM_THREADS=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42

# 2. Run parity test
cargo test -p crossval --features crossval,integration-tests -- parity_bitnetcpp --nocapture

# 3. Inspect receipt
jq . docs/baselines/$(date +%Y-%m-%d)/parity-bitnetcpp.json
```

**Expected Output:**

```
✓ Tokenization exact match
C++ parity check completed:
  Cosine similarity: 0.999876
  Cosine OK (≥0.99): true
  Exact match rate: 1.0000
  No divergence detected
✓ Parity receipt written to: docs/baselines/2025-10-16/parity-bitnetcpp.json
```

## Definition of Done

### ✅ Completed

- [x] C++ shim compiles and links against llama.cpp
- [x] llama.cpp API signatures correctly mapped (separate `add_bos` and `parse_special` flags)
- [x] Token IDs match exactly between Rust and C++
- [x] Logits cosine similarity ≥ 0.99
- [x] N-step greedy decode exact match (rate = 1.0)
- [x] Greedy decode positions use `llama_get_kv_cache_token_count` for correct alignment
- [x] Template/BOS/parse_special contract verified
- [x] Receipt written atomically with SHA256 fingerprint
- [x] Receipt path workspace-anchored (no chrono dependency)
- [x] Workspace passes `cargo fmt` and `clippy`
- [x] Testing guide documented

### 🔄 Follow-up (Separate PRs)

- [ ] CI workflow (label-triggered `crossval` job)
- [ ] Nightly baseline updates
- [ ] Replace fabricated TPS with real benches
- [ ] Multi-model test suite

## Technical Details

### Parameter Mapping Fixes

**1. Tokenization Flags (BOS vs parse_special):**

**After (Correct):**
```cpp
// C API signature
int bitnet_tokenize(bitnet_model_t*, const char* text,
                    int add_bos, int parse_special,
                    int32_t* out_ids, int out_cap);

// Shim implementation
llama_tokenize(model, text, text_len, tokens, n_max,
    (bool)add_bos,        // ✅ add_special: controls BOS insertion
    (bool)parse_special); // ✅ parse_special: parses "<|eot_id|>" markers
```

**Rust Usage:**
```rust
// For Instruct/Raw templates
let parse_special = false;  // Don't parse special markers
let add_bos = true;         // Add BOS token

// For LLaMA-3 Chat template
let parse_special = true;   // Parse "<|eot_id|>" etc.
let add_bos = false;        // No BOS (template handles it)
```

**2. Greedy Decode Positions:**

**After (Correct):**
```cpp
// Get actual KV cache length after prefill
int n_past = llama_get_kv_cache_token_count(c->context);

for (int step = 0; step < max_new_tokens; ++step) {
    // ... sample next_token ...
    batch.pos[0] = n_past + step;  // ✅ Correct position
    llama_decode(c->context, batch);
}
```

### Atomic Receipt Writing

Receipts are written using temp-file + atomic rename to prevent corruption:

```rust
let tmp_path = receipt_dir.join("parity-bitnetcpp.json.tmp");
fs::write(&tmp_path, serde_json::to_vec_pretty(&receipt)?)?;
fs::rename(&tmp_path, &receipt_path)?; // Atomic on POSIX
```

### Model Fingerprinting

SHA256 hashing ensures parity results are tied to specific model versions:

```rust
let model_sha = sha256_file(&gguf_path)?;
receipt["model_sha256"] = model_sha;
```

## Files Changed

### Modified
- `crates/bitnet-sys/csrc/bitnet_c_shim.cc` - C++ shim implementation
- `crates/bitnet-sys/src/wrapper.rs` - Rust FFI wrappers
- `crossval/tests/parity_bitnetcpp.rs` - Parity test harness
- `xtask/src/main.rs` - Allow unused `license` field in LockEntry

### Added
- `CROSSVAL_TESTING.md` - Comprehensive testing guide
- `docs/baselines/YYYY-MM-DD/parity-bitnetcpp.json` - Parity receipts (runtime)

## Dependencies

### Required
- `BITNET_CPP_DIR`: Path to built BitNet.cpp (llama.cpp fork)
- `CROSSVAL_GGUF`: Path to GGUF model for testing

### Optional (for determinism)
- `RAYON_NUM_THREADS=1`: Single-threaded execution
- `BITNET_DETERMINISTIC=1`: Enable deterministic mode
- `BITNET_SEED=42`: Fixed RNG seed

## Rationale

**Why this approach?**

1. **No Mocks:** Real C++ implementation validates Rust against production reference
2. **Deterministic:** Single-threaded, seeded execution ensures reproducible results
3. **Template-Aware:** Matches CLI behavior for LLaMA-3 chat and Instruct models
4. **Receipt-Based:** Atomic writes with SHA256 fingerprints provide audit trail
5. **Feature-Gated:** FFI only required when `crossval` feature enabled

**Why not use higher-level llama.cpp APIs?**

The C shim uses low-level `llama_tokenize`, `llama_batch`, and `llama_decode` to match BitNet.rs inference patterns exactly. This ensures apples-to-apples comparison without abstraction overhead.

## Migration Path (Future)

1. **Label-triggered CI:** Add `crossval` label to PRs affecting inference
2. **Nightly baselines:** Automated receipt generation and archival
3. **Benchmark integration:** Replace fabricated TPS with real metrics from C++ parity
4. **Policy validation:** Use receipts to validate performance regression thresholds

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| llama.cpp API changes | Comprehensive build.rs error messages with upgrade hints |
| Non-deterministic results | Enforce `RAYON_NUM_THREADS=1`, `BITNET_SEED`, single-threaded llama.cpp |
| Model corruption | SHA256 fingerprinting detects file changes |
| Template mismatch | Auto-detection with explicit override support |
| Link errors | RPATH auto-injection in build.rs (Linux/macOS) |

## References

- **Issue:** #439 (Remove cross-validation mocks, establish real parity)
- **Microsoft BitNet.cpp:** https://github.com/microsoft/BitNet
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **GGUF Spec:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

**Review Checklist:**

- [ ] C++ shim compiles on reviewer's machine (with `BITNET_CPP_DIR` set)
- [ ] Parity test produces `status: "ok"` receipt
- [ ] `cargo fmt` and `clippy` pass
- [ ] Testing guide is clear and actionable
- [ ] No regressions in existing workspace tests

**Merge Criteria:**

- All checkboxes in "Definition of Done" ✅
- At least one successful parity receipt in `docs/baselines/`
- CI passing (existing tests only - crossval CI comes in follow-up)
