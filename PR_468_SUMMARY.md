# PR #468: BitNet.cpp Cross-Validation Parity Harness

## Why

Removes cross-val mocks by implementing real Rust↔bitnet.cpp parity: tokenization equality, prefill logits cosine, and deterministic greedy decode. Emits atomic parity receipts with sha256.

## What Changed

* C++ shim compiled and linked via `build.rs` (llama C API)
* Parity harness compares tokens, logits (cosine), and N-step greedy; writes receipt to `docs/baselines/YYYY-MM-DD/parity-bitnetcpp.json`
* Template/BOS/parse_special contract matches CLI (GGUF metadata → `TemplateType::detect` → fallback Instruct)
* Deterministic flags recorded; workspace-anchored receipt path
* Fixed C++ greedy decode position tracking (`n_past + (generated - 1)`)
* Cosine similarity handles both-zero-vector case correctly
* Receipt path uses safer date parsing and supports `BASELINES_DIR` env override

## How to Run

```bash
# Fetch model
cargo run -p xtask -- fetch-models --lock crossval-models.lock.json | tee /tmp/fetch.json
export CROSSVAL_GGUF=$(jq -r '.local // .[0].local' /tmp/fetch.json)

# Point to bitnet.cpp build
export BITNET_CPP_DIR=/path/to/bitnetcpp/build

# Set deterministic flags
export RAYON_NUM_THREADS=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42

# Run parity test
cargo test -p crossval --features crossval,integration-tests -- parity_bitnetcpp -- --nocapture

# Inspect receipt
jq . docs/baselines/$(date +%Y-%m-%d)/parity-bitnetcpp.json
```

**Expected Receipt:**
```json
{
  "cpp_available": true,
  "cosine_similarity": 0.999876,
  "cosine_ok": true,
  "exact_match_rate": 1.0,
  "first_divergence_step": null,
  "status": "ok",
  "model_sha256": "<hash>",
  "validation": {
    "deterministic": true,
    "threads": 1,
    "seed": 0
  }
}
```

## Definition of Done

* [x] Shim compiles/links (llama C API)
* [x] Formatted prompt used on both sides
* [x] BOS/parse_special mapped per template
* [x] Cosine ≥ 0.99; exact-match 1.0; no divergence; `"status":"ok"`
* [x] Atomic receipt with `model_sha256`
* [x] Receipt path workspace-anchored with safe date parsing
* [x] C++ greedy position tracking fixed (`n_past + (generated - 1)`)
* [x] Cosine similarity handles zero vectors correctly
* [x] fmt+clippy green

## Technical Details

### Parameter Mapping

**Tokenization Flags:**
```cpp
// C API signature
int bitnet_tokenize(bitnet_model_t*, const char* text,
                    int add_bos, int parse_special,
                    int32_t* out_ids, int out_cap);

// Shim implementation
llama_tokenize(model, text, text_len, tokens, n_max,
    (bool)add_bos,        // add_special: controls BOS insertion
    (bool)parse_special); // parse_special: parses "<|eot_id|>" markers
```

**Template Contract:**
```rust
// For Instruct/Raw templates
add_bos = true;         parse_special = false;

// For LLaMA-3 Chat template
add_bos = false;        parse_special = true;
```

**Greedy Decode Position Fix:**
```cpp
// Get actual KV cache length after prefill
int n_past = llama_get_kv_cache_token_count(c->context);

for (int step = 0; step < max_new_tokens; ++step) {
    // ... sample next_token ...
    io_ids[generated] = next_token;
    ++generated;

    // Decode at correct position (off-by-one fix)
    batch.pos[0] = n_past + (generated - 1);  // ✅ Correct
    llama_decode(c->context, batch);
}
```

**Receipt Path Safety:**
```rust
// Safer than slicing fixed indices: split at 'T'
let date_str = ts.split_once('T').map(|(d, _)| d).unwrap_or("1970-01-01");

// Allow override for CI: BASELINES_DIR=/path/to/workspace/docs/baselines
let base_dir = std::env::var("BASELINES_DIR")
    .ok()
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/baselines"));
```

## Files Changed

### Modified
- `crates/bitnet-sys/csrc/bitnet_c_shim.cc` - Fixed greedy decode position tracking
- `crates/bitnet-sys/include/bitnet_c.h` - Tokenization signature (already correct)
- `crates/bitnet-sys/src/wrapper.rs` - Safe wrappers (already correct)
- `crossval/tests/parity_bitnetcpp.rs` - Parity harness with fixes:
  - Receipt path uses safer date parsing and env override
  - Formatted prompt consistency
  - Cosine similarity zero-vector handling
  - Updated documentation
- `xtask/src/main.rs` - Documented unused `license` field

## Follow-up PRs

* Label `crossval` CI job (fetch models, run parity, upload receipts)
* Honest benches (replace fabricated TPS)
* `xtask gen-baselines` for baseline management
* Production mock removal (fail-fast with actionable errors)

## References

- **Issue:** #439
- **Microsoft BitNet.cpp:** https://github.com/microsoft/BitNet
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
