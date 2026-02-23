# BitNet.rs FFI & Tracing - Quick Reference

## 1-Minute Overview

| Component | Location | Purpose |
|-----------|----------|---------|
| **C FFI API** | `crates/bitnet-ffi/src/c_api.rs` | 30+ C functions for model loading, inference, streaming |
| **Trace Capture** | `crates/bitnet-trace/src/lib.rs` | Per-layer activation capture (Blake3 hashing) |
| **C++ Bridge** | `crossval/src/cpp_bindings.rs` | Safe Rust wrappers around C++ reference |
| **Logits Compare** | `crossval/src/logits_compare.rs` | Per-position divergence detection |
| **Weight Mapping** | `crates/bitnet-models/src/transformer.rs` | Tied weights, transposition, quantization handling |
| **Trace Diff Tool** | `xtask/src/trace_diff.rs` | Compare Rust vs C++ traces |

---

## 4 Key Capabilities

### 1. Token Passing
**Status**: Strings only in public C API; tokens available in `bitnet_sys::wrapper::Session`
```bash
# Current: Prompt string only
bitnet_inference(model_id, "What is 2+2?", output, max_len);

# Need: Token-level API (NOT YET EXPOSED)
# Use bitnet_sys::wrapper::Session for direct token access
```

### 2. Trace Capture
**Easy 1-line setup**:
```bash
BITNET_TRACE_DIR=/tmp/traces cargo run -p bitnet-cli -- run --model model.gguf --prompt "test"
# Output: /tmp/traces/*.trace (JSON files with Blake3 hashes)
```

### 3. Logits Divergence
**Metrics**: Cosine similarity, L2 distance, max absolute difference
```rust
let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);
// Returns: first_divergence_token, per_token_cosine_sim, per_token_l2_dist, max_absolute_diff
```

### 4. Head-Tie Validation
**Automatic handling** of tied weights (embedding ↔ LM head):
```rust
// Tied weights (no dedicated LM head)
if lm_head.is_none() {
    // Pre-transpose embeddings once at load → cached_weight
    hidden.matmul(&embed_tied_weight)?
}

// Dedicated LM head
else {
    lm_head.forward(hidden)?
}
```

---

## Essential Commands

```bash
# 1. Capture Rust traces
BITNET_TRACE_DIR=/tmp/rs BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu,trace -- run \
  --model model.gguf --tokenizer tok.json --prompt "test" --max-tokens 4 --greedy

# 2. Compare with C++ traces
cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp

# 3. Debug weight mapping
BITNET_DEBUG_LOGITS=1 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "test" --max-tokens 1

# 4. Per-position logits divergence (requires C++ setup)
BITNET_CPP_DIR=/path/to/bitnet.cpp \
  cargo test -p crossval --test per_position_logits -- --nocapture
```

---

## Weight Mapping Rules

| Scenario | Path | Handling |
|----------|------|----------|
| **Tied weights** | No `lm_head` | Pre-transpose [V,H]→[H,V] at load, cache in `embed_tied_weight` |
| **Dedicated head** | Has `lm_head` | Use Linear layer directly |
| **Transposed embed** | Stored as [H,V] | Use raw embedding matrix (no transpose) |
| **Standard embed** | Stored as [V,H] | Use cached pre-transposed weight |

---

## Trace Capture Points

**Key stages instrumented**:
1. **Embeddings** (layer=-1, stage="embeddings")
2. **Q/K/V Projections** (stage="q_proj", "k_proj", "v_proj")
3. **Attention Output** (stage="attn_out")
4. **FFN Output** (stage="ffn_out")
5. **Layer Norm** (intermediate)
6. **Logits** (layer=-1, stage="logits")

Each trace includes:
- Shape, dtype, Blake3 hash
- RMS (Root Mean Square)
- Token position (seq=0 is prefill, seq=1+ is decode)
- Layer index (-1 for embeddings/logits, 0+ for transformer)

---

## Troubleshooting

### No traces captured?
```bash
# Check BITNET_TRACE_DIR is set
echo $BITNET_TRACE_DIR
# Check directory is writable
touch $BITNET_TRACE_DIR/test && rm $BITNET_TRACE_DIR/test
```

### C++ FFI not available?
```bash
# Set BITNET_CPP_DIR
export BITNET_CPP_DIR=/path/to/bitnet.cpp
# Or use scalar crossval (doesn't require C++)
cargo test -p bitnet-quantization --features cpu
```

### Token-level control needed?
```rust
// Use lower-level API:
use bitnet_sys::wrapper::Session as CppSession;
let mut session = CppSession::load_deterministic(model_path)?;
let tokens = session.tokenize("prompt")?;  // Tokens directly
let logits = session.eval_and_get_logits(&tokens, 0)?;
```

### Tied weights not working?
```bash
# Debug with:
BITNET_DEBUG_LOGITS=1 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "test" --max-tokens 1
# Look for: "Pre-transposing tied embeddings" message
```

---

## File Reference

**Core FFI**:
- `crates/bitnet-ffi/src/c_api.rs` — C API functions
- `crates/bitnet-ffi/src/inference.rs` — Inference manager

**Tracing**:
- `crates/bitnet-trace/src/lib.rs` — Trace capture
- `xtask/src/trace_diff.rs` — Trace comparison

**Cross-Validation**:
- `crossval/src/cpp_bindings.rs` — C++ FFI wrappers
- `crossval/src/logits_compare.rs` — Per-position divergence
- `crossval/tests/per_position_logits.rs` — Divergence tests

**Weight Mapping**:
- `crates/bitnet-models/src/transformer.rs:1609-1740` — Logits computation
- `crates/bitnet-models/src/transformer.rs:1355-1375` — Head-tie loading

---

## Next Steps

1. **Need token-level FFI?** Implement `bitnet_tokenize()` and `bitnet_inference_tokens()` in `c_api.rs`
2. **Need streaming logits?** Extend `BitNetCStreamConfig` to support logits callbacks
3. **Need multi-GPU?** Add `device_id` parameter to FFI functions
4. **Need trace filtering?** Add `--trace-filter layer:stage` to CLI

See `docs/reports/FFI_TRACE_INFRASTRUCTURE.md` for comprehensive documentation.
