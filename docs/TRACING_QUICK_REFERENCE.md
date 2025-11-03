# BitNet.rs Tracing Infrastructure - Quick Reference

## Current State at a Glance

### Struct Definition
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub name: String,              // e.g., "blk0/attn_norm"
    pub shape: Vec<usize>,         // e.g., [1, 2560]
    pub dtype: String,             // Data type before F32 conversion
    pub blake3: String,            // 64-char hex hash
    pub rms: f64,                  // Root mean square
    pub num_elements: usize,       // Product of shape dimensions
}
```

### JSON Output
```json
{
  "name": "blk0/attn_norm",
  "shape": [1, 2560],
  "dtype": "F32",
  "blake3": "abc123def456...",
  "rms": 0.9982,
  "num_elements": 2560
}
```

### API
```rust
pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()>
```

## File Organization

```
$BITNET_TRACE_DIR/
├── blk0_attn_norm.trace
├── blk0_q_proj.trace
├── blk0_attn_scores_softmax.trace
└── logits.trace
```

**Filename pattern:** `{sanitize(name)}.trace` where sanitize replaces `/` and `\` with `_`

## Environment Control

- **Enable:** `export BITNET_TRACE_DIR=/tmp/bitnet-traces`
- **Disable:** Unset or empty string
- **Zero-cost when disabled:** ~1-10 µs env var check

## Feature Gating

```toml
# In bitnet-models/Cargo.toml
trace = ["dep:bitnet-trace"]
```

```rust
// In transformer.rs
#[cfg(feature = "trace")]
{
    dump_trace("blk0/attn_norm", &x)?;
}
```

**When disabled:** Feature removed entirely from binary

## Blake3 Hashing

```rust
let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
let hash = blake3::hash(&bytes);
let blake3_hex = hash.to_hex().to_string();  // 64 chars
```

- 256-bit hash
- Little-endian F32 bytes
- Deterministic and cryptographically secure

## Performance

| State | Overhead |
|-------|----------|
| **Disabled** | ~1-10 µs per call |
| **Single tensor** | ~1-20 ms (file I/O dominant) |
| **2B model, 24 layers** | ~500 ms for prefill, ~50s for 128 tokens |

**Tip:** Use ramdisk (`/dev/shm`) for `BITNET_TRACE_DIR` to reduce I/O overhead

## Current Tracepoints

### In `forward_full()` (full sequence)
1. `t0/embeddings` - After embedding layer
2. `t0/blk{}/attn_norm` - Attention layer norm
3. `t0/blk{}/q_proj` - Query projection
4. `t0/blk{}/attn_scores_softmax` - Attention softmax
5. `t0/logits` - Final logits

### In `forward_incremental()` (token-by-token)
1. `t0/embeddings` - Current token embedding
2. `t0/logits` - Current token logits

## Extending for seq/layer/stage

### Phase 1: Add Optional Fields (Backward Compatible)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub blake3: String,
    pub rms: f64,
    pub num_elements: usize,
    
    // New optional fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq_index: Option<usize>,   // Token position
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_idx: Option<usize>,   // Layer number
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,      // "embeddings", "attention", "mlp"
}
```

**Key points:**
- Uses `skip_serializing_if` to omit None values from JSON
- Old JSON without these fields still parses (defaults to None)
- New JSON with fields parses in old code (fields ignored)

### Phase 2: New API

```rust
pub fn dump_trace_with_context(
    name: &str,
    tensor: &Tensor,
    seq_index: Option<usize>,
    layer_idx: Option<usize>,
    stage: Option<&str>,
) -> candle_core::Result<()>;

pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()> {
    dump_trace_with_context(name, tensor, None, None, None)
}
```

### Phase 3: Update Call Sites

```rust
// Old style (still works)
dump_trace("blk0/attn_norm", &x)?;

// New style with context
dump_trace_with_context("attn_norm", &x, Some(0), Some(0), Some("attention"))?;
```

## Testing

### Unit Tests (src/lib.rs)
- `test_sanitize_filename()` - Path sanitization
- `test_dump_trace_disabled()` - Silent return when disabled
- `test_trace_record_serialization()` - JSON round-trip

### Integration Tests (tests/integration_test.rs)
- `test_dump_trace_integration()` - Full workflow
- `test_dump_trace_creates_directory()` - Auto-creation
- `test_dump_trace_different_dtypes()` - F32/F64 handling
- `test_dump_trace_various_shapes()` - 1D-4D tensors
- `test_dump_trace_empty_trace_dir()` - Disabled state

## Cross-Validation Workflow

```bash
# 1. Capture Rust traces
export BITNET_TRACE_DIR=/tmp/rust-traces
cargo run -p bitnet-cli --features cpu,full-cli,trace -- run \
  --model model.gguf --prompt "2+2=" --max-tokens 4

# 2. Compare with C++ reference (if available)
export BITNET_TRACE_DIR=/tmp/cpp-traces
./bitnet-cpp --model model.gguf --prompt "2+2=" --n-predict 4

# 3. Analyze
jq -r '.blake3' /tmp/rust-traces/*.trace | sort > rust.hashes
jq -r '.blake3' /tmp/cpp-traces/*.trace | sort > cpp.hashes
diff rust.hashes cpp.hashes

# 4. Automated sweep (runs 3 scenarios)
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `crates/bitnet-trace/src/lib.rs` | 194 | Main implementation |
| `crates/bitnet-trace/README.md` | 120 | User guide |
| `crates/bitnet-trace/tests/integration_test.rs` | 177 | Integration tests |
| `crates/bitnet-models/src/transformer.rs` | ~8 calls | Call sites |
| `scripts/run_crossval_sweep.sh` | - | Complete workflow |

## Backward Compatibility Strategy

**Current:** Simple schema with 6 fields

**To add seq/layer/stage:**
1. Add `Option<T>` fields with `#[serde(skip_serializing_if = "Option::is_none")]`
2. Create new API function, keep old one
3. Migrate call sites gradually
4. Old JSON and old code continue to work

**Result:** Zero breaking changes, progressive adoption possible

## Design Principles

✅ **Zero-cost when disabled** - No overhead if env var not set
✅ **Feature-gated** - Compiled out when `--features trace` not used
✅ **Stateless** - No global mutable state
✅ **Sequential I/O** - No locks or synchronization needed
✅ **Simple API** - Single public function for common case
✅ **Self-describing** - Field names and format self-documenting

---

See `docs/TRACING_INFRASTRUCTURE_ANALYSIS.md` for comprehensive deep dive.
