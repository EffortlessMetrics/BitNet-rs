# BitNet.rs Tensor Tracing Infrastructure - Comprehensive Analysis

## Executive Summary

The `bitnet-trace` crate provides a lightweight, environment-controlled tensor activation tracing system for cross-validation debugging. It captures Blake3 hashes, RMS statistics, and tensor metadata during inference, writing results to JSON trace files.

**Key characteristics:**
- **Zero-cost when disabled**: No overhead if `BITNET_TRACE_DIR` is not set
- **Feature-gated**: Compiled in only when `--features trace` is used
- **Simple API**: Single public function `dump_trace(name, tensor)`
- **Stateless**: No global mutable state; traces written directly to disk
- **Sequential I/O**: No thread synchronization needed

---

## Current Struct Definition: `TraceRecord`

### Complete Definition

```rust
/// Trace record containing tensor metadata and hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Tensor name/identifier
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Blake3 hash of raw F32 bytes
    pub blake3: String,
    /// Root mean square of tensor values
    pub rms: f64,
    /// Total number of elements
    pub num_elements: usize,
}
```

**Field analysis:**
- `name: String` — User-provided identifier (e.g., "blk0/attn_norm"). Sanitized by replacing `/` and `\` with `_` for filenames.
- `shape: Vec<usize>` — Tensor dimensions (e.g., `[1, 2560]`)
- `dtype: String` — Data type **before conversion to F32** (captured as `format!("{:?}", tensor.dtype())`)
- `blake3: String` — 64-character hex hash of tensor data as F32 little-endian bytes
- `rms: f64` — Root mean square (numerical stability indicator)
- `num_elements: usize` — Product of shape dimensions

**Serialization:**
- Uses `serde` with `#[derive(Serialize, Deserialize)]`
- Writes as pretty-printed JSON
- 6 fields total in JSON output

---

## Current JSON Output Example

```json
{
  "name": "blk0/attn_norm",
  "shape": [
    1,
    2560
  ],
  "dtype": "F32",
  "blake3": "abc123def456789...",
  "rms": 0.9982,
  "num_elements": 2560
}
```

**Notes:**
- Blake3 hash is 64 characters (256-bit hash in hex)
- RMS uses f64 precision (double)
- Shape is always a vector of `usize` (works for any dimensionality)
- Element count computed as product of shape dimensions

---

## File Naming Convention

**Pattern:** `{sanitized_name}.trace`

**Sanitization:**
```rust
fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\'], "_")
}
```

**Examples:**
- Input: `"blk0/attn_norm"` → File: `blk0_attn_norm.trace`
- Input: `"layer\weights"` → File: `layer_weights.trace`
- Input: `"simple_name"` → File: `simple_name.trace`

**Directory structure:**
```
$BITNET_TRACE_DIR/
├── blk0_embeddings.trace
├── blk0_attn_norm.trace
├── blk0_q_proj.trace
├── blk0_attn_scores_softmax.trace
├── blk0_ffn_norm.trace
├── blk0_ffn_output.trace
├── blk1_embeddings.trace
├── blk1_attn_norm.trace
...
└── logits.trace
```

---

## API Surface: Public Functions

### Main Function: `dump_trace()`

```rust
pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()>
```

**Behavior:**
1. Check if `BITNET_TRACE_DIR` environment variable is set
2. If not set or empty → return `Ok(())` immediately (silent, zero-cost)
3. If set:
   - Create trace directory (with parent dirs) if it doesn't exist
   - Convert tensor to F32 and flatten
   - Compute Blake3 hash of F32 little-endian bytes
   - Compute RMS (root mean square) of values
   - Create `TraceRecord` struct
   - Write JSON to `{trace_dir}/{sanitized_name}.trace`
   - Return `Ok(())`

**Error handling:**
- Returns `Err` if:
  - Tensor to F32 conversion fails
  - Directory creation fails
  - JSON serialization fails
  - File write fails
- All errors wrapped in `candle_core::Error`

**Performance:**
- **When disabled** (no `BITNET_TRACE_DIR`): Single env var check, ~0ns overhead
- **When enabled**: 
  - F32 conversion: O(n) where n = element count
  - Blake3 hash: O(n) high-throughput cryptographic hash
  - RMS: O(n) single pass over flattened data
  - JSON serialization: O(m) where m = JSON size (~200-500 bytes per trace)
  - File I/O: Sequential disk write

### Public Struct: `TraceRecord`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord { /* ... */ }
```

**Visibility:** Fully public with all fields public. Can be:
- Constructed manually: `TraceRecord { name, shape, dtype, blake3, rms, num_elements }`
- Serialized/deserialized by external code
- Used in tests and integration code

---

## Blake3 Hashing Implementation

**Location:** Line 117-120 in `lib.rs`

```rust
// Compute Blake3 hash of raw bytes
let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
let hash = blake3::hash(&bytes);
let blake3_hex = hash.to_hex().to_string();
```

**Details:**
1. Convert each F32 value to 4 bytes (little-endian)
2. Collect into single byte vector
3. Pass to `blake3::hash()` from `blake3 v1.5` crate
4. Convert resulting hash to 64-character hex string

**Cryptographic properties:**
- 256-bit hash (64 hex chars)
- Fast and secure
- Deterministic (same tensor → same hash)
- Sensitive to byte-level changes

**Data format:**
- Little-endian F32 bytes (standard on x86/ARM)
- Handles both positive and negative floats
- IEEE-754 NaN values represented consistently

---

## Environment Variable Control

### Primary Control: `BITNET_TRACE_DIR`

**Checking logic** (line 96-99):
```rust
let trace_dir = match env::var("BITNET_TRACE_DIR") {
    Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
    _ => return Ok(()), // Tracing disabled - return silently
};
```

**Behavior:**
- If `BITNET_TRACE_DIR` is not set → Disabled (silent return)
- If `BITNET_TRACE_DIR=""` (empty string) → Disabled (silent return)
- If `BITNET_TRACE_DIR=/path/to/dir` → Enabled, uses specified directory

**Typical usage:**
```bash
export BITNET_TRACE_DIR=/tmp/bitnet-traces
cargo run -p bitnet-cli -- run --model model.gguf --prompt "test"
```

**No other environment variables control tracing.**

---

## Feature Gating

### In `bitnet-models/Cargo.toml`

```toml
[features]
trace = ["dep:bitnet-trace"]  # Enable tensor activation tracing for cross-validation
```

### In `transformer.rs` usage

```rust
// Tracepoint 1: Embeddings output (after embed, before layers)
#[cfg(feature = "trace")]
{
    let first_token_emb = hidden.narrow(1, 0, 1)?;
    dump_trace("t0/embeddings", &first_token_emb)
        .map_err(BitNetError::from)?;
}
```

**Compilation:**
- When `--features trace` is NOT specified: `dump_trace` calls are removed entirely
- When `--features trace` is specified: Trace calls compiled in and may execute

**Zero-cost when disabled:**
- No `bitnet-trace` dependency at all when feature not enabled
- No binary size impact
- No runtime overhead

---

## Current Usage in `bitnet-models/transformer.rs`

### Tracepoints Currently Implemented

**Path: `forward_full()` method** (full sequence inference):
1. **Embeddings output** (`t0/embeddings`)
   - First token's embedding [B, 1, H] after embedding layer
   - Before any transformer blocks

2. **Attention norm** (`t0/blk{}/attn_norm`)
   - Per-layer output of attention layer norm
   - Layer-specific index `self.attention.layer_idx`

3. **Q projection output** (`t0/blk{}/q_proj`)
   - Query projection [B, H, T, D]
   - Layer-specific tracepoint

4. **Attention scores softmax** (`t0/blk{}/attn_scores_softmax`)
   - After softmax normalization
   - Attention weights [B, H, T, T]

5. **Final logits** (`t0/logits`)
   - Output logits [B, 1, V]
   - First token only (narrowed to single position)

**Path: `forward_incremental()` method** (token-by-token generation):
1. **Embeddings output** (`t0/embeddings`)
   - Current token embedding [B, H]
   - No narrowing needed (already single token)

2. **Final logits** (`t0/logits`)
   - Output logits [B, V]
   - Single token (no narrowing)

### Naming Convention

- Prefix: `t0/` (sequence/transaction identifier, currently hardcoded as "t0")
- Format: `t0/{path}/{optional_index}`
- Examples:
  - `t0/embeddings`
  - `t0/blk0/attn_norm`
  - `t0/blk0/q_proj`
  - `t0/logits`

**Observation:** 
- No explicit "sequence number" field
- No "generation step" tracking
- Name string encodes layer information manually

---

## What Fields Need to Be Added for seq/layer/stage

### Proposed Extensions

Based on the current usage patterns and the CLAUDE.md requirements for comprehensive tracing, we should add:

#### Option 1: Add Explicit Struct Fields (Backward Compatible)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    // Existing fields
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub blake3: String,
    pub rms: f64,
    pub num_elements: usize,
    
    // New optional fields for structured hierarchy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_index: Option<usize>,  // Token position in generation (0 for prefill)
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<usize>,     // Transformer layer (0, 1, 2, ...)
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,          // Computation stage (e.g., "embeddings", "attention", "mlp", "output")
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_type: Option<String>,     // Block variant (e.g., "attention_norm", "q_proj", "attn_scores")
}
```

**Advantages:**
- Backward compatible: old code without new fields still parses
- Self-documenting: field names are clear
- Easy to filter/search: `jq 'select(.layer_index == 0)'`

**Disadvantages:**
- Slightly larger JSON (though `skip_serializing_if` mitigates)
- Requires changes to all call sites

#### Option 2: Keep Name String, Add Parse Helper (Zero-Change Compatibility)

Keep current `TraceRecord` unchanged, provide utility functions:

```rust
impl TraceRecord {
    /// Parse sequence index from trace name (e.g., "t42/..." → Some(42))
    pub fn parse_sequence_index(&self) -> Option<usize> {
        self.name.split('/').next()
            .and_then(|s| s.strip_prefix('t'))
            .and_then(|s| s.parse().ok())
    }
    
    /// Parse layer index from trace name (e.g., "t0/blk3/..." → Some(3))
    pub fn parse_layer_index(&self) -> Option<usize> {
        self.name.split('/').find_map(|part| {
            part.strip_prefix("blk").and_then(|s| s.parse().ok())
        })
    }
    
    /// Extract stage from trace name (e.g., "t0/blk0/attn_norm" → "attn_norm")
    pub fn parse_stage(&self) -> Option<String> {
        self.name.split('/').last().map(|s| s.to_string())
    }
}
```

**Advantages:**
- Zero breaking changes
- Existing code continues to work unchanged
- New features available via parsing

**Disadvantages:**
- Coupling between name format and logic
- Name format is implicit contract

---

## Recommended Schema Extension

### For Phase 1: Backward-Compatible Addition

Add optional fields with `#[serde(skip_serializing_if)]`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    // Existing fields (unchanged)
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub blake3: String,
    pub rms: f64,
    pub num_elements: usize,
    
    // New optional fields (default to None/skipped in JSON if not set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq_index: Option<usize>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_idx: Option<usize>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
}
```

### For Phase 2: Update API

```rust
pub fn dump_trace_with_context(
    name: &str,
    tensor: &Tensor,
    seq_index: Option<usize>,
    layer_idx: Option<usize>,
    stage: Option<&str>,
) -> candle_core::Result<()> {
    // Implementation uses new fields when provided
}

// Keep existing dump_trace() for backward compatibility
pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()> {
    dump_trace_with_context(name, tensor, None, None, None)
}
```

### For Phase 3: Update Call Sites

```rust
// Current usage
dump_trace("blk0/attn_norm", &x)?;

// Future usage with context
dump_trace_with_context("attn_norm", &x, Some(0), Some(0), Some("attention"))?;
```

---

## How to Extend Schema Without Breaking Existing Code

### Strategy 1: Serde's `skip_serializing_if` (Recommended)

```rust
#[serde(skip_serializing_if = "Option::is_none")]
pub field: Option<T>,
```

**Effect:**
- If field is `None`, JSON omits it entirely
- Old code parsing without the field still works (defaults to `None`)
- New code can provide values

**Example:**
```rust
// Old JSON format (no new field)
{ "name": "blk0", "blake3": "abc..." }

// Parses successfully to:
TraceRecord {
    name: "blk0",
    blake3: "abc...",
    seq_index: None,  // Default
    // ... other fields
}

// New JSON format (with new field)
{ "name": "blk0", "blake3": "abc...", "seq_index": 0 }

// Old code using struct literal still works
let record = TraceRecord {
    name: "blk0".into(),
    blake3: "abc...".into(),
    seq_index: None,  // Can be None
    // ...
};
```

### Strategy 2: Enum-Based Versioning (If Major Changes Needed)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "version")]
pub enum TraceRecord {
    #[serde(rename = "1.0")]
    V1 {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        blake3: String,
        rms: f64,
        num_elements: usize,
    },
    #[serde(rename = "2.0")]
    V2 {
        // All V1 fields plus new ones
        name: String,
        shape: Vec<usize>,
        dtype: String,
        blake3: String,
        rms: f64,
        num_elements: usize,
        seq_index: usize,
        layer_idx: usize,
        stage: String,
    },
}
```

**Not recommended for current use case** (simpler to use optional fields).

### Best Practice

1. **Use `Option<T>` with `#[serde(skip_serializing_if)]`** for new fields
2. **Update API** with new functions (keep old ones)
3. **Gradually migrate** call sites
4. **Document** in README and examples

---

## Performance Characteristics

### Disabled (BITNET_TRACE_DIR not set)

```
Time per dump_trace() call:
├─ env::var() lookup: ~1-10 microseconds
├─ String comparison: negligible
└─ Return Ok(()):  ~0 ns
───────────────────────
TOTAL: ~1-10 microseconds per call (effectively free)
```

**Memory:** Zero additional allocation

**Code size:** When `--features trace` not used, entire `bitnet-trace` crate excluded from binary

---

### Enabled (BITNET_TRACE_DIR set)

**Per tensor traced:**

```
Conversion to F32:     O(n)    ~100-500 ns/element (high-bandwidth)
                              = 0.1-0.5 ms for 1M-element tensor

Compute RMS:           O(n)    ~100-300 ns/element
                              = 0.1-0.3 ms for 1M-element tensor

Blake3 hash:           O(n)    ~50-100 ns/element (highly optimized)
                              = 0.05-0.1 ms for 1M-element tensor

JSON serialization:    O(m)    where m = JSON size
                              ~1-5 µs per field
                              = 0.01-0.05 ms per record

File I/O:              O(m)    ~1-10 ms per file write (disk dependent)

env::var() check:      ~1 µs
────────────────────────────
TOTAL per trace:       ~1-20 ms (dominated by file I/O)
```

**Characteristics:**
- **CPU-bound operations** (F32 conversion, hashing, RMS) are bandwidth-limited
- **I/O is the bottleneck** (disk write ~1-10 ms per file)
- **Sequential model**: No parallelism needed, no locks

### Real-World Overhead (Inference)

**Example: 2B model with ~24 layers**

```
Scenario 1: Single token (prefill)
├─ Embeddings: 1 trace    =  ~20 ms file I/O
├─ 24 layer traces         = ~480 ms file I/O
├─ Final logits: 1 trace   =  ~20 ms file I/O
└─ Total: ~520 ms

Scenario 2: 128 tokens generated
├─ 128 × (24 layers + embeddings + logits) traces
├─ ~4K trace files
├─ ~40-60 seconds file I/O
└─ Inference time: ~1-3 seconds QK256 MVP (~0.1 tok/s)
   Trace overhead: ~10-60x depending on disk speed
```

**Recommendation:**
- Use ramdisk for `BITNET_TRACE_DIR` (e.g., `/dev/shm` on Linux)
- Trace only key layers for production
- Use SSD for development tracing

---

## Testing & Validation

### Unit Tests (in `src/lib.rs`)

```rust
#[test]
fn test_sanitize_filename()
    // Verifies path separator replacement

#[test]
fn test_dump_trace_disabled()
    // Confirms silent return when BITNET_TRACE_DIR not set

#[test]
fn test_trace_record_serialization()
    // Round-trip JSON serialization
```

### Integration Tests (in `tests/integration_test.rs`)

```rust
#[test]
fn test_dump_trace_integration()
    // Full workflow: create tensor, dump trace, verify file

#[test]
fn test_dump_trace_creates_directory()
    // Directory auto-creation

#[test]
fn test_dump_trace_different_dtypes()
    // F32, F64 handling

#[test]
fn test_dump_trace_various_shapes()
    // 1D through 4D tensors

#[test]
fn test_dump_trace_empty_trace_dir()
    // Empty string disables tracing
```

**Test strategy:**
- Use `tempfile::TempDir` for isolated test directories
- Serialize environment variable access with `Mutex`
- Verify file existence and JSON parsing
- Validate Blake3 computation deterministically
- Check RMS calculation accuracy

---

## Dependencies

```toml
[dependencies]
blake3 = "1.5"              # Fast cryptographic hashing
candle-core = { workspace = true }  # Tensor operations
serde = { version = "1.0", features = ["derive"] }  # Serialization
serde_json = "1.0"          # JSON format

[dev-dependencies]
tempfile = "3.8"            # Temporary test directories
```

---

## Typical Workflow: Cross-Validation

### 1. Capture Rust Traces

```bash
export BITNET_TRACE_DIR=/tmp/rust-traces
RUSTFLAGS="-C target-cpu=native" cargo run -p bitnet-cli \
  --no-default-features --features cpu,full-cli,trace -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "2+2=" \
  --max-tokens 4 \
  --temperature 0.0 --greedy
```

### 2. Capture Reference Traces (C++ if available)

```bash
export BITNET_TRACE_DIR=/tmp/cpp-traces
./bitnet-cpp/bin/main \
  --model models/model.gguf \
  --prompt "2+2=" \
  --n-predict 4
```

### 3. Compare Traces

```bash
# Extract Blake3 hashes
jq -r '.blake3' /tmp/rust-traces/*.trace | sort > /tmp/rust-hashes
jq -r '.blake3' /tmp/cpp-traces/*.trace | sort > /tmp/cpp-hashes

# Diff
diff /tmp/rust-hashes /tmp/cpp-hashes

# Extract RMS values
jq -r '.rms' /tmp/rust-traces/*.trace | sort > /tmp/rust-rms
jq -r '.rms' /tmp/cpp-traces/*.trace | sort > /tmp/cpp-rms

# Visual comparison
paste /tmp/rust-rms /tmp/cpp-rms
```

### 4. Automated Script

See `scripts/run_crossval_sweep.sh` for complete workflow:
- Runs 3 deterministic scenarios (1, 2, 4 tokens)
- Captures traces for each
- Compares Blake3 hashes
- Generates cosine similarity metrics
- Produces markdown report

---

## Summary: Schema for seq/layer/stage Support

### Minimal Non-Breaking Extension

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub blake3: String,
    pub rms: f64,
    pub num_elements: usize,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq_index: Option<usize>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_idx: Option<usize>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
}
```

### New API (alongside existing)

```rust
pub fn dump_trace_with_context(
    name: &str,
    tensor: &Tensor,
    seq_index: Option<usize>,
    layer_idx: Option<usize>,
    stage: Option<&str>,
) -> candle_core::Result<()>;

pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()>;
```

### Advantages

✅ **Backward compatible**: Old code/JSON files work unchanged
✅ **Zero-cost when fields not used**: JSON omits None fields
✅ **Progressive adoption**: Gradual migration possible
✅ **Self-documenting**: Field names explicit
✅ **Query-friendly**: `jq` easily filters by layer/stage
✅ **Version-safe**: Can deserialize both old and new formats

---

## Files Modified/Created

### Current
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-trace/src/lib.rs` (194 lines)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-trace/README.md` (120 lines)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-trace/tests/integration_test.rs` (177 lines)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-trace/Cargo.toml` (16 lines)

### Used in
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs` (~8 trace calls)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/Cargo.toml` (trace feature)

---

## Recommendations

### For Immediate Use (MVP)
- Keep current API
- Add optional fields with `#[serde(skip_serializing_if)]`
- Provide helper functions to parse seq/layer/stage from name strings

### For Phase 2 (Post-MVP)
- Create new `dump_trace_with_context()` API
- Migrate call sites gradually
- Keep `dump_trace()` for backward compatibility

### For Testing
- Expand integration tests to verify new fields
- Add property tests for Blake3 determinism
- Test round-trip serialization with new fields

---

## References

- **Crate**: `crates/bitnet-trace/`
- **Usage**: `crates/bitnet-models/src/transformer.rs` (7 call sites, 2 paths)
- **Scripts**: `scripts/run_crossval_sweep.sh` (comprehensive workflow)
- **Docs**: `crates/bitnet-trace/README.md` (usage guide)
- **Tests**: `crates/bitnet-trace/tests/integration_test.rs` (5 tests)

