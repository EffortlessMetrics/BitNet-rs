# C++ FFI for Cross-Validation: Quick Reference

## Current Capabilities (All Ready)

### Per-Position Logits Extraction
```rust
// 1. Initialize C++ backend
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());

// 2. Load model (deterministic, single-threaded)
let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path)?;

// 3. Tokenize
let tokens = cpp_session.tokenize("The capital of France is")?;

// 4. Extract all logits (all positions enabled via logits_all=true)
cpp_session.context.eval(&tokens, 0)?;
let all_cpp_logits = cpp_session.context.get_all_logits(tokens.len())?;
// Result: Vec<Vec<f32>> where outer = positions, inner = vocab logits

// 5. Compare with Rust
let rust_logits_per_position = /* ... extract same way ... */;
let divergence = bitnet_crossval::logits_compare::compare_per_position_logits(
    &rust_logits_per_position,
    &all_cpp_logits,
);

// 6. Analyze divergence
if let Some(div_pos) = divergence.first_divergence_token {
    println!("Divergence at position {}", div_pos);
    println!("Cosine similarity: {}", divergence.per_token_cosine_sim[div_pos]);
}
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `crates/bitnet-sys/csrc/bitnet_c_shim.cc` | C++ wrapper to llama.cpp | ✅ Ready |
| `crates/bitnet-sys/src/wrapper.rs` | Rust safe wrappers | ✅ Ready |
| `crates/bitnet-inference/src/ffi_session.rs` | Global reusable FFI session | ✅ Ready |
| `crossval/src/logits_compare.rs` | Per-position comparison metrics | ✅ Ready |
| `crossval/tests/per_position_logits.rs` | Integration tests | ✅ Ready |
| `crossval/README_PER_POSITION_LOGITS.md` | Detailed API docs | ✅ Ready |

## Essential APIs

### C++ Session (Wrapper)
```rust
pub struct Session {
    pub model: Model,
    pub context: Context,
}

impl Session {
    pub fn load_deterministic(model_path: &str) -> Result<Self>
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>>
    pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) 
        -> Result<Vec<f32>>  // Last token only
}

pub struct Context {
    pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>>  // Position i
    pub fn get_all_logits(&self, n_tokens: usize) 
        -> Result<Vec<Vec<f32>>>  // All positions
}
```

### Logits Comparison
```rust
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence {
    pub struct LogitsDivergence {
        pub first_divergence_token: Option<usize>,
        pub per_token_cosine_sim: Vec<f32>,    // 1.0 = identical
        pub per_token_l2_dist: Vec<f32>,       // 0.0 = identical
        pub max_absolute_diff: f32,
    }
}

const COSINE_SIMILARITY_THRESHOLD: f32 = 1e-4;  // Divergence threshold
```

### FFI Session Management
```rust
#[cfg(feature = "ffi")]
pub struct ParityCppSession { ... }

#[cfg(feature = "ffi")]
pub fn parity_cpp_session(model_path: &str) 
    -> Result<&'static Mutex<ParityCppSession>>
```

## Common Patterns

### Pattern 1: Extract Last Token Logits
```rust
let mut session = Session::load_deterministic(model)?;
let tokens = session.tokenize(prompt)?;
let logits = session.eval_and_get_logits(&tokens, 0)?;  // Last token only
```

### Pattern 2: Extract All Position Logits
```rust
let mut session = Session::load_deterministic(model)?;
let tokens = session.tokenize(prompt)?;
session.context.eval(&tokens, 0)?;
let logits_all = session.context.get_all_logits(tokens.len())?;  // All positions
```

### Pattern 3: Compare Rust vs C++
```rust
let rs_logits = eval_logits_once(model, &tokens)?;
let mut cpp_session = Session::load_deterministic(model)?;
let cpp_logits = cpp_session.eval_and_get_logits(&tokens, 0)?;

let comparison = compare_per_position_logits(
    &vec![rs_logits],
    &vec![cpp_logits],
);

assert!(comparison.first_divergence_token.is_none(), "Parity failed");
assert!(comparison.per_token_cosine_sim[0] > 0.9999, "Low cosine similarity");
```

### Pattern 4: Track Multi-Token Divergence
```rust
let mut session = Session::load_deterministic(model)?;
let initial_tokens = session.tokenize(prompt)?;

let mut rust_all = Vec::new();
let mut cpp_all = Vec::new();
let mut tokens = initial_tokens.clone();

for step in 0..max_tokens {
    let cpp_logits = session.eval_and_get_logits(&tokens, 0)?;
    let rust_logits = eval_logits_once(model, &tokens)?;
    
    rust_all.push(rust_logits.clone());
    cpp_all.push(cpp_logits.clone());
    
    // Sample and continue
    let next_token = session.context.sample_greedy(&cpp_logits);
    tokens.push(next_token);
}

let divergence = compare_per_position_logits(&rust_all, &cpp_all);
println!("First divergence at token {}", divergence.first_divergence_token.unwrap_or(tokens.len()));
```

## Build Requirements

```bash
# To enable C++ FFI features
export BITNET_CPP_DIR=/path/to/bitnet.cpp  # Required for crossval feature
cargo test --features crossval --test per_position_logits

# Without C++ available (tests skip gracefully)
cargo test --features integration-tests
```

## Test Execution

```bash
# All parity tests (152+ passing)
cargo test --features crossval --test parity

# Per-position logits tests
cargo test --features crossval --test per_position_logits -- --nocapture

# With model path
export CROSSVAL_GGUF=/path/to/model.gguf
cargo test --features crossval --test per_position_logits -- --nocapture
```

## Performance Notes

### Current Bottleneck
- `get_logits_ith()` called N times (where N = token positions)
- Each call crosses FFI boundary to C++
- For 10-position sequence: 10 FFI calls

### Optional Optimization (Not Blocking)
- Add `bitnet_get_all_logits()` C++ function (30 lines)
- Reduces to 1 FFI call per evaluation
- Performance gain: ~10-30% for multi-token sequences
- **Status**: Recommended but not required for Sprint 1.3

## Debugging

### Check C++ Availability
```rust
if bitnet_sys::is_available() {
    println!("C++ backend available");
} else {
    println!("C++ backend not available (set BITNET_CPP_DIR)");
}
```

### Verify Determinism
```bash
# Set single-threaded environment
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
cargo test --features crossval test_name
```

### Check Logits Configuration
```rust
// Verify logits_all is enabled
assert!(cpp_session.context.logits_all);  // Should be true
```

## Known Issues & Workarounds

### Issue #469: Tokenizer Parity Mismatch
- C++ tokenization may differ slightly from Rust
- **Workaround**: Tests skip if tokenization differs
- **Impact**: Some cross-validation tests may skip

### Memory Corruption (RESOLVED)
- **Previous**: `munmap_chunk()` crashes
- **Solution**: Global reusable FFI session pattern
- **Status**: ✅ Fixed and tested

### GPU FFI Not Available
- **Current**: CPU-only for parity testing
- **Future**: GPU FFI extension planned
- **Workaround**: Use CPU feature for validation

## Next Steps for Sprint 1.3

### Week 1: Validation (Ready Now)
- [ ] Run existing parity tests with `--features crossval`
- [ ] Verify single-token logits cosine similarity > 0.9999
- [ ] Check multi-token generation tracking

### Week 2: Per-Position Extraction (Choose Path)
- [ ] **Path A (C++ Optimization)**: Add `bitnet_get_all_logits()` shim
  - Add C++ function (30 lines in csrc/bitnet_c_shim.cc)
  - Update header (5 lines in include/bitnet_c.h)
  - Generate Rust bindings (automatic via build.rs)
  - Add wrapper (20 lines in wrapper.rs)
  
- [ ] **Path B (Pure Rust)**: Extend parity.rs with per-position extraction
  - Add `eval_logits_per_position()` function
  - Implement incremental forward passes
  - Document performance notes

### Week 3: Testing & Documentation
- [ ] Run per_position_logits.rs integration tests
- [ ] Verify cosine similarity metrics
- [ ] Document any divergence findings
- [ ] Update CLAUDE.md with per-position test guidance

## References

- Full analysis: `docs/C_FFI_INTEGRATION_ANALYSIS.md`
- Per-position API: `crossval/README_PER_POSITION_LOGITS.md`
- Parity tests: `crossval/tests/parity.rs`
- C++ shim: `crates/bitnet-sys/csrc/bitnet_c_shim.cc`
- FFI session: `crates/bitnet-inference/src/ffi_session.rs`
