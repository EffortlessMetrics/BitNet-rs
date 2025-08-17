# Cross-Validation Status Report

## Completed Work

### 1. UTF-8 GGUF Parsing Fix ✅
- **Issue**: GGUF reader was failing on non-UTF-8 strings (common in token pieces)
- **Solution**: Implemented lossy UTF-8 decoding in `read_string()` function
- **Status**: Fixed and tested - warning shows but doesn't crash
- **File**: `crates/bitnet-models/src/formats/gguf/types.rs`

### 2. Improved Benchmarking Script ✅
- **Created**: `scripts/crossval_bench.sh` with:
  - Warm-up passes to stabilize timings
  - Configurable thread matrix via `THREADS` env var
  - Strict mode (`STRICT=1`) to fail on mock/disabled paths
  - Robust parsing of tokens/sec from C++ output
  - CSV output for tracking results over time

### 3. CLI Updated to Use Real Loader ✅
- **Changed**: `bitnet-cli` now uses `ModelLoader` instead of mock `gguf_simple`
- **File**: `crates/bitnet-cli/src/main.rs`

## Current Status

### C++ Implementation (llama.cpp)
- **Working**: ✅ Fully functional
- **Performance**: 
  - 1 thread: ~13-15 tokens/sec
  - 8 threads: ~17-21 tokens/sec
  - Model loads correctly, inference produces coherent text

### Rust Implementation (BitNet-rs)
- **GGUF Loading**: ⚠️ Partially working
  - UTF-8 issue fixed (lossy decode implemented)
  - Still has array bounds issue during metadata parsing
  - Error: "String extends beyond data bounds" during array parsing
- **Inference**: Using mock tensors when GGUF fails to load
- **Performance**: N/A (mock path is not meaningful)

## Next Steps

1. **Fix GGUF Array Parsing**
   - Issue is in `GgufValue::read()` array handling
   - The temporary buffer offset calculation is incorrect
   - Need to fix the array element parsing logic

2. **Complete Model Loading Pipeline**
   - Once GGUF loads, need proper tensor mapping
   - Tokenizer integration (currently using mock)

3. **Enable Real Benchmarks**
   - Once model loads, can compare real performance
   - Use `STRICT=1` mode to ensure no mock paths

## Benchmark Results

| Implementation | Tokens | Threads | Tokens/sec | Status |
|----------------|--------|---------|------------|--------|
| C++ (llama.cpp) | 32 | 1 | 13.19 | OK |
| C++ (llama.cpp) | 64 | 1 | 15.00 | OK |
| C++ (llama.cpp) | 64 | 8 | 17.77 | OK |
| Rust (BitNet-rs) | 64 | auto | N/A | GGUF_LOAD_FAIL |

## Files Modified

1. `crates/bitnet-models/src/formats/gguf/types.rs` - UTF-8 fix
2. `crates/bitnet-cli/src/main.rs` - Use real loader
3. `scripts/crossval_bench.sh` - New robust benchmark script
4. `crossval_results.csv` - Benchmark results tracking