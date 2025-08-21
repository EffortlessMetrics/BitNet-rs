# Cross-Validation System Improvements

## Summary
Production-grade improvements to the BitNet-rs cross-validation system that make C++ compatibility testing robust and predictable across all environments.

## Key Improvements Implemented

### 1. Real GGUF Loader Integration
- Uses actual `GgufReader` from `bitnet-models` for validation
- No more "looks like GGUF" - now validates actual parsing capability
- Provides detailed error messages when parsing fails

### 2. Bulletproof GGUF Test Fixtures
- Always generates GGUF v3 format (224 bytes with 4 KV pairs)
- Backpatching ensures n_kv and data_offset stay consistent
- When `--version 2` requested, emits v3 with `compat.v2_requested=true` tag
- Files are 32-byte aligned with data_offset == file_size for 0-tensor files
- Contains metadata: general.architecture, general.name, general.file_type, compat.v2_requested

### 3. C++ Header Preflight Check
- Runs `llama-gguf -l` before full model load
- Early detection of GGUF incompatibilities
- Separates format issues from runtime/backend problems
- Enables clean XFAIL for known C++ limitations

### 4. JSON Cross-Validation Reports
- Machine-readable results saved to `target/crossval_report.json`
- Tracks: rust_ok, cpp_header_ok, cpp_full_ok, xfail status
- Includes timestamps, platform info, and detailed notes
- Perfect for CI artifact uploads and automated analysis

### 5. Cross-Platform Environment Setup
- Automatic library path configuration:
  - Linux: LD_LIBRARY_PATH
  - macOS: DYLD_LIBRARY_PATH  
  - Windows: PATH
- Static C++ builds to avoid runtime library issues
- Deterministic test environment (single-threaded, fixed seeds)

### 6. Enhanced Error Detection
- 9 different C++ failure patterns recognized
- Case-insensitive matching for robustness
- Covers GGUF errors, tensor errors, assertions
- Soft-fail mechanism with `CROSSVAL_ALLOW_CPP_FAIL=1`

### 7. Strong Test Invariants  
- Validates mini GGUF fixtures parse correctly
- Verifies alignment: `file_size % 32 == 0`
- Verifies 0-tensor files: `data_offset == file_size`
- Verifies metadata presence and correctness
- Ensures v3 always emitted (even when v2 requested)

## Usage Examples

### Generate Test Fixtures
```bash
# Generate mini GGUF files for testing
cargo xtask gen-mini-gguf --version 3 --output target/test.gguf
```

### Run Cross-Validation
```bash
# With soft-fail for C++ (CI-friendly)
export CROSSVAL_ALLOW_CPP_FAIL=1
cargo xtask crossval --model target/test.gguf

# Check the report
cat target/crossval_report.json
```

### Static C++ Build
```bash
# Fetch and build C++ with static linking
cargo xtask fetch-cpp --force
```

### Full Workflow
```bash
# Download model, build C++, run tests
cargo xtask full-crossval
```

## CI Integration

The system is designed for CI with:
- Soft-fail mechanism keeps CI green when C++ has known issues
- JSON reports for artifact uploads
- Mini fixtures for fast testing without downloads
- Deterministic execution for reproducible results

## Files Modified

1. **xtask/src/main.rs**: Complete cross-validation improvements
2. **xtask/Cargo.toml**: Added bitnet-models dependency
3. **xtask/tests/mini_gguf.rs**: Snapshot tests for fixtures
4. **scripts/build_cpp_static.sh**: Static build helper
5. **docs/CPP_CROSSVAL_GUIDE.md**: Comprehensive documentation

## Known Limitations

- C++ may fail on certain GGUF variants that Rust handles
- Always generates v3 format (v2 support via metadata tag only)
- Mini fixtures are 0-tensor files for header validation only

## Next Steps

To improve further:
1. Fix GgufReader to properly handle v2 string lengths
2. Add metadata-only comparison mode (no tensor loading)
3. Create C++ inspect-to-JSON helper for deeper comparison
4. Add CI matrix testing across Linux/macOS/Windows