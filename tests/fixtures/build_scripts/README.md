# Build Script Test Fixtures (Issue #439 AC2)

## Purpose

Validate that `build.rs` scripts use **unified GPU feature detection** by checking both `CARGO_FEATURE_GPU` and `CARGO_FEATURE_CUDA` environment variables.

## Fixture Files

### Valid Patterns

1. **`valid_gpu_check.rs`**
   - ✓ Checks both `CARGO_FEATURE_GPU` and `CARGO_FEATURE_CUDA`
   - ✓ Uses logical OR: `gpu || cuda`
   - ✓ Emits `cargo:rustc-cfg=bitnet_build_gpu`
   - Usage: Reference implementation for Issue #439 AC2

2. **`valid_with_debug_output.rs`**
   - ✓ Same unified detection as `valid_gpu_check.rs`
   - ✓ Includes debug eprintln! statements for troubleshooting
   - Usage: Debugging GPU feature detection issues

### Invalid Patterns (Anti-Patterns)

3. **`invalid_cuda_only.rs`**
   - ✗ Only checks `CARGO_FEATURE_CUDA`
   - ✗ Ignores `CARGO_FEATURE_GPU`
   - Failure: `cargo build --features gpu` → GPU NOT compiled
   - Tests: AC2 validation should reject this pattern

4. **`invalid_gpu_only.rs`**
   - ✗ Only checks `CARGO_FEATURE_GPU`
   - ✗ Ignores `CARGO_FEATURE_CUDA` alias
   - Failure: `cargo build --features cuda` → GPU NOT compiled
   - Tests: AC2 validation should reject this pattern

## Testing Usage

### Load Valid Pattern
```rust
use std::fs;

let valid_build_rs = fs::read_to_string(
    "tests/fixtures/build_scripts/valid_gpu_check.rs"
)?;

assert!(valid_build_rs.contains("CARGO_FEATURE_GPU"));
assert!(valid_build_rs.contains("CARGO_FEATURE_CUDA"));
assert!(valid_build_rs.contains("||"));
```

### Validate Against Anti-Patterns
```rust
let invalid_build_rs = fs::read_to_string(
    "tests/fixtures/build_scripts/invalid_cuda_only.rs"
)?;

// Should fail AC2 validation
assert!(
    !(invalid_build_rs.contains("CARGO_FEATURE_GPU")
      && invalid_build_rs.contains("CARGO_FEATURE_CUDA")),
    "This pattern violates AC2"
);
```

## Integration with Tests

These fixtures are consumed by:
- `crates/bitnet-kernels/tests/build_script_validation.rs` (AC2 tests)
- `xtask/tests/preflight.rs` (Build configuration validation)

## Specification Reference

- **Issue**: #439 GPU feature-gate hardening
- **Acceptance Criteria**: AC2 - Build script parity
- **Specification**: `docs/explanation/issue-439-spec.md#ac2-build-script-parity`

## Expected Unified Pattern

```rust
let gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some()
       || std::env::var_os("CARGO_FEATURE_CUDA").is_some();

if gpu {
    println!("cargo:rustc-cfg=bitnet_build_gpu");
}
```

## Common Pitfalls

1. **Only checking one feature**: Breaks either `--features gpu` or `--features cuda`
2. **Using .unwrap()**: Build scripts should handle missing features gracefully
3. **Emitting multiple cfg flags**: Stick to one unified `bitnet_build_gpu` flag
4. **Forgetting rerun-if-changed**: Add `println!("cargo:rerun-if-changed=Cargo.toml");`
