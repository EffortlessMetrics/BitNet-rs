# Feature Gate Pattern Fixtures (Issue #439 AC1)

## Purpose

Validate that GPU code uses **unified feature predicates** `#[cfg(any(feature="gpu", feature="cuda"))]` rather than standalone `#[cfg(feature="cuda")]` to prevent compile-time drift.

## Fixture Files

### Valid Patterns

1. **`valid_unified_predicate.rs`**
   - ✓ Uses `#[cfg(any(feature = "gpu", feature = "cuda"))]`
   - ✓ Runtime check: `cfg!(any(feature = "gpu", feature = "cuda"))`
   - ✓ Includes CPU fallback with inverted predicate
   - ✓ Works with both `--features gpu` AND `--features cuda`
   - Usage: Reference implementation for AC1

2. **`valid_nested_predicates.rs`**
   - ✓ Complex nested predicates (e.g., GPU + mixed_precision)
   - ✓ All GPU checks use unified predicate
   - ✓ Demonstrates proper feature composition
   - Usage: Advanced feature gate patterns

### Invalid Patterns (Anti-Patterns)

3. **`invalid_standalone_cuda.rs`**
   - ✗ Uses `#[cfg(feature = "cuda")]` without `any()`
   - ✗ Ignores `feature = "gpu"`
   - Failure: `cargo build --features gpu` → GPU NOT compiled
   - Tests: AC1 validation should reject this pattern

4. **`invalid_standalone_gpu.rs`**
   - ✗ Uses `#[cfg(feature = "gpu")]` without checking `cuda` alias
   - ✗ Breaks backward compatibility
   - Failure: `cargo build --features cuda` → GPU NOT compiled
   - Tests: AC1 validation should reject this pattern

## Testing Usage

### Load Valid Pattern
```rust
use std::fs;

let valid_code = fs::read_to_string(
    "tests/fixtures/code_patterns/valid_unified_predicate.rs"
)?;

assert!(valid_code.contains(r#"any(feature = "gpu", feature = "cuda")"#));
```

### Detect Anti-Patterns
```rust
use std::process::Command;

// Search for standalone cuda gates
let output = Command::new("rg")
    .args(&[r#"#\[cfg\(feature\s*=\s*"cuda"\)\]"#])
    .arg("invalid_standalone_cuda.rs")
    .output()?;

// Should find violations (for testing validation logic)
assert!(!String::from_utf8_lossy(&output.stdout).is_empty());
```

## Integration with Tests

These fixtures are consumed by:
- `crates/bitnet-kernels/tests/feature_gate_consistency.rs` (AC1 tests)
- Workspace-wide consistency validation

## Specification Reference

- **Issue**: #439 GPU feature-gate hardening
- **Acceptance Criteria**: AC1 - Kernel gate unification
- **Specification**: `docs/explanation/issue-439-spec.md#ac1-kernel-gate-unification`

## Expected Unified Pattern

### Module-Level Gate
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu_module {
    // GPU-specific code
}
```

### CPU Fallback
```rust
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
pub mod gpu_module {
    // Stubs or panic!()
}
```

### Runtime Check
```rust
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}
```

### Nested Predicates
```rust
#[cfg(all(
    any(feature = "gpu", feature = "cuda"),
    feature = "mixed_precision"
))]
pub mod mixed_precision_gpu {
    // Requires BOTH GPU and mixed_precision
}
```

## Common Pitfalls

1. **Standalone cuda gate**: Only checks `feature = "cuda"` → Breaks `--features gpu`
2. **Standalone gpu gate**: Only checks `feature = "gpu"` → Breaks `--features cuda`
3. **Inconsistent runtime checks**: Using `cfg!(feature = "cuda")` instead of unified check
4. **Missing CPU fallback**: Not providing `#[cfg(not(any(...)))]` alternative

## Validation Commands

```bash
# Search for standalone cuda gates (should find none in production code)
rg '#\[cfg\(feature\s*=\s*"cuda"\)\]' crates/ --glob '*.rs'

# Search for standalone cfg! runtime checks
rg 'cfg!\(feature\s*=\s*"cuda"\)' crates/ --glob '*.rs'

# Verify unified predicates exist
rg 'any\(feature\s*=\s*"gpu",\s*feature\s*=\s*"cuda"\)' crates/ --glob '*.rs'
```
