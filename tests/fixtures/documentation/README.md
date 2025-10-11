# Documentation Example Fixtures (Issue #439 AC7)

## Purpose

Validate that documentation uses **standardized feature flag examples** and avoids standalone `cuda` feature references without proper context.

## Fixture Files

### Valid Documentation Patterns

1. **`valid_feature_flags.md`**
   - ✓ Uses `--no-default-features --features cpu|gpu` pattern
   - ✓ Mentions `cuda` is temporary alias for `gpu`
   - ✓ Comprehensive feature flag reference
   - ✓ Includes common pitfalls section
   - Usage: Reference documentation template for Issue #439

2. **`valid_comprehensive_guide.md`**
   - ✓ Complete feature flag guide with philosophy section
   - ✓ Architecture-specific details (AVX2, NEON, CUDA)
   - ✓ Feature composition examples
   - ✓ Troubleshooting section
   - Usage: Comprehensive documentation reference

### Invalid Documentation Patterns (Anti-Patterns)

3. **`invalid_cuda_examples.md`**
   - ✗ Uses `--features cuda` without alias context
   - ✗ Missing `--no-default-features` flag
   - ✗ No migration guidance
   - Tests: AC7 validation should flag these patterns

4. **`invalid_bare_features.md`**
   - ✗ Uses `--features cpu` without `--no-default-features`
   - ✗ No explanation of empty default features
   - ✗ Inconsistent with BitNet.rs standards
   - Tests: AC7 validation should detect bare features

## Testing Usage

### Search for Valid Patterns
```bash
# Find examples with --no-default-features
rg --no-default-features tests/fixtures/documentation/

# Verify cuda alias context
rg "cuda.*alias|alias.*cuda" tests/fixtures/documentation/
```

### Detect Invalid Patterns
```bash
# Find bare --features usage
rg "\\-\\-features\\s+(cpu|gpu)" tests/fixtures/documentation/ | \
  grep -v "no-default-features"

# Find standalone cuda without context
rg "\\-\\-features\\s+cuda" tests/fixtures/documentation/ | \
  grep -v "alias"
```

## Integration with Tests

These fixtures are consumed by:
- `xtask/tests/documentation_audit.rs` (AC7 tests)
- Workspace-wide documentation consistency checks

## Specification Reference

- **Issue**: #439 GPU feature-gate hardening
- **Acceptance Criteria**: AC7 - Documentation updates
- **Specification**: `docs/explanation/issue-439-spec.md#ac7-documentation-updates`

## Expected Documentation Patterns

### Build Commands
```markdown
# CPU build
cargo build --no-default-features --features cpu

# GPU build (unified flag)
cargo build --no-default-features --features gpu
```

### Feature Alias Note
```markdown
**Note**: `cuda` is a temporary alias for `gpu` (will be removed in future versions).
Prefer using `--features gpu`.
```

### Test Commands
```markdown
cargo test --workspace --no-default-features --features cpu
```

### Quality Commands
```markdown
cargo clippy --all-targets --no-default-features --features cpu -- -D warnings
```

## Common Documentation Mistakes

1. **Bare features flag**: `cargo build --features cpu` (missing `--no-default-features`)
2. **Standalone cuda**: `cargo build --features cuda` (no alias context)
3. **Missing workspace flag**: `cargo test --features cpu` (should be `--workspace`)
4. **No migration path**: Using `cuda` without noting it's deprecated

## Validation Checklist

Documentation should:
- [ ] Always use `--no-default-features` with feature flags
- [ ] Prefer `--features gpu` over `--features cuda`
- [ ] Mention `cuda` is temporary alias if used
- [ ] Include `--workspace` for test commands
- [ ] Explain empty default features philosophy
- [ ] Provide troubleshooting for common issues

## Example Validation Test

```rust
use std::fs;

#[test]
fn validate_documentation_standards() {
    let doc = fs::read_to_string(
        "tests/fixtures/documentation/valid_feature_flags.md"
    ).unwrap();

    // Check for standardized patterns
    assert!(doc.contains("--no-default-features"));
    assert!(doc.contains("--features cpu"));
    assert!(doc.contains("--features gpu"));

    // Check for cuda alias context
    if doc.contains("cuda") {
        assert!(
            doc.contains("alias") || doc.contains("temporary"),
            "cuda feature should be documented as temporary alias"
        );
    }
}
```
