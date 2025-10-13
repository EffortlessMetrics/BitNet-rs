# Implement opt-in cross-validation harness with C++ reference

**Labels:** `enhancement`, `testing`, `cross-validation`

**Priority:** Medium

**Depends on:** PR #452 (receipt verification)

## Summary

Create feature-gated cross-validation tests that compare BitNet.rs outputs with C++ reference implementation.

## Problem

Currently, cross-validation is manual and not integrated into the test suite. Need automated, opt-in tests that validate inference accuracy.

## Acceptance Criteria

- [ ] Feature-gated behind `#[cfg(feature = "crossval")]`
- [ ] Tests require `BITNET_GGUF` environment variable set to valid model path
- [ ] If `BITNET_GGUF` not set: tests print "SKIP: BITNET_GGUF not set" and pass
- [ ] If model present:
  - Run same prompts through Rust and C++ implementations
  - Assert correlation ≥ 0.995
  - Assert element-wise error bounds (e.g., max abs error < 0.01)
  - Compare token generation (first 10 tokens should match)
- [ ] Add to CI as optional job (only runs when model fixture available)
- [ ] Document in `docs/development/validation-framework.md`

## Implementation Notes

- Use existing `crossval` crate as foundation
- Tests in `tests/crossval/*.rs`
- Don't block CI on this - make it opt-in for contributors with models

## Test Example

```rust
#[test]
#[cfg(feature = "crossval")]
fn test_inference_matches_cpp_reference() {
    let model_path = match std::env::var("BITNET_GGUF") {
        Ok(path) => path,
        Err(_) => {
            println!("SKIP: BITNET_GGUF not set (cross-validation skipped)");
            return;
        }
    };

    let rust_output = run_rust_inference(&model_path, "Hello world");
    let cpp_output = run_cpp_reference(&model_path, "Hello world");

    let correlation = compute_correlation(&rust_output, &cpp_output);
    assert!(
        correlation >= 0.995,
        "Correlation too low: {} (expected ≥ 0.995)",
        correlation
    );

    let max_error = compute_max_abs_error(&rust_output, &cpp_output);
    assert!(
        max_error < 0.01,
        "Max abs error too high: {} (expected < 0.01)",
        max_error
    );
}
```

## Usage

```bash
# Download or provide model
export BITNET_GGUF=models/bitnet-2b.gguf

# Run cross-validation tests
cargo test --features crossval

# Skip cross-validation (tests pass without model)
cargo test
```

## CI Integration (Optional)

```yaml
test-crossval:
  runs-on: ubuntu-latest
  # Only run if we have the model fixture
  if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[crossval]')
  steps:
    - name: Download reference model
      run: |
        mkdir -p models
        # TODO: Add model download logic

    - name: Run cross-validation
      env:
        BITNET_GGUF: models/bitnet-2b.gguf
      run: cargo test --features crossval
```

## Files to Modify

- `tests/crossval/*.rs` - Add opt-in cross-validation tests
- `docs/development/validation-framework.md` - Document usage
- `.github/workflows/nightly-crossval.yml` - Optional CI job

## Related

- Depends on: PR #452 (receipt verification)
- Related: `docs/development/cross-validation-setup.md`

## Estimated Effort

~2 days
