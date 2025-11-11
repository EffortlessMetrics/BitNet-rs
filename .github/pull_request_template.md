## Summary

<!-- Brief description of what this PR accomplishes -->

## CI Requirements (check all that apply)

<!-- These are enforced by the Guards workflow - violations will block merge -->

- [ ] Actions are **SHA-pinned** (no @vN/@main/@stable/@latest)
- [ ] Workflow `cargo`/`cross` commands use **--locked**
- [ ] Toolchain respects **rust-toolchain.toml** (MSRV 1.89.0)
- [ ] Receipt workflow: builds exclude Python/WASM (`--exclude bitnet-py --exclude bitnet-wasm`) in CPU+GPU lanes (if touching `verify-receipts.yml`)
- [ ] **Guards** passes in CI (and locally if you run it)

<!-- optional local preflight -->
<!-- make guards  # or run scripts/guard equivalents locally -->

## Changes

<!-- List the main changes in this PR -->

- <!-- Add a bullet point for each meaningful change -->
-

## Testing

<!-- Describe how you tested these changes -->

- [ ] Tests pass locally with `cargo test --workspace --no-default-features --features cpu`
- [ ] Code formatted with `cargo fmt --all`
- [ ] Linting passes with `cargo clippy --all-targets --all-features -- -D warnings`

## CI Labels (opt-in heavy checks)

<!-- Only select labels for checks relevant to this PR -->
<!-- See docs/ci/labels.md for detailed label documentation -->

- [ ] `coverage` - Run code coverage analysis (heavy, only for core changes)
- [ ] `receipts` - Run CPU receipt verification gates (for inference/quantization changes)
- [ ] `framework` - Full integration test suite (for major architectural changes)
- [ ] `gpu` - GPU-specific tests (requires CUDA, only for GPU-related changes)
- [ ] `quant` - Quantization matrix testing (for quantization algorithm changes)
- [ ] `crossval` - Cross-validation determinism checks (for inference parity validation)
- [ ] `perf` - Performance regression gates (for performance-critical changes)
- [ ] `lut` - TL-LUT stress testing (for lookup table quantization changes)

## Documentation

- [ ] README updated (if user-facing changes)
- [ ] CLAUDE.md updated (if development workflow changes)
- [ ] API documentation updated (if public API changes)

## Checklist

- [ ] This PR is focused on a single concern
- [ ] Commit messages are clear and descriptive
- [ ] Breaking changes are documented
- [ ] All conversation threads resolved before merge
