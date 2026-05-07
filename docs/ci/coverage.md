# Coverage

Coverage is execution-surface evidence. It answers one question precisely:

> **Did tests execute this Rust CPU code?**

## What coverage does not answer

Coverage explicitly does **not** answer:

- Whether tests would catch the wrong behavior (see `ripr`, property tests)
- Whether the inference engine produces correct output (see crossval, hardware validation)
- Whether GPU backends (Metal, Vulkan, OpenCL, ROCm, CUDA) are correct (see GPU scaffolding status in README)
- Whether model predictions are sound (see model validation in `docs/howto/`)
- Whether hardware acceleration is proven
- Whether cross-validation against C++ passes
- Whether mutation adequacy is strong
- Whether the code is fast enough

Those are separate proof lanes.

## Coverage in BitNet-rs

Coverage runs are **gated by label or main branch**:

- **PR runs:** only when explicitly labeled `coverage` or `full-ci`
- **Main runs:** automatic after every merge (cost: ~45 LEM, included in release validation)
- **Flag:** `rust-cpu` — CPU path execution surface only
- **Threshold policy:** currently informational; will ratchet after baseline collection

Coverage artifacts (`coverage.json`, `coverage.txt`, `lcov.info`) are stored on every run, enabling trend analysis and per-crate surface inspection.

## Codecov integration

Codecov integration is configured in `codecov.yml` with:

- **Project status:** tracks overall coverage %
- **Patch status:** tracks changes in PR diffs
- **Comments:** disabled — the GitHub check and Codecov dashboard are the primary signals
- **Flags:** scoped to `rust-cpu` for now; GPU flags deferred until backend validation is real

## Future: baseline and ratchet

After 10–20 runs with real project coverage, we will review:

1. Coverage % distribution across crate types
2. Lowest-covered core paths
3. Runtime cost and flake rate
4. Whether `--ignore-run-fail` is masking relevant failures

Then we will decide whether to tighten thresholds and move from informational to enforced status. Decisions will be based on observed data, not aspiration.
