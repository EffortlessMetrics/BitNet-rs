# Coverage

Coverage is execution-surface evidence. It answers one narrow question:

> Did tests execute this Rust CPU code?

## What Coverage Does Not Prove

Coverage does not prove:

- tests would catch the wrong behavior
- inference output is correct
- GPU backends are correct
- model predictions are sound
- hardware acceleration is proven
- cross-validation against C++ passes
- mutation adequacy is strong
- runtime performance is acceptable

Those are separate proof lanes.

## Coverage In BitNet-rs

Coverage runs are gated by branch, label, or manual dispatch:

- PR runs: only when labeled `coverage` or `full-ci`
- Main runs: automatic after every merge that touches coverage-relevant paths
- Manual runs: available through `workflow_dispatch`
- Flag: `rust-cpu`, covering CPU-path execution surface only
- Timeout: 60 minutes
- Threshold: 70% line coverage on pushes to `main`

The workflow lives in `.github/workflows/coverage.yml`. It runs as a job
container using the pinned Rust CI image, then runs `cargo llvm-cov nextest`
and emits JSON, LCOV, and text reports. The hosted runner is not cleaned before
coverage starts; the image keeps the large Rust coverage toolchain out of the
runner payload instead.

## Artifacts

The `coverage-report` artifact is uploaded on every run and contains:

- `coverage.json`
- `coverage.txt`
- `lcov.info`
- `target/bitnet/reports/coverage-receipt.json`

Artifacts are retained for 7 days.

## Codecov

Codecov integration is configured in `codecov.yml`:

- project and patch statuses are informational
- comments are disabled
- annotations are disabled
- the current flag is `rust-cpu`

Coverage uploads require `CODECOV_TOKEN`. Missing tokens skip upload instead
of failing unrelated CI.

## Ratchet Policy

Coverage thresholds should ratchet only after enough real runs establish a
stable baseline. Treat coverage as one signal in the verification ladder, not
as a substitute for correctness, cross-validation, hardware receipts, or model
quality evidence.
