## Summary

<!-- Brief description of what this PR accomplishes -->

## Lane ownership

Lane:
Campaign:
Work item:
Orchestrator:
Branch:
Base main SHA:
Allowed paths:

- <!-- path or none -->

Shared surfaces touched:

- <!-- shared surface or none -->

Closeout required:

## Repo boundary and merge type

Merge type:

- [ ] squash ordinary swarm PR
- [ ] regular merge source/sync/promotion PR

Repo boundary:

- [ ] swarm-only
- [ ] source-sync
- [ ] source-promotion
- [ ] source-only

Ancestry impact:

- [ ] no ancestry impact
- [ ] must preserve public source history
- [ ] must preserve swarm history into source

Boundary packet:

Promotion or sync packet path:
Source repo commit:
Swarm repo commit:
Included PRs:
Source impact:
Release/publish/signing impact:
Excluded work:
Machine clone or cutover impact:

## Source-of-truth links

Proposal:
Spec:
ADR:
Plan item:
Active goal or campaign manifest:

## Scope

- [ ] Proposal / why
- [ ] Spec / behavior contract
- [ ] ADR / durable decision
- [ ] Plan / sequencing
- [ ] Active goal / current execution state
- [ ] Runtime / implementation
- [ ] Policy ledger
- [ ] Support-tier update
- [ ] Generated status / receipt

## Non-goals

<!-- What this PR explicitly does not do -->

## Claim boundary

<!-- What may be claimed after this PR, and what may not be claimed yet -->

Model/hardware/proof claims added:
Claims explicitly not promoted:
Receipts or proof manifests:

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

## CI cost and verification discipline

<!--
BitNet-rs intentionally targets ordinary PR CI cost far below common
agentic-development defaults. The goal is not lighter testing; it is
stronger, better-scoped verification per CI minute.

Before requesting broad CI, prefer targeted Rust-native checks that match
the changed risk surface. Use the labels below only when the PR genuinely
needs the extra verification.

See: docs/ci/cost-and-verification-policy.md
-->

## CI Labels (opt-in heavy checks)

<!-- Only select labels for checks relevant to this PR -->
<!-- See docs/ci/labels.md for detailed label documentation -->
<!-- See docs/ci/cost-and-verification-policy.md for what these labels authorize -->

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

## Rollback

<!-- How to revert safely if needed -->

## Checklist

- [ ] This PR is focused on a single concern
- [ ] Commit messages are clear and descriptive
- [ ] Breaking changes are documented
- [ ] All conversation threads resolved before merge
