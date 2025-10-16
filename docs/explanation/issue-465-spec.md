# Issue #465: CPU Path Followup

## Context

After PR #464 merged CPU forward pass implementation with TL LUT helper and receipt validation, the BitNet.rs project needs final polish for v0.1.0-mvp release. The core CPU inference pipeline is complete and validated, but gaps remain in documentation, baseline establishment, and CI enforcement of honest compute receipts.

**Current State (post-#464):**
- CPU forward pass, KV cache, CLI priming/greedy decode implemented and tested
- TL LUT helper with bounds/overflow protection landed
- Receipt honesty enforcement with CPU quantized prefix matchers operational
- All integrative gates green through merge

**Remaining Gaps:**
- README lacks quickstart flow for deterministic inference + receipt verification
- No pinned CPU baseline receipt for deterministic comparison
- Branch protection not enforcing Model Gates (CPU) as required status check
- PR #435 (mock-elimination & baselines) ready to merge but pending
- Documentation drift with legacy performance claims and inconsistent feature flags

**Affected Components:**
- Documentation: README, quickstart guides, performance claims
- CI/CD: Branch protection rules, receipt verification gates
- Baselines: Deterministic CPU baseline for yardstick comparison
- Testing: Receipt validation workflow integration

## User Story

As a BitNet.rs maintainer preparing for v0.1.0-mvp release, I want comprehensive post-merge polish including documentation updates, baseline establishment, and CI gate enforcement so that users have clear quickstart paths with deterministic inference verification and the repository enforces honest compute receipts automatically.

## Acceptance Criteria

AC1: README contains 10-line CPU quickstart block with build → deterministic run → answer flow that renders output and verified receipt via copy-paste

AC2: README contains receipts documentation block mirroring xtask commands and environment variables (BITNET_STRICT_MODE, BITNET_DETERMINISTIC, RAYON_NUM_THREADS)

AC3: Pinned CPU baseline receipt exists in `docs/baselines/YYYYMMDD-cpu.json` with `compute_path: "real"` and CPU kernel IDs (`i2s_*`, `tl*_*`) generated via deterministic benchmark run

AC4: CPU baseline receipt verification passes with `cargo run -p xtask -- verify-receipt` command against pinned JSON file

AC5: GitHub branch protection rules enforce Model Gates (CPU) as required status check that blocks PRs with mocked/empty receipts

AC6: Smoke test with mocked/empty receipt demonstrates branch protection blocking behavior

AC7: PR #435 (mock-elimination & baselines) merged with Verification section mentioning `verify-receipt` workflow

AC8: Mock-inference tracking issue closed after #435 merge

AC9: Documentation standardizes all cargo commands to use `--no-default-features --features cpu|gpu` pattern across README and guides

AC10: Documentation removes all legacy performance claims (e.g., "200 tok/s CPU") and replaces with receipt-driven evidence references

AC11: Pre-tag verification passes all quality gates (clippy, tests, deterministic benchmark, receipt verification) with strict mode enabled

AC12: v0.1.0-mvp tag created with bitnet and st2gguf binaries linked to pinned baseline receipt

## Technical Implementation Notes

**Affected Crates:**
- Documentation: README.md, docs/quickstart.md, docs/getting-started.md
- CI/CD: .github/workflows/, branch protection settings
- Baselines: docs/baselines/ directory structure
- Testing: xtask benchmark and verify-receipt commands

**Pipeline Stages:**
- Documentation: Quickstart flow updates, receipt verification examples
- Baseline Establishment: Deterministic CPU benchmark → pinned JSON receipt
- CI Enforcement: Branch protection rules for Model Gates (CPU)
- Quality Assurance: Pre-release verification checklist

**Performance Considerations:**
- Deterministic inference with `BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1`
- CPU baseline receipt establishes performance yardstick with real kernel IDs
- Receipt verification ensures honest compute (no mock inference) with `compute_path: "real"`

**Feature Flags:**
- All commands must use `--no-default-features --features cpu` for CPU operations
- Documentation must standardize feature flag usage across examples

**Testing Strategy:**
- TDD approach with `// AC:ID` tags mapping to acceptance criteria
- Deterministic benchmark runs: `cargo run -p xtask -- benchmark --model tests/models/tiny.gguf --tokens 128 --deterministic`
- Receipt verification: `cargo run -p xtask -- verify-receipt ci/inference.json`
- Pre-tag verification checklist:
  ```bash
  cargo clippy --workspace --all-targets --no-default-features --features cpu -D warnings
  cargo test --workspace --no-default-features --features cpu
  export BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1
  cargo run -p xtask -- benchmark --model tests/models/tiny.gguf --tokens 128 --deterministic
  cargo run -p xtask -- verify-receipt ci/inference.json
  ```

**Documentation Requirements:**
- README quickstart: 10-line CPU flow (build → run → verify)
- README receipts: xtask commands + environment variables reference
- Standardize feature flags: `--no-default-features --features cpu|gpu`
- Remove legacy claims: grep for "200 tok/s", replace with receipt evidence
- Cross-reference: Link baseline receipts to quickstart examples

**CI/CD Requirements:**
- Branch protection: Require Model Gates (CPU) status check
- Smoke test: Verify mocked receipts are blocked (negative test case)
- Merge #435: Review Verification section, ensure `verify-receipt` workflow documented

**Baseline Requirements:**
- Deterministic CPU baseline: `docs/baselines/YYYYMMDD-cpu.json`
- Receipt schema: v1.0.0 with `compute_path: "real"` + CPU kernel IDs
- Verification: Passes `cargo run -p xtask -- verify-receipt` validation
- Fingerprint: Includes model hash, token count, kernel IDs for reproducibility

**Release Checklist (v0.1.0-mvp):**
1. All ACs validated with evidence
2. Pre-tag verification passes (clippy, tests, deterministic benchmark, receipt verification)
3. Baseline receipt pinned and linked from README
4. Branch protection enforcing Model Gates (CPU)
5. #435 merged and mock-inference issue closed
6. Documentation standardized (feature flags, receipt-driven claims)
7. Tag created with binaries and baseline reference

**Optional (Not Blocking MVP):**
- Cross-validation quick lane: `cargo run -p xtask -- crossval --quick`
- GPU fingerprint allowlist: `.ci/fingerprints.yml` for fast GPUs (50-100 tok/s envelope)

## Dependencies

- PR #435: Mock-elimination & baselines (ready to merge, blocks AC7/AC8)
- Deterministic test model: `tests/models/tiny.gguf` (required for baseline generation)
- xtask commands: `benchmark`, `verify-receipt` (already implemented in #464)

## Success Metrics

- Copy-paste quickstart works end-to-end (build → inference → verification)
- Pinned baseline receipt verifies successfully with strict mode
- Branch protection blocks mocked receipts (smoke test validation)
- Documentation standardized (no legacy claims, consistent feature flags)
- v0.1.0-mvp tag released with linked baseline evidence
