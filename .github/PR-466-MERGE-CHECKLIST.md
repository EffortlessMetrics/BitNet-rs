# Merge Checklist — CPU MVP Finalization (PR-466)

## Pre-Merge Validation

### Code Quality
- [ ] Freshness: branch at HEAD of `main` (no conflicts)
- [ ] Format/Lint: `cargo fmt` / `cargo clippy -D warnings` (CPU lane) pass
- [ ] CPU Tests: `cargo test --workspace --no-default-features --features cpu`
- [ ] TL Correctness: `cargo test -p bitnet-kernels --test tl_packed_correctness`
- [ ] TL Stress: `cargo test -p bitnet-kernels --test fuzz_tl_lut_stress`

### Documentation
- [ ] Docs build: `cargo doc --workspace --no-default-features --features cpu --no-deps`
- [ ] Doctests pass: `cargo test --doc --workspace --no-default-features --features cpu`
- [ ] Link check pass (CI or manual)
- [ ] CLAUDE.md updated with receipt gate procedures

### Receipt Gate Validation (CPU)
- [ ] `xtask benchmark --tokens 128 --deterministic` → generates `ci/inference.json`
- [ ] `xtask verify-receipt --path ci/inference.json` → **PASS**
- [ ] Receipt fields validated:
  - [ ] `compute_path = "real"`
  - [ ] `backend = "cpu"`
  - [ ] `strict = true`
  - [ ] `deterministic = true`
  - [ ] `kernels` non-empty with CPU kernel IDs (i2s_*, tl*_*)
  - [ ] `tokens_per_second` present and reasonable (>0)
  - [ ] `rust_version` populated

### GPU Compatibility
- [ ] GPU builds: `cargo build --no-default-features --features gpu`
- [ ] GPU tests skip cleanly without CUDA: tests check `BITNET_ENABLE_GPU_TESTS`
- [ ] No GPU tests run in default CPU CI lane

### Repository Hygiene
- [ ] Artifacts: receipt attached or pinned baseline referenced in PR description
- [ ] Timeout policy: CI `timeout-minutes` / test TIMEOUTs aligned
- [ ] No leftover debug prints or temporary TODOs in critical paths
- [ ] Pre-commit hooks passing

## Post-Merge Tasks

### Branch Protection (AC5)
- [ ] GitHub Settings → Branches → `main`
- [ ] Add required status check: **Model Gates (CPU Receipt Verification)**

### Smoke Test (AC6)
- [ ] Create throwaway PR with mocked receipt (`compute_path = "mock"`)
- [ ] Verify CI fails with actionable error message
- [ ] Screenshot/document failure for tracking
- [ ] Close smoke test PR

### Release Preparation
- [ ] Tag created: `v0.1.0-mvp`
- [ ] Release notes drafted with:
  - [ ] Quickstart (10 lines: build → run → answer)
  - [ ] Receipt flow (run → emit → verify)
  - [ ] Link to pinned baseline: `docs/baselines/20251015-cpu.json`
  - [ ] Attached binaries: `bitnet`, `st2gguf`, `SHA256SUMS`
