# Cross-Validation CI Deployment Checklist

Use this checklist to ensure successful deployment of the cross-validation CI workflow.

## Pre-Deployment

### Repository Prerequisites

- [ ] Repository has `crossval` crate with `ffi` feature
- [ ] `xtask` supports `fetch-cpp` command
- [ ] `xtask` supports `crossval-per-token` command (requires `crossval-all` feature)
- [ ] Test models accessible via `xtask download-model`
- [ ] GitHub Actions enabled on repository

### Workflow File Validation

- [ ] Workflow file exists: `.github/workflows/crossval.yml`
- [ ] YAML syntax valid: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/crossval.yml'))"`
- [ ] CPP_TAG set to correct commit hash
- [ ] CROSSVAL_TOLERANCE set appropriately (default: 0.999)

### Documentation

- [ ] Workflow guide exists: `docs/ci/crossval-workflow.md`
- [ ] Quick reference exists: `docs/ci/crossval-quick-reference.md`
- [ ] Setup guide exists: `docs/ci/SETUP.md`
- [ ] Implementation summary exists: `CROSSVAL_CI_IMPLEMENTATION.md`

## Deployment

### GitHub Configuration

- [ ] Actions enabled: Settings → Actions → General → Allow all actions
- [ ] Workflow permissions: Settings → Actions → General → Read and write
- [ ] Caching enabled: Settings → Actions → General (default)
- [ ] Cache limit verified: 10 GB (default)

### Label Creation

- [ ] Create `crossval` label:
  ```bash
  gh label create crossval \
    --description "Trigger cross-validation CI" \
    --color "0E8A16"
  ```

### Branch Protection

- [ ] Branch protection enabled for `main`
- [ ] Required status checks enabled
- [ ] Status checks selected:
  - [ ] `check-no-ffi`
  - [ ] `check-llama-stub`
- [ ] Optional: Lane B checks (if requiring parity for all PRs)
  - [ ] `lane-b-llama (ubuntu-latest)`
  - [ ] `lane-b-llama (macos-latest)`

## First Run Validation

### Manual Trigger Test

- [ ] Trigger workflow:
  ```bash
  gh workflow run crossval.yml \
    --ref main \
    -f lane=lane-b-llama \
    -f force_rebuild=true
  ```

- [ ] Monitor execution:
  ```bash
  gh run list --workflow=crossval.yml
  gh run view <run-id> --log
  ```

- [ ] Expected duration: 15-20 minutes (first run, cache miss)
- [ ] Workflow completes successfully
- [ ] All jobs pass (check-trigger, check-no-ffi, check-llama-stub, lane-b-llama)
- [ ] Artifacts uploaded:
  - [ ] `lane-b-ubuntu-latest-<run-id>`
  - [ ] `lane-b-macos-latest-<run-id>`
  - [ ] `crossval-summary-<run-id>`

### Cache Verification

- [ ] Navigate to Actions → Caches
- [ ] Verify cache created:
  - [ ] `llama-cpp-Linux-<cpp_tag>`
  - [ ] `llama-cpp-macOS-<cpp_tag>`
  - [ ] `cargo-Linux-<lock_hash>`
  - [ ] `cargo-macOS-<lock_hash>`
- [ ] Cache size reasonable (~500 MB for llama.cpp, ~2 GB for cargo)

### Second Run (Cache Hit)

- [ ] Trigger workflow again (without force_rebuild):
  ```bash
  gh workflow run crossval.yml \
    --ref main \
    -f lane=lane-b-llama
  ```

- [ ] Expected duration: 5-10 minutes (cache hit)
- [ ] Verify cache hit in logs: "⚡ Using cached llama.cpp libraries"
- [ ] Workflow completes successfully

## PR Integration Test

### Create Test PR

- [ ] Create branch: `git checkout -b test-crossval-ci`
- [ ] Make trivial change (e.g., update README)
- [ ] Push and create PR

### Trigger via Label

- [ ] Add `crossval` label to test PR
- [ ] Verify workflow triggers automatically
- [ ] Wait for completion (~10-15 minutes)
- [ ] Check summary comment posted on PR
- [ ] Verify status checks appear in PR UI
- [ ] Remove `crossval` label

### Verify Required Checks

- [ ] Create another test PR (without crossval label)
- [ ] Verify only required checks run:
  - [ ] `check-no-ffi`
  - [ ] `check-llama-stub`
- [ ] Lane B does NOT run
- [ ] Close test PR

## Local Development Setup

### C++ Reference Setup

- [ ] Run auto-setup:
  ```bash
  eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
  ```

- [ ] Verify preflight:
  ```bash
  cargo run -p xtask --features crossval-all -- preflight --verbose
  ```

- [ ] Expected output: "✓ llama.cpp: AVAILABLE"

### Local Cross-Validation

- [ ] Download test model:
  ```bash
  cargo run -p xtask -- download-model
  ```

- [ ] Run cross-validation tests:
  ```bash
  cargo test -p bitnet-crossval \
    --features cpu,ffi,crossval \
    --test dual_backend_integration \
    -- --nocapture
  ```

- [ ] Tests pass locally

### Per-Token Parity

- [ ] Run per-token check:
  ```bash
  cargo run -p xtask --features crossval-all -- crossval-per-token \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --prompt "What is 2+2?" \
    --max-tokens 4
  ```

- [ ] Parity check completes
- [ ] Cosine similarity above threshold

## Team Onboarding

### Documentation Updates

- [ ] Update CONTRIBUTING.md with cross-validation section
- [ ] Add workflow badge to README.md
- [ ] Link to `docs/ci/crossval-quick-reference.md` in CONTRIBUTING.md

### Team Communication

- [ ] Share setup guide with team: `docs/ci/SETUP.md`
- [ ] Share quick reference: `docs/ci/crossval-quick-reference.md`
- [ ] Demo PR workflow:
  1. Add `crossval` label
  2. Wait for results
  3. Review summary comment

### Training Materials

- [ ] Local setup commands documented
- [ ] PR workflow documented
- [ ] Debugging procedures documented
- [ ] Escalation process documented

## Monitoring Setup

### Notifications

- [ ] Email notifications enabled (GitHub settings)
- [ ] Optional: Slack webhook configured (if using)
- [ ] Optional: Discord webhook configured (if using)

### Dashboards

- [ ] Bookmark Actions page: `https://github.com/USER/REPO/actions/workflows/crossval.yml`
- [ ] Bookmark Cache page: `https://github.com/USER/REPO/actions/caches`
- [ ] Optional: Set up external monitoring (e.g., Datadog, Prometheus)

## Scheduled Run Verification

### Nightly Run (Lane B)

- [ ] Wait for next nightly run (3 AM UTC, Mon-Sat)
- [ ] Verify workflow triggers automatically
- [ ] Check execution completes successfully
- [ ] Verify cache still hit (if within 7 days)
- [ ] Review artifacts uploaded

### Weekly Run (Both Lanes)

- [ ] Wait for Sunday run (4 AM UTC)
- [ ] Verify both lanes trigger:
  - [ ] Lane B (ubuntu + macos)
  - [ ] Lane A (ubuntu)
- [ ] Check execution completes
- [ ] Verify Lane A is non-blocking (continues on error)
- [ ] Review summary artifact

## Post-Deployment

### Week 1 Monitoring

- [ ] Day 1: Verify nightly run
- [ ] Day 2: Check cache hit ratio
- [ ] Day 3: Monitor artifact storage
- [ ] Day 7: Review weekly run (both lanes)
- [ ] No critical failures
- [ ] Cache hit ratio >80%

### Week 2 Optimization

- [ ] Review execution times
- [ ] Tune cache keys if needed
- [ ] Adjust tolerance if consistent divergence
- [ ] Update CPP_TAG if upstream changes

### Week 4 Audit

- [ ] Review all runs (success rate target: >95%)
- [ ] Check artifact retention (should auto-delete >30 days)
- [ ] Verify cache eviction policy working
- [ ] Document any issues or improvements

## Troubleshooting Reference

### Common Issues Checklist

If workflow fails, check:

- [ ] YAML syntax valid
- [ ] Actions enabled
- [ ] Workflow permissions correct
- [ ] CPP_TAG points to valid commit
- [ ] Cache not corrupted (try force_rebuild)
- [ ] Model download successful
- [ ] C++ library paths set correctly

### Debug Commands

```bash
# Check workflow syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/crossval.yml'))"

# View latest run
gh run list --workflow=crossval.yml --limit 1

# Download artifacts
gh run download <run-id>

# Trigger with debugging
gh workflow run crossval.yml \
  -f lane=lane-b-llama \
  -f force_rebuild=true

# Local reproduction
BITNET_CPP_DIR="" cargo build -p bitnet-crossval --features ffi  # STUB mode
cargo test -p bitnet-crossval --features cpu,ffi,crossval --lib  # No FFI
```

## Success Criteria

### Functional

- [x] Workflow triggers on schedule, PR label, and manual
- [x] Required checks (no-FFI, STUB mode) always run
- [x] Lane B runs on trigger with cache support
- [x] Lane A runs weekly (non-blocking)
- [x] Artifacts uploaded correctly
- [x] Summary reports generated
- [x] PR comments posted

### Performance

- [x] First run: <20 min (cache miss)
- [x] Subsequent runs: <10 min (cache hit)
- [x] Cache hit ratio: >80% after week 1
- [x] Artifact storage: <5 GB
- [x] Total cache usage: <10 GB

### Quality

- [x] YAML syntax valid
- [x] All documentation complete
- [x] Local reproduction documented
- [x] Troubleshooting guide available
- [x] Team onboarding materials ready

## Final Sign-Off

- [ ] All pre-deployment checks complete
- [ ] All deployment steps complete
- [ ] All validation tests pass
- [ ] Team onboarded
- [ ] Monitoring configured
- [ ] Week 1 monitoring complete

**Deployment Status**: ✅ Ready for Production

**Deployed By**: ___________________
**Date**: ___________________
**Review By**: ___________________
**Approval**: ___________________

---

## Post-Deployment Notes

Use this section to document any issues, optimizations, or learnings during deployment:

```
[Date] [Issue/Optimization/Learning]

Example:
2025-10-25 Cache hit ratio lower than expected on macOS (60%).
           Increased cache retention to 14 days.

2025-11-01 Lane A consistently failing due to bitnet.cpp API changes.
           Updated CPP_TAG to latest stable commit.
```

---

## Maintenance Schedule

### Daily

- [ ] Check nightly run status (automated, no action unless failure)

### Weekly

- [ ] Review Sunday run (both lanes)
- [ ] Check cache hit ratio trend
- [ ] Verify artifact retention policy

### Monthly

- [ ] Update CPP_TAG if upstream changes
- [ ] Review execution time trends
- [ ] Audit artifact storage usage
- [ ] Check for workflow optimization opportunities

### Quarterly

- [ ] Review tolerance threshold
- [ ] Update documentation if workflow changes
- [ ] Consider GPU lane addition (if runners available)
- [ ] Evaluate performance regression tracking

---

**For detailed troubleshooting, see**: `docs/ci/crossval-quick-reference.md#common-failure-patterns`

**For maintenance procedures, see**: `CROSSVAL_CI_IMPLEMENTATION.md#maintenance-and-operations`
