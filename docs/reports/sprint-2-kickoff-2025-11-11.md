# Sprint-2 Kickoff - BitNet-rs MVP v0.1.0

**Date**: November 11, 2025
**Status**: Ready to Execute
**Milestone**: MVP v0.1.0 (due 2025-12-31)

---

## Executive Summary

Repository organization complete. Sprint-2 focuses on two parallel tracks:
- **Track A (P0)**: QK256 SIMD optimization (#417) - THE critical performance blocker
- **Track B (P1)**: MVP polish (#469) - Foundation for SIMD work

**Critical Path**: Complete #469 polish (5-7 days) → Launch #417 SIMD (2-4 weeks) → MVP release

---

## Sprint-2 Objectives

### Primary Goals (P0)

1. **QK256 SIMD Optimization (#417)** - ~0.1 → 3+ tok/s
   - Current: Scalar kernels only (~0.1 tok/s for 2B models)
   - Foundation: AVX2 dequant path merged (~1.2× observed)
   - Target: ≥3× improvement with nibble-LUT + FMA tiling
   - Effort: 2-4 weeks, 4 reviewable PRs

2. **MVP Polish (#469)** - 8 acceptance criteria
   - Prerequisites for SIMD work
   - Effort: 5-7 developer-days
   - Status: Ready for implementation

### Secondary Goals (P1)

3. **GGUF Mapping Bug (#393)** - Silent corruption fix
   - Impact: Q4/Q5/Q8 tensors wrongly mapped to I2S/TL types
   - Blocks: #346 (TL1), #401 (TL2)
   - Effort: 1-2 weeks

4. **KV Cache Memory Pool (#319)** - Production memory management
   - Current: Metadata-only allocation, cache bypasses pool
   - Impact: Memory leaks and fragmentation
   - Effort: 2-3 weeks

5. **CUDA Backend MVP (#450)** - GPU inference validation
   - Unblocked by: PR #475 (feature gate unification)
   - Blocks: #455 (GPU receipt gate), #317 (GPU forward), #414 (GPU crossval)
   - Effort: 2-3 weeks

6. **Documentation Audit (#459)** - Receipt-driven performance examples
   - Replace performance claims with measured receipts
   - Effort: 2-3 days

---

## Track A: QK256 SIMD Optimization (#417)

### Implementation Plan (4 PRs)

**PR 1: Dispatch + Microbench Harness** (Week 1, Days 1-3)
- CPU feature detection (`is_x86_feature_detected!`)
- Runtime dispatch scaffolding (`target_feature` annotations)
- Microbench harness for QK256 GEMV
- Baseline: Scalar kernel performance recorded

**Acceptance:**
- [ ] CPU feature detection works on AVX2/non-AVX2 machines
- [ ] Scalar baseline bench recorded in `docs/baselines/qk256-scalar-baseline.json`
- [ ] Dispatch compiles without errors on x86_64 and ARM
- [ ] `make guards` passes

**PR 2: Unpack Path** (Week 1-2, Days 4-7)
- Nibble LUT / bit-twiddle expand to `i8/f32`
- Cache-aware load patterns
- Unit tests for exact expand correctness (property-based)

**Acceptance:**
- [ ] Unpack path passes property-based tests (10K random inputs)
- [ ] Exact parity with scalar unpack (element-wise comparison)
- [ ] Benches show unpack overhead < 5% of total GEMV time
- [ ] `make guards` passes

**PR 3: AVX2 Kernel v1** (Week 2-3, Days 8-14)
- 8-wide FMA tiling (8–16 rows, interleaved accumulators)
- Integration with dispatch (runtime switch)
- Smoke parity test vs scalar (cosine ≥ .99999)

**Acceptance:**
- [ ] AVX2 kernel achieves ≥1.5× speedup over scalar (incremental target)
- [ ] Cosine similarity ≥ .99999 on seeded runs (1000 random shapes)
- [ ] No regressions on non-AVX2 platforms (scalar fallback works)
- [ ] Benchmark results recorded in `docs/baselines/qk256-avx2-v1.json`
- [ ] `make guards` passes

**PR 4: Integration + Scaling** (Week 3-4, Days 15-21)
- Row-parallel scheduling via Rayon
- `RAYON_NUM_THREADS` documentation
- `target-cpu=native` build instructions
- End-to-end throughput receipt

**Acceptance:**
- [ ] Full model inference reaches ≥3× throughput (2B model, CPU-only)
- [ ] Receipt written to `docs/baselines/<date>/qk256-avx2-production.json`
- [ ] README updated with build instructions (`RUSTFLAGS="-C target-cpu=native"`)
- [ ] Thread scaling documented (1, 2, 4, 8 threads benched)
- [ ] `make guards` passes

### Gate Criteria (Each PR)

All PRs must satisfy:
1. ✅ `make guards` green
2. ✅ Scalar parity maintained (cosine ≥ .99999)
3. ✅ Benchmarks updated and recorded
4. ✅ CI wall-time budget respected (<30min)
5. ✅ No regressions on ARM/non-AVX2 platforms

---

## Track B: MVP Polish (#469)

### Implementation Plan (8 ACs, 3 PRs)

**PR 1: Tolerance Single-Sourcing** (Days 1-2)
- AC1: Tolerance constant used in loader + constructor
- AC2: Update tests to use new tolerance constant

**Acceptance:**
- [ ] `BITNET_DEFAULT_TOLERANCE` constant defined in one location
- [ ] Loader uses tolerance from constant (no hardcoded values)
- [ ] Constructor uses tolerance from constant
- [ ] Tests updated to use tolerance constant
- [ ] `make guards` passes

**PR 2: K/V Cache Assertions** (Days 3-4)
- AC3: K/V post-slice assertions with `warn_once!`
- AC4: Update K/V cache tests

**Acceptance:**
- [ ] Post-slice assertions added to K/V cache operations
- [ ] `warn_once!` macro used for first-occurrence warnings
- [ ] Tests validate assertion triggers correctly
- [ ] No performance regression in K/V operations
- [ ] `make guards` passes

**PR 3: Parity Harness Hygiene** (Days 5-7)
- AC5: Parity timeout constant used consistently
- AC6: Receipt path rooted at workspace root
- AC7: Cross-validation tests updated
- AC8: Documentation updated

**Acceptance:**
- [ ] `PARITY_TEST_TIMEOUT_SECS` constant defined and used everywhere
- [ ] Receipt path logic rooted at workspace root (no `../` traversal)
- [ ] Cross-validation tests pass with new timeout/path logic
- [ ] `docs/development/validation-framework.md` updated
- [ ] `make guards` passes

---

## Milestone Health (MVP v0.1.0)

### Current Status

**Assigned Issues**: 10
**MVP Blockers**: 6
**Target Date**: 2025-12-31

| Issue | Title | Priority | Effort | Status |
|-------|-------|----------|--------|--------|
| #417 | QK256 SIMD optimization | P0 | 2-4 weeks | Ready |
| #469 | MVP polish (8 ACs) | P1 | 5-7 days | Ready |
| #393 | GGUF mapping bug | P0 | 1-2 weeks | Ready |
| #319 | KV cache memory pool | P0 | 2-3 weeks | Ready |
| #450 | CUDA backend MVP | P0 | 2-3 weeks | Unblocked |
| #459 | Doc audit (receipts) | P0 | 2-3 days | Ready |
| #472 | MVP Sprint 2 roadmap | - | - | Meta |
| #482 | Dependabot automation | - | 1 hour | PR #507 |
| #474 | Integer overflow (fuzz) | P1 | 1-2 days | Ready |
| #434 | CPU SIMD test hangs | P2 | Verify | Needs testing |

### Burndown Forecast

**Week 1** (Days 1-7):
- Complete #469 (MVP polish) - 7 days
- Launch #417 PR1 (Dispatch + benches) - 3 days

**Week 2** (Days 8-14):
- Complete #417 PR2 (Unpack path) - 4 days
- Launch #417 PR3 (AVX2 kernel) - 3 days
- Start #459 (Doc audit) - 2 days

**Week 3** (Days 15-21):
- Complete #417 PR3 (AVX2 kernel) - 4 days
- Complete #459 (Doc audit) - 1 day
- Launch #417 PR4 (Integration) - 2 days

**Week 4** (Days 22-28):
- Complete #417 PR4 (Integration) - 5 days
- Buffer for testing and refinement - 2 days

**Blockers for v0.2.0**:
- #393, #319, #450 pushed to v0.2.0 (production hardening)

---

## Repository State

### ✅ What's Locked In (Post-Cleanup)

1. **CI Guardrails** (PRs #486-#505)
   - SHA pins (40-hex with PCRE2)
   - MSRV enforcement (1.89.0, single-source)
   - `--locked` flags everywhere
   - Receipt excludes enforced
   - Runner pinning (ubuntu-22.04)
   - Template path triggers
   - `make guards` local preflight

2. **Issue Triage** (Nov 11, 2025)
   - 8 issues closed (resolved or duplicates)
   - 6 MVP blockers escalated to P0
   - 3 production blockers labeled for v0.2.0
   - Milestones created (MVP v0.1.0, v0.2.0)
   - Labels organized (mvp:blocker, mvp:polish, production-blocker, perf:simd, parity)

3. **Documentation** (Complete)
   - `docs/ci/guardrails.md` (comprehensive)
   - `docs/reports/comprehensive-issue-analysis-2025-11-11.md` (17KB)
   - `docs/reports/documentation-issues-analysis-2025-11-11.md` (35KB)
   - `tech_debt_analysis_343_420.md` (23KB)
   - `epic_templates.md` (23KB, 4 ready-to-create epics)
   - README QK256 quick-start section
   - README CI guardrails pointer (PR #508)

4. **Automation** (Ready)
   - Dependabot enabled (PR #507)
   - SHA repin bot (weekly automation)
   - Nightly guards (mirror PR checks)

### 📋 Pending Merges

**PR #507**: Dependabot configuration
- **Status**: All checks passing (2 SUCCESS, 3 SKIPPED)
- **Reviews**: 2 automated comments
- **Action**: Requires manual approval + squash merge

**PR #508**: README guardrails pointer
- **Status**: All checks passing (12 SUCCESS)
- **Reviews**: 2 automated comments
- **Action**: Requires manual approval + squash merge

**To merge**:
```bash
gh pr review 507 --approve && gh pr merge 507 --squash --delete-branch
gh pr review 508 --approve && gh pr merge 508 --squash --delete-branch
```

---

## Next Actions (This Week)

### Immediate (30 min)

1. **Merge PRs**
   ```bash
   gh pr review 507 --approve && gh pr merge 507 --squash --delete-branch
   gh pr review 508 --approve && gh pr merge 508 --squash --delete-branch
   ```

2. **Verify guards after merge**
   ```bash
   make guards
   gh workflow run "Guards (nightly)" --ref main
   ```

### This Week (5-7 days)

1. **Implement #469 (MVP polish)** - 3 PRs, 5-7 days
   - PR1: Tolerance single-sourcing (Days 1-2)
   - PR2: K/V cache assertions (Days 3-4)
   - PR3: Parity harness hygiene (Days 5-7)

2. **Start #417 (QK256 SIMD)** - PR1 (Dispatch + benches), 3 days
   - Scaffold dispatch infrastructure
   - Create microbench harness
   - Record scalar baseline

3. **Quick wins**
   - #474: Integer overflow fix (1-2 days)
   - #434: Verify EnvGuard fixed test hangs (30 min testing)

---

## Optional Enhancements

### Dependabot Grouping (Future PR)

To reduce PR noise, consider grouping related dependencies:

**File**: `.github/dependabot-enhanced.yml` (created)

**Groups added**:
- `tokio-ecosystem`: tokio*, bytes, hyper*, h2, mio
- `serde-ecosystem`: serde*
- `test-dependencies`: proptest*, criterion*, quickcheck*
- `github-actions`: actions/*, dtolnay/*, Swatinem/*, taiki-e/*

**Benefits**:
- 1 PR per group instead of N individual PRs
- Easier to review related updates together
- Lower CI cost (fewer workflow runs)

**To apply**:
```bash
mv .github/dependabot-enhanced.yml .github/dependabot.yml
git add .github/dependabot.yml
git commit -m "ci: group Dependabot updates to reduce PR noise"
git push
```

### Auto-Merge Policy (Future)

Once Dependabot runs for a few weeks and proves stable:
- Consider auto-merge for **patch** Cargo bumps only (leaf crates)
- Keep Actions manual (SHA pins already handle immutability)
- Requires enabling `enablePullRequestAutoMerge` in repo settings

---

## Watch List (Week 1)

### CI Health Monitors

1. **Receipt lane latency** (post-exclude changes)
   - Quantization matrix should skip bitnet-py/bitnet-wasm
   - Expected: <5min per job (down from 15min)

2. **Nightly guard summaries**
   - Watch for `ubuntu-latest` drift
   - Verify receipt exclude patterns work

3. **Dependabot PRs** (starting this week)
   - Confirm CI footprint acceptable
   - Review grouping effectiveness after 2-3 weeks

### Code Quality

1. **AVX2 kernel parity** (during #417 implementation)
   - Cosine similarity ≥ .99999 on all test shapes
   - No silent divergence on edge cases

2. **Memory safety** (#319 KV cache work)
   - Valgrind clean
   - No leaks in long-running inference

---

## Success Criteria (End of Sprint-2)

### MVP v0.1.0 Release Ready

**Performance**:
- [ ] QK256 throughput ≥3× baseline (2B model, CPU-only)
- [ ] Receipt recorded with measured TPS
- [ ] Benchmarks reproducible with `RUSTFLAGS="-C target-cpu=native"`

**Quality**:
- [ ] All 8 MVP polish ACs complete (#469)
- [ ] Parity tests passing (cosine ≥ .99999)
- [ ] `make guards` green
- [ ] Receipt verification gates passing

**Documentation**:
- [ ] Performance claims replaced with receipt examples (#459)
- [ ] QK256 SIMD build instructions in README
- [ ] Thread scaling documented

**Automation**:
- [ ] Dependabot running weekly
- [ ] Nightly guards monitoring CI drift
- [ ] SHA repin bot keeping actions immutable

---

## Sprint Retrospective (To Be Completed)

Track outcomes for Sprint-3 planning:
- Actual vs estimated effort
- Blockers encountered
- Process improvements
- CI/CD refinements

---

**Generated**: 2025-11-11
**Next Review**: End of Week 1 (2025-11-18)
**Milestone**: MVP v0.1.0 (2025-12-31)
