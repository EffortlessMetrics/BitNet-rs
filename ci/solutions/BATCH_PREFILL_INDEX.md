# Batch Prefill Performance Test Quarantine - Document Index

**Navigation:** [ci/](../) → [solutions/](./00_NAVIGATION_INDEX.md) → This Document
**Related:** [PR #475 Summary](../PR_475_FINAL_SUMMARY.md)

---

## Quick Navigation

### Primary Documents (Start Here)

1. **IMPLEMENTATION_SUMMARY.md** (181 lines)
   - Overview of findings
   - Root cause summary (5 causes)
   - Checklist for implementation
   - Quick reference for commands and baselines
   - **START HERE** if you want the executive summary

2. **batch_prefill_perf_quarantine.md** (741 lines)
   - Complete detailed analysis
   - Full root cause investigation
   - Complete quarantine code pattern
   - Alternative approaches with rationale
   - CI workflow recommendations
   - Verification checklist
   - **READ THIS** for complete implementation guide

## Document Structure

### batch_prefill_perf_quarantine.md Sections

| Section | Content | Purpose |
|---------|---------|---------|
| 1 | Performance Test Analysis | Identify assertions causing flakiness |
| 2 | Root Cause Analysis | 5 primary causes + why current quarantine insufficient |
| 3 | Quarantine Pattern Implementation | Complete code with detailed docstrings |
| 4 | Alternative Approaches | 5 alternatives evaluated, pros/cons |
| 5 | CI Workflow Recommendations | Nightly job setup, local development, functional split |
| 6 | Verification Checklist | 16 items to validate before deployment |
| 7 | Implementation Summary | Current state → target state |
| 8 | References | Documentation, issues, external links |
| 9 | Appendix | System load detection code samples |

## Key Findings at a Glance

### Problem
Test at `crates/bitnet-inference/tests/batch_prefill.rs::test_batch_prefill_performance_consistency` (lines 219-269) has non-deterministic CI failures.

### Root Causes (5)
1. **Timer Resolution Variance** - 0.5ms threshold near system minimum
2. **Scheduler Jitter** - 50+ ms preemption on shared CI CPU
3. **Async Overhead** - tokio executor context switching
4. **Parallel Interference** - Other tests consume CPU
5. **Mock Realism Gap** - Fixed delay doesn't scale with input

### Solution Overview
- Primary: `#[ignore]` + relax assertions + load detection
- Secondary: Split into functional (always) + performance (nightly) tests
- CI: New nightly job + unchanged standard CI

## Implementation Steps

### Phase 1: Test Code (30 minutes)
- Replace lines 249-269 in `batch_prefill.rs`
- Change hard assertions to informational warnings
- Relax thresholds from (0.5ms, 8-100ms) to ±200%
- Add system load detection code
- Add comprehensive docstring

### Phase 2: CI Configuration (30 minutes)
- Create `.github/workflows/testing-framework-performance.yml`
- Set environment: `RUN_PERF_TESTS=1`, `RAYON_NUM_THREADS=1`
- Configure single-threaded execution
- Archive JUnit metrics

### Phase 3: Documentation (1 hour)
- Update `docs/development/test-suite.md`
- Add reference to CLAUDE.md
- Document local testing procedure
- Document expected baselines

### Phase 4: Verification (30 minutes)
- Verify standard CI still passes
- Verify performance test skipped by default
- Verify nightly job executes
- Local testing walkthrough

**Total Effort**: 2-3 hours

## Test Location Reference

```
File: crates/bitnet-inference/tests/batch_prefill.rs
Lines: 219-269
Current: #[ignore] + basic env guard (INSUFFICIENT)
Target: #[ignore] + relaxed assertions + load detection (COMPLETE)

Adjacent Tests (Run in Standard CI):
- test_batch_prefill_timing (lines 176-217) ✓ FUNCTIONAL, NO TIMING
- test_prefill_error_recovery (lines 271-288) ✓ FUNCTIONAL, NO TIMING
- test_empty_batch_handling (lines 290-301) ✓ FUNCTIONAL, NO TIMING
- test_single_prompt_batch (lines 303-320) ✓ FUNCTIONAL, NO TIMING
```

## Performance Baselines (Reference)

When test runs on **idle system, single-threaded**:
- Tokenization: 1.0-1.5ms (1ms mock sleep + overhead)
- Prefill: 10-15ms (10ms mock sleep + overhead)

When test runs on **high-load system**:
- Tokenization: 5-20ms (scheduler jitter)
- Prefill: 50-300ms (scheduler preemption)
- **Test will warn but NOT fail** (with new pattern)

## Code Changes Summary

### Assertion Change Example

**Before** (line 251-256):
```rust
assert!(
    (8.0..=100.0).contains(&prefill_time),
    "Prompt {} prefill time {} should be reasonable",
    i,
    prefill_time
);
```

**After**:
```rust
if prefill_time < 5.0 || prefill_time > 200.0 {
    eprintln!("⚠️  PERF-WARNING: Prompt {} prefill time {:.2}ms is unusual", 
             i, prefill_time);
    eprintln!("   (Expected ~10ms ± 200%% due to system load)");
    eprintln!("   This is NOT a test failure, just a system load indicator.");
}
```

**Key Difference**: Warning instead of assertion failure

## Local Testing Procedure

```bash
# Prerequisites: Close other applications, check system load with 'top'

# Run on single thread, with performance test enabled
RUN_PERF_TESTS=1 RAYON_NUM_THREADS=1 cargo test \
  --test batch_prefill \
  test_batch_prefill_performance_consistency \
  -- --test-threads=1 --nocapture

# Expected output on idle system:
# ✓ All functional tests passed
# Performance Results (System-Dependent):
# =====================================
# Prompt 0: prefill = 11.23ms
# Prompt 0: tokenize = 1.05ms
# ...

# Expected on high-load system:
# ✓ All functional tests passed
# ⚠️  PERF-WARNING: Prompt 2 prefill time 145.67ms is unusual
#   (Expected ~10ms ± 200% due to system load)
```

## CI Workflow Changes

### Standard CI (No Changes)
- Continues to skip `#[ignore]` tests
- All other tests pass normally
- Zero test failures from performance tests

### New Nightly Job
- File: `.github/workflows/testing-framework-performance.yml`
- Trigger: Scheduled daily (2 AM UTC) or manual dispatch
- Environment: `RUN_PERF_TESTS=1`, single-threaded
- Output: JUnit metrics archived
- Status: Informational (doesn't block merge)

## Verification Checklist

### Before Implementing
- [ ] Read both summary and full analysis documents
- [ ] Understand all 5 root causes
- [ ] Review code pattern in Section 3.1
- [ ] Understand CI workflow (Section 5.1)

### During Implementation
- [ ] Test code changes compile without warnings
- [ ] Test skipped by default (0 failures in standard CI)
- [ ] Nightly workflow runs successfully
- [ ] Documentation is clear and complete

### After Deployment
- [ ] Standard CI passes all tests
- [ ] Performance test only runs with `RUN_PERF_TESTS=1`
- [ ] Nightly job executes on schedule
- [ ] Local development docs work as written
- [ ] No increase in CI failure rate

## References in BitNet-rs Codebase

### Configuration Files
- `.config/nextest.toml` - Test runner settings (profiles: default, ci)
- `Cargo.toml` - Dependencies, workspace structure

### Related Test Files
- `tests/support/env_guard.rs` - Environment isolation pattern (use as reference)
- `crates/bitnet-inference/tests/` - Other integration tests

### Related Issues & PRs
- **Issue #254**: Shape mismatch in layer-norm
- **Issue #439**: Feature gate consistency ✅ RESOLVED
- **PR #475**: Comprehensive integration and EnvGuard (foundation for this pattern)

### Documentation
- `CLAUDE.md` - Project status, test patterns
- `docs/development/test-suite.md` - Testing framework reference

## Document Version History

| Version | Date | Status | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-23 | COMPLETE | Initial analysis, all sections complete |

## FAQ

### Q: Will this change affect standard CI?
**A**: No. Standard CI continues to skip `#[ignore]` tests. Zero changes to production CI.

### Q: Can I run the performance test locally?
**A**: Yes, with `RUN_PERF_TESTS=1 RAYON_NUM_THREADS=1` on an idle system. See local testing section.

### Q: What if system load is high?
**A**: Test will print warnings but NOT fail. This is by design - timing is system-dependent.

### Q: When should I split into functional test?
**A**: Optional but recommended. Provides guaranteed-passing functional coverage in standard CI.

### Q: How do I know if my system is suitable for running the test?
**A**: The test checks `/proc/loadavg` on Linux and warns if load > num_cpus × 1.5.

## Support

For questions or clarifications:
1. Read the full analysis in `batch_prefill_perf_quarantine.md`
2. Check FAQ section above
3. Review code examples in Section 3.1 and Section 5
4. Reference similar patterns in `tests/support/env_guard.rs`

---

**Document Status**: COMPLETE & READY FOR IMPLEMENTATION
**Created**: 2025-10-23
**Thoroughness**: Medium (comprehensive but focused)
**Total Analysis**: ~800 lines across 2 documents

---

**Document Metadata**

- **Created:** 2025-10-23
- **Last Reviewed:** 2025-10-23
- **Status:** Active
- **Next Review:** 2025-11-23

---
