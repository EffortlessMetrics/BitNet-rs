# Batch Prefill Performance Quarantine - Implementation Summary

**Navigation:** [ci/](../) → [solutions/](./00_NAVIGATION_INDEX.md) → This Document
**Related:** [PR #475 Summary](../PR_475_FINAL_SUCCESS_REPORT.md)

---

## Overview

This directory contains the comprehensive analysis of flakiness in `test_batch_prefill_performance_consistency` and recommended quarantine pattern.

## Files

### Primary Analysis Document
- **`batch_prefill_perf_quarantine.md`** (741 lines, 25KB)
  - Complete performance test analysis
  - Root cause analysis (5 primary causes identified)
  - Quarantine pattern with complete code
  - Alternative approaches evaluated
  - CI workflow recommendations
  - Verification checklist

## Key Findings

### Problem
The test at `crates/bitnet-inference/tests/batch_prefill.rs::test_batch_prefill_performance_consistency` has non-deterministic CI failures due to timing-sensitive assertions.

### Root Causes (5 Identified)

1. **Timer Resolution Variance**
   - Assertion threshold (0.5ms) at edge of system timer precision
   - CI VMs often have degraded precision

2. **Scheduler Jitter**
   - CI CPU is shared resource
   - Scheduler preemption can add 50+ ms to measured times
   - 100ms upper bound easily exceeded

3. **Async Runtime Overhead**
   - Test uses `#[tokio::test]` - executor adds 5-20ms overhead
   - Multiple awaits create context switching opportunities

4. **Parallel Test Interference**
   - Other tests consume CPU even when target test runs
   - Nextest config allows up to 4 parallel threads in CI

5. **Mock Realism Gap**
   - Mock tokenizer uses fixed 1ms delay regardless of input length
   - Real tokenizers scale with input, creating false precision expectations

### Solution: Quarantine Pattern

**Primary Approach**: 
- Mark test `#[ignore]` (removes from standard CI)
- Environment guard: `RUN_PERF_TESTS=1` required to run
- Relax assertions from hard failures to informational warnings
- Maintain functional coverage (batch size, result validity)
- Add system load detection warnings

**Secondary Approach** (Recommended):
- Split into two tests:
  1. `test_batch_prefill_functional` - Always runs, no timing
  2. `test_batch_prefill_performance_consistency` - Nightly only

**CI Scheduling**:
- Standard CI: Runs functional test only (no changes to existing setup)
- Nightly job: Runs performance test on dedicated runner
- New workflow: `.github/workflows/testing-framework-performance.yml`

## Implementation Checklist

### Changes to Test File
- [ ] Replace performance assertions with informational warnings
- [ ] Relax thresholds from (0.5, 8-100ms) to ±200% from baseline
- [ ] Keep functional assertions (batch size, result validity)
- [ ] Add detailed docstring explaining quarantine rationale
- [ ] Add system load detection code
- [ ] Optional: Create separate functional test file

### CI Configuration
- [ ] Create nightly performance job (`.github/workflows/testing-framework-performance.yml`)
- [ ] Set `RUN_PERF_TESTS=1` in nightly job environment
- [ ] Set `--test-threads=1` for deterministic execution
- [ ] Archive JUnit metrics for trend analysis
- [ ] No changes to standard CI (tests remain `#[ignore]`)

### Documentation
- [ ] Update `docs/development/test-suite.md` with performance testing guidance
- [ ] Add reference to this quarantine analysis in CLAUDE.md
- [ ] Document local testing procedure (single-threaded, low-load conditions)
- [ ] Document expected timing baseline (1ms tokenize, 10ms prefill)

### Verification
- [ ] Standard CI still passes all functional tests
- [ ] Performance test skipped by default (0 failures)
- [ ] Performance test runs with `RUN_PERF_TESTS=1` (metrics printed, no failures)
- [ ] Nightly job executes successfully
- [ ] Local development docs work as written

## Quick Reference

### Test Location
```
crates/bitnet-inference/tests/batch_prefill.rs
Lines: 219-269
Current: #[ignore] with basic env guard
Target: #[ignore] with relaxed assertions + load detection
```

### Assertion Changes

**Before** (Flaky):
```rust
assert!((8.0..=100.0).contains(&prefill_time));  // 92ms window
assert!(tokenize_time >= 0.5);                   // 0.5ms threshold
```

**After** (Informational):
```rust
if prefill_time < 5.0 || prefill_time > 200.0 {
    eprintln!("⚠️ PERF-WARNING: ...");  // Warning, not failure
}
if tokenize_time < 0.2 || tokenize_time > 20.0 {
    eprintln!("⚠️ PERF-WARNING: ...");  // Warning, not failure
}
```

### Local Testing Command

```bash
# Run on idle system, single-threaded
RUN_PERF_TESTS=1 RAYON_NUM_THREADS=1 cargo test \
  --test batch_prefill \
  test_batch_prefill_performance_consistency \
  -- --test-threads=1 --nocapture
```

## Performance Baselines

**Expected** (when test runs on idle system):
- Tokenization: 1.0-1.5ms (mock 1ms sleep + overhead)
- Prefill: 10-15ms (mock 10ms sleep + overhead)
- Tolerance: ±200% for system load variance

**With High Load**:
- Tokenization: 5-20ms (scheduler jitter)
- Prefill: 50-300ms (scheduler preemption)
- Test will warn but not fail

## Status

- Analysis: COMPLETE
- Pattern: READY FOR IMPLEMENTATION
- Estimated effort: 2-3 hours
  - Test code: 30 minutes
  - CI workflow: 30 minutes
  - Documentation: 1 hour
  - Verification: 30 minutes

## Next Steps

1. Review `batch_prefill_perf_quarantine.md` in detail
2. Implement test code changes (Section 3.1)
3. Create nightly CI workflow (Section 5.1)
4. Update documentation (Section 5.3)
5. Verify standard CI passes without performance tests
6. Verify nightly job runs successfully

## Related Issues

- **Issue #254**: Shape mismatch in layer-norm (affects other tests)
- **Issue #439**: Feature gate consistency (✅ RESOLVED in PR #475)
- **PR #475**: Comprehensive integration and EnvGuard implementation (foundation for this pattern)

## Contact / Review

This analysis is ready for:
- Code review (implementation details)
- CI review (workflow configuration)
- Documentation review (user-facing guidance)

---

**Document Status**: READY FOR IMPLEMENTATION
**Created**: 2025-10-23
**Analysis**: Medium thoroughness with complete recommendations
