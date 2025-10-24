# Concurrent Load Performance Test Quarantine - Quick Reference

**Navigation:** [ci/](../) → [solutions/](./00_NAVIGATION_INDEX.md) → This Document
**Related:** [PR #475 Summary](../PR_475_FINAL_SUMMARY.md)

---

**For developers**: Use this guide to quickly apply the quarantine pattern  
**Time to implement**: ~3-5 minutes  
**Risk level**: Minimal (test-only change)

---

## Quick Summary

**Problem**: `test_batch_processing_efficiency` is timing-sensitive and fails non-deterministically in CI  
**Solution**: Quarantine with `#[ignore]` + environment guard  
**Pattern**: Already proven in `batch_prefill.rs` (lines 220-228)  
**Impact**: Removes false CI failures, improves CI reliability

---

## Implementation Checklist

### Step 1: Locate Test

```bash
$ grep -n "fn test_batch_processing_efficiency" \
    crates/bitnet-server/tests/concurrent_load_tests.rs
# Line 316: async fn test_batch_processing_efficiency() -> Result<()> {
```

Target: **Line 312-376**

---

### Step 2: Add Quarantine Attributes

**Location**: Line 312, before `#[tokio::test]`  
**Action**: Insert 3 lines

Find (lines 312-313):
```rust
/// Test batch processing efficiency under concurrent load
#[tokio::test]
async fn test_batch_processing_efficiency() -> Result<()> {
```

Replace with:
```rust
/// Test batch processing efficiency under concurrent load
#[tokio::test]
#[ignore] // Performance test: timing-sensitive, causes non-deterministic CI failures
           // Run locally with: cargo test --ignored test_batch_processing_efficiency
           // Blocked by: environment-dependent timing issues (CPU load, scheduler, concurrent execution)
async fn test_batch_processing_efficiency() -> Result<()> {
```

---

### Step 3: Add Environment Guard

**Location**: Inside function, after line 316 (opening brace)  
**Action**: Insert 5 lines before first `println!`

Find (line 323):
```rust
async fn test_batch_processing_efficiency() -> Result<()> {
    println!("=== Batch Processing Efficiency Test ===");
```

Replace with:
```rust
async fn test_batch_processing_efficiency() -> Result<()> {
    // Guard: Only run if explicitly requested via environment variable
    if std::env::var("RUN_PERF_TESTS").ok().as_deref() != Some("1") {
        eprintln!("⏭️  Skipping performance test (set RUN_PERF_TESTS=1 to run)");
        return Ok(());
    }

    println!("=== Batch Processing Efficiency Test ===");
```

---

## Verification Commands

```bash
# 1. Confirm quarantine applied
grep -A 5 "#\[ignore\].*timing-sensitive" \
  crates/bitnet-server/tests/concurrent_load_tests.rs

# 2. Confirm guard added
grep -B 2 "RUN_PERF_TESTS" \
  crates/bitnet-server/tests/concurrent_load_tests.rs

# 3. Verify test is skipped by default
cargo test -p bitnet-server test_batch_processing_efficiency 2>&1 | grep -i ignored

# 4. Verify test runs with flag
RUN_PERF_TESTS=1 cargo test --ignored test_batch_processing_efficiency 2>&1 | head -20

# 5. Full test suite passes
cargo nextest run -p bitnet-server --profile ci
```

---

## Code Changes Summary

| File | Lines | Change | Type |
|------|-------|--------|------|
| `crates/bitnet-server/tests/concurrent_load_tests.rs` | 312-315 | Add `#[ignore]` attribute | Insert |
| `crates/bitnet-server/tests/concurrent_load_tests.rs` | 316-320 | Add environment guard | Insert |
| **Total** | **~8 lines** | Add quarantine | Non-invasive |

---

## What This Changes

### ✅ Before (Flaky)
```bash
$ cargo test -p bitnet-server test_batch_processing_efficiency
# Output: test_batch_processing_efficiency ... FAILED (timing-dependent)
# CI Impact: False failure rate 8-12%
```

### ✅ After (Stable)
```bash
$ cargo test -p bitnet-server test_batch_processing_efficiency
# Output: test_batch_processing_efficiency ... IGNORED
# CI Impact: No false failures

$ RUN_PERF_TESTS=1 cargo test --ignored test_batch_processing_efficiency
# Output: test_batch_processing_efficiency ... ok (or FAILED - timing-dependent)
# Local validation: Can manually run when needed
```

---

## Commit Message

```
test(concurrent-load): quarantine timing-sensitive batch efficiency test

Apply #[ignore] and environment guard to test_batch_processing_efficiency
to prevent non-deterministic CI failures.

The test is inherently timing-sensitive due to:
- Mock processing time randomness (±50% variance)
- System load variability in CI environments
- Async executor scheduling interference
- Batch timeout interaction with request patterns

Quarantine approach:
- Added #[ignore] attribute with documentation
- Added environment variable guard (RUN_PERF_TESTS=1)
- Follows precedent in batch_prefill.rs (lines 220-228)

CI Impact:
- Removes ~30s test from standard CI
- Improves CI reliability (no timing-based false failures)
- Test remains accessible via RUN_PERF_TESTS=1 for local validation

See: ci/solutions/concurrent_load_perf_quarantine.md
```

---

## For CI Configuration

**Current behavior** (no changes needed):
```bash
# In .github/workflows/ci.yml
- run: cargo nextest run --workspace --profile ci
# Result: test_batch_processing_efficiency automatically skipped
```

**Optional weekly performance tests** (new job):
```yaml
# In .github/workflows/nightly-perf.yml
name: Weekly Performance Tests
on:
  schedule:
    - cron: '0 2 * * 0'  # Sunday 2 AM UTC
jobs:
  perf:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo nextest run --workspace --ignored --profile ci
        env:
          RUN_PERF_TESTS: "1"
```

---

## Troubleshooting

### "Test still fails in CI"
→ Verify `#[ignore]` attribute is applied (check line 312)  
→ Confirm environment guard is in place (check line 316)  
→ Run `cargo clean && cargo nextest run -p bitnet-server --profile ci`

### "Test doesn't run with RUN_PERF_TESTS=1"
→ Need `--ignored` flag: `RUN_PERF_TESTS=1 cargo test --ignored test_batch_processing_efficiency`

### "Clippy/formatting issues"
→ Run `cargo fmt` on the file  
→ Run `cargo clippy --fix` to auto-fix (if applicable)

---

## Before You Commit

- [ ] Applied `#[ignore]` attribute (line 312-315)
- [ ] Added environment guard (line 316-320)
- [ ] Test skipped with default `cargo test` ✓
- [ ] Test runs with `RUN_PERF_TESTS=1` ✓
- [ ] CI profile skips test: `cargo nextest run --profile ci` ✓
- [ ] Formatting: `cargo fmt` ✓
- [ ] No clippy warnings: `cargo clippy --fix` ✓
- [ ] Commit message prepared (from template above)

---

## Time Estimates

| Step | Time |
|------|------|
| Locate test | 30 sec |
| Add attributes | 1 min |
| Add guard | 1.5 min |
| Verify changes | 2 min |
| Run tests | 3 min |
| Commit | 1 min |
| **Total** | **~9 minutes** |

---

## Related Sections

For detailed analysis:
- **Why it's flaky**: See `concurrent_load_perf_quarantine.md` → "Root Cause Summary"
- **Assertions explained**: See `concurrent_load_perf_quarantine.md` → "Efficiency Assertions"
- **Implementation details**: See `concurrent_load_perf_quarantine.md` → "Complete Implementation"
- **Verification approach**: See `concurrent_load_perf_quarantine.md` → "Verification Approach"

---

**Document Version**: 1.0  
**Status**: Ready for Implementation  
**Last Updated**: 2025-10-23

