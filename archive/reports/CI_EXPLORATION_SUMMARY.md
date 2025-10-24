# BitNet.rs CI Infrastructure Exploration Summary

**Date**: 2025-10-23
**Scope**: Link-check configuration, CI job dependencies, feature matrix testing
**Status**: Complete - Comprehensive analysis with 2 detailed reports generated

---

## Quick Reference

### Files Generated
1. **CI_LINK_CHECK_AND_DAG_ANALYSIS.md** (16 KB)
   - Detailed link-check configuration (lychee + markdown-link-check)
   - Complete CI job DAG with dependencies
   - Feature matrix breakdown
   - Guard jobs documentation
   - Nextest configuration analysis

2. **CI_JOB_DEPENDENCY_GRAPH.txt** (13 KB)
   - Visual ASCII dependency graph
   - Tier-based job organization
   - Gating vs non-blocking classification
   - Feature matrix detail
   - Critical path identification

---

## Key Findings

### 1. Link-Check Configuration

**Primary Tool**: lychee (in `quality` job)
- **Config file**: `.lychee.toml`
- **Mode**: Offline (no external URL checks)
- **Excluded**: `docs/archive/` directory (line 46)

**Critical Issue**: 
```toml
exclude = [
    ...
    "docs/archive/",  # Historical documentation - not maintained (archived 2025-10-23)
]
```

**Impact**: Any broken links in `/docs/archive/**` are NOT caught by the link checker.

**Secondary Tool**: markdown-link-check (in `documentation-validation.yml`)
- Runs independently from main CI workflow
- Only on doc-specific changes or manual trigger
- Checks external links (non-blocking)

---

### 2. CI Job Architecture

**Total Jobs**: 22 (across all workflows)

**Gating Jobs** (11 - MUST PASS):
- `test` - Primary gate (ubuntu, windows, macos, aarch64)
- `feature-matrix` - 6 feature combinations
- `doctest-matrix` - 3 doctest variants
- `doctest` - CPU-specific doctests
- `guard-fixture-integrity` - GGUF fixture validation
- `guard-serial-annotations` - #[serial(bitnet_env)] enforcement
- `guard-feature-consistency` - #[cfg(feature)] validation
- `env-mutation-guard` - No raw env mutations
- `quality` - Includes lychee link check
- `security` - CVE audit + license check
- `ffi-zero-warning-windows` - MSVC zero-warning requirement

**Non-Blocking Jobs** (5 - observational):
- `feature-hack-check` - Full powerset analysis
- `guard-ignore-annotations` - Unannotated #[ignore] tests
- `ffi-smoke` - FFI compile check
- `perf-smoke` - 4-token inference timing
- `api-compat` - API surface changes (PR-only)

**Conditional Jobs** (6 - special triggers):
- `crossval-cpu-smoke` - PR or main (fast smoke)
- `crossval-cpu` - Main or dispatch (full validation)
- `build-test-cuda` - GPU runner (main/dispatch/schedule)
- `crossval-cuda` - GPU runner (main/dispatch/schedule)
- `benchmark` - Main pushes only (performance tracking)

---

### 3. Job Dependency Hierarchy

```
Tier 1: test (primary gate)
    ↓ (all downstream depend on this)
Tier 2: 13 jobs (mostly parallel after test)
    ├─ feature-hack-check [O]
    ├─ feature-matrix [G]
    ├─ doctest-matrix [G]
    ├─ doctest [G]
    ├─ guard-* jobs [G except ignore-annotations]
    ├─ env-mutation-guard [G]
    ├─ quality [G] ← INCLUDES LYCHEE
    └─ security [G]
    ↓
Tier 3: Special gates (parallel with Tier 2)
    ├─ ffi-smoke [O]
    ├─ perf-smoke [O]
    └─ ffi-zero-warning-windows [G]
    ↓
Tier 4: Cross-validation (conditional)
    ├─ crossval-cpu-smoke [C]
    ├─ crossval-cpu [C]
    ├─ build-test-cuda [C]
    └─ crossval-cuda [C]
    ↓
Tier 5: Benchmarking (main only)
    └─ benchmark [C][O]
```

**Legend**:
- [G] = Gating (must pass)
- [O] = Observable (non-blocking)
- [C] = Conditional (special triggers)

---

### 4. Feature Matrix Coverage

**feature-matrix** job tests (GATING):
1. `cpu` - Baseline CPU inference
2. `cpu,avx2` - SIMD optimizations
3. `cpu,fixtures` - GGUF fixture integration
4. `cpu,avx2,fixtures` - SIMD + fixtures
5. `ffi` - C++ FFI bridge
6. `gpu` - Compile-only (no GPU in CI)

Each feature set:
- Builds with `cargo build --no-default-features --features`
- Runs tests with appropriate nextest profile
- Runs doctests with `cargo test --doc`

**feature-hack-check** job (NON-BLOCKING):
- Full powerset analysis with `cargo-hack --depth 2`
- Comprehensive feature interaction testing
- Continue-on-error: true (observational only)

---

### 5. Nextest Configuration

**Location**: `.config/nextest.toml`

**Profiles**:
- `default` - Full test suite, fail-fast
- `ci` - 4 fixed threads, 300s timeout, no retries (primary)
- `fixtures` - 2 threads (I/O bound), 600s timeout
- `gpu` - 1 thread (GPU memory), 300s timeout
- `doctests` - num_cpus threads, 120s timeout

**Common Settings**:
- `retries = 0` (no flaky test tolerance)
- `success-output = "never"` (clean logs)
- `failure-output = "immediate"` (fast feedback)
- `slow-timeout = 300s` (prevents CI hangs)

---

### 6. Guard Jobs (Quality Enforcement)

| Job | Purpose | Config | Blocking |
|-----|---------|--------|----------|
| `guard-fixture-integrity` | GGUF checksums & alignment | `scripts/validate-fixtures.sh` | Yes |
| `guard-serial-annotations` | #[serial(bitnet_env)] validation | `scripts/check-serial-annotations.sh` | Yes |
| `guard-feature-consistency` | Feature gate hygiene | `scripts/check-feature-gates.sh` | Yes |
| `guard-ignore-annotations` | #[ignore] documentation | `scripts/check-ignore-annotations.sh` | No |
| `env-mutation-guard` | No raw env mutations | ripgrep pattern search | Yes |

---

### 7. Link Checking Strategy

**What IS Checked**:
- ✅ All `**/*.md` files (recursive)
- ✅ Internal links (file references)
- ✅ Markdown syntax (via markdownlint)
- ✅ Code examples in doctests

**What IS NOT Checked**:
- ❌ `docs/archive/` directory (explicitly excluded)
- ❌ External HTTP(S) URLs in CI (offline mode)
- ❌ `.html` files (depends on offline mode)

**Two Separate Workflows**:
1. **ci.yml** (primary)
   - Runs on every push/PR (code changes)
   - Uses lychee in offline mode
   - Hard gate: fails merge if broken links
   - Excludes docs/archive/

2. **documentation-validation.yml** (secondary)
   - Runs on doc-specific changes
   - Uses markdown-link-check
   - External link validation
   - Non-blocking (observational)

---

## Critical Issues

### Issue #1: docs/archive/ Excluded from Link Checks
**Severity**: Medium
**Status**: By design (archived 2025-10-23)
**Workaround**: Manual review of archived docs before archiving

### Issue #2: Offline Mode Limits External Link Validation
**Severity**: Low
**Status**: By design (CI performance)
**Mitigation**: markdown-link-check provides external validation (non-blocking)

### Issue #3: markdown-link-check is Non-Blocking
**Severity**: Low
**Status**: By design (separate validation workflow)
**Mitigation**: Runs on doc-specific changes independently

---

## Architecture Summary

### Job Orchestration
- **Parallel execution**: Most jobs run in parallel after `test` passes
- **Dependency isolation**: Guard jobs are independent (no cross-dependencies)
- **Conditional triggers**: Cross-validation and benchmarks only on main/dispatch
- **Nested workflows**: Separate documentation validation workflow

### Feature Coverage
- **6 feature combinations** tested (cpu, cpu+avx2, cpu+fixtures, ffi, gpu)
- **Powerset analysis** with cargo-hack (non-blocking)
- **3 doctest variants** (cpu, cpu+avx2, all-features)

### Quality Gates
- **11 hard gates** (must pass for merge)
- **5 non-blocking observatories** (informational only)
- **Guard jobs enforce**: fixtures, env isolation, feature consistency, annotations

### Link Checking
- **Primary**: lychee (offline, in quality job)
- **Secondary**: markdown-link-check (external links, separate workflow)
- **Archive handling**: Explicitly excluded from checks

---

## Recommendations

### For Link Checking
1. **Review archived docs manually** before archiving (since they're excluded from CI)
2. **Consider enabling external link checks** on PRs (currently offline for performance)
3. **Add notification** when `docs/archive/` is modified (since links won't be validated)

### For CI Infrastructure
1. **Monitor job execution times** (test is critical path)
2. **Consider caching** between feature matrix jobs (they have overlapping builds)
3. **Add explicit test count expectations** (for regression detection)

### For Feature Testing
1. **Current coverage is comprehensive** (6 combinations + powerset analysis)
2. **Consider adding performance regression detection** (currently non-blocking)
3. **Guard jobs are well-designed** (isolated, orthogonal concerns)

---

## Additional Resources

### Files Generated
- **CI_LINK_CHECK_AND_DAG_ANALYSIS.md**: Comprehensive technical details
- **CI_JOB_DEPENDENCY_GRAPH.txt**: Visual representation with tier organization

### Key Configuration Files
- `.lychee.toml` (lines 1-75) - Link checker configuration
- `.github/workflows/ci.yml` (lines 1-1098) - Main CI workflow
- `.github/workflows/documentation-validation.yml` - Documentation validation
- `.config/nextest.toml` (lines 1-96) - Test execution configuration

### Related Documentation
- `CLAUDE.md` - Project status and test infrastructure
- `docs/development/test-suite.md` - Test framework documentation
- `docs/development/validation-ci.md` - CI validation details

---

## Summary Statistics

| Aspect | Count |
|--------|-------|
| **Total CI Jobs** | 22 |
| **Gating Jobs** | 11 |
| **Non-Blocking Jobs** | 5 |
| **Conditional Jobs** | 6 |
| **Feature Combinations** | 6 (feature-matrix) + powerset (feature-hack) |
| **Doctest Variants** | 3 |
| **Guard Job Types** | 5 |
| **Nextest Profiles** | 5 |
| **Excluded Paths** | 10+ |
| **Link Checkers** | 2 (lychee + markdown-link-check) |

---

**Exploration Complete**
Generated by: CI Infrastructure Analysis Script
Date: 2025-10-23
