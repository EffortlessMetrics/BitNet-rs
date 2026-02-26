# CI Job Dependencies & Git Hooks Analysis

**Date**: 2025-10-23  
**Repository**: BitNet-rs (feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2)  
**Thorough Exploration**: CI DAG hygiene and hook dependency checks

---

## Executive Summary

The BitNet-rs CI pipeline has a well-structured DAG with clear separation between:
- **Primary gates** (blocking merge): `test` job gates all others
- **Feature validation gates**: `feature-matrix`, `doctest-matrix`, `feature-hack-check`
- **Quality guards**: Fixture integrity, serial annotations, feature consistency, ignore annotations, env mutations
- **Performance observers**: `perf-smoke` (non-blocking)
- **Cross-validation jobs**: CPU smoke, full CPU, CUDA smoke, full CUDA
- **Security & API checks**: Separate concerns with optional PR gating

The pre-commit hook provides local enforcement of CI guards before commits reach CI/CD.

---

## Job Dependency Structure (DAG)

### Job Classes

#### Tier 0: Primary Gate
```
test (ubuntu-latest, windows-latest, macos-latest)
├─ Formats & linting (cargo fmt, clippy, banned patterns)
├─ Compilation (CPU, ARM64 cross-compile)
└─ Tests (nextest, all feature combinations)
```

#### Tier 1: Core Feature Validation (All depend on `test`)
```
test ──┬─→ feature-hack-check (non-blocking)
       ├─→ feature-matrix (gates CI)
       │   ├─ cpu
       │   ├─ cpu,avx2
       │   ├─ cpu,fixtures
       │   ├─ cpu,avx2,fixtures
       │   ├─ ffi
       │   └─ gpu (compile-only)
       └─→ doctest-matrix (gates CI)
           ├─ cpu
           ├─ cpu,avx2
           └─ all-features (non-blocking)
```

#### Tier 2: Guards (Independent - No dependencies)
```
guard-fixture-integrity
guard-serial-annotations
guard-feature-consistency
guard-ignore-annotations (non-blocking)
env-mutation-guard
```

#### Tier 3: Performance & Observability (depend on `test`)
```
test ──→ perf-smoke (non-blocking)
         ├─ Downloads model
         ├─ Builds CLI (release)
         ├─ Runs 4-token inference
         └─ Generates receipt (with verification)
```

#### Tier 4: Legacy/Conditional Features
```
test ──→ doctest (gates CI)
```

#### Tier 5: API & Security (Independent or PR-gated)
```
api-compat (PR-only gate)
security (always runs)
```

#### Tier 6: FFI Validation (depend on `test`)
```
test ──┬─→ ffi-smoke
       ├─→ ffi-zero-warning-windows
       └─→ ffi-zero-warning-linux
```

#### Tier 7: Cross-Validation (depend on `test`, limited by conditions)
```
test ──┬─→ crossval-cpu (main/dispatch only, gates)
       ├─→ crossval-cpu-smoke (all PRs, gates)
       ├─→ build-test-cuda (self-hosted GPU, main/dispatch/schedule)
       └─→ crossval-cuda (self-hosted GPU, main/dispatch/schedule)
```

#### Tier 8: Performance Benchmarks (main branch only)
```
benchmark (main branch push only)
quality (always runs)
```

---

## Job Classification

### Blocking Gates (Must Pass for Merge)
1. **test** - Primary Rust test suite across 3 OSes
2. **feature-matrix** - Curated critical feature sets
3. **doctest-matrix** - Documentation examples
4. **doctest** - CPU doctests
5. **crossval-cpu-smoke** - Fast smoke test validation
6. **guard-fixture-integrity** - Fixture checksum/schema validation
7. **guard-serial-annotations** - EnvGuard pattern enforcement
8. **guard-feature-consistency** - Feature gate definition checks
9. **env-mutation-guard** - Raw env mutation prevention

### Non-Blocking Observers (Informational Only)
1. **feature-hack-check** - Full powerset feature combinations (`continue-on-error: true`)
2. **guard-ignore-annotations** - Bare #[ignore] detection (`continue-on-error: true`)
3. **perf-smoke** - Performance regression tracking (`continue-on-error: true`)
4. **doctest-matrix:all-features** - GPU compilation (may fail on non-GPU runners)

### Conditional Gates (Depends on Trigger)
- **crossval-cpu** - Only on `main` branch push or workflow_dispatch
- **build-test-cuda** - Only on self-hosted GPU runners
- **crossval-cuda** - Only on self-hosted GPU runners
- **benchmark** - Only on `main` branch push
- **api-compat** - Only on pull_request events

---

## Ripgrep Usage in CI & Hooks

### Pre-commit Hook (`/.githooks/pre-commit`)

**Check 1: Bare #[ignore] Annotation**
```bash
rg -n -P '#\[ignore\](?!\s*=)' --hidden --glob '!**/target/**' --glob '*.rs' crates tests tests-new xtask
```
- **Pattern**: Negative lookahead for `=` (requires annotated form)
- **Scope**: Explicit glob filters for test directories
- **Purpose**: Prevents bare `#[ignore]` markers; enforces `#[ignore = "reason"]`

**Check 2: Raw Environment Mutations**
```bash
rg -n '(std::env::set_var|std::env::remove_var)\(' --glob '*.rs' --glob '!**/tests/helpers/**' --glob '!**/support/**' --glob '!**/env_guard.rs' crates tests tests-new xtask
```
- **Pattern**: Literal function call matching
- **Scope**: Excludes helper/support directories where EnvGuard is defined
- **Purpose**: Enforces EnvGuard pattern with `#[serial(bitnet_env)]` in tests

---

### CI Guard Scripts (Using Ripgrep)

#### 1. Guard: Serial Annotations (`scripts/check-serial-annotations.sh`)
```bash
rg -n 'EnvGuard::new|temp_env::with_var' crates tests --type rust -B 5
```
- **Pattern**: Searches for EnvGuard usage and backtracks 5 lines
- **Purpose**: Ensures `#[serial(bitnet_env)]` annotation exists
- **Severity**: Gating guard (exit 1 on failure)

#### 2. Guard: Feature Consistency (`scripts/check-feature-gates.sh`)
```bash
# Extract defined features
grep -A 100 '^\[features\]' Cargo.toml | grep '^[a-z0-9_-]* ='

# Find #[cfg(feature = "...")] usage
rg -oI '#\[cfg.*feature\s*=\s*"([^"]+)"' --replace '$1' crates --type rust

# Check for antipattern (gpu without cuda fallback)
rg -n '#\[cfg\(feature = "gpu"\)\]' crates --type rust
```
- **Patterns**: Feature definition extraction + cfg macro scanning + antipattern detection
- **Purpose**: Cross-checks feature definitions with actual usage
- **Severity**: Gating guard (exit 1 on undefined features); warning on antipatterns

#### 3. Guard: Ignore Annotations (`scripts/check-ignore-annotations.sh`)
```bash
rg -n '#\[ignore\]' crates tests --type rust
```
- **Pattern**: Simple match for ignore attribute
- **Purpose**: Verifies each has issue reference or justification via line context
- **Severity**: Non-blocking observer (continue-on-error: true)

#### 4. Guard: Environment Mutation (`scripts/env-mutation-guard in ci.yml`)
```bash
rg -n '(std::env::set_var|std::env::remove_var)\(' crates \
  --glob '!**/tests/support/**' \
  --glob '!**/support/**' \
  --glob '!**/helpers/**' \
  --type rust
```
- **Pattern**: Same as pre-commit but without custom helper exclusions
- **Purpose**: Inline CI check without shell script
- **Severity**: Gating guard (exit 1 on detection)

---

## Guard Job Dependencies

### Independent Guards (No external dependencies)
```
guard-fixture-integrity
  └─ Reads: ci/fixtures/qk256/SHA256SUMS
  └─ Uses: sha256sum, cargo run inspect, jq
  └─ Exit: 1 on checksum mismatch or GGUF structure failure

guard-serial-annotations
  └─ Reads: Cargo.toml + all .rs files
  └─ Uses: ripgrep (rg)
  └─ Exit: 1 on missing #[serial(bitnet_env)]

guard-feature-consistency
  └─ Reads: Cargo.toml + all .rs files
  └─ Uses: ripgrep (rg) + grep
  └─ Exit: 1 on undefined features (warns on antipatterns)

guard-ignore-annotations
  └─ Reads: all .rs files
  └─ Uses: ripgrep (rg) + sed + grep
  └─ Exit: 1 on bare #[ignore] (non-blocking in CI)

env-mutation-guard
  └─ Reads: all .rs files in crates/
  └─ Uses: ripgrep (rg) inline
  └─ Exit: 1 on raw env mutations
```

**Critical Observation**: All guards are **standalone** with no cross-guard dependencies. This allows:
- Parallel execution
- Independent failure diagnosis
- No cascading failures
- Clear owner responsibility per guard

---

## Pre-commit Hook Dependencies

The `.githooks/pre-commit` script provides **local enforcement before commits reach CI**:

### Hook Checks (Sequential)
```
1. Install ripgrep (must exist locally)
2. Check bare #[ignore] → exit 1 if violated
3. Check raw env mutations → exit 1 if violated
4. Exit 0 if all checks pass
```

### Hook Configuration
- **Enable**: `git config core.hooksPath .githooks`
- **Bypass** (not recommended): `git commit --no-verify`
- **Matched by CI**: All checks are mirrored in CI guard jobs

---

## Preflight Checks in CI

### `test` Job Preflight Steps
```
1. Checkout with submodules
2. Install Rust + targets
3. Setup BitNet.cpp cache (if crossval label or main branch)
4. Install cross-compilation tools (for ARM64)
5. Cache cargo registry
6. Install ripgrep (Ubuntu x86_64 only)
7. Install nextest
8. Check formatting → exit 1 on failure
9. Run clippy → exit 1 on failure
10. Check banned patterns via bash script
11. Compile tests → exit 1 on failure
12. Build workspace (CPU)
13. Run nextest
14. Final compile check (tests)
15. Final build (CPU)
16. Cross-compile (ARM64 if applicable)
```

### `crossval-cpu-smoke` Preflight Steps
```
1. Checkout
2. Install Rust
3. Cache setup
4. Install build tools (gcc, cmake)
5. Cache BitNet C++ (smoke variant)
6. Fetch C++ (pinned commit)
7. Download model
8. Verify model SHA vs lock
9. Run smoke tests
```

---

## DAG Hygiene Assessment

### Strengths
1. **Single primary gate**: `test` job gates all dependent jobs
   - Clear bottleneck for resource allocation
   - Quick feedback on compilation/test failures
   - Multi-platform coverage (3 OSes)

2. **Independent guard jobs**:
   - No guard-to-guard dependencies
   - Parallel execution reduces CI time
   - Isolated failure modes

3. **Feature validation coverage**:
   - Curated critical sets (feature-matrix)
   - Full powerset check (feature-hack-check, non-blocking)
   - Doctest examples validated

4. **Cross-validation structure**:
   - Smoke test for all PRs (gates)
   - Full validation on main/dispatch (gates)
   - Separate CUDA runners avoid blocking CPU tests

5. **Performance observability**:
   - Non-blocking smoke test
   - Receipt generation for parity tracking
   - No gate failure on performance regression

### Weaknesses & Improvements

1. **Missing Direct Dependencies in Some Jobs**
   - Problem: `doctest` job doesn't declare explicit `needs: [test]` (implicit ordering)
   - Risk: Potential race condition if runner scheduling changes
   - Recommendation: Add explicit `needs: test` to all Tier 1+ jobs

2. **Guard Job Visibility**
   - Problem: Guard jobs have no explicit ordering/grouping in DAG
   - Risk: Unclear which guards block merge vs inform
   - Recommendation: Document guard classification in job names (e.g., "GATE - ..." vs "OBSERVE - ...")

3. **Ripgrep Installation Per Job**
   - Problem: `sudo apt-get install ripgrep` duplicated in multiple guards
   - Risk: Network failures; slower CI; version inconsistency
   - Recommendation: Install once in `test` job, cache as artifact, or use pre-built runner

4. **Feature Matrix Scope**
   - Problem: `feature-hack-check` is non-blocking; untested combinations could slip to main
   - Risk: Silent feature gate bugs in production combinations
   - Recommendation: Consider making depth-1 or depth-2 powerset blocking for critical crates

5. **Conditional Guard Enforcement**
   - Problem: Some guards only run on Ubuntu; Windows/macOS skip format checks
   - Risk: Platform-specific violations not caught
   - Recommendation: Run key guards (`guard-feature-consistency`, etc.) on all platforms

6. **Fixture Integrity Not Cached**
   - Problem: Guard rebuilds CLI each time to inspect fixtures
   - Risk: Slow fixture validation; no artifact caching
   - Recommendation: Cache `bitnet-cli` binary or move fixture validation to separate step

---

## Dependency Graph Visualization

```
                        ┌─────────────────────────────────────┐
                        │         test (Tier 0)                │
                        │  All platforms, all tests, clippy   │
                        └────────┬────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────────┐
                 │               │                   │
        ┌────────▼────────┐      │       ┌──────────▼──────────┐
        │ feature-matrix  │      │       │ doctest-matrix      │
        │ (gates CI)      │      │       │ (gates CI)          │
        └─────────────────┘      │       └─────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼──────┐      ┌────────────▼────┐      ┌────────▼────────┐
   │ perf-smoke│      │ ffi-smoke        │      │ crossval-cpu     │
   │(observe)  │      │ (gates)          │      │ (gates)          │
   └───────────┘      └──────────────────┘      └──────────────────┘
        
        ┌──────────────┬────────────────────┬──────────────────────┐
        │              │                    │                      │
   ┌────▼──────────┐   │   ┌────────────────▼──────┐   ┌──────────▼─────────┐
   │ crossval-cuda │   │   │ build-test-cuda       │   │ quality            │
   │ (GPU runners) │   │   │ (GPU runners)         │   │ (always)           │
   └───────────────┘   │   └───────────────────────┘   └────────────────────┘
                       │
                ┌──────▼──────────────┐
                │ benchmark (main)    │
                └─────────────────────┘
                
    ┌──────────────────────────────────────────────────────────────┐
    │                    Independent Guards                         │
    ├──────────────────────────────────────────────────────────────┤
    │ guard-fixture-integrity                                      │
    │ guard-serial-annotations                                     │
    │ guard-feature-consistency                                    │
    │ guard-ignore-annotations (observe)                           │
    │ env-mutation-guard                                           │
    │ api-compat (PR-only)                                         │
    │ security (always)                                            │
    └──────────────────────────────────────────────────────────────┘
```

---

## Git Hook Preflight Checks

### Hook Dependencies

The `.githooks/pre-commit` script enforces **two orthogonal checks**:

```
pre-commit Hook
├─ Check 1: Bare #[ignore] Annotations
│  ├─ Requires: ripgrep (rg)
│  ├─ Pattern: #\[ignore\](?!\s*=)
│  ├─ Scope: crates/, tests/, tests-new/, xtask/
│  └─ Failure: exit 1, prevents commit
│
└─ Check 2: Raw Environment Mutations
   ├─ Requires: ripgrep (rg)
   ├─ Pattern: std::env::(set_var|remove_var)\(
   ├─ Exclusions: tests/helpers/**, support/**, env_guard.rs
   └─ Failure: exit 1, prevents commit
```

### Hook Mirroring in CI

Each pre-commit check has a **corresponding CI guard**:

| Hook Check | CI Guard | Blocking |
|-----------|----------|----------|
| Bare #[ignore] | `guard-ignore-annotations` | No (observe) |
| Raw env mutations | `env-mutation-guard` | Yes (gate) |
| Feature gates* | `guard-feature-consistency` | Yes (gate) |
| Serial annotations* | `guard-serial-annotations` | Yes (gate) |

*Note: These checks exist in CI but not in pre-commit hook (could be added)

---

## Recommendations for CI DAG Hygiene

### Priority 1: Improve Job Dependency Clarity
```yaml
# Add explicit dependencies to eliminate implicit ordering
doctest:
  needs: test  # Currently missing explicit declaration

api-compat:
  needs: [test]  # May benefit from explicit gate dependency
```

### Priority 2: Guard Job Naming Convention
```yaml
# Rename for clarity in job list
- guard-fixture-integrity     # GATE - Fixture Integrity
- guard-serial-annotations    # GATE - Serial Annotations
- guard-feature-consistency   # GATE - Feature Consistency
- guard-ignore-annotations    # OBSERVE - Ignore Annotations
- env-mutation-guard          # GATE - Environment Mutations
```

### Priority 3: Optimize Ripgrep Installation
```yaml
# Option A: Centralize in test job, cache as artifact
# Option B: Use matrix variable, single install step
# Option C: Use pre-installed ripgrep on ubuntu-latest runners

# Current: Duplicated in 3+ guard jobs
# Proposed: Shared step or pre-built environment
```

### Priority 4: Add Hook Checks to CI
```bash
# Enhance ci.yml guards to match all pre-commit checks
# Current hook checks not in CI:
# - Bare #[ignore] detection (only in observe-only guard)
# - Feature gate checks (already in gating guard)
# - Serial annotation checks (already in gating guard)

# Recommended: Add feature/serial checks to pre-commit hook
```

### Priority 5: Conditional Guard Simplification
```yaml
# Run guards on primary test platform (ubuntu-latest x86_64)
# to avoid duplication across matrix

guard-*:
  runs-on: ubuntu-latest  # Simplify from current scattered approach
  # Avoid running same check 3x on test matrix
```

### Priority 6: Feature Matrix Gating
```yaml
feature-hack-check:
  continue-on-error: false  # Consider making depth-2 blocking
  # or: Create separate job for blocking coverage:
  feature-hack-check-minimal:
    needs: test
    continue-on-error: false  # Small subset, always blocks
```

---

## Files Referenced in Analysis

| File | Purpose | Ripgrep Usage |
|------|---------|---------------|
| `.github/workflows/ci.yml` | Main CI configuration | N/A |
| `.githooks/pre-commit` | Local enforcement hook | `rg` for #[ignore] and env mutations |
| `.githooks/README.md` | Hook documentation | N/A |
| `scripts/check-serial-annotations.sh` | CI guard for EnvGuard pattern | `rg` to find mutations, `sed` for context |
| `scripts/check-feature-gates.sh` | CI guard for feature consistency | `rg` for feature usage + `grep` for antipatterns |
| `scripts/check-ignore-annotations.sh` | CI guard for ignore justification | `rg` to find ignores, `sed` for context |
| `scripts/validate-fixtures.sh` | CI guard for fixture integrity | `cargo run inspect` (not ripgrep) |

---

## Summary Table

| Job | Type | Gate? | Dependencies | Ripgrep Used? |
|-----|------|-------|--------------|---------------|
| test | Primary | Yes | None | Yes (in hook) |
| feature-matrix | Validation | Yes | test | No |
| doctest-matrix | Validation | Yes | test | No |
| feature-hack-check | Validation | No | test | No |
| guard-fixture-integrity | Guard | Yes | None | No |
| guard-serial-annotations | Guard | Yes | None | Yes |
| guard-feature-consistency | Guard | Yes | None | Yes |
| guard-ignore-annotations | Guard | No | None | Yes |
| env-mutation-guard | Guard | Yes | None | Yes (inline) |
| perf-smoke | Observe | No | test | No |
| crossval-cpu | Cross-val | Yes | test | No |
| crossval-cpu-smoke | Cross-val | Yes | test | No |
| build-test-cuda | Build | Yes | test | No |
| crossval-cuda | Cross-val | Yes | test | No |
| api-compat | Check | Partial | None | No |
| security | Audit | No | None | No |
| quality | Audit | No | None | No |
| benchmark | Perf | No | None | No |
| ffi-smoke | Build | Yes | test | No |
| ffi-zero-warning-* | Build | Yes | test | No |

