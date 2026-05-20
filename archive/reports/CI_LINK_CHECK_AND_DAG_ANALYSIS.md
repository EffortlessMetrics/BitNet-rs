> **Archived CI/docs report claim boundary**
>
> This file is a historical CI/docs report from active development. Status,
> ready, validated, merge, production, backend, performance, and publication
> wording below is historical context only and is not a current project,
> support, CI, quality, release, backend, performance, or publication claim.
> Current claims must come from active docs, trackers, receipts, specs, and
> claim gates.
# BitNet-rs CI Infrastructure Analysis

## 1. Link-Check Configuration

### Lychee Configuration File
**Location**: `/home/steven/code/Rust/BitNet-rs/.lychee.toml`

**Key Settings**:
- **Offline mode**: `offline = true` (disabled for CI performance, skips external URL checks)
- **Max concurrency**: 8
- **Timeout**: 10 seconds
- **Max retries**: 2
- **Accept HTTP codes**: 200, 429
- **Cache enabled**: true
- **Progress bar disabled**: true (for CI)

**Excluded Paths**:
```toml
exclude = [
    "target/",
    "vendor/",
    "node_modules/",
    ".git/",
    ".vscode/",
    ".idea/",
    ".tmp",
    ".temp",
    "Cargo.lock",
    "package-lock.json",
    "docs/archive/",  # ⚠️ IMPORTANT: Historical documentation excluded
]
```

**CRITICAL FINDING**: `docs/archive/` is **explicitly excluded** from link checking (line 46):
```
"docs/archive/",  # Historical documentation - not maintained (archived 2025-10-23)
```

This means any broken links in `/docs/archive/**` files are **NOT** caught by the link checker.

---

## 2. Link-Check CI Job

### In Main CI Workflow (ci.yml)
**Job Name**: `quality`
**Location**: Lines 785-831 in `.github/workflows/ci.yml`
**Runs on**: ubuntu-latest
**Non-blocking**: No (fails CI if links are broken)

**Job Details**:
```yaml
quality:
  name: Code Quality
  runs-on: ubuntu-latest
  steps:
    # ... other quality checks ...
    
    # Link validation (offline mode for CI performance)
    - name: Check links
      run: |
        cargo install lychee || true
        lychee --accept 200,429 --no-progress --offline --config .lychee.toml "**/*.md"
```

**Key Points**:
- Uses `.lychee.toml` configuration file ✓
- Runs in `--offline` mode (no external link checks) ✓
- Checks all `**/*.md` files recursively
- `continue-on-error: false` (default) - **FAILS CI if broken links detected**
- Installed with `cargo install lychee || true` (fallback if already installed)

### In Documentation Workflow (documentation-validation.yml)
**Job Name**: `validate-documentation` 
**Location**: Lines 1-297 in `.github/workflows/documentation-validation.yml`
**Trigger**: On push/PR to docs, weekly schedule (Monday 8 AM UTC)

**Link Checking Tools Used**:
1. **markdown-link-check** (npm) - lines 69-78
   ```bash
   markdown-link-check README.md || echo "README.md has broken links"
   markdown-link-check INSTALLATION.md || echo "INSTALLATION.md has broken links"
   find docs -name "*.md" -exec markdown-link-check {} \; || echo "Some docs have broken links"
   ```
   - Does NOT use lychee.toml
   - Checks external links (interactive mode)
   - **Continue-on-error**: implicit (allows failure)

2. **Lychee** - NOT used in this workflow

---

## 3. CI Job DAG (Dependency Graph)

### Primary Test Gates
```
test (ubuntu-latest, windows-latest, macos-latest)
  ├─ Matrix: [x86_64, aarch64]
  └─ Runs: clippy, format check, banned patterns check, nextest
  └─ Output: Feeds 9 downstream jobs
```

### Feature Matrix & Doctest Gates
```
test
  ├─→ feature-hack-check (non-blocking, observability)
  │   └─ cargo-hack powerset (depth=2)
  │
  ├─→ feature-matrix (GATE - must pass)
  │   ├─ cpu, cpu+avx2, cpu+fixtures, cpu+avx2+fixtures, ffi
  │   └─ gpu (compile-only)
  │
  └─→ doctest-matrix (GATE - must pass)
      ├─ cpu, cpu+avx2, all-features
      └─ all-features: continue-on-error=true (GPU may not be available)
```

### Guard Jobs (Gating)
```
  ├─→ guard-fixture-integrity (GATE)
  │   └─ Runs: bash scripts/validate-fixtures.sh
  │
  ├─→ guard-serial-annotations (GATE)
  │   └─ Runs: bash scripts/check-serial-annotations.sh
  │
  ├─→ guard-feature-consistency (GATE)
  │   └─ Runs: bash scripts/check-feature-gates.sh
  │
  └─→ guard-ignore-annotations (NON-BLOCKING, observability)
      └─ Runs: bash scripts/check-ignore-annotations.sh
      └─ continue-on-error: true (134 bare markers exist)
```

### Documentation & Quality Jobs
```
  ├─→ doctest (GATE)
  │   ├─ CPU features only
  │   └─ all-features (continue-on-error: true - GPU may not be available)
  │
  ├─→ quality (GATE)
  │   ├─ cargo machete (unused deps)
  │   ├─ cargo outdated (outdated deps)
  │   ├─ cargo llvm-cov (coverage)
  │   ├─ cargo doc (documentation)
  │   ├─ markdownlint (markdown linting)
  │   └─ lychee (link checking) **Uses .lychee.toml with docs/archive/ excluded**
  │
  ├─→ security (GATE)
  │   ├─ cargo audit (CVE audit)
  │   └─ cargo deny (license/dependency checks)
  │
  └─→ api-compat (PR-only, non-blocking observability)
      ├─ cargo-semver-checks
      ├─ cargo public-api
      ├─ cbindgen FFI headers
      └─ CLI help diff
```

### Performance & Validation Jobs
```
  ├─→ perf-smoke (NON-BLOCKING, observability)
  │   ├─ needs: [test]
  │   ├─ Download model
  │   ├─ Build CLI (release)
  │   ├─ Run 4-token inference with /usr/bin/time
  │   ├─ Benchmark with receipt generation
  │   ├─ Verify receipt examples
  │   └─ Comment results on PR
  │
  └─→ env-mutation-guard (GATE)
      └─ Checks for raw std::env::{set_var,remove_var}() calls
```

### FFI & Build Jobs
```
  ├─→ ffi-smoke (NON-BLOCKING, observability)
  │   ├─ Matrix: [gcc/g++, clang/clang++]
  │   └─ Smoke build only (no tests)
  │
  └─→ ffi-zero-warning-windows (GATE)
      └─ needs: [test]
      └─ MSVC build must have zero warnings
```

### Cross-Validation & GPU Jobs
```
  ├─→ crossval-cpu-smoke (PR gate - fast smoke)
  │   ├─ needs: [test]
  │   ├─ Fetch C++ (pinned to CPP_TAG)
  │   ├─ Download model
  │   ├─ Run smoke tests: parity preflight + tiny checks
  │   └─ Upload results
  │
  ├─→ crossval-cpu (FULL - main branch only)
  │   ├─ needs: [test]
  │   ├─ Conditional: github.event == 'workflow_dispatch' || main
  │   └─ Full cross-validation suite
  │
  ├─→ build-test-cuda (GPU - self-hosted)
  │   ├─ needs: [test]
  │   ├─ Conditional: workflow_dispatch || main || schedule
  │   └─ GPU kernel tests
  │
  └─→ crossval-cuda (GPU - self-hosted)
      ├─ needs: [test]
      ├─ Conditional: workflow_dispatch || main || schedule
      └─ Full CUDA cross-validation
```

### Performance Benchmarks (Main branch only)
```
  └─→ benchmark (NON-BLOCKING, observability)
      ├─ Conditional: github.event == 'push' && main
      ├─ Runs: cargo bench --all-features
      ├─ Stores: Criterion reports
      └─ Auto-push to gh-pages (alert at 105% threshold)
```

---

## 4. Complete Job Dependency Map

### Jobs that run UNCONDITIONALLY on every PR/push:
1. ✅ `test` (primary gate)
2. ✅ `feature-matrix` (curated feature sets gate)
3. ✅ `doctest-matrix` (doctest validation gate)
4. ✅ `doctest` (CPU doctest gate)
5. ✅ `quality` (includes lychee link check)
6. ✅ `security` (cargo audit + deny)
7. ✅ `guard-fixture-integrity` (fixture checksums)
8. ✅ `guard-serial-annotations` (env isolation)
9. ✅ `guard-feature-consistency` (feature gates)
10. ✅ `env-mutation-guard` (no raw env mutations)
11. ⚠️ `guard-ignore-annotations` (non-blocking, observability)
12. ⚠️ `feature-hack-check` (non-blocking, observability)
13. ⚠️ `api-compat` (PR-only, non-blocking)
14. ⚠️ `ffi-smoke` (non-blocking observability)
15. ⚠️ `perf-smoke` (non-blocking observability)

### Jobs that run CONDITIONALLY:
- 🔷 `crossval-cpu-smoke` - On PR or main (fast smoke test)
- 🟦 `crossval-cpu` - Only on main or workflow_dispatch
- 🟦 `build-test-cuda` - Only on main/schedule or workflow_dispatch (GPU runner)
- 🟦 `crossval-cuda` - Only on main/schedule or workflow_dispatch (GPU runner)
- 🟦 `benchmark` - Only on main branch pushes
- 🟦 `ffi-zero-warning-windows` - Needs test (windows-latest)

### All jobs (27 total):
```
Primary: test
↓
├─ feature-hack-check (non-blocking)
├─ feature-matrix (gate)
├─ doctest-matrix (gate)
├─ guard-fixture-integrity (gate)
├─ guard-serial-annotations (gate)
├─ guard-feature-consistency (gate)
├─ guard-ignore-annotations (non-blocking)
├─ doctest (gate)
├─ perf-smoke (non-blocking)
├─ env-mutation-guard (gate)
├─ api-compat (PR-only, non-blocking)
├─ security (gate)
├─ ffi-smoke (non-blocking)
├─ ffi-zero-warning-windows (gate)
├─ quality (gate - includes lychee link check)
├─ crossval-cpu-smoke (PR/main conditional)
├─ crossval-cpu (main/dispatch only)
├─ build-test-cuda (GPU/main/dispatch)
├─ crossval-cuda (GPU/main/dispatch)
└─ benchmark (main/dispatch only)
```

---

## 5. Link Checking Strategy

### Current Implementation
- **Primary tool**: lychee (in `quality` job)
- **Config file**: `.lychee.toml`
- **Mode**: Offline (no external checks)
- **Scope**: All `**/*.md` files
- **Archive handling**: `docs/archive/` is **EXCLUDED** (not checked)

### Secondary Tools
- **markdown-link-check** (in `documentation-validation.yml`)
  - Runs independently from CI main workflow
  - Only on doc changes (separate trigger)
  - Checks external links (interactive mode)
  - **Continue-on-error**: implicit (non-blocking)

### What IS Checked
- ✅ All markdown files in: `docs/`, `crates/`, root level (`**/*.md`)
- ✅ Internal links (file references)
- ✅ Markdown syntax (via markdownlint in `quality` job)
- ✅ Code examples (doctests)

### What IS NOT Checked
- ❌ `docs/archive/` directory (explicitly excluded in .lychee.toml line 46)
- ❌ External HTTP(S) URLs (offline mode in CI)
- ❌ `.html` files (excluded from markdown linting, checked by lychee but depends on offline mode)

---

## 6. Feature Matrix Testing

### feature-hack-check (Non-blocking, observability)
```yaml
needs: test
continue-on-error: true
strategy:
  matrix: None (single job)
steps:
  - cargo hack check --feature-powerset --depth 2
  - cargo hack build --feature-powerset --depth 2 (lib only)
```

### feature-matrix (GATING - must pass)
```yaml
needs: test
strategy:
  matrix:
    features:
      - cpu
      - cpu,avx2
      - cpu,fixtures
      - cpu,avx2,fixtures
      - ffi
    include:
      - features: gpu
        compile-only: true
steps:
  - cargo build --no-default-features --features "${{ matrix.features }}"
  - cargo nextest run --features "${{ matrix.features }}" (unless compile-only)
  - cargo test --doc (unless compile-only)
```

**Profiles**:
- Standard: `cargo nextest run --profile ci`
- With fixtures: `cargo nextest run --profile fixtures`

### doctest-matrix (GATING - must pass)
```yaml
needs: test
strategy:
  matrix:
    features:
      - cpu
      - cpu,avx2
      - all-features
steps:
  - cargo test --doc --features "${{ matrix.features }}"
  - all-features: continue-on-error: true (GPU may not be available)
```

---

## 7. Guard Jobs (Quality Gates)

| Guard Job | Purpose | Config | Blocking |
|-----------|---------|--------|----------|
| `guard-fixture-integrity` | Validates GGUF fixture checksums, schema, alignment | `scripts/validate-fixtures.sh` | ✅ Yes (Gate) |
| `guard-serial-annotations` | Ensures #[serial(bitnet_env)] on env-mutating tests | `scripts/check-serial-annotations.sh` | ✅ Yes (Gate) |
| `guard-feature-consistency` | Cross-checks #[cfg(feature)] with Cargo.toml | `scripts/check-feature-gates.sh` | ✅ Yes (Gate) |
| `guard-ignore-annotations` | Ensures #[ignore] tests have issue refs | `scripts/check-ignore-annotations.sh` | ⚠️ No (Non-blocking) |
| `env-mutation-guard` | Detects raw std::env::{set_var,remove_var}() | ripgrep pattern search | ✅ Yes (Gate) |

---

## 8. Nextest Configuration

**Location**: `.config/nextest.toml`
**Profiles**:
- `default`: Full test suite, fail-fast, 300s timeout
- `ci`: 4 fixed threads, 300s timeout, no retries
- `fixtures`: 2 threads (I/O contention), 600s timeout
- `gpu`: 1 thread (GPU memory), 300s timeout
- `doctests`: num_cpus threads, 120s timeout

All profiles:
- `retries = 0` (no flaky test tolerance)
- `success-output = "never"` (reduce noise)
- `failure-output = "immediate"` (fast feedback)

---

## 9. Key Observations

### Link Checking
1. **Lychee is the primary link checker** (in `quality` job)
2. **Runs in offline mode** - no external URL validation in CI
3. **`docs/archive/` is explicitly excluded** from lychee checks
4. **markdown-link-check is separate** - runs only on doc-specific changes
5. **Both tools are non-blocking** in practical terms (no automated exclusion of bad PRs)

### Job Organization
1. **Test** is the foundational gate - 9 jobs depend on it
2. **Feature matrix** is comprehensive (6 tested combinations + 1 compile-only)
3. **Guard jobs** enforce environmental hygiene (4 blocking + 1 non-blocking)
4. **Cross-validation** runs only on main/dispatch (expensive)
5. **GPU jobs** require self-hosted runners (conditional)

### Non-Blocking Observability
- `feature-hack-check` - Full powerset analysis (expensive)
- `perf-smoke` - 4-token inference timing
- `api-compat` - API surface changes
- `ffi-smoke` - FFI build health
- `guard-ignore-annotations` - Unannotated #[ignore] tests
- `benchmark` - Performance tracking

### Gating (Hard Requirements)
- `test` (primary)
- `feature-matrix` (curated features)
- `doctest-matrix` (doc examples)
- `guard-fixture-integrity` (fixture integrity)
- `guard-serial-annotations` (env isolation)
- `guard-feature-consistency` (feature gates)
- `env-mutation-guard` (no raw env mutations)
- `doctest` (CPU examples)
- `quality` (including lychee)
- `security` (CVE/license audit)
- `ffi-zero-warning-windows` (MSVC warnings)

---

## 10. Critical Issues

### Issue #1: docs/archive/ Excluded from Link Checks
**Severity**: Medium
**Description**: The `.lychee.toml` configuration explicitly excludes `docs/archive/` from link validation
**Impact**: Broken internal links in archived documentation won't be caught
**Location**: `.lychee.toml`, line 46
**Justification**: "Historical documentation - not maintained (archived 2025-10-23)"

### Issue #2: Offline Mode for Link Checking
**Severity**: Low
**Description**: Lychee runs in `offline = true` mode, skipping external URL validation
**Impact**: External links (e.g., to GitHub docs, examples) aren't validated in CI
**Rationale**: "offline mode for CI performance"
**Mitigation**: `markdown-link-check` validates external links but runs separately and is non-blocking

### Issue #3: markdown-link-check is Non-Blocking
**Severity**: Low
**Description**: The `documentation-validation.yml` workflow uses `markdown-link-check` with implicit continue-on-error
**Impact**: Documentation link validation runs separately and doesn't block merges
**Triggering**: Only on doc-specific changes or manual trigger

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Primary Link Checker** | Lychee (offline mode) in `quality` job |
| **Config File** | `.lychee.toml` |
| **Excluded Paths** | `docs/archive/`, `target/`, `vendor/`, `.git/`, etc. |
| **Archive Handling** | Excluded from checks (archived 2025-10-23) |
| **Total CI Jobs** | 20 total jobs (9 gates + 11 non-blocking/conditional) |
| **Mandatory Gates** | 10 jobs must pass for merge |
| **Nextest Profile** | `ci` = 4 threads, 300s timeout, no retries |
| **Feature Matrix** | 6 tested combinations (cpu, cpu+avx2, cpu+fixtures, cpu+avx2+fixtures, ffi, gpu compile-only) |
| **Guard Jobs** | 5 jobs (4 gating + 1 non-blocking) |
| **Workflow Conditional** | crossval-cpu-smoke on PR/main; others only on main/dispatch |
