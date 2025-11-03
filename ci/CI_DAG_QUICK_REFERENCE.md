# CI DAG & Git Hooks - Quick Reference

## Job Dependency Tree (One-liner view)

```
test (PRIMARY GATE)
├─ feature-matrix (gates) ─→ feature-hack-check (observe)
├─ doctest-matrix (gates)
├─ doctest (gates)
├─ ffi-smoke (gates) ─→ ffi-zero-warning-* (gates)
├─ perf-smoke (observe)
├─ crossval-cpu-smoke (gates) ─→ crossval-cpu (main only, gates)
├─ build-test-cuda (GPU, gates)
└─ crossval-cuda (GPU, main only, gates)

INDEPENDENT GUARDS (run in parallel):
├─ guard-fixture-integrity (gates)
├─ guard-serial-annotations (gates)
├─ guard-feature-consistency (gates)
├─ guard-ignore-annotations (observe)
└─ env-mutation-guard (gates)

ALWAYS-RUN JOBS:
├─ api-compat (PR only)
├─ security
├─ quality
└─ benchmark (main only)
```

## Gating Guards (Must Pass for Merge)

| Guard | Ripgrep? | Checks |
|-------|----------|--------|
| guard-fixture-integrity | No | SHA256 checksums, GGUF magic/version, alignment |
| guard-serial-annotations | Yes | `EnvGuard::new` has `#[serial(bitnet_env)]` |
| guard-feature-consistency | Yes | Defined features match `#[cfg(feature = "...")]` |
| env-mutation-guard | Yes | No raw `std::env::set_var/remove_var` |

## Pre-commit Hook Checks

**Location**: `.githooks/pre-commit`  
**Enable**: `git config core.hooksPath .githooks`  
**Tool**: ripgrep (rg)

```bash
# Check 1: Bare #[ignore] annotations
rg -n -P '#\[ignore\](?!\s*=)' --glob '*.rs' crates tests tests-new xtask
# Prevents: #[ignore]
# Requires: #[ignore = "reason"] or comment before

# Check 2: Raw environment mutations
rg -n '(std::env::set_var|std::env::remove_var)\(' --glob '*.rs' \
  --glob '!**/tests/helpers/**' --glob '!**/support/**' --glob '!**/env_guard.rs' \
  crates tests tests-new xtask
# Prevents: direct env::set_var/remove_var in tests
# Requires: EnvGuard + #[serial(bitnet_env)]
```

## Ripgrep Usage Summary

| File | Ripgrep Patterns | Purpose |
|------|------------------|---------|
| `.githooks/pre-commit` | Negative lookahead, literal matching | Local enforcement |
| `scripts/check-serial-annotations.sh` | `EnvGuard\|\|temp_env` + backtrack 5 | Verify annotation pattern |
| `scripts/check-feature-gates.sh` | Feature definition extraction + `#[cfg...]` | Cross-check feature usage |
| `scripts/check-ignore-annotations.sh` | `#\[ignore\]` + context extraction | Verify issue references |
| `ci.yml:env-mutation-guard` | Inline ripgrep (no script) | CI enforcement |

## Key Metrics

- **Total Jobs**: 18 named jobs (excluding matrix expansion)
- **Blocking Gates**: 9 jobs
- **Non-blocking Observers**: 4 jobs  
- **Guard Jobs**: 5 independent jobs
- **Primary Gate**: `test` (blocks all downstream)
- **Ripgrep Usage**: 4 guard scripts + 1 pre-commit hook

## Hygiene Issues & Fixes

| Issue | Severity | Fix |
|-------|----------|-----|
| `doctest` missing explicit `needs: test` | Low | Add `needs: test` declaration |
| Ripgrep installed per-guard | Low | Centralize in `test` job or use cached artifact |
| `feature-hack-check` non-blocking | Medium | Consider blocking depth-1/2 powerset |
| Guards only run on Ubuntu | Medium | Run all guards on primary platform only |
| Fixture validation rebuilds CLI | Low | Cache `bitnet-cli` binary across jobs |
| Pre-commit hook not in CI | Low | Mirror all pre-commit checks in CI guards |

## Job Execution Times (Tier 0)

1. **test** (parallel across 6 configurations): ~15-20 min
2. **feature-matrix** (6 configurations): ~10-15 min each in parallel
3. **doctest-matrix** (3 configurations): ~5-10 min each in parallel
4. **Guards** (5 independent): ~1-3 min each in parallel
5. **crossval-cpu-smoke**: ~5-10 min (depends on model download)

**Total Critical Path**: ~20-30 min (test → feature-matrix/doctest-matrix)  
**Total Parallel Wall Time**: ~30-50 min (including all guards)

## Recommended Reading Order

1. `.github/workflows/ci.yml` (full DAG structure)
2. `.githooks/pre-commit` (local enforcement)
3. `ci/CI_DAG_HYGIENE_AND_HOOKS_ANALYSIS.md` (detailed analysis)
4. `scripts/check-*.sh` (individual guard implementations)
