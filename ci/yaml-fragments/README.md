# CI YAML Fragments - Feature Matrix Testing and Guards

This directory contains CI job definitions for SPEC-2025-006: Feature Matrix Testing and CI Guards.

## Overview

These YAML fragments are ready to insert into `.github/workflows/ci.yml`. They implement:

1. **Feature Matrix Testing** - Comprehensive feature combination testing
2. **CI Guards** - Quality gates for test annotations and fixture integrity

## Files

### Feature Matrix Testing

- **feature-hack-check.yml** - cargo-hack powerset check (non-blocking observability)
  - Tests all feature combinations at depth=2 (~700 combos)
  - Runtime: ~12 minutes
  - Non-blocking (continue-on-error: true)

- **feature-matrix.yml** - Curated feature matrix (gating)
  - Tests 5 critical feature combos: cpu, cpu+avx2, cpu+fixtures, cpu+avx2+fixtures, ffi
  - Plus GPU compile-only check
  - Runtime: ~8 minutes (parallel)
  - **Gates CI** - must pass for merge

- **doctest-matrix.yml** - Doctest feature matrix (gating)
  - Tests documentation examples with 3 feature combos: cpu, cpu+avx2, all-features
  - Runtime: ~5 minutes (parallel)
  - **Gates CI** - must pass for merge (except all-features)

### CI Guards

- **guard-ignore-annotations.yml** - Unannotated #[ignore] detector
  - Ensures all ignored tests have issue reference or justification
  - Runtime: ~30 seconds

- **guard-fixture-integrity.yml** - Fixture integrity validator
  - Validates checksums, schema, and alignment
  - Runtime: ~2 minutes

- **guard-serial-annotations.yml** - Serial annotation validator
  - Ensures env-mutating tests have #[serial(bitnet_env)]
  - Runtime: ~30 seconds

- **guard-feature-consistency.yml** - Feature gate consistency checker
  - Cross-checks #[cfg(feature = "...")] with defined features
  - Runtime: ~30 seconds

## Integration Instructions

### Step 1: Insert Jobs into .github/workflows/ci.yml

Add the following jobs to `.github/workflows/ci.yml` after the existing `test` job:

```yaml
jobs:
  # ... existing test job ...

  # Insert feature matrix testing jobs
  # (paste contents of feature-hack-check.yml here)
  # (paste contents of feature-matrix.yml here)
  # (paste contents of doctest-matrix.yml here)

  # Insert guard jobs
  # (paste contents of guard-ignore-annotations.yml here)
  # (paste contents of guard-fixture-integrity.yml here)
  # (paste contents of guard-serial-annotations.yml here)
  # (paste contents of guard-feature-consistency.yml here)
```

### Step 2: Verify Scripts Exist

Ensure the guard scripts are executable:

```bash
chmod +x scripts/check-ignore-annotations.sh
chmod +x scripts/validate-fixtures.sh
chmod +x scripts/check-serial-annotations.sh
chmod +x scripts/check-feature-gates.sh
```

### Step 3: Verify Nextest Profiles

Ensure `.config/nextest.toml` contains the new profiles:

- `[profile.fixtures]` - For fixture-heavy tests
- `[profile.gpu]` - For GPU kernel tests
- `[profile.doctests]` - For documentation examples

### Step 4: Test Locally

Before enabling in CI, test locally:

```bash
# Test guard scripts
bash scripts/check-ignore-annotations.sh
bash scripts/validate-fixtures.sh
bash scripts/check-serial-annotations.sh
bash scripts/check-feature-gates.sh

# Test curated feature matrix
cargo nextest run --no-default-features --features cpu,avx2,fixtures --profile fixtures

# Test cargo-hack (requires installation)
cargo install cargo-hack --locked
cargo hack check --feature-powerset --depth 2 --workspace --exclude xtask
```

## Performance Budget

**Baseline CI Time**: ~6 minutes (primary test job)

**New Jobs**:
- feature-hack-check: ~12 minutes (non-blocking, parallel)
- feature-matrix: ~8 minutes (gating, parallel)
- doctest-matrix: ~5 minutes (gating, parallel)
- guard jobs: ~4 minutes total (gating, parallel)

**Total CI Time**: ~8 minutes (longest gating job)
**Increase**: +2 minutes (+33% vs. baseline)

## Gradual Rollout

### Phase 1: Foundation (Week 1)
- ✅ Create guard scripts
- ✅ Create nextest profiles
- ✅ Create YAML fragments
- [ ] Test locally
- [ ] Enable feature-matrix and feature-hack-check in CI
- [ ] Measure CI time impact

### Phase 2: Guard Jobs (Week 2)
- [ ] Fix any violations found by guards
- [ ] Enable guard jobs in CI
- [ ] Verify no false positives

### Phase 3: Doctest Matrix (Week 3)
- [ ] Fix any failing doc examples
- [ ] Enable doctest-matrix in CI
- [ ] Verify all-features runs (continue-on-error)

### Phase 4: Optimization (Week 4)
- [ ] Profile CI job runtimes
- [ ] Optimize slow paths
- [ ] Document CI architecture

## Troubleshooting

### False Positives in Guards

If a guard script reports false positives:

1. Check the pattern matching logic in the script
2. Add escape hatches if needed (e.g., `// guard-ignore: <reason>`)
3. Update the script with better detection logic

### CI Time Exceeds Budget

If CI time exceeds +3 minutes:

1. Consider reducing cargo-hack depth to 1
2. Run cargo-hack as non-blocking only
3. Profile jobs to identify bottlenecks
4. Use GitHub Actions matrix parallelism

### Feature Gate Bugs Slip Through

If feature gate bugs are not caught:

1. Check cargo-hack output for failures
2. Consider making cargo-hack blocking (remove continue-on-error)
3. Add failing combo to curated matrix

## References

- **SPEC-2025-006**: Feature Matrix Testing and CI Guards
- **CLAUDE.md**: Test status and scaffolding documentation
- **cargo-hack**: https://github.com/taiki-e/cargo-hack
- **nextest**: https://nexte.st/

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-23 | 1.0.0 | Initial YAML fragments and guard scripts |
