# CI DAG Dependencies Quick Reference

**Last Updated**: 2025-10-23

## Dependency Graph

### Level 0: Independent Gates (No Dependencies)

Run immediately in parallel:

```yaml
test:                         # Primary test suite
guard-fixture-integrity:      # Fixture validation
guard-serial-annotations:     # Serial annotation checks
guard-feature-consistency:    # Feature gate consistency
guard-ignore-annotations:     # Ignore annotation validation
env-mutation-guard:          # Environment mutation checks
security:                    # Security audit
quality:                     # Code quality checks
```

### Level 1: Depends on `test`

Run after `test` passes:

```yaml
# Feature validation
feature-matrix:              needs: [test]
doctest-matrix:              needs: [test]
doctest:                     needs: [test]

# FFI validation
ffi-smoke:                   needs: [test]
ffi-zero-warning-windows:    needs: [test]
ffi-zero-warning-linux:      needs: [test]

# Cross-validation
crossval-cpu-smoke:          needs: [test]

# Performance
perf-smoke:                  needs: [test]

# Conditional jobs
api-compat:                  needs: [test]  # PR only, non-blocking
crossval-cpu:                needs: [test]  # main/dispatch only
build-test-cuda:             needs: [test]  # main/dispatch/schedule only
crossval-cuda:               needs: [test]  # main/dispatch/schedule only
benchmark:                   needs: [test]  # main only, non-blocking
```

### Level 2: Depends on `test` + `feature-matrix`

Run after both `test` and `feature-matrix` pass:

```yaml
feature-hack-check:          needs: [test, feature-matrix]  # Non-blocking
```

## Non-Blocking Observers

These jobs run but don't block CI:

```yaml
feature-hack-check:          continue-on-error: true
api-compat:                  continue-on-error: true
benchmark:                   continue-on-error: true
perf-smoke:                  continue-on-error: true (within steps)
```

## Conditional Execution

### PR Only
- `api-compat`: `github.event_name == 'pull_request'`

### Main Branch Only
- `benchmark`: `github.event_name == 'push' && github.ref == 'refs/heads/main'`

### Main Branch or Manual Dispatch
- `crossval-cpu`: `github.event_name == 'workflow_dispatch' || github.ref == 'refs/heads/main'`

### Main Branch, Manual, or Scheduled
- `build-test-cuda`: `github.event_name == 'workflow_dispatch' || github.ref == 'refs/heads/main' || github.event_name == 'schedule'`
- `crossval-cuda`: Same as above

## Fail-Fast Behavior

### Scenario: `test` fails

**Result**: All Level 1+ jobs are skipped

**Continues running**:
- `guard-fixture-integrity`
- `guard-serial-annotations`
- `guard-feature-consistency`
- `guard-ignore-annotations`
- `env-mutation-guard`
- `security`
- `quality`

**Skipped**:
- All Level 1 jobs
- All Level 2 jobs

### Scenario: `feature-matrix` fails

**Result**: Only `feature-hack-check` is skipped

**Continues running**:
- All Level 0 jobs
- All other Level 1 jobs

## Adding New Jobs

### Independent Gate (Parallel)

```yaml
guard-new-check:
  name: Guard - New Check
  runs-on: ubuntu-latest
  # No needs: dependency - runs immediately
  steps:
    - uses: actions/checkout@v4
    - run: ./scripts/new-check.sh
```

### Dependent Gate (After Tests)

```yaml
new-validation:
  name: New Validation Check
  runs-on: ubuntu-latest
  needs: [test]  # Wait for primary tests
  steps:
    - uses: actions/checkout@v4
    - run: ./scripts/new-validation.sh
```

### Observer (Non-Blocking)

```yaml
new-observer:
  name: New Observability Check
  runs-on: ubuntu-latest
  needs: [test]  # Wait for primary tests
  continue-on-error: true  # Don't block CI
  steps:
    - uses: actions/checkout@v4
    - run: ./scripts/new-observer.sh
```

## Verification Commands

```bash
# List all jobs
rg "^  [a-z-]+:" .github/workflows/ci.yml | sed 's/:$//' | awk '{print $1}'

# Show all dependencies
rg "^\s+needs:" .github/workflows/ci.yml

# Find non-blocking jobs
rg "continue-on-error: true" .github/workflows/ci.yml

# Show conditional jobs
rg "^\s+if:" .github/workflows/ci.yml
```

## Related Documentation

- `ci/CI_DAG_OPTIMIZATION_SUMMARY.md` - Detailed optimization rationale
- `ci/CI_EXPLORATION_SUMMARY.md` - Full CI architecture
- `ci/CI_DAG_HYGIENE_AND_HOOKS_ANALYSIS.md` - Original analysis
