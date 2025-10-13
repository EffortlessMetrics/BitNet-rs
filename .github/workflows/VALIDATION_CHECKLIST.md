# Validation Workflow Checklist

Quick reference checklist for the validation workflow.

## Pre-Merge Checklist (for Contributors)

### Before Creating PR

- [ ] Test validation tools locally with strict mode:
  ```bash
  export BITNET_STRICT_MODE=1
  export BITNET_DETERMINISTIC=1
  export BITNET_SEED=42
  export RAYON_NUM_THREADS=1
  ```

- [ ] Build all validation tools:
  ```bash
  cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
  cargo build -p bitnet-st2gguf --release
  cargo build -p bitnet-st-tools --release
  ```

- [ ] Run validation tests locally:
  ```bash
  cargo test -p bitnet-cli --test validation_workflow \
    --no-default-features --features cpu,full-cli -- --nocapture
  ```

- [ ] If modifying models, validate without corrections:
  ```bash
  cargo run -p bitnet-cli -- inspect --ln-stats --json your-model.gguf
  # Check that status is "ok" and suspicious counts are 0
  ```

- [ ] Verify no correction flags in your changes:
  ```bash
  git diff main | grep -i "BITNET_ALLOW_RUNTIME_CORRECTIONS"
  git diff main | grep -i "BITNET_CORRECTION_POLICY"
  git diff main | grep -i "BITNET_FIX_LN_SCALE"
  # Should return nothing
  ```

### After PR Creation

- [ ] Check that validation workflow runs automatically
- [ ] Verify security-guard passes (no correction flags detected)
- [ ] Verify build-tools passes on all platforms (Ubuntu, Windows, macOS)
- [ ] Verify validation-tests passes
- [ ] Verify validate-models passes (or is skipped if appropriate)
- [ ] Check quality-gate status (must be green for merge)

### If Validation Fails

- [ ] Review workflow logs in GitHub Actions
- [ ] Check job summaries for specific failures
- [ ] Download artifacts for detailed reports
- [ ] Fix issues locally and test before pushing
- [ ] Re-run failed jobs if needed

## CI Maintenance Checklist (for Maintainers)

### Regular Maintenance

- [ ] Review validation workflow performance monthly
- [ ] Check cache effectiveness (aim for < 10 minute builds)
- [ ] Update Rust toolchain versions as needed
- [ ] Verify all platforms still supported
- [ ] Review artifact retention policies

### When Adding New Validation Features

- [ ] Add integration tests to `validation_workflow.rs`
- [ ] Update security-guard checks if new env vars added
- [ ] Document new features in `docs/development/validation-ci.md`
- [ ] Add examples to workflow summary
- [ ] Test locally on all platforms if possible

### When Updating Dependencies

- [ ] Test validation tools still build
- [ ] Run validation tests with new dependencies
- [ ] Check for breaking changes in actions (Swatinem/rust-cache, etc.)
- [ ] Update cache keys if dependency structure changes

### Security Reviews

- [ ] Verify correction flags still blocked in security-guard
- [ ] Check no new correction mechanisms bypass security-guard
- [ ] Review workflow permissions (minimal required)
- [ ] Audit artifact contents (no secrets leaked)

## Troubleshooting Checklist

### Security Guard Fails

- [ ] Check for `BITNET_ALLOW_RUNTIME_CORRECTIONS` in workflow files
- [ ] Check for `BITNET_CORRECTION_POLICY` in workflow files
- [ ] Check for `BITNET_FIX_LN_SCALE` in workflow files
- [ ] Verify `BITNET_STRICT_MODE=1` present in validation.yml
- [ ] Remove any correction flags from workflow files

### Build Tools Fails

- [ ] Check compilation errors in logs
- [ ] Verify features are correct: `--no-default-features --features cpu,full-cli`
- [ ] Test local build with same features
- [ ] Check Cargo.lock is up to date
- [ ] Verify dependencies resolve correctly

### Validation Tests Fail

- [ ] Review test output with `--nocapture`
- [ ] Check if validation logic changed (update tests if intentional)
- [ ] Verify test fixtures/models are available
- [ ] Check if model paths in tests are correct
- [ ] Test locally to reproduce failure

### Model Validation Fails

- [ ] Check if model has suspicious LayerNorm weights
- [ ] Verify model was exported with proper LayerNorm preservation
- [ ] Check if model exists at expected path
- [ ] Verify expected ruleset matches detected ruleset
- [ ] Regenerate model with `bitnet-st2gguf --strict` if needed

### Performance Issues

- [ ] Check cache hit rates in logs
- [ ] Verify Swatinem/rust-cache is working
- [ ] Check if incremental compilation disabled (`CARGO_INCREMENTAL=0`)
- [ ] Review parallel execution (build-tools and validation-tests should run in parallel)
- [ ] Consider skipping model validation for tooling-only changes

## Quick Commands

### View Workflow Status

```bash
# List recent runs
gh run list --workflow=validation.yml --limit 10

# View specific run
gh run view <run-id>

# Watch current run
gh run watch
```

### Download Artifacts

```bash
# Download all artifacts from specific run
gh run download <run-id>

# Download specific artifact
gh run download <run-id> -n ubuntu-latest-validation-tools
```

### Manual Workflow Dispatch

```bash
# Run validation workflow manually
gh workflow run validation.yml

# Skip model validation (for tooling changes)
gh workflow run validation.yml -f skip_model_validation=true
```

### Local Validation Commands

```bash
# Set environment to match CI
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Build tools
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
cargo build -p bitnet-st2gguf --release
cargo build -p bitnet-st-tools --release

# Test tools execute
./target/release/bitnet --version
./target/release/st2gguf --help
./target/release/st-ln-inspect --help
./target/release/st-merge-ln-f16 --help

# Run validation tests
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli -- --nocapture

# Validate a model
./target/release/bitnet inspect --ln-stats --json your-model.gguf

# Check JSON output
./target/release/bitnet inspect --ln-stats --json your-model.gguf | \
  jq '{status, ruleset, layernorm, projection}'
```

### Security Checks

```bash
# Check for correction flags in workflows
rg -n 'BITNET_ALLOW_RUNTIME_CORRECTIONS' .github/workflows
rg -n 'BITNET_CORRECTION_POLICY' .github/workflows
rg -n 'BITNET_FIX_LN_SCALE' .github/workflows

# Should return no results (or only in comments)
```

## Quick Reference

### Required Jobs (Must Pass)

- ✅ `security-guard` - Critical
- ✅ `build-tools` - Critical
- ✅ `validation-tests` - Critical
- ✅ `quality-gate` - Critical

### Optional Jobs (Can Skip)

- ⚠️ `validate-models` - Optional (skip with workflow_dispatch input)

### Typical Timeline

```
0-1 min:   security-guard
1-10 min:  build-tools (parallel across 3 platforms)
10-15 min: validation-tests + validate-models (parallel)
15-20 min: validation-summary + quality-gate
Total:     15-25 minutes with caching
```

### Key Environment Variables

```bash
BITNET_STRICT_MODE=1              # Required - fail on suspicious weights
BITNET_DETERMINISTIC=1            # Required - reproducible tests
BITNET_SEED=42                    # Required - deterministic seed
RAYON_NUM_THREADS=1               # Required - single-threaded

# BLOCKED in CI (security-guard enforces):
# BITNET_ALLOW_RUNTIME_CORRECTIONS  ❌
# BITNET_CORRECTION_POLICY          ❌
# BITNET_FIX_LN_SCALE               ❌ (deprecated)
```

### Monitored Paths

```
crates/bitnet-cli/**
crates/bitnet-st-tools/**
crates/bitnet-st2gguf/**
crates/bitnet-models/**
scripts/validate_gguf.sh
scripts/export_clean_gguf.sh
.github/workflows/validation.yml
```

## Documentation Links

- **Full Documentation**: `docs/development/validation-ci.md`
- **Workflow Diagram**: `.github/workflows/VALIDATION_WORKFLOW_DIAGRAM.md`
- **Workflow Summary**: `.github/workflows/VALIDATION_WORKFLOW_SUMMARY.md`
- **Feature Spec**: `docs/features/validation-workflow.md`
- **Correction Policy**: `docs/explanation/correction-policy.md`

## Common Exit Codes

- `0` - Success (all checks passed)
- `1` - General failure
- `8` - Suspicious weights detected in strict mode (from bitnet-cli inspect)

## Support

For issues with validation workflow:

1. ✅ Check this checklist
2. ✅ Review workflow logs
3. ✅ Test locally with provided commands
4. ✅ Check full documentation
5. ✅ Open issue with workflow run ID and error details

---

Last updated: 2025-10-13
Workflow version: validation.yml (initial release)
