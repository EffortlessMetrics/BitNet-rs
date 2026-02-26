# Validation Workflow Summary

## Overview

The `validation.yml` workflow implements comprehensive validation gates for bitnet-rs model operations and validation tooling.

## Key Features

### 1. Security Guard
- **Blocks correction flags in CI** (BITNET_ALLOW_RUNTIME_CORRECTIONS, BITNET_CORRECTION_POLICY)
- Verifies strict mode is enabled
- Fails immediately if forbidden flags are detected

### 2. Multi-Platform Tool Builds
- Builds validation tools on Ubuntu, Windows, macOS
- Tools: bitnet-cli (full-cli), bitnet-st2gguf, bitnet-st-tools
- Verifies binaries execute correctly
- Uploads artifacts for downstream jobs

### 3. Integration Testing
- Runs validation workflow tests (`validation_workflow.rs`)
- Tests inspect command, LayerNorm validation, architecture detection
- Validates JSON output format and exit codes
- Coverage across all platforms

### 4. Model Validation
- Validates GGUF models with strict mode enabled
- Checks LayerNorm RMS values for suspicious weights
- Verifies correct ruleset detection
- Can be skipped for tooling-only changes

### 5. Quality Gate
- Final gate that blocks PR merge on failure
- Always runs, even if previous jobs fail
- Provides detailed troubleshooting guidance

## Integration with Existing CI

### Complements Main CI (`ci.yml`)
- Main CI: Broad test suite including crossval
- Validation workflow: Focused on validation tooling
- Both: Enforce strict mode and block corrections

### Extends GGUF Build Workflow (`gguf_build_and_validate.yml`)
- GGUF Build: Exports and validates new models
- Validation workflow: Validates existing models and tools
- Both: Use strict mode, block corrections

### Reinforces Guards (`guards.yml`)
- Guards: Broader checks for scripts and workflows
- Validation workflow: Validation-specific security checks
- Both: Check for forbidden correction flags

## Environment Configuration

```bash
# Strict validation (fail on suspicious weights)
BITNET_STRICT_MODE=1

# Deterministic inference
BITNET_DETERMINISTIC=1
BITNET_SEED=42
RAYON_NUM_THREADS=1

# Blocked flags (security-guard enforces)
# BITNET_ALLOW_RUNTIME_CORRECTIONS - NOT ALLOWED
# BITNET_CORRECTION_POLICY - NOT ALLOWED
# BITNET_FIX_LN_SCALE - NOT ALLOWED (deprecated)
```

## Trigger Conditions

- **Push** to main/develop (validation-related paths)
- **Pull Request** to main/develop (validation-related paths)
- **Manual Dispatch** with optional skip_model_validation

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

## Job Dependencies

```
security-guard (blocks everything on failure)
    ↓
build-tools (parallel across platforms)
    ↓
validation-tests + validate-models (parallel)
    ↓
validation-summary (aggregates results)
    ↓
quality-gate (blocks PR merge on failure)
```

## Success Criteria

For PR merge, these jobs must pass:

- ✅ `security-guard` - Critical
- ✅ `build-tools` - Critical
- ✅ `validation-tests` - Critical
- ⚠️ `validate-models` - Optional (may be skipped)
- ✅ `quality-gate` - Critical

## Artifacts Generated

1. **{os}-validation-tools** (7 days)
   - Built binaries for bitnet-cli, st2gguf, st-tools

2. **validation-test-report-{os}** (30 days)
   - Test execution reports per platform

3. **model-validation-{model_name}** (30 days)
   - Validation JSON output and reports

## Common Use Cases

### PR Workflow
1. Developer makes changes to validation code
2. Workflow triggers automatically on PR
3. Security guard checks for forbidden flags
4. Tools build across platforms
5. Integration tests run
6. Models validated (if applicable)
7. Quality gate blocks merge if validation fails

### Manual Testing
```bash
# Trigger workflow manually
gh workflow run validation.yml

# Skip model validation for tooling changes
gh workflow run validation.yml -f skip_model_validation=true
```

### Local Validation
```bash
# Match CI environment
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Build tools
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
cargo build -p bitnet-st2gguf --release
cargo build -p bitnet-st-tools --release

# Run tests
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli

# Validate model
cargo run -p bitnet-cli -- inspect --ln-stats --json your-model.gguf
```

## Troubleshooting

### Security Guard Fails
- **Problem**: Correction flags detected in workflow files
- **Solution**: Remove BITNET_ALLOW_RUNTIME_CORRECTIONS, BITNET_CORRECTION_POLICY from workflows

### Build Tools Fails
- **Problem**: Compilation errors
- **Solution**: Test locally with same features: `--no-default-features --features cpu,full-cli`

### Validation Tests Fail
- **Problem**: Integration test failures
- **Solution**: Run tests locally with `--nocapture` to see detailed output

### Model Validation Fails
- **Problem**: Suspicious LayerNorm weights detected
- **Solution**: Regenerate model with `bitnet-st2gguf --strict` or `just model-clean`

## Performance

- **Typical Duration**: 15-25 minutes (with caching)
- **Parallel Execution**: build-tools and validation-tests run in parallel
- **Caching**: Cargo dependencies cached per OS
- **Optimization**: Skip model validation for tooling-only changes

## Best Practices

### For Contributors
1. Test locally with strict mode before pushing
2. Run validation tests: `cargo test -p bitnet-cli --test validation_workflow`
3. Ensure models pass inspection without corrections

### For Maintainers
1. Never disable strict mode in CI
2. Keep validation fast with aggressive caching
3. Provide clear error messages and troubleshooting guidance
4. Block correction flags in security-guard

## Documentation

- **Full Documentation**: `docs/development/validation-ci.md`
- **Feature Spec**: `docs/features/validation-workflow.md`
- **Correction Policy**: `docs/explanation/correction-policy.md`
- **GGUF Export**: `docs/howto/export-clean-gguf.md`

## Support

For issues:
1. Review workflow logs in GitHub Actions
2. Check `docs/development/validation-ci.md` for detailed troubleshooting
3. Test locally with provided commands
4. Open issue with workflow run ID and error details
