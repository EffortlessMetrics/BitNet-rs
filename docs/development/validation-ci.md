# Validation CI Workflow

Comprehensive documentation for the BitNet-rs validation CI workflow (`.github/workflows/validation.yml`).

## Overview

The validation workflow enforces quality standards for all model operations and validation tooling in BitNet-rs. It ensures that:

1. **Security**: No runtime correction flags are enabled in CI
2. **Tooling**: All validation tools build and execute correctly across platforms
3. **Testing**: Integration tests pass for validation workflows
4. **Models**: GGUF models pass strict validation without corrections

## Workflow Structure

### Trigger Conditions

The workflow runs on:

- **Push** to `main` or `develop` branches (affecting validation-related paths)
- **Pull requests** to `main` or `develop` branches (affecting validation-related paths)
- **Manual dispatch** with optional `skip_model_validation` input

### Monitored Paths

```yaml
paths:
  - "crates/bitnet-cli/**"
  - "crates/bitnet-st-tools/**"
  - "crates/bitnet-st2gguf/**"
  - "crates/bitnet-models/**"
  - "scripts/validate_gguf.sh"
  - "scripts/export_clean_gguf.sh"
  - ".github/workflows/validation.yml"
```

## Environment Configuration

### Global Environment Variables

```bash
CARGO_TERM_COLOR=always
RUST_BACKTRACE=1
CARGO_INCREMENTAL=0
RUSTFLAGS="-D warnings"

# Strict validation mode - fail on suspicious weights
BITNET_STRICT_MODE=1

# Deterministic inference for reproducible tests
BITNET_DETERMINISTIC=1
BITNET_SEED=42
RAYON_NUM_THREADS=1

# Git metadata for vergen-gix
VERGEN_GIT_SHA=${{ github.sha }}
VERGEN_GIT_BRANCH=${{ github.ref_name }}
VERGEN_GIT_DESCRIBE=${{ github.ref_name }}-${{ github.sha }}
VERGEN_IDEMPOTENT=1
```

### Security Configuration

**Blocked Environment Variables** (enforced by `security-guard` job):

- `BITNET_ALLOW_RUNTIME_CORRECTIONS` - Must not be set in CI
- `BITNET_CORRECTION_POLICY` - Must not be set in CI
- `BITNET_FIX_LN_SCALE` - Deprecated, must not be set in CI

These flags are blocked to ensure models pass validation without runtime corrections. Runtime corrections are only allowed for known-bad models in local development with explicit fingerprinting.

## Jobs

### 1. Security Guard (`security-guard`)

**Purpose**: Block correction flags and verify strict mode configuration

**Key Checks**:

- Scans workflow files for forbidden correction environment variables
- Verifies `BITNET_STRICT_MODE=1` is enabled in validation workflow
- Fails immediately if any correction flags are found

**Why This Matters**:

Runtime corrections mask underlying issues. CI must validate models with their actual weights, not corrected versions. This ensures:

- Models exported with proper LayerNorm weights (F16/F32, not quantized)
- Detection of suspicious weights early in development
- No accidental use of correction policies in production

**Exit Codes**:

- `0` - No forbidden flags found, strict mode verified
- `1` - Forbidden flags detected or strict mode not enabled

### 2. Build Tools (`build-tools`)

**Purpose**: Build validation tools across all platforms

**Tools Built**:

1. **bitnet-cli** (with `--features cpu,full-cli`)
   - Main CLI with inspection and validation commands
   - Includes `inspect` command with LayerNorm validation

2. **bitnet-st2gguf**
   - SafeTensors to GGUF converter
   - Preserves LayerNorm weights in float format

3. **bitnet-st-tools**
   - `st-ln-inspect`: Inspect LayerNorm weights in SafeTensors
   - `st-merge-ln-f16`: Merge F16 LayerNorm weights into SafeTensors

**Platform Matrix**:

- Ubuntu (Linux x86_64)
- Windows (x86_64)
- macOS (x86_64)

**Verification Steps**:

1. Build all tools in release mode
2. Verify binary files exist at expected paths
3. Execute each binary to confirm they run (version/help commands)
4. Upload binaries as artifacts for downstream jobs

**Artifacts**:

- `{os}-validation-tools` - Contains all built validation binaries
- Retention: 7 days

### 3. Validation Tests (`validation-tests`)

**Purpose**: Run integration tests for validation workflows

**Tests Executed**:

1. **Validation Workflow Tests** (`validation_workflow.rs`)
   - Basic inspect command invocation
   - LayerNorm RMS validation
   - Architecture detection and ruleset selection
   - Gate modes (auto, none, policy)
   - JSON output format validation
   - Exit code verification in strict mode
   - Error handling for missing/corrupted files

2. **Inspect Tests** (`inspect_ln_stats.rs`)
   - LayerNorm tensor identification
   - Projection weight validation
   - Quantized tensor handling
   - Text and JSON output formats

**Platform Coverage**: Ubuntu, Windows, macOS

**Test Features**:

```bash
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli \
  -- --nocapture
```

**Artifacts**:

- `validation-test-report-{os}` - Test execution report
- Retention: 30 days

### 4. Validate Models (`validate-models`)

**Purpose**: Validate GGUF models with strict mode enabled

**Model Matrix**:

| Model | Path | Expected Ruleset |
|-------|------|------------------|
| BitNet-I2S-2B | `models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` | `bitnet-b1.58:i2_s` |
| Clean-F16 | `models/clean/clean-f16.gguf` | `generic` |

**Validation Process**:

1. Download validation tools from build-tools job
2. Check if model exists (skip if not available)
3. Run `inspect --ln-stats --json` with strict mode enabled
4. Parse JSON output and verify:
   - Correct ruleset detected
   - Validation status (ok/warning/failed)
   - Suspicious weight counts for LayerNorm and projection
5. Fail if suspicious weights detected in strict mode

**JSON Output Structure**:

```json
{
  "model_sha256": "abc123...",
  "ruleset": "bitnet-b1.58:i2_s",
  "layernorm": {
    "total": 64,
    "suspicious": 0
  },
  "projection": {
    "total": 2,
    "suspicious": 0
  },
  "strict_mode": true,
  "status": "ok",
  "tensors": [...]
}
```

**Skip Condition**:

Can be skipped via workflow_dispatch input `skip_model_validation: true` for tooling-only changes.

**Artifacts**:

- `model-validation-{model_name}` - Validation output and report
- Retention: 30 days

### 5. Validation Summary (`validation-summary`)

**Purpose**: Aggregate results from all jobs and generate summary

**Summary Contents**:

- Job status for each gate (security-guard, build-tools, validation-tests, validate-models)
- Configuration summary (strict mode, deterministic, correction flags)
- Validation coverage (tools, tests, models)
- Platform coverage

**Output**: GitHub Actions step summary (visible in workflow run page)

**Success Criteria**:

- All jobs must pass (or validate-models may be skipped)
- Fails if any required job fails

### 6. Quality Gate (`quality-gate`)

**Purpose**: Final gate that blocks PR merge on validation failure

**Permissions**:

```yaml
permissions:
  checks: write
  pull-requests: write
```

**Behavior**:

- Always runs (even if previous jobs fail)
- Checks validation-summary result
- Fails with detailed error message if validation did not pass
- Provides common troubleshooting guidance

**Common Issues Reported**:

1. Correction flags set in CI
2. Suspicious LayerNorm weights detected in strict mode
3. Build failures for validation tools
4. Integration test failures

## Usage Examples

### Running the Workflow Manually

```bash
# Via GitHub CLI
gh workflow run validation.yml

# Skip model validation (for tooling-only changes)
gh workflow run validation.yml -f skip_model_validation=true
```

### Local Testing Equivalent

```bash
# Security checks
rg -n 'BITNET_ALLOW_RUNTIME_CORRECTIONS' .github/workflows
rg -n 'BITNET_CORRECTION_POLICY' .github/workflows

# Build tools
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
cargo build -p bitnet-st2gguf --release --no-default-features --features cpu
cargo build -p bitnet-st-tools --release --no-default-features --features cpu

# Run validation tests
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli

# Validate a model
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats --json \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

## Integration with Existing CI

The validation workflow complements other CI workflows:

### Relationship to Other Workflows

1. **`ci.yml`** (Main CI)
   - Runs broader test suite including crossval
   - Validation workflow focuses specifically on validation tooling
   - Both enforce strict mode and block correction flags

2. **`gguf_build_and_validate.yml`**
   - Handles GGUF export and validation for new models
   - Validation workflow validates existing models and tools
   - Both use strict mode and block corrections

3. **`guards.yml`**
   - Broader guards for scripts and workflow files
   - Validation workflow adds validation-specific security checks
   - Both check for forbidden correction flags

4. **`testing-framework-unit.yml`**
   - Runs unit tests for all crates
   - Validation workflow focuses on integration tests for validation
   - Complementary coverage

### Status Check Requirements

For PR merges, the following validation jobs must pass:

- `security-guard` - Critical (blocks merge)
- `build-tools` - Critical (blocks merge)
- `validation-tests` - Critical (blocks merge)
- `validate-models` - Optional (may be skipped)
- `quality-gate` - Critical (blocks merge)

## Troubleshooting

### Common Failures

#### 1. Security Guard Fails

**Error**: `BITNET_ALLOW_RUNTIME_CORRECTIONS must not be set in CI`

**Cause**: Correction flags were added to a workflow file

**Solution**: Remove correction environment variables from workflow files. Use corrections only in local development with explicit policy files and fingerprinting.

#### 2. Build Tools Fail

**Error**: Binary not found or build failure

**Cause**: Compilation errors in validation tools

**Solution**:

```bash
# Test locally
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
cargo build -p bitnet-st2gguf --release --no-default-features --features cpu
cargo build -p bitnet-st-tools --release --no-default-features --features cpu

# Check for compilation errors
cargo check --workspace
```

#### 3. Validation Tests Fail

**Error**: Integration test failures

**Cause**: Validation logic changes broke tests

**Solution**:

```bash
# Run tests locally with verbose output
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli \
  -- --nocapture

# Update tests if validation behavior changed intentionally
```

#### 4. Model Validation Fails

**Error**: Suspicious LayerNorm weights detected

**Cause**: Model has quantized LayerNorm weights (should be F16/F32)

**Solution**:

1. Regenerate model with proper LayerNorm preservation:
   ```bash
   cargo run -p bitnet-st2gguf --no-default-features --features cpu -- --input model.safetensors \
     --output model.gguf --strict
   ```

2. Or use export scripts:
   ```bash
   just model-clean <model_dir> <tokenizer.json>
   ```

3. If model is known-bad and cannot be regenerated:
   - Create a correction policy (for local use only)
   - Document in `docs/explanation/correction-policy.md`
   - Do NOT use in CI

### Debugging Tips

#### View Job Logs

```bash
# List recent workflow runs
gh run list --workflow=validation.yml

# View specific run
gh run view <run_id>

# Download artifacts
gh run download <run_id>
```

#### Test Individual Components

```bash
# Test security guard locally
bash -c 'if grep -r "BITNET_ALLOW_RUNTIME_CORRECTIONS" .github/workflows/*.yml | grep -v "^\s*#"; then echo "Found forbidden flag"; exit 1; fi'

# Test model validation
export BITNET_STRICT_MODE=1
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats --json models/your-model.gguf

# Check JSON output
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats --json models/your-model.gguf | jq '.status'
```

## Best Practices

### For Model Developers

1. **Always use F16/F32 for LayerNorm weights**
   - Never quantize LayerNorm weights
   - Use `bitnet-st2gguf` with strict mode

2. **Test locally with strict mode**
   ```bash
   export BITNET_STRICT_MODE=1
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats your-model.gguf
   ```

3. **Validate before committing**
   - Run validation tests locally
   - Ensure models pass inspection

### For Tool Developers

1. **Update integration tests**
   - Add tests for new validation features
   - Keep tests in sync with validation logic

2. **Document validation behavior**
   - Update feature specs
   - Add examples to help text

3. **Test across platforms**
   - Ensure tools build on Windows, macOS, Linux
   - Test platform-specific code paths

### For CI Maintainers

1. **Never disable strict mode in CI**
   - Always use `BITNET_STRICT_MODE=1`
   - Block correction flags in security-guard

2. **Keep validation fast**
   - Cache dependencies aggressively
   - Skip model validation for tooling-only changes

3. **Provide clear feedback**
   - Generate detailed reports
   - Include troubleshooting guidance in error messages

## Performance

### Typical Execution Times

- **security-guard**: < 1 minute
- **build-tools**: 5-10 minutes (with cache)
- **validation-tests**: 2-5 minutes
- **validate-models**: 1-3 minutes per model
- **Total**: 15-25 minutes (parallel execution)

### Caching Strategy

1. **Cargo dependencies**: Cached per OS with `Swatinem/rust-cache@v2`
2. **Validation tools**: Uploaded as artifacts, reused by downstream jobs
3. **Model files**: Not cached (checked into repository)

### Optimization Tips

1. Use `shared-key` in Swatinem/rust-cache to share cache across jobs
2. Skip model validation for non-model changes via workflow_dispatch
3. Use `fail-fast: false` to continue other jobs on failure
4. Cache tool binaries in `build-tools` job for reuse

## Security Considerations

### Why Block Correction Flags?

Runtime corrections mask fundamental issues:

1. **Quantized LayerNorm weights**
   - Should be fixed at export time
   - Corrections hide the root cause

2. **Policy-based corrections**
   - Only for known-bad models with fingerprinting
   - Should not be used in CI to catch regressions

3. **Deprecated flags**
   - `BITNET_FIX_LN_SCALE` is deprecated
   - Enforces migration to correction policies

### Strict Mode Rationale

`BITNET_STRICT_MODE=1` provides:

1. **Early detection**: Catches suspicious weights immediately
2. **Fail-fast**: Prevents bad models from entering production
3. **Determinism**: Combined with deterministic settings for reproducible validation

### Permissions

Workflow uses minimal permissions:

```yaml
permissions:
  checks: write          # Update check status
  pull-requests: write   # Comment on PRs (quality-gate)
```

No access to:

- Repository contents (read-only via checkout)
- Secrets
- Package publishing

## Future Improvements

### Planned Enhancements

1. **Model caching**: Cache common test models to speed up validation
2. **Parallel model validation**: Validate multiple models concurrently
3. **Detailed tensor reports**: Per-tensor validation details in artifacts
4. **Baseline comparison**: Compare validation metrics against baselines
5. **Integration with release gates**: Block releases on validation failures

### Experimental Features

1. **GPU validation**: Validate GPU-specific code paths
2. **Quantization validation**: Verify quantization quality metrics
3. **Inference validation**: Run inference tests with known outputs
4. **Performance regression detection**: Track validation performance over time

## References

- **Feature Spec**: `docs/features/validation-workflow.md`
- **Correction Policy**: `docs/explanation/correction-policy.md`
- **GGUF Validation**: `docs/howto/export-clean-gguf.md`
- **Integration Tests**: `crates/bitnet-cli/tests/validation_workflow.rs`
- **Main CI**: `.github/workflows/ci.yml`
- **GGUF Build**: `.github/workflows/gguf_build_and_validate.yml`
- **Guards**: `.github/workflows/guards.yml`

## Support

For issues with the validation workflow:

1. Check this documentation for troubleshooting guidance
2. Review workflow logs in GitHub Actions
3. Test components locally with provided commands
4. Open an issue with:
   - Workflow run ID
   - Full error message
   - Local test results
   - Model details (if applicable)
