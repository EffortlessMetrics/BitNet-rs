# Validation Workflow Documentation

This directory contains the comprehensive validation CI workflow for bitnet-rs.

## Quick Links

- **Workflow File**: [`validation.yml`](./validation.yml) - The actual GitHub Actions workflow
- **Full Documentation**: [`../../docs/development/validation-ci.md`](../../docs/development/validation-ci.md)
- **Quick Reference**: [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md)
- **Visual Diagram**: [`VALIDATION_WORKFLOW_DIAGRAM.md`](./VALIDATION_WORKFLOW_DIAGRAM.md)
- **Summary**: [`VALIDATION_WORKFLOW_SUMMARY.md`](./VALIDATION_WORKFLOW_SUMMARY.md)

## What This Workflow Does

The validation workflow (`validation.yml`) enforces quality standards for bitnet-rs:

### 🔒 Security Guard
- Blocks correction flags in CI (BITNET_ALLOW_RUNTIME_CORRECTIONS, BITNET_CORRECTION_POLICY)
- Verifies strict mode is enabled
- Ensures models are validated without runtime corrections

### 🔨 Build Validation Tools
- Builds bitnet-cli with full-cli features
- Builds bitnet-st2gguf (SafeTensors to GGUF converter)
- Builds bitnet-st-tools (st-ln-inspect, st-merge-ln-f16)
- Cross-platform: Ubuntu, Windows, macOS

### 🧪 Integration Testing
- Runs validation workflow tests
- Tests inspect command functionality
- Validates LayerNorm RMS calculations
- Tests architecture detection and rulesets

### 📊 Model Validation
- Validates GGUF models in strict mode
- Checks for suspicious LayerNorm weights
- Verifies correct ruleset detection
- Can be skipped for tooling-only changes

### ✅ Quality Gate
- Final gate that blocks PR merge on validation failure
- Always runs, provides detailed troubleshooting guidance

## File Structure

```
.github/workflows/
├── validation.yml                      # Main workflow (562 lines)
├── VALIDATION_WORKFLOW_SUMMARY.md      # Quick overview (237 lines)
├── VALIDATION_WORKFLOW_DIAGRAM.md      # Visual diagrams (696 lines)
├── VALIDATION_CHECKLIST.md             # Checklist reference (388 lines)
└── README_VALIDATION.md                # This file (242 lines)

docs/development/
└── validation-ci.md                    # Full documentation (640 lines)

Total: 2,765 lines of workflow + documentation
```

## Usage

### For Contributors

**Before creating a PR:**

```bash
# Set environment to match CI
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Build validation tools
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
cargo build -p bitnet-st2gguf --release
cargo build -p bitnet-st-tools --release

# Run validation tests
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli -- --nocapture

# Validate models (if changed)
cargo run -p bitnet-cli -- inspect --ln-stats --json your-model.gguf
```

**After creating a PR:**

1. Check workflow runs automatically
2. Verify all jobs pass (security-guard, build-tools, validation-tests, quality-gate)
3. Fix any failures locally before pushing

### For Maintainers

**Manual workflow dispatch:**

```bash
# Run validation workflow
gh workflow run validation.yml

# Skip model validation (for tooling-only changes)
gh workflow run validation.yml -f skip_model_validation=true
```

**View workflow status:**

```bash
# List recent runs
gh run list --workflow=validation.yml --limit 10

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

## Key Features

### ✨ Comprehensive Validation
- Security checks for forbidden correction flags
- Cross-platform tool builds (Ubuntu, Windows, macOS)
- Integration tests for validation workflows
- Model validation with strict mode
- Quality gate blocking PR merge

### 🚀 Performance Optimized
- Aggressive caching with Swatinem/rust-cache
- Parallel execution (build-tools, validation-tests run in parallel)
- Typical duration: 15-25 minutes with caching
- Optional model validation skip for tooling changes

### 🔐 Security Enforced
- No correction flags allowed in CI
- Strict mode always enabled (BITNET_STRICT_MODE=1)
- Minimal workflow permissions
- Early detection of suspicious weights

### 📦 Rich Artifacts
- Built binaries for all platforms (7 days retention)
- Validation test reports (30 days retention)
- Model validation outputs (30 days retention)
- Detailed job summaries in GitHub Actions UI

## Integration with Other Workflows

### Main CI (`ci.yml`)
- Main CI: Broad test suite including crossval
- Validation workflow: Focused on validation tooling
- Both: Enforce strict mode and block corrections

### GGUF Build (`gguf_build_and_validate.yml`)
- GGUF Build: Exports and validates new models
- Validation workflow: Validates existing models and tools
- Both: Use strict mode, block corrections

### Guards (`guards.yml`)
- Guards: Broader checks for scripts and workflows
- Validation workflow: Validation-specific security checks
- Both: Check for forbidden correction flags

## Environment Configuration

### Always Enabled

```bash
BITNET_STRICT_MODE=1              # Fail on suspicious weights
BITNET_DETERMINISTIC=1            # Reproducible inference
BITNET_SEED=42                    # Deterministic seed
RAYON_NUM_THREADS=1               # Single-threaded
```

### Always Blocked

```bash
BITNET_ALLOW_RUNTIME_CORRECTIONS  ❌ Not allowed in CI
BITNET_CORRECTION_POLICY          ❌ Not allowed in CI
BITNET_FIX_LN_SCALE               ❌ Deprecated, not allowed
```

## Troubleshooting

### Common Issues

1. **Security Guard Fails**
   - Problem: Correction flags detected in workflow files
   - Solution: Remove BITNET_ALLOW_RUNTIME_CORRECTIONS, BITNET_CORRECTION_POLICY

2. **Build Tools Fails**
   - Problem: Compilation errors
   - Solution: Test locally with `--no-default-features --features cpu,full-cli`

3. **Validation Tests Fail**
   - Problem: Integration test failures
   - Solution: Run tests locally with `--nocapture` flag

4. **Model Validation Fails**
   - Problem: Suspicious LayerNorm weights detected
   - Solution: Regenerate model with `bitnet-st2gguf --strict`

See [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md) for detailed troubleshooting steps.

## Documentation

### For Quick Reference
- **Checklist**: [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md) - Quick reference checklist
- **Summary**: [`VALIDATION_WORKFLOW_SUMMARY.md`](./VALIDATION_WORKFLOW_SUMMARY.md) - One-page overview

### For Visual Understanding
- **Diagram**: [`VALIDATION_WORKFLOW_DIAGRAM.md`](./VALIDATION_WORKFLOW_DIAGRAM.md) - Job flow diagrams

### For Deep Dive
- **Full Docs**: [`../../docs/development/validation-ci.md`](../../docs/development/validation-ci.md) - Comprehensive guide

## Timeline

Typical execution times (with caching):

```
0-1 min:   security-guard
1-10 min:  build-tools (parallel: Ubuntu, Windows, macOS)
10-15 min: validation-tests + validate-models (parallel)
15-20 min: validation-summary + quality-gate
Total:     15-25 minutes
```

## Success Criteria

For PR merge, these jobs must pass:

- ✅ `security-guard` - Critical (blocks merge)
- ✅ `build-tools` - Critical (blocks merge)
- ✅ `validation-tests` - Critical (blocks merge)
- ⚠️ `validate-models` - Optional (may be skipped)
- ✅ `quality-gate` - Critical (blocks merge)

## Support

For issues with the validation workflow:

1. Check [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md) for troubleshooting
2. Review workflow logs in GitHub Actions
3. Test locally with provided commands
4. Check [`../../docs/development/validation-ci.md`](../../docs/development/validation-ci.md) for detailed documentation
5. Open issue with workflow run ID and error details

## Credits

Created: 2025-10-13
Version: 1.0 (initial release)
Part of: bitnet-rs validation infrastructure

---

**Quick Command Reference:**

```bash
# View workflow status
gh run list --workflow=validation.yml

# Run workflow manually
gh workflow run validation.yml

# Test locally
export BITNET_STRICT_MODE=1
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli
```
