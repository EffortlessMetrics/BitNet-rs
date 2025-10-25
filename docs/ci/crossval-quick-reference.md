# Cross-Validation Quick Reference

Fast lookup guide for common cross-validation CI tasks.

## Workflow Triggers

| Trigger | Lane A | Lane B | When |
|---------|--------|--------|------|
| **Manual** | ✓ Optional | ✓ Default | User selects via UI |
| **PR with `crossval` label** | ✗ | ✓ | On label add |
| **Nightly (Mon-Sat)** | ✗ | ✓ | 3 AM UTC |
| **Weekly (Sunday)** | ✓ | ✓ | 4 AM UTC |

## Job Status Matrix

| Job | Type | Blocks PR | Platforms | Timeout |
|-----|------|-----------|-----------|---------|
| `check-no-ffi` | ✅ Required | Yes | Ubuntu | 5 min |
| `check-llama-stub` | ✅ Required | Yes | Ubuntu | 5 min |
| `lane-b-llama` | ✅ Required* | Yes | Ubuntu, macOS | 45 min |
| `lane-a-bitnet` | ⚠️ Non-blocking | No | Ubuntu | 60 min |

\* Only when triggered (PR label, nightly, manual)

## Manual Trigger Examples

### Fast Lane B Validation (Default)

```bash
gh workflow run crossval.yml \
  --ref main \
  -f lane=lane-b-llama
```

**Use Case**: Quick parity check before PR merge

**Duration**: ~10 minutes (cached)

### Full Validation (Both Lanes)

```bash
gh workflow run crossval.yml \
  --ref main \
  -f lane=both \
  -f tolerance=0.999
```

**Use Case**: Pre-release validation

**Duration**: ~20 minutes

### Force Rebuild (Cache Bust)

```bash
gh workflow run crossval.yml \
  --ref main \
  -f lane=lane-b-llama \
  -f force_rebuild=true
```

**Use Case**: Cache corruption, upstream C++ changes

**Duration**: ~15 minutes (fresh build)

### Relaxed Tolerance

```bash
gh workflow run crossval.yml \
  --ref main \
  -f lane=lane-b-llama \
  -f tolerance=0.995
```

**Use Case**: Known numerical differences, model quality testing

**Duration**: ~10 minutes

## PR Workflow Checklist

### Adding Cross-Validation to PR

- [ ] Add `crossval` label to PR
- [ ] Wait for Lane B to start (~2 min)
- [ ] Monitor workflow progress (Actions tab)
- [ ] Review summary comment (~10 min total)
- [ ] Fix failures if any
- [ ] Remove label if testing complete

### Interpreting Results

| Status | Icon | Meaning | Action |
|--------|------|---------|--------|
| **PASSED** | ✅ | All checks green | Ready to merge |
| **FAILED** | ❌ | Parity divergence | Review artifacts |
| **SKIPPED** | ⏭️ | Lane not triggered | Expected |
| **CANCELLED** | 🚫 | New commit pushed | Re-run if needed |

## Cache Management

### Cache Keys

```
llama-cpp-{os}-{cpp_tag}     # Lane B libraries
bitnet-cpp-{os}-{cpp_tag}    # Lane A libraries
cargo-{os}-{lock_hash}       # Rust dependencies
```

### Cache Lifetime

- **Hit**: Uses cached libraries (~30 seconds)
- **Miss**: Builds from source (~7-10 minutes)
- **Expiry**: 7 days since last use

### Force Cache Refresh

**Option 1: Manual Dispatch**
```bash
gh workflow run crossval.yml -f force_rebuild=true
```

**Option 2: Update CPP_TAG**
```yaml
# .github/workflows/crossval.yml
env:
  CPP_TAG: '<new-commit-hash>'  # Changes cache key
```

## Artifact Downloads

### GitHub UI

1. Navigate to workflow run
2. Scroll to "Artifacts" section
3. Click artifact name to download

### GitHub CLI

```bash
# List artifacts
gh run view <run-id> --log

# Download specific artifact
gh run download <run-id> -n lane-b-ubuntu-latest-<run-id>

# Download all artifacts
gh run download <run-id>
```

## Common Failure Patterns

### No-FFI Compilation Failure

**Error**: `cargo check -p bitnet-crossval` fails

**Cause**: Unconditional FFI dependency

**Fix**:
```rust
// Bad: unconditional extern
extern "C" { fn foo(); }

// Good: feature-gated extern
#[cfg(feature = "ffi")]
extern "C" { fn foo(); }
```

### STUB Mode Failure

**Error**: Build succeeds, tests fail with missing symbols

**Cause**: Missing `#[cfg(feature = "ffi")]` guards

**Fix**: Audit crossval crate for unconditional FFI usage

### Parity Divergence

**Error**: Cosine similarity below threshold (e.g., 0.998 < 0.999)

**Diagnostics**:
1. Download `crossval-parity-*.json` artifact
2. Check `divergence_token` field
3. Review `min_cosine_similarity` value

**Common Causes**:
- Model fingerprint mismatch
- C++ library version skew
- Numerical precision differences
- Tokenizer parity issues

**Fixes**:
- Verify model SHA256
- Update `CPP_TAG` to match baseline
- Adjust tolerance for legitimate differences
- File issue if reproducible divergence

### Cache Corruption

**Symptoms**:
- Inconsistent test results
- Linker errors
- Missing symbols

**Fix**:
```bash
# Force fresh build
gh workflow run crossval.yml -f force_rebuild=true
```

Or manually clear cache:
1. Go to Actions → Caches
2. Delete relevant cache keys
3. Re-run workflow

## Status Check Configuration

### Required Checks (Branch Protection)

Add these to branch protection rules:

- `check-no-ffi`
- `check-llama-stub`
- `lane-b-llama (ubuntu-latest)` *(optional)*
- `lane-b-llama (macos-latest)` *(optional)*

**Recommendation**: Require no-FFI and STUB mode checks only. Lane B is triggered manually via label.

### GitHub Settings

```
Settings → Branches → Branch protection rules → main
├── Require status checks before merging: ✓
├── Require branches to be up to date: ✓
└── Status checks found:
    ├─ check-no-ffi
    └─ check-llama-stub
```

## Debugging Commands

### Local Reproduction

```bash
# No-FFI check
cargo check -p bitnet-crossval --no-default-features --lib

# STUB mode check
BITNET_CPP_DIR="" cargo build -p bitnet-crossval --features ffi

# Lane B simulation (requires llama.cpp setup)
cargo test -p bitnet-crossval --features cpu,ffi,crossval \
  --test dual_backend_integration -- --nocapture

# Per-token parity
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 4 \
  --cpp-backend llama
```

### Log Analysis

**Find first failure in logs**:
```bash
gh run view <run-id> --log | grep -A 10 "ERROR"
```

**Extract parity metrics**:
```bash
gh run download <run-id> -n lane-b-ubuntu-latest-<run-id>
jq '.metrics' crossval-parity-ubuntu-latest.json
```

## Notification Configuration

### Slack/Discord Integration

Add webhook to workflow:

```yaml
- name: Notify on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "❌ Cross-Validation Failed",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "Lane B cross-validation failed on ${{ github.ref }}"
            }
          }
        ]
      }
```

### Email Notifications

Configure in GitHub personal settings:
```
Settings → Notifications → Actions
├── Email notifications: ✓
└── Send notifications for failed workflows: ✓
```

## Performance Baselines

### Expected Execution Times

| Scenario | Cache Hit | Cache Miss | Notes |
|----------|-----------|------------|-------|
| **Lane B (single OS)** | 5-8 min | 12-15 min | Includes test execution |
| **Lane B (both OS)** | 10-15 min | 20-25 min | Parallel matrix |
| **Lane A** | 8-10 min | 15-20 min | Ubuntu only |
| **Both lanes** | 15-20 min | 25-30 min | Full validation |

### Cache Hit Ratio Targets

- **Daily runs**: 90%+ (stable CPP_TAG)
- **Weekly runs**: 60%+ (cache expiry edge cases)
- **PR runs**: 85%+ (fresh cache from nightly)

## Escalation

### Lane B Failure (Blocking)

1. Review artifacts
2. Check for known issues
3. File bug report with:
   - Run ID
   - Parity JSON
   - Model fingerprint
   - Tolerance used

### Lane A Failure (Non-Blocking)

1. Review weekly report
2. Check upstream bitnet.cpp changes
3. Update CPP_TAG if needed
4. File issue if persistent

### Cache Issues

1. Try `force_rebuild=true`
2. Check GitHub cache limits (10 GB)
3. Clear old caches manually
4. File infrastructure issue

## Quick Links

- [Workflow File](../../.github/workflows/crossval.yml)
- [Detailed Documentation](./crossval-workflow.md)
- [Dual-Backend Architecture](../explanation/dual-backend-crossval.md)
- [Backend Detection Spec](../reference/backend-detection.md)
- [CLAUDE.md Reference](../../CLAUDE.md#cross-validation-cli-reference)
