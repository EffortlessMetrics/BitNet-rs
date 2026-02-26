# Cross-Validation CI Setup Guide

Step-by-step instructions for integrating the cross-validation workflow into your BitNet-rs repository.

## Prerequisites

Before setting up the cross-validation CI:

- [ ] Repository has `crossval` crate configured
- [ ] `xtask` supports `fetch-cpp` and `crossval-per-token` commands
- [ ] Test models accessible (via download or fixtures)
- [ ] GitHub Actions enabled on repository

## Step 1: Workflow File Setup

### 1.1 Copy Workflow File

The workflow file is already present at:
```
.github/workflows/crossval.yml
```

Verify file exists:
```bash
ls -la .github/workflows/crossval.yml
```

### 1.2 Validate Workflow Syntax

```bash
# Install actionlint (optional, for local validation)
brew install actionlint  # macOS
# or
go install github.com/rhysd/actionlint/cmd/actionlint@latest

# Validate workflow
actionlint .github/workflows/crossval.yml
```

## Step 2: Configure Environment Variables

### 2.1 Update CPP_TAG

Set the C++ reference commit hash in `.github/workflows/crossval.yml`:

```yaml
env:
  CPP_TAG: '0e6fbf6d92a7db6a2f44528a41fb1f35d2c223aa'  # Update this
```

**How to get the correct commit**:

```bash
# For bitnet.cpp
cd legacy/bitnet.cpp
git log -1 --format="%H"

# For llama.cpp
cd legacy/llama.cpp
git log -1 --format="%H"
```

### 2.2 Set Tolerance (Optional)

Default cosine similarity threshold is `0.999`. Adjust if needed:

```yaml
env:
  CROSSVAL_TOLERANCE: '0.999'  # Lower = more permissive
```

## Step 3: Enable GitHub Actions Features

### 3.1 Enable Actions in Repository

1. Go to **Settings → Actions → General**
2. Under "Actions permissions", select:
   - ✓ Allow all actions and reusable workflows
3. Under "Workflow permissions", select:
   - ✓ Read and write permissions
   - ✓ Allow GitHub Actions to create and approve pull requests

### 3.2 Configure Action Caching

Caching is automatic, but verify settings:

1. Go to **Settings → Actions → General**
2. Under "Cache settings":
   - ✓ Enable caching (default)
   - Cache limit: 10 GB (GitHub default)

### 3.3 Set Up Secrets (Optional)

For notifications or private model downloads:

```
Settings → Secrets and variables → Actions → New repository secret

Name: SLACK_WEBHOOK
Value: https://hooks.slack.com/services/...
```

## Step 4: Branch Protection Setup

### 4.1 Add Required Status Checks

1. Go to **Settings → Branches**
2. Click "Add rule" for `main` branch
3. Enable:
   - ✓ Require status checks to pass before merging
   - ✓ Require branches to be up to date before merging

4. Select required checks:
   - ✓ `check-no-ffi`
   - ✓ `check-llama-stub`

**Optional**: Add Lane B checks if you want to require parity for all PRs:
   - ✓ `lane-b-llama (ubuntu-latest)`
   - ✓ `lane-b-llama (macos-latest)`

**Recommendation**: Keep only `check-no-ffi` and `check-llama-stub` as required. Trigger Lane B manually via PR labels.

### 4.2 Configure PR Labels

Create the `crossval` label:

```bash
# Using GitHub CLI
gh label create crossval \
  --description "Trigger cross-validation CI" \
  --color "0E8A16"

# Or via web UI:
# Issues → Labels → New label
# Name: crossval
# Description: Trigger cross-validation CI
# Color: #0E8A16 (green)
```

## Step 5: First Run Validation

### 5.1 Manual Trigger Test

Test the workflow with manual dispatch:

```bash
# Trigger Lane B (fast, recommended for first test)
gh workflow run crossval.yml \
  --ref main \
  -f lane=lane-b-llama \
  -f force_rebuild=true  # Force fresh build for first run

# Monitor progress
gh run list --workflow=crossval.yml

# View logs
gh run view <run-id> --log
```

**Expected first run time**: 15-20 minutes (cache miss)

**Subsequent runs**: 5-10 minutes (cache hit)

### 5.2 Verify Cache Creation

After first successful run:

1. Go to **Actions → Caches**
2. Verify presence of:
   - `llama-cpp-{os}-{cpp_tag}`
   - `cargo-{os}-{lock_hash}`

Cache should persist for 7 days.

### 5.3 Test PR Label Trigger

1. Create a test PR
2. Add `crossval` label
3. Verify workflow triggers automatically
4. Review summary comment on PR

## Step 6: Local Development Setup

### 6.1 Install xtask Dependencies

Ensure local xtask supports cross-validation:

```bash
# Check xtask features
cargo run -p xtask -- --help | grep crossval

# Expected output:
#   crossval-per-token  Per-token parity comparison
#   preflight          Check C++ backend availability
#   setup-cpp-auto     One-command C++ setup
```

### 6.2 Setup C++ References Locally

```bash
# Auto-setup (recommended)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Verify setup
cargo run -p xtask --features crossval-all -- preflight --verbose

# Expected output:
# ✓ llama.cpp: AVAILABLE
#   Libraries: libllama*, libggml*
```

### 6.3 Local Cross-Validation Test

```bash
# Download test model
cargo run -p xtask -- download-model

# Run local cross-validation
cargo test -p bitnet-crossval \
  --features cpu,ffi,crossval \
  --test dual_backend_integration \
  -- --nocapture

# Per-token parity check
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

## Step 7: Monitoring and Observability

### 7.1 Set Up Notifications

**Email** (GitHub native):
```
Your profile → Settings → Notifications → Actions
├── Email notifications: ✓
└── Send notifications for failed workflows: ✓
```

**Slack** (requires webhook):

Add to `.github/workflows/crossval.yml`:

```yaml
# After crossval-summary job
- name: Notify Slack on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "❌ Cross-Validation Failed: ${{ github.ref }}"
      }
```

### 7.2 Dashboard Setup

Track metrics over time:

1. **Actions Dashboard**: `https://github.com/USER/REPO/actions/workflows/crossval.yml`
2. **Cache Usage**: Settings → Actions → Caches
3. **Artifact Storage**: Actions → Workflow run → Artifacts section

### 7.3 Baseline Tracking (Future)

Placeholder for performance baseline tracking:

```bash
# TODO: Implement baseline comparison
# cargo run -p xtask -- compare-baselines \
#   --current crossval/results/latest.json \
#   --baseline baselines/crossval-baseline.json
```

## Step 8: Team Onboarding

### 8.1 Update CONTRIBUTING.md

Add cross-validation section:

```markdown
## Cross-Validation

Before merging PRs that affect inference:

1. Add `crossval` label to PR
2. Wait for Lane B to complete (~10 min)
3. Review summary comment
4. Fix any parity divergences

See [docs/ci/crossval-quick-reference.md](docs/ci/crossval-quick-reference.md).
```

### 8.2 Update CLAUDE.md

The cross-validation commands are already documented in `CLAUDE.md`:

```markdown
# Cross-validation sweep (comprehensive multi-scenario testing)
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval

# Per-token logits divergence detection
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

### 8.3 Team Training

Share with team:

- [Workflow Documentation](./crossval-workflow.md)
- [Quick Reference](./crossval-quick-reference.md)
- [Local Setup Commands](#step-6-local-development-setup)

## Troubleshooting Setup

### Workflow Not Triggering

**Check**:
1. Actions enabled: Settings → Actions → General
2. Workflow file valid YAML: `actionlint .github/workflows/crossval.yml`
3. Branch protection allows Actions: Settings → Branches

**Fix**:
```bash
# Re-commit workflow file to trigger
git commit --amend --no-edit
git push --force-with-lease
```

### Cache Not Working

**Check**:
1. Cache keys match: `llama-cpp-{os}-{cpp_tag}`
2. Cache not expired (7 days)
3. Cache size under 10 GB limit

**Fix**:
```bash
# Force rebuild to recreate cache
gh workflow run crossval.yml -f force_rebuild=true
```

### Status Checks Not Appearing

**Check**:
1. Workflow has run at least once
2. Check names match exactly: `check-no-ffi`, `check-llama-stub`
3. Branch protection rules updated

**Fix**:
1. Trigger workflow manually
2. Wait for completion
3. Re-add branch protection rules

### Local C++ Setup Fails

**Check**:
```bash
# Verify environment
echo $BITNET_CPP_DIR
echo $LD_LIBRARY_PATH

# Check libraries
ls -la ~/.cache/llama_cpp/lib
ls -la ~/.cache/bitnet_cpp/build
```

**Fix**:
```bash
# Clean and rebuild
rm -rf ~/.cache/llama_cpp ~/.cache/bitnet_cpp
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

## Validation Checklist

After setup, verify:

- [ ] Workflow file syntax valid (`actionlint`)
- [ ] Manual trigger works (Lane B)
- [ ] Cache created and persists
- [ ] PR label trigger works
- [ ] Required status checks enforced
- [ ] Artifacts uploaded correctly
- [ ] Summary comment posts on PR
- [ ] Local cross-validation works
- [ ] Team documentation updated
- [ ] Notifications configured

## Next Steps

1. **Run Weekly Validation**: Let Sunday cron run both lanes
2. **Monitor Cache Hit Ratio**: Track in Actions → Caches
3. **Tune Tolerance**: Adjust `CROSSVAL_TOLERANCE` based on results
4. **Add GPU Lane**: When GPU runners available
5. **Baseline Tracking**: Implement performance regression detection

## Support

### Documentation

- [Workflow Details](./crossval-workflow.md)
- [Quick Reference](./crossval-quick-reference.md)
- [CLAUDE.md Cross-Val Section](../../CLAUDE.md#cross-validation-cli-reference)

### Debugging

- [Common Issues](./crossval-quick-reference.md#common-failure-patterns)
- [Artifact Analysis](./crossval-quick-reference.md#artifact-downloads)
- [Local Reproduction](./crossval-quick-reference.md#debugging-commands)

### Escalation

File issues with:
- Run ID
- Workflow logs
- Artifacts (if available)
- Expected vs actual behavior
