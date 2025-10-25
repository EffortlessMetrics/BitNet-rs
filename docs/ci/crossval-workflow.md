# Cross-Validation CI Workflow

Comprehensive documentation for the BitNet.rs cross-validation CI workflow.

## Overview

The cross-validation workflow validates BitNet.rs against C++ reference implementations through a dual-lane architecture:

- **Lane A**: BitNet.rs vs bitnet.cpp (BitNet-specific models)
- **Lane B**: BitNet.rs vs llama.cpp (LLaMA-compatible models)

## Workflow Structure

### Status Checks (Required)

These checks gate all PRs and must pass:

1. **check-no-ffi**: Ensures crossval crate compiles without FFI dependencies
2. **check-llama-stub**: Verifies STUB mode (FFI feature with no libraries)

Both checks ensure graceful degradation when C++ libraries are unavailable.

### Cross-Validation Lanes

#### Lane B - llama.cpp (Primary)

- **Trigger**: Daily (3 AM UTC), PR with `crossval` label, manual
- **Execution Time**: ~5-10 min with cached libraries
- **Status**: **Required** - must pass for PRs
- **Platforms**: Ubuntu, macOS

**Features**:
- Cached llama.cpp libraries for fast execution
- Per-token parity validation
- Cosine similarity threshold checking
- Receipt generation for audit trail

**Cache Strategy**:
```yaml
Cache Key: llama-cpp-${{ runner.os }}-${{ env.CPP_TAG }}
Path: ~/.cache/llama_cpp
Retention: 7 days
```

#### Lane A - BitNet.cpp (Optional)

- **Trigger**: Weekly (Sundays 4 AM UTC), manual only
- **Execution Time**: ~10-15 min (source build often required)
- **Status**: **Non-blocking** - continues on error
- **Platforms**: Ubuntu only

**Features**:
- BitNet-specific model validation
- Source build fallback when cache misses
- Non-blocking to prevent weekly schedule breakage

**Cache Strategy**:
```yaml
Cache Key: bitnet-cpp-${{ runner.os }}-${{ env.CPP_TAG }}
Path: ~/.cache/bitnet_cpp
Retention: 7 days
```

## Trigger Conditions

### Manual Dispatch

```yaml
workflow_dispatch:
  inputs:
    lane:
      - lane-b-llama      # Default: fast lane
      - lane-a-bitnet     # BitNet-specific validation
      - both              # Run both lanes
    force_rebuild: boolean
    tolerance: string (default: '0.999')
```

**Usage**:
1. Go to Actions → Cross-Validation
2. Click "Run workflow"
3. Select lane and options
4. Trigger workflow

### Scheduled Execution

- **Daily (3 AM UTC)**: Lane B only (weekdays)
- **Weekly (Sunday 4 AM UTC)**: Both lanes

### Pull Request

Add `crossval` label to PR to trigger Lane B validation.

## Execution Flow

### check-trigger Job

Determines which lanes to run based on:

1. **Manual dispatch**: Uses input selection
2. **PR with label**: Lane B only
3. **Scheduled**: Lane B daily, both lanes on Sunday
4. **Default**: Skip all lanes

Outputs:
- `should_run_lane_a`: boolean
- `should_run_lane_b`: boolean
- `lane_label`: human-readable description

### check-no-ffi Job

**Purpose**: Ensure crossval crate compiles without FFI dependencies

**Steps**:
1. Checkout repository
2. Install Rust toolchain (1.82.0)
3. Cache Cargo dependencies
4. Run `cargo check -p bitnet-crossval --no-default-features --lib`
5. Verify no unconditional bitnet-sys dependency

**Success Criteria**:
- Compilation succeeds without FFI features
- No `bitnet-sys` in dependency tree

### check-llama-stub Job

**Purpose**: Verify STUB mode compilation (FFI feature, no libs)

**Steps**:
1. Checkout repository
2. Install Rust toolchain (1.82.0)
3. Cache Cargo dependencies
4. Build with `ffi` feature, `BITNET_CPP_DIR=""` (forces STUB)
5. Run STUB mode tests

**Success Criteria**:
- Compilation succeeds in STUB mode
- Tests pass without C++ libraries

### lane-b-llama Job

**Purpose**: Cross-validate Rust vs llama.cpp

**Matrix**:
- OS: ubuntu-latest, macos-latest
- Features: cpu,ffi,crossval

**Steps**:
1. Checkout repository
2. Install system dependencies (cmake, ninja, jq)
3. Cache Cargo + llama.cpp libraries
4. Setup llama.cpp (cached or build)
5. Download test model (xtask download-model)
6. Build Rust with crossval features
7. Run cross-validation tests
8. Run per-token parity check
9. Generate receipt
10. Upload artifacts

**Artifacts**:
- `crossval-receipt-{os}.json`: Execution metadata
- `crossval-parity-{os}.json`: Per-token parity results

### lane-a-bitnet Job

**Purpose**: Cross-validate Rust vs bitnet.cpp

**Platform**: Ubuntu only

**Steps**:
1. Checkout repository
2. Install system dependencies
3. Cache Cargo + bitnet.cpp libraries
4. Setup bitnet.cpp (cached or build)
5. Download BitNet test model
6. Build Rust with crossval features
7. Run cross-validation tests (including ignored tests)
8. Upload artifacts

**Non-Blocking**: Uses `continue-on-error` for scheduled runs

### crossval-summary Job

**Purpose**: Aggregate results and post summary

**Always runs** (even on failure) to provide visibility.

**Steps**:
1. Download all artifacts
2. Generate summary markdown report
3. Upload summary artifact
4. Comment on PR (if applicable)
5. Check overall status (fail if required checks failed)

**Summary Contents**:
- Status of required checks (no-FFI, STUB mode)
- Lane results (PASSED/FAILED/Skipped)
- Key findings (accuracy, performance, safety)
- Artifact availability

## Artifacts

### Receipt Format

```json
{
  "lane": "B",
  "backend": "llama.cpp",
  "timestamp": "2025-10-25T12:34:56Z",
  "platform": "ubuntu-latest",
  "tolerance": "0.999",
  "cpp_version": "0e6fbf6...",
  "status": "completed"
}
```

### Parity Report Format

```json
{
  "status": "ok",
  "backend": "llama",
  "divergence_token": -1,
  "metrics": {
    "min_cosine_similarity": 0.9995,
    "max_l2_distance": 0.0042,
    "mean_abs_difference": 0.0018,
    "token_count": 4
  }
}
```

### Retention

- **Lane artifacts**: 30 days
- **Summary reports**: 90 days

## Configuration

### Environment Variables

```yaml
CARGO_TERM_COLOR: always
RUST_BACKTRACE: 1
CARGO_INCREMENTAL: 0
CPP_TAG: '0e6fbf6...'           # C++ reference commit hash
CROSSVAL_TOLERANCE: '0.999'     # Cosine similarity threshold
```

### Concurrency

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

Cancels in-progress runs when new commit pushed to same PR/branch.

### Timeouts

- **Lane B**: 45 minutes per OS
- **Lane A**: 60 minutes
- **Overall**: 120 minutes (GitHub default)

## Failure Handling

### Required Checks Fail

If `check-no-ffi` or `check-llama-stub` fails:
- Workflow fails immediately
- PR cannot merge
- Fix issue before re-running

### Lane B Fails

If Lane B cross-validation fails:
- Workflow fails (blocking)
- PR cannot merge (if triggered by PR)
- Summary reports failure details
- Artifacts available for debugging

### Lane A Fails

If Lane A cross-validation fails:
- Workflow continues (non-blocking)
- Weekly schedule not disrupted
- Summary reports warning
- Investigation recommended but not urgent

## Debugging

### View Logs

1. Go to Actions → Cross-Validation
2. Click on failed run
3. Expand failed job
4. Review step logs

### Download Artifacts

1. Navigate to workflow run
2. Scroll to "Artifacts" section
3. Download relevant artifacts:
   - `lane-b-{os}-{run_id}`: Lane B results
   - `lane-a-bitnet-{run_id}`: Lane A results
   - `crossval-summary-{run_id}`: Aggregated summary

### Common Issues

#### Cache Miss (Slow Build)

**Symptom**: llama.cpp or bitnet.cpp builds from source (~7-10 min)

**Solution**:
- First run after cache expiry (7 days) will rebuild
- Force rebuild: Use `force_rebuild: true` in manual dispatch
- Check cache key matches expected pattern

#### STUB Mode False Positive

**Symptom**: check-llama-stub passes but tests fail with missing symbols

**Solution**:
- Verify `BITNET_CPP_DIR=""` is set during build
- Check `#[cfg(feature = "ffi")]` guards in source
- Ensure no unconditional extern "C" declarations

#### Per-Token Parity Failure

**Symptom**: Cosine similarity below threshold

**Solution**:
- Review parity JSON artifact for first divergence token
- Check model fingerprint matches expected
- Verify C++ library version (CPP_TAG)
- Adjust tolerance if legitimate numerical differences

## Best Practices

### PR Workflow

1. Add `crossval` label to PR
2. Wait for Lane B to complete (~10 min)
3. Review summary comment
4. Fix failures before merging
5. Remove label if testing not needed

### Weekly Maintenance

1. Review Sunday cross-validation results
2. Investigate Lane A failures (non-blocking but informative)
3. Update `CPP_TAG` if upstream changes
4. Refresh cache if performance degrades

### Manual Testing

```bash
# Trigger Lane B only (fast)
gh workflow run crossval.yml \
  --ref main \
  -f lane=lane-b-llama \
  -f tolerance=0.999

# Trigger both lanes with fresh build
gh workflow run crossval.yml \
  --ref main \
  -f lane=both \
  -f force_rebuild=true \
  -f tolerance=0.995
```

## Status Badge

Add to README.md:

```markdown
[![Cross-Validation](https://github.com/USER/REPO/workflows/Cross-Validation/badge.svg)](https://github.com/USER/REPO/actions/workflows/crossval.yml)
```

## Future Enhancements

### Planned

- [ ] GPU cross-validation lane (requires GPU runners)
- [ ] Performance regression tracking
- [ ] Automatic tolerance adjustment based on model type
- [ ] Multi-model test matrix (SmolLM3, LLaMA-3, BitNet)
- [ ] Cache warming on upstream C++ changes

### Under Consideration

- [ ] Differential fuzzing integration
- [ ] Coverage-guided cross-validation
- [ ] Automatic issue creation on nightly failures
- [ ] Integration with benchmark tracking

## References

- [Dual-Backend Architecture](../explanation/dual-backend-crossval.md)
- [Backend Detection Spec](../reference/backend-detection.md)
- [Per-Token Parity Spec](../reference/per-token-parity.md)
- [CLAUDE.md Cross-Validation Section](../../CLAUDE.md#cross-validation-cli-reference)
