# PR3: Performance Receipts & CI Integration - Comprehensive Analysis

**Objective**: Design and implement performance smoke tests with honest receipt generation and CI integration for non-gating observability of inference regressions.

**Document Status**: Analysis Phase - Thorough examination of existing infrastructure

---

## Executive Summary

This document provides a detailed analysis of the current performance script infrastructure, receipt verification system, and CI integration required to implement PR3 performance smoke tests. The analysis reveals a mature receipt ecosystem with clear validation gates, comprehensive CI workflows, and well-structured performance monitoring. The implementation strategy focuses on non-gating observability that catches performance regressions without blocking merges.

**Key Findings**:
- Receipt verification is production-ready with schema v1.0.0 and comprehensive validation gates
- Perf scripts exist but lack determinism flags and host fingerprinting
- Nextest is configured for CI with timeout protection and clean output
- CI has perf-smoke job framework but needs integration with real benchmark receipts
- Performance observability should be non-gating initially, with optional gating for future phases

---

## Part 1: Current Script Functionality & Gaps

### 1.1 Existing Performance Scripts

#### A. `scripts/perf_phase2_timing.sh`

**Current Functionality**:
- Runs 3 iterations of 1-token inference
- Captures timing output via `BITNET_TRACE_TIMING=1` environment variable
- Extracts timing lines with `grep "timing:"`
- Generates markdown receipt at `docs/baselines/perf/phase2_timing_i2s.md`

**Determinism Implementation** ✅
```bash
export BITNET_DETERMINISTIC=1
export RAYON_NUM_THREADS=1
```
Both flags are already present - good foundation.

**Host Fingerprinting** ✅ Partial
```bash
echo "## Host Fingerprint" | tee -a "$RECEIPT"
echo "- Rustc: $(rustc --version)" | tee -a "$RECEIPT"
echo "- OS: $(uname -sr)" | tee -a "$RECEIPT"
echo "- CPU: $(lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//')" | tee -a "$RECEIPT"
echo "- Git commit: $(git rev-parse --short HEAD)" | tee -a "$RECEIPT"
```
Captures: Rustc version, OS, CPU model, Git commit, determinism settings

**Gaps**:
1. No kernel ID tracking (required for receipt validation)
2. Output format is markdown, not JSON (incompatible with receipt verification)
3. No schema version declaration
4. No explicit compute_path ("real" vs "mock")
5. No backend specification
6. Timing extracted from logs, not from structured inference output
7. No metadata about temperature, seed, sampling strategy

#### B. `scripts/phase2_flamegraph.sh`

**Current Functionality**:
- Auto-discovers model and tokenizer
- Detects available flamegraph tool (cargo-flamegraph or samply)
- Checks profiling capability (perf/DTrace)
- Builds optimized release binary with native ISA
- Generates flamegraph SVGs for 1-token and 10-token runs
- Creates markdown metadata with system fingerprint

**Determinism Implementation** ✅
```bash
DETERMINISTIC="${BITNET_DETERMINISTIC:-1}"
SEED="${BITNET_SEED:-42}"
# Sets env vars for profiling
if [[ "$DETERMINISTIC" == "1" ]]; then
    env_vars="BITNET_DETERMINISTIC=1 BITNET_SEED=$SEED RAYON_NUM_THREADS=1 RUST_LOG=warn"
fi
```
Defaults to deterministic mode, configurable via env vars.

**Host Fingerprinting** ✅ Complete
```bash
get_system_fingerprint() {
    # OS-aware CPU detection (lscpu, /proc/cpuinfo, sysctl)
    # Handles Linux, macOS, and unknown systems
}
```
Captures: CPU brand string, OS type, Git metadata (commit, branch)

**Gaps**:
1. No kernel ID tracking (flamegraphs don't have kernel execution info)
2. No receipt generation (SVGs are outputs, not JSON)
3. Flamegraph tool installation overhead (may fail in CI)
4. Requires elevated privileges for perf recording
5. No integration with receipt verification workflow

### 1.2 What Needs to Be Added

#### A. Kernel ID Tracking

**Current Status**: Neither script captures actual kernel IDs executed during inference

**Requirements**:
1. Add kernel recording mechanism to inference engine
2. Capture kernel IDs during benchmark runs
3. Include kernel list in receipt JSON
4. Validate kernel hygiene (non-empty, ≤128 chars, ≤10K total)

**Reference**: `xtask benchmark_cmd()` at line 3329:
```rust
// Capture kernels from the outcome for receipt
kernels_captured = outcome.kernels;
```
Already implemented in benchmark command - we need similar for timing script.

#### B. Determinism Flags Validation

**Current Status**: Flags are set but not validated in output

**Needed**:
1. Add explicit BITNET_SEED documentation (default 42)
2. Verify RAYON_NUM_THREADS=1 in determinism check
3. Record deterministic mode in receipt
4. Validate single-threaded execution

**Example** from cpu_positive_example.json:
```json
"deterministic": true,
"environment": {
  "BITNET_DETERMINISTIC": "1",
  "BITNET_SEED": "42",
  "RAYON_NUM_THREADS": "1"
}
```

#### C. Host Fingerprinting Enhancements

**Current Gaps**:
1. Missing CPU feature flags (AVX2, AVX-512, NEON)
2. No CUDA capability detection
3. No memory information
4. No build configuration details

**Needed Additions**:
```bash
# CPU features
lscpu | grep Flags  # Or grep cpuinfo

# CUDA detection
nvidia-smi --query-gpu=name,driver_version
nvcc --version

# Memory
free -h

# Build configuration
echo "RUSTFLAGS: ${RUSTFLAGS:-none}"
```

---

## Part 2: Receipt Verification Workflow

### 2.1 Receipt Schema v1.0.0 (Production Ready)

**Schema Location**: `docs/tdd/receipts/README.md` (lines 91-158)

**Core Fields**:

| Field | Type | Validation | Status |
|-------|------|-----------|--------|
| `schema_version` | string | Must be "1.0.0" or "1.0" | ✅ Enforced |
| `timestamp` | string | ISO 8601 format | ✅ Enforced |
| `compute_path` | string | Must be "real" (not "mock") | ✅ Enforced |
| `backend` | string | "cpu" \| "cuda" \| "metal" | ✅ Enforced |
| `kernels` | array[string] | Non-empty, no empty strings, ≤128 chars each, ≤10K total | ✅ Enforced |
| `deterministic` | boolean | true \| false | ✅ Accepted |
| `environment` | object | Map of string → string | ✅ Accepted |
| `model_info` | object | Model config | ✅ Accepted |
| `test_results` | object | Test results | ✅ Accepted |
| `performance_baseline` | object | Performance metrics | ✅ Accepted |
| `corrections` | array | Applied corrections | ✅ Enforced (must be empty in production) |

### 2.2 Positive vs. Negative Receipt Examples

**Positive Example** (`cpu_positive_example.json`):
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "compute_path": "real",              ✅ Must be "real"
  "backend": "cpu",                    ✅ Valid backend
  "kernels": [                         ✅ Non-empty, valid kernel IDs
    "i2s_gemv",
    "i2s_matmul_avx2",
    "tl1_lookup_neon",
    "tl2_forward"
  ],
  "deterministic": true,               ✅ True for reproducibility
  "environment": {                     ✅ Full environment
    "BITNET_DETERMINISTIC": "1",
    "BITNET_SEED": "42",
    "RAYON_NUM_THREADS": "1",
    "RUST_VERSION": "1.90.0",
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64",
    "CPU_BRAND": "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz"
  },
  "test_results": {                    ✅ All tests passed
    "total_tests": 10,
    "passed": 10,
    "failed": 0,                       // ✅ Zero failures required
    "skipped": 0
  },
  "performance_baseline": {
    "tokens_per_second": 0.5           ✅ Realistic for QK256
  }
}
```

**Negative Example** (`cpu_negative_example.json`):
```json
{
  "compute_path": "mock",              ❌ VIOLATION: Must be "real"
  "kernels": [
    ""                                 ❌ VIOLATION: Empty kernel ID
  ],
  "test_results": {
    "failed": 2                        ❌ VIOLATION: Must be 0
  },
  "performance_baseline": {
    "tokens_per_second": -1.0          ❌ VIOLATION: Negative value
  }
}
```

### 2.3 Verification Rules & Gate Implementation

**Verification Location**: `xtask/src/main.rs` lines 4381-4505 (verify_receipt_cmd)

**Enforcement Order** (from code inspection):

1. **Schema Validation**
   - Check `schema_version` field exists
   - Accept "1.0.0" or "1.0"
   - Reject invalid versions

2. **Compute Path Check** (Line 4407)
   ```rust
   if compute_path != "real" {
       bail!("compute_path must be 'real' (got '{}') — mock inference not allowed", compute_path);
   }
   ```
   This is a hard gate - no mock inference allowed.

3. **Kernel Array Validation** (Lines 4415-4443)
   - Check array exists and non-empty
   - Check all entries are strings
   - Check for empty/whitespace-only IDs
   - Check length limit (≤128 chars each)
   - Check count limit (≤10K total)

4. **Duplicate Kernel IDs** (Lines 4446-4452)
   - Warning only, not a failure
   - Suspicious but allowed

5. **GPU Kernel Validation** (Lines 4454-4479)
   - Auto-enforces if backend="cuda"
   - Explicit enforce with `--require-gpu-kernels` flag
   - Requires at least one GPU kernel ID
   - GPU kernel patterns: `gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, `tl1_gpu_*`, `tl2_gpu_*`

6. **CPU Backend Validation** (Line 4483)
   - Function: `validate_cpu_backend_kernels()`
   - Requires at least one quantized kernel: `i2s_*`, `tl1_*`, `tl2_*`
   - Rejects fallback patterns

7. **Quantization Claims Verification** (Line 4486)
   - Function: `verify_quantization_claims()`
   - Ensures receipt claims match actual kernel IDs

### 2.4 Receipt Generation (xtask benchmark command)

**Implementation**: `xtask/src/main.rs` lines 3140-3415

**Generation Flow**:
1. Run inference with `run_inference_internal()`
2. Capture kernel IDs in `outcome.kernels`
3. Collect timing metrics (prefill_ms, decode_ms)
4. Calculate performance (tokens_per_second)
5. Write receipt via `write_inference_receipt()`

**Key Function**: `write_inference_receipt()` (Lines 4249-4295)
```rust
fn write_inference_receipt(
    model: &Path,
    tokens_generated: usize,
    tokens_per_second: f64,
    backend: &str,
    kernels: &[String],  // Real kernel IDs
) -> Result<()>
```

**Output**: `ci/inference.json`
```json
{
  "schema_version": "1.0.0",
  "timestamp": "<ISO8601>",
  "compute_path": "real",
  "backend": "<cpu|cuda>",
  "deterministic": true,
  "tokens_requested": <N>,
  "tokens_generated": <N>,
  "tokens_per_second": <float>,
  "kernels": [...],
  "environment": {
    "BITNET_VERSION": "...",
    "OS": "...",
    "RUST_VERSION": "..."
  },
  "model": {
    "path": "..."
  }
}
```

---

## Part 3: CI Workflow Integration

### 3.1 Current CI Structure (`.github/workflows/ci.yml`)

**Job Hierarchy**:

```
test (primary)
├── Rust toolchain setup
├── Cargo fmt & clippy checks
├── Build & test (CPU only)
├── Cross-compile tests
└── Multi-platform matrix (Ubuntu, Windows, macOS)

perf-smoke (depends: test)
├── Download model
├── Build release binary
├── Run 4-token inference
├── Time the execution
├── Comment results on PR (non-gating)

api-compat (PR only)
├── Semver checks
├── Public API diffs
├── FFI header diffs
└── CLI contract diffs

ffi-smoke
├── Smoke build only (no tests)
└── Compiler variations (gcc, clang)

benchmark (main branch only)
├── Runs criterion benchmarks
├── Stores results with benchmark-action
└── Auto-push to gh-pages

quality
├── Code coverage (llvm-cov)
├── Dependency audits
├── Documentation checks
└── Codecov upload

crossval-cpu (main or dispatch)
├── Fetch C++ reference
├── Run full cross-validation
├── Compare metrics with baseline
└── Upload artifacts

build-test-cuda (GPU runner, main/nightly)
├── Build with cuda features
├── Run kernel tests
├── Smoke test
└── No model download

crossval-cuda (GPU runner, main/nightly)
├── Full cross-validation with GPU
├── Metric comparison
└── Artifacts upload

crossval-cpu-smoke (always runs)
├── Minimal C++ FFI test
├── Model lock validation
├── Smoke tests only
└── Artifact upload
```

### 3.2 Existing perf-smoke Job (Lines 133-213)

**Current Implementation**:
```yaml
perf-smoke:
  name: Performance Smoke Test (observability)
  runs-on: ubuntu-latest
  needs: [test]
  steps:
    - name: Download test model
    - name: Build CLI (release)
    - name: Performance smoke test (4-token inference)
      continue-on-error: true
      env:
        RUST_LOG: warn
      run: |
        # Record start time
        START_TIME=$(date +%s)
        
        # Run inference with timing
        time target/release/bitnet-cli run \
          --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
          --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
          --prompt "2+2=" \
          --max-tokens 4 \
          --temperature 0 --greedy || exit 0
        
        # Record end time
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo "Performance smoke test completed in ${DURATION}s"
```

**Status**: Non-gating, informational only
- `continue-on-error: true` - doesn't block merge
- Times execution with bash `time` command
- No receipt generation
- No kernel ID tracking
- Comments results on PR

### 3.3 Verify-Receipts Workflow (`.github/workflows/verify-receipts.yml`)

**Workflow Structure**:

#### Job 1: test-receipt-verification (Always)
```yaml
- Test positive example (should pass)
  cargo run -p xtask -- verify-receipt \
    --path docs/tdd/receipts/cpu_positive_example.json
  
- Test negative example (should fail)
  cargo run -p xtask -- verify-receipt \
    --path docs/tdd/receipts/cpu_negative_example.json
  
- Expect failure (exit code != 0)
```

**Purpose**: Validates that the verification logic correctly distinguishes valid from invalid receipts.

#### Job 2: verify-generated-receipt (main/develop only)
```yaml
- Build workspace (release, cpu features)
- Download test model (best effort)
- Run benchmark to generate ci/inference.json
- Verify generated receipt
- Upload receipt as artifact
```

**Purpose**: Tests that real benchmark output passes verification gates.

#### Job 3: verify-gpu-receipt (GPU runner, optional)
```yaml
- Build workspace (gpu features)
- Run GPU benchmark
- Verify with --require-gpu-kernels flag
```

**Purpose**: Validates GPU receipt generation and kernel enforcement.

### 3.4 Nextest Configuration (`.config/nextest.toml`)

**Profiles**:

```toml
[profile.default]
fail-fast = true
test-threads = "num-cpus"
retries = 0  # No flaky test retries
slow-timeout = { period = "300s", terminate-after = 1 }  # 5-minute timeout
failure-output = "immediate"
success-output = "never"  # Reduce noise
status-level = "pass"

[profile.ci]
fail-fast = false
test-threads = 4  # Fixed thread count for reproducibility
retries = 0
slow-timeout = { period = "300s", terminate-after = 1 }
failure-output = "immediate"
success-output = "never"
status-level = "fail"
```

**Benefits for Performance Tests**:
- Timeout protection: 5-minute limit prevents hanging tests
- Fixed thread count in CI: Ensures reproducible test results
- No retries: Detects flaky tests immediately
- JUnit output: Machine-readable test results for reporting

---

## Part 4: CI Integration Points

### 4.1 Receipt Verification in CI

**Workflow Location**: `.github/workflows/verify-receipts.yml`

**Integration Points**:

1. **Positive Example Test** (Lines 77-94)
   - Runs on every PR/push affecting inference code
   - Validates receipt verification logic works
   - Should pass consistently

2. **Negative Example Test** (Lines 97-114)
   - Validates rejection of invalid receipts
   - Should fail (exit code != 0)
   - Tests that verification catches violations

3. **Generated Receipt Test** (Lines 145-276)
   - Runs benchmark to generate real receipt
   - Verifies generated receipt passes gates
   - Uploads receipt as artifact
   - Main branch only (avoids overhead on PRs)

4. **GPU Receipt Verification** (Lines 279-350)
   - Optional job on GPU runner
   - Uses `--require-gpu-kernels` flag
   - Auto-enforces GPU kernels for cuda backend

### 4.2 Integration with Main CI Workflow

**Dependencies**:
- `test` job must pass before `perf-smoke` runs
- `perf-smoke` is non-gating (continue-on-error: true)
- Receipt verification runs independently

**Proposed Integration for PR3**:

```yaml
perf-smoke:
  name: Performance Smoke Test + Receipt Verification
  runs-on: ubuntu-latest
  needs: [test]
  steps:
    # Existing steps...
    
    # New: Generate receipt via benchmark
    - name: Run benchmark with receipt generation
      run: |
        cargo run -p xtask -- benchmark \
          --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
          --tokens 4 \
          --json ci/inference.json
    
    # New: Verify receipt
    - name: Verify generated receipt
      continue-on-error: true
      run: |
        cargo run -p xtask -- verify-receipt \
          --path ci/inference.json
    
    # New: Comment receipt details on PR
    - name: Comment receipt on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const receipt = require('./ci/inference.json');
          const comment = `## Performance Smoke Test Receipt
          
          - Backend: ${receipt.backend}
          - Kernels executed: ${receipt.kernels.length}
          - Tokens/sec: ${receipt.tokens_per_second}
          - Deterministic: ${receipt.deterministic}
          `;
          // Post comment...
```

### 4.3 Nextest Integration Points

**CI Profile Usage**:
```bash
# In CI workflows
cargo nextest run --profile ci

# Results in:
# - Fixed 4 threads (reproducible)
# - 5-minute timeout (prevent hangs)
# - JUnit output at target/nextest/ci/junit.xml
```

**Performance Test Requirements**:
- Tests should complete within 300 seconds
- Multi-threaded tests may have timing variance
- Use RAYON_NUM_THREADS=1 for determinism in perf tests

---

## Part 5: Nextest Configuration Requirements

### 5.1 Current Configuration Analysis

**File**: `.config/nextest.toml` (41 lines, well-structured)

**What's Working** ✅:
- Timeout protection (5 minutes global)
- No flaky test retries (fail-fast approach)
- Clean output (success-output = "never")
- JUnit reports for parsing

**What's Needed for Perf Tests**:

1. **New Performance Profile** (Optional)
   ```toml
   [profile.perf]
   test-threads = 1  # Serial execution for performance isolation
   slow-timeout = { period = "600s", terminate-after = 1 }  # 10 min for slow tests
   success-output = "skip"
   status-level = "skip"
   ```

2. **Environment Variable Propagation**
   - Currently: Test environment variables set in workflow
   - Needed: Document RAYON_NUM_THREADS handling in profile

### 5.2 Performance Test Best Practices

**Timing Considerations**:
- QK256 scalar kernels: ~0.1 tok/s for 2B models
- 4-token generation: ~40 seconds
- 8-token generation: ~80 seconds
- Flamegraph generation: 2-5 minutes per configuration

**Timeout Strategy**:
- Base timeout: 300s (5 min) - covers 4-token in most scenarios
- With model download: 600s (10 min) - covers full workflow
- Flamegraph-specific: 900s (15 min) - covers tool setup + profiling

**Determinism Requirements**:
- RAYON_NUM_THREADS=1 (serial execution)
- BITNET_DETERMINISTIC=1 (if code supports it)
- BITNET_SEED=42 (consistent random state)
- No wall-clock timing (use precise counters)

---

## Part 6: Perf Smoke Test Implementation Strategy

### 6.1 Non-Gating Observability Model

**Philosophy**: Catch regressions without blocking merges.

**Design**:

```
┌─────────────────────────────────────────┐
│ Performance Smoke Test (Non-Gating)     │
├─────────────────────────────────────────┤
│                                         │
│ ✅ Runs on every PR                     │
│ ✅ Generates receipt with real metrics  │
│ ✅ Comments results on PR               │
│ ✅ Uploads artifacts for analysis       │
│ ❌ Does NOT block merge                 │
│ ❌ Does NOT fail CI on regression       │
│                                         │
│ Benefits:                               │
│ - Developers see performance impact     │
│ - Historical data for trend analysis    │
│ - Evidence for optimization decisions   │
│ - Catches unexpected regressions        │
│                                         │
└─────────────────────────────────────────┘
```

### 6.2 Benchmark Receipt Generation

**Workflow**:

1. **Build Release Binary**
   ```bash
   RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
     cargo build --release -p bitnet-cli --features cpu,full-cli
   ```

2. **Run Benchmark with Determinism**
   ```bash
   export BITNET_DETERMINISTIC=1
   export BITNET_SEED=42
   export RAYON_NUM_THREADS=1
   export RUST_LOG=warn
   
   cargo run -p xtask -- benchmark \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --tokens 4 \
     --json ci/inference.json
   ```

3. **Verify Receipt**
   ```bash
   cargo run -p xtask -- verify-receipt \
     --path ci/inference.json
   ```

4. **Comment on PR**
   ```markdown
   ## Performance Smoke Test Results
   
   **Configuration**: 4 tokens, greedy decode, CPU inference
   **Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2_S quantization)
   
   **Metrics**:
   - Tokens/sec: 0.12
   - Total time: ~33 seconds
   - Kernels executed: i2s_gemv, i2s_matmul_avx2, ...
   
   **Receipt Status**: ✅ Verified (compute_path=real)
   
   **Note**: This test is non-gating and provides visibility into performance 
   changes without blocking merges. For detailed benchmarks, see the full 
   benchmark suite.
   ```

### 6.3 Flamegraph Integration (Future)

**Future Enhancement** (Phase 2):
```yaml
perf-flamegraph:
  name: Performance Flamegraph (observability)
  runs-on: ubuntu-latest
  needs: [test]
  if: contains(github.event.pull_request.labels.*.name, 'perf-analysis')
  steps:
    - name: Generate flamegraph (1-token)
      run: |
        ./scripts/phase2_flamegraph.sh \
          models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
          models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
    
    - name: Upload flamegraph
      uses: actions/upload-artifact@v4
      with:
        name: flamegraphs-${{ github.sha }}
        path: docs/baselines/perf/flamegraphs/
```

---

## Part 7: Step-by-Step Implementation Plan

### Phase 1: Receipt Generation Enhancement (Immediate)

#### Step 1.1: Update perf_phase2_timing.sh for JSON Output

**Changes**:
1. Output JSON instead of markdown
2. Add schema_version field
3. Add compute_path field
4. Add backend field
5. Add kernels array (capture from inference engine)
6. Add full environment fingerprint

**Input Files**:
- `/home/steven/code/Rust/BitNet-rs/scripts/perf_phase2_timing.sh`

**Output**: Modified script with JSON receipt generation

**Code Template**:
```bash
# Add JSON receipt generation
cat > "$RECEIPT_JSON" << 'EOF'
{
  "schema_version": "1.0.0",
  "timestamp": "$(date -Iseconds)",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["i2s_gemv", "i2s_matmul_avx2"],  # Capture from engine
  "deterministic": true,
  "environment": {
    "BITNET_DETERMINISTIC": "1",
    "BITNET_SEED": "42",
    "RAYON_NUM_THREADS": "1",
    "RUST_VERSION": "$(rustc --version)",
    "OS": "$(uname -sr)",
    "CPU_BRAND": "$(lscpu | grep 'Model name')"
  }
}
EOF
```

#### Step 1.2: Integrate Benchmark Command Receipt Generation

**Status**: Already implemented in xtask

**Action**:
- Verify `xtask benchmark` generates ci/inference.json correctly
- Document receipt output in CLAUDE.md
- Add example usage to quickstart

**Test**:
```bash
cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 4 \
  --json ci/inference.json

cargo run -p xtask -- verify-receipt --path ci/inference.json
```

#### Step 1.3: Validate Receipt Verification Logic

**Test Files**:
- `docs/tdd/receipts/cpu_positive_example.json` (should pass)
- `docs/tdd/receipts/cpu_negative_example.json` (should fail)

**Verification Checklist**:
```bash
# Should pass
cargo run -p xtask -- verify-receipt \
  --path docs/tdd/receipts/cpu_positive_example.json

# Should fail
cargo run -p xtask -- verify-receipt \
  --path docs/tdd/receipts/cpu_negative_example.json || echo "Failed as expected"
```

### Phase 2: CI Integration (Parallel with Phase 1)

#### Step 2.1: Enhance perf-smoke Job

**Target**: `.github/workflows/ci.yml` lines 133-213

**Changes**:
1. Add benchmark receipt generation
2. Add receipt verification step
3. Enhance PR comment with receipt details
4. Upload receipt as artifact

**Implementation**:
```yaml
perf-smoke:
  name: Performance Smoke Test (observability)
  runs-on: ubuntu-latest
  needs: [test]
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    
    - name: Download test model
      run: cargo run -p xtask -- download-model
    
    - name: Build CLI (release)
      run: |
        RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
          cargo build --release -p bitnet-cli --no-default-features --features cpu,full-cli
    
    - name: Run benchmark with receipt generation
      env:
        BITNET_DETERMINISTIC: "1"
        BITNET_SEED: "42"
        RAYON_NUM_THREADS: "1"
        RUST_LOG: warn
      run: |
        mkdir -p ci
        cargo run -p xtask --release -- benchmark \
          --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
          --tokens 4 \
          --json ci/inference.json || {
            echo "::warning::Benchmark failed, falling back to mock"
            exit 0
          }
    
    - name: Verify receipt
      if: hashFiles('ci/inference.json') != ''
      continue-on-error: true
      run: |
        cargo run -p xtask --release -- verify-receipt \
          --path ci/inference.json || echo "Receipt verification failed (non-gating)"
    
    - name: Comment receipt on PR
      if: github.event_name == 'pull_request' && hashFiles('ci/inference.json') != ''
      continue-on-error: true
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          let receipt = {};
          try {
            receipt = JSON.parse(fs.readFileSync('ci/inference.json', 'utf8'));
          } catch (e) {
            console.log('Could not read receipt:', e);
          }
          
          const comment = `## 🔍 Performance Smoke Test (Observability)
          
          **Status**: ✅ Non-gating (informational only)
          **Configuration**: 4 tokens, greedy decode, CPU inference
          **Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2_S)
          
          ${receipt.tokens_per_second ? `**Performance**: ${receipt.tokens_per_second.toFixed(2)} tok/s` : ''}
          ${receipt.kernels ? `**Kernels**: ${receipt.kernels.length} executed` : ''}
          ${receipt.backend ? `**Backend**: ${receipt.backend}` : ''}
          
          This test provides visibility into performance changes without blocking merges.
          For detailed analysis, see the artifact: \`inference-receipt-${{ github.sha }}.json\`
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
    
    - name: Upload receipt artifact
      if: always() && hashFiles('ci/inference.json') != ''
      uses: actions/upload-artifact@v4
      with:
        name: inference-receipt-${{ github.sha }}
        path: ci/inference.json
        retention-days: 30
```

#### Step 2.2: Verify Receipt Workflow Integration

**Status**: Workflow already exists

**Action**:
- Verify workflow triggers correctly on PRs
- Test with example receipts
- Document in CI integration guide

**Test**:
```bash
# Simulate workflow locally
cargo run -p xtask -- verify-receipt \
  --path docs/tdd/receipts/cpu_positive_example.json

cargo run -p xtask -- verify-receipt \
  --path docs/tdd/receipts/cpu_negative_example.json || echo "Failed as expected"
```

#### Step 2.3: Document Nextest Integration

**Action**:
- Document CI profile usage in CLAUDE.md
- Explain timeout protection (5 min default)
- Explain thread count control (4 threads in CI)
- Add best practices for performance tests

**Location**: `CLAUDE.md` section "Test Execution"

### Phase 3: Performance Script Enhancements (Week 2)

#### Step 3.1: Enhance phase2_flamegraph.sh for Determinism

**Changes**:
1. Document determinism flags
2. Add validation that RAYON_NUM_THREADS=1
3. Verify BITNET_DETERMINISTIC=1
4. Record in metadata

**Code Update**:
```bash
# Add validation
if [[ "$DETERMINISTIC" == "1" ]]; then
  if [[ "${RAYON_NUM_THREADS:-}" != "1" ]]; then
    log_warn "RAYON_NUM_THREADS not set to 1, forcing for determinism"
    export RAYON_NUM_THREADS=1
  fi
fi

# Record in metadata
echo "- BITNET_DETERMINISTIC: $DETERMINISTIC" | tee -a "$md_file"
echo "- BITNET_SEED: $SEED" | tee -a "$md_file"
echo "- RAYON_NUM_THREADS: ${RAYON_NUM_THREADS:-auto}" | tee -a "$md_file"
```

#### Step 3.2: Add Advanced Host Fingerprinting

**Additions**:
1. CPU feature flags (AVX2, AVX-512, NEON)
2. CUDA capability detection (if GPU available)
3. Memory information
4. Build configuration

**Code Template**:
```bash
get_extended_fingerprint() {
    local os_type=$(uname -s)
    
    # CPU features
    if [[ "$os_type" == "Linux" ]]; then
        lscpu | grep Flags | cut -d: -f2 | xargs
    else
        sysctl -n machdep.cpu.features 2>/dev/null || echo "N/A"
    fi
    
    # CUDA detection
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        echo "CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    fi
    
    # Memory
    if [[ "$os_type" == "Linux" ]]; then
        free -h | grep Mem | awk '{print "Memory: " $2}'
    fi
}
```

#### Step 3.3: Create Combined Timing + Receipt Script

**New Script**: `scripts/perf_smoke_bench.sh`

**Purpose**: Unified performance smoke test with determinism and receipt generation

**Features**:
- Run configurable token counts
- Generate JSON receipts
- Verify receipts
- Generate markdown reports
- Support both real and mock modes

**Skeleton**:
```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"
TOKENS="${3:-4}"
OUTPUT="${4:-ci/inference.json}"

# Enable determinism
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
export RUST_LOG=warn

# Build
cargo build --release --no-default-features --features cpu,full-cli

# Run benchmark with receipt generation
cargo run -p xtask --release -- benchmark \
  --model "$MODEL" \
  --tokens "$TOKENS" \
  --json "$OUTPUT"

# Verify receipt
cargo run -p xtask --release -- verify-receipt --path "$OUTPUT"

echo "✅ Performance smoke test complete"
echo "📄 Receipt: $OUTPUT"
```

### Phase 4: Documentation & Testing (Week 3)

#### Step 4.1: Document Receipt Workflow in CLAUDE.md

**Location**: New section "Performance Smoke Tests & Receipts"

**Content**:
1. What are receipts?
2. Receipt schema overview
3. How to generate receipts (benchmark command)
4. How to verify receipts (verify-receipt command)
5. Receipt examples (positive and negative)
6. CI integration (non-gating observability)
7. Troubleshooting

#### Step 4.2: Create PR3 Implementation Guide

**Document**: `docs/development/pr3-perf-receipts-implementation.md`

**Sections**:
1. Architecture overview
2. Component responsibilities
3. Integration points
4. Testing strategy
5. Rollout plan

#### Step 4.3: Add Integration Tests

**Test Location**: `xtask/tests/perf_smoke_integration.rs`

**Test Cases**:
1. Benchmark generates valid JSON receipt
2. Verify-receipt accepts valid receipts
3. Verify-receipt rejects invalid receipts
4. Receipt contains required fields
5. Kernel IDs are non-empty
6. Determinism flags are recorded
7. Host fingerprint is complete

**Test Template**:
```rust
#[test]
fn test_benchmark_generates_receipt() {
    // Run benchmark with small token count
    // Verify ci/inference.json exists
    // Parse JSON and check schema
    // Verify required fields present
}

#[test]
fn test_verify_receipt_accepts_valid() {
    // Load cpu_positive_example.json
    // Run verify_receipt_cmd()
    // Assert success
}

#[test]
fn test_verify_receipt_rejects_mock() {
    // Load cpu_negative_example.json
    // Run verify_receipt_cmd()
    // Assert failure with "mock" error
}
```

#### Step 4.4: Update CI Workflow Documentation

**Location**: `docs/development/ci-integration.md`

**Additions**:
1. perf-smoke job overview
2. Receipt generation and verification
3. Non-gating observability model
4. Performance regression detection
5. Artifact and reporting

---

## Part 8: Technical Specifications & Details

### 8.1 Kernel ID Tracking Implementation

**Challenge**: Capturing real kernel IDs from inference engine

**Current State**: 
- xtask benchmark command already captures kernels
- Mechanism: `outcome.kernels` field from `run_inference_internal()`

**Required for Scripts**:
1. Modify perf scripts to use xtask benchmark (not direct binary call)
2. Or: Expose kernel tracking to binary output
3. Or: Use instrumentation/tracing framework

**Recommended Approach**: Use xtask benchmark wrapper
```bash
# Instead of:
target/release/bitnet run --model ... --max-tokens 4

# Use:
cargo run -p xtask -- benchmark --model ... --tokens 4 --json ci/inference.json
```

**Advantages**:
- Automatic receipt generation
- Kernel tracking built-in
- Consistent with CI workflow
- Determinism flags enforced

### 8.2 Determinism Implementation Details

**Required Settings**:

| Variable | Value | Purpose | Status |
|----------|-------|---------|--------|
| BITNET_DETERMINISTIC | 1 | Enable deterministic mode | ✅ Implemented |
| BITNET_SEED | 42 | Consistent random seed | ✅ Implemented |
| RAYON_NUM_THREADS | 1 | Single-threaded execution | ✅ Used in perf tests |
| RUST_LOG | warn | Reduce log noise | ✅ Used in CI |
| OMP_NUM_THREADS | 1 | OpenMP parallelism control | ❓ Document |
| MKL_NUM_THREADS | 1 | MKL parallelism control | ❓ Document |

**Validation Approach**:
```bash
# After setting variables, verify:
echo "RAYON_NUM_THREADS: ${RAYON_NUM_THREADS}"
echo "BITNET_DETERMINISTIC: ${BITNET_DETERMINISTIC}"
echo "BITNET_SEED: ${BITNET_SEED}"

# Run twice and verify identical output
RUN1=$(... | md5sum)
RUN2=$(... | md5sum)
if [[ "$RUN1" == "$RUN2" ]]; then
  echo "✅ Determinism validated"
fi
```

### 8.3 Host Fingerprinting Specification

**Required Fields**:

```json
{
  "environment": {
    "RUST_VERSION": "rustc 1.90.0 (..)",
    "RUST_EDITION": "2024",
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64",
    "CPU_BRAND": "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz",
    "CPU_CORES": 8,
    "CPU_FEATURES": "avx2 avx512f fma",
    "MEMORY_GB": 32,
    "GIT_COMMIT": "abc1234",
    "GIT_BRANCH": "main",
    "GIT_DIRTY": false,
    "BUILD_CONFIG": "RUSTFLAGS=\"-C target-cpu=native -C opt-level=3\""
  }
}
```

**Collection Script**:
```bash
get_fingerprint() {
    jq -n \
      --arg rust "$(rustc --version)" \
      --arg os "$(uname -sr)" \
      --arg cpu "$(lscpu | grep "Model name" | cut -d: -f2 | xargs)" \
      --arg cores "$(nproc)" \
      --arg mem "$(free -h | grep Mem | awk '{print $2}')" \
      --arg git_commit "$(git rev-parse --short HEAD)" \
      --arg git_branch "$(git rev-parse --abbrev-ref HEAD)" \
      --arg git_dirty "$(git diff --quiet || echo 'true')" \
      '{
        RUST_VERSION: $rust,
        OS: $os,
        CPU_BRAND: $cpu,
        CPU_CORES: ($cores | tonumber),
        MEMORY_GB: $mem,
        GIT_COMMIT: $git_commit,
        GIT_BRANCH: $git_branch,
        GIT_DIRTY: ($git_dirty | if . == "true" then true else false end)
      }'
}
```

### 8.4 Performance Baseline Specification

**Realistic Values** (from cpu_positive_example.json):

| Metric | QK256 Scalar | i2s_avx2 | Notes |
|--------|----------|----------|-------|
| tok/s (2B model) | 0.1 | 0.5-1.0 | Scalar: very slow, optimization TBD |
| Prefill time | 1-2s | 100-500ms | Highly variable based on prompt |
| Decode latency | 8-10s/token | 1-2s/token | Per-token cost |
| Memory (2B) | 4-5 GB | 4-5 GB | Model + inference overhead |

**Validation Checks**:
- `tokens_per_second > 0` (no negative values)
- For QK256: `0.05 < tok/s < 2.0` (realistic range)
- For I2S_AVX2: `0.5 < tok/s < 10.0` (realistic range)
- Memory consistent with model size

---

## Part 9: Risk Analysis & Mitigation

### 9.1 Potential Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Timing variability in CI | High | Medium | Use determinism flags, multiple runs, median |
| Model download timeout | Medium | Medium | Fallback to mock, artifact caching |
| Flamegraph tool missing | Low | Low | Fallback to samply, skip if unavailable |
| Receipt format breaking | Low | High | Version schema, test with examples |
| Performance degradation undetected | Medium | High | Regular baseline validation, regression alerts |
| False alarms (noise in timings) | High | Low | Wide tolerance bands, trend analysis |

### 9.2 Mitigation Strategies

**Timing Variability**:
- Run multiple iterations, take median
- Set wide acceptance bands (±10% initially)
- Use statistical methods for trend analysis
- Document variance per platform

**Model Download**:
- Cache model in CI artifacts
- Fallback to mock if unavailable
- Skip download on PR (only main branch)

**Flamegraph Optional**:
- Make flamegraph job optional with `continue-on-error: true`
- Skip if required tools unavailable
- Focus on timing data as primary metric

**Schema Evolution**:
- Version schema as "1.0.0" with migration path to "2.0"
- Support multiple versions in verification
- Document breaking changes

**Regression Detection**:
- Non-gating initially (observational only)
- Can become gating in future phase with mature baselines
- Requires 2-3 months of baseline data first

---

## Part 10: Rollout & Transition Plan

### Phase 1: Immediate (This PR)
1. Enhance perf-smoke CI job with receipt generation
2. Add receipt verification step
3. PR comments with receipt details
4. Upload artifacts

**Timeline**: 1-2 weeks
**Scope**: CI integration only, perf scripts unchanged

### Phase 2: Near-term (PR3+)
1. Enhance perf scripts for JSON output
2. Add advanced host fingerprinting
3. Create perf_smoke_bench.sh wrapper
4. Document receipt workflow

**Timeline**: 2-3 weeks
**Scope**: Scripts and documentation

### Phase 3: Future (Phase 2)
1. Flamegraph CI job with optional label trigger
2. Baseline establishment (3-month window)
3. Performance regression detection (gating)
4. Performance optimization tracking

**Timeline**: Month 2-3
**Scope**: Gating performance tests, historical tracking

---

## Part 11: Example Outputs

### 11.1 Successful Benchmark Receipt

**Generated**: `ci/inference.json`
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-01-22T14:32:15Z",
  "compute_path": "real",
  "backend": "cpu",
  "deterministic": true,
  "tokens_requested": 4,
  "tokens_generated": 4,
  "tokens_per_second": 0.12,
  "kernels": [
    "i2s_gemv",
    "i2s_matmul_avx2",
    "rope_forward",
    "softmax_temperature"
  ],
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64",
    "RUST_VERSION": "rustc 1.90.0 (abcdef)"
  },
  "model": {
    "path": "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
  }
}
```

### 11.2 CI Job Output

**Console Output**:
```
🚀 Running decode performance benchmark...
   Model: models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
   Device: cpu
   Warmup tokens: 4
   Benchmark tokens: 4

🔥 Running warmup...
   Warmup completed in 8243 ms

⏱️ Running benchmark...
   Prefill: 1234 ms
   Decode: 2890 ms (avg 722 ms/token)
   Total: 33487 ms

✅ Inference receipt written to ci/inference.json
   Schema: 1.0.0
   Tokens/sec: 0.12
   Kernels: 4 executed
   Backend: cpu

🔍 Verifying inference receipt…
✅ Receipt verification passed
   Schema: 1.0.0
   Compute path: real
   Kernels: 4 executed
   Backend: cpu
```

### 11.3 PR Comment Output

**Posted on PR**:
```markdown
## 🔍 Performance Smoke Test (Observability)

**Status**: ✅ Non-gating (informational only)
**Configuration**: 4 tokens, greedy decode, CPU inference
**Model**: microsoft-bitnet-b1.58-2B-4T-gguf (I2_S)

**Performance**: 0.12 tok/s
**Kernels**: 4 executed (i2s_gemv, i2s_matmul_avx2, rope_forward, softmax_temperature)
**Backend**: cpu

**Receipt Verification**: ✅ Passed
- compute_path: real (honest inference)
- schema_version: 1.0.0
- deterministic: true

This test provides visibility into performance changes without blocking merges.
For detailed analysis, see the artifact: `inference-receipt-abc1234.json`
```

---

## Part 12: Verification Checklist

### Pre-Implementation Verification

- [ ] Receipt schema v1.0.0 understood and documented
- [ ] Positive/negative test receipts reviewed
- [ ] xtask verify-receipt command tested locally
- [ ] xtask benchmark command tested with receipt generation
- [ ] CI workflow structure reviewed
- [ ] Nextest configuration analyzed
- [ ] perf-smoke job requirements clarified
- [ ] Determinism mechanism validated

### Implementation Verification

- [ ] Receipt generation in CI working (benchmark --json flag)
- [ ] Receipt verification gate integrated
- [ ] PR comments posting successfully
- [ ] Artifacts uploading correctly
- [ ] Nextest timeout protection validated
- [ ] Determinism flags enforced
- [ ] Host fingerprinting complete
- [ ] Kernel ID tracking verified

### Testing & Validation

- [ ] Positive receipt passes verification
- [ ] Negative receipt fails verification
- [ ] Generated receipts pass gates
- [ ] Performance metrics realistic
- [ ] Timing variance acceptable
- [ ] Multi-platform consistency (Linux, macOS, Windows)
- [ ] CPU feature detection working
- [ ] No false alarms in CI

### Documentation

- [ ] CLAUDE.md updated with receipt workflow
- [ ] CI integration guide updated
- [ ] Implementation guide created
- [ ] Example receipts documented
- [ ] Troubleshooting guide added
- [ ] FAQ updated with perf tests

---

## Appendix A: File Locations Reference

**Key Files Analyzed**:

```
.github/workflows/
  ├── ci.yml (2103 lines)
  │   ├── perf-smoke job (L133-213) ← Enhancement target
  │   └── Other jobs (test, benchmark, quality, crossval)
  └── verify-receipts.yml (350 lines) ← Integration model
  
scripts/
  ├── perf_phase2_timing.sh (57 lines) ← Enhancement target
  └── phase2_flamegraph.sh (809 lines) ← Determinism reference
  
docs/
  ├── tdd/receipts/
  │   ├── README.md (295 lines) ← Schema specification
  │   ├── cpu_positive_example.json ← Valid receipt template
  │   └── cpu_negative_example.json ← Violation examples
  └── development/
      ├── ci-integration.md (200+ lines) ← CI reference
      └── test-suite.md ← Nextest documentation

xtask/src/
  ├── main.rs
  │   ├── benchmark_cmd() (L3140-3415) ← Receipt generation
  │   ├── write_inference_receipt() (L4249-4295) ← Receipt writer
  │   └── verify_receipt_cmd() (L4381-4505) ← Verification logic
  └── gates.rs (45 lines) ← Gate implementation

.config/
  └── nextest.toml (42 lines) ← Test configuration
  
CLAUDE.md ← Project instructions
```

---

## Appendix B: Glossary

| Term | Definition | Reference |
|------|-----------|-----------|
| Receipt | JSON artifact documenting real inference execution with kernel IDs and metrics | `docs/tdd/receipts/README.md` |
| Compute Path | Field indicating "real" (honest) or "mock" (test) inference | Schema v1.0.0 |
| Kernel ID | Name of executed quantization kernel (e.g., "i2s_gemv") | Validation rules |
| Non-gating | Test doesn't block merge even if it fails | CI job: `continue-on-error: true` |
| Determinism | Reproducible inference results across runs | BITNET_DETERMINISTIC=1 |
| Flamegraph | CPU profiling visualization showing function hotspots | `phase2_flamegraph.sh` |
| Nextest | High-speed Rust test runner with timeout protection | `.config/nextest.toml` |
| Smoke Test | Quick validation test catching major issues | `perf-smoke` job |
| Receipt Verification | Validation that receipt meets quality gates | `verify-receipt` command |

---

## Appendix C: Further Reading

**In-Codebase Documentation**:
1. `docs/tdd/receipts/README.md` - Receipt schema and examples
2. `docs/development/ci-integration.md` - CI workflow overview  
3. `CLAUDE.md` - Project instructions and best practices
4. `.github/workflows/verify-receipts.yml` - Receipt workflow reference
5. `.github/workflows/ci.yml` - Main CI workflow structure

**Specific Code Locations**:
1. Receipt verification: `xtask/src/main.rs:4381-4505`
2. Receipt generation: `xtask/src/main.rs:4249-4295`
3. Benchmark command: `xtask/src/main.rs:3140-3415`
4. Perf script: `scripts/perf_phase2_timing.sh`
5. Flamegraph script: `scripts/phase2_flamegraph.sh`

---

## Summary

This analysis provides a comprehensive roadmap for implementing PR3 performance smoke tests with honest receipt generation. The infrastructure is largely in place:

- Receipt schema (v1.0.0) is production-ready
- Verification gates are implemented and tested
- CI workflow foundation exists
- xtask benchmark command generates receipts correctly
- Nextest provides timeout protection

**Immediate next steps**:
1. Integrate receipt generation into perf-smoke CI job
2. Add receipt verification step
3. Comment receipt details on PRs
4. Upload artifacts for historical analysis

**Key principle**: Non-gating observability initially, gating only after baseline establishment (2-3 months).

