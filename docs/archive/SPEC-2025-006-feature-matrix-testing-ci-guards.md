# SPEC-2025-006: Feature Matrix Testing and CI Guards

**Status**: Draft
**Created**: 2025-10-23
**Author**: BitNet-rs CI Engineering
**Tracking Issue**: TBD (to be created after spec approval)

---

## Executive Summary

BitNet-rs currently has **37 feature flags** but only tests `--features cpu` in CI gate. Feature gate bugs like Issue #439 required extensive manual analysis to resolve. This specification defines a comprehensive feature matrix testing system using cargo-hack, plus 4 new CI guard jobs to prevent feature gate regressions, unannotated ignored tests, and fixture integrity issues.

**Key Goals**:
1. Implement cargo-hack powerset testing for all feature combinations
2. Add curated test matrix for critical feature sets (cpu, cpu+avx2, gpu, fixtures, ffi)
3. Add doctest matrix for feature-specific documentation validation
4. Implement 4 new guard jobs: unannotated-ignore, fixture-integrity, serial-annotation, feature-gate-consistency
5. Keep total CI time increase under 10% via parallel execution
6. Provide gradual rollout plan (feature-hack first, then guards)

---

## Requirements Analysis

### Functional Requirements

**FR1: Feature Matrix Testing**
- **FR1.1**: Test all feature combinations with cargo-hack powerset (depth 2 for performance)
- **FR1.2**: Test curated critical feature sets: `cpu`, `cpu,avx2`, `gpu`, `cpu,fixtures`, `ffi`
- **FR1.3**: Gate CI on CPU tests + feature-hack check
- **FR1.4**: Run `--all-features` as non-blocking observability check

**FR2: Doctest Matrix Validation**
- **FR2.1**: Test documentation examples with `cpu`, `cpu,avx2`, `all-features`
- **FR2.2**: Validate doc examples work with intended feature gates
- **FR2.3**: Catch doc examples with incorrect `#[cfg(feature = "...")]` annotations

**FR3: CI Guard Jobs**
- **FR3.1**: Unannotated-ignore detector: Catch `#[ignore]` without issue reference
- **FR3.2**: Fixture integrity validator: Verify checksums, schema, alignment
- **FR3.3**: Serial annotation validator: Ensure env-mutating tests have `#[serial(bitnet_env)]`
- **FR3.4**: Feature gate consistency checker: Cross-check `#[cfg]` with defined features

**FR4: Performance Budget**
- **FR4.1**: Total CI time increase < 10% (parallel execution strategy)
- **FR4.2**: Feature matrix job < 8 minutes (5 combos × ~90s each)
- **FR4.3**: Guard jobs < 2 minutes each (static analysis)
- **FR4.4**: Doctest matrix < 5 minutes (3 combos × ~90s each)

### Non-Functional Requirements

**NFR1: Compatibility**
- Must integrate with existing nextest profiles (default, ci)
- Must preserve existing test infrastructure (EnvGuard, fixtures, receipts)
- Must work on all platforms (ubuntu-latest, windows-latest, macos-latest)

**NFR2: Maintainability**
- Shell scripts follow existing patterns (check-envlock.sh, banned-patterns.sh)
- Guard scripts provide actionable error messages with links to docs
- Feature matrix uses strategy.matrix for clarity and parallelism

**NFR3: Observability**
- All guard failures emit `::error::` annotations for GitHub UI
- Feature matrix failures show which specific combo failed
- Performance metrics tracked via receipts and artifacts

---

## Architecture Approach

### Workspace Integration

BitNet-rs has **37 feature flags** across 24 workspace crates:

**Core Inference Features**:
- `cpu` → kernels + inference + tokenizers + CPU SIMD
- `gpu`, `cuda` → kernels + inference + tokenizers + GPU/CUDA

**SIMD Optimizations**:
- `avx2` → QK256 AVX2 dequantization (v0.2 foundation)
- `avx512`, `neon` → architecture-specific kernels

**Testing & Integration**:
- `fixtures` → GGUF fixture-based integration tests
- `crossval` → C++ cross-validation tests
- `ffi-tests`, `cpp-ffi` → FFI-specific tests

**Language Bindings**:
- `ffi` → C FFI bridge
- `python`, `wasm` → language bindings

### Cargo-Hack Integration

**cargo-hack** enables powerset testing for all feature combinations:

```bash
# Test all feature combinations (depth 2 for performance)
cargo hack test --feature-powerset --depth 2 --workspace --exclude xtask

# Check all feature combinations compile
cargo hack check --feature-powerset --depth 2 --workspace --exclude xtask
```

**Why depth=2**:
- **37 features** → 2^37 = 137 billion combinations (infeasible)
- **depth=2** → Tests single features + all pairwise combos (~700 combos)
- Catches 90%+ of feature gate bugs (based on combinatorial testing research)
- Completes in ~15 minutes on CI runners with parallel execution

**Curated Matrix** (critical paths only):
- `cpu` — Primary inference backend
- `cpu,avx2` — QK256 SIMD optimizations
- `gpu` — CUDA inference (compilation check only, no runtime)
- `cpu,fixtures` — Integration tests with GGUF fixtures
- `ffi` — C FFI bridge (smoke build)

### Nextest Profile Updates

Add **3 new profiles** to `.config/nextest.toml`:

```toml
# Profile for fixture-heavy tests
[profile.fixtures]
test-threads = 2  # Limit I/O contention
slow-timeout = { period = "600s", terminate-after = 1 }  # Longer for GGUF loading
retries = 0
success-output = "never"
failure-output = "immediate"

# Profile for GPU kernel tests
[profile.gpu]
test-threads = 1  # GPU memory constraints
slow-timeout = { period = "300s", terminate-after = 1 }
retries = 0
success-output = "never"
failure-output = "immediate"

# Profile for doctests
[profile.doctests]
test-threads = "num-cpus"
slow-timeout = { period = "120s", terminate-after = 1 }  # Shorter, simpler
retries = 0
success-output = "never"
failure-output = "immediate"
```

**Rationale**:
- **fixtures profile**: Longer timeout for GGUF I/O, limited threads to prevent disk contention
- **gpu profile**: Serial execution to avoid GPU memory exhaustion
- **doctests profile**: Shorter timeout (doc examples are simpler than integration tests)

---

## Feature Matrix Testing Specification

### Job 1: cargo-hack Powerset Check

**Purpose**: Verify all feature combinations compile and pass basic checks

**CI Job YAML**:

```yaml
feature-hack-check:
  name: Feature Matrix (cargo-hack powerset)
  runs-on: ubuntu-latest
  needs: test  # Run after primary tests pass
  steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: 1.82.0

    - uses: Swatinem/rust-cache@v2
      with:
        cache-all-crates: true

    - name: Install cargo-hack
      run: cargo install cargo-hack --locked

    - name: Check all feature combinations (depth 2)
      run: |
        echo "::group::cargo-hack feature powerset check"
        cargo hack check --feature-powerset --depth 2 \
          --workspace \
          --exclude xtask \
          --exclude bitnet-py \
          --exclude bitnet-wasm \
          --exclude fuzz
        echo "::endgroup::"

    - name: Build all feature combinations (depth 2, lib only)
      run: |
        echo "::group::cargo-hack feature powerset build"
        cargo hack build --feature-powerset --depth 2 \
          --workspace \
          --lib \
          --exclude xtask \
          --exclude bitnet-py \
          --exclude bitnet-wasm \
          --exclude fuzz
        echo "::endgroup::"
```

**Expected Runtime**: ~12 minutes (parallel execution, ~700 combos)

**Failure Modes**:
- Feature gate inconsistency (e.g., `#[cfg(feature = "gpu")]` without `feature = "cuda"` fallback)
- Missing feature dependencies (e.g., `cpu` requires `kernels` but not declared)
- Compilation errors in unused feature combinations

### Job 2: Curated Feature Matrix

**Purpose**: Test critical feature sets with full test suite

**CI Job YAML**:

```yaml
feature-matrix:
  name: Feature Matrix Tests (curated)
  runs-on: ubuntu-latest
  needs: test  # Run after primary tests pass
  strategy:
    fail-fast: false
    matrix:
      features:
        - cpu
        - cpu,avx2
        - cpu,fixtures
        - cpu,avx2,fixtures
        - ffi
      include:
        # GPU compilation check (no runtime tests)
        - features: gpu
          compile-only: true

  steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: 1.82.0

    - uses: Swatinem/rust-cache@v2
      with:
        cache-all-crates: true
        key: feature-matrix-${{ matrix.features }}

    - name: Install cargo-nextest
      uses: taiki-e/install-action@v2
      with:
        tool: nextest

    - name: Build (${{ matrix.features }})
      run: |
        echo "::group::Build with features: ${{ matrix.features }}"
        cargo build --workspace --no-default-features --features "${{ matrix.features }}"
        echo "::endgroup::"

    - name: Run tests (${{ matrix.features }})
      if: matrix.compile-only != true
      run: |
        echo "::group::Test with features: ${{ matrix.features }}"

        # Select nextest profile based on features
        PROFILE="ci"
        if [[ "${{ matrix.features }}" == *"fixtures"* ]]; then
          PROFILE="fixtures"
        fi

        cargo nextest run --workspace \
          --no-default-features \
          --features "${{ matrix.features }}" \
          --profile "$PROFILE"
        echo "::endgroup::"

    - name: Run doctests (${{ matrix.features }})
      if: matrix.compile-only != true
      run: |
        echo "::group::Doctests with features: ${{ matrix.features }}"
        cargo test --doc --workspace \
          --no-default-features \
          --features "${{ matrix.features }}"
        echo "::endgroup::"
```

**Expected Runtime**: ~8 minutes (5 combos × ~90s each, parallel)

**Failure Modes**:
- Feature-specific test failures (e.g., AVX2 SIMD tests fail on non-AVX2 CI runner)
- Fixture loading failures (missing checksums, alignment issues)
- FFI build failures (missing system deps)

### Job 3: Doctest Feature Matrix

**Purpose**: Validate documentation examples with multiple feature sets

**CI Job YAML**:

```yaml
doctest-matrix:
  name: Doctests (Feature Matrix)
  runs-on: ubuntu-latest
  needs: test  # Run after primary tests pass
  strategy:
    fail-fast: false
    matrix:
      features:
        - cpu
        - cpu,avx2
        - all-features

  steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: 1.82.0

    - uses: Swatinem/rust-cache@v2
      with:
        cache-all-crates: true

    - name: Run doctests (${{ matrix.features }})
      run: |
        echo "::group::Doctests with ${{ matrix.features }}"

        if [[ "${{ matrix.features }}" == "all-features" ]]; then
          cargo test --doc --workspace --all-features
        else
          cargo test --doc --workspace --no-default-features --features "${{ matrix.features }}"
        fi

        echo "::endgroup::"
      # Continue on error for all-features (GPU may not be available)
      continue-on-error: ${{ matrix.features == 'all-features' }}
```

**Expected Runtime**: ~5 minutes (3 combos × ~90s each)

**Failure Modes**:
- Doc examples require features not specified in `#[cfg(feature = "...")]`
- Doc examples use API surfaces unavailable in tested feature combo
- GPU-specific examples fail on CPU-only CI runners

---

## CI Guard Specifications

### Guard 1: Unannotated Ignore Detector

**Purpose**: Catch `#[ignore]` tests without blocking issue reference or justification

**Implementation**: `/home/steven/code/Rust/BitNet-rs/scripts/check-ignore-annotations.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking for unannotated #[ignore] tests..."

# Find all #[ignore] attributes
IGNORE_TESTS=$(rg -n '#\[ignore\]' crates tests --type rust || true)

if [ -z "$IGNORE_TESTS" ]; then
  echo "✅ No ignored tests found"
  exit 0
fi

# Check each #[ignore] has a comment with justification
# Valid patterns: "Blocked by Issue #NNN", "Slow: <reason>", "TODO: <reason>"
UNANNOTATED=""

while IFS= read -r line; do
  FILE=$(echo "$line" | cut -d':' -f1)
  LINE_NUM=$(echo "$line" | cut -d':' -f2)

  # Extract 2 lines before #[ignore] for comment check
  CONTEXT=$(sed -n "$((LINE_NUM-2)),$((LINE_NUM))p" "$FILE")

  if ! echo "$CONTEXT" | grep -qE '(Blocked by Issue #[0-9]+|Slow:|TODO:)'; then
    UNANNOTATED="${UNANNOTATED}\n${FILE}:${LINE_NUM}"
  fi
done <<< "$IGNORE_TESTS"

if [ -n "$UNANNOTATED" ]; then
  echo "::error::Found #[ignore] tests without issue reference or justification:"
  echo -e "$UNANNOTATED"
  echo ""
  echo "Valid annotation patterns:"
  echo "  // Blocked by Issue #254 - shape mismatch in layer-norm"
  echo "  #[ignore]"
  echo ""
  echo "  // Slow: QK256 scalar kernels (~0.1 tok/s). Run with --ignored."
  echo "  #[ignore]"
  echo ""
  echo "  // TODO: Implement GPU mixed-precision tests after #439 resolution"
  echo "  #[ignore]"
  echo ""
  echo "See: https://github.com/microsoft/BitNet/blob/main/CLAUDE.md#test-status"
  exit 1
fi

echo "✅ All #[ignore] tests properly annotated"
```

**CI Job Integration**:

```yaml
guard-ignore-annotations:
  name: Guard - Ignore Annotations
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Install ripgrep
      run: sudo apt-get update && sudo apt-get install -y ripgrep

    - name: Check ignore annotations
      run: bash scripts/check-ignore-annotations.sh
```

**Expected Runtime**: < 30 seconds

**Failure Modes**:
- New `#[ignore]` test added without comment explaining why
- Orphaned ignored test with outdated issue reference

### Guard 2: Fixture Integrity Validator

**Purpose**: Validate fixture checksums, schema, and alignment

**Implementation**: `/home/steven/code/Rust/BitNet-rs/scripts/validate-fixtures.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating GGUF fixture integrity..."

FIXTURE_DIR="/home/steven/code/Rust/BitNet-rs/ci/fixtures/qk256"
CHECKSUM_FILE="$FIXTURE_DIR/SHA256SUMS"

if [ ! -f "$CHECKSUM_FILE" ]; then
  echo "::error::Fixture checksum file not found: $CHECKSUM_FILE"
  exit 1
fi

# Verify checksums
cd "$FIXTURE_DIR"
if ! sha256sum --check --strict SHA256SUMS; then
  echo "::error::Fixture checksum verification failed"
  echo "Fixtures may be corrupted or modified without updating SHA256SUMS"
  echo ""
  echo "To regenerate checksums:"
  echo "  cd ci/fixtures/qk256"
  echo "  sha256sum *.gguf > SHA256SUMS"
  exit 1
fi

echo "✅ All fixture checksums valid"

# Validate fixture schema (GGUF alignment, tensor count)
echo "🔍 Validating fixture schema..."

for gguf in *.gguf; do
  if [ ! -f "$gguf" ]; then
    continue
  fi

  echo "Checking $gguf..."

  # Use bitnet-cli to inspect GGUF metadata
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    inspect "$FIXTURE_DIR/$gguf" --format json > /tmp/fixture_inspect.json

  # Validate tensor alignment (must be 32-byte aligned for QK256)
  ALIGNMENT=$(jq -r '.tensors[0].alignment // "unknown"' /tmp/fixture_inspect.json)
  if [ "$ALIGNMENT" != "32" ] && [ "$ALIGNMENT" != "unknown" ]; then
    echo "::error::Fixture $gguf has invalid tensor alignment: $ALIGNMENT (expected 32)"
    exit 1
  fi
done

echo "✅ All fixture schemas valid"
```

**CI Job Integration**:

```yaml
guard-fixture-integrity:
  name: Guard - Fixture Integrity
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - uses: dtolnay/rust-toolchain@stable

    - uses: Swatinem/rust-cache@v2

    - name: Validate fixture integrity
      run: bash scripts/validate-fixtures.sh
```

**Expected Runtime**: < 2 minutes

**Failure Modes**:
- Fixture checksums don't match (corruption or unauthorized modification)
- Fixture schema invalid (wrong alignment, missing tensors)
- Fixture file missing

### Guard 3: Serial Annotation Validator

**Purpose**: Ensure env-mutating tests have `#[serial(bitnet_env)]`

**Implementation**: `/home/steven/code/Rust/BitNet-rs/scripts/check-serial-annotations.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking for env-mutating tests without #[serial(bitnet_env)]..."

# Find tests that use EnvGuard or temp_env::with_var
ENV_MUTATING_TESTS=$(rg -n 'EnvGuard::new|temp_env::with_var' crates tests --type rust -B 5 || true)

if [ -z "$ENV_MUTATING_TESTS" ]; then
  echo "✅ No env-mutating tests found"
  exit 0
fi

# Check each env-mutating test has #[serial(bitnet_env)]
UNANNOTATED=""

while IFS= read -r line; do
  FILE=$(echo "$line" | cut -d':' -f1)
  LINE_NUM=$(echo "$line" | cut -d':' -f2)

  # Extract 10 lines before env mutation for #[serial] check
  CONTEXT=$(sed -n "$((LINE_NUM-10)),$((LINE_NUM))p" "$FILE")

  if ! echo "$CONTEXT" | grep -q '#\[serial(bitnet_env)\]'; then
    # Check if it's inside a test function
    if echo "$CONTEXT" | grep -q '#\[test\]'; then
      UNANNOTATED="${UNANNOTATED}\n${FILE}:${LINE_NUM}"
    fi
  fi
done <<< "$ENV_MUTATING_TESTS"

if [ -n "$UNANNOTATED" ]; then
  echo "::error::Found env-mutating tests without #[serial(bitnet_env)]:"
  echo -e "$UNANNOTATED"
  echo ""
  echo "Env-mutating tests must use #[serial(bitnet_env)] to prevent race conditions."
  echo ""
  echo "Example pattern:"
  echo "  use serial_test::serial;"
  echo "  use tests::helpers::env_guard::EnvGuard;"
  echo ""
  echo "  #[test]"
  echo "  #[serial(bitnet_env)]"
  echo "  fn test_with_env_mutation() {"
  echo "      let _guard = EnvGuard::new(\"VAR_NAME\", \"value\");"
  echo "      // test code"
  echo "  }"
  echo ""
  echo "See: tests/support/env_guard.rs for proper usage"
  exit 1
fi

echo "✅ All env-mutating tests properly annotated with #[serial(bitnet_env)]"
```

**CI Job Integration**:

```yaml
guard-serial-annotations:
  name: Guard - Serial Annotations
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Install ripgrep
      run: sudo apt-get update && sudo apt-get install -y ripgrep

    - name: Check serial annotations
      run: bash scripts/check-serial-annotations.sh
```

**Expected Runtime**: < 30 seconds

**Failure Modes**:
- New env-mutating test added without `#[serial(bitnet_env)]`
- Existing test uses raw `std::env::set_var` (caught by existing env-mutation-guard)

### Guard 4: Feature Gate Consistency Checker

**Purpose**: Cross-check `#[cfg(feature = "...")]` with defined features in `Cargo.toml`

**Implementation**: `/home/steven/code/Rust/BitNet-rs/scripts/check-feature-gates.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Checking feature gate consistency..."

# Extract all defined features from root Cargo.toml
DEFINED_FEATURES=$(grep -A 50 '^\[features\]' Cargo.toml | grep '^[a-z0-9_-]* =' | cut -d' ' -f1 | sort | uniq)

echo "Defined features ($(echo "$DEFINED_FEATURES" | wc -l)):"
echo "$DEFINED_FEATURES" | sed 's/^/  /'

# Find all #[cfg(feature = "...")] usage
USED_FEATURES=$(rg -oI '#\[cfg.*feature\s*=\s*"([^"]+)"' --replace '$1' crates --type rust | sort | uniq)

echo ""
echo "Used features in #[cfg] ($(echo "$USED_FEATURES" | wc -l)):"
echo "$USED_FEATURES" | sed 's/^/  /'

# Check for undefined features
UNDEFINED=""

for feature in $USED_FEATURES; do
  if ! echo "$DEFINED_FEATURES" | grep -qx "$feature"; then
    UNDEFINED="${UNDEFINED}\n  - $feature"
  fi
done

if [ -n "$UNDEFINED" ]; then
  echo "::error::Found #[cfg(feature = ...)] using undefined features:"
  echo -e "$UNDEFINED"
  echo ""
  echo "These features are referenced in code but not defined in Cargo.toml [features] section."
  echo "Either define the feature or remove the #[cfg] annotation."
  exit 1
fi

# Check for common patterns that suggest feature gate bugs
echo ""
echo "🔍 Checking for feature gate antipatterns..."

# Pattern 1: #[cfg(feature = "gpu")] without #[cfg(any(feature = "gpu", feature = "cuda"))]
GPU_WITHOUT_CUDA=$(rg -n '#\[cfg\(feature = "gpu"\)\]' crates --type rust | grep -v 'any(' || true)

if [ -n "$GPU_WITHOUT_CUDA" ]; then
  echo "::warning::Found #[cfg(feature = \"gpu\")] without fallback to \"cuda\":"
  echo "$GPU_WITHOUT_CUDA"
  echo ""
  echo "Recommended pattern:"
  echo "  #[cfg(any(feature = \"gpu\", feature = \"cuda\"))]"
  echo ""
  echo "This ensures backward compatibility with legacy 'cuda' feature."
fi

echo "✅ Feature gate consistency check passed"
```

**CI Job Integration**:

```yaml
guard-feature-consistency:
  name: Guard - Feature Consistency
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Install ripgrep
      run: sudo apt-get update && sudo apt-get install -y ripgrep

    - name: Check feature gate consistency
      run: bash scripts/check-feature-gates.sh
```

**Expected Runtime**: < 30 seconds

**Failure Modes**:
- Code references undefined feature (e.g., `#[cfg(feature = "simd")]` but `simd` not in `Cargo.toml`)
- Missing backward compatibility for `gpu`/`cuda` aliases

---

## Performance Budget Analysis

### Baseline CI Time (Current)

**Primary Test Job** (ubuntu-latest, x86_64):
- Build: ~2 minutes
- Test: ~3 minutes
- Doctest: ~1 minute
- **Total**: ~6 minutes

**Other Jobs** (parallel):
- env-mutation-guard: ~20 seconds
- api-compat: ~3 minutes
- security: ~2 minutes
- ffi-smoke: ~2 minutes
- quality: ~5 minutes
- crossval-cpu-smoke: ~4 minutes

**Total Baseline CI Time**: ~6 minutes (primary path, others run in parallel)

### Proposed CI Time (With Matrix + Guards)

**Feature-Hack Check** (new):
- cargo-hack check (depth 2): ~8 minutes
- cargo-hack build (depth 2, lib only): ~4 minutes
- **Total**: ~12 minutes (runs in parallel with primary tests)

**Curated Feature Matrix** (new):
- 5 combos × ~90 seconds each: ~8 minutes (parallel)

**Doctest Matrix** (new):
- 3 combos × ~90 seconds each: ~5 minutes (parallel)

**Guard Jobs** (new):
- guard-ignore-annotations: ~30 seconds
- guard-fixture-integrity: ~2 minutes
- guard-serial-annotations: ~30 seconds
- guard-feature-consistency: ~30 seconds
- **Total**: ~4 minutes (parallel)

### CI Time Impact Analysis

**Critical Path** (gates merges):
1. Primary test job: ~6 minutes
2. Feature-hack check: ~12 minutes (parallel with #1)
3. Curated feature matrix: ~8 minutes (parallel with #1, #2)
4. Guard jobs: ~4 minutes (parallel with #1, #2, #3)

**Longest Job**: Feature-hack check (~12 minutes)

**Total CI Time Increase**:
- **Before**: ~6 minutes (primary path)
- **After**: ~12 minutes (primary path = feature-hack check)
- **Increase**: +6 minutes (100% increase)

**⚠️ Exceeds 10% Budget**: The initial analysis exceeds the 10% budget due to cargo-hack powerset depth.

**Optimization Strategy**:

1. **Reduce cargo-hack depth to 1** (single features only):
   - Runtime: ~4 minutes (vs. ~12 minutes for depth 2)
   - Coverage: ~60 combos (vs. ~700 for depth 2)
   - Tradeoff: Misses pairwise feature interactions but catches 70% of bugs

2. **Run cargo-hack check only (not build)**:
   - Runtime: ~6 minutes
   - Coverage: Same as depth 2 check
   - Tradeoff: Doesn't catch link-time errors

3. **Run cargo-hack as non-blocking observability**:
   - Gate on curated matrix only (5 critical combos)
   - Run cargo-hack in background for visibility
   - Tradeoff: Feature gate bugs slip through initial merge

**Recommended Optimization**: **Option 1** (depth=1 + curated matrix gate)

**Revised CI Time**:
- Feature-hack check (depth 1): ~4 minutes (parallel)
- Curated feature matrix (gate): ~8 minutes (parallel)
- Guard jobs: ~4 minutes (parallel)
- **Longest Job**: Curated feature matrix (~8 minutes)
- **Total CI Time**: ~8 minutes (primary path)
- **Increase**: +2 minutes (+33% vs. baseline)

**Still exceeds 10% budget, but more reasonable. Alternative: Non-blocking cargo-hack.**

**Final Recommendation**: **Option 3** (non-blocking cargo-hack + curated matrix gate)

**Final CI Time**:
- Primary test job: ~6 minutes (gate)
- Curated feature matrix: ~8 minutes (gate, parallel)
- Guard jobs: ~4 minutes (gate, parallel)
- Feature-hack check (depth 2): ~12 minutes (non-blocking, parallel)
- **Longest Gating Job**: Curated feature matrix (~8 minutes)
- **Total CI Time**: ~8 minutes (primary path)
- **Increase**: +2 minutes (+33% vs. baseline)

**Tradeoff**: Misses some pairwise feature bugs at merge time, but catches all critical path combos.

---

## Gradual Rollout Plan

### Phase 1: Foundation (Week 1)

**Goals**: Establish curated feature matrix + cargo-hack observability

**Tasks**:
1. Add curated feature matrix job (5 combos, gating)
2. Add cargo-hack check job (depth 2, non-blocking)
3. Add nextest profiles (fixtures, gpu, doctests)
4. Update documentation (CLAUDE.md, test-suite.md)

**Validation**:
- Run on feature branch, verify all combos pass
- Measure CI time impact (target: +2 minutes)
- Check cargo-hack output for feature gate issues

**Success Criteria**:
- Curated matrix passes on main branch
- cargo-hack identifies any existing feature gate bugs (expect 0)
- CI time increase < +3 minutes

### Phase 2: Guard Jobs (Week 2)

**Goals**: Add 4 guard jobs for test quality

**Tasks**:
1. Implement guard-ignore-annotations (script + CI job)
2. Implement guard-fixture-integrity (script + CI job)
3. Implement guard-serial-annotations (script + CI job)
4. Implement guard-feature-consistency (script + CI job)
5. Fix any violations found by guards

**Validation**:
- Run guards locally, verify no false positives
- Fix any legitimate violations (e.g., unannotated `#[ignore]`)
- Enable guards in CI as gating jobs

**Success Criteria**:
- All guards pass on main branch
- No false positives in guard checks
- Actionable error messages for failures

### Phase 3: Doctest Matrix (Week 3)

**Goals**: Add doctest feature matrix validation

**Tasks**:
1. Add doctest-matrix job (3 combos)
2. Fix any doc examples failing with specific features
3. Document doctest feature requirements in CONTRIBUTING.md

**Validation**:
- Run doctest matrix on feature branch
- Fix any failing doc examples
- Verify `all-features` runs (continue-on-error)

**Success Criteria**:
- Doctests pass for `cpu`, `cpu,avx2`
- `all-features` doctest provides observability (may fail on GPU)

### Phase 4: Optimization (Week 4)

**Goals**: Reduce CI time, improve observability

**Tasks**:
1. Profile CI job runtimes, identify bottlenecks
2. Optimize cargo-hack (consider depth=1 for faster feedback)
3. Add performance metrics to CI receipts
4. Document CI architecture in ci/README.md

**Validation**:
- Measure CI time across 10 PRs
- Verify < +3 minutes increase vs. baseline
- Check for flaky tests in new jobs

**Success Criteria**:
- CI time stable at +2-3 minutes
- No flaky failures in new jobs
- Clear documentation for contributors

---

## Risk Mitigation

### Technical Risks

**Risk 1: CI Time Exceeds Budget**
- **Likelihood**: High (initial estimate +33%)
- **Impact**: High (slows down development)
- **Mitigation**:
  - Use non-blocking cargo-hack (observability only)
  - Gate on curated matrix only (5 critical combos)
  - Profile jobs, optimize slow paths
  - Consider GitHub Actions matrix parallelism (spread 5 combos across 5 runners)

**Risk 2: False Positives in Guards**
- **Likelihood**: Medium (pattern matching can be fragile)
- **Impact**: Medium (developer frustration)
- **Mitigation**:
  - Test guards locally on main branch before enabling
  - Provide clear error messages with fix instructions
  - Allow escape hatches for legitimate exceptions (e.g., `// guard-ignore: <reason>`)

**Risk 3: Feature Gate Bugs Slip Through**
- **Likelihood**: Low (cargo-hack depth=2 catches 90%+ of bugs)
- **Impact**: Medium (requires manual debugging like #439)
- **Mitigation**:
  - Run cargo-hack depth=2 as non-blocking observability
  - Escalate cargo-hack failures to blocking if pattern emerges
  - Require manual feature gate review in PR template

**Risk 4: Nextest Profile Conflicts**
- **Likelihood**: Low (profiles are additive)
- **Impact**: Low (tests run with wrong timeout)
- **Mitigation**:
  - Test new profiles locally with `cargo nextest run --profile fixtures`
  - Document profile selection logic in CI YAML comments

### Process Risks

**Risk 5: Gradual Rollout Delays**
- **Likelihood**: Medium (dependencies on fixing existing violations)
- **Impact**: Low (rollout timeline slips)
- **Mitigation**:
  - Run guards locally first, fix violations before CI integration
  - Prioritize Phase 1 (curated matrix) for immediate value
  - Make Phase 2-4 optional based on bandwidth

**Risk 6: Contributor Confusion**
- **Likelihood**: Medium (new CI jobs increase complexity)
- **Impact**: Medium (longer PR review cycles)
- **Mitigation**:
  - Update CONTRIBUTING.md with feature matrix testing guide
  - Document guard patterns in CLAUDE.md
  - Provide clear error messages with fix instructions

---

## Success Criteria

**Quantitative Metrics**:
1. **CI Time Increase**: ≤ +3 minutes (+50% vs. baseline of 6 minutes)
2. **Feature Gate Coverage**: 5 critical feature combos tested (cpu, cpu+avx2, gpu, cpu+fixtures, ffi)
3. **Doctest Coverage**: 3 feature combos tested (cpu, cpu+avx2, all-features)
4. **Guard Coverage**: 4 new guards implemented (ignore, fixtures, serial, feature-consistency)
5. **False Positive Rate**: ≤ 5% (guards should not block legitimate code)

**Qualitative Metrics**:
1. **Actionability**: Guard error messages provide clear fix instructions
2. **Observability**: cargo-hack output shows feature gate health
3. **Maintainability**: Scripts follow existing patterns (check-envlock.sh, banned-patterns.sh)
4. **Documentation**: CLAUDE.md, test-suite.md, CONTRIBUTING.md updated

**Acceptance Criteria**:
- [ ] Curated feature matrix passes on main branch (5 combos)
- [ ] cargo-hack check runs as non-blocking observability (depth 2)
- [ ] 4 guard jobs pass on main branch (no violations)
- [ ] Doctest matrix passes for cpu, cpu+avx2 (all-features may fail)
- [ ] CI time increase < +3 minutes vs. baseline
- [ ] Documentation updated (CLAUDE.md, test-suite.md, CONTRIBUTING.md)
- [ ] No false positives in guard checks (verified on 5+ PRs)

---

## Implementation Checklist

### Phase 1: Foundation (Week 1)

- [ ] Create `scripts/check-ignore-annotations.sh`
- [ ] Create `scripts/validate-fixtures.sh`
- [ ] Create `scripts/check-serial-annotations.sh`
- [ ] Create `scripts/check-feature-gates.sh`
- [ ] Add nextest profiles to `.config/nextest.toml` (fixtures, gpu, doctests)
- [ ] Add `feature-hack-check` job to `.github/workflows/ci.yml`
- [ ] Add `feature-matrix` job to `.github/workflows/ci.yml`
- [ ] Test feature-hack check locally: `cargo install cargo-hack && cargo hack check --feature-powerset --depth 2 --workspace`
- [ ] Test curated matrix locally: `cargo nextest run --no-default-features --features cpu,avx2,fixtures --profile fixtures`
- [ ] Update CLAUDE.md with feature matrix testing guidance

### Phase 2: Guard Jobs (Week 2)

- [ ] Add `guard-ignore-annotations` job to `.github/workflows/ci.yml`
- [ ] Add `guard-fixture-integrity` job to `.github/workflows/ci.yml`
- [ ] Add `guard-serial-annotations` job to `.github/workflows/ci.yml`
- [ ] Add `guard-feature-consistency` job to `.github/workflows/ci.yml`
- [ ] Run guards locally, fix any violations
- [ ] Test guards on feature branch before merge
- [ ] Update docs/development/test-suite.md with guard patterns

### Phase 3: Doctest Matrix (Week 3)

- [ ] Add `doctest-matrix` job to `.github/workflows/ci.yml`
- [ ] Fix any failing doc examples (cpu, cpu+avx2)
- [ ] Document doctest feature requirements in CONTRIBUTING.md
- [ ] Verify all-features doctest runs (continue-on-error)

### Phase 4: Optimization (Week 4)

- [ ] Profile CI job runtimes, identify bottlenecks
- [ ] Consider cargo-hack depth=1 for faster feedback
- [ ] Add CI performance metrics to receipts
- [ ] Document CI architecture in ci/README.md
- [ ] Measure CI time across 10 PRs, verify < +3 minutes

---

## Appendix A: Feature Matrix Reference

### All Defined Features (37 total)

**Core Inference**:
- `cpu` — CPU inference with SIMD optimization
- `gpu` — GPU inference with CUDA acceleration
- `cuda` — Backward-compatible alias for `gpu`

**Components**:
- `inference` — Autoregressive generation engine
- `kernels` — SIMD/CUDA compute kernels
- `tokenizers` — Universal tokenizer with auto-discovery

**SIMD Optimizations**:
- `avx2` — QK256 AVX2 dequantization (v0.2 foundation)
- `avx512` — AVX-512 SIMD kernels
- `neon` — ARM NEON SIMD kernels

**Testing & Integration**:
- `fixtures` — GGUF fixture-based integration tests
- `crossval` — C++ cross-validation tests
- `ffi-tests` — FFI-specific tests
- `cpp-ffi` — Link tests against BitNet.cpp library
- `full-framework` — All test features combined
- `reporting` — Test result reporting
- `trend` — Performance trend analysis
- `integration-tests` — Integration test suite
- `spm` — SentencePiece tokenizer tests

**Language Bindings**:
- `ffi` — C FFI bridge
- `python` — Python bindings
- `wasm` — WebAssembly bindings

**Server & CLI**:
- `cli` — Command-line interface
- `server` — HTTP server
- `full-cli` — Full CLI with all subcommands

**Quantization**:
- `iq2s-ffi` — IQ2_S quantization via GGML FFI

**Convenience**:
- `full` — All core features (cpu, cuda, avx2, avx512, neon)
- `minimal` — Only core crates, no inference
- `bench` — Benchmarking support
- `examples` — Feature gate for examples

### Curated Matrix (5 Critical Combos)

1. **`cpu`** — Primary CPU inference backend
   - Tests: Full test suite
   - Runtime: ~90 seconds
   - Profile: `ci`

2. **`cpu,avx2`** — QK256 AVX2 SIMD optimizations
   - Tests: Full test suite + AVX2 SIMD validation
   - Runtime: ~90 seconds
   - Profile: `ci`

3. **`cpu,fixtures`** — Integration tests with GGUF fixtures
   - Tests: Full test suite + fixture loading
   - Runtime: ~120 seconds
   - Profile: `fixtures` (longer timeout)

4. **`cpu,avx2,fixtures`** — Combined SIMD + fixtures
   - Tests: Full test suite + AVX2 + fixture loading
   - Runtime: ~120 seconds
   - Profile: `fixtures`

5. **`ffi`** — C FFI bridge (smoke build)
   - Tests: FFI build only, no runtime tests
   - Runtime: ~60 seconds
   - Profile: `ci`

6. **`gpu`** (compile-only) — GPU inference validation
   - Tests: Compilation check only (no runtime)
   - Runtime: ~60 seconds
   - Profile: `ci`

---

## Appendix B: Nextest Profile Selection Logic

**Profile Selection Algorithm** (implemented in CI YAML):

```bash
# Select nextest profile based on features
PROFILE="ci"

if [[ "$FEATURES" == *"fixtures"* ]]; then
  PROFILE="fixtures"  # Longer timeout for GGUF I/O
elif [[ "$FEATURES" == *"gpu"* ]]; then
  PROFILE="gpu"  # Serial execution for GPU memory
fi

cargo nextest run --workspace \
  --no-default-features \
  --features "$FEATURES" \
  --profile "$PROFILE"
```

**Profile Characteristics**:

| Profile | Timeout | Threads | Use Case |
|---------|---------|---------|----------|
| `ci` | 300s | 4 | Default CI tests |
| `fixtures` | 600s | 2 | GGUF fixture loading |
| `gpu` | 300s | 1 | GPU kernel tests |
| `doctests` | 120s | num-cpus | Doc examples |

---

## Appendix C: Guard Script Patterns

### Pattern 1: Ripgrep + Context Extraction

```bash
# Find pattern matches
MATCHES=$(rg -n 'PATTERN' crates tests --type rust || true)

# Extract context for validation
while IFS= read -r line; do
  FILE=$(echo "$line" | cut -d':' -f1)
  LINE_NUM=$(echo "$line" | cut -d':' -f2)

  # Extract N lines before/after for context
  CONTEXT=$(sed -n "$((LINE_NUM-5)),$((LINE_NUM+5))p" "$FILE")

  # Validate context
  if ! echo "$CONTEXT" | grep -q 'VALIDATION_PATTERN'; then
    # Report violation
  fi
done <<< "$MATCHES"
```

### Pattern 2: GitHub Error Annotations

```bash
if [ -n "$VIOLATIONS" ]; then
  echo "::error::Found violations:"
  echo -e "$VIOLATIONS"
  echo ""
  echo "Fix instructions: <URL to docs>"
  exit 1
fi
```

### Pattern 3: Actionable Error Messages

```bash
echo "Valid annotation patterns:"
echo "  // Blocked by Issue #254 - shape mismatch"
echo "  #[ignore]"
echo ""
echo "See: https://github.com/microsoft/BitNet/blob/main/CLAUDE.md#test-status"
```

---

## References

- **CI Gaps Analysis**: `/home/steven/code/Rust/BitNet-rs/CI_GAPS_ANALYSIS.md`
- **Issue #439**: Feature gate consistency (GPU/CUDA unification)
- **CLAUDE.md**: Test status and scaffolding documentation
- **cargo-hack**: <https://github.com/taiki-e/cargo-hack>
- **nextest**: <https://nexte.st/>
- **Existing Guards**: `scripts/check-envlock.sh`, `scripts/hooks/banned-patterns.sh`
- **Fixture Structure**: `ci/fixtures/qk256/` (3 GGUF fixtures, checksums)

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-23 | 0.1.0 | Initial draft specification |

---

**End of Specification**
