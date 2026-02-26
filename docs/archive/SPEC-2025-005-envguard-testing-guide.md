# SPEC-2025-005: Document Environment Testing Standards with EnvGuard Guide

**Status**: Draft
**Created**: 2025-10-23
**Priority**: P1
**Category**: Documentation & Test Standards
**Related Issues**: None
**Related PRs**: #475

---

## Executive Summary

Document environment variable testing standards with comprehensive EnvGuard usage guide in `docs/development/test-suite.md`. This addresses critical test hygiene gap identified in audit: 39 tests mutating environment variables WITHOUT proper `#[serial(bitnet_env)]` guards, creating race conditions in parallel test execution.

**Current State**:
- EnvGuard pattern exists and works (7/7 tests passing from PR #475)
- Pattern documented in CLAUDE.md (lines 345-360) but not in comprehensive dev docs
- 39 tests mutating env vars without `#[serial]` protection (P0 risk)

**Target State**:
- Comprehensive EnvGuard guide in `docs/development/test-suite.md`
- All env-mutating tests use `#[serial(bitnet_env)]` pattern
- Clear examples and anti-patterns documented
- CI check to enforce pattern usage

**Impact**:
- **CI Reliability**: Eliminate race conditions from parallel test execution
- **Developer Guidance**: Clear patterns for env-aware testing
- **Test Safety**: Automatic env cleanup prevents test pollution

---

## Requirements Analysis

### Functional Requirements

1. **FR1: EnvGuard Guide Section**
   - Add "Environment Variable Testing with EnvGuard" section to `docs/development/test-suite.md`
   - Document the `#[serial(bitnet_env)]` pattern with code examples
   - Explain race condition risks in parallel testing
   - Provide anti-patterns (what NOT to do)

2. **FR2: Pattern Examples**
   - Show correct usage: `EnvGuard` + `#[serial(bitnet_env)]`
   - Show incorrect usage: env mutation without guards (anti-pattern)
   - Provide real-world examples from BitNet-rs test suite
   - Document test execution modes (serial vs. parallel)

3. **FR3: CI Integration**
   - Document how to run env-aware tests: `cargo test #[serial(bitnet_env)]`
   - Explain nextest profile for serialized execution
   - Provide validation commands for detecting unguarded env mutations

4. **FR4: Cross-Linking**
   - Link from `ci/README.md` to test-suite.md EnvGuard section
   - Reference from CLAUDE.md (already has brief mention, expand with link)
   - Add to test hygiene checklist in PR template

### Non-Functional Requirements

1. **NFR1: Documentation Quality**
   - Clear explanations with code examples
   - Warnings about common pitfalls
   - Easy-to-follow implementation checklist

2. **NFR2: Developer Experience**
   - Self-service guide (developers can learn without asking)
   - Copy-paste ready code snippets
   - Clear success criteria for each pattern

3. **NFR3: Maintainability**
   - Keep EnvGuard implementation in `tests/helpers/env_guard.rs`
   - Document when to update guide (new env vars, new patterns)

---

## Architecture Approach

### Documentation Structure

**New Section in `docs/development/test-suite.md`**:
```markdown
## Environment Variable Testing with EnvGuard

### Overview

BitNet-rs uses environment variables for runtime configuration (e.g., `BITNET_DETERMINISTIC`, `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE`). Tests that mutate environment variables must use the **EnvGuard pattern** to prevent race conditions in parallel test execution.

### The Problem: Race Conditions

**Without EnvGuard** (ANTI-PATTERN):
```rust
#[test]
fn test_determinism() {
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    // Test logic
    std::env::remove_var("BITNET_DETERMINISTIC");
}
```

**Issue**: When tests run in parallel (default with `cargo test`), multiple tests can mutate the same environment variable simultaneously, causing:
- Flaky test failures (test sees wrong env value)
- Test pollution (env changes leak to other tests)
- Non-deterministic CI failures

### The Solution: EnvGuard + #[serial]

**Correct Pattern**:
```rust
use serial_test::serial;
use tests::helpers::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // Ensures serial execution with other env-mutating tests
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Test logic - env automatically restored on drop
}
```

**Benefits**:
- **Automatic cleanup**: Environment restored when `_guard` drops (even on panic)
- **Serial execution**: `#[serial(bitnet_env)]` prevents parallel execution with other env tests
- **No pollution**: Other tests unaffected by env changes

### Implementation Guide

#### Step 1: Add Dependencies

```toml
# Cargo.toml (workspace or crate-level)
[dev-dependencies]
serial_test = "3.1"
```

#### Step 2: Use EnvGuard Pattern

```rust
use serial_test::serial;
use tests::helpers::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // Critical: prevents parallel execution
fn test_with_env_mutation() {
    // Set environment variable with automatic cleanup
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");

    // Test logic that depends on BITNET_DETERMINISTIC=1
    let result = my_deterministic_function();
    assert_eq!(result, expected_value);

    // No explicit cleanup needed - _guard drops here and restores env
}
```

#### Step 3: Multiple Environment Variables

```rust
#[test]
#[serial(bitnet_env)]
fn test_with_multiple_env_vars() {
    let _guard1 = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    let _guard2 = EnvGuard::new("BITNET_SEED", "42");
    let _guard3 = EnvGuard::new("RAYON_NUM_THREADS", "1");

    // All guards drop in reverse order (LIFO), restoring env correctly
}
```

### Common Pitfalls

#### Pitfall 1: Missing #[serial] Attribute

```rust
// WRONG: EnvGuard without #[serial] - still has race condition risk!
#[test]
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Can still run in parallel with other env-mutating tests
}

// CORRECT: Always use #[serial(bitnet_env)] with EnvGuard
#[test]
#[serial(bitnet_env)]
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Guaranteed serial execution
}
```

#### Pitfall 2: Manual Env Mutation

```rust
// WRONG: Manual set_var without cleanup
#[test]
#[serial(bitnet_env)]
fn test_determinism() {
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    // If test panics, env not restored!
}

// CORRECT: EnvGuard handles cleanup automatically
#[test]
#[serial(bitnet_env)]
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Cleanup guaranteed even on panic
}
```

#### Pitfall 3: Forgetting Guard Lifetime

```rust
// WRONG: Guard dropped too early
#[test]
#[serial(bitnet_env)]
fn test_determinism() {
    {
        let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    } // _guard drops here, env restored immediately!

    // Test logic here sees restored (original) env - BUG!
}

// CORRECT: Guard lives for full test duration
#[test]
#[serial(bitnet_env)]
fn test_determinism() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Test logic - env still set
} // _guard drops here after test completes
```

### Running EnvGuard Tests

```bash
# Run all tests (env-aware tests execute serially)
cargo test --workspace --no-default-features --features cpu

# Run only env-aware tests (serialized via #[serial])
cargo test --workspace --no-default-features --features cpu -- serial

# Run with nextest (recommended - better parallel scheduling)
cargo nextest run --workspace --no-default-features --features cpu

# Force single-threaded execution (for debugging race conditions)
cargo test --workspace --no-default-features --features cpu -- --test-threads=1
```

### CI Configuration

BitNet-rs CI uses nextest with fixed thread count for deterministic execution:

```yaml
# .github/workflows/ci.yml
- name: Run tests with nextest
  run: |
    cargo nextest run --profile ci --workspace --no-default-features --features cpu
```

**Nextest Config** (`.config/nextest.toml`):
```toml
[profile.ci]
test-threads = 4           # Fixed thread count for reproducibility
retries = 0                # No retries (tests must pass first time)
fail-fast = true           # Stop on first failure
```

### When to Use EnvGuard

**Use EnvGuard when**:
- Setting environment variables in tests (`BITNET_*`, `RAYON_*`, etc.)
- Testing environment-dependent behavior (determinism, strict mode, GPU fallback)
- Validating environment variable parsing/validation

**Don't use EnvGuard when**:
- Test doesn't mutate environment variables
- Testing with command-line flags (not env vars)
- Using mocked configuration (no real env mutation)

### BitNet-rs Environment Variables

**Common Env Vars in Tests** (all require EnvGuard):
- `BITNET_DETERMINISTIC`: Enable deterministic inference
- `BITNET_SEED`: Set random seed for reproducibility
- `BITNET_STRICT_MODE`: Enable strict validation (fail on warnings)
- `BITNET_GPU_FAKE`: Override GPU detection for testing
- `BITNET_GGUF`: Model path override
- `RAYON_NUM_THREADS`: Control parallelism

**See also**: `docs/environment-variables.md` for complete reference

### EnvGuard Implementation

**Location**: `tests/helpers/env_guard.rs`

**Source** (for reference):
```rust
pub struct EnvGuard {
    key: String,
    old_value: Option<String>,
}

impl EnvGuard {
    pub fn new(key: &str, value: &str) -> Self {
        let old_value = std::env::var(key).ok();
        std::env::set_var(key, value);
        EnvGuard {
            key: key.to_string(),
            old_value,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.old_value {
            Some(val) => std::env::set_var(&self.key, val),
            None => std::env::remove_var(&self.key),
        }
    }
}
```

### Real-World Examples

**Example 1: Deterministic Inference Test**
```rust
#[test]
#[serial(bitnet_env)]
fn test_deterministic_inference_with_seed() {
    let _det_guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    let _seed_guard = EnvGuard::new("BITNET_SEED", "42");
    let _threads_guard = EnvGuard::new("RAYON_NUM_THREADS", "1");

    // Run inference twice - should get identical results
    let result1 = run_inference();
    let result2 = run_inference();
    assert_eq!(result1, result2);
}
```

**Example 2: Strict Mode Validation Test**
```rust
#[test]
#[serial(bitnet_env)]
fn test_strict_mode_fails_on_warnings() {
    let _guard = EnvGuard::new("BITNET_STRICT_MODE", "1");

    // Should fail when model has suspicious LayerNorm weights
    let result = load_model_with_warnings();
    assert!(result.is_err(), "Strict mode should fail on warnings");
}
```

**Example 3: GPU Fallback Test**
```rust
#[test]
#[serial(bitnet_env)]
fn test_gpu_fallback_when_unavailable() {
    let _guard = EnvGuard::new("BITNET_GPU_FAKE", "none");

    // Should fall back to CPU when GPU faked unavailable
    let backend = select_backend();
    assert_eq!(backend, Backend::Cpu);
}
```

### Validation Checklist

When adding env-mutating tests:
- [ ] Import `serial_test::serial` and `tests::helpers::env_guard::EnvGuard`
- [ ] Add `#[serial(bitnet_env)]` attribute to test
- [ ] Use `EnvGuard::new()` for all env mutations (never `std::env::set_var` directly)
- [ ] Bind guard to `_guard` variable (underscore prefix prevents unused warning)
- [ ] Ensure guard lives for full test duration (not dropped early in scope)
- [ ] Run test with `cargo test <test_name>` to verify serial execution
- [ ] Check for race conditions: `cargo test -- --test-threads=4` (should pass)

### References

- **EnvGuard Implementation**: `tests/helpers/env_guard.rs`
- **Example Tests**: `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`
- **Serial Test Docs**: https://docs.rs/serial_test/latest/serial_test/
- **Environment Variables Reference**: `docs/environment-variables.md`
```

---

## Quantization Strategy

**Not Applicable**: Documentation task with no impact on quantization algorithms.

---

## GPU/CPU Implementation

**Not Applicable**: EnvGuard pattern is backend-agnostic and used for both CPU and GPU tests.

**Note**: The guide includes examples of testing GPU fallback behavior using `BITNET_GPU_FAKE` environment variable.

---

## GGUF Integration

**Not Applicable**: EnvGuard pattern used for testing GGUF loading configuration (e.g., `BITNET_GGUF` path override), but does not affect GGUF parsing itself.

---

## Performance Specifications

**Not Applicable**: Documentation task with no runtime performance impact.

**Test Execution Impact**:
- Serial tests (`#[serial(bitnet_env)]`) run sequentially, slightly slower than parallel
- Typical overhead: <5% for test suites with few env-mutating tests
- Benefit: 100% reliability (no race conditions) justifies small overhead

---

## Cross-Validation Plan

### Documentation Quality Validation

**Readability Check**:
```bash
# 1. Verify section added to test-suite.md
grep -A 5 "Environment Variable Testing with EnvGuard" docs/development/test-suite.md
# Expected: Section exists with overview

# 2. Check code examples present
grep -c "```rust" docs/development/test-suite.md
# Expected: ≥10 code examples (patterns + anti-patterns)

# 3. Verify cross-links
grep "env_guard.rs" docs/development/test-suite.md
grep "environment-variables.md" docs/development/test-suite.md
# Expected: References to implementation and env var docs
```

### Pattern Compliance Validation

**Audit Env-Mutating Tests** (identify non-compliant tests):
```bash
# 1. Find all tests using std::env::set_var
grep -rn "std::env::set_var" --include="*.rs" crates/*/tests/

# 2. Check which tests have #[serial(bitnet_env)]
grep -B 3 "std::env::set_var" --include="*.rs" crates/*/tests/ | grep "#\[serial"

# 3. Identify non-compliant tests (set_var without #[serial])
# Manual review of output from steps 1-2

# Expected: 39 tests identified in audit report need migration
```

### CI Integration Validation

**Test Execution Check**:
```bash
# 1. Run env-aware tests serially
cargo test --workspace --no-default-features --features cpu -- serial
# Expected: All tests pass (no race conditions)

# 2. Run with parallel threads (stress test)
cargo test --workspace --no-default-features --features cpu -- --test-threads=4
# Expected: All tests pass (EnvGuard prevents pollution)

# 3. Run with nextest (production CI mode)
cargo nextest run --profile ci --workspace --no-default-features --features cpu
# Expected: All tests pass with fixed 4-thread execution
```

---

## Feature Flag Analysis

**Not Applicable**: EnvGuard pattern is feature-agnostic and used across all feature configurations.

---

## Testing Strategy

### Documentation Tests

**Code Example Validation** (doc tests):
```rust
// Add to docs/development/test-suite.md as doc tests
/// ```rust,no_run
/// use serial_test::serial;
/// use tests::helpers::env_guard::EnvGuard;
///
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_determinism() {
///     let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
///     // Test logic
/// }
/// ```
```

**Validation**:
```bash
# Run doc tests
cargo test --doc

# Check for doc test failures
cargo test --doc 2>&1 | grep "test result:"
# Expected: All doc tests pass
```

### Pattern Compliance Tests

**EnvGuard Usage Audit Test** (`tests/test_env_guard_compliance.rs` - NEW):
```rust
#[test]
#[ignore] // Manual audit required
fn test_all_env_mutations_use_envguard() {
    // This test requires manual code review
    // Automated check would be complex (needs AST parsing)

    // Manual audit checklist:
    // 1. Find all tests using std::env::set_var
    // 2. Verify each has #[serial(bitnet_env)] attribute
    // 3. Verify each uses EnvGuard (not raw set_var)

    println!("Run manual audit:");
    println!("  grep -rn 'std::env::set_var' crates/*/tests/");
    println!("  grep -B 3 'std::env::set_var' crates/*/tests/ | grep '#\\[serial'");

    // Mark as ignored - this is a maintenance reminder test
}
```

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Incomplete migration** | Medium | High | Provide audit commands to find non-compliant tests |
| **Developer non-compliance** | Medium | Medium | Add PR checklist: "Env-mutating tests use EnvGuard?" |
| **Pattern misuse** | Low | Medium | Comprehensive examples and anti-patterns in guide |
| **CI race conditions** | Low | High | Nextest with fixed thread count; serial execution for env tests |

### Validation Commands

**Risk Validation**:
```bash
# 1. Find all env-mutating tests without EnvGuard (manual audit)
grep -rn "std::env::set_var" --include="*.rs" crates/*/tests/ \
  | grep -v "EnvGuard" \
  | wc -l
# Expected: 39 tests (from audit report) → 0 after migration

# 2. Find tests with EnvGuard but missing #[serial]
grep -rn "EnvGuard::new" --include="*.rs" crates/*/tests/ -A 5 \
  | grep -B 5 "EnvGuard::new" \
  | grep "#\[test\]" \
  | grep -v "#\[serial"
# Expected: No output (all EnvGuard tests have #[serial])

# 3. Verify no race conditions (stress test)
for i in {1..10}; do
  cargo test --workspace --no-default-features --features cpu -- --test-threads=8 \
    || echo "FAILED on iteration $i"
done
# Expected: All iterations pass (no flaky failures)
```

---

## Success Criteria

### Measurable Acceptance Criteria

**AC1: EnvGuard Guide Section Added**
- ✅ Section added to `docs/development/test-suite.md`
- ✅ ≥2000 words covering patterns, anti-patterns, examples
- ✅ ≥10 code examples (Rust code blocks)

**Validation**:
```bash
# Check section exists
grep -A 10 "Environment Variable Testing with EnvGuard" docs/development/test-suite.md
# Expected: Section with overview

# Count words in section
awk '/Environment Variable Testing with EnvGuard/,/^## [^E]/' docs/development/test-suite.md | wc -w
# Expected: ≥2000 words

# Count code examples
awk '/Environment Variable Testing with EnvGuard/,/^## [^E]/' docs/development/test-suite.md | grep -c '```rust'
# Expected: ≥10 code examples
```

**AC2: Cross-Links Established**
- ✅ Link from `ci/README.md` to test-suite.md#envguard
- ✅ Link from `CLAUDE.md` to test-suite.md#envguard
- ✅ Reference from PR template checklist

**Validation**:
```bash
# Check ci/README.md link
grep "test-suite.md.*EnvGuard" ci/README.md
# Expected: Link to EnvGuard section

# Check CLAUDE.md reference
grep "test-suite.md" CLAUDE.md
# Expected: Link to comprehensive guide

# Verify PR template
grep -i "envguard\|environment.*test" .github/PULL_REQUEST_TEMPLATE.md
# Expected: Checklist item for env-aware tests
```

**AC3: Pattern Examples Clear**
- ✅ Correct pattern example with `EnvGuard` + `#[serial]`
- ✅ Anti-pattern examples (what NOT to do)
- ✅ Real-world examples from BitNet-rs codebase
- ✅ Common pitfalls documented

**Validation**:
```bash
# Check for correct pattern
grep -A 10 "Correct Pattern" docs/development/test-suite.md | grep "#\[serial(bitnet_env)\]"
# Expected: Pattern documented

# Check for anti-patterns
grep -A 5 "ANTI-PATTERN\|WRONG:" docs/development/test-suite.md | wc -l
# Expected: ≥3 anti-pattern examples

# Check for real-world examples
grep -A 5 "Real-World Examples" docs/development/test-suite.md
# Expected: Section with actual test examples
```

**AC4: CI Validation Documented**
- ✅ Commands for running env-aware tests
- ✅ Nextest profile configuration explained
- ✅ Audit commands for detecting non-compliant tests

**Validation**:
```bash
# Check CI commands documented
grep "cargo test.*serial" docs/development/test-suite.md
grep "cargo nextest" docs/development/test-suite.md
# Expected: Commands for running serialized tests

# Check audit commands
grep "grep.*set_var" docs/development/test-suite.md
# Expected: Audit commands for finding non-compliant tests
```

---

## Performance Thresholds

**Not Applicable**: Documentation task with no runtime performance impact.

**Documentation Quality Metrics**:
- Section length: ≥2000 words (comprehensive coverage)
- Code examples: ≥10 examples (patterns + anti-patterns)
- Cross-links: ≥3 incoming references (ci/README.md, CLAUDE.md, PR template)

---

## Implementation Notes

### Section Placement in test-suite.md

**Proposed Structure**:
```markdown
# Test Suite Guide

## Test Status Summary
## Running Tests
### Standard Test Execution with cargo test
### Using cargo nextest (Recommended for CI)

## Fixture Management
## Environment Variable Testing with EnvGuard  ← NEW SECTION (insert here)

## GPU Testing
## Cross-Validation Tests
## Performance Tests
## Mutation Testing
```

**Rationale**: Place after "Fixture Management" (related to test setup) and before "GPU Testing" (uses env vars for GPU_FAKE).

### Code Example Style Guide

**Consistent Formatting**:
```rust
// Good: Clear section labels
// CORRECT: Pattern description
#[test]
#[serial(bitnet_env)]
fn test_example() { /* ... */ }

// WRONG: Anti-pattern description
#[test]
fn test_bad_example() { /* ... */ }

// Explanation of why wrong
```

**Annotations**:
- Use `// CORRECT:` for recommended patterns
- Use `// WRONG:` for anti-patterns
- Use `// Note:` for important clarifications
- Use `// Expected:` for validation command outputs

---

## BitNet-rs Alignment

### TDD Practices

✅ **Alignment**: EnvGuard guide supports TDD by ensuring test isolation and reliability.

### Feature-Gated Architecture

✅ **Alignment**: Pattern applies to all feature configurations (CPU, GPU, fixtures).

### Workspace Structure

✅ **Alignment**: EnvGuard implementation in `tests/helpers/` follows workspace conventions.

### Cross-Platform Support

✅ **Alignment**: EnvGuard pattern is platform-agnostic (works on Linux, macOS, Windows).

---

## Neural Network References

**Not Applicable**: EnvGuard pattern is infrastructure-focused, not neural network-specific.

**Note**: The guide includes examples of testing neural network configurations via environment variables (e.g., deterministic inference, strict mode validation).

---

## Related Documentation

- **EnvGuard Implementation**: `tests/helpers/env_guard.rs` (already exists from PR #475)
- **Environment Variables Reference**: `docs/environment-variables.md`
- **Test Suite Guide**: `docs/development/test-suite.md` (needs EnvGuard section)
- **CLAUDE.md**: Brief EnvGuard mention (lines 345-360, needs link to comprehensive guide)
- **Test Audit Report**: `COMPREHENSIVE_TEST_AUDIT_REPORT.md` (identifies 39 non-compliant tests)

---

## Implementation Checklist

**Phase 1: Draft EnvGuard Guide Section** (2 hours)
- [ ] Write overview explaining race condition problem
- [ ] Document correct pattern: `EnvGuard` + `#[serial(bitnet_env)]`
- [ ] Add anti-pattern examples (what NOT to do)
- [ ] Include implementation guide (step-by-step)
- [ ] Document common pitfalls (3+ examples)
- [ ] Add running tests section (CI commands)
- [ ] Include when to use / when NOT to use guidance
- [ ] List BitNet-rs env vars requiring EnvGuard
- [ ] Add real-world examples (3+ from codebase)
- [ ] Include validation checklist

**Phase 2: Add Code Examples** (1 hour)
- [ ] Correct pattern example (basic)
- [ ] Multiple env vars example
- [ ] Anti-pattern: missing #[serial]
- [ ] Anti-pattern: manual set_var
- [ ] Anti-pattern: guard dropped too early
- [ ] Real example: deterministic inference test
- [ ] Real example: strict mode test
- [ ] Real example: GPU fallback test
- [ ] EnvGuard implementation (reference)
- [ ] Doc test examples (validatable)

**Phase 3: Cross-Link Integration** (30 minutes)
- [ ] Add link from `ci/README.md` to test-suite.md#envguard
- [ ] Update `CLAUDE.md` reference (lines 345-360) with link
- [ ] Add checklist to `.github/PULL_REQUEST_TEMPLATE.md`
- [ ] Verify all links with lychee: `lychee docs/development/test-suite.md`

**Phase 4: CI Validation Section** (30 minutes)
- [ ] Document `cargo test -- serial` command
- [ ] Document `cargo nextest run --profile ci` command
- [ ] Add audit commands (find non-compliant tests)
- [ ] Explain nextest profile configuration
- [ ] Include CI workflow example

**Phase 5: Review and Validation** (1 hour)
- [ ] Self-review: Read guide as first-time developer
- [ ] Check all code examples compile (copy-paste test)
- [ ] Verify cross-links work (click-through test)
- [ ] Run audit commands to find non-compliant tests
- [ ] Update word count and example count
- [ ] Commit guide section

**Phase 6: Optional - Migrate Non-Compliant Tests** (3-5 hours)
- [ ] Run audit: `grep -rn "std::env::set_var" crates/*/tests/`
- [ ] Identify 39 non-compliant tests (from audit report)
- [ ] Add `#[serial(bitnet_env)]` to each test
- [ ] Replace `std::env::set_var` with `EnvGuard::new`
- [ ] Run tests to verify no race conditions
- [ ] Commit test migrations

---

## Status

**Current Phase**: Draft Specification
**Next Steps**: Review and approval → Implementation
**Estimated Implementation Time**: 5 hours (guide writing + cross-linking + validation)
**Optional Migration**: +3-5 hours (migrate 39 non-compliant tests)
**Risk Level**: Low (documentation only, no breaking changes)

---

**Last Updated**: 2025-10-23
**Spec Author**: BitNet-rs Spec Analyzer Agent
**Review Status**: Pending
