# ADR-009: CPU Reference Testing for Every GPU Kernel

**Status**: ACCEPTED
**Date**: 2025-06-24
**Context**: Testing strategy for GPU kernel correctness

---

## Context

GPU kernels are notoriously difficult to debug.  Differences in floating-
point rounding, driver versions, and hardware errata can produce subtly
wrong results that are invisible in unit tests but corrupt model output.

BitNet-rs needs a testing strategy that:

1. Works **without** GPU hardware (CI runners, developer laptops)
2. Catches numerical drift across driver or hardware changes
3. Validates correctness, not just "doesn't crash"

---

## Decision

**Every GPU kernel must have a corresponding CPU reference implementation
used for cross-validation testing.**

The CPU reference is a simple, readable, scalar implementation of the same
mathematical operation.  It is the single source of truth for expected
output.

---

## Rationale

### 1. Test Without Hardware
CPU reference tests run on any machine.  CI can validate kernel
correctness on standard ubuntu-22.04 runners without GPU instances,
keeping CI costs low.

### 2. Cross-Validate GPU Output
When GPU hardware is available (local dev, GPU CI lane), the test
harness runs both the GPU kernel and the CPU reference on the same
input, then asserts the outputs match within a configurable tolerance
(`correlation > 0.999`, `max_relative_error < 1e-3`).

### 3. Regression Detection
CPU reference outputs are pinned as insta snapshots (see ADR for snapshot
testing).  Any change to the reference implementation or its inputs
is caught immediately, preventing "drift by a thousand commits."

### 4. Documentation by Example
The CPU reference serves as executable documentation of what the kernel
**should** compute.  New contributors can read the scalar code without
understanding OpenCL, CUDA, or Vulkan.

---

## Consequences

### Positive
- ✅ Kernel correctness tested on every PR (no GPU required)
- ✅ GPU vs CPU cross-validation when hardware is available
- ✅ Snapshot-pinned reference outputs catch silent drift
- ✅ CPU reference doubles as documentation
- ✅ Property tests can use the CPU reference as an oracle

### Negative
- ⚠️ Dual maintenance: kernel logic exists in both GPU and CPU code.
  Changes to the kernel algorithm must be mirrored in the reference.
- ⚠️ CPU reference may mask GPU-specific bugs that only appear with
  GPU-specific input patterns (e.g., warp divergence, bank conflicts).
- ⚠️ Tolerance thresholds must be carefully tuned per kernel to avoid
  false positives (too tight) or false negatives (too loose).

### Mitigations
1. **Property Tests**: Use `proptest` to generate random inputs and
   verify invariants (e.g., softmax sums to 1.0, matmul is associative)
   against the CPU reference.
2. **Golden Output Tests**: Pin a small set of known-good outputs per
   kernel to catch both GPU and CPU regressions.
3. **Tolerance Registry**: A per-kernel tolerance table in
   `crates/bitnet-kernels/tests/support/` defines acceptable error
   bounds, reviewed and updated when algorithms change.
4. **GPU-Only Stress Tests**: Separate `#[ignore]` tests exercise
   GPU-specific patterns (large work-groups, shared memory, subgroup
   operations) that the CPU reference cannot cover.

---

## Implementation

Each kernel follows this test structure:

```rust
/// CPU reference: simple scalar implementation
fn cpu_matmul_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // Triple-loop matmul — clear, correct, slow
    ...
}

#[test]
fn matmul_cpu_reference_correctness() {
    let a = deterministic_input(m * k, seed);
    let b = deterministic_input(k * n, seed + 1);
    let expected = cpu_matmul_reference(&a, &b, m, n, k);
    insta::assert_debug_snapshot!(expected);
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "Requires GPU hardware"]
fn matmul_gpu_matches_cpu_reference() {
    let a = deterministic_input(m * k, seed);
    let b = deterministic_input(k * n, seed + 1);
    let expected = cpu_matmul_reference(&a, &b, m, n, k);
    let actual = gpu_matmul(&a, &b, m, n, k);
    assert_correlation(&expected, &actual, 0.999);
}
```

---

## Alternatives Considered

### Test GPU Kernels Only on GPU Hardware
**Rejected**: Would require GPU CI runners for every PR.  Expensive and
fragile (driver updates break tests).

### Use Third-Party Math Libraries as Reference
**Rejected**: Adds large dependencies (`ndarray`, `faer`).  The scalar
CPU reference is trivial to implement and has zero dependencies.

### Compare Against C++ Reference (bitnet.cpp)
**Complementary**: The `crossval` crate already validates end-to-end
against bitnet.cpp.  CPU reference testing validates individual kernels
at a finer granularity.

---

## References

- Snapshot regression tests: `crates/bitnet-kernels/tests/gpu_snapshot_regression.rs`
- Cross-validation framework: `crossval/`
- `KernelProvider` trait: `crates/bitnet-kernels/src/lib.rs`

---

## Changelog

- **2025-06-24**: Initial decision
