# BITNET-SPEC-FALCON-E-FAMILY-I2S

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no I2_S runtime claim until layout proof passes
Policy impact: no policy exception

## Required I2_S definition

Falcon-E I2_S proof must define:

```text
Falcon-E GGUF metadata required for I2_S
QK256 / grouped block layout verification
weight scale semantics
activation quantization to I8_S
act_scale
act_sum
integer dot correction
tail-column behavior
row stride behavior
embedding policy
LM head / tied-head policy
compatibility with existing qk256-scalar-i8s-scaled-gemv
```

## Kernel IDs

Kernel IDs may alias existing QK256 kernels only after compatibility proof:

```text
falcon-e-i2s-scalar-reference-gemv
falcon-e-i2s-avx2-gemv
falcon-e-i2s-avx512-gemv
falcon-e-i2s-cuda-gemv
falcon-e-i2s-apple-neon-gemv
falcon-e-i2s-opencl-gemv
```

## Layout proof requirements

- Synthetic fixtures for exact block packing, scale layout, row stride, and tail
  columns.
- At least one real Falcon-E tensor-role fixture when an artifact is available.
- Scalar oracle output with deterministic comparison tolerance.
- Explicit statement whether each existing QK256 kernel may be used unchanged,
  used with adapter metadata, or rejected.

## Hard rule

```text
Existing Microsoft 2B QK256 proof does not automatically prove Falcon-E I2_S compatibility.
```
