# BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B conversion](BITNET-SPEC-B158-3B-CONVERSION.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; TL layout contract only
Policy impact: no policy exception

## Purpose

Define what BitNet-rs must know before it can load, fixture, or execute TL1/TL2
artifacts for the 3B lane. This spec is intentionally stricter than a support
matrix row: kernels may not execute until the route layout is known and a scalar
oracle exists.

## Layout fields to specify before runtime proof

A TL1 or TL2 artifact-authority decision must define:

- TL1 tensor layout and tensor role mapping;
- TL2 tensor layout and tensor role mapping;
- bit packing order and values for ternary weights;
- lookup-table semantics and table indexing;
- weight scale and group scale semantics;
- activation type expected by TL kernels;
- embedding quantization policy;
- row stride, block size, and alignment;
- tail behavior for non-multiple rows and columns;
- endianness;
- GGUF metadata required to identify `tl1` or `tl2` routes;
- explicit differences from `I2_S` and QK256.

## Required kernel IDs

The route registry and receipts must use stable kernel IDs:

- `tl1-scalar-reference-gemv`
- `tl1-neon-reference-gemv`
- `tl2-scalar-reference-gemv`
- `tl2-avx2-reference-gemv`
- `tl2-avx512-reference-gemv`
- `tl2-cuda-reference-gemv`

Additional OpenCL, Metal, or platform-specific IDs may be added later only after
scalar TL fixture coverage exists for the same route family.

## Fixture requirements

TL fixtures must include:

- tiny synthetic TL1 and TL2 tensors independent of 3B model binaries;
- row stride and tail cases;
- lookup-table and scale cases;
- known output vectors generated from the scalar definition;
- negative fixtures that prove `I2_S`/QK256 code paths are not called.

## Hard rules

- TL1/TL2 are not QK256.
- TL1/TL2 fixtures must not call `I2_S` scalar or CUDA QK256 kernels.
- No TL accelerator work may start before a scalar TL oracle exists for that
  route family.
- TL1 proof does not prove TL2, and TL2 proof does not prove TL1.
