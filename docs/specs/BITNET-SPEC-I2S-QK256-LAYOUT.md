# BITNET-SPEC-I2S-QK256-LAYOUT

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0015-i2s-productization.md
Linked specs: self
Linked ADRs: n/a
Linked plan: plans/i2s/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

Contract draft for I2_S lane. This spec captures route-specific acceptance, fallback-explicit receipt requirements, and non-inheritance boundaries.

## Hard rule

Production I2_S claims require explicit selected kernel, , and  for strict accelerated proofs.

## Canonical constants

- `QK256_BLOCK_COLS = 256`
- `QK256_PACKED_BYTES_PER_BLOCK = 64`
- `row_stride_bytes = ceil(cols / 256) * 64`

## Canonical layout

- GGML grouped bitplane layout with two 128-value chunks per block.
- 32 bytes per chunk; four 32-value lanes per chunk; high-bit-first lane bit order.
- Code map: `0 -> -1`, `1 -> 0`, `2 -> +1`, `3 -> 0`.

## Acceptance

- layout-core pack/unpack matches quantization grouped layout.
- legacy simple offset/4 helpers must be removed or diagnostic-only.
- tail fixtures: 1, 2, 127, 128, 129, 255, 256, 257, 300, 512, 1024.
