# I2_S Source Map

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0015-i2s-productization.md
Linked specs: docs/specs/BITNET-SPEC-I2S-*.md
Linked ADRs: n/a
Linked plan: plans/i2s/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

## Purpose

This source map defines the I2_S/QK256 lane as a first-class BitNet route. It records what is already proven, what remains unproven, and the hard claim boundaries.

## Current state summary

- Official Microsoft 2B I2_S/QK256 row remains `product_cli_ready` with `reference_good`, `cpu_answer_ready`, `accelerator_answer_ready`, and `bitnet_packed_i2s_qk256_proof` true.
- Speedup, full residency, and broad server readiness remain false.
- Answer-artifact gate remains mandatory before answer/backend claims.

## Hard boundaries

- I2_S/QK256 proof is not TL1/TL2 proof.
- Dense SLM proof is not BitNet packed I2_S proof.
- No-scale F32 QK256 paths are diagnostic/reference only.
- Production I2_S route uses scaled I2_S × I8_S math.
- Model-family support is per artifact and cannot be inherited from official 2B.
