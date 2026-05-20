# BITNET-PROP-0015: I2_S productization

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: self
Linked specs: docs/specs/BITNET-SPEC-I2S-*.md
Linked ADRs: n/a
Linked plan: plans/i2s/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

## Thesis

I2_S/QK256 is the current primary production BitNet route in BitNet-rs. It must be governed as a model-family/kernel-family/backend-family contract, not a generic quantization helper.

## Why now

- Official Microsoft 2B I2_S route is the current answer-ready/product CLI lane.
- Route claims need stronger layout authority and precise scaled-kernel identity.
- Model-family expansion must stay artifact-gated and conservative.

## Non-goals

- No TL1/TL2 promotion via I2_S proof.
- No dense SLM inheritance.
- No global speedup/residency claims without exact-profile proof.
