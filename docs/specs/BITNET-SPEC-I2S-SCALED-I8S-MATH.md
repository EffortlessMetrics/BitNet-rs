# BITNET-SPEC-I2S-SCALED-I8S-MATH

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

Production I2_S claims require explicit selected kernel, selected route, selected backend, fallback status, and proof-family booleans for strict accelerated proofs.

## Production math semantics

`q = quantize_row_i8_s(x)`

`act_scale = 127 / max(abs(x), 0.00001)`

`act_sum = sum(q)`

`int_dot = dot(i2_s_codes, q)`

`output = (int_dot - act_sum) / act_scale * weight_scale`

No-scale F32 dequant GEMV is diagnostic only and cannot satisfy production answer-readiness.
