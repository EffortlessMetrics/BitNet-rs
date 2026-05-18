# BITNET-SPEC-INTEL-GPU-DENSE-SLM

## Purpose

Define Arc/OpenVINO GPU dense SLM support without conflating it with BitNet
QK256/I2_S, native OpenCL, A770, CPU, or NPU proof.

## Initial target

Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric export on Lunar Lake Arc 140V
`GPU.0`.

## Proof ladder

1. Export manifest.
2. Runtime/device identity.
3. OpenVINO GPU bounded smoke.
4. Operator ask.
5. Corpus v2.
6. Phase timing.
7. Profile comparison.
8. Promotion review.
9. Model status.
10. Optional server exact-profile proof.

## Promotion rule

OpenVINO GPU can be promoted only per profile after `fallback=false`, quality
passes for that profile, profile timing is applicable, benchmark-qualified
advantage exists, telemetry context is present or explicitly unavailable, and
generated-token limitations are recorded.

Current known blockers include corpus quality failures, missing direct generated
token IDs in some OpenVINO paths, missing prompt-token timing applicability,
incomplete phase splits, missing profile regression bundle coverage, and no
benchmark-qualified speed/power advantage for candidate GPU routes.
