# BitNet CPU Scalar Parity Contract

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar as the packed CPU oracle for parity; does not promote optimized lanes.
Policy impact: New tolerances require a deliberate parity-policy update.

## Purpose

This spec defines scalar parity levels and oracle direction. Optimized lanes
compare to scalar for correctness. Scalar does not compare to optimized lanes for
correctness.

Scalar evidence must be strong enough that AVX2, AVX-512, NEON, CUDA, Metal,
OpenVINO, and future lanes can use scalar receipts as the truth plate for packed
BitNet math after their own lane-specific proof work starts.

## Oracle Direction

The scalar lane is authoritative for packed CPU correctness after the relevant
layout, kernel, fixture, answer-corpus, and long-decode proofs pass:

```text
optimized lane output -> compare against scalar
scalar output -> compare against spec/fixtures/reference artifact, not optimized lane
```

An optimized lane matching scalar is parity evidence for that optimized lane. An
optimized lane disagreeing with scalar is not evidence that scalar is wrong unless
fixture, BitNet.cpp reference, or scalar-spec evidence also points to scalar
failure.

## Parity Levels

| Level | Proof |
| --- | --- |
| byte layout | Exact bytes, offsets, block geometry, row stride, and tail layout match the canonical QK256/I2_S authority. |
| block unpack | Exact code map `0 -> -1`, `1 -> 0`, `2 -> +1`, `3 -> 0` for every position and tail. |
| integer dot | Exact integer accumulation for packed I2_S codes and I8_S activations, including documented wrapping behavior where applicable. |
| scaled I8_S output | Exact or documented scalar tolerance for `(dot - act_sum) / act_scale * weight_scale`; no new tolerance without policy update. |
| model logits | Bounded top-k/token evidence with model, tokenizer, prompt, backend, selected kernel, and fallback fields. |
| generated IDs | Exact greedy equality where comparing scalar variants; explicit divergence classification otherwise. |
| answer text | Quality-gated corpus result with no broad chat-quality claim. |

## Fixture Coverage Requirements

Scaled scalar fixture suites should cover:

```text
cols: 1, 2, 127, 128, 129, 255, 256, 257, 300, 512, 1024
rows: 1, 2, 7, 32
weight_scale: 0.125, 0.5, 1.0
patterns: all 0, all 1, all 2, all 3, cyclic, pseudorandom
activations: zero-ish, constant, ramp, signed, pseudorandom
tails: every non-multiple boundary
```

The fixtures must verify activation quantization, activation sum, integer dot,
scaled output, tail behavior, and repeatability.

## Tolerance Policy

Do not invent new tolerances inside scalar implementation PRs. Any tolerance not
covered by the existing parity policy must be introduced by an explicit parity
policy update that names:

- the operation and data type;
- the scalar and comparator paths;
- max absolute and mean absolute bounds;
- token/logit evidence if model-level behavior is affected;
- why the tolerance does not weaken scalar's oracle role.

## Answer-Corpus Boundary

A tiny answer corpus can prove a strict deterministic smoke and quality gate for
specific prompts. It must not be described as broad chat quality, serving
readiness, or generalized benchmark quality.

Answer-corpus parity artifacts must preserve prompt IDs, generated IDs, decoded
text, tokenizer source, model SHA, selected backend/kernel, fallback status, and
first divergence when comparing variants.

## Acceptance

Scalar parity PRs must include exact fixture evidence or scoped receipt evidence,
strict fallback fields, no hidden dense/dequantized substitution, no new tolerance
without policy update, and a rollback path.
