# BITNET-SPEC-CPU-SCALAR-PARITY: CPU Scalar Parity Contract

Status: proposed

Linked plan:
[CPU scalar implementation plan](../../plans/cpu-scalar/implementation-plan.md)

Linked specs:
[CPU scalar kernel contract](BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md),
[CPU scalar hot-path contract](BITNET-SPEC-CPU-SCALAR-HOTPATH.md)

## Purpose

This spec defines scalar as the CPU oracle for packed BitNet QK256/I2_S work.
Optimized lanes compare to scalar. Scalar does not compare to optimized lanes
for correctness.

## Parity Direction

Correctness direction is one-way:

```text
optimized lane -> scalar oracle
scalar oracle -/-> optimized lane authority
```

AVX2, AVX-512, NEON, CUDA, Metal, OpenCL, OpenVINO, and future lanes may use
scalar receipts as reference evidence. A mismatch is an optimized-lane defect or
a classified model/prompt/tokenizer/shared-decode issue until a scalar defect is
proven by fixtures or independent reference evidence.

## Scalar Parity Levels

| Level | Proof |
| --- | --- |
| byte layout | exact |
| block unpack | exact |
| integer dot | exact |
| scaled I8_S output | exact or documented scalar tolerance |
| model logits | bounded top-k/token evidence |
| generated IDs | exact greedy equality where comparing scalar variants |
| answer text | quality-gated |

No new tolerance may be introduced by an implementation PR unless the parity
policy is deliberately updated and linked from the proof receipt.

## Required Evidence

Scalar parity receipts must preserve:

```text
model SHA-256
tokenizer source and strictness
prompt bytes or prompt ID
prompt token IDs
generated token IDs
decoded text
greedy/sampling settings
requested backend
selected backend
requested kernel
selected kernel
fallback_used
reference kernel or receipt path
first divergence, or null when none
```

For model-level comparisons, first-step top-k/logit evidence should be captured
when available so prompt/tokenizer/template issues can be separated from shared
decode math or backend-specific execution.

## Strict Scalar Equality Rules

When comparing scalar variants under deterministic greedy decoding:

- prompt IDs must match exactly;
- generated IDs must match exactly unless the receipt records the first
  divergence and classification;
- decoded text must match when generated IDs match;
- `fallback_used` must be `false`;
- selected kernels must use precise scalar IDs.

## Claim Boundary

A passing scalar parity proof can support claims about scalar oracle status for
the exact artifact, prompt set, tokenizer, backend, and profile tested. It must
not promote broad chat quality, accelerated speed, GPU/NPU execution, dense SLM
support, or server readiness.
