# BitNet CPU Scalar Parity Contract

Status: draft
Owner: cpu-proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md; docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md
Linked ADRs: n/a
Linked plan: plans/cpu-scalar/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar as the CPU oracle used by optimized lanes.
Policy impact: Tolerance changes must update the parity policy instead of being introduced ad hoc.

## Purpose

This specification defines scalar as the correctness oracle for BitNet CPU packed
execution. Optimized lanes compare to scalar. Scalar does not compare to
optimized lanes for correctness.

## Parity levels

| Level | Proof |
| --- | --- |
| byte layout | exact |
| block unpack | exact |
| integer dot | exact |
| scaled I8S output | exact or documented scalar tolerance |
| model logits | bounded top-k/token evidence |
| generated IDs | exact greedy equality where comparing scalar variants |
| answer text | quality-gated |

Exact scalar-vs-scalar comparisons must remain exact unless scalar semantics
change through a dedicated spec and proof update. New tolerances must not be
invented inside runtime PRs; they must update the repository parity policy and
carry receipt-backed justification.

## Oracle rule

Scalar is the truth plate for accelerated packed kernels:

```text
AVX2 / AVX-512 / NEON / CUDA / OpenCL / future packed lanes -> compare to scalar
scalar -> does not depend on accelerated output for correctness
```

An optimized kernel may be faster, but speed cannot override scalar parity. If
an optimized lane diverges from scalar, the divergence must be classified before
claiming correctness. If scalar itself changes, the change must include scalar
fixtures, answer or scoped receipts, and a rollback path.

## Required evidence fields

Parity receipts must preserve:

```text
model_sha256
tokenizer_source
prompt bytes or prompt file identity
prompt token IDs
generated token IDs
decoded text
requested backend
selected backend
requested kernel
selected kernel
fallback_used
first divergence or null
max_abs_error and mean_abs_error where logits/activation tensors are compared
top-k/logit evidence where available
```

## Answer-quality boundary

Passing a tiny answer corpus can prove that the selected scalar lane satisfies a
specific answer gate for a specific model/tokenizer/prompt corpus. It must not be
used as a broad chat-quality claim.
