# Reference Topology

This document defines validation authority across scalar CPU code, architecture
CPU references, Apple Silicon lanes, x86 lanes, and accelerator lanes. It exists
to keep correctness, product UX, and hardware acceleration claims separate.

> Claim boundary: this topology defines validation authority and required
> comparison relationships. It does not assert that every listed accelerator is
> currently validated or product-ready. Current hardware/model support must come
> from active receipts, model coverage, status docs, specs, and claim gates.

## Core Rule

Scalar is the smallest correctness oracle. AVX2 and NEON are the architecture
reference paths. Hardware acceleration claims must validate against the nearest
architecture CPU reference before they can be promoted to user-facing support.

```text
scalar
  -> validates AVX2 and NEON

AVX2
  -> x86 CPU reference
  -> validates A770, AVX-512, CUDA, and Lunar Lake GPU/NPU paths

NEON
  -> Apple/ARM CPU reference
  -> validates Apple Silicon Metal, MPSGraph, future ANE paths, and constrained ARM paths

M4 Mac mini + MacBook
  -> Apple Silicon cross-reference pair
  -> validates Apple Silicon behavior across stable desktop power and mobile/thermal hardware
```

Scalar remains important because it is simple enough to debug and audit. It is
not the long-term comparison target for every accelerator. Once scalar has
validated the architecture reference path, accelerators should compare against
the CPU reference nearest to their platform.

## Lane Authority

| Lane | Authority |
|---|---|
| Scalar | Minimal correctness oracle for CPU reference implementations. |
| Kaby Lake / AVX2 | x86 SLM and BitNet reference lane for x86 CPU and accelerator validation. |
| AVX-512 | High-end x86 CPU acceleration that must validate against AVX2 behavior before support claims. |
| CUDA | NVIDIA acceleration that must validate against the x86 CPU reference before support claims. |
| Intel A770 | Intel GPU acceleration that must validate against the x86 CPU reference before support claims. |
| Lunar Lake | Intel CPU/GPU/NPU platform that must validate against local x86 CPU and AVX2 behavior before support claims. |
| NEON | Apple/ARM CPU reference for Apple Silicon and constrained ARM validation. |
| M4 Mac mini | Stable Apple Silicon dense SLM product, reference, and performance lane. |
| MacBook Apple Silicon | Mobile Apple Silicon cross-reference and larger-artifact exploration lane. |
| Apple Metal | Apple acceleration path that must validate against NEON and scoped receipts before support claims. |
| MPSGraph | Apple graph/reference evidence unless target resolution proves more. |
| Future ANE | Requires explicit resolved-target proof; MPSGraph visibility alone is insufficient. |

## Model Family Boundaries

Dense SLM evidence and BitNet evidence are not interchangeable.

```text
Qwen2.5 0.5B Instruct
  = dense regular SLM
  = Apple Silicon user-facing local-answer baseline
  = validates Mac UX, receipts, model cache, warm sessions, CLI, and quality harness

BitNet b1.58 / Falcon-E / 1bitLLM
  = 1-bit / 1.58-bit model family
  = validates BitNet kernels, I2_S/TL1/TL2 layouts, BitLinear paths, and ternary execution
```

Qwen success on the M4 Mac mini proves the dense Mac runtime path is useful. It
does not prove BitNet local-answer quality, QK256 on Apple Silicon, full Metal
model inference, Neural Engine execution, or 1-bit math.

## Promotion Rules

Acceleration claims move through these gates:

1. Scalar correctness validates the architecture CPU reference.
2. AVX2 or NEON validates platform-local CPU behavior.
3. Accelerator output validates against the nearest CPU reference.
4. Greedy output parity or bounded numeric parity is receipt-backed.
5. Fallback status is explicit and cannot count as acceleration.
6. Timing is recorded only for the exact model, prompt/profile, backend, machine, and run settings.
7. User-facing support is documented only after the relevant receipt gate passes.

Examples:

```text
CUDA:
  validate against AVX2 / AVX-512 reference behavior, not only scalar.

Intel A770:
  validate against AVX2 and local x86 CPU behavior.

Apple Metal:
  validate against NEON and record phase-level fallback status.

MacBook Apple Silicon:
  cross-check M4 Mac mini dense SLM behavior and run larger artifact sweeps,
  but do not replace the M4 performance envelope with one mobile run.
```

## Apple Silicon Split

The Apple lanes have distinct jobs:

```text
M4 Mac mini dense SLM:
  stable product/performance lane for Qwen-backed local answers,
  model cache UX, Mac CLI, warm sessions, release profiles, and phase-scoped Metal evidence.

MacBook Apple Silicon:
  mobile cross-reference and larger-artifact lane for dense SLM mirroring and BitNet candidate sweeps.

Apple BitNet:
  artifact-qualified 1-bit / 1.58-bit proof lane.
  It requires reference-good model/tokenizer authority and strict backend receipts before local-answer claims.
```

The M4 Mac mini should be used for stable, repeatable Apple Silicon proof and
operator UX. The MacBook should be used for cross-checking mobile Apple Silicon
behavior and for larger model/artifact exploration when storage and thermal
context permit.

## Claim Boundaries

Do not claim:

```text
BitNet local-answer quality from dense Qwen evidence
QK256 support from dense SLM or I2_S/TL1 evidence
full Apple Metal inference from phase-scoped Metal receipts
Neural Engine execution from MPSGraph visibility
broad M4 or Apple Silicon performance from one machine/profile
MacBook performance regression from M4-only receipts, or the reverse
```

Do claim only what the receipt proves:

```text
selected backend
requested backend
runtime API
model and tokenizer authority
kernel family or dense model family
execution phase
fallback status
machine context
timing scope
quality or parity result
```
