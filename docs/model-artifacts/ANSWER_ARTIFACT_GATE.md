# Answer Artifact Gate

## Purpose

Answer-readiness claims need a model artifact that is known to produce coherent
short answers before any CPU, CUDA, Apple, NPU, SLM, or server lane can claim a
usable local-answer path.

A GGUF can be structurally valid and still fail answer readiness. Structural
loading proves that the bytes can be parsed. Answer readiness also requires
tokenizer and pre-tokenizer authority, prompt-template authority, a passing
reference runner, deterministic prompt-suite output, and an exact artifact hash.

This gate is shared across hardware lanes so each backend does not rediscover
the same model-artifact failure and misattribute it to its own kernels.

## Required States

Machine-readable manifests must use one of these states:

| State | Meaning |
|---|---|
| `unknown` | Artifact is known by name only. It is not valid evidence for answer claims. |
| `structurally_valid` | Artifact loads or passes structural checks, but answer quality is not proven. |
| `rejected_missing_tokenizer_authority` | Tokenizer or pre-tokenizer metadata is missing or ambiguous for answer claims. |
| `rejected_reference_runner_failed` | A reference runner could not produce coherent output or could not run the artifact. |
| `rejected_prompt_suite_failed` | The deterministic prompt suite failed under the reference runner. |
| `answer_ready` | The artifact passes this gate and can unblock backend answer-readiness work. |
| `blocked` | Search or regeneration is blocked; do not claim coherent local answers. |

Project-specific manifests may retain more detailed legacy statuses, but they
must map to one of these shared states before they are used as a cross-lane
precondition.

## Answer-Ready Requirements

An `answer_ready` artifact must record:

- Repository or source path.
- Exact file name.
- Exact SHA256.
- Byte size.
- Format and architecture.
- Quantization family.
- Tokenizer source.
- Tokenizer model family.
- Pre-tokenizer authority or an explicit compatibility decision.
- Prompt-template family and stop-token policy.
- Reference runner name and version or commit.
- Reference runner command shape.
- Deterministic prompt suite path.
- Per-prompt pass/fail output summary.

The deterministic prompt suite must include constrained prompts such as
`math_2_plus_2` and `capital_france`. Constrained prompts must pass their
explicit gates. Open prompts need readable, non-empty, printable UTF-8 output
without raw special-token garbage or repetition failures.

## Claim Boundaries

Allowed before this gate passes:

- Structural GGUF validity.
- Loader and tokenizer diagnostics.
- Backend execution proof.
- CPU/CUDA/Metal/NPU diagnostic receipts marked `diagnostic_only`.
- Failure classification such as `model_artifact_blocked`.

Not allowed before this gate passes:

- Coherent local-answer claims.
- CPU answer-readiness claims.
- CUDA answer-readiness claims.
- Apple M4 local-answer success claims.
- NPU, SLM, server, or other hardware answer claims.
- Speedup claims based on generated text quality.

## Current Shared Blocker

The current Microsoft BitNet I2_S GGUF is structurally valid, but it is rejected
for answer-readiness claims because reference-runner evidence records missing
pre-tokenizer authority and deterministic prompt-suite failure.

Shared manifests:

- `ci/model-artifacts/artifact-manifest.toml`
- `ci/model-artifacts/rejected-artifacts.toml`

Apple-local evidence remains at:

- `ci/quality/apple-m4-local-answer-model-artifacts.toml`

## Hardware-Lane Use

CPU and accelerator lanes can still run diagnostics against rejected artifacts,
but receipts must keep claims narrow:

```text
claim = diagnostic_only
answer_readiness_claim = false
speedup_claim = false
```

Once an artifact becomes `answer_ready`, hardware lanes may use it for their own
strict answer-readiness gates while preserving lane-specific backend, runtime,
fallback, kernel, and timing proof requirements.
