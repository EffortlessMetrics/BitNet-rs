# Apple BitNet Artifact Sweep Campaign

Campaign ID: `apple-bitnet-artifact-sweep`

Status: active

## Objective

Use the MacBook Apple Silicon lane to qualify 1-bit / 1.58-bit BitNet-family
artifacts before any M4 Mac mini Apple CPU/NEON or Metal BitNet local-answer
claim.

## Why This Exists

The dense Qwen Apple Silicon path is the practical Mac user-facing SLM lane. It
proves Mac UX, model cache, warm sessions, receipts, quality corpus behavior,
and Apple CPU/NEON routing for a regular dense SLM. It does not prove BitNet
model quality or 1-bit math.

The shared model-artifact gate now records that the official Microsoft I2_S
artifact can be answer-ready under external tokenizer pre-tokenizer authority,
but Apple lanes still need their own strict artifact and backend receipts. The
MacBook is the right first Apple machine for larger artifact sweeps because it
is the Apple Silicon cross-reference and larger-artifact lane.

## End State

- Official Microsoft 2B I2_S is accepted or rejected on MacBook under recorded
  tokenizer authority.
- The smaller 0.7B `1bitLLM/bitnet_b1_58-large` candidate is accepted or
  rejected as an Apple BitNet control artifact.
- The 3B 1bitLLM candidate is evaluated only on supported TL diagnostic routes.
- Falcon-E candidates remain secondary BitNet-like family evidence after
  Microsoft and 1bitLLM behavior is understood.
- Accepted candidates feed a separate M4 Mac mini strict Apple CPU/NEON proof
  item; they do not become M4 answer claims by themselves.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| ABAS-001 | proposed | Validate official Microsoft 2B I2_S on MacBook with external tokenizer authority. |
| ABAS-002 | proposed | Evaluate 0.7B `1bitLLM/bitnet_b1_58-large` as the smaller Apple BitNet control. |
| ABAS-003 | proposed | Evaluate 3B only on supported TL1/TL2 diagnostic routes. |
| ABAS-004 | proposed | Evaluate Falcon-E as a secondary BitNet-like family after primary candidates. |
| ABAS-005 | proposed | Hand off the best accepted artifact to a separate M4 strict proof item. |

## Review Policy

Each PR owns one artifact decision. Candidate downloads must stay under cache or
`target/`, rejected candidates should be deleted unless a later item explicitly
keeps them for a bounded diagnostic reason, and model binaries must never be
committed.

Do not claim Rust Apple BitNet local answers until the target backend runs a
strict local-answer receipt with real model, tokenizer authority, selected
backend, fallback status, generated text, token IDs, and timing.

## Claim Boundary

Do not claim:

```text
BitNet local-answer quality from dense Qwen evidence
M4 BitNet local answers from MacBook artifact-only evidence
QK256 support on Apple Silicon
full Apple Metal inference
Neural Engine execution
MPSGraph model inference
broad Apple Silicon performance
```

Do claim only:

```text
artifact accepted or rejected for the recorded source, file, hash, tokenizer
authority, runner, prompt suite, machine context, and cleanup status
```
