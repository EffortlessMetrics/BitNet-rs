# Apple Silicon MacBook Cross-Reference Campaign

Campaign ID: `apple-silicon-macbook`

Status: active

## Objective

Use a MacBook as the Apple Silicon cross-reference and larger-artifact validation lane for dense SLM behavior and Apple BitNet candidate sweeps, while keeping M4 Mac mini product/performance claims and BitNet artifact claims separate.

## Why This Exists

The M4 Mac mini is the stable Apple Silicon dense SLM product/performance lane. It proves the practical Mac UX for the dense Qwen2.5 path: model cache, `bitnet mac ask`, Apple CPU/NEON routing, warm sessions, quality receipts, performance receipts, and phase-scoped Metal evidence.

The MacBook has a different job. It should cross-check Apple Silicon behavior on mobile hardware and handle larger model/artifact exploration when storage allows. It is the right place to sweep Apple BitNet candidates before sending accepted artifacts back to the M4 Mac mini for strict local-answer proof.

Dense Qwen success remains dense SLM evidence. It validates Mac UX and Apple CPU/NEON routing, not BitNet math, 1-bit / 1.58-bit layouts, I2_S/TL1/TL2 kernels, QK256, or Apple BitNet local-answer quality.

## End State

- MacBook machine, storage, thermal/mobile context, cache root, CPU/NEON, Metal, and MPSGraph visibility are receipt-backed.
- The known-good dense Qwen Mac path is mirrored on MacBook with the same model hash, tokenizer metadata, quality corpus, deterministic greedy settings, and receipt schema used by the M4 lane.
- Larger BitNet / 1-bit candidates are tested on MacBook first with source, size, hash, tokenizer authority, reference output, and cleanup status recorded.
- Accepted BitNet candidates remain artifact-qualified until a strict backend local-answer receipt proves coherent output on that backend.
- MacBook receipts can cross-check Apple Silicon behavior without turning one mobile run into a fleet-wide Apple performance guarantee.

## Initial Work Items

| Work item | Status | Notes |
|---|---|---|
| MB-AS-001 | merged | Add a MacBook machine/storage/profile receipt contract. |
| MB-AS-002 | blocked | Mirror the dense Qwen Apple CPU/NEON baseline once an actual MacBook runner is available. |
| MB-AS-003 | merged | Add the Apple BitNet candidate artifact matrix for MacBook sweeps. |
| MB-AS-004 | proposed | Validate official Microsoft 2B I2_S with external tokenizer authority. |
| MB-AS-005 | proposed | Evaluate 0.7B `1bitLLM/bitnet_b1_58-large` as the smaller Apple BitNet target. |
| MB-AS-006 | proposed | Evaluate 3B only on supported TL1/TL2 diagnostic routes. |

## Review Policy

Each PR owns one MacBook item. Keep model downloads under cache or `target/`, delete rejected candidates when the item requires cleanup, and never commit model binaries. Hardware and model claims must be scoped to the exact machine, model, tokenizer authority, backend, fallback status, and receipt evidence.

Human direction is required only when a candidate needs a new dependency, a license-sensitive decision, a destructive cleanup, a claim boundary is ambiguous, or branch/repository policy blocks the allowed merge path.
