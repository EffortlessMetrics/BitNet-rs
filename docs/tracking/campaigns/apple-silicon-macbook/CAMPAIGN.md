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

The live M3 MacBook Air execution lane now has its own campaign-local control
plane at `docs/tracking/campaigns/apple-m3-macbook-air/`. Treat this campaign as
the broader MacBook umbrella and historical cross-reference record; use
`apple-m3-macbook-air` for current M3 Air item state, validation commands,
output paths, and merge gates.

Do not open new live M3 Air execution work here unless it is explicitly an
umbrella/proxy note. Machine-profile receipts, dense SLM MacBook runs, large
artifact downloads, storage audits, and M4 handoff reports belong in
`apple-m3-macbook-air` first; this campaign should link to those items instead
of duplicating them.

| Work item | Status | Notes |
|---|---|---|
| MB-AS-001 | merged | Add a MacBook machine/storage/profile receipt contract. |
| MB-AS-002 | superseded | Live M3 dense Qwen work now belongs to `M3MBA-004A`/`M3MBA-004B` in `apple-m3-macbook-air`. |
| MB-AS-003 | merged | Add the Apple BitNet candidate artifact matrix for MacBook sweeps. |
| MB-AS-004 | superseded | Official Microsoft 2B I2_S validation now belongs to `M3MBA-005A`/`M3MBA-005B`/`M3MBA-005C`. |
| MB-AS-005 | superseded | 0.7B `1bitLLM/bitnet_b1_58-large` evaluation now belongs to `M3MBA-006`. |
| MB-AS-006 | superseded | 3B TL1/TL2 diagnostics now belong to `M3MBA-007`. |
| MB-AS-007 | merged | Add the live M3 MacBook Air lane roadmap, storage policy, sequencing, and claim boundaries. |
| MB-AS-008 | superseded | Real M3 Air machine/profile evidence now belongs to `M3MBA-002`. |

## Review Policy

Each PR owns one MacBook item. Keep model downloads under cache or `target/`, delete rejected candidates when the item requires cleanup, and never commit model binaries. Hardware and model claims must be scoped to the exact machine, model, tokenizer authority, backend, fallback status, and receipt evidence.

CI validates schemas, TOML, docs, generated tracker pages, committed receipts, and
small fixtures only. Model downloads, Hugging Face artifact fetches, reference
runs, thermal/mobile measurements, and large-cache cleanup are local hardware
tasks unless a future PR adds an explicit label-gated CI job for them. Any
committed artifact receipt must include source revision, filename, byte size,
SHA256, tokenizer authority when relevant, and cleanup or retention status.

Human direction is required only when a candidate needs a new dependency, a license-sensitive decision, a destructive cleanup, a claim boundary is ambiguous, or branch/repository policy blocks the allowed merge path.
