# Falcon3 Family Implementation Plan

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../../docs/proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: [Falcon3 specs](../../docs/specs/INDEX.md#falcon3-family-onboarding)
Linked ADRs: [BITNET-ADR-0005](../../docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: sequences registered/candidate proof only until receipts promote exact rows
Policy impact: no policy exception

## Goal

Make Falcon3 Family a first-class multi-size BitNet-family onboarding lane without overclaiming. The lane starts with artifact authority, tokenizer/prompt authority, I2_S layout proof, and reference-good output for 1B/7B direct GGUFs; backend acceleration and performance come later.

## Hard Rails

```text
Do not commit model binaries.
Do not claim all Falcon3 works from one model size.
Do not claim Falcon3 1B proof proves Falcon3 7B/10B.
Do not claim Falcon3 proof inherits Microsoft 2B, Falcon-E, Llama3, or dense SLM proof.
Do not claim I2_S/QK256 compatibility before layout proof.
Do not route TL1/TL2 through QK256/I2_S kernels.
Do not claim CUDA/Apple/A770/CPU answer readiness before the artifact gate and reference-good gate pass.
Do not claim speedup before exact-profile benchmark review.
Keep all receipts fallback-explicit and claim-boundary-explicit.
Keep docs/spec PRs free of runtime/kernel changes.
```

## PR Sequence

### F3-000 — Source-of-truth proposal, source map, specs, and registered rows

Add the Falcon3 proposal, source map, implementation plan, campaign tracker, all Falcon3 family specs, and candidate-only matrix rows. Update Apple candidate matrices to mention Falcon3 separately from Falcon-E.

Acceptance:

- Docs/specs only; no runtime/kernel changes.
- No model binaries.
- No answer/backend/speed/server claims.
- Falcon3 vs Falcon-E boundary explicit.
- 1B/7B direct GGUF priority explicit.
- 3B/10B conversion-route status explicit.
- Model coverage rows are registered-only with all answer/backend/speed booleans false.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check falcon3-family
cargo run --locked -p xtask --no-default-features -- campaign generate --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

Rollback: revert the Falcon3 docs/spec/matrix/campaign files and regenerated campaign dashboards.

### F3-001 — Artifact inventory for direct GGUFs

Probe `tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF` and `tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF` for exact revision, file list, `ggml-model-i2_s.gguf` size, SHA256, GGUF metadata, license, tokenizer metadata if embedded, HF displayed model size, nominal model size, storage context, and cleanup status.

### F3-002 — Tokenizer and prompt authority audit

Record tokenizer files, hashes, BOS/EOS/PAD/UNK IDs, chat template text, completion fallback, stop policy, rendered prompts, prompt token IDs, reference runner command, and conversation mode policy.

### F3-003 — Reference-good corpus for 1B

Run the deterministic reference prompt suite for Falcon3 1B I2_S and record pass/fail output without claiming Rust CPU readiness.

### F3-004 — Reference-good corpus for 7B

Run the deterministic reference prompt suite for Falcon3 7B I2_S and record pass/fail output without claiming Rust CPU readiness.

### F3-005 — Structural loader recognition

Teach or verify Rust structural loading for Falcon3 1B/7B I2_S GGUFs, tensor-role classification, family tagging, and rejection of unapproved TL routes. This is the first runtime/loader PR and is not part of documentation-only registration.

### F3-006 — I2_S scalar fixture parity

Prove Falcon3 I2_S layout compatibility before aliasing QK256 kernels.

### F3-007 — Strict CPU answer proof for 1B

Promote only the exact Falcon3 1B CPU path after artifact, tokenizer/prompt, reference-good, layout, and CPU corpus receipts pass with fallback=false.

### F3-008 — Strict CPU answer proof for 7B

Promote only the exact Falcon3 7B CPU path after independent receipts pass, adding memory envelope and load timing.

### F3-009 — AVX2/AVX512 parity

Compare scalar versus AVX generated-token parity and record first divergence, selected kernels, and fallback=false without speed claims.

### F3-010 — CUDA all-layer route plan

Classify Falcon3 BitLinear layers, QK256/I2_S invocation count, unsupported ops, and route requirements without execution claims.

### F3-011 — Falcon3 1B CUDA one-token proof

After CPU answer-ready, prove one-token CUDA for the exact 1B artifact/profile with fallback=false and speedup=false.

### F3-012 — CUDA short-decode and warm-session proof

Extend exact 1B CUDA proof to short decode and warm sessions.

### F3-013 — Apple CPU/NEON proof

Run strict Apple CPU/NEON proof on MacBook or M4 with fallback=false and no Metal claim.

### F3-014 — 3B/10B conversion authority

Verify exact conversion paths, tool commits, input/output hashes, and runner commands before any reference-good or backend claim.

### F3-015 — TL fixture corpus and scalar oracle

Define TL1/TL2 fixtures and scalar oracles before any TL accelerator work.

### F3-016 — Exact-profile benchmarks and product promotion

Benchmark only exact artifacts/profiles with fallback=false, accepted answer quality, CPU comparators, and explicit review. Product CLI and server promotion require later exact receipts.
