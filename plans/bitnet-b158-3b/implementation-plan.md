# BitNet b1.58 3B TL candidate implementation plan

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../../docs/proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [artifact](../../docs/specs/BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md), [conversion](../../docs/specs/BITNET-SPEC-B158-3B-CONVERSION.md), [TL layout](../../docs/specs/BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [tokenizer/prompt](../../docs/specs/BITNET-SPEC-B158-3B-TOKENIZER-PROMPT.md), [quality](../../docs/specs/BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md), [CPU](../../docs/specs/BITNET-SPEC-B158-3B-CPU.md), [CUDA](../../docs/specs/BITNET-SPEC-B158-3B-CUDA.md), [Apple](../../docs/specs/BITNET-SPEC-B158-3B-APPLE.md), [performance](../../docs/specs/BITNET-SPEC-B158-3B-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion until proof receipts pass
Policy impact: no policy exception

## Scope

Make `1bitLLM/bitnet_b1_58-3B` a first-class BitNet-rs TL-model candidate
without overclaiming. This plan does not enable runtime inference, commit model
binaries, promote support tiers, or treat the model as an `I2_S`/QK256 sibling
of the official Microsoft 2B artifact.

## PR sequence

1. **Docs and source-of-truth rails.** Add the proposal, source map, campaign,
   plan, and candidate-matrix clarifications. Acceptance: docs only, no model
   binaries, no runtime claims, 3B `I2_S` blocked explicitly, TL1/TL2
   verification path explicit.
2. **Artifact and conversion contracts.** Add the 3B artifact inventory and
   conversion specs. Acceptance: exact source revision, file/hash requirements,
   no official-GGUF assumption, blocked-conversion receipts allowed.
3. **TL layout and tokenizer contracts.** Add TL1/TL2 layout and tokenizer /
   prompt specs. Acceptance: TL1/TL2 are not QK256, tokenizer/prompt authority
   is model-specific, no Microsoft 2B prompt inheritance.
4. **Quality, backend, and performance contracts.** Add reference-quality, CPU,
   CUDA, Apple, and performance specs. Acceptance: each backend has prerequisite
   gates, fallback is explicit, speedup is false until exact-profile review.
5. **Coverage candidate registration.** Add a guarded model coverage row with
   `current_tier = "registered"` and forbidden answer/backend/speed claims.
6. **Artifact inventory receipt.** Probe the Hugging Face repo in cache only,
   record source revision, sizes, hashes, tokenizer/config hashes, storage
   context, cleanup status, and no official GGUF if true.
7. **Conversion and runner authority.** Verify or block upstream TL1/TL2
   conversion and runner commands with tool commits, input hashes, output hashes
   when produced, route metadata, and diagnostic-only claim boundaries.
8. **Tokenizer and reference output.** Audit tokenizer/prompt authority and run
   a deterministic reference corpus only after an approved reference route
   exists.
9. **TL fixtures, loader, and scalar oracle.** Add synthetic TL1/TL2 fixtures,
   structural loader recognition, unsupported `I2_S` rejection receipts, and
   scalar TL oracle tests before accelerator work.
10. **CPU answer proof.** Prove x86 TL2 CPU or ARM TL1 CPU/NEON only after
    reference-good and scalar TL oracle prerequisites pass.
11. **Accelerator proof.** Add AVX, CUDA, Apple, or OpenCL proof only after the
    exact CPU route is answer-ready for the same artifact and route family.
12. **Benchmark and product promotion.** Promote CLI/server support only from
    exact-profile receipts with fallback=false, quality passed, and speedup
    claims separately reviewed.

## Validation for docs/spec PRs

```bash
cargo run --locked -p xtask --no-default-features -- campaign check bitnet-b158-3b
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If a command cannot run because of environment limits, record the unavailable
command, why it could not run, substitute evidence, and whether that blocks
merge.

## Claim boundaries

- Do not commit model binaries.
- Do not claim 3B `I2_S` support.
- Do not route 3B through QK256/`I2_S` except diagnostic rejection receipts.
- Do not claim x86 TL2 until runner/conversion evidence is verified.
- Do not claim ARM TL1 until runner/conversion evidence is verified.
- Do not substitute third-party GGUFs without an artifact-authority decision.
- Do not inherit official Microsoft 2B `I2_S` proof.
- Do not inherit dense Qwen or dense SLM proof.
- Do not claim CPU, CUDA, Apple, server, or speed readiness before the artifact
  gate and route-specific receipts pass.
