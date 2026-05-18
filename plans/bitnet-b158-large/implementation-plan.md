# bitnet_b1_58-large control-model implementation plan

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../../docs/proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact](../../docs/specs/BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [conversion](../../docs/specs/BITNET-SPEC-B158-LARGE-CONVERSION.md), [tokenizer/prompt](../../docs/specs/BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md), [reference quality](../../docs/specs/BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [CPU](../../docs/specs/BITNET-SPEC-B158-LARGE-CPU.md), [CUDA](../../docs/specs/BITNET-SPEC-B158-LARGE-CUDA.md), [Apple](../../docs/specs/BITNET-SPEC-B158-LARGE-APPLE.md), [performance](../../docs/specs/BITNET-SPEC-B158-LARGE-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion until receipts prove each tier
Policy impact: no policy exception

## Scope

Make `1bitLLM/bitnet_b1_58-large` a first-class, claim-safe BitNet-rs control
model. This plan does not change runtime code, commit model binaries, promote
coverage tiers, or claim answer/backend/speed support.

## PR sequence

### Phase 0 — Source-of-truth rails

1. **docs(bitnet-large): add control-model proposal and source map**
   - Add the proposal, source map, this plan directory, index links, and Apple
     candidate/campaign references.
   - Acceptance: docs/rails only, no model binaries, no runtime claims, no model
     coverage promotion.

### Phase 1 — Specs

2. **docs(spec): add b1.58-large artifact and conversion contracts**
   - Add artifact inventory and conversion-lane specs.
3. **docs(spec): add b1.58-large tokenizer and reference quality contracts**
   - Add tokenizer/prompt and reference-quality specs.
4. **docs(spec): add b1.58-large backend and performance contracts**
   - Add CPU, CUDA, Apple, and performance specs.

### Phase 2 — Claim-control registration

5. **models(bitnet-large): add coverage matrix row**
   - Register only: `current_tier = "registered"` and every answer/backend/speed
     claim remains false or forbidden.

### Phase 3 — Artifact inventory

6. **models(bitnet-large): add artifact inventory command/receipt**
   - Record revision, file list, sizes, SHA256, tokenizer files, config fields,
     storage context, and cleanup status.
   - Acceptance: no answer, conversion, backend, or speed claim.

### Phase 4 — Conversion authority

7. **convert(bitnet-large): F16 structural GGUF conversion receipt**
   - Use `bitnet-st2gguf` only for F16 structural/reference output.
8. **convert(bitnet-large): upstream-compatible I2_S/TL1/TL2 conversion research receipt**
   - Investigate upstream commands and runner paths; if blocked, commit a
     blocked receipt rather than a workaround.

### Phase 5 — Reference output

9. **test(bitnet-large): tokenizer and prompt authority audit**
   - Record tokenizer/config hashes, special token IDs, prompt renderings,
     prompt token IDs, runner policy, and stop tokens.
10. **test(bitnet-large): reference answer corpus**
    - Run the approved reference artifact through the deterministic corpus and
      promote only if output is coherent and bounded.

### Phase 6 — CPU path

11. **cpu(bitnet-large): strict scalar/CPU answer receipt**
    - Requires reference-good; records fallback false, generated IDs, decoded
      text, quality result, and no speedup claim.
12. **cpu(bitnet-large): AVX2/AVX512 parity**
    - Records scalar-vs-SIMD parity or first divergence with selected kernels.

### Phase 7 — CUDA path

13. **cuda(bitnet-large): all-layer route plan**
    - Classify all tensor roles and selected routes with no execution claim.
14. **cuda(bitnet-large): one-token proof**
    - Requires CPU answer-ready; records strict backend, fallback false, and no
      speedup claim.
15. **cuda(bitnet-large): short-decode and warm-session**
    - Records short/warm receipts and upload-once evidence where applicable.

### Phase 8 — Apple path

16. **apple(bitnet-large): MacBook reference artifact run**
    - Records MacBook storage context, artifact inventory, reference output, and
      cleanup with no M4 claim.
17. **apple(bitnet-large): M4 CPU/NEON strict answer proof**
    - Records strict Apple CPU/NEON answer receipts with no Metal claim.

### Phase 9 — Benchmark and product surfacing

18. **bench(bitnet-large): exact-profile benchmark receipts**
    - Uses the same artifact, tokenizer, and prompt profile with fallback false;
      speedup remains false until review.
19. **models(bitnet-large): product CLI promotion review**
    - Updates status/CLI surfaces only for tiers proven by receipts.

## Validation for docs/spec PRs

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-bitnet-artifact-sweep
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If campaign context changes require it, also run:

```bash
cargo run --locked -p xtask --no-default-features -- campaign doctor
cargo fmt --all -- --check
```

## Hard rules

- Do not commit model binaries.
- Do not substitute third-party GGUFs without an artifact-authority decision.
- Do not claim answer readiness before reference output passes.
- Do not treat `bitnet-st2gguf` F16 conversion as `I2_S`, `TL1`, or `TL2` proof.
- Do not inherit official Microsoft 2B proof.
- Do not inherit dense Qwen SLM proof.
- Do not claim speedup before exact-profile benchmark review.
- Do not claim CUDA, Apple, or CPU answer readiness before the artifact gate
  passes.
- Keep all receipts fallback-explicit and claim-boundary-explicit.
