# Falcon-E Family implementation plan

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-TOKENIZER-PROMPT.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-REFERENCE-QUALITY.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-I2S.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-TL1-TL2.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CPU.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CUDA.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-APPLE.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-A770-OPENCL.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-PERFORMANCE.md
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no runtime support promotion until receipts pass
Policy impact: no policy exception

## Scope

Make Falcon-E Family a first-class compact 1.58-bit planning lane without
runtime overclaims. The first phase is source-of-truth registration and specs;
later phases gather artifact identity, tokenizer/prompt authority, I2_S layout
proof, reference quality, Rust CPU answers, and backend receipts.

## PR sequence

1. `docs(falcon-e): add Falcon-E family proposal and source map`.
2. `docs(spec): add Falcon-E artifact and route contracts`.
3. `docs(spec): add Falcon-E tokenizer and reference-quality contracts`.
4. `docs(spec): add Falcon-E I2_S and TL contracts`.
5. `docs(spec): add Falcon-E backend/performance contracts`.
6. `models(falcon-e): add model coverage candidate rows`.
7. `models(falcon-e): add 1B and 3B artifact inventory receipts`.
8. `test(falcon-e): tokenizer and prompt authority audit`.
9. `test(falcon-e): reference answer corpus for 1B`.
10. `test(falcon-e): reference answer corpus for 3B`.
11. `loader(falcon-e): recognize Falcon-E I2_S artifacts structurally`.
12. `quant(falcon-e): I2_S scalar fixture parity`.
13. `cpu(falcon-e): strict CPU answer proof for 1B`.
14. `cpu(falcon-e): strict CPU answer proof for 3B`.
15. `cpu(falcon-e): AVX2/AVX512 I2_S parity`.
16. `cuda(falcon-e): I2_S all-layer route plan`.
17. `cuda(falcon-e): Falcon-E 1B one-token proof`.
18. `cuda(falcon-e): short-decode and warm-session proof`.
19. `apple(falcon-e): MacBook/M4 I2_S CPU-NEON proof`.
20. `a770(falcon-e): A770 I2_S route plan and fixture parity`.
21. `spec(falcon-e): TL1/TL2 fixture corpus`.
22. `quant(falcon-e): scalar TL1/TL2 oracle`.
23. `cpu(falcon-e): x86 TL2 or ARM TL1 answer proof`.
24. `bench(falcon-e): exact-profile benchmark receipts`.
25. `models(falcon-e): product CLI promotion review`.
26. `server(falcon-e): exact-profile server smoke`.

## Validation for docs/spec PRs

```bash
cargo run --locked -p xtask --no-default-features -- campaign check falcon-e-family
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Claim boundaries

- No model binaries.
- No all-Falcon-E claim from one model size.
- No Falcon-E 3B claim from Falcon-E 1B proof.
- No inheritance from Microsoft BitNet 2B, Falcon3, 1bitLLM, or dense SLM proof.
- No I2_S/QK256 compatibility claim before Falcon-E layout proof.
- No TL1/TL2 backend work before TL layout specs and scalar oracles.
- No CPU/CUDA/Apple/A770 answer readiness before artifact and reference-good
  receipts.
- No speedup before exact-profile benchmark review.
