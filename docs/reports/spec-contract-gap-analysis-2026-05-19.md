Status: Draft
Owner: Codex
Created: 2026-05-19
Linked proposal: n/a
Linked specs: docs/reference/SPEC_SYSTEM.md
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Clarifies cross-cutting contracts required before future promotion claims.
Policy impact: Recommends policy-file placement for CI and generated-tracker discipline.

# Cross-cutting spec contract gap analysis

## Core rule

Anything that future PRs must obey belongs in a spec, ADR, or policy file.
Anything that only describes the current PR sequence belongs in a plan or active
campaign item. Anything that proves one run belongs in a receipt.

## Highest-priority gaps to lock into durable source-of-truth artifacts

1. **Model readiness/status semantics**
   - Add `docs/specs/BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md`.
   - Lock canonical meanings for `tier/current_tier`, readiness booleans,
     route/backend/fallback provenance, and `next_proof`.
   - Require parity across:
     - `bitnet model status --device <device> --format json`
     - `bitnet receipts explain <receipt> --format json`
     - `bitnet support bundle --latest --device <device> --format json`

2. **Receipt-explain/support-bundle schema contracts**
   - Add:
     - `docs/specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md`
     - `docs/specs/BITNET-SPEC-SUPPORT-BUNDLE-SCHEMA.md`
   - Lock required/optional fields, `--latest` resolution, unknown-vs-false
     semantics, proof-family booleans, and promotion-warning behavior.

3. **Common route identity and proof-family non-inheritance**
   - Add:
     - `docs/specs/BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md`
     - `docs/specs/BITNET-SPEC-ROUTE-IDENTITY-COMMON.md`
   - Codify hard non-transfer rules across family/backend/route/artifact/profile/hardware.

4. **Complete I2_S CUDA + status-surface specs**
   - Complete/fix:
     - `docs/specs/BITNET-SPEC-I2S-CUDA.md`
     - `docs/specs/BITNET-SPEC-I2S-STATUS-SURFACE.md`
   - Lock route IDs, kernel requirements, fallback rules, ladder criteria,
     and exact acceptance profiles for `bitnet_packed_i2s_qk256_proof=true`.

5. **Route-contract template convergence**
   - Use OpenVINO/NPU route contract structure as template for:
     - `docs/specs/BITNET-SPEC-CUDA-ROUTE-CONTRACT.md`
     - `docs/specs/BITNET-SPEC-APPLE-ROUTE-CONTRACT.md`
     - `docs/specs/BITNET-SPEC-CPU-ROUTE-CONTRACT.md`
     - `docs/specs/BITNET-SPEC-A770-OPENCL-ROUTE-CONTRACT.md`

6. **Model promotion ladder**
   - Add `docs/specs/BITNET-SPEC-MODEL-PROMOTION-LADDER.md`.
   - Define exact tier entry/reset rules and `next_proof` computation:
     `registered -> structurally_valid -> reference_good -> cpu_answer_ready -> accelerator_answer_ready -> product_cli_ready -> benchmark_qualified -> server_ready`.

7. **Performance/server/residency promotion safety**
   - Add:
     - `docs/specs/BITNET-SPEC-PERFORMANCE-PROMOTION.md`
     - `docs/specs/BITNET-SPEC-SERVER-READINESS.md`
     - `docs/specs/BITNET-SPEC-RESIDENCY-PROMOTION.md`
   - Prevent overclaiming by forcing exact-profile proof contracts.

8. **Runtime correctness invariants (from A770 diagnostics)**
   - Add:
     - `docs/specs/BITNET-SPEC-GGUF-TOKENIZER-PROMPT-AUTHORITY.md`
     - `docs/specs/BITNET-SPEC-GGUF-EMBEDDING-LOGITS-INVARIANTS.md`
     - `docs/specs/BITNET-SPEC-BITNET-ATTENTION-PRECISION.md`
     - `docs/specs/BITNET-SPEC-REFERENCE-TRACE-COMPARE-CONTRACT.md`

9. **A770 lineage disposition (ADR + plan + report, not only spec)**
   - Add:
     - `docs/adr/ADR-00xx-a770-diagnostic-lineage-disposition.md`
     - `plans/a770-diagnostic-salvage/implementation-plan.md`
     - `docs/reports/a770-diagnostic-lineage-map.md`

10. **SLM CPU / Q8 selector-readiness boundaries**
    - Add:
      - `docs/specs/BITNET-SPEC-SLM-Q8-DENSE-LOCALITY.md`
      - `docs/specs/BITNET-SPEC-SLM-CPU-WARM-SESSION.md`

11. **Rust 1.95 / CI / ripr / no-panic / file-policy contracts (policy docs + TOMLs)**
    - Add/update docs and ledgers in `docs/ci/`, `docs/development/`, and `policy/`.

12. **Generated-tracker branch-race discipline (policy + CI doc)**
    - Add:
      - `docs/ci/generated-tracker-discipline.md`
      - `policy/generated-tracker-policy.toml`

## What should not be encoded in specs

Keep the following out of specs and in their owning artifacts:

- current PR number/order -> plan or campaign active item;
- generated dashboard rows -> generated tracker artifacts;
- one hardware run -> receipt JSON;
- temporary diagnosis -> report/PR body;
- queue disposition -> campaign ledger/events;
- local validation gaps -> PR closeout;
- branch-race notes -> policy/CI operation docs;
- exact benchmark result -> receipt + benchmark report.

## Suggested PR sequence

1. `docs(spec): define model readiness and receipt explain contracts`
2. `docs(spec): complete I2_S CUDA and status-surface contracts`
3. `docs(spec): codify BitNet tokenizer, embedding, and attention invariants`
4. `docs(adr): record A770 diagnostic lineage disposition policy`
5. `docs(spec): define SLM Q8 locality and selector-readiness boundaries`
6. `docs(policy): encode Rust 1.95 proof and CI lane contracts`
