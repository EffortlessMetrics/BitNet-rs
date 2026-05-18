# A770 Diagnostic Lineage Map

Status: active disposition map
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs:

- `docs/specs/intel-arc-a770-gpu-roadmap.md`
- `docs/specs/a770-bitnet-claim-boundary.md`

Linked ADRs:

- `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`

Linked plan:

- `plans/a770-bitnet-claim-boundary-implementation.md`

Linked issues: n/a
Linked PRs: #4744 through #5722 A770 diagnostic branch chain
Support-tier impact: no promotion
Policy impact: none

## Scope

This report maps the open A770 diagnostic PR chain into mainline disposition
buckets. It is a queue-recovery aid, not proof that any branch-chain PR is
mergeable into `main`.

The committed A770 source of truth remains diagnostic. This report does not add
kernels, model support, runtime behavior, benchmark claims, receipt promotion,
or A770 support claims.

## Current Queue Snapshot

Snapshot command:

```powershell
rtk gh pr list --state open --limit 220 --json number,title,headRefName,baseRefName,isDraft,mergeable,mergeStateStatus,updatedAt
```

Observed snapshot on 2026-05-18:

| Scope | Count |
| --- | ---: |
| Open scoped PRs with `a770/*`, `codex/*`, or `claude/*` heads | 154 |
| Open `a770/*` PRs | 153 |
| Open `codex/*` PRs | 0 |
| Open `claude/*` PRs | 1 |
| Draft PRs in that scope | 1 |
| PRs in that scope based directly on `main` | 1 |

The direct `main` PR is the draft AVX2 performance PR #5092. The A770 PRs are
stacked on other `a770/*` branches and should not be merged linearly.

Post-refresh note on 2026-05-18: #5717 clean-ported #4744's non-claiming
dispatch-status slice to `main`, #4741 through #4744 are closed/superseded, and
PR #5722 merged only into the A770 diagnostic branch chain. A follow-up queue
refresh showed 153 open scoped PRs: 152 `a770/*`, 0 `codex/*`, and 1
`claude/*`.

## Disposition Rule

Do not merge current A770 diagnostic probes directly into `main`.

Use them as source material for replacement PRs only when the replacement:

- is based on current `main`;
- has one semantic purpose;
- keeps A770 support at `diagnostic` unless claim-grade receipts exist;
- removes transcript-only or temp-target artifacts;
- includes focused tests or documented no-reuse/no-promotion proof;
- preserves the not-claims in the A770 claim-boundary spec.

## Extraction Buckets

| Bucket | Representative PRs | Durable value | Disposition |
| --- | --- | --- | --- |
| Backend identity and claim guards | #4744, #4745, #4750 | Backend/fallback/status vocabulary and non-claiming route identity. | Rebuild as A770-003/A770-004 replacement PRs only. Do not inherit old generated/dependency edits. |
| Loader, tokenizer, and model invariants | #4751-#4756, #4801, #4841, #4847, #4887, #4959, #4961, #4966 | Candidate strict GGUF, tied-logit, embedding-row, tokenizer, and model-contract fixes. | Compare against current `main`; port only confirmed invariants with direct tests. |
| QK256/OpenCL mechanics | #4763, #4764, #4767, #4770, #4774, #4776, #4850, #4853, #4855, #4856 | QK256 layout, activation quantization, and OpenCL dispatch evidence. | Hold until A770-005/A770-006 smoke/parity path exists. No support or performance claim. |
| Reference setup and compare tools | #4782, #4788, #4793, #4796, #4799, #4819, #4821, #4823, #4825, #4827, #4830, #4831, #4833, #4862-#4908, #4910-#4949 | Reference setup, prompt identity, hidden/logit/layer trace planning, run, and compare tooling. | Collapse into one or two durable `xtask` trace/compare PRs with stable command names and tests. |
| Attention score, softmax, and value-mix hypotheses | #4990-#5064, #5098-#5131, #5711 | Diagnostic localization of score input, value cache, probability, value mix, history, and selected query boundary rows. | Archive lineage first. Port no runtime math fix until contradictory hypotheses are reconciled against current `main`. |
| Transient probe rows | Most one-off `diag-*history*`, `diag-*boundary*`, and selected-row probes in #4976-#5131 and #5711 | Local investigation evidence. | Close as superseded after the durable trace tools and lineage summary preserve the useful conclusion. |
| Draft AVX2 perf branch | #5092 | Possible QK256 AVX2 optimization. | Leave draft until parity proof, repeatable benchmark context, CPU flags, samples, and claim boundary are current. |

## Immediate Queue Decisions

### #5711 and #5722

`diag(bitnet): bind selected query boundary rows` is diagnostic-only and stacked
on `a770/diag-rust-score-input-operand-drift`, not `main`.

The PR body reports:

- `query_boundary_present = false`;
- `boundary_rows_present = false`;
- `claim_allowed = false`.

Disposition: do not merge as-is. If the selected-query boundary handling remains
useful, port it into the durable reference trace/compare replacement PR after
the branch-chain lineage is flattened.

PR #5722 added selected-key score history decision evidence on the same branch
chain. It merged there, not to `main`, and remains diagnostic lineage evidence
only.

### #5092

`perf(qk256-avx2): VPERMPS LUT decode + shared byte loads` remains draft.

Disposition: do not merge from this lane until repeatable benchmark receipts,
CPU/flags/sample context, and parity tests support the stated performance
claim.

## Replacement PR Order

1. `docs(a770): archive diagnostic lineage and current blockers`
   - Preserve the branch-chain decision map and close/supersession criteria.
   - No runtime, route, or support-tier change.
2. `identity(a770): preserve requested and selected backend identity`
   - Implement the A770-003 identity slice from current `main`.
   - No kernels or A770 execution claim.
3. `diag(bitnet): add durable reference trace and compare tools`
   - Collapse the reusable plan/run/compare pieces from the trace branches.
   - Keep command outputs diagnostic and fallback/claim fields explicit.
4. `fix(bitnet): preserve confirmed GGUF/tokenizer/logit invariants`
   - Port only small confirmed loader/tokenizer/model fixes with direct tests.
   - No A770 quality, reference-parity, residency, or performance claim.

After each replacement lands, close the corresponding transient branch-chain PRs
with a link to the replacement and this map.

## Claim Boundary

This report may claim only:

```text
The open A770 diagnostic queue has been mapped into replacement and closure
buckets, and none of the current diagnostic probes is a direct merge candidate.
```

It must not claim:

```text
A770 OpenCL BitNet execution works.
A770 semantic quality is proven.
A770 performance is proven.
Selected attention is proven.
Resident KV, attention scores, softmax, or value mix are resident.
Full support-op or full device residency is proven.
Reference parity is complete.
BitNet inference completion is achieved.
```
