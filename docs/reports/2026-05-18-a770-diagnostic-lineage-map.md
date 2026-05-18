# A770 Diagnostic Lineage Map

Status: active lineage and content-audit map
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
Linked PRs: #4744 through #5725 A770 diagnostic branch chain
Support-tier impact: no promotion
Policy impact: none

## Scope

This report maps the open A770 diagnostic PR chain into mainline content-audit
buckets. It is a queue-recovery aid, not proof that any branch-chain PR is
mergeable into `main`, and not proof that any branch-chain PR is obsolete.

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
PR #5722 merged only into the A770 diagnostic branch chain. PR #5725 then
merged into the same branch chain as selected-key score-input bucket source
evidence. A follow-up queue refresh showed 153 open scoped PRs: 152 `a770/*`,
0 `codex/*`, and 1 `claude/*`.

Post-reopen repair note on 2026-05-18: the earlier bulk-close framing was
reversed. A live refresh after reopening showed 157 open scoped PRs: 150
`a770/*`, 6 `codex/*`, and 1 `claude/*`. The A770 diagnostic PRs are open
content inventory until each one is audited by exact content, not age or branch
distance.

Post-cleanup note on 2026-05-18: `#5724` landed the canonical tracker/CI-plan
fix, `#5740` was closed as its duplicate, `#5731` was closed as a duplicate of
`#5730`, and `#5730` then merged after rebase/review. `#5733` also merged as a
3B TL candidate docs lane. After the queue guardrail merge in `#5741`, the
Llama3-8B-1.58 candidate docs lane in `#5732` merged after rebase/review. The
remaining direct-to-main Codex docs/source-map PRs are `#5746`, `#5747`, and
`#5748`, plus the temporary follow-up refresh `#5751`.

## Disposition Rule

Do not merge current A770 diagnostic probes directly into `main`.

Do not close a PR because it is old, far behind `main`, noisy, or part of an
earlier branch chain.

Close only after a content audit proves one of:

- the exact useful content already landed, with successor PR or commit named;
- the exact useful content was clean-ported elsewhere, with successor PR or
  commit named;
- it is a true duplicate of another open PR, with the kept PR named;
- it is historical diagnostic evidence already captured here or in another
  ledger, with no unique code/test/report left to port;
- the idea is no longer wanted for a content reason recorded in the PR.

Use them as source material for replacement PRs only when the replacement:

- is based on current `main`;
- has one semantic purpose;
- keeps A770 support at `diagnostic` unless claim-grade receipts exist;
- removes transcript-only or temp-target artifacts;
- includes focused tests or documented no-reuse/no-promotion proof;
- preserves the not-claims in the A770 claim-boundary spec.

## Extraction Buckets

| Bucket | Representative PRs | Durable value | Audit action |
| --- | --- | --- | --- |
| Backend identity and claim guards | #4744, #4745, #4750 | Backend/fallback/status vocabulary and non-claiming route identity. | Rebuild as A770-003/A770-004 replacement PRs only. Do not inherit old generated/dependency edits. |
| Loader, tokenizer, and model invariants | #4751-#4756, #4801, #4841, #4847, #4887, #4959, #4961, #4966 | Candidate strict GGUF, tied-logit, embedding-row, tokenizer, and model-contract fixes. | Compare against current `main`; port only confirmed invariants with direct tests. |
| QK256/OpenCL mechanics | #4763, #4764, #4767, #4770, #4774, #4776, #4850, #4853, #4855, #4856 | QK256 layout, activation quantization, and OpenCL dispatch evidence. | Hold until A770-005/A770-006 smoke/parity path exists. No support or performance claim. |
| Reference setup and compare tools | #4782, #4788, #4793, #4796, #4799, #4819, #4821, #4823, #4825, #4827, #4830, #4831, #4833, #4862-#4908, #4910-#4949 | Reference setup, prompt identity, hidden/logit/layer trace planning, run, and compare tooling. | Collapse into one or two durable `xtask` trace/compare PRs with stable command names and tests. |
| Attention score, softmax, and value-mix hypotheses | #4990-#5064, #5098-#5131, #5711 | Diagnostic localization of score input, value cache, probability, value mix, history, and selected query boundary rows. | Archive lineage first. Port no runtime math fix until contradictory hypotheses are reconciled against current `main`. |
| Transient probe rows | Most one-off `diag-*history*`, `diag-*boundary*`, and selected-row probes in #4976-#5131 and #5711 | Local investigation evidence. | Audit for unique report/test/tool value. Close only after an exact successor, duplicate, or historical-only ledger entry is recorded. |
| Draft AVX2 perf branch | #5092 | Possible QK256 AVX2 optimization. | Leave draft until parity proof, repeatable benchmark context, CPU flags, samples, and claim boundary are current. |

## Post-Reopen Audit Guardrails

The current default for reopened A770 diagnostic PRs is `keep open pending
content audit`. A PR being old, stacked, or far behind `main` is not evidence
that its content is obsolete.

Read-only audit on 2026-05-18 found:

| PRs | Finding | Next action |
| --- | --- | --- |
| #4774-#5131 | No exact-superseded group was proven. Many PRs preserve unique layer scope, trace infrastructure, report shape, or diagnostic evidence. | Keep open while replacement PRs are built; do not close by range. |
| #4751-#4770, #4801, #4837, #4845, #4850, #4853, #4855, #4883, #4885, #4892, #4959, #4961, #5010, #5012, #5020 | Runtime, loader, tokenizer, QK256, embedding, attention, CLI, and proof candidates may still contain useful invariants. | Audit against current `main`; port smallest confirmed fixes with direct tests. |
| #5730, #5731 | Exact duplicate docs/specs pair by stable patch-id. | #5731 was closed after naming #5730 as the survivor; #5730 merged after rebase/review. |
| #5733 | Unique BitNet 3B TL candidate docs/specs/campaign lane. | Merged after rebase/review; do not treat its merge as runtime/model proof. |
| #5732 | Unique Llama3-8B-1.58 candidate docs/specs/campaign lane. | Merged after rebase/review; do not treat its merge as runtime/model proof. |
| #5746, #5747, #5748 | New direct-to-main model-family docs/source-map lanes opened after #5730 merged. | Treat as docs/spec source-of-truth waves; compare overlap before merging independently. |

No diagnostic PR should be closed as "historical evidence" unless this map or a
successor ledger names the exact report, test, or behavior that preserves its
useful content and confirms there is no unique content left to port.

## Immediate Queue Decisions

### #5711, #5722, and #5725

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

PR #5725 added selected-key score-input bucket source evidence on the same
branch chain. It merged there, not to `main`, and is the current selected-key
source-frontier lineage point for the next diagnostic slice. It is not A770
execution, semantic quality, residency, performance, reference parity, or
completion proof.

### #5092

`perf(qk256-avx2): VPERMPS LUT decode + shared byte loads` remains draft.

Disposition: do not merge from this lane until repeatable benchmark receipts,
CPU/flags/sample context, and parity tests support the stated performance
claim.

## Replacement PR Order

1. `docs(a770): archive diagnostic lineage and current blockers`
   - Preserve the branch-chain decision map and content-audit criteria.
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

After each replacement lands, audit the corresponding branch-chain PRs for
remaining unique code, tests, reports, or receipts. Close only those proven to
have no unique remaining value, with a link to the replacement and this map.

## Claim Boundary

This report may claim only:

```text
The open A770 diagnostic queue has been mapped into content-audit and
replacement buckets, and none of the current diagnostic probes is a direct
support-claim candidate.
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
