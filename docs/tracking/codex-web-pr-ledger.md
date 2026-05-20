# Codex Web PR Ledger

Status: refreshed queue-recovery summary; historical snapshot retained below
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs: docs/reference/SPEC_SYSTEM.md
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: open GitHub PR queue snapshot
Support-tier impact: no support-tier promotion
Policy impact: no policy exception

This ledger is the durable tracked summary for the Codex/web PR recovery lane.
It is a review and disposition aid, not proof that a PR is mergeable. Every
merge still needs a narrow diff review, the PR's stated proof commands, and
`git diff --check`.

The live queue is generated locally under `target/pr-ledger/` because it is a
throwaway refresh surface. This tracked file records the current durable
disposition and retains the original large snapshot below for provenance.

## Current Queue State

- Date: 2026-05-20
- Command: `rtk gh pr list --state open --json number,title,isDraft,mergeable,mergeStateStatus,updatedAt,headRefName,labels,author,url --limit 100`
- Open PR count: 1
- Duplicate open PR clusters: none
- Direct `main` PRs: #5092 only
- Worktree audit: clean on `main` when this refresh was recorded

| PR | Lane | Current signal | Disposition |
|---:|---|---|---|
| #5092 | CPU AVX2 / perf | Draft, mergeable, GitHub check rollup green, one changed file: `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`. Labels: `blocked`, `disp:draft-hold`, `claim:perf-evidence-required`, `proof:partial`, `lane:cpu-avx2`, `lane:perf`, `prio:p2`. | Keep blocked as draft. Direct `bitnet-quantization --features cpu,avx2` proof is not product-route proof. Undraft or merge only after accepted selected-route/counter proof, exact-profile benchmark evidence, and claim review. |

Current #5092 claim boundary:

- local F32/no-scale AVX2 QK256 kernel candidate only;
- no scaled BitNet I2_S x I8_S AVX2 execution claim;
- no generated-token parity, model answer, server-readiness, residency, or
  accepted speedup claim;
- `speedup_claim` remains false until an exact-profile performance review
  accepts a narrow claim.

Durable value already salvaged from #5092:

- #6136 merged the test-only subset at commit
  `34a4aaaf0239513f0b4f603ff1c822325d1940d0`.
- #6136 landed QK256 byte-packing fixture cleanup, strict AVX2 full-block
  position identity coverage, and selected-kernel/fallback assertions without
  changing the AVX2 runtime kernel.

Recent closed-unmerged items from the recovery wave include duplicate or
superseded PRs such as #6121, #6080, #6002, #5984, #5980, #5978, #5977, #5965,
and #5960, #5950, and #5944. Closed PRs must still name their successor or
content reason in the PR discussion; this file does not by itself authorize
closure.

The historical sections below are not a live open-queue statement. They remain
as lineage/provenance for how earlier Codex, A770, SRP, and diagnostics waves
were classified before the queue was reduced.

## Non-Closure Rule

Do not close a PR because it is old, far behind `main`, noisy, or part of an
earlier branch chain.

Closing is allowed only after a content audit proves one of these conditions:

- the exact useful content already landed, with successor PR or commit named;
- the exact useful content was clean-ported elsewhere, with successor PR or
  commit named;
- it is a true duplicate of another open PR, with the kept PR named;
- it is historical diagnostic evidence already captured in this ledger or a
  linked source map, with no unique code/test/report left to port;
- the idea is no longer wanted for a content reason recorded in the PR.

Every close comment must name the disposition, successor when applicable,
remaining unique content, and claim boundary. "Stale", "behind", and "old" are
not valid closure reasons.

Historical snapshot source:

- Date: 2026-05-18
- Command: `rtk proxy gh pr list --state open --limit 200 --json number,title,headRefName,baseRefName,isDraft,mergeable,mergeStateStatus,updatedAt,body,changedFiles,files,url`
- Scope counted here: open PRs with `codex/*`, `a770/*`, or `claude/*` heads.
- Initial count: 185 total: 15 `codex/*`, 169 `a770/*`, 1 `claude/*`.
- Refresh note: the queue is actively moving. After processing #5488, a
  follow-up `gh pr list` refresh showed 175 open scoped PRs: 12 `codex/*`, 162
  `a770/*`, and 1 `claude/*`.
- Later refresh note: after this ledger branch opened, #5545, #5546, and #5547
  appeared as additional AVX2 BitNet hot-path planning variants. They belong
  to the same duplicate/overlap cluster as #5540 through #5544.
- Queue refresh: after merging #5461 and #5571, a follow-up `gh pr list`
  refresh showed 166 open scoped PRs: 4 `codex/*`, 160 `a770/*`, 1 `cuda-*`,
  and 1 `claude/*`. Current non-A770 queue head: #5569, #5568, #5549, #5547,
  and #5541.
- Queue refresh: after #5697, #5709, and #5710 merged, a follow-up
  `gh pr list` refresh showed 154 open scoped PRs: 153 `a770/*`, 0
  `codex/*`, and 1 `claude/*`. The only direct `main` PR in that scope is
  draft perf PR #5092; the A770 queue remains a stacked diagnostic chain.
- Queue refresh: after #5715 and #5717 merged, #4741 through #4744 were
  closed/superseded, and #5722 merged only into the A770 branch chain, a
  follow-up `gh pr list` refresh showed 153 open scoped PRs: 152 `a770/*`, 0
  `codex/*`, and 1 `claude/*`. The only direct `main` PR in that scope remains
  draft perf PR #5092.
- Queue refresh: #5725 subsequently merged into the same A770 branch chain as
  selected-key score-input bucket source evidence. It is lineage evidence for
  the next diagnostic slice, not mainline A770 execution, semantic quality,
  residency, performance, reference parity, or completion proof.
- Queue recovery refresh: after the unsafe close/reopen repair pass, a live
  `gh pr list --limit 300` refresh showed 157 open scoped PRs: 150 `a770/*`,
  6 `codex/*`, and 1 `claude/*`. The increased open count is intentional until
  each PR has a content-audited merge, rebase, port, duplicate, or historical
  evidence disposition.
- Follow-up cleanup on 2026-05-18: #5724 landed as the canonical
  SLM-CPU-040/041 tracker sync and `xtask/src/ci/plan.rs` dead
  `changed_count` fix; #5740 was closed as superseded by #5724; #5731 was
  closed as a duplicate of #5730; #5730 then merged after rebase/review.
- Live follow-up refresh on 2026-05-18 after #5730, #5732, #5733, and #5741
  merged leaves three direct `codex/*` model-doc/source-map PRs to `main`
  (`#5746`, `#5747`, `#5748`) plus draft perf PR `#5092`. The temporary
  queue-refresh PR `#5751` exists only to update this ledger state.

## Post-Reopen Content Audit Addendum

This addendum records read-only audit results after the reopen repair. It is
not approval to close PRs; it is the minimum routing evidence for the next
queue pass.

| PRs | Content audit result | Required action |
|---|---|---|
| #5730, #5731 | Exact duplicate BitNet-large docs/specs/plans by stable patch-id `d3a8898c691015cdf8fcc5f7468d9beb0bd792c1`. | #5731 was closed with #5730 named as the survivor; #5730 merged after rebase/review. |
| #5732 | Unique Llama3-8B-1.58 candidate docs/specs/campaign lane. | Merged after rebase/review; this is docs/source-map state only, not runtime/model proof. |
| #5733 | Unique BitNet 3B TL candidate docs/specs/campaign lane. | Merged after rebase/review; this is docs/source-map state only, not runtime/model proof. |
| #5746, #5747, #5748 | New direct-to-main model-family docs/source-map lanes opened after the #5730 merge. | Classify as docs/spec source-of-truth waves; compare overlap before merging independently. |
| #5724, #5740 | Both touched `xtask/src/ci/plan.rs` dead `changed_count` cleanup; #5724 also carried SLM tracker updates. | #5724 merged after green CI and #5740 was closed as superseded. |
| #5092 | Draft AVX2 QK256 runtime/perf candidate; no accepted speedup claim may land from it. | Keep draft/proof-gated until product-route execution proof, benchmark context, CPU flags, samples, and receipts are current. |
| #4774-#5131 | Reopened A770 diagnostic chain contains content-bearing trace, score, probability, value-mix, cache, layer, and reference tooling evidence. | Keep open by default. Port or merge useful reports; close only after exact successor, duplicate, or historical-only evidence is recorded. |

Runtime/proof PRs in the old loader, tokenizer, QK256, embedding, attention,
CLI, and proof clusters remain content-bearing until current `main` is checked
for the exact invariant or a replacement PR lands. This includes PRs #4751
through #4770 plus #4801, #4837, #4845, #4850, #4853, #4855, #4883, #4885,
and #4892, #4959, #4961, #5010, #5012, and #5020. Some may be superseded by
later work, but no closure disposition is valid without naming the successor PR
or commit.

## Initial Queue Summary

| Lane | PRs | Intent | Base/head classification | Mergeability | Proof commands | Generated files | Dependencies | Claim boundary | Recommended disposition |
|---|---|---|---|---|---|---|---|---|---|
| Source-of-truth and generated tracker closeout | #5536 | Close Apple M4 reproduction manifest item and generated campaign status. | `main` <- `codex/apple-m4-inference-excellence/M4-REPRO-002-closeout` | Mergeable | PR body lists `campaign check apple-m4-inference-excellence`, `campaign generate --check`, `campaign doctor`, `git diff --check`. | Yes: campaign/generated status and global generated dashboards. | None detected. | Does not prove new Apple M4 execution or performance. | Review generated files against generator output before merge; reject hand-edited generated drift. |
| Lunar Lake timing applicability receipts | #5537 | Record LNL258V route profile timing applicability and refreshed hardware receipts. | `main` <- `codex/lunar-lake/LNL258V-ROUTE-009-profile-timing-applicability` | Mergeable | PR body lists targeted `bitnet-cli` route-profile tests, `cargo build`, JSON validation, `campaign check intel-258v-platform`, `campaign generate --check`. | Yes: campaign/generated status and global generated dashboards. | None detected. | CPU/platform timing applicability only; no accelerator promotion. | Review after #5536 or as a separate generated-tracker lane; require generator proof. |
| Fast tests | #5488 | Add unit coverage for startup diagnostics. | `main` <- `codex/add-unit-testing-kmbnst` | Merged | PR body lists `cargo test --locked -p bitnet-startup-contract-diagnostics-core --no-default-features`, `cargo fmt --all -- --check`, and package clippy. | No. | None detected. | Test-only; no behavior or public API claim. | Landed after replacing test `.expect(...)` calls with `Result<()>`/`?`, rerunning targeted proof, and confirming green CI. |
| SRP refactor wave | #5461, #5464, #5465, #5466, #5467, #5468, #5469, #5470, #5471, #5472, #5473, #5474 | Split existing modules without intended behavior change. | All `main` <- `codex/refactor-codebase-into-srp-submodules*` heads. | All mergeable. | PR bodies list crate-scoped fmt/test/clippy or `git diff --check`; exact command differs per crate. | No. | None detected. | Behavior-preserving only; no public API drift unless already exported through same facade. | Review and merge one crate/module at a time after proving no behavior drift. Do not batch the wave. |
| Draft perf PR | #5092 | AVX2 QK256 runtime/perf candidate. | `main` <- `claude/improve-avx2-performance-fdrIb` | Mergeable, draft. | Not sufficient for merge until product-route execution, repeatable benchmark, and parity proof are current. | No. | None detected. | No speedup claim may land without CPU/flags/sample context, route/counter proof, and parity tests. | Leave draft/proof-gated; do not close or merge from this lane without content audit and current receipts. |
| A770 root experience/history chain | #4738 | Bench/experience history rails. | `main` <- `a770/llm-experience-history` | Conflicting. | PR body lists xtask `llm_experience`, help, docs, and bench receipt tests. | `Cargo.lock`. | `xtask/Cargo.toml`. | History rails cannot promote A770 quality, performance, full residency, or completion. | Do not merge as a giant root. Reconstruct useful receipt/history parts into smaller replacement PRs if needed. |
| A770 claim gates and runbooks | #4739, #4740 | Gate A770 promotion on experience receipts and document clean rerun flow. | Stacked on A770 branches, not `main`. | Mergeable. | #4739 lists `claims verify` and docs checks; #4740 lists diff check only. | No. | None detected. | Claim gate may only prevent promotion; it must not imply support. | Keep as durable candidates if they can be rebased onto `main` and proven independently. |
| A770 backend/CLI route identity tools | #4741, #4742, #4743, #4744, #5697, #5717 | Preserve route identity, backend fallback classification, strict backend proof guard, and non-claiming OpenCL dispatch status. | Stacked A770 chain plus main-based replacements. | #4741 through #4744 closed/superseded; #5697 and #5717 merged to `main`. | Replacement PRs ran local formatting, metadata, and `git diff --check` proof; long local cargo checks timed out without diagnostics and hosted PR Gate/CI supplied the merge proof. | No. | #5717 intentionally propagated OpenCL/oneAPI feature flags through the status path; no lockfile landed. | May preserve identity/fallback receipts; must not claim A770 OpenCL execution or BitNet inference works. | Done for fallback/status pieces. Remaining identity/strict-backend ideas are replacement-only A770-003/A770-004 source material. |
| A770 OpenCL launcher and route label branch | #4745, #4750 | Add OpenCL launcher and route labels. | Stacked A770 chain. | #4745 mergeable; #4750 conflicting. | PR bodies list QK256 dispatch tests and OpenCL/OneAPI checks. | #4745 has generated tracker/dashboard edits and `Cargo.lock`; #4750 has `Cargo.lock`. | Both touch Cargo manifests/lockfiles. | Launcher/status work must remain non-claiming and fallback-explicit. | Hold. Generated edits and conflicts make these poor direct merge candidates; salvage only after comparing against newer diagnostics. |
| A770 loader/tokenizer/transformer/runtime fixes | #4751, #4752, #4753, #4755, #4756, #4757, #4758, #4759, #4767, #4770, #4788, #4801, #4837, #4845, #4853, #4892, #4959, #4961, #5010, #5012, #5020, #5077 | Candidate correctness fixes across loader, tokenizer, transformer, CLI, QK256, reference setup, embedding layout, and attention precision. | Stacked A770 chain, not standalone `main` PRs. | Mergeable in current stack except where blocked by ancestor conflicts. | PR bodies list crate-scoped tests/checks; several older fixes also report missing-fixture gaps or hardware/reference assumptions. | No generated files detected in this group. | No dependency files detected in this group. | May claim only the specific corrected invariant after direct proof; no A770 semantic quality, performance, selected attention, resident KV, full support/device residency, reference parity, or completion. | Highest-value A770 salvage pool. Compare against current `main`, port or merge the smallest confirmed fixes into replacement PRs, and keep originals open until content audit proves no unique value remains. |
| A770 tests/proof/quality hardening | #4761, #4850, #4855, #4883, #4885, #5004 | Test-only, proof oracle, and prompt-suite quality hardening. | Stacked A770 chain. | Mergeable. | PR bodies list targeted CLI, QK256, model, or prompt-suite tests. | No generated files detected. | No dependency files detected. | Test/proof only unless reviewed diff shows behavior change. | Good salvage candidates after verifying they are not coupled to transient branch-chain assumptions. |
| A770 diagnostic probes and trace/compare tools | #4760, #4763, #4764, #4774, #4776, #4782, #4793, #4796, #4799, #4807, #4809, #4815, #4819, #4821, #4823, #4825, #4827, #4830, #4831, #4833, #4841, #4847, #4848, #4856, #4857, #4859, #4861, #4862, #4863, #4865, #4867, #4868, #4869, #4870, #4871, #4872, #4873, #4874, #4875, #4877, #4878, #4880, #4882, #4887, #4893, #4897, #4901, #4906, #4908, #4910, #4912, #4913, #4915, #4919, #4923, #4925, #4927, #4928, #4934, #4936, #4939, #4941, #4947, #4949, #4952, #4953, #4966, #4972, #4976, #4977, #4978, #4982, #4983, #4984, #4986, #4990, #4991, #4992, #4993, #4994, #4997, #4998, #4999, #5002, #5006, #5008, #5015, #5016, #5018, #5019, #5022, #5024, #5026, #5028, #5033, #5034, #5038, #5039, #5040, #5044, #5046, #5047, #5049, #5050, #5051, #5053, #5056, #5059, #5064, #5068, #5070, #5075, #5079, #5098, #5101, #5103, #5105, #5111, #5113, #5114, #5117, #5120, #5122, #5123, #5126, #5129, #5130, #5131, #5132, #5134, #5136, #5137 | Diagnose A770/reference/BitNet drift, trace locality, history, score/probability/value-mix differences, and reference plan behavior. | Mostly stacked A770 chain; #4815 is conflicting. | 137 mergeable, one conflicting in this group. | Bodies repeatedly list xtask trace/reference tests, `cargo check`, `git diff --check`, and selected manual trace compares. | #4776 has generated tracker/dashboard edits and `Cargo.lock`; most others do not. | #4776 touches dependency metadata. | Diagnostic evidence only. Must not imply A770 quality, selected attention, resident KV, attention-score/softmax/value-mix residency, full support/device residency, reference parity, or completion. | Do not merge or close probe-by-probe from age/range alone. Audit content first; port or merge useful reports, and close only after the non-closure rule is satisfied. |

## Initial Blockers And Risk Flags

| PR | Flag | Disposition impact |
|---:|---|---|
| #4738 | Conflicting root branch, `Cargo.lock`, `xtask/Cargo.toml`. | Replacement/salvage only. |
| #4745 | Generated tracker/dashboard edits and dependency files. | Requires generator proof and dependency review; likely replacement. |
| #4750 | Conflicting plus `Cargo.lock` and CLI manifest. | Do not merge as-is. |
| #4776 | Generated tracker/dashboard edits, `Cargo.lock`, dependency files, and timed-out proof note in body. | Do not merge as standalone diagnostic. |
| #4815 | Conflicting diagnostic PR. | Keep open or port until content audit proves exact successor, duplicate, or historical-only disposition. |
| #5092 | Draft perf/runtime candidate with no accepted speedup claim. | Requires route/counter proof, parity, and repeatable benchmark context before any merge. |
| #5536 | Generated dashboards. | Merged by the refresh checkpoint; keep generator proof as the rule for similar PRs. |
| #5537 | Generated dashboards and hardware receipt JSON. | Merged by the refresh checkpoint; keep JSON and generator proof as the rule for similar PRs. |

## Processing Order

1. #5488 is done: test-only, one file, no generated files, no dependency edits,
   no claim promotion, and green post-fix CI.
2. Then process the SRP refactor wave one PR at a time, starting with the
   smallest crate-local split whose public facade is unchanged.
3. Process source-of-truth/generated tracker PRs only with generator proof.
4. For A770, rebuild the lineage before merging anything: keep durable claim
   gates, trace/compare tools, and confirmed runtime/model fixes. Do not close
   transient probes until a content audit proves their exact successor,
   duplicate, or historical-only disposition.
5. Leave perf PR #5092 draft until benchmark, CPU/flags/sample context, and
   parity evidence are current.

## Disposition Log

| Date | PR | Decision | Evidence |
|---|---:|---|---|
| 2026-05-18 | #5488 | Merged | Local `cargo test --locked -p bitnet-startup-contract-diagnostics-core --no-default-features`, package fmt check, package clippy, `git diff --check`, and refreshed GitHub checks all passed after removing no-panic-family test debt. |
| 2026-05-18 | #5536 | Merged before this ledger follow-up | GitHub reports merged as source-of-truth/generated tracker closeout; keep generator-proof requirement for similar PRs. |
| 2026-05-18 | #5537 | Merged before this ledger follow-up | GitHub reports merged as Lunar Lake timing applicability/receipt update; keep JSON and generator-proof requirement for similar PRs. |
| 2026-05-18 | #5465, #5466, #5472, #5473, #5474 | Merged by concurrent queue activity | GitHub reports these SRP refactors merged; remaining SRP queue still needs one-at-a-time review. |
| 2026-05-18 | #5461 | Merged by concurrent queue activity after local review | Local review confirmed the API gateway facade preserved exports; package fmt, check, API-gateway tests, clippy, and GitHub merge state were inspected. A follow-up format issue from current `main` was split to #5571. |
| 2026-05-18 | #5567 | Closed as superseded | The intended no-panic-family cleanup was already present on `origin/main`; closing avoided a redundant/conflicting cleanup PR. |
| 2026-05-18 | #5571 | Merged by concurrent queue activity | One-file BDD grid rustfmt repair for the format gate. Local package fmt/check and GitHub merge state were inspected. |
| 2026-05-18 | #5697 | Merged by concurrent queue activity after local review | The CLI proof summary now exposes `backend_fallback_used` and `backend_fallback_reason`; local package fmt/checks passed before merge, and GitHub Policy/PR Gate were green at merge. |
| 2026-05-18 | #5706 | Closed as duplicate | The same no-panic-family transformer cleanup landed in #5697, so the standalone repair branch was no longer needed. |
| 2026-05-18 | #5709 | Merged | Tracker-only M4-ROBUSTNESS-001 closeout after #5699. Campaign tracker, PR Gate, CI core, docs, and link checks were green. |
| 2026-05-18 | #5710 | Merged | Tracker-only SLM-CPU-040 setup after SLM-CPU-039. Campaign tracker, generated dashboards, PR Gate, CI core, docs, and link checks were green. |
| 2026-05-18 | #5715 | Merged by concurrent queue activity | SLM-CPU-040 down-projection storage boundary classification landed on `main` before the post-#5717 refresh. |
| 2026-05-18 | #5717 | Merged | Clean `main` replacement for #4744. Local review added canonical A770 not-claims to `qk256_dispatch_status`; package fmt, `git diff --check`, OpenCL metadata, OneAPI metadata, and hosted GitHub PR Gate were green. The long local QK256 cargo test/check probes timed out without diagnostics and were recorded as environmental validation gaps. |
| 2026-05-18 | #4741-#4744 | Closed as superseded | Backend route/fallback/status ideas were either replaced by #5697/#5717 or left as A770-003/A770-004 source material. None is a direct merge candidate. |
| 2026-05-18 | #5722 | Merged into A770 branch chain only | Diagnostic selected-key score history evidence merged to `a770/diag-rust-score-input-operand-drift`, not `main`; it remains lineage evidence, not mainline proof. |
| 2026-05-18 | #5725 | Merged into A770 branch chain only | Diagnostic selected-key score-input bucket source evidence merged to `a770/diag-rust-score-input-operand-drift`, not `main`; it remains lineage evidence for the next selected-key source-boundary diagnostic, not mainline proof. |

## New Open Cluster After Refresh

| PRs | Intent | Disposition |
|---|---|---|
| #5540, #5541, #5542, #5543, #5544, #5545, #5546, #5547 | AVX2/QK256 hot-path audit, diagnostic counters, receipt fields, and implementation planning. | Treat as a duplicate/overlap cluster. Compare claims and proofs, keep at most one canonical implementation plus one docs/plan PR if evidence supports it, and close only exact duplicates or superseded attempts with successor PRs named. |

## Current Queue Head

| PR | Lane | Current signal | Disposition rule |
|---:|---|---|---|
| #5569 | Apple M4 source-of-truth tracker | Mergeable tracker/generated update. Proof commands in the body include campaign check, campaign generate `--check`, campaign doctor, and diff check. | Process only with generator proof; no runtime/model/quality/perf claim. |
| #5568 | CUDA UX receipt/status schema tests | Mergeable two-file code/test change, but GitHub Policy failed on the current run. | Triage policy report before review or merge. Preserve receipt/status claim boundaries. |
| #5549 | Apple M4 prompt generation identity receipts | Conflicting receipt/schema/docs/generated update. | Do not merge as-is; refresh or salvage only after resolving conflicts and rerunning campaign generators. |
| #5547 | AVX2 hot-path docs/plan | Conflicting member of the AVX2 duplicate planning cluster. | Compare against #5541 and related #5540-#5547 attempts; keep one canonical plan if still useful. |
| #5541 | AVX2 QK256 hot-path counters and receipt fields | Mergeable code/test change, but Policy failed and the PR body lacks final test completion evidence. | Triage Policy and rerun exact QK256 proof before considering as the canonical implementation. |

## Latest Queue Head

| PR or cluster | Lane | Current signal | Disposition rule |
|---:|---|---|---|
| #5131 | A770 layer0 FFN diagnostic | Latest open A770 PR by the post-refresh list; stacked on `a770/diag-rmsnorm-f64-trace-effect`, not `main`. | Do not merge as-is. Salvage only through a durable trace/compare replacement PR after lineage flattening. |
| #4745-#5131 plus merged branch-chain evidence through #5725 | A770 diagnostic branch chain | 152 open `a770/*` PRs remain, mostly one-PR-per-probe branch-chain diagnostics. #4741 through #4744 are closed/superseded; #5722 and #5725 merged only into the A770 branch chain. | Use `docs/reports/2026-05-18-a770-diagnostic-lineage-map.md`; no probe-by-probe merges and no closure by age/range. |
| #5092 | Draft AVX2 QK256 perf | Only direct `main` PR in the scoped queue; still draft and proof-gated with no accepted speedup claim. | Leave draft until product-route execution proof, parity proof, repeatable benchmark context, CPU flags, samples, and claim boundary are current. |
