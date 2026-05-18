# Codex Web PR Ledger

Status: active queue-recovery ledger
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

This ledger is the working source for the Codex/web PR recovery lane. It is a
review and disposition aid, not proof that a PR is mergeable. Every merge still
needs a narrow diff review, the PR's stated proof commands, and `git diff
--check`.

Snapshot source:

- Date: 2026-05-18
- Command: `rtk proxy gh pr list --state open --limit 200 --json number,title,headRefName,baseRefName,isDraft,mergeable,mergeStateStatus,updatedAt,body,changedFiles,files,url`
- Scope counted here: open PRs with `codex/*`, `a770/*`, or `claude/*` heads.
- Initial count: 185 total: 15 `codex/*`, 169 `a770/*`, 1 `claude/*`.
- Refresh note: the queue is actively moving. After processing #5488, a
  follow-up `gh pr list` refresh showed 175 open scoped PRs: 12 `codex/*`, 162
  `a770/*`, and 1 `claude/*`.

## Initial Queue Summary

| Lane | PRs | Intent | Base/head classification | Mergeability | Proof commands | Generated files | Dependencies | Claim boundary | Recommended disposition |
|---|---|---|---|---|---|---|---|---|---|
| Source-of-truth and generated tracker closeout | #5536 | Close Apple M4 reproduction manifest item and generated campaign status. | `main` <- `codex/apple-m4-inference-excellence/M4-REPRO-002-closeout` | Mergeable | PR body lists `campaign check apple-m4-inference-excellence`, `campaign generate --check`, `campaign doctor`, `git diff --check`. | Yes: campaign/generated status and global generated dashboards. | None detected. | Does not prove new Apple M4 execution or performance. | Review generated files against generator output before merge; reject hand-edited generated drift. |
| Lunar Lake timing applicability receipts | #5537 | Record LNL258V route profile timing applicability and refreshed hardware receipts. | `main` <- `codex/lunar-lake/LNL258V-ROUTE-009-profile-timing-applicability` | Mergeable | PR body lists targeted `bitnet-cli` route-profile tests, `cargo build`, JSON validation, `campaign check intel-258v-platform`, `campaign generate --check`. | Yes: campaign/generated status and global generated dashboards. | None detected. | CPU/platform timing applicability only; no accelerator promotion. | Review after #5536 or as a separate generated-tracker lane; require generator proof. |
| Fast tests | #5488 | Add unit coverage for startup diagnostics. | `main` <- `codex/add-unit-testing-kmbnst` | Merged | PR body lists `cargo test --locked -p bitnet-startup-contract-diagnostics-core --no-default-features`, `cargo fmt --all -- --check`, and package clippy. | No. | None detected. | Test-only; no behavior or public API claim. | Landed after replacing test `.expect(...)` calls with `Result<()>`/`?`, rerunning targeted proof, and confirming green CI. |
| SRP refactor wave | #5461, #5464, #5465, #5466, #5467, #5468, #5469, #5470, #5471, #5472, #5473, #5474 | Split existing modules without intended behavior change. | All `main` <- `codex/refactor-codebase-into-srp-submodules*` heads. | All mergeable. | PR bodies list crate-scoped fmt/test/clippy or `git diff --check`; exact command differs per crate. | No. | None detected. | Behavior-preserving only; no public API drift unless already exported through same facade. | Review and merge one crate/module at a time after proving no behavior drift. Do not batch the wave. |
| Draft perf PR | #5092 | AVX2 QK256 optimization claim. | `main` <- `claude/improve-avx2-performance-fdrIb` | Mergeable, draft. | Not sufficient for merge until repeatable benchmark and parity proof are current. | No. | None detected. | No speedup claim may land without CPU/flags/sample context and parity tests. | Leave draft or close/supersede; do not merge from this lane yet. |
| A770 root experience/history chain | #4738 | Bench/experience history rails. | `main` <- `a770/llm-experience-history` | Conflicting. | PR body lists xtask `llm_experience`, help, docs, and bench receipt tests. | `Cargo.lock`. | `xtask/Cargo.toml`. | History rails cannot promote A770 quality, performance, full residency, or completion. | Do not merge as a giant root. Reconstruct useful receipt/history parts into smaller replacement PRs if needed. |
| A770 claim gates and runbooks | #4739, #4740 | Gate A770 promotion on experience receipts and document clean rerun flow. | Stacked on A770 branches, not `main`. | Mergeable. | #4739 lists `claims verify` and docs checks; #4740 lists diff check only. | No. | None detected. | Claim gate may only prevent promotion; it must not imply support. | Keep as durable candidates if they can be rebased onto `main` and proven independently. |
| A770 backend/CLI route identity tools | #4741, #4742, #4743, #4744 | Preserve route identity, backend fallback classification, strict backend proof guard, and non-claiming OpenCL dispatch status. | Stacked A770 chain. | Mergeable. | PR bodies list targeted cargo tests/checks plus `git diff --check`. | No. | #4741 touches `crates/bitnet-cli/Cargo.toml`; #4744 touches several crate `Cargo.toml` files. | May preserve identity/fallback receipts; must not claim A770 OpenCL execution or BitNet inference works. | Review as possible replacement PRs, starting with the smallest no-dependency identity/fallback slice. |
| A770 OpenCL launcher and route label branch | #4745, #4750 | Add OpenCL launcher and route labels. | Stacked A770 chain. | #4745 mergeable; #4750 conflicting. | PR bodies list QK256 dispatch tests and OpenCL/OneAPI checks. | #4745 has generated tracker/dashboard edits and `Cargo.lock`; #4750 has `Cargo.lock`. | Both touch Cargo manifests/lockfiles. | Launcher/status work must remain non-claiming and fallback-explicit. | Hold. Generated edits and conflicts make these poor direct merge candidates; salvage only after comparing against newer diagnostics. |
| A770 loader/tokenizer/transformer/runtime fixes | #4751, #4752, #4753, #4755, #4756, #4757, #4758, #4759, #4767, #4770, #4788, #4801, #4837, #4845, #4853, #4892, #4959, #4961, #5010, #5012, #5020, #5077 | Candidate correctness fixes across loader, tokenizer, transformer, CLI, QK256, reference setup, embedding layout, and attention precision. | Stacked A770 chain, not standalone `main` PRs. | Mergeable in current stack except where blocked by ancestor conflicts. | PR bodies list crate-scoped tests/checks; several older fixes also report missing-fixture gaps or hardware/reference assumptions. | No generated files detected in this group. | No dependency files detected in this group. | May claim only the specific corrected invariant after direct proof; no A770 semantic quality, performance, selected attention, resident KV, full support/device residency, reference parity, or completion. | Highest-value A770 salvage pool. Compare against current `main`, port the smallest confirmed fixes into replacement PRs, and close original probes once evidence is preserved. |
| A770 tests/proof/quality hardening | #4761, #4850, #4855, #4883, #4885, #5004 | Test-only, proof oracle, and prompt-suite quality hardening. | Stacked A770 chain. | Mergeable. | PR bodies list targeted CLI, QK256, model, or prompt-suite tests. | No generated files detected. | No dependency files detected. | Test/proof only unless reviewed diff shows behavior change. | Good salvage candidates after verifying they are not coupled to transient branch-chain assumptions. |
| A770 diagnostic probes and trace/compare tools | #4760, #4763, #4764, #4774, #4776, #4782, #4793, #4796, #4799, #4807, #4809, #4815, #4819, #4821, #4823, #4825, #4827, #4830, #4831, #4833, #4841, #4847, #4848, #4856, #4857, #4859, #4861, #4862, #4863, #4865, #4867, #4868, #4869, #4870, #4871, #4872, #4873, #4874, #4875, #4877, #4878, #4880, #4882, #4887, #4893, #4897, #4901, #4906, #4908, #4910, #4912, #4913, #4915, #4919, #4923, #4925, #4927, #4928, #4934, #4936, #4939, #4941, #4947, #4949, #4952, #4953, #4966, #4972, #4976, #4977, #4978, #4982, #4983, #4984, #4986, #4990, #4991, #4992, #4993, #4994, #4997, #4998, #4999, #5002, #5006, #5008, #5015, #5016, #5018, #5019, #5022, #5024, #5026, #5028, #5033, #5034, #5038, #5039, #5040, #5044, #5046, #5047, #5049, #5050, #5051, #5053, #5056, #5059, #5064, #5068, #5070, #5075, #5079, #5098, #5101, #5103, #5105, #5111, #5113, #5114, #5117, #5120, #5122, #5123, #5126, #5129, #5130, #5131, #5132, #5134, #5136, #5137 | Diagnose A770/reference/BitNet drift, trace locality, history, score/probability/value-mix differences, and reference plan behavior. | Mostly stacked A770 chain; #4815 is conflicting. | 137 mergeable, one conflicting in this group. | Bodies repeatedly list xtask trace/reference tests, `cargo check`, `git diff --check`, and selected manual trace compares. | #4776 has generated tracker/dashboard edits and `Cargo.lock`; most others do not. | #4776 touches dependency metadata. | Diagnostic evidence only. Must not imply A770 quality, selected attention, resident KV, attention-score/softmax/value-mix residency, full support/device residency, reference parity, or completion. | Do not merge probe-by-probe. Collapse into a small number of durable trace/compare tools and written learnings, then close transient PRs as superseded. |

## Initial Blockers And Risk Flags

| PR | Flag | Disposition impact |
|---:|---|---|
| #4738 | Conflicting root branch, `Cargo.lock`, `xtask/Cargo.toml`. | Replacement/salvage only. |
| #4745 | Generated tracker/dashboard edits and dependency files. | Requires generator proof and dependency review; likely replacement. |
| #4750 | Conflicting plus `Cargo.lock` and CLI manifest. | Do not merge as-is. |
| #4776 | Generated tracker/dashboard edits, `Cargo.lock`, dependency files, and timed-out proof note in body. | Do not merge as standalone diagnostic. |
| #4815 | Conflicting diagnostic PR. | Close or supersede after lineage review. |
| #5092 | Draft perf PR with a strong speedup claim. | Requires parity plus repeatable benchmark context before any merge. |
| #5536 | Generated dashboards. | Merged by the refresh checkpoint; keep generator proof as the rule for similar PRs. |
| #5537 | Generated dashboards and hardware receipt JSON. | Merged by the refresh checkpoint; keep JSON and generator proof as the rule for similar PRs. |

## Processing Order

1. #5488 is done: test-only, one file, no generated files, no dependency edits,
   no claim promotion, and green post-fix CI.
2. Then process the SRP refactor wave one PR at a time, starting with the
   smallest crate-local split whose public facade is unchanged.
3. Process source-of-truth/generated tracker PRs only with generator proof.
4. For A770, rebuild the lineage before merging anything: keep durable claim
   gates, trace/compare tools, and confirmed runtime/model fixes; close
   transient probes once their evidence is preserved.
5. Leave perf PR #5092 draft until benchmark, CPU/flags/sample context, and
   parity evidence are current.

## Disposition Log

| Date | PR | Decision | Evidence |
|---|---:|---|---|
| 2026-05-18 | #5488 | Merged | Local `cargo test --locked -p bitnet-startup-contract-diagnostics-core --no-default-features`, package fmt check, package clippy, `git diff --check`, and refreshed GitHub checks all passed after removing no-panic-family test debt. |
| 2026-05-18 | #5536 | Merged before this ledger follow-up | GitHub reports merged as source-of-truth/generated tracker closeout; keep generator-proof requirement for similar PRs. |
| 2026-05-18 | #5537 | Merged before this ledger follow-up | GitHub reports merged as Lunar Lake timing applicability/receipt update; keep JSON and generator-proof requirement for similar PRs. |
| 2026-05-18 | #5465, #5466, #5472, #5473, #5474 | Merged by concurrent queue activity | GitHub reports these SRP refactors merged; remaining SRP queue still needs one-at-a-time review. |

## New Open Cluster After Refresh

| PRs | Intent | Disposition |
|---|---|---|
| #5540, #5541, #5542, #5543, #5544 | AVX2/QK256 hot-path audit, diagnostic counters, receipt fields, and implementation planning. | Treat as a duplicate/overlap cluster. Compare claims and proofs, keep at most one canonical implementation plus one docs/plan PR if evidence supports it, and close superseded attempts. |
