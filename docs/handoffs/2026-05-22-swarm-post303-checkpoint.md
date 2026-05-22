# Swarm Post-303 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #289, #290, #291, #292, #293, #295, #296, #297, #298, #299, #300, #301, #302, #303, #304

## Landed

- #289 `docs(handoff): record post-288 swarm checkpoint`
  - Squash commit: `5bf25feec2f1f5ece1cb0726802ff8bd16eee346`
  - Merged: `2026-05-22T06:49:03Z`
- #290 `docs(a770): close A770-022 tracker`
  - Squash commit: `a69b7d2952fbf0cb8b95493d518beea87aabb9f4`
  - Merged: `2026-05-22T06:55:46Z`
- #291 `docs(slm): add SLM-CPU-079 aligned matvec artifact`
  - Squash commit: `52bc1c0e9f813b20681dd7aa30f05657630198b2`
  - Merged: `2026-05-22T07:18:24Z`
- #292 `diag(cuda): probe Qwen3 RoPE table upload`
  - Squash commit: `6a8fba58a2668f2ebd9276e2764f780a2c5672c0`
  - Merged: `2026-05-22T07:58:18Z`
  - Claim boundary: diagnostic upload probe only; no CUDA speed, residency,
    server readiness, broad dense GGUF, or BitNet QK256 proof claim.
- #293 `fix(a770): keep missing frontier chosen-id unknown`
  - Squash commit: `694038c180738edd74fb546d0ebbdd33636bac61`
  - Merged: `2026-05-22T07:36:28Z`
- #295 `docs(a770): sync A770-022 campaign status`
  - Squash commit: `72029e8e17b358363ed9a91f9677081bad79f198`
  - Merged: `2026-05-22T07:24:42Z`
- #296 `docs(slm): close SLM-CPU-079 tracker`
  - Squash commit: `d2f505e11e283c35dacc300100ed0fcc4bf8da77`
  - Merged: `2026-05-22T07:31:15Z`
- #297 `sync: import BitNet-rs M4 lifecycle closeout`
  - Regular merge commit: `16ae8c17f15e102192557e1b6e9d0641d87142f8`
  - Parents: `8e5583baf624aec418b3142ea3904cfe52499384`,
    `4feab7cced8d4bda13055745eca93be5851b52df`
  - Merged: `2026-05-22T08:10:31Z`
  - Merge method: `MERGE`, preserving source-to-swarm ancestry.
- #298 `docs(lunar-lake): refresh LNL258V-GOAL-AUDIT-024 after SLM-CPU-079`
  - Squash commit: `63357f7d9cc28a9c27725eedd886c046d7234f74`
  - Merged: `2026-05-22T07:51:53Z`
- #299 `docs(lunar-lake): close LNL258V-GOAL-AUDIT-024`
  - Squash commit: `8e5583baf624aec418b3142ea3904cfe52499384`
  - Merged: `2026-05-22T08:05:25Z`
- #300 `diag(a770): report first mismatch logit margin frontier`
  - Squash commit: `46680a5efe29918991c6e81d9adb781ee189a2ca`
  - Merged: `2026-05-22T08:32:12Z`
  - Follow-up fix before merge: corrected `rows_truncated` so a single missing
    context row is not double-counted as truncation.
  - Local focused proof during refresh included `answer_parity` tests for
    `bitnet-cli`, `rustfmt` on `answer_parity.rs`, JSON parse of the A770
    artifact, and `git diff --check`.
  - Claim boundary: diagnostic mismatch frontier only; no A770 support,
    quality, speed, selected attention, resident KV, full residency, or parity
    claim.
- #301 `docs: refresh Lunar Lake audit to current swarm main`
  - Squash commit: `d3ab1515377122210f467cd16e21e96195103152`
  - Merged: `2026-05-22T08:41:41Z`
- #302 `test(a770): remove logit margin unwraps`
  - Squash commit: `2b7794313b0c5eeae9c604707d98b20aae3c2093`
  - Merged: `2026-05-22T09:05:25Z`
  - Policy rerun cleared after the branch removed the remaining logit-margin
    row unwrap introduced by the #300 regression test.
- #303 `fix(cuda): stage Qwen3 RoPE tables on CPU`
  - Squash commit and current promotion candidate SHA:
    `2e8152e3282f3e1960b5ee5db0e3a0b18da47a9f`
  - Merged: `2026-05-22T09:18:23Z`
  - Routed proof: `Route BitNet Rust Small`, `BitNet Rust Small on CX53`, and
    `BitNet Rust Small Result` passed on the final head.
  - Local focused proof: `cargo fmt -p bitnet-transformer -- --check` and
    `git diff --check origin/main...HEAD` passed; local
    `cargo check --locked -p bitnet-transformer --no-default-features --features cpu`
    timed out on this Windows checkout under Cargo/process contention.
  - Claim boundary: CUDA Qwen3 RoPE table staging fix only; no speed,
    residency, server readiness, broad CUDA support, or dense GGUF proof claim.
- #304 `docs: close LNL258V-GOAL-AUDIT-025`
  - Squash commit: `ad2707f9109ec8461af5907e980921165adce479`
  - Merged: `2026-05-22T08:52:48Z`

## Source Delta

- `source/main` was fetched while writing this handoff.
- `source/main` currently has two commits not yet reachable from
  `origin/main`:
  - `cae130284` `M4-COMPAT-001: add compatibility refresh contract (#6188)`
  - `23f4f17ff` `M4-COMPAT-001: close tracker after merge (#6189)`
- This handoff does not start a source-to-swarm sync. Those commits should be
  imported only in an explicit source-sync merge window, with ancestry
  preserved by regular merge commit or another approved non-rewriting update.
- The raw `origin/main..source/main` diff crosses source-owned and swarm-owned
  surfaces, including workflows and generated/hardware artifacts. Treat that
  diff as a sync-planning signal, not as a direct replacement patch.

## Evidence

- Open swarm PR queue at checkpoint:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt`
  returned `[]`.
- `origin/main` at checkpoint:
  `2e8152e3282f3e1960b5ee5db0e3a0b18da47a9f`.
- Latest `origin/main` history at checkpoint:
  `#303`, `#302`, `#304`, `#301`, `#300`, `#299`, `#292`, `#298`, `#293`,
  source import `#6187`, `#296`, `#295`, `#291`, source import `#6186`,
  `#290`, `#289`.
- Source-sync merge method proof:
  - #297 commit `16ae8c17f15e102192557e1b6e9d0641d87142f8` is a two-parent
    merge commit.
- Normal swarm merge method proof:
  - #303 commit `2e8152e3282f3e1960b5ee5db0e3a0b18da47a9f` has one parent,
    consistent with the required squash merge path for ordinary swarm work.
- Source delta proof:
  - `rtk git rev-list --count source/main ^origin/main` returned `2`.
  - `rtk git merge-base --is-ancestor source/main origin/main` did not pass.

## Validation

- Queue refresh: no open PRs were present when this handoff was written.
- `origin/main` and `source/main` were fetched before selecting the promotion
  candidate SHA and recording the source delta.
- GitHub merge/check evidence was read for #303.
- This handoff is docs-only and does not regenerate tracker dashboards or run
  Cargo proof.
- Local Cargo validation can stall on this Windows checkout behind
  package-cache locks or long process queues. Remote normalized routed CI is
  the required merge authority for normal swarm PRs when local proof is blocked.

## Remaining Work

- Do not start a swarm-to-source promotion from this handoff alone. Use
  `2e8152e3282f3e1960b5ee5db0e3a0b18da47a9f` as the current promotion
  candidate SHA once the source-promotion operator chooses the batch boundary.
- Schedule a short source-to-swarm sync window for source commits `cae130284`
  and `23f4f17ff` if those M4 compatibility updates should enter swarm before
  the next promotion batch.
- Continue processing new swarm PRs as they arrive:
  classify lane, inspect exact diff, verify claim boundaries, run focused
  proof, and squash-merge normal swarm PRs only when green.
- Keep source-history attach, source-to-swarm sync, and swarm-to-source
  promotion ancestry-preserving. Do not squash those repository-boundary PRs.
- A770 remains a salvage lane. Port durable trace, compare, claim-gate, and
  test value forward, but do not promote A770 support, quality, parity,
  selected attention, resident KV, full residency, or speed without explicit
  accepted receipts and claim gates.

## Blockers

- None for the current empty queue checkpoint.

## Claim Boundary

This handoff records operator state only. It does not claim A770 support,
CUDA support, Apple M4 support, quality, speed, server readiness, selected
attention, resident KV cache, full residency, reference parity, source release
readiness, source-to-swarm sync completion for M4 compatibility, or
swarm-to-source promotion completion.

## Next Operator Commands

```powershell
rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt,statusCheckRollup
rtk git fetch origin --prune
rtk git fetch source --prune
rtk git rev-parse origin/main
rtk git log --oneline -16 origin/main
rtk git log --oneline source/main ^origin/main
rtk git rev-list --count source/main ^origin/main
```
