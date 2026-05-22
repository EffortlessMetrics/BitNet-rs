# Swarm Post-369 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #360, #361, #362, #363, #364, #365, #366, #367, #368, #369

## Landed

- #360 `xtask(repo-boundary): generate promotion packets`
  - Squash commit: `d0d0cbaa867c766b704c3baeafe3d85845c7dd0f`
  - Added conservative promotion-packet generation for future swarm-to-source
    handoffs.
- #361 `docs(slm-cpu): queue SLM-CPU-082 repeated packed q8 receipts`
  - Squash commit: `e2675bb96ca48a9f8a79dbf1bdb6249401e34fc9`
  - Shared generated-tracker work; branch was repaired and refreshed before
    merge.
- #362 `diag(a770): normalize readiness scoring for label cases`
  - Squash commit: `89378b4e48fe80a454cf1d9569cb23b765bbe419`
  - A770 diagnostic salvage only; no A770 support, quality, speed, selected
    attention, resident KV, full residency, reference parity, completion,
    server readiness, or hardware-support claim.
- #363 `docs(lunar-lake): refresh LNL258V-GOAL-AUDIT-032`
  - Squash commit: `eef9cd94095de7e0ea19b6b637a2056f02d4cb6d`
  - Tracker refresh only.
- #364 `sync(repo): merge BitNet-rs source main 9b655ae47`
  - Regular merge commit: `6ff756f9d67aea11607edfd6a589d03f74b007a1`
  - Parents: `4cebd4874234364e425e046b28a6c4791a4d6bb2` and
    `b92505ccf9dc6d8598525d5ded7275eda956b505`
  - This was a source-to-swarm sync and correctly preserved ancestry instead of
    using squash.
- #365 `docs(lunar-lake): close out LNL258V-GOAL-AUDIT-032`
  - Squash commit: `c0020eca8e32c07df58952b9881a6c92aa5b6ab9`
  - Closed the Lunar Lake tracker refresh.
- #366 `docs(a770): close A770-028`
  - Squash commit: `4cebd4874234364e425e046b28a6c4791a4d6bb2`
  - A770 closeout only; no public hardware-support claim.
- #367 `docs(slm-cpu): keep SLM-CPU-082 ready after queue merge`
  - Squash commit: `7c979cf14f83005563452f080513a7dddab570ef`
  - Reconciled the SLM-CPU-082 tracker state after the queue merge.
- #368 `docs(repo-boundary): document merge window discipline`
  - Squash commit: `b7f303b1decea3907b253ad6b816dae7daaafaa3`
  - Added short shared-surface merge windows and exclusive repository-boundary
    windows to the swarm authority docs.
- #369 `docs(repo-boundary): canonicalize promotion contract`
  - Squash commit and current promotion candidate SHA:
    `c8164e6e5ea2f172faa390dbf5931aa8b5725cb9`
  - Kept `docs/release/SWARM_PROMOTION.md` as a compatibility pointer and made
    `docs/release/PROMOTE_TO_BITNET_RS.md` the single active promotion contract.

## Source Delta

- Current source ref:
  `source/main` = `9b655ae474e6ebe96e832d20c219dd8a92c0c63a`
- Current swarm ref:
  `origin/main` = `c8164e6e5ea2f172faa390dbf5931aa8b5725cb9`
- `source/main` is reachable from `origin/main`.
- `rtk git rev-list --count source/main ^origin/main` returned `0`.
- `rtk git rev-list --count origin/main ^source/main` returned `369`.
- The post-357 source delta was resolved by #364 using a regular merge commit.

## Evidence

The queue evidence below was captured before this handoff PR was opened. While
this handoff is open, live PR-list commands are expected to show this PR.

- Open swarm PR queue at checkpoint:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --limit 50 --json number,title`
  returned `[]`.
- Independent open PR count:
  `rtk gh api 'repos/EffortlessMetrics/bitnet-rs-swarm/pulls?state=open&per_page=1' --jq 'length'`
  returned `0`.
- Generated active PR dashboard at checkpoint:
  `docs/tracking/generated/active-prs.md` contained only the header table and no
  active rows.
- Campaign manifest stale-PR scan:
  `Get-ChildItem docs\tracking\campaigns -Filter active.toml -Recurse | Select-String -SimpleMatch 'status = "pr_open"'`
  returned no matches.
- #364 merge-shape proof:
  `rtk git show --no-patch --pretty=raw 6ff756f9d67aea11607edfd6a589d03f74b007a1`
  showed two parents.
- #368 remote proof:
  `BitNet Rust Small Result`, `PR Gate Success`, Markdownlint, docs automation,
  and `Doctor + Generated Dashboards` completed successfully.
- #369 remote proof:
  `BitNet Rust Small Result`, Markdownlint, docs link checking, and guards
  completed successfully before this handoff was updated. Later non-required
  informational jobs were still settling when #369 squash-merged.

## Validation

- Queue refresh and independent REST count both proved zero open PRs before this
  handoff was written.
- `origin/main` and `source/main` were fetched before selecting the promotion
  candidate SHA.
- `rtk git merge-base --is-ancestor source/main origin/main` returned success.
- This handoff is docs-only and does not regenerate tracker dashboards or run
  Cargo proof.

## Remaining Work

- Do not start a swarm-to-source promotion from this handoff alone. Use
  `c8164e6e5ea2f172faa390dbf5931aa8b5725cb9` as the current promotion candidate
  only after the source-promotion operator chooses the batch boundary, prepares
  the promotion packet, and verifies source-owned release surfaces.
- Continue processing new swarm PRs as they arrive:
  classify lane, inspect exact diff, verify claim boundaries, run focused proof,
  and squash-merge normal swarm PRs only when green.
- Keep source-history repair, source-to-swarm sync, and swarm-to-source
  promotion ancestry-preserving. Do not squash those repository-boundary PRs.
- Keep generated dashboards, campaign `active.toml`, workflows, `Cargo.lock`,
  root policy files, and authority docs under short merge-window discipline.

## Blockers

- None for the current empty queue checkpoint.
- Source promotion has not started and remains a separate, ancestry-preserving
  release/source-owner operation.

## Claim Boundary

This handoff records operator state only. It does not claim A770 support, CUDA
support, Apple M4 support, Lunar Lake readiness, SLM CPU release readiness,
quality, speed, server readiness, selected attention, resident KV cache, full
residency, reference parity, source release readiness, source-to-swarm sync
completion beyond #364's named source commit, or swarm-to-source promotion
completion.

## Next Operator Commands

```powershell
rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt,statusCheckRollup
rtk gh api 'repos/EffortlessMetrics/bitnet-rs-swarm/pulls?state=open&per_page=1' --jq 'length'
rtk git fetch origin --prune
rtk git fetch source --prune
rtk git rev-parse origin/main
rtk git rev-parse source/main
rtk git merge-base --is-ancestor source/main origin/main
rtk git rev-list --count source/main ^origin/main
rtk git rev-list --count origin/main ^source/main
rtk git log --oneline -12 origin/main
```
