# Swarm Post-357 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #349, #350, #351, #353, #354, #357

## Landed

- #349 `docs(slm-cpu): define SLM-CPU-081 repeated packed q8 timing gate`
  - Squash commit: `149247d66b04b8297f32d8b7d4096b401bd9f73c`
  - Merged before this checkpoint's final queue proof.
  - Claim boundary: repeated packed-Q8 timing gate only; no default runtime
    enablement, sustained throughput, broad answer quality, server, accelerator,
    Qwen3.5, or BitNet QK256 claim.
- #354 `docs(slm-cpu): close out SLM-CPU-081`
  - Squash commit: `68c328093b0c281a61587ea20b14debb2b2916a0`
  - Closed the generated tracker state for #349.
- #350 `docs(lunar-lake): refresh LNL258V-GOAL-AUDIT-031`
  - Squash commit: `9ae3b3de6fd7840f483b1be083d798c40c28c0d1`
  - Merged after generated-dashboard branch refresh churn.
  - Claim boundary: no-inference Lunar Lake audit refresh only; no model load,
    route promotion, speedup, power advantage, native accelerator, broad quality,
    Qwen3 Lunar Lake promotion, or BitNet QK256/I2_S behavior-change claim.
- #351 `xtask(repo-boundary): reduce false drift warnings`
  - Squash commit: `08b87a26d0b68d14f135d585e6e3cbfda1196be7`
  - Kept repo-boundary status usable in source-origin and swarm-origin
    checkouts without changing merge policy.
- #353 `docs(nvidia): record Qwen3 source capture timeout`
  - Squash commit: `52e8b3374a262b176899c0ad4dfa2b3810803964`
  - Claim boundary: diagnostic CUDA-MODEL-017M source-capture timeout only; no
    Qwen3 speed, benchmark-qualified speed, full CUDA residency, broad dense
    GGUF readiness, Qwen2.5 inheritance, BitNet packed I2_S/QK256 proof, server
    readiness, or repeated-comparator availability claim.
- #357 `docs(lunar-lake): close out LNL258V-GOAL-AUDIT-031`
  - Squash commit and current promotion candidate SHA:
    `952dcf775a0a478bb88ae5ec21c745d2c644ce1a`
  - Merged: `2026-05-22T17:56:27Z`
  - Cleared the Lunar Lake audit refresh from active tracker surfaces.

## Closed Unmerged

- #352 `docs(a770): close readiness divergence logit tracker`
  - Closed as duplicate after #348 already landed the A770-027 closeout.
- #355 `docs(lunar-lake): refresh LNL258V-GOAL-AUDIT-031`
  - Closed as an intermediate replacement after #350 landed.
- #356 `docs(lunar-lake): refresh LNL258V-GOAL-AUDIT-031`
  - Closed as duplicate after #350 landed; its remaining diff only retargeted
    tracker metadata from the landed PR to the replacement PR.
- #358 `docs(slm-cpu): close SLM-CPU-081 tracker`
  - Closed as duplicate/superseded by #354. `origin/main` already had
    SLM-CPU-081 marked merged, absent from active PRs, and represented by the
    canonical merged event.

## Source Delta

- `source/main` was fetched while writing this handoff, then fetched again
  during PR review after #357 landed.
- Current review state:
  - `source/main`: `9b655ae474e6ebe96e832d20c219dd8a92c0c63a`
  - `origin/main`: `952dcf775a0a478bb88ae5ec21c745d2c644ce1a`
- `source/main` has commits missing from `origin/main`:
  `rtk git rev-list --count source/main ^origin/main` returned `4`.
- `source/main` is not currently reachable from `origin/main`:
  `rtk git merge-base --is-ancestor source/main origin/main` failed.
- Missing source commits at review time:
  - `9b655ae47` `[codex] Close Apple M4 inference excellence audit (#6199)`
  - `982f65334` `M4 campaign overview status sync (#6198)`
  - `bdceac95d` `M4-METAL-EX-002: Record merged tracker state (#6197)`
  - `59a7c1ab0` `M4-METAL-EX-002: Add attention-score phase parity (#6196)`
- This handoff does not start a swarm-to-source promotion. The recorded SHA is a
  promotion candidate only after the source-promotion operator chooses the batch
  boundary, accounts for the current source delta, and prepares an
  ancestry-preserving promotion.

## Evidence

- Open swarm PR queue at checkpoint:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --limit 50 --json number,title`
  returned `[]`.
- Independent open PR count:
  `rtk gh api -X GET repos/EffortlessMetrics/bitnet-rs-swarm/pulls -f state=open --jq 'length'`
  returned `0`.
- During review of this handoff PR, the same open-PR commands returned this PR
  as the only open swarm PR.
- Generated active PR dashboard at checkpoint:
  `docs/tracking/generated/active-prs.md` contained only the header table and no
  active rows.
- Campaign manifest stale-PR scan:
  `Get-ChildItem docs\tracking\campaigns -Filter active.toml -Recurse | Select-String -SimpleMatch 'status = "pr_open"'`
  returned no matches.
- `origin/main` and local `main` at checkpoint:
  `952dcf775a0a478bb88ae5ec21c745d2c644ce1a`.

## Validation

- Queue refresh and independent REST count both proved zero open PRs.
- `origin/main` and `source/main` were fetched before selecting the promotion
  candidate SHA, then fetched again during review to correct the source delta.
- This handoff is docs-only and does not regenerate tracker dashboards or run
  Cargo proof.
- Local Cargo validation can stall on this Windows checkout behind
  package-cache locks or long process queues. Remote normalized routed CI remains
  the required merge authority for normal swarm PRs when local proof is blocked.

## Remaining Work

- Do not start a swarm-to-source promotion from this handoff alone. Use
  `952dcf775a0a478bb88ae5ec21c745d2c644ce1a` as the current promotion candidate
  SHA only after the source-promotion operator chooses the batch boundary and
  accounts for the four source commits currently missing from swarm.
- Before source promotion, either sync the missing source M4 commits into swarm
  with an ancestry-preserving source-to-swarm merge or explicitly include their
  disposition in the promotion plan.
- Continue processing new swarm PRs as they arrive:
  classify lane, inspect exact diff, verify claim boundaries, run focused proof,
  and squash-merge normal swarm PRs only when green.
- Keep source-history attach, source-to-swarm sync, and swarm-to-source
  promotion ancestry-preserving. Do not squash those repository-boundary PRs.
- A770 remains a salvage lane. Port durable trace, compare, claim-gate, and
  test value forward, but do not promote A770 support, quality, parity, selected
  attention, resident KV, full residency, or speed without explicit accepted
  receipts and claim gates.

## Blockers

- None for merging this docs-only handoff.
- Source promotion from the recorded candidate SHA is blocked until the current
  `source/main` delta is handled deliberately.

## Claim Boundary

This handoff records operator state only. It does not claim A770 support, CUDA
support, Apple M4 support, Lunar Lake readiness, quality, speed, server
readiness, selected attention, resident KV cache, full residency, reference
parity, source release readiness, source-to-swarm sync completion, or
swarm-to-source promotion completion.

## Next Operator Commands

```powershell
rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt,statusCheckRollup
rtk gh api -X GET repos/EffortlessMetrics/bitnet-rs-swarm/pulls -f state=open --jq 'length'
rtk git fetch origin --prune
rtk git fetch source --prune
rtk git rev-parse origin/main
rtk git merge-base --is-ancestor source/main origin/main
rtk git rev-list --count source/main ^origin/main
rtk git log --oneline -12 origin/main
```
