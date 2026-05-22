# Swarm Post-288 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #273, #274, #275, #276, #277, #278, #279, #280, #282, #283, #284, #285, #287, #288

## Landed

- #273 `docs: align agentic PR merge authority`
  - Squash commit: `e6c6dd3906d8ff4a32e423ca6929ba9ee468cfdb`
  - Merged: `2026-05-22T04:19:22Z`
- #274 `docs(slm): queue SLM-CPU-078 matvec cleanup`
  - Squash commit: `720b8fd579ca1f0533af41cf33dcc05e5ae86fb1`
  - Merged: `2026-05-22T04:27:57Z`
- #275 `docs(cuda): record Qwen3 transformer init trace`
  - Squash commit: `4280aa1b07cf3221d26a8f31a9789b816d7ea59d`
  - Merged: `2026-05-22T05:12:58Z`
- #276 `docs(lunar-lake): refresh swarm migration audit cutoff`
  - Squash commit: `0ebf4cabdb8aa6fd6c20c1a8e6dc27686c77dd6b`
  - Merged: `2026-05-22T04:50:28Z`
- #277 `perf(slm): add aligned packed q8 matvec path (SLM-CPU-078)`
  - Squash commit: `3b39511e9e1a1d42ffad31de95c7faf248a47130`
  - Merged: `2026-05-22T05:47:21Z`
  - Local focused proof during refresh:
    `rtk cargo test --locked -p bitnet-transformer --no-default-features --features cpu exact_q8_sidecar_runtime_hook`
    passed with 4 tests.
  - Local `campaign generate` and `campaign doctor` attempts were blocked by
    Windows Cargo/package-cache contention; hosted Campaign Tracker was used as
    the generated-dashboard authority.
- #278 `diag(a770): summarize answer logits topk frontier`
  - Squash commit: `91e36faeed8c184771f07ba2adb1ae3976edb380`
  - Merged: `2026-05-22T05:01:23Z`
  - Claim boundary: diagnostic A770 logits frontier only; no A770 quality,
    speed, residency, support, or parity claim.
- #279 `xtask: report repo-boundary status`
  - Squash commit: `074382edef1ee371b4cb3c5c751a39f368a9e588`
  - Merged: `2026-05-22T05:29:29Z`
- #280 `docs(a770): close logits topk frontier item`
  - Squash commit: `bda91f3278b1925606dc40a643056a02380713d4`
  - Merged: `2026-05-22T05:41:20Z`
  - Shared generated-dashboard window. The final branch had
    `Doctor + Generated Dashboards` and `BitNet Rust Small Result` green.
- #282 `xtask: fail repo-boundary status on missing source history`
  - Squash commit: `73f94073e8acc6160a11082db5bd001db1cc4bbe`
  - Merged: `2026-05-22T05:37:06Z`
- #283 `docs(slm): sync SLM-CPU-078 merge state`
  - Squash commit: `c1a48a61bb6ec029e35095c2ad9b77c03664c441`
  - Merged: `2026-05-22T05:57:14Z`
- #284 `sync: import BitNet-rs main history`
  - Regular merge commit: `c8b17e10bc453cec97bbd58f62002a00937b5dce`
  - Parents: `c1a48a61bb6ec029e35095c2ad9b77c03664c441`,
    `7dbc7b9c7b437af302952b192832e56ba851b6ef`
  - Merged: `2026-05-22T06:01:58Z`
  - Merge method: `MERGE`, preserving source-to-swarm ancestry.
  - Note: `Doctor + Generated Dashboards` failed on the import branch before
    later generated reconciliation. #288 is the final source-import
    reconciliation point for this checkpoint.
- #285 `diag(cuda): trace Qwen3 RoPE initialization`
  - Squash commit: `730914ff659ba8ebddfbf47e02211a6018fcae5c`
  - Merged: `2026-05-22T06:06:24Z`
  - Claim boundary: diagnostic constructor trace/report only; no CUDA speed,
    residency, server readiness, broad dense GGUF, Qwen2.5 inheritance, or
    BitNet QK256 proof claim.
- #287 `diag(a770): capture multistep answer parity frontier`
  - Squash commit and current promotion candidate SHA:
    `b01c5223d9b56b0e617959173a412aa371809baa`
  - Merged: `2026-05-22T06:34:41Z`
  - Local refresh proof:
    `rtk rustfmt --edition 2024 --check crates\bitnet-cli\src\commands\answer_parity.rs`
    and `rtk git diff --check` passed.
  - Claim boundary: diagnostic-only; no CPU/A770 answer parity, A770 quality,
    speed, selected attention, resident KV, full residency, reference parity,
    strict answer readiness, or trusted-partial acceleration claim.
- #288 `sync: import BitNet-rs M4 trend closeout`
  - Regular merge commit: `0342e761a77fb3f4e2f773d80a9d6d60d431e996`
  - Parents: `730914ff659ba8ebddfbf47e02211a6018fcae5c`,
    `49c042280a40e48a28915c82f64d57fe965a4951`
  - Merged: `2026-05-22T06:26:40Z`
  - Merge method: `MERGE`, preserving source-to-swarm ancestry.
  - Remote proof: `Doctor + Generated Dashboards`, `PR Gate Success`,
    `PR Plan`, and `BitNet Rust Small Result` all passed.

## Closed Unmerged

- #286 `docs(tracking): refresh generated dashboards after import`
  - Closed: `2026-05-22T06:14:15Z`
  - Merged: no
  - Disposition: superseded by #288, which imported the durable source closeout
    commit and reconciled the generated tracker state through an
    ancestry-preserving source-sync PR.

## Evidence

- Open swarm PR queue at checkpoint:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,mergeable,autoMergeRequest,updatedAt`
  returned `[]`.
- `origin/main` at checkpoint:
  `b01c5223d9b56b0e617959173a412aa371809baa`.
- Latest merged PRs inspected at checkpoint:
  `#288`, `#287`, `#285`, `#284`, `#283`, `#282`, `#280`, `#279`, `#278`,
  `#277`, `#276`, `#275`, `#274`, `#273`.
- Source-history reachability at checkpoint:
  - `rtk git merge-base --is-ancestor source/main origin/main` passed.
  - `rtk git rev-list --count source/main ^origin/main` returned `0`.
- Source-sync merge method proof:
  - #284 commit `c8b17e10bc453cec97bbd58f62002a00937b5dce` is a two-parent
    merge commit.
  - #288 commit `0342e761a77fb3f4e2f773d80a9d6d60d431e996` is a two-parent
    merge commit.
- Normal swarm merge method proof:
  - #287 commit `b01c5223d9b56b0e617959173a412aa371809baa` has one parent,
    consistent with the required squash merge path for ordinary swarm work.

## Validation

- Queue refresh: no open PRs were present when this handoff was written.
- GitHub merge/check evidence was read for the active source-sync and hardware
  closeout work.
- `origin/main` and `source/main` were fetched before selecting the promotion
  candidate SHA.
- This handoff is docs-only and does not regenerate tracker dashboards or run
  Cargo proof.
- Local generator/doctor proof can stall on this Windows checkout behind
  package-cache locks or long build queues. Hosted `Doctor + Generated
  Dashboards` was treated as authority for generated tracker branches when
  local proof was blocked and the diff was otherwise scoped.

## Remaining Work

- Do not start a swarm-to-source promotion from this handoff alone. Use
  `b01c5223d9b56b0e617959173a412aa371809baa` as the next promotion candidate
  SHA once the source-promotion operator chooses the batch boundary.
- Continue processing new swarm PRs as they arrive:
  classify lane, inspect exact diff, verify claim boundaries, run focused proof,
  and squash-merge normal swarm PRs only when green.
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
readiness, or swarm-to-source promotion completion.

## Next Operator Commands

```powershell
rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,mergeable,autoMergeRequest,updatedAt,statusCheckRollup
rtk git fetch origin --prune
rtk git fetch source --prune
rtk git rev-parse origin/main
rtk git merge-base --is-ancestor source/main origin/main
rtk git rev-list --count source/main ^origin/main
rtk git log --oneline -12 origin/main
```
