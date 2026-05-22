# Swarm History Repair Closeout

Status: active
Owner: swarm maintainers
Created: 2026-05-21
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: `docs/release/PROMOTE_TO_BITNET_RS.md`
Linked campaign: n/a
Linked issues: n/a
Linked PRs: `EffortlessMetrics/bitnet-rs-swarm#104`
Claim impact: no runtime, model, hardware, quality, performance, residency, or
  release claim
Policy impact: records the repaired source/swarm history boundary
Generated-file impact: none

## Purpose

This closeout records the one-time history-preserving repair that made
`EffortlessMetrics/bitnet-rs-swarm` a descendant of both the old swarm history
and `EffortlessMetrics/BitNet-rs/main`.

The repair was not a hard reset, not a squash merge, and not a file copy. It
preserved the old swarm mainline, imported the real source commit graph, and
kept active swarm PR branches related to the old swarm history.

## Repair Commits

| Item | SHA | Notes |
| --- | --- | --- |
| Source main imported during repair | `6b4f3964431eab7c8d77d499bdee1e75014c50e5` | `EffortlessMetrics/BitNet-rs#6163`, the source repo release-boundary closeout present at import time. |
| Original swarm base connected to source | `4d9dcd250dd27a241d4c8653ad1c854c37cde0c1` | Pre-repair swarm commit kept as an ancestor. |
| History connection merge | `1dd9e185e5d7ba5bab96f11f7b4b0f2481e025a1` | Merge commit with parents `4d9dcd250dd27a241d4c8653ad1c854c37cde0c1` and `6b4f3964431eab7c8d77d499bdee1e75014c50e5`. |
| Source-tree update commit | `18b3955fa03cd51b863affc8066b1cffbc19a21b` | Updated the swarm tree to the source tree after connecting histories. |
| Current swarm main absorbed before import PR merge | `1d31bbe2de7498fe137751f3b0dfa113eb5f69d4` | `docs(repo): define swarm development authority (#98)`. |
| Import PR head | `b375b5281df92b4750847bfb05a6eebaeca0e3c6` | Guarded source release workflows before the final merge. |
| Import PR merge commit | `d6f57e922d15008568c6b742b13b9a4c6aed314d` | Real merge commit for `EffortlessMetrics/bitnet-rs-swarm#104`; not squash-merged. |
| Current source main checked during this closeout | `3d2660b85b123fb8448b9561717a0147e4ff421a` | `EffortlessMetrics/BitNet-rs#6164`, currently reachable from swarm `origin/main`. |
| Current swarm main checked during this closeout | `60bbf7fe1c2f6d9946fa3638cdcd3cfb6b9fdbe9` | Current `EffortlessMetrics/bitnet-rs-swarm/main` after refreshing the closeout branch. |

## Required Ancestry Invariants

These are the repaired-history invariants future operators should preserve:

```bash
git merge-base --is-ancestor 1d31bbe2de7498fe137751f3b0dfa113eb5f69d4 origin/main
git merge-base --is-ancestor source/main origin/main
git rev-list --count source/main ^origin/main
```

Expected results at this closeout:

```text
old swarm main at import ancestor of origin/main: yes
current source/main ancestor of origin/main: yes
git rev-list --count source/main ^origin/main: 0
```

The current `source/main` includes source repo commit
`3d2660b85b123fb8448b9561717a0147e4ff421a`, so the import remains current as
of this closeout. If a later source commit appears, do another
history-preserving merge or promotion operation. Do not hard-reset swarm main.

## Latest Source-To-Swarm Sync Checkpoint

The original history repair imported the public source graph and made the old
swarm main reachable. Later source-to-swarm sync PRs keep the repaired graph
current while `EffortlessMetrics/BitNet-rs` remains the public source of truth.

Latest verified checkpoint:

| Item | SHA / PR | Notes |
| --- | --- | --- |
| Source main imported | `36f93356b93d9ce98bfbcb37a301bf53dd68314c` | `EffortlessMetrics/BitNet-rs#6195`, M4 Metal phase-choice tracker closeout. |
| Swarm main before sync merge | `fdcbb124a3ad9aa8d2b4a7c78e29085d0bd20e19` | `EffortlessMetrics/bitnet-rs-swarm#320`, A770-024 yes-no corpus scoring. |
| Source-to-swarm sync PR | `EffortlessMetrics/bitnet-rs-swarm#317` | Regular merge; auto-merge used `MERGE`, not squash. |
| Source-to-swarm merge commit | `0ac5a1ac456717c4ac740df87fd63b5f06354ade` | Parents are current swarm main and the source-import branch head. |

Verified after #317 merged:

```bash
git merge-base --is-ancestor origin/main swarm/main
git rev-list --count origin/main ^swarm/main
```

Expected:

```text
current source/main ancestor of swarm/main: yes
git rev-list --count origin/main ^swarm/main: 0
```

This checkpoint is still a source-to-swarm import, not a source cutover and not
a release promotion. It imports source history into the high-throughput swarm
workspace so active swarm PRs can continue from a current source base.

## Workflow Guard Boundary

The import preserved swarm routed CI and guarded source-owned release surfaces.

Source-release and publishing workflows in swarm must remain guarded by
repository checks, disabled in swarm, or excluded from promotion. At this
closeout, the imported sensitive workflows include repository guards such as:

- `.github/workflows/release.yml`
  - release validation, artifact build, package build, release creation,
    crates.io publish, PyPI publish, npm publish, and post-release jobs require
    `github.repository == 'EffortlessMetrics/BitNet-rs'`.
- `.github/workflows/cache-bitnet-cpp.yml`
  - version lookup, cache publish, summary, and dependent update jobs require
    `github.repository == 'EffortlessMetrics/BitNet-rs'`.
- `.github/workflows/rust-ci-image.yml`
  - image publication requires source repo, `main`, and an allowed triggering
    event through `PUBLISH_IMAGE`.

Do not enable release, signing, publish, secrets-heavy deployment, model-cache,
or full-platform release workflows in swarm without a separate explicit
release/security review.

## Remaining Open Swarm PRs At Closeout

This snapshot is for coordination only. It is not a closure authority.

| PR | Title | State at closeout |
| ---: | --- | --- |
| #168 | `wasm: harden memory arithmetic and zero-size progress handling` | behind |
| #167 | `bitnet-models: make head/group ratios exact and tested` | behind |
| #166 | `wasm-inference/WASM-002: centralize generate messages and add branch tests` | behind |
| #165 | `M4-SERVE-EX-003: close local safety defaults tracker` | dirty |
| #163 | `kernels: harden matmul size checks to prevent usize overflow` | behind |
| #161 | `docs(slm-cpu): add SLM-CPU-071 timing gate` | dirty |
| #159 | `docs(lunar-lake): close stale BitNet watchpoints` | dirty |
| #153 | `Refactor tool call parsing into SRP-focused submodules` | behind |
| #152 | `Refactor SamplingStrategy::sample_in_place into SRP helper methods` | behind |
| #149 | `Refactor partition_work into SRP helpers` | behind |
| #148 | `Refactor startup diagnostics line assembly into SRP submodule helpers` | behind |
| #147 | `refactor(op_pool): split OpPool::acquire into SRP helper methods` | behind |
| #146 | `refactor(common): split matmul_shape into SRP helpers` | behind |
| #145 | `refactor: split API versioning into SRP submodules` | behind |
| #144 | `Refactor simple_inference into SRP-style helper submodule` | behind |
| #142 | `refactor: split envlock validation into SRP submodule` | behind |
| #141 | `refactor: extract JSON string parser into SRP submodule` | behind |
| #140 | `Refactor API key auth core into SRP-oriented modules` | behind |
| #139 | `refactor: split workgroup tuner into SRP submodules` | behind |
| #138 | `Refactor NVIDIA default workgroup builder into SRP submodule` | behind |
| #137 | `refactor(test-env): decompose EnvScope into SRP submodules` | behind |
| #135 | `refactor(bitnet-logits): split transforms into SRP submodules` | behind |

Open PRs should be rebased or merged from current swarm `main` as needed.
Preserve PR identity where feasible. Do not close PRs merely because the
history repair or later main movement made them behind.

## Future Sync Rule

Future source/swarm syncs must preserve both histories:

- do not hard-reset swarm main to source main;
- do not squash history imports or promotions;
- do not copy source files as a single content commit;
- do not force-push main for freshness;
- do not retarget existing machine clones by editing `origin`;
- keep release/publish/signing workflows source-owned unless separately
  migrated.

Use the promotion process in `docs/release/PROMOTE_TO_BITNET_RS.md` for
selected swarm-to-source work.

## Claim Boundary

This document records repository history and governance state only. It does not
promote runtime support, model quality, A770/OpenCL/CUDA/AVX execution,
server-readiness, speedup, TTFT, throughput, residency, reference parity,
release readiness, or publish readiness.
