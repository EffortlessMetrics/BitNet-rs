<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# AMD CPU baselines Campaign Status

- Campaign: `amd-cpu-baselines`
- State: `active`
- Objective: Validate AMD 5700X and 9950X3D as CPU proof and benchmark lanes while preserving scalar, AVX2, AVX-512, cache, memory, and sustained-power context.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| AMD5700X-003 | ready | TBD | `codex/amd-cpu-baselines/AMD5700X-003-scalar-avx2-dispatch` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Prove 5700X scalar and AVX2 dispatch with selected CPU kernel receipts and no GPU/NPU fallback. |
| AMD9950X3D-003 | ready | TBD | `codex/amd-cpu-baselines/AMD9950X3D-003-scalar-avx2-avx512-dispatch` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Prove 9950X3D scalar, AVX2, and AVX-512 dispatch with selected CPU kernel receipts and no GPU/NPU fallback. |

## Hard Constraints

- These lanes are CPU proof lanes, not accelerator lanes.
- The 5700X lane must not claim AVX-512.
- 9950X3D receipts must record scheduler/core placement and cache-domain context before performance claims.
