# Apple Silicon Team — Process & Branch Management

- **id**: 20260301T1420-apple-silicon-team-process
- **tags**: #apple-silicon #process #branch-management
- **links**: [[00-index]], [[10-ci]]

## Context

Initial session assessing the state of Apple Silicon / Metal support in bitnet-rs.
The repo had accumulated significant branch sprawl with no clear merge strategy.

## Decision

Established a phased approach: merge green PRs first, then batch-rebase stale branches,
then prioritize Metal kernel integration with macOS CI.

## Evidence

- 740 remote branches, only 1 open PR (#1711) at time of assessment
- Metal feature flags already wired into workspace `Cargo.toml` but no macOS CI lane
- `bitnet-metal` crate exists on branch but not merged to `main`
- Many small branches are only 5-8 commits behind `main` — easy rebase targets

## Consequences

1. PR #1711 (green) must merge first — unblocks all subsequent rebases
2. Parallel agent dispatch per branch for rebase & PR creation
3. Apple Silicon priority: metal-kernels PR + macOS CI lane
4. Worktrees used for parallel agent work to avoid conflicts

## Follow-ups

- [ ] Merge PR #1711
- [ ] Batch rebase stale branches (5-8 commits behind)
- [ ] Create macOS ARM64 CI lane in `ci-core.yml`
- [ ] Merge `bitnet-metal` crate to `main`
