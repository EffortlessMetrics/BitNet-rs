# Apple Silicon Team — Friction Log & Process Notes

## Session: 2026-03-01

### Observations
- 740 remote branches, only 1 open PR (#1711). Massive backlog of unsubmitted work.
- Metal feature flags already wired into workspace Cargo.toml but no macOS CI lane.
- `bitnet-metal` crate exists on branch but not merged.
- No macOS runner in ci-core.yml — need to add for Apple Silicon validation.
- Many small branches are only 5-8 commits behind main — easy rebase targets.

### Process Decisions
1. Merge first: PR #1711 (green) unblocks all subsequent rebases.
2. Batch rebase: dispatch parallel agents per branch to rebase & PR.
3. Apple Silicon priority: metal-kernels PR + macOS CI lane.
4. Use worktrees for parallel agent work to avoid conflicts.

### Tweaks Applied
- (initial) Structured merge board in SQL for real-time tracking.
- (initial) Dependencies modeled: all rebases depend on #1711 merge.
