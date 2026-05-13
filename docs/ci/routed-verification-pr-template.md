# Routed Verification PR Template

Use this body for every PR in the routed verification rollout.

```md
## Summary

## CI economics
- Default PR LEM before:
- Default PR LEM after:
- Lanes removed from default:
- Lanes still available by label/main/manual:
- Branch protection impact:

## Verification preserved
- What failure mode this still catches:
- What moved to main/label/nightly:
- Why that is acceptable:

## Boundaries
- No macOS default PR runner
- No Windows default PR runner
- No Docker/model/download default PR work
- No branch-protection change unless explicitly scoped
- No unrelated Rust/runtime changes

## Validation
- [ ] command
- [ ] command
```

## Filling guidance

- **Default PR LEM before/after:** use lane estimates from
  `policy/ci-lanes.toml` and `policy/ci-lane-whitelist.toml`.
- **Lanes removed from default:** name stable lane IDs, not just workflow names.
- **Still available:** identify labels and non-PR triggers that preserve proof.
- **Branch protection impact:** say `None` unless the work item explicitly scopes
  branch protection.
- **Verification preserved:** describe the failure mode, not only the workflow.
- **Validation:** paste exact commands run locally, then update checkboxes before
  opening the PR.
