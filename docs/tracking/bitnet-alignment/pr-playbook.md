# PR Playbook

## PR size

One PR should do one of:

- add or update tracking
- collapse one leaf crate
- collapse one small cluster into a destination module
- fix one truth-boundary bug
- add one verification gate
- wire one runtime path

Avoid mixing structural collapse with behavior changes.

## PR body template

```md
## Summary

- ...

## Work item

- `TRUTH-001`

## Scope

Allowed paths:
- ...

Out of scope:
- ...

## Acceptance

- [ ] ...

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] ...

## Notes

- No behavior change / behavior change explained
- Follow-up items
```

## Merge rules

- Prefer small PRs that keep main green.
- Close duplicates aggressively.
- Do not reopen superseded cleanup.
- If a PR discovers a larger issue, add a new ledger item instead of expanding scope.
- If CI fails outside the changed scope, record it in `status.md`; do not opportunistically rewrite unrelated systems.

## Collapse rules

- Move code first.
- Preserve tests.
- Update imports.
- Remove workspace member.
- Remove dependency edges.
- Keep module names SRP.
- Avoid public re-export sprawl.
